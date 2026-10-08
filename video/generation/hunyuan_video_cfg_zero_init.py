#!/usr/bin/env python3
"""Zero-initial-velocity controller for HunyuanVideo-1.5 T2V/I2V CFG.

This is deliberately only the ``zero init`` half of CFG-Zero*.  It does not
implement or approximate the paper's optimized-scale projection.  The native
Hunyuan pipeline still evaluates its packed unconditional/conditional CFG
batch normally; this controller replaces the raw Transformer prediction with
zero for the requested first denoising steps.  Consequently the post-CFG
velocity passed to the scheduler is exactly zero as well.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CFGZeroInitConfig:
    total_steps: int
    zero_init_steps: int
    expected_task: str = "t2v"

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if not 1 <= self.zero_init_steps <= self.total_steps:
            raise ValueError("zero_init_steps must be in [1, total_steps]")
        if self.expected_task not in {"t2v", "i2v"}:
            raise ValueError("expected_task must be 't2v' or 'i2v'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": "cfg_zero_init_only",
            "total_steps": self.total_steps,
            "zero_init_steps": self.zero_init_steps,
            "zero_init_fraction": self.zero_init_steps / self.total_steps,
            "expected_task": self.expected_task,
            "optimized_scale_projection_enabled": False,
            "intervention_location": "raw_packed_transformer_prediction",
            "post_cfg_velocity_is_exact_zero": True,
        }


def _replace_prediction(output: Any, prediction: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        if not output:
            raise RuntimeError("Transformer returned an empty tuple")
        return (prediction, *output[1:])
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Transformer returned an empty list")
        return [prediction, *output[1:]]
    raise TypeError(
        "Hunyuan zero-init requires the Transformer return_dict=False tuple/list output"
    )


class HunyuanCFGZeroInit:
    """Patch one Transformer instance and audit exact per-sample scheduling."""

    def __init__(
        self,
        transformer: Any,
        *,
        total_steps: int,
        zero_init_steps: int,
        expected_task: str = "t2v",
    ):
        if getattr(transformer, "_sfg_guidance_controller", None) is not None:
            raise RuntimeError("another SFG guidance controller is already installed")
        self.config = CFGZeroInitConfig(
            total_steps=total_steps,
            zero_init_steps=zero_init_steps,
            expected_task=expected_task,
        )
        self.transformer = transformer
        self.original_forward = transformer.forward
        self.logical_forward_calls = 0
        self.zeroed_forward_calls = 0
        self.completed_samples = 0
        self.aborted_samples = 0
        self.current_sample_seed: int | None = None
        self.current_step = 0
        self.first_pre_zero_prediction_rms: float | None = None
        self.maximum_returned_abs_on_zero_steps = 0.0

    def begin_sample(self, seed: int) -> None:
        if self.current_sample_seed is not None:
            raise RuntimeError("zero-init sample already active")
        self.current_sample_seed = int(seed)
        self.current_step = 0

    def finish_sample(self) -> None:
        if self.current_sample_seed is None:
            raise RuntimeError("zero-init sample is not active")
        if self.current_step != self.config.total_steps:
            raise RuntimeError(
                "zero-init denoising-step mismatch: "
                f"actual={self.current_step}, expected={self.config.total_steps}"
            )
        self.completed_samples += 1
        self.current_sample_seed = None
        self.current_step = 0

    def abort_sample(self) -> None:
        if self.current_sample_seed is not None:
            self.aborted_samples += 1
        self.current_sample_seed = None
        self.current_step = 0

    def guided_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.current_sample_seed is None:
            raise RuntimeError("call begin_sample(seed) before zero-init generation")
        actual_task = str(kwargs.get("mask_type", "t2v")).strip().lower()
        if actual_task != self.config.expected_task:
            raise RuntimeError(
                f"zero-init expected task={self.config.expected_task!r}, got {actual_task!r}"
            )
        if self.current_step >= self.config.total_steps:
            raise RuntimeError("zero-init received too many denoising calls for one sample")

        step_index = self.current_step
        self.current_step += 1
        self.logical_forward_calls += 1
        output = self.original_forward(*args, **kwargs)
        if step_index >= self.config.zero_init_steps:
            return output

        prediction = output[0]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Transformer prediction is not a tensor")
        if self.first_pre_zero_prediction_rms is None:
            self.first_pre_zero_prediction_rms = float(
                prediction.detach().float().square().mean().sqrt().item()
            )
        zero_prediction = torch.zeros_like(prediction)
        returned_absmax = float(zero_prediction.detach().float().abs().max().item())
        self.maximum_returned_abs_on_zero_steps = max(
            self.maximum_returned_abs_on_zero_steps, returned_absmax
        )
        self.zeroed_forward_calls += 1
        return _replace_prediction(output, zero_prediction)

    def install(self) -> "HunyuanCFGZeroInit":
        controller = self

        def wrapped_forward(_transformer_self: Any, *args: Any, **kwargs: Any) -> Any:
            return controller.guided_forward(*args, **kwargs)

        self.transformer.forward = types.MethodType(wrapped_forward, self.transformer)
        self.transformer._sfg_guidance_controller = self
        return self

    def runtime_stats(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "logical_forward_calls": self.logical_forward_calls,
            "zeroed_forward_calls": self.zeroed_forward_calls,
            "expected_zeroed_forward_calls": (
                self.completed_samples * self.config.zero_init_steps
            ),
            "completed_samples": self.completed_samples,
            "aborted_samples": self.aborted_samples,
            "sample_active": self.current_sample_seed is not None,
            "first_pre_zero_prediction_rms": self.first_pre_zero_prediction_rms,
            "maximum_returned_abs_on_zero_steps": (
                self.maximum_returned_abs_on_zero_steps
            ),
        }


def install_hunyuan_cfg_zero_init(
    transformer: Any,
    *,
    total_steps: int,
    zero_init_steps: int,
    expected_task: str = "t2v",
) -> HunyuanCFGZeroInit:
    return HunyuanCFGZeroInit(
        transformer,
        total_steps=total_steps,
        zero_init_steps=zero_init_steps,
        expected_task=expected_task,
    ).install()


def cfg_zero_init_runtime_stats(transformer: Any) -> dict[str, Any] | None:
    controller = getattr(transformer, "_sfg_guidance_controller", None)
    if not isinstance(controller, HunyuanCFGZeroInit):
        return None
    return controller.runtime_stats()
