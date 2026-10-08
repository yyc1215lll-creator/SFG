#!/usr/bin/env python3
"""Auditable STG controller for HunyuanVideo-1.5 T2V.

The authors' HunyuanVideo implementation evaluates a clean prediction and a
perturbed prediction whose selected Transformer block is replaced by an
identity mapping, then returns

    D_stg = D_clean + scale * (D_clean - D_perturbed).

The authors' published Hunyuan example uses the original CFG-distilled
HunyuanVideo architecture (20 dual-stream plus 40 single-stream blocks),
embedded guidance 6, block index 2, scale 1.0, and 30 denoising steps.  Those
model-specific defaults are not directly transferable to HunyuanVideo-1.5
Base, whose backbone has 54 double-stream blocks and no single-stream blocks.

This module therefore supports both the historical literal-default port and
an explicitly labelled HV1.5 Base adaptation.  The latter keeps the official
clean-minus-perturbed direction but exposes an active step window and records
the architecture mapping.  It does not use an unconditional text branch,
train weights, or introduce an external model.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch


OFFICIAL_REPOSITORY = "https://github.com/junhahyung/STGuidance"
OFFICIAL_COMMIT = "d9e7be5dadfc8d0f53855b2a56e478a5e64b2ca4"
OFFICIAL_HUNYUAN_SOURCE_SHA256 = (
    "f0c911bddb818d050b45e801f2a14b432ac7b588ed5af3b3c2c2ede2b1715c99"
)


@dataclass(frozen=True)
class STGConfig:
    scale: float
    total_steps: int
    active_start: int
    active_end: int
    block_indices: tuple[int, ...]
    block_count: int
    port_kind: str
    expected_task: str = "t2v"

    def __post_init__(self) -> None:
        if self.scale < 0.0:
            raise ValueError("STG scale must be non-negative")
        if self.total_steps < 1:
            raise ValueError("STG total_steps must be positive")
        if not 0 <= self.active_start < self.active_end <= self.total_steps:
            raise ValueError("invalid STG active-step interval")
        if self.block_count != 54:
            raise ValueError("STG is pinned to HunyuanVideo-1.5's 54 double blocks")
        if self.expected_task != "t2v":
            raise ValueError("the official Hunyuan STG port is currently pinned to T2V")
        if not self.block_indices:
            raise ValueError("STG requires at least one perturbed block")
        if len(set(self.block_indices)) != len(self.block_indices):
            raise ValueError("STG block indices must be unique")
        if any(index < 0 or index >= self.block_count for index in self.block_indices):
            raise ValueError("STG block index is outside the double-block stack")
        if self.port_kind not in ("literal_default_port", "hv15_base_adapted"):
            raise ValueError(f"unsupported STG port kind: {self.port_kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": (
                "stg_hv15_base_adapted_noguidance"
                if self.port_kind == "hv15_base_adapted"
                else "stg_literal_hunyuan_default_port_noguidance"
            ),
            "formula": "D_clean + scale * (D_clean - D_perturbed)",
            "scale": self.scale,
            "total_steps": self.total_steps,
            "active_start": self.active_start,
            "active_end_exclusive": self.active_end,
            "active_steps": self.active_end - self.active_start,
            "block_indices": list(self.block_indices),
            "block_count": self.block_count,
            "port_kind": self.port_kind,
            "reference_architecture": "20_dual_plus_40_single_cfg_distilled",
            "target_architecture": "54_double_base",
            "block_mapping": "zero_based_global_depth_position",
            "unconditional_branch_enabled": False,
            "guidance_rescaling_enabled": False,
            "expected_task": self.expected_task,
            "clean_perturbed_condition_reuse": "exact_same_forward_args",
            "perturbation": "selected_double_block_identity_mapping",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "official_hunyuan_source_sha256": OFFICIAL_HUNYUAN_SOURCE_SHA256,
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
    raise TypeError("STG requires return_dict=False tuple/list Transformer output")


def _identity_double_block(_block_self: Any, *args: Any, **kwargs: Any) -> Any:
    """Match the official Hunyuan STG block perturbation exactly."""
    if "img" in kwargs and "txt" in kwargs:
        return kwargs["img"], kwargs["txt"]
    if len(args) >= 2:
        return args[0], args[1]
    raise RuntimeError("could not recover img/txt inputs for STG identity block")


class HunyuanSTG:
    """Patch a HunyuanVideo-1.5 Transformer with auditable STG."""

    def __init__(
        self,
        transformer: Any,
        *,
        scale: float,
        total_steps: int,
        active_start: int = 0,
        active_end: int | None = None,
        block_indices: tuple[int, ...] = (2,),
        port_kind: str = "literal_default_port",
        expected_task: str = "t2v",
    ) -> None:
        if getattr(transformer, "_sfg_guidance_controller", None) is not None:
            raise RuntimeError("another SFG guidance controller is already installed")
        double_blocks = getattr(transformer, "double_blocks", None)
        single_blocks = getattr(transformer, "single_blocks", ())
        if double_blocks is None:
            raise TypeError("Hunyuan Transformer has no double_blocks")
        if len(double_blocks) != 54 or len(single_blocks) != 0:
            raise RuntimeError(
                "STG requires the pinned 54-double/0-single HunyuanVideo-1.5 model"
            )
        self.config = STGConfig(
            scale=float(scale),
            total_steps=int(total_steps),
            active_start=int(active_start),
            active_end=(
                int(total_steps) if active_end is None else int(active_end)
            ),
            block_indices=tuple(int(index) for index in block_indices),
            block_count=len(double_blocks),
            port_kind=str(port_kind),
            expected_task=expected_task,
        )
        self.transformer = transformer
        self.double_blocks = double_blocks
        self.original_forward = transformer.forward
        self.logical_forward_calls = 0
        self.clean_forward_calls = 0
        self.perturbed_forward_calls = 0
        self.identity_block_calls = 0
        self.restored_block_calls = 0
        self.completed_samples = 0
        self.aborted_samples = 0
        self.current_sample_seed: int | None = None
        self.current_step = 0
        self.first_residual_relative_rms: float | None = None
        self.maximum_residual_relative_rms = 0.0
        self.first_guidance_correction_relative_rms: float | None = None
        self.maximum_guidance_correction_relative_rms = 0.0

    def begin_sample(self, seed: int) -> None:
        if self.current_sample_seed is not None:
            raise RuntimeError("STG sample already active")
        self.current_sample_seed = int(seed)
        self.current_step = 0

    def finish_sample(self) -> None:
        if self.current_sample_seed is None:
            raise RuntimeError("STG sample is not active")
        if self.current_step != self.config.total_steps:
            raise RuntimeError(
                "STG denoising-step mismatch: "
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

    def _run_perturbed(self, *args: Any, **kwargs: Any) -> Any:
        originals: list[tuple[Any, Any]] = []
        try:
            for block_index in self.config.block_indices:
                block = self.double_blocks[block_index]
                originals.append((block, block.forward))
                block.forward = types.MethodType(_identity_double_block, block)
                self.identity_block_calls += 1
            return self.original_forward(*args, **kwargs)
        finally:
            for block, original_forward in originals:
                block.forward = original_forward
                self.restored_block_calls += 1

    def guided_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.current_sample_seed is None:
            raise RuntimeError("call begin_sample(seed) before STG generation")
        actual_task = str(kwargs.get("mask_type", "t2v")).strip().lower()
        if actual_task != self.config.expected_task:
            raise RuntimeError(
                f"STG expected task={self.config.expected_task!r}, got {actual_task!r}"
            )
        if self.current_step >= self.config.total_steps:
            raise RuntimeError("STG received too many denoising calls for one sample")

        step_index = self.current_step
        self.current_step += 1
        self.logical_forward_calls += 1
        clean_output = self.original_forward(*args, **kwargs)
        self.clean_forward_calls += 1
        if not (
            self.config.active_start <= step_index < self.config.active_end
        ) or self.config.scale == 0.0:
            return clean_output

        perturbed_output = self._run_perturbed(*args, **kwargs)
        self.perturbed_forward_calls += 1

        clean_prediction = clean_output[0]
        perturbed_prediction = perturbed_output[0]
        if not isinstance(clean_prediction, torch.Tensor) or not isinstance(
            perturbed_prediction, torch.Tensor
        ):
            raise TypeError("STG Transformer predictions must be tensors")
        residual = clean_prediction - perturbed_prediction
        guided_prediction = clean_prediction + self.config.scale * residual
        residual_rms = residual.detach().float().square().mean().sqrt()
        clean_rms = clean_prediction.detach().float().square().mean().sqrt().clamp_min(1e-8)
        relative_rms = float((residual_rms / clean_rms).item())
        correction_relative_rms = self.config.scale * relative_rms
        if self.first_residual_relative_rms is None:
            self.first_residual_relative_rms = relative_rms
            self.first_guidance_correction_relative_rms = correction_relative_rms
        self.maximum_residual_relative_rms = max(
            self.maximum_residual_relative_rms, relative_rms
        )
        self.maximum_guidance_correction_relative_rms = max(
            self.maximum_guidance_correction_relative_rms,
            correction_relative_rms,
        )
        return _replace_prediction(clean_output, guided_prediction)

    def install(self) -> "HunyuanSTG":
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
            "clean_forward_calls": self.clean_forward_calls,
            "perturbed_forward_calls": self.perturbed_forward_calls,
            "identity_block_calls": self.identity_block_calls,
            "expected_identity_block_calls": (
                self.perturbed_forward_calls * len(self.config.block_indices)
            ),
            "restored_block_calls": self.restored_block_calls,
            "completed_samples": self.completed_samples,
            "aborted_samples": self.aborted_samples,
            "sample_active": self.current_sample_seed is not None,
            "first_residual_relative_rms": self.first_residual_relative_rms,
            "maximum_residual_relative_rms": self.maximum_residual_relative_rms,
            "first_guidance_correction_relative_rms": (
                self.first_guidance_correction_relative_rms
            ),
            "maximum_guidance_correction_relative_rms": (
                self.maximum_guidance_correction_relative_rms
            ),
        }


def install_hunyuan_stg(
    transformer: Any,
    *,
    scale: float,
    total_steps: int,
    active_start: int = 0,
    active_end: int | None = None,
    block_indices: tuple[int, ...] = (2,),
    port_kind: str = "literal_default_port",
    expected_task: str = "t2v",
) -> HunyuanSTG:
    return HunyuanSTG(
        transformer,
        scale=scale,
        total_steps=total_steps,
        active_start=active_start,
        active_end=active_end,
        block_indices=block_indices,
        port_kind=port_kind,
        expected_task=expected_task,
    ).install()


def stg_runtime_stats(transformer: Any) -> dict[str, Any] | None:
    controller = getattr(transformer, "_sfg_guidance_controller", None)
    if not isinstance(controller, HunyuanSTG):
        return None
    return controller.runtime_stats()
