#!/usr/bin/env python3
"""S²-Guidance controller for HunyuanVideo-1.5 T2V/I2V NoGuidance.

The S² paper defines

    D_s2 = D_uncond + lambda (D_cond - D_uncond)
           - omega (D_sub - D_cond).

For the deliberately CFG-free comparison used here, ``lambda=1`` and the
formula is exactly ``D_clean + omega * (D_clean - D_sub)``.  No unconditional
text branch is constructed.  ``D_sub`` is evaluated with a deterministic,
per-timestep random subset of residual Transformer blocks skipped.

The paper's public repository does not currently include runnable code, so
this module is an explicit paper-formula implementation rather than a port of
an unpublished reference implementation.
"""

from __future__ import annotations

import hashlib
import json
import random
import types
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class S2Config:
    omega: float
    total_steps: int
    active_start: int
    active_end: int
    drop_count: int
    block_count: int
    controller_seed: int
    excluded_block_indices: tuple[int, ...] = (0,)
    expected_task: str = "t2v"

    def __post_init__(self) -> None:
        if self.omega < 0.0:
            raise ValueError("S2 omega must be non-negative")
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.active_start < self.active_end <= self.total_steps:
            raise ValueError("S2 active range must satisfy 0 <= start < end <= steps")
        if self.block_count != 54:
            raise ValueError("S2 is pinned to HunyuanVideo-1.5's 54 double blocks")
        if self.expected_task not in {"t2v", "i2v"}:
            raise ValueError("expected_task must be 't2v' or 'i2v'")
        candidates = set(range(self.block_count)) - set(self.excluded_block_indices)
        if not 1 <= self.drop_count <= len(candidates):
            raise ValueError("S2 drop_count exceeds the eligible block count")

    @property
    def active_steps(self) -> int:
        return self.active_end - self.active_start

    @property
    def candidate_block_indices(self) -> tuple[int, ...]:
        excluded = set(self.excluded_block_indices)
        return tuple(index for index in range(self.block_count) if index not in excluded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": "s2_guidance_noguidance",
            "formula": "D_clean + omega * (D_clean - D_sub)",
            "cfg_lambda": 1.0,
            "unconditional_branch_enabled": False,
            "omega": self.omega,
            "total_steps": self.total_steps,
            "active_start": self.active_start,
            "active_end_exclusive": self.active_end,
            "active_steps": self.active_steps,
            "active_fraction": self.active_steps / self.total_steps,
            "drop_count": self.drop_count,
            "block_count": self.block_count,
            "drop_fraction_of_all_blocks": self.drop_count / self.block_count,
            "excluded_block_indices": list(self.excluded_block_indices),
            "candidate_block_indices": list(self.candidate_block_indices),
            "controller_seed": self.controller_seed,
            "expected_task": self.expected_task,
            "clean_subnetwork_condition_reuse": "exact_same_forward_args",
            "mask_sampling": "sha256_seeded_python_random_without_replacement",
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
    raise TypeError("S2 requires the Transformer return_dict=False tuple/list output")


def _identity_double_block(_block_self: Any, *args: Any, **kwargs: Any) -> Any:
    if "img" in kwargs and "txt" in kwargs:
        return kwargs["img"], kwargs["txt"]
    if len(args) >= 2:
        return args[0], args[1]
    raise RuntimeError("could not recover img/txt inputs for a dropped double block")


class HunyuanS2Guidance:
    """Patch one Hunyuan Transformer with auditable stochastic block dropping."""

    def __init__(
        self,
        transformer: Any,
        *,
        omega: float,
        total_steps: int,
        active_start: int,
        active_end: int,
        drop_count: int,
        controller_seed: int,
        expected_task: str = "t2v",
    ):
        if getattr(transformer, "_sfg_guidance_controller", None) is not None:
            raise RuntimeError("another SFG guidance controller is already installed")
        double_blocks = getattr(transformer, "double_blocks", None)
        single_blocks = getattr(transformer, "single_blocks", ())
        if double_blocks is None:
            raise TypeError("Hunyuan Transformer has no double_blocks")
        if len(double_blocks) != 54 or len(single_blocks) != 0:
            raise RuntimeError(
                "S2 requires the pinned 54-double/0-single HunyuanVideo-1.5 model"
            )
        self.config = S2Config(
            omega=omega,
            total_steps=total_steps,
            active_start=active_start,
            active_end=active_end,
            drop_count=drop_count,
            block_count=len(double_blocks),
            controller_seed=int(controller_seed),
            expected_task=expected_task,
        )
        self.transformer = transformer
        self.double_blocks = double_blocks
        self.original_forward = transformer.forward
        self.logical_forward_calls = 0
        self.clean_forward_calls = 0
        self.subnetwork_forward_calls = 0
        self.dropped_block_calls = 0
        self.completed_samples = 0
        self.aborted_samples = 0
        self.current_sample_seed: int | None = None
        self.current_step = 0
        self.first_residual_relative_rms: float | None = None
        self._mask_schedule_digest = hashlib.sha256()
        self._first_sample_seed: int | None = None
        self._first_sample_masks: list[dict[str, Any]] = []
        self._capture_current_sample_masks = False

    def begin_sample(self, seed: int) -> None:
        if self.current_sample_seed is not None:
            raise RuntimeError("S2 sample already active")
        self.current_sample_seed = int(seed)
        self.current_step = 0
        if self._first_sample_seed is None:
            self._first_sample_seed = int(seed)
            self._capture_current_sample_masks = True
        else:
            self._capture_current_sample_masks = False

    def finish_sample(self) -> None:
        if self.current_sample_seed is None:
            raise RuntimeError("S2 sample is not active")
        if self.current_step != self.config.total_steps:
            raise RuntimeError(
                "S2 denoising-step mismatch: "
                f"actual={self.current_step}, expected={self.config.total_steps}"
            )
        self.completed_samples += 1
        self.current_sample_seed = None
        self.current_step = 0
        self._capture_current_sample_masks = False

    def abort_sample(self) -> None:
        if self.current_sample_seed is not None:
            self.aborted_samples += 1
        self.current_sample_seed = None
        self.current_step = 0
        self._capture_current_sample_masks = False

    def mask_for(self, sample_seed: int, step_index: int) -> tuple[int, ...]:
        if not 0 <= step_index < self.config.total_steps:
            raise ValueError("step_index outside the denoising schedule")
        material = (
            f"hunyuan-s2-v1|controller={self.config.controller_seed}|"
            f"sample={int(sample_seed)}|step={int(step_index)}"
        ).encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
        generator = random.Random(local_seed)
        selected = generator.sample(
            self.config.candidate_block_indices, self.config.drop_count
        )
        return tuple(sorted(selected))

    def _run_subnetwork(
        self, selected: tuple[int, ...], *args: Any, **kwargs: Any
    ) -> Any:
        originals: list[tuple[Any, Any]] = []
        try:
            for block_index in selected:
                block = self.double_blocks[block_index]
                originals.append((block, block.forward))
                block.forward = types.MethodType(_identity_double_block, block)
            return self.original_forward(*args, **kwargs)
        finally:
            for block, original_forward in originals:
                block.forward = original_forward

    def guided_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.current_sample_seed is None:
            raise RuntimeError("call begin_sample(seed) before S2 generation")
        actual_task = str(kwargs.get("mask_type", "t2v")).strip().lower()
        if actual_task != self.config.expected_task:
            raise RuntimeError(
                f"S2 expected task={self.config.expected_task!r}, got {actual_task!r}"
            )
        if self.current_step >= self.config.total_steps:
            raise RuntimeError("S2 received too many denoising calls for one sample")

        step_index = self.current_step
        sample_seed = self.current_sample_seed
        self.current_step += 1
        self.logical_forward_calls += 1
        clean_output = self.original_forward(*args, **kwargs)
        self.clean_forward_calls += 1
        if not self.config.active_start <= step_index < self.config.active_end:
            return clean_output

        selected = self.mask_for(sample_seed, step_index)
        if 0 in selected or len(selected) != self.config.drop_count:
            raise RuntimeError(f"invalid S2 block mask: {selected}")
        sub_output = self._run_subnetwork(selected, *args, **kwargs)
        self.subnetwork_forward_calls += 1
        self.dropped_block_calls += len(selected)

        mask_record = {
            "sample_seed": sample_seed,
            "step_index": step_index,
            "dropped_blocks": list(selected),
        }
        self._mask_schedule_digest.update(
            (json.dumps(mask_record, sort_keys=True) + "\n").encode("utf-8")
        )
        if self._capture_current_sample_masks:
            self._first_sample_masks.append(mask_record)

        clean_prediction = clean_output[0]
        sub_prediction = sub_output[0]
        if not isinstance(clean_prediction, torch.Tensor) or not isinstance(
            sub_prediction, torch.Tensor
        ):
            raise TypeError("S2 Transformer predictions must be tensors")
        residual = clean_prediction - sub_prediction
        guided_prediction = clean_prediction + self.config.omega * residual
        if self.first_residual_relative_rms is None:
            residual_rms = residual.detach().float().square().mean().sqrt()
            clean_rms = (
                clean_prediction.detach().float().square().mean().sqrt().clamp_min(1e-8)
            )
            self.first_residual_relative_rms = float((residual_rms / clean_rms).item())
        return _replace_prediction(clean_output, guided_prediction)

    def install(self) -> "HunyuanS2Guidance":
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
            "subnetwork_forward_calls": self.subnetwork_forward_calls,
            "dropped_block_calls": self.dropped_block_calls,
            "expected_dropped_block_calls": (
                self.subnetwork_forward_calls * self.config.drop_count
            ),
            "completed_samples": self.completed_samples,
            "aborted_samples": self.aborted_samples,
            "sample_active": self.current_sample_seed is not None,
            "first_residual_relative_rms": self.first_residual_relative_rms,
            "mask_schedule_sha256": self._mask_schedule_digest.hexdigest(),
            "first_sample_seed": self._first_sample_seed,
            "first_sample_masks": list(self._first_sample_masks),
        }


def install_hunyuan_s2_guidance(
    transformer: Any,
    *,
    omega: float,
    total_steps: int,
    active_start: int,
    active_end: int,
    drop_count: int,
    controller_seed: int,
    expected_task: str = "t2v",
) -> HunyuanS2Guidance:
    return HunyuanS2Guidance(
        transformer,
        omega=omega,
        total_steps=total_steps,
        active_start=active_start,
        active_end=active_end,
        drop_count=drop_count,
        controller_seed=controller_seed,
        expected_task=expected_task,
    ).install()


def s2_runtime_stats(transformer: Any) -> dict[str, Any] | None:
    controller = getattr(transformer, "_sfg_guidance_controller", None)
    if not isinstance(controller, HunyuanS2Guidance):
        return None
    return controller.runtime_stats()
