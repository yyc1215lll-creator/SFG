#!/usr/bin/env python3
"""Official CFG-Zero* controller for HunyuanVideo-1.5 packed native CFG.

The algebra is ported from the authors' Apache-2.0 reference implementation:
https://github.com/WeichenFan/CFG-Zero-star/tree/3162be1fba5dd0129ac8423ad6919d928f420a8d

HunyuanVideo-1.5 evaluates native CFG as one packed Transformer batch ordered
``[unconditional, conditional]`` and combines it after the Transformer call.
For non-zero-init steps this controller replaces only the packed unconditional
prediction ``v_u`` by ``alpha * v_u``.  The unchanged native CFG expression
then becomes exactly the authors' expression::

    alpha = <v_c, v_u> / (||v_u||^2 + 1e-8)
    v = alpha*v_u + w*(v_c - alpha*v_u)

For the requested initial steps both packed predictions are returned as exact
zero, making the scheduler velocity exact zero.  The legacy
``hunyuan_video_cfg_zero_init.py`` controller remains a separate baseline and
is intentionally not changed by this module.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch


OFFICIAL_REPOSITORY = "https://github.com/WeichenFan/CFG-Zero-star"
OFFICIAL_COMMIT = "3162be1fba5dd0129ac8423ad6919d928f420a8d"
OFFICIAL_EPSILON = 1e-8


@dataclass(frozen=True)
class CFGZeroStarConfig:
    total_steps: int
    zero_init_steps: int
    guidance_scale: float
    expected_task: str = "t2v"
    epsilon: float = OFFICIAL_EPSILON

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if not 1 <= self.zero_init_steps <= self.total_steps:
            raise ValueError("zero_init_steps must be in [1, total_steps]")
        if self.guidance_scale <= 1.0:
            raise ValueError("official CFG-Zero* requires native CFG scale > 1")
        if self.expected_task not in {"t2v", "i2v"}:
            raise ValueError("expected_task must be 't2v' or 'i2v'")
        if self.epsilon != OFFICIAL_EPSILON:
            raise ValueError(
                f"official CFG-Zero* pins epsilon to {OFFICIAL_EPSILON:g}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": "cfg_zero_star_official",
            "total_steps": self.total_steps,
            "zero_init_steps": self.zero_init_steps,
            "zero_init_fraction": self.zero_init_steps / self.total_steps,
            "guidance_scale": self.guidance_scale,
            "expected_task": self.expected_task,
            "optimized_scale_projection_enabled": True,
            "optimized_scale_formula": (
                "alpha=<v_cond,v_uncond>/(||v_uncond||^2+1e-8)"
            ),
            "guided_velocity_formula": (
                "alpha*v_uncond+w*(v_cond-alpha*v_uncond)"
            ),
            "epsilon": self.epsilon,
            "packed_branch_order": "unconditional_then_conditional",
            "intervention_location": "raw_packed_transformer_prediction",
            "post_cfg_velocity_is_exact_zero_on_zero_init_steps": True,
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
        }


def optimized_scale_official(
    conditional_prediction: torch.Tensor,
    unconditional_prediction: torch.Tensor,
    *,
    epsilon: float = OFFICIAL_EPSILON,
) -> torch.Tensor:
    """Return the authors' per-sample optimized scale in float32.

    The reference code applies the reduction over every non-batch element.  We
    explicitly promote that reduction to float32, matching the intent of the
    reference function's float32 autocast decorator while avoiding BF16
    accumulation error for a full video velocity field.
    """

    if conditional_prediction.shape != unconditional_prediction.shape:
        raise ValueError("conditional/unconditional prediction shapes must match")
    if conditional_prediction.ndim < 1 or conditional_prediction.shape[0] < 1:
        raise ValueError("CFG-Zero* predictions require a non-empty batch")
    if epsilon != OFFICIAL_EPSILON:
        raise ValueError(f"official CFG-Zero* pins epsilon to {OFFICIAL_EPSILON:g}")

    positive_flat = conditional_prediction.float().reshape(
        conditional_prediction.shape[0], -1
    )
    negative_flat = unconditional_prediction.float().reshape(
        unconditional_prediction.shape[0], -1
    )
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat.square(), dim=1, keepdim=True) + epsilon
    alpha = dot_product / squared_norm
    return alpha.reshape(
        conditional_prediction.shape[0],
        *([1] * (conditional_prediction.ndim - 1)),
    )


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
        "Hunyuan CFG-Zero* requires return_dict=False tuple/list Transformer output"
    )


class HunyuanCFGZeroStar:
    """Patch a Hunyuan Transformer and audit the complete official method."""

    def __init__(
        self,
        transformer: Any,
        *,
        total_steps: int,
        zero_init_steps: int,
        guidance_scale: float,
        expected_task: str = "t2v",
    ):
        if getattr(transformer, "_sfg_guidance_controller", None) is not None:
            raise RuntimeError("another SFG guidance controller is already installed")
        self.config = CFGZeroStarConfig(
            total_steps=total_steps,
            zero_init_steps=zero_init_steps,
            guidance_scale=guidance_scale,
            expected_task=expected_task,
        )
        self.transformer = transformer
        self.original_forward = transformer.forward
        self.logical_forward_calls = 0
        self.zeroed_forward_calls = 0
        self.optimized_scale_calls = 0
        self.packed_cfg_validation_calls = 0
        self.completed_samples = 0
        self.aborted_samples = 0
        self.current_sample_seed: int | None = None
        self.current_step = 0
        self.first_pre_zero_prediction_rms: float | None = None
        self.maximum_returned_abs_on_zero_steps = 0.0
        self.first_alpha: float | None = None
        self.minimum_alpha: float | None = None
        self.maximum_alpha: float | None = None
        self.nonfinite_alpha_count = 0
        self.first_scaled_unconditional_relative_change_rms: float | None = None
        self.maximum_scaled_unconditional_relative_change_rms = 0.0
        self.maximum_native_equivalence_error = 0.0

    def begin_sample(self, seed: int) -> None:
        if self.current_sample_seed is not None:
            raise RuntimeError("CFG-Zero* sample already active")
        self.current_sample_seed = int(seed)
        self.current_step = 0

    def finish_sample(self) -> None:
        if self.current_sample_seed is None:
            raise RuntimeError("CFG-Zero* sample is not active")
        if self.current_step != self.config.total_steps:
            raise RuntimeError(
                "CFG-Zero* denoising-step mismatch: "
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
            raise RuntimeError("call begin_sample(seed) before CFG-Zero* generation")
        actual_task = str(kwargs.get("mask_type", "t2v")).strip().lower()
        if actual_task != self.config.expected_task:
            raise RuntimeError(
                f"CFG-Zero* expected task={self.config.expected_task!r}, "
                f"got {actual_task!r}"
            )
        if self.current_step >= self.config.total_steps:
            raise RuntimeError("CFG-Zero* received too many denoising calls")

        step_index = self.current_step
        self.current_step += 1
        self.logical_forward_calls += 1
        output = self.original_forward(*args, **kwargs)
        prediction = output[0]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Transformer prediction is not a tensor")
        if prediction.ndim < 1 or prediction.shape[0] < 2 or prediction.shape[0] % 2:
            raise RuntimeError(
                "CFG-Zero* requires an even packed [unconditional, conditional] batch; "
                f"got shape={tuple(prediction.shape)}"
            )
        self.packed_cfg_validation_calls += 1

        if step_index < self.config.zero_init_steps:
            if self.first_pre_zero_prediction_rms is None:
                self.first_pre_zero_prediction_rms = float(
                    prediction.detach().float().square().mean().sqrt().item()
                )
            zero_prediction = torch.zeros_like(prediction)
            self.maximum_returned_abs_on_zero_steps = max(
                self.maximum_returned_abs_on_zero_steps,
                float(zero_prediction.detach().float().abs().max().item()),
            )
            self.zeroed_forward_calls += 1
            return _replace_prediction(output, zero_prediction)

        unconditional, conditional = prediction.chunk(2, dim=0)
        alpha_float = optimized_scale_official(conditional, unconditional)
        finite = torch.isfinite(alpha_float)
        nonfinite = int((~finite).sum().item())
        self.nonfinite_alpha_count += nonfinite
        if nonfinite:
            raise RuntimeError("official CFG-Zero* optimized scale became non-finite")
        alpha = alpha_float.to(dtype=unconditional.dtype)
        scaled_unconditional = unconditional * alpha

        change = (scaled_unconditional - unconditional).detach().float()
        denominator = unconditional.detach().float().square().mean().sqrt()
        relative_change_rms = float(
            (change.square().mean().sqrt() / denominator.clamp_min(1e-12)).item()
        )
        self.maximum_scaled_unconditional_relative_change_rms = max(
            self.maximum_scaled_unconditional_relative_change_rms,
            relative_change_rms,
        )

        alpha_min = float(alpha_float.detach().min().item())
        alpha_max = float(alpha_float.detach().max().item())
        if self.first_alpha is None:
            self.first_alpha = float(alpha_float.detach().flatten()[0].item())
            self.first_scaled_unconditional_relative_change_rms = relative_change_rms
        self.minimum_alpha = (
            alpha_min if self.minimum_alpha is None else min(self.minimum_alpha, alpha_min)
        )
        self.maximum_alpha = (
            alpha_max if self.maximum_alpha is None else max(self.maximum_alpha, alpha_max)
        )

        # Prove that reusing the frozen native packed-CFG combiner is exactly
        # the official expression, rather than an approximate reinterpretation.
        direct = scaled_unconditional + self.config.guidance_scale * (
            conditional - scaled_unconditional
        )
        packed = torch.cat([scaled_unconditional, conditional], dim=0)
        packed_unconditional, packed_conditional = packed.chunk(2, dim=0)
        native = packed_unconditional + self.config.guidance_scale * (
            packed_conditional - packed_unconditional
        )
        equivalence_error = float(
            (direct.detach().float() - native.detach().float()).abs().max().item()
        )
        self.maximum_native_equivalence_error = max(
            self.maximum_native_equivalence_error, equivalence_error
        )
        self.optimized_scale_calls += 1
        return _replace_prediction(output, packed)

    def install(self) -> "HunyuanCFGZeroStar":
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
            "optimized_scale_calls": self.optimized_scale_calls,
            "expected_optimized_scale_calls": (
                self.completed_samples
                * (self.config.total_steps - self.config.zero_init_steps)
            ),
            "packed_cfg_validation_calls": self.packed_cfg_validation_calls,
            "completed_samples": self.completed_samples,
            "aborted_samples": self.aborted_samples,
            "sample_active": self.current_sample_seed is not None,
            "first_pre_zero_prediction_rms": self.first_pre_zero_prediction_rms,
            "maximum_returned_abs_on_zero_steps": (
                self.maximum_returned_abs_on_zero_steps
            ),
            "first_alpha": self.first_alpha,
            "minimum_alpha": self.minimum_alpha,
            "maximum_alpha": self.maximum_alpha,
            "nonfinite_alpha_count": self.nonfinite_alpha_count,
            "first_scaled_unconditional_relative_change_rms": (
                self.first_scaled_unconditional_relative_change_rms
            ),
            "maximum_scaled_unconditional_relative_change_rms": (
                self.maximum_scaled_unconditional_relative_change_rms
            ),
            "maximum_native_equivalence_error": (
                self.maximum_native_equivalence_error
            ),
        }


def install_hunyuan_cfg_zero_star(
    transformer: Any,
    *,
    total_steps: int,
    zero_init_steps: int,
    guidance_scale: float,
    expected_task: str = "t2v",
) -> HunyuanCFGZeroStar:
    return HunyuanCFGZeroStar(
        transformer,
        total_steps=total_steps,
        zero_init_steps=zero_init_steps,
        guidance_scale=guidance_scale,
        expected_task=expected_task,
    ).install()


def cfg_zero_star_runtime_stats(transformer: Any) -> dict[str, Any] | None:
    controller = getattr(transformer, "_sfg_guidance_controller", None)
    if not isinstance(controller, HunyuanCFGZeroStar):
        return None
    return controller.runtime_stats()
