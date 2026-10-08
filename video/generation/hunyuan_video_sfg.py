#!/usr/bin/env python3
"""Explicit directional Bridge Guidance for HunyuanVideo-1.5.

This is the HunyuanVideo analogue of the explicit SD3/SD3.5 SFGS/SFGX path in
``latent_sd35.py``.  A combined counterfactual can weaken the useful
image-query <- text-value bridge (SFGS) while strengthening the adverse
text-query <- image-value bridge (SFGX):

    D_guided = D_clean + omega * (D_clean - D_counterfactual)

``D_counterfactual`` uses the same latent, timestep, and prompt as ``D_clean``.  Under
the project convention, SFGS ``u=+0.1`` gives a 0.9x image<-text bridge and SFGX
``u=-0.1`` gives a 1.1x text<-image bridge.  The same-sign SFGX control
``u=+0.1`` instead gives a 0.9x text<-image bridge.  The two model evaluations are
executed sequentially to retain the same counterfactual as SD3.5 without
doubling peak activation memory.  A first-N-step schedule can restrict this
extra branch to the early denoising trajectory.

The upstream HunyuanVideo source is not edited.  Installation wraps one loaded
transformer instance and temporarily replaces the module-level
``parallel_attention`` call used by its double-stream blocks.

For SFGS-containing branches, image and text queries are evaluated directly with
their respective value scaling.  This preserves the full-key softmax while
avoiding a second, expensive image-query attention decomposition.
"""

from __future__ import annotations

import importlib
import types
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.utils.communications import all_gather, all_to_all_4D


@dataclass(frozen=True)
class ExplicitSFGConfig:
    """Frozen runtime definition of the SD3.5-style SFGS/SFGX intervention."""

    sfg_s_u: float = 0.0
    sfg_strength_u: float = -0.1
    omega: float = 1.0
    layer_indices: tuple[int, ...] = ()
    direction: str = "text_to_image"
    branch_mode: str = "sequential"
    total_steps: int = 1
    active_steps: int = 1
    normclip_tau: float = 999.0
    orthogonal: bool = False
    condition_scope: str = "joint_encoder"
    expected_task: str | None = None
    expected_vision_tokens: int | None = None

    @property
    def sfg_s_bridge_delta(self) -> float:
        return -float(self.sfg_s_u)

    @property
    def sfg_x_bridge_delta(self) -> float:
        return -float(self.sfg_strength_u)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sfg_s_bridge_delta"] = self.sfg_s_bridge_delta
        payload["sfg_s_bridge_multiplier"] = 1.0 + self.sfg_s_bridge_delta
        payload["sfg_x_bridge_delta"] = self.sfg_x_bridge_delta
        payload["sfg_x_bridge_multiplier"] = 1.0 + self.sfg_x_bridge_delta
        # Backward-compatible names for historical SFGX-only audits.
        payload["strong_bridge_delta"] = self.sfg_x_bridge_delta
        payload["strong_bridge_multiplier"] = 1.0 + self.sfg_x_bridge_delta
        payload["active_fraction"] = self.active_steps / self.total_steps
        payload["formula"] = "D_clean + omega * (D_clean - D_counterfactual)"
        return payload


def _normalize_layers(layers: str | Iterable[int], depth: int) -> tuple[int, ...]:
    if depth < 1:
        raise ValueError("SFGX requires at least one double-stream block")
    if isinstance(layers, str):
        value = layers.strip().lower()
        if value in {"", "all"}:
            selected = tuple(range(depth))
        else:
            try:
                selected = tuple(int(item.strip()) for item in value.split(",") if item.strip())
            except ValueError as exc:
                raise ValueError(f"invalid SFGX layer list: {layers!r}") from exc
    else:
        selected = tuple(int(item) for item in layers)
    selected = tuple(sorted(set(selected)))
    if not selected:
        raise ValueError("SFGX layer selection is empty")
    invalid = [index for index in selected if index < 0 or index >= depth]
    if invalid:
        raise ValueError(f"SFGX layer indices outside [0, {depth - 1}]: {invalid}")
    return selected


def _torch_value_slice_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_mask: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    """SDPA fallback for the contribution of a value slice.

    Tensors use Hunyuan's ``[batch, sequence, heads, head_dim]`` layout.
    ``value`` already contains zeros outside the selected source slice.
    """

    attended = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=key_mask[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
    )
    attended = attended * query_mask[:, None, :, None].to(attended.dtype)
    return attended.transpose(1, 2).contiguous()


def _flash_value_slice_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_mask: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    """FlashAttention varlen evaluation with separate query and KV masks."""

    from einops import rearrange
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, query_len, heads, _head_dim = query.shape
    query_flat, query_indices, cu_query, max_query, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), query_mask
    )
    key_flat, _key_indices, cu_key, max_key, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_mask
    )
    value_flat, _value_indices, _cu_value, _max_value, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_mask
    )
    query_unpad = rearrange(query_flat, "n (h d) -> n h d", h=heads)
    key_unpad = rearrange(key_flat, "n (h d) -> n h d", h=heads)
    value_unpad = rearrange(value_flat, "n (h d) -> n h d", h=heads)
    output_unpad = flash_attn_varlen_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_query,
        cu_key,
        max_query,
        max_key,
        0.0,
        softmax_scale=None,
        causal=False,
    )
    return rearrange(
        pad_input(
            rearrange(output_unpad, "n h d -> n (h d)"),
            query_indices,
            batch_size,
            query_len,
        ),
        "b s (h d) -> b s h d",
        h=heads,
    )


def _value_slice_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_mask: torch.Tensor,
    key_mask: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    if query.is_cuda and query.dtype in {torch.float16, torch.bfloat16}:
        return (
            _flash_value_slice_attention(query, key, value, query_mask, key_mask),
            "flash_attn_varlen",
        )
    return _torch_value_slice_attention(query, key, value, query_mask, key_mask), "torch_sdpa"


def _direct_split_bridge_attention(
    q: tuple[torch.Tensor, torch.Tensor],
    k: tuple[torch.Tensor, torch.Tensor],
    v: tuple[torch.Tensor, torch.Tensor],
    text_mask: torch.Tensor | None,
    *,
    sfg_s_bridge_delta: float,
    sfg_x_bridge_delta: float,
    condition_scope: str = "joint_encoder",
    vision_tokens: int | None = None,
) -> tuple[torch.Tensor, str]:
    """Evaluate the modified joint attention without a delta recomputation.

    Video and encoder queries share the original full key set and therefore
    the original softmax normalization.  They use different value tensors so
    only the requested cross-modal value slice is scaled.  ``joint_encoder``
    preserves the historical two-query-partition operator.  ``text_only``
    splits the I2V encoder queries at the leading vision-token boundary so no
    attention edge involving those reference-image semantic tokens is scaled.
    """

    image_query, text_query = q
    image_key, text_key = k
    image_value, text_value = v
    batch_size, text_len = text_query.shape[:2]
    if text_mask is None:
        text_mask_bool = torch.ones(
            (batch_size, text_len), device=text_query.device, dtype=torch.bool
        )
    else:
        if tuple(text_mask.shape) != (batch_size, text_len):
            raise ValueError(
                f"unexpected text mask {tuple(text_mask.shape)} "
                f"for text Q {tuple(text_query.shape)}"
            )
        text_mask_bool = text_mask.to(device=text_query.device, dtype=torch.bool)

    condition_scope = str(condition_scope).strip().lower()
    if condition_scope not in {"joint_encoder", "text_only"}:
        raise ValueError(f"unsupported SFGS/SFGX condition scope: {condition_scope!r}")
    if condition_scope == "text_only":
        if vision_tokens is None:
            raise ValueError("text_only SFG requires the I2V vision-token boundary")
        vision_tokens = int(vision_tokens)
        if not 1 <= vision_tokens < text_len:
            raise ValueError(
                "text_only SFG requires 1 <= vision_tokens < encoder length; "
                f"vision={vision_tokens}, encoder={text_len}"
            )
        if not bool(torch.all(text_mask_bool[:, :vision_tokens]).item()):
            raise ValueError(
                "text_only SFG requires fully-valid leading reference-image tokens"
            )
        if not bool(torch.all(text_mask_bool[:, vision_tokens:].any(dim=1)).item()):
            raise ValueError("text_only SFG requires at least one valid prompt token")

    image_mask = torch.ones(
        (batch_size, image_key.shape[1]), device=image_query.device, dtype=torch.bool
    )
    key_mask = torch.cat((image_mask, text_mask_bool), dim=1)
    key_all = torch.cat((image_key, text_key), dim=1)

    if condition_scope == "joint_encoder":
        # Historical/full SFG: I2V image-semantic and prompt values are one
        # joint encoder condition and receive the same SFGS multiplier.
        sfg_s_text_value = text_value * (1.0 + float(sfg_s_bridge_delta))
    else:
        # Text-only SFG: preserve the first ``vision_tokens`` values exactly;
        # only the following prompt-value slice is scaled for video queries.
        sfg_s_text_value = torch.cat(
            (
                text_value[:, :vision_tokens],
                text_value[:, vision_tokens:] * (1.0 + float(sfg_s_bridge_delta)),
            ),
            dim=1,
        )
    image_query_value = torch.cat((image_value, sfg_s_text_value), dim=1)
    image_output, image_backend = _value_slice_attention(
        image_query,
        key_all,
        image_query_value,
        image_mask,
        key_mask,
    )
    del sfg_s_text_value, image_query_value

    sfg_x_image_value = (
        image_value * (1.0 + float(sfg_x_bridge_delta))
        if sfg_x_bridge_delta
        else image_value
    )
    if condition_scope == "joint_encoder":
        text_query_value = torch.cat((sfg_x_image_value, text_value), dim=1)
        text_output, text_backend = _value_slice_attention(
            text_query,
            key_all,
            text_query_value,
            text_mask_bool,
            key_mask,
        )
        text_backends = (text_backend,)
    else:
        # Reference-image semantic queries use the clean value tensor.  Prompt
        # queries alone see scaled video values.  The three query partitions
        # still add up to one full joint-attention query sequence.
        clean_value = torch.cat((image_value, text_value), dim=1)
        vision_output, vision_backend = _value_slice_attention(
            text_query[:, :vision_tokens],
            key_all,
            clean_value,
            text_mask_bool[:, :vision_tokens],
            key_mask,
        )
        prompt_query_value = torch.cat((sfg_x_image_value, text_value), dim=1)
        prompt_output, prompt_backend = _value_slice_attention(
            text_query[:, vision_tokens:],
            key_all,
            prompt_query_value,
            text_mask_bool[:, vision_tokens:],
            key_mask,
        )
        text_output = torch.cat((vision_output, prompt_output), dim=1)
        text_backends = (vision_backend, prompt_backend)
    if any(backend != image_backend for backend in text_backends):
        raise RuntimeError(
            "SFGS/SFGX direct attention backend mismatch: "
            f"video={image_backend}, encoder={text_backends}"
        )
    attended = torch.cat((image_output, text_output), dim=1)
    backend_suffix = "direct_split" if condition_scope == "joint_encoder" else "direct_split_text_only"
    return attended.reshape(batch_size, attended.shape[1], -1), f"{image_backend}_{backend_suffix}"


class HunyuanExplicitSFG:
    """Controller installed on one HunyuanVideo-1.5 transformer instance."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        sfg_s_u: float,
        sfg_strength_u: float,
        omega: float,
        layers: str | Iterable[int],
        total_steps: int,
        active_steps: int,
        expected_task: str | None = None,
        expected_vision_tokens: int | None = None,
        condition_scope: str = "joint_encoder",
    ) -> None:
        sfg_s_u = float(sfg_s_u)
        sfg_strength_u = float(sfg_strength_u)
        omega = float(omega)
        total_steps = int(total_steps)
        active_steps = int(active_steps)
        if expected_task is not None:
            expected_task = str(expected_task).strip().lower()
            if expected_task not in {"t2v", "i2v"}:
                raise ValueError("expected_task must be one of: t2v, i2v")
        if expected_vision_tokens is not None:
            expected_vision_tokens = int(expected_vision_tokens)
            if expected_vision_tokens < 1:
                raise ValueError("expected_vision_tokens must be positive")
        if expected_task == "i2v" and expected_vision_tokens is None:
            raise ValueError("I2V SFG requires expected_vision_tokens")
        if expected_task != "i2v" and expected_vision_tokens is not None:
            raise ValueError("expected_vision_tokens is only valid for expected_task='i2v'")
        condition_scope = str(condition_scope).strip().lower()
        if condition_scope not in {"joint_encoder", "text_only"}:
            raise ValueError(
                "condition_scope must be one of: joint_encoder, text_only"
            )
        if condition_scope == "text_only" and expected_task != "i2v":
            raise ValueError("text_only SFG is only defined for expected_task='i2v'")
        if not 0.0 <= sfg_s_u <= 1.0:
            raise ValueError(
                "explicit SFGS requires sfg_s_u in [0, 1]; "
                "positive u weakens the image-query <- text-value counterfactual"
            )
        if not -1.0 <= sfg_strength_u <= 1.0:
            raise ValueError(
                "explicit SFGX requires sfg_strength_u in [-1, 1]; "
                "negative u strengthens and positive u weakens the "
                "text-query <- image-value counterfactual"
            )
        if sfg_s_u == 0.0 and sfg_strength_u == 0.0:
            raise ValueError("explicit SFGS/SFGX requires at least one non-zero bridge u")
        if omega < 0.0:
            raise ValueError("explicit SFGS/SFGX omega must be >= 0")
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if not 1 <= active_steps <= total_steps:
            raise ValueError("active_steps must be in [1, total_steps]")

        double_blocks = getattr(transformer, "double_blocks", None)
        single_blocks = getattr(transformer, "single_blocks", None)
        if double_blocks is None or single_blocks is None:
            raise TypeError("expected a HunyuanVideo transformer with double_blocks/single_blocks")
        if len(single_blocks) != 0:
            raise ValueError(
                "this Hunyuan SFGX path is intentionally pinned to the all-double-stream "
                f"HunyuanVideo-1.5 checkpoint; found {len(single_blocks)} single blocks"
            )
        selected = _normalize_layers(layers, len(double_blocks))
        if sfg_s_u and sfg_strength_u:
            direction = "both"
        elif sfg_s_u:
            direction = "image_to_text"
        else:
            direction = "text_to_image"
        self.config = ExplicitSFGConfig(
            sfg_s_u=sfg_s_u,
            sfg_strength_u=sfg_strength_u,
            omega=omega,
            layer_indices=selected,
            direction=direction,
            total_steps=total_steps,
            active_steps=active_steps,
            condition_scope=condition_scope,
            expected_task=expected_task,
            expected_vision_tokens=expected_vision_tokens,
        )
        self.transformer = transformer
        self.selected_layers = frozenset(selected)
        self.strong_branch_active = False
        self.clean_forward_calls = 0
        self.strong_forward_calls = 0
        self.logical_forward_calls = 0
        self.bridge_attention_calls = 0
        self.sfg_s_attention_calls = 0
        self.sfg_x_attention_calls = 0
        self.direct_bridge_attention_calls = 0
        self.legacy_bridge_attention_calls = 0
        self.bridge_backend: str | None = None
        self.first_residual_relative_rms: float | None = None
        self.task_validation_calls = 0
        self.observed_vision_tokens: int | None = None
        self.observed_encoder_tokens: int | None = None
        self.observed_valid_encoder_tokens: int | None = None
        self.observed_valid_prompt_tokens: int | None = None
        self._joint_attention_layout_validated = False

        transformer_module = importlib.import_module(
            "hyvideo.models.transformers.hunyuanvideo_1_5_transformer"
        )
        existing = getattr(transformer_module, "_sfg_explicit_sfg_controller", None)
        if existing is not None:
            raise RuntimeError("Hunyuan explicit SFGX is already installed in this process")
        self.transformer_module = transformer_module
        self.original_parallel_attention = transformer_module.parallel_attention
        self.original_forward = transformer.forward

    def _bridge_contribution(
        self,
        q: tuple[torch.Tensor, torch.Tensor],
        k: tuple[torch.Tensor, torch.Tensor],
        v: tuple[torch.Tensor, torch.Tensor],
        text_mask: torch.Tensor | None,
        *,
        direction: str,
    ) -> torch.Tensor:
        image_query, text_query = q
        image_key, text_key = k
        image_value, text_value = v
        parallel_dims = get_parallel_state()
        sp_enabled = parallel_dims.sp_enabled

        if sp_enabled:
            sp_group = parallel_dims.sp_group
            sp_size = parallel_dims.sp
            sp_rank = parallel_dims.sp_rank
            image_query = all_to_all_4D(
                image_query, sp_group, scatter_dim=2, gather_dim=1
            )
            image_key = all_to_all_4D(
                image_key, sp_group, scatter_dim=2, gather_dim=1
            )
            image_value = all_to_all_4D(
                image_value, sp_group, scatter_dim=2, gather_dim=1
            )

            def local_heads(tensor: torch.Tensor) -> torch.Tensor:
                head_count = tensor.shape[2]
                if head_count % sp_size != 0:
                    raise ValueError(
                        f"text attention heads {head_count} are not divisible by SP size {sp_size}"
                    )
                width = head_count // sp_size
                return tensor.narrow(2, sp_rank * width, width)

            text_query = local_heads(text_query)
            text_key = local_heads(text_key)
            text_value = local_heads(text_value)

        batch_size, text_len = text_query.shape[:2]
        if text_mask is None:
            text_mask_bool = torch.ones(
                (batch_size, text_len), device=text_query.device, dtype=torch.bool
            )
        else:
            if tuple(text_mask.shape) != (batch_size, text_len):
                raise ValueError(
                    f"unexpected text mask {tuple(text_mask.shape)} for text Q {tuple(text_query.shape)}"
                )
            text_mask_bool = text_mask.to(device=text_query.device, dtype=torch.bool)

        key_all = torch.cat((image_key, text_key), dim=1)
        image_mask = torch.ones(
            (batch_size, image_key.shape[1]), device=text_query.device, dtype=torch.bool
        )
        key_mask = torch.cat((image_mask, text_mask_bool), dim=1)
        if direction == "text_to_image":
            # Full attention weights, but only image values, isolate
            # A_text<-image V_image for text-query targets.
            if self.config.condition_scope == "text_only":
                vision_tokens = int(self.config.expected_vision_tokens or 0)
                target_query = text_query[:, vision_tokens:]
                target_mask = text_mask_bool[:, vision_tokens:]
                target_len = text_len - vision_tokens
            else:
                target_query = text_query
                target_mask = text_mask_bool
                target_len = text_len
            source_value = torch.cat(
                (image_value, torch.zeros_like(text_value)), dim=1
            )
        elif direction == "image_to_text":
            # Full attention weights, but only text values, isolate
            # A_image<-text V_text for image-query targets.
            target_query = image_query
            target_mask = image_mask
            target_len = image_query.shape[1]
            if self.config.condition_scope == "text_only":
                vision_tokens = int(self.config.expected_vision_tokens or 0)
                text_source_value = torch.cat(
                    (
                        torch.zeros_like(text_value[:, :vision_tokens]),
                        text_value[:, vision_tokens:],
                    ),
                    dim=1,
                )
            else:
                text_source_value = text_value
            source_value = torch.cat(
                (torch.zeros_like(image_value), text_source_value), dim=1
            )
        else:
            raise ValueError(f"unsupported bridge direction: {direction}")
        contribution, backend = _value_slice_attention(
            target_query,
            key_all,
            source_value,
            target_mask,
            key_mask,
        )
        self.bridge_backend = backend
        if sp_enabled:
            contribution = all_gather(contribution, dim=2, group=parallel_dims.sp_group).contiguous()
        return contribution.reshape(batch_size, target_len, -1)

    def parallel_attention(self, q, k, v, img_q_len, img_kv_len, **kwargs):
        block_idx = kwargs.get("block_idx")
        intervention_active = (
            self.strong_branch_active and block_idx in self.selected_layers
        )
        parallel_dims = get_parallel_state()
        attn_mode = str(kwargs.get("attn_mode") or "").lower()
        direct_backend_supported = (
            not parallel_dims.sp_enabled
            and (
                (q[0].is_cuda and attn_mode in {"flash", "flash2"})
                or (q[0].device.type == "cpu" and attn_mode == "torch")
            )
        )
        if (
            intervention_active
            and self.config.expected_task == "i2v"
            and not self._joint_attention_layout_validated
        ):
            vision_tokens = int(self.config.expected_vision_tokens or 0)
            text_mask = kwargs.get("text_mask")
            encoder_tokens = int(q[1].shape[1])
            if encoder_tokens <= vision_tokens:
                raise RuntimeError(
                    "I2V SFG expected prompt tokens after the reference-image "
                    f"segment: encoder={encoder_tokens}, vision={vision_tokens}"
                )
            if text_mask is None or tuple(text_mask.shape) != (
                q[1].shape[0],
                encoder_tokens,
            ):
                raise RuntimeError(
                    "I2V SFG requires the complete encoder attention mask"
                )
            vision_mask = text_mask[:, :vision_tokens].to(dtype=torch.bool)
            if not bool(torch.all(vision_mask).item()):
                raise RuntimeError(
                    "I2V reference-image semantic tokens are not the first fully-valid "
                    f"{vision_tokens} encoder tokens"
                )
            self.observed_encoder_tokens = encoder_tokens
            self.observed_valid_encoder_tokens = int(
                text_mask.to(dtype=torch.bool).sum(dim=1).min().item()
            )
            self.observed_valid_prompt_tokens = int(
                text_mask[:, vision_tokens:]
                .to(dtype=torch.bool)
                .sum(dim=1)
                .min()
                .item()
            )
            if (
                self.config.condition_scope == "text_only"
                and self.observed_valid_prompt_tokens < 1
            ):
                raise RuntimeError(
                    "I2V text-only SFG requires at least one valid prompt token "
                    "after the reference-image semantic segment"
                )
            self._joint_attention_layout_validated = True
        if (
            intervention_active
            and self.config.sfg_s_bridge_delta
            and direct_backend_supported
        ):
            if q[0].shape[1] != int(img_q_len) or k[0].shape[1] != int(img_kv_len):
                raise RuntimeError(
                    "SFGS/SFGX direct attention image length mismatch: "
                    f"q={q[0].shape[1]} vs {img_q_len}, "
                    f"kv={k[0].shape[1]} vs {img_kv_len}"
                )
            attended, backend = _direct_split_bridge_attention(
                q,
                k,
                v,
                kwargs.get("text_mask"),
                sfg_s_bridge_delta=self.config.sfg_s_bridge_delta,
                sfg_x_bridge_delta=self.config.sfg_x_bridge_delta,
                condition_scope=self.config.condition_scope,
                vision_tokens=self.config.expected_vision_tokens,
            )
            expected_sequence = int(img_q_len) + q[1].shape[1]
            if attended.shape[1] != expected_sequence:
                raise RuntimeError(
                    "SFGS/SFGX direct attention sequence mismatch: "
                    f"output={tuple(attended.shape)}, expected_sequence={expected_sequence}"
                )
            self.bridge_backend = backend
            self.sfg_s_attention_calls += 1
            if self.config.sfg_x_bridge_delta:
                self.sfg_x_attention_calls += 1
            self.bridge_attention_calls += 1
            self.direct_bridge_attention_calls += 1
            return attended

        attended = self.original_parallel_attention(
            q,
            k,
            v,
            img_q_len,
            img_kv_len,
            **kwargs,
        )
        if not intervention_active:
            return attended

        if self.config.sfg_s_bridge_delta:
            contribution = self._bridge_contribution(
                q,
                k,
                v,
                kwargs.get("text_mask"),
                direction="image_to_text",
            )
            image_output = attended[:, : int(img_q_len)]
            if image_output.shape != contribution.shape:
                raise RuntimeError(
                    "SFGS bridge contribution/output mismatch: "
                    f"output={tuple(image_output.shape)} "
                    f"contribution={tuple(contribution.shape)}"
                )
            image_output.add_(contribution, alpha=self.config.sfg_s_bridge_delta)
            self.sfg_s_attention_calls += 1
        if self.config.sfg_x_bridge_delta:
            contribution = self._bridge_contribution(
                q,
                k,
                v,
                kwargs.get("text_mask"),
                direction="text_to_image",
            )
            text_output = attended[:, int(img_q_len) :]
            if self.config.condition_scope == "text_only":
                vision_tokens = int(self.config.expected_vision_tokens or 0)
                text_output = text_output[:, vision_tokens:]
            if text_output.shape != contribution.shape:
                raise RuntimeError(
                    "SFGX bridge contribution/output mismatch: "
                    f"output={tuple(text_output.shape)} "
                    f"contribution={tuple(contribution.shape)}"
                )
            text_output.add_(contribution, alpha=self.config.sfg_x_bridge_delta)
            self.sfg_x_attention_calls += 1
        self.bridge_attention_calls += 1
        self.legacy_bridge_attention_calls += 1
        return attended

    def guided_forward(self, *args, **kwargs):
        if bool(kwargs.get("output_features", False)):
            raise ValueError("Hunyuan explicit SFGS/SFGX does not support output_features=True")
        if kwargs.get("return_dict", False):
            raise ValueError("Hunyuan explicit SFGS/SFGX requires return_dict=False")

        if self.config.expected_task is not None:
            actual_task = str(kwargs.get("mask_type", "")).strip().lower()
            if actual_task != self.config.expected_task:
                raise RuntimeError(
                    "SFG task mismatch: "
                    f"expected={self.config.expected_task!r}, actual={actual_task!r}"
                )
            if actual_task == "i2v":
                vision_states = kwargs.get("vision_states")
                if not isinstance(vision_states, torch.Tensor):
                    raise RuntimeError("I2V joint SFG requires vision_states")
                actual_vision_tokens = int(vision_states.shape[1])
                if actual_vision_tokens != self.config.expected_vision_tokens:
                    raise RuntimeError(
                        "I2V vision-token mismatch: "
                        f"expected={self.config.expected_vision_tokens}, "
                        f"actual={actual_vision_tokens}"
                    )
                if self.task_validation_calls == 0 and bool(
                    torch.all(vision_states == 0).item()
                ):
                    raise RuntimeError(
                        "I2V joint SFG received an all-zero reference-image semantic stream"
                    )
                self.observed_vision_tokens = actual_vision_tokens
            self.task_validation_calls += 1

        step_index = self.logical_forward_calls % self.config.total_steps
        self.logical_forward_calls += 1
        self.strong_branch_active = False
        clean_output = self.original_forward(*args, **kwargs)
        self.clean_forward_calls += 1
        if step_index >= self.config.active_steps:
            return clean_output
        try:
            self.strong_branch_active = True
            strong_output = self.original_forward(*args, **kwargs)
        finally:
            self.strong_branch_active = False
        self.strong_forward_calls += 1

        clean_prediction = clean_output[0]
        bridge_prediction = strong_output[0]
        residual = clean_prediction - bridge_prediction
        guided_prediction = clean_prediction + self.config.omega * residual

        if self.first_residual_relative_rms is None:
            residual_rms = residual.float().square().mean().sqrt()
            clean_rms = clean_prediction.float().square().mean().sqrt().clamp_min(1e-8)
            self.first_residual_relative_rms = float((residual_rms / clean_rms).item())
        return (guided_prediction, clean_output[1])

    def install(self) -> "HunyuanExplicitSFG":
        self.transformer_module.parallel_attention = self.parallel_attention
        self.transformer_module._sfg_explicit_sfg_controller = self

        controller = self

        def wrapped_forward(_transformer_self, *args, **kwargs):
            return controller.guided_forward(*args, **kwargs)

        self.transformer.forward = types.MethodType(wrapped_forward, self.transformer)
        self.transformer._sfg_explicit_sfg_controller = self
        return self

    def runtime_stats(self) -> dict[str, Any]:
        expected_per_strong_forward = len(self.config.layer_indices)
        expected_sfg_s_calls = (
            self.strong_forward_calls * expected_per_strong_forward
            if self.config.sfg_s_bridge_delta
            else 0
        )
        expected_sfg_x_calls = (
            self.strong_forward_calls * expected_per_strong_forward
            if self.config.sfg_x_bridge_delta
            else 0
        )
        return {
            **self.config.to_dict(),
            "logical_forward_calls": self.logical_forward_calls,
            "clean_forward_calls": self.clean_forward_calls,
            "strong_forward_calls": self.strong_forward_calls,
            "bridge_attention_calls": self.bridge_attention_calls,
            "expected_bridge_attention_calls": (
                self.strong_forward_calls * expected_per_strong_forward
            ),
            "sfg_s_attention_calls": self.sfg_s_attention_calls,
            "expected_sfg_s_attention_calls": expected_sfg_s_calls,
            "sfg_x_attention_calls": self.sfg_x_attention_calls,
            "expected_sfg_x_attention_calls": expected_sfg_x_calls,
            "direct_bridge_attention_calls": self.direct_bridge_attention_calls,
            "legacy_bridge_attention_calls": self.legacy_bridge_attention_calls,
            "bridge_backend": self.bridge_backend,
            "first_residual_relative_rms": self.first_residual_relative_rms,
            "task_validation_calls": self.task_validation_calls,
            "observed_vision_tokens": self.observed_vision_tokens,
            "observed_encoder_tokens": self.observed_encoder_tokens,
            "observed_valid_encoder_tokens": self.observed_valid_encoder_tokens,
            "observed_valid_prompt_tokens": self.observed_valid_prompt_tokens,
            "joint_attention_layout_validated": self._joint_attention_layout_validated,
            "condition_attention_layout_validated": self._joint_attention_layout_validated,
            "vision_attention_scaling_excluded": (
                self.config.condition_scope == "text_only"
            ),
            "unscaled_leading_encoder_tokens": (
                self.config.expected_vision_tokens
                if self.config.condition_scope == "text_only"
                else 0
            ),
            "scaled_encoder_token_start": (
                self.config.expected_vision_tokens
                if self.config.condition_scope == "text_only"
                else 0
            ),
        }


def install_hunyuan_explicit_sfg(
    transformer: torch.nn.Module,
    *,
    sfg_s_u: float = 0.0,
    sfg_strength_u: float = -0.1,
    omega: float = 1.0,
    layers: str | Iterable[int] = "all",
    total_steps: int = 1,
    active_steps: int | None = None,
    expected_task: str | None = None,
    expected_vision_tokens: int | None = None,
    condition_scope: str = "joint_encoder",
) -> HunyuanExplicitSFG:
    """Install SD3.5-style explicit SFGS/SFGX on a loaded Hunyuan transformer."""

    if active_steps is None:
        active_steps = total_steps

    return HunyuanExplicitSFG(
        transformer,
        sfg_s_u=sfg_s_u,
        sfg_strength_u=sfg_strength_u,
        omega=omega,
        layers=layers,
        total_steps=total_steps,
        active_steps=active_steps,
        expected_task=expected_task,
        expected_vision_tokens=expected_vision_tokens,
        condition_scope=condition_scope,
    ).install()


def sfg_runtime_stats(transformer: torch.nn.Module) -> dict[str, Any] | None:
    controller = getattr(transformer, "_sfg_explicit_sfg_controller", None)
    return None if controller is None else controller.runtime_stats()
