from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


@dataclass
class BridgeCausalConfig:
    mode: str = "feedback_tokens"
    num_feedback_tokens: int = 16
    adapter_layers: str = "all"
    lock_strength: float = 1.0
    routing_scale_init: float = 1.0
    split_feedback_init: float = 0.25
    split_feedback_bias: int = 0
    split_feedback_bias_init_std: float = 0.0
    split_feedback_source_conditioning: int = 0
    split_feedback_source_names: str = "geneval,t2i,hps,coco,unknown"
    split_feedback_source_bias_init_std: float = 0.0
    split_feedback_source_gain_init_std: float = 0.0
    split_feedback_source_route_conditioning: int = 0
    split_feedback_source_routing_init_std: float = 0.0
    split_feedback_source_i2f_gate_init_std: float = 0.0
    split_feedback_source_i2f_lora_gain_init_std: float = 0.0
    split_feedback_source_token_lora_rank: int = 0
    split_feedback_source_token_lora_alpha: float = 1.0
    split_feedback_source_token_lora_up_init_std: float = 0.0
    gate_init_i2t: float = 0.0
    gate_init_t2i: float = 0.0
    gate_init_i2f: float = 0.0
    i2f_lora_rank: int = 0
    i2f_lora_alpha: float = 1.0
    i2f_lora_norm: int = 1
    use_timestep_gates: int = 0
    timestep_gate_hidden_dim: int = 64
    timestep_gate_scale: float = 0.10
    max_semantic_leak: float = 0.05
    timestep_max: float = 1000.0


def _parse_layer_spec(spec: str, num_layers: int) -> List[int]:
    text = str(spec or "all").strip().lower()
    if text in {"", "all", "*"}:
        return list(range(num_layers))
    if text in {"early", "first"}:
        return list(range(0, max(1, num_layers // 3)))
    if text in {"mid", "middle"}:
        return list(range(num_layers // 3, max(num_layers // 3 + 1, (2 * num_layers) // 3)))
    if text in {"late", "last"}:
        return list(range((2 * num_layers) // 3, num_layers))
    if text in {"mid_late", "middle_late", "mid-late"}:
        return list(range(num_layers // 3, num_layers))

    layers = []
    for item in re.split(r"[\s,;]+", text):
        if not item:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", item)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            layers.extend(range(start, end + 1))
            continue
        if item.isdigit():
            layers.append(int(item))
            continue
        raise ValueError(f"Unsupported adapter layer spec item: {item!r}")
    unique = sorted({i for i in layers if 0 <= i < num_layers})
    if not unique:
        raise ValueError(f"Adapter layer spec {spec!r} selected no valid layers for num_layers={num_layers}.")
    return unique


def _parse_source_names(spec: str) -> list[str]:
    names = []
    for item in re.split(r"[\s,;]+", str(spec or "")):
        normalized = _normalize_prompt_source(item)
        if normalized and normalized not in names:
            names.append(normalized)
    if "unknown" not in names:
        names.append("unknown")
    return names or ["unknown"]


def _normalize_prompt_source(source: object) -> str:
    text = str(source or "").strip().lower()
    aliases = {
        "gen_eval": "geneval",
        "geneval6c": "geneval",
        "t2icomp": "t2i",
        "t2i_comp": "t2i",
        "t2i-comp": "t2i",
        "hpsv21": "hps",
    }
    return aliases.get(text, text or "unknown")


class BridgeCausalJointAttnProcessor2_0(nn.Module):
    """SD3 joint-attention processor for trainable bridge gates and S/F routing."""

    def __init__(
        self,
        *,
        mode: str,
        semantic_token_count: int = 0,
        feedback_token_count: int = 0,
        lock_strength: float = 1.0,
        routing_scale_init: float = 1.0,
        gate_init_i2t: float = 0.0,
        gate_init_t2i: float = 0.0,
        gate_init_i2f: float = 0.0,
        inner_dim: int = 0,
        i2f_lora_rank: int = 0,
        i2f_lora_alpha: float = 1.0,
        i2f_lora_norm: bool = True,
        train_gates: bool = True,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("BridgeCausalJointAttnProcessor2_0 requires PyTorch scaled_dot_product_attention.")
        self.mode = str(mode)
        self.semantic_token_count = int(semantic_token_count)
        self.feedback_token_count = int(feedback_token_count)
        self.lock_strength = float(lock_strength)
        train_gates = bool(train_gates) and self.mode in {
            "bridge_gate",
            "feedback_tokens",
            "split_text",
            "contrast_readout",
            "readout_lora",
        }
        self.routing_scale = (
            nn.Parameter(torch.tensor(float(routing_scale_init)))
            if train_gates and self.mode in {"feedback_tokens", "split_text"}
            else None
        )
        directional_gates = train_gates and self.mode not in {"contrast_readout", "readout_lora"}
        self.gate_i2t = nn.Parameter(torch.tensor(float(gate_init_i2t))) if directional_gates else None
        self.gate_t2i = nn.Parameter(torch.tensor(float(gate_init_t2i))) if directional_gates else None
        self.gate_i2f = nn.Parameter(torch.tensor(float(gate_init_i2f))) if train_gates else None
        self.i2f_lora_rank = int(i2f_lora_rank)
        self.i2f_lora_alpha = float(i2f_lora_alpha)
        if self.i2f_lora_rank > 0:
            if int(inner_dim) <= 0:
                raise ValueError("inner_dim must be positive when i2f_lora_rank > 0.")
            # LayerNorm rescales arbitrarily small inputs to unit variance. For the
            # contrast pathway that destroys the contamination-proportional property,
            # so contrast jobs should disable it (i2f_lora_norm=0).
            self.i2f_norm = nn.LayerNorm(int(inner_dim)) if bool(i2f_lora_norm) else None
            self.i2f_down = nn.Linear(int(inner_dim), self.i2f_lora_rank, bias=False)
            self.i2f_up = nn.Linear(self.i2f_lora_rank, int(inner_dim), bias=False)
            nn.init.normal_(self.i2f_down.weight, std=1.0 / max(1, int(inner_dim)))
            nn.init.zeros_(self.i2f_up.weight)
        else:
            self.i2f_norm = None
            self.i2f_down = None
            self.i2f_up = None
        if self.mode == "readout_lora" and train_gates:
            # Frozen scale-matching constant c_l for the no-branch control:
            # multiplies the plain image->text readout so the gate/LoRA see
            # inputs at the same magnitude the contrast pathway feeds them.
            # Measured offline from a trained contrast checkpoint; a buffer,
            # never trained, persisted through adapter.pt like any state.
            self.register_buffer("readout_scale", torch.tensor(1.0, dtype=torch.float32))
        else:
            self.readout_scale = None
        self._runtime_routing_delta: torch.Tensor | None = None
        self._runtime_gate_i2f_delta: torch.Tensor | None = None
        self._runtime_i2f_lora_gain_delta: torch.Tensor | None = None
        self._runtime_gate_i2s_delta: torch.Tensor | None = None
        self._runtime_gate_f2i_delta: torch.Tensor | None = None
        self._runtime_semantic_leak: torch.Tensor | None = None
        self._runtime_visual_swap_partner: torch.Tensor | None = None
        self._runtime_visual_swap_scale: float = 0.0

    def set_token_layout(self, semantic_token_count: int, feedback_token_count: int) -> None:
        self.semantic_token_count = int(semantic_token_count)
        self.feedback_token_count = int(feedback_token_count)

    def set_source_gate_adjustments(
        self,
        routing_delta: torch.Tensor | None = None,
        gate_i2f_delta: torch.Tensor | None = None,
        i2f_lora_gain_delta: torch.Tensor | None = None,
        gate_i2s_delta: torch.Tensor | None = None,
        gate_f2i_delta: torch.Tensor | None = None,
        semantic_leak: torch.Tensor | None = None,
    ) -> None:
        self._runtime_routing_delta = routing_delta
        self._runtime_gate_i2f_delta = gate_i2f_delta
        self._runtime_i2f_lora_gain_delta = i2f_lora_gain_delta
        self._runtime_gate_i2s_delta = gate_i2s_delta
        self._runtime_gate_f2i_delta = gate_f2i_delta
        self._runtime_semantic_leak = semantic_leak

    def set_visual_swap(
        self,
        partner_indices: torch.Tensor | None = None,
        scale: float = 0.0,
    ) -> None:
        self._runtime_visual_swap_partner = partner_indices
        self._runtime_visual_swap_scale = float(scale)

    @staticmethod
    def _apply_scale(contrib: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
        if torch.is_tensor(scale):
            scale = scale.to(device=contrib.device, dtype=contrib.dtype)
            if scale.ndim == 1:
                scale = scale.view(-1, 1, 1, 1)
        return scale * contrib

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        image_seq_len = hidden_states.shape[1]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        semantic_count = 0
        feedback_count = 0
        s_start = image_seq_len
        s_end = image_seq_len
        f_start = image_seq_len
        f_end = image_seq_len
        if encoder_hidden_states is not None:
            total_seq_len = key.shape[2]
            semantic_count = max(0, min(self.semantic_token_count, total_seq_len - image_seq_len))
            feedback_count = max(0, min(self.feedback_token_count, total_seq_len - image_seq_len - semantic_count))
            s_start = image_seq_len
            s_end = image_seq_len + semantic_count
            f_start = s_end
            f_end = f_start + feedback_count

        if encoder_hidden_states is not None and self.mode in {"feedback_tokens", "split_text"} and feedback_count > 0:
            attended = torch.zeros_like(query)
            base_end = f_start
            attended[:, :, :base_end] = F.scaled_dot_product_attention(
                query[:, :, :base_end],
                key[:, :, :base_end],
                value[:, :, :base_end],
                dropout_p=0.0,
                is_causal=False,
            )
            attended[:, :, f_start:f_end] = F.scaled_dot_product_attention(
                query[:, :, f_start:f_end],
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
        elif encoder_hidden_states is not None and self.mode == "contrast_readout" and feedback_count > 0:
            # Base tokens [image, original text] attend exactly as the base model;
            # the appended clean-text copy attends only to itself so it never
            # absorbs image state and stays a semantic reference stream.
            attended = torch.zeros_like(query)
            base_end = f_start
            attended[:, :, :base_end] = F.scaled_dot_product_attention(
                query[:, :, :base_end],
                key[:, :, :base_end],
                value[:, :, :base_end],
                dropout_p=0.0,
                is_causal=False,
            )
            attended[:, :, f_start:f_end] = F.scaled_dot_product_attention(
                query[:, :, f_start:f_end],
                key[:, :, f_start:f_end],
                value[:, :, f_start:f_end],
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        if encoder_hidden_states is not None:
            if (
                self._runtime_visual_swap_partner is not None
                and abs(float(self._runtime_visual_swap_scale)) > 0.0
                and semantic_count > 0
                and image_seq_len > 0
            ):
                attended = self._apply_text_image_visual_swap(
                    attended,
                    query,
                    key,
                    value,
                    image_seq_len=image_seq_len,
                    target_start=s_start,
                    target_end=s_end,
                    key_end=(f_start if feedback_count > 0 else total_seq_len),
                    partner_indices=self._runtime_visual_swap_partner,
                    scale=float(self._runtime_visual_swap_scale),
                )

            updates = []
            if self.mode == "bridge_gate":
                if self.gate_i2t is not None:
                    updates.append((0, image_seq_len, s_start, total_seq_len, self.gate_i2t, total_seq_len))
                if self.gate_t2i is not None and semantic_count > 0:
                    updates.append((s_start, s_end, 0, image_seq_len, self.gate_t2i, total_seq_len))
            elif self.mode in {"contrast_readout", "readout_lora"}:
                # Handled below with standalone-softmax readouts; the generic
                # value-slice update list uses a different normalization.
                pass
            else:
                routing_scale = self.routing_scale if self.routing_scale is not None else 1.0
                if self._runtime_routing_delta is not None:
                    routing_scale = routing_scale + self._runtime_routing_delta.to(
                        device=query.device,
                        dtype=query.dtype,
                    )
                effective_lock = self.lock_strength * routing_scale
                if self._runtime_semantic_leak is not None:
                    semantic_leak = self._runtime_semantic_leak.to(
                        device=query.device,
                        dtype=query.dtype,
                    )
                    effective_lock = torch.clamp(effective_lock - semantic_leak, min=0.0, max=1.5)
                if self.lock_strength != 0.0 and semantic_count > 0:
                    # In feedback/split modes S already attends only over image+S.
                    # Compute the image contribution under that same key set, and do
                    # not subtract S<-F because that path is already masked out.
                    updates.append((s_start, s_end, 0, image_seq_len, -effective_lock, f_start))
                if semantic_count > 0 and self._runtime_gate_i2s_delta is not None:
                    gate_i2s_delta = self._runtime_gate_i2s_delta.to(
                        device=query.device,
                        dtype=query.dtype,
                    )
                    updates.append((0, image_seq_len, s_start, s_end, gate_i2s_delta, f_start))
                if feedback_count > 0 and self.gate_i2f is not None:
                    gate_i2f = self.gate_i2f
                    if self._runtime_gate_i2f_delta is not None:
                        gate_i2f = gate_i2f + self._runtime_gate_i2f_delta.to(
                            device=query.device,
                            dtype=query.dtype,
                        )
                    updates.append((0, image_seq_len, f_start, f_end, gate_i2f, total_seq_len))
                if feedback_count > 0 and self._runtime_gate_f2i_delta is not None:
                    gate_f2i_delta = self._runtime_gate_f2i_delta.to(
                        device=query.device,
                        dtype=query.dtype,
                    )
                    updates.append((f_start, f_end, 0, image_seq_len, gate_f2i_delta, total_seq_len))

            if updates:
                attended = attended.clone()
                for target_start, target_end, source_start, source_end, scale, key_end in updates:
                    if target_end <= target_start or source_end <= source_start:
                        continue
                    contrib = self._value_slice_contribution(
                        query[:, :, target_start:target_end],
                        key[:, :, :key_end],
                        value[:, :, :key_end],
                        source_start,
                        source_end,
                    )
                    delta = self._apply_scale(contrib, scale)
                    if (
                        self.i2f_down is not None
                        and target_start == 0
                        and target_end == image_seq_len
                        and source_start == f_start
                        and source_end == f_end
                    ):
                        lora_delta = self._i2f_lora_delta(contrib)
                        if self._runtime_i2f_lora_gain_delta is not None:
                            lora_gain = 1.0 + self._runtime_i2f_lora_gain_delta.to(
                                device=query.device,
                                dtype=query.dtype,
                            )
                            lora_delta = self._apply_scale(lora_delta, lora_gain)
                        delta = delta + lora_delta
                    attended[:, :, target_start:target_end] = attended[:, :, target_start:target_end] + delta

            if (
                self.mode == "contrast_readout"
                and feedback_count > 0
                and image_seq_len > 0
                and self.gate_i2f is not None
            ):
                # Image tokens read the clean semantic copy minus the (possibly
                # contaminated) original text stream: an in-forward counterfactual
                # contrast. Zero gate and zero LoRA-up keep this an exact no-op.
                gate = self.gate_i2f
                if self._runtime_gate_i2f_delta is not None:
                    gate = gate + self._runtime_gate_i2f_delta.to(
                        device=query.device,
                        dtype=query.dtype,
                    )
                query_image = query[:, :, :image_seq_len]
                readout_clean = F.scaled_dot_product_attention(
                    query_image,
                    key[:, :, f_start:f_end],
                    value[:, :, f_start:f_end],
                    dropout_p=0.0,
                    is_causal=False,
                )
                readout_orig = F.scaled_dot_product_attention(
                    query_image,
                    key[:, :, s_start:s_end],
                    value[:, :, s_start:s_end],
                    dropout_p=0.0,
                    is_causal=False,
                )
                contrast = readout_clean - readout_orig
                capture = getattr(self, "capture_readout_norms", None)
                if capture is not None:
                    # Diagnostic-only: per-call median L2 norm of the full
                    # per-token vector (heads folded), used to measure the
                    # readout_lora scale-matching constant c_l offline.
                    with torch.no_grad():
                        def _median_token_norm(t: torch.Tensor) -> float:
                            flat = t.transpose(1, 2).reshape(t.shape[0], t.shape[2], -1)
                            return float(flat.float().norm(dim=-1).median().item())

                        capture.append(
                            {
                                "contrast": _median_token_norm(contrast),
                                "orig": _median_token_norm(readout_orig),
                            }
                        )
                delta = self._apply_scale(contrast, gate)
                if self.i2f_down is not None:
                    lora_delta = self._i2f_lora_delta(contrast)
                    if self._runtime_i2f_lora_gain_delta is not None:
                        lora_gain = 1.0 + self._runtime_i2f_lora_gain_delta.to(
                            device=query.device,
                            dtype=query.dtype,
                        )
                        lora_delta = self._apply_scale(lora_delta, lora_gain)
                    delta = delta + lora_delta
                attended = attended.clone()
                attended[:, :, :image_seq_len] = attended[:, :, :image_seq_len] + delta

            if (
                self.mode == "readout_lora"
                and semantic_count > 0
                and image_seq_len > 0
                and self.gate_i2f is not None
            ):
                # Scale-matched NO-BRANCH control for contrast_readout: image
                # tokens re-read the ORIGINAL text stream through the same
                # standalone softmax (this is exactly the readout_orig half of
                # the contrast pathway, which needs no appended branch). The
                # frozen readout_scale constant c_l rescales it so the gate and
                # LoRA see inputs at contrast magnitude. Single differing
                # variable vs contrast_readout: the signal content (plain
                # readout vs counterfactual contrast). Zero gate and zero
                # LoRA-up keep this an exact no-op.
                gate = self.gate_i2f
                if self._runtime_gate_i2f_delta is not None:
                    gate = gate + self._runtime_gate_i2f_delta.to(
                        device=query.device,
                        dtype=query.dtype,
                    )
                query_image = query[:, :, :image_seq_len]
                readout_orig = F.scaled_dot_product_attention(
                    query_image,
                    key[:, :, s_start:s_end],
                    value[:, :, s_start:s_end],
                    dropout_p=0.0,
                    is_causal=False,
                )
                scaled_readout = readout_orig * self.readout_scale.to(
                    device=readout_orig.device,
                    dtype=readout_orig.dtype,
                )
                delta = self._apply_scale(scaled_readout, gate)
                if self.i2f_down is not None:
                    lora_delta = self._i2f_lora_delta(scaled_readout)
                    if self._runtime_i2f_lora_gain_delta is not None:
                        lora_gain = 1.0 + self._runtime_i2f_lora_gain_delta.to(
                            device=query.device,
                            dtype=query.dtype,
                        )
                        lora_delta = self._apply_scale(lora_delta, lora_gain)
                    delta = delta + lora_delta
                attended = attended.clone()
                attended[:, :, :image_seq_len] = attended[:, :, :image_seq_len] + delta

        hidden_states = attended.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states

    @staticmethod
    def _value_slice_contribution(query_target, key_all, value_all, source_start: int, source_end: int):
        value_source_only = torch.zeros_like(value_all)
        value_source_only[:, :, source_start:source_end] = value_all[:, :, source_start:source_end]
        return F.scaled_dot_product_attention(
            query_target,
            key_all,
            value_source_only,
            dropout_p=0.0,
            is_causal=False,
        )

    def _i2f_lora_delta(self, contrib: torch.Tensor) -> torch.Tensor:
        batch_size, heads, seq_len, head_dim = contrib.shape
        x = contrib.transpose(1, 2).reshape(batch_size, seq_len, heads * head_dim)
        if self.i2f_norm is not None:
            x = self.i2f_norm(x)
        x = self.i2f_up(self.i2f_down(x))
        x = x * (self.i2f_lora_alpha / max(1, self.i2f_lora_rank))
        return x.reshape(batch_size, seq_len, heads, head_dim).transpose(1, 2).to(dtype=contrib.dtype)

    def _apply_text_image_visual_swap(
        self,
        attended: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        image_seq_len: int,
        target_start: int,
        target_end: int,
        key_end: int,
        partner_indices: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        partner = partner_indices.to(device=query.device, dtype=torch.long)
        if partner.numel() != query.shape[0]:
            raise ValueError(
                f"visual-swap partner count {partner.numel()} does not match batch size {query.shape[0]}"
            )
        if int(partner.min().item()) < 0 or int(partner.max().item()) >= query.shape[0]:
            raise ValueError(f"visual-swap partner indices out of range: {partner.detach().cpu().tolist()}")

        key_context = key[:, :, :key_end]
        value_context = value[:, :, :key_end]
        key_swapped = key_context.clone()
        value_swapped = value_context.clone()
        key_swapped[:, :, :image_seq_len] = key_context.index_select(0, partner)[:, :, :image_seq_len]
        value_swapped[:, :, :image_seq_len] = value_context.index_select(0, partner)[:, :, :image_seq_len]
        query_target = query[:, :, target_start:target_end]
        original_contrib = self._value_slice_contribution(
            query_target,
            key_context,
            value_context,
            0,
            image_seq_len,
        )
        swapped_contrib = self._value_slice_contribution(
            query_target,
            key_swapped,
            value_swapped,
            0,
            image_seq_len,
        )
        updated = attended.clone()
        updated[:, :, target_start:target_end] = (
            updated[:, :, target_start:target_end]
            + float(scale) * (swapped_contrib - original_contrib)
        )
        return updated


class BridgeCausalSD3Adapter(nn.Module):
    """Small trainable retrofit module for frozen SD3 MMDiT transformers."""

    def __init__(self, transformer, config: BridgeCausalConfig):
        super().__init__()
        self.config_obj = config
        self.mode = str(config.mode)
        self._runtime_enabled = True
        if self.mode not in {"bridge_gate", "feedback_tokens", "split_text", "contrast_readout", "readout_lora"}:
            raise ValueError(f"Unsupported bridge-causal adapter mode: {self.mode}")
        self.inner_dim = int(transformer.inner_dim)
        self.num_layers = len(transformer.transformer_blocks)
        self.adapter_layer_indices = _parse_layer_spec(config.adapter_layers, self.num_layers)
        self.use_timestep_gates = bool(int(config.use_timestep_gates)) and self.mode in {
            "feedback_tokens",
            "split_text",
            "contrast_readout",
        }
        self.timestep_gate_scale = float(config.timestep_gate_scale)
        self.max_semantic_leak = float(config.max_semantic_leak)
        self.timestep_max = max(1.0, float(config.timestep_max))
        if self.use_timestep_gates:
            gate_hidden_dim = int(config.timestep_gate_hidden_dim)
            self.timestep_layer_embed = nn.Embedding(self.num_layers, gate_hidden_dim)
            self.timestep_gate_mlp = nn.Sequential(
                nn.Linear(gate_hidden_dim + 1, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 4),
            )
            nn.init.zeros_(self.timestep_gate_mlp[-1].weight)
            nn.init.zeros_(self.timestep_gate_mlp[-1].bias)
        else:
            self.timestep_layer_embed = None
            self.timestep_gate_mlp = None
        self._last_semantic_reference: torch.Tensor | None = None
        self._last_semantic_states: list[torch.Tensor] = []
        self._last_timestep_gate_values: list[dict[str, torch.Tensor]] = []
        self._last_rho: torch.Tensor | None = None

        if self.mode == "feedback_tokens":
            token_scale = self.inner_dim ** -0.5
            self.feedback_tokens = nn.Parameter(
                torch.randn(int(config.num_feedback_tokens), self.inner_dim, dtype=torch.float32) * token_scale
            )
        else:
            self.feedback_tokens = None

        if self.mode == "split_text":
            self.split_feedback_gain = nn.Parameter(torch.tensor(float(config.split_feedback_init), dtype=torch.float32))
            if int(config.split_feedback_bias):
                bias = torch.zeros(self.inner_dim, dtype=torch.float32)
                init_std = float(config.split_feedback_bias_init_std)
                if init_std > 0.0:
                    bias.normal_(mean=0.0, std=init_std)
                self.split_feedback_bias = nn.Parameter(bias)
            else:
                self.split_feedback_bias = None
            self.split_feedback_source_names = _parse_source_names(config.split_feedback_source_names)
            self._source_to_index = {name: index for index, name in enumerate(self.split_feedback_source_names)}
            if int(config.split_feedback_source_conditioning):
                num_sources = len(self.split_feedback_source_names)
                gain_delta = torch.zeros(num_sources, dtype=torch.float32)
                gain_std = float(config.split_feedback_source_gain_init_std)
                if gain_std > 0.0:
                    gain_delta.normal_(mean=0.0, std=gain_std)
                self.split_feedback_source_gain_delta = nn.Parameter(gain_delta)
                source_bias = torch.zeros(num_sources, self.inner_dim, dtype=torch.float32)
                source_bias_std = float(config.split_feedback_source_bias_init_std)
                if source_bias_std > 0.0:
                    source_bias.normal_(mean=0.0, std=source_bias_std)
                self.split_feedback_source_bias = nn.Parameter(source_bias)
            else:
                self.split_feedback_source_gain_delta = None
                self.split_feedback_source_bias = None
            if int(config.split_feedback_source_route_conditioning):
                num_sources = len(self.split_feedback_source_names)
                routing_delta = torch.zeros(num_sources, dtype=torch.float32)
                routing_std = float(config.split_feedback_source_routing_init_std)
                if routing_std > 0.0:
                    routing_delta.normal_(mean=0.0, std=routing_std)
                self.split_feedback_source_routing_delta = nn.Parameter(routing_delta)
                i2f_gate_delta = torch.zeros(num_sources, dtype=torch.float32)
                i2f_gate_std = float(config.split_feedback_source_i2f_gate_init_std)
                if i2f_gate_std > 0.0:
                    i2f_gate_delta.normal_(mean=0.0, std=i2f_gate_std)
                self.split_feedback_source_i2f_gate_delta = nn.Parameter(i2f_gate_delta)
                i2f_lora_gain_delta = torch.zeros(num_sources, dtype=torch.float32)
                i2f_lora_gain_std = float(config.split_feedback_source_i2f_lora_gain_init_std)
                if i2f_lora_gain_std > 0.0:
                    i2f_lora_gain_delta.normal_(mean=0.0, std=i2f_lora_gain_std)
                self.split_feedback_source_i2f_lora_gain_delta = nn.Parameter(i2f_lora_gain_delta)
            else:
                self.split_feedback_source_routing_delta = None
                self.split_feedback_source_i2f_gate_delta = None
                self.split_feedback_source_i2f_lora_gain_delta = None
            self.split_feedback_source_token_lora_rank = int(config.split_feedback_source_token_lora_rank)
            self.split_feedback_source_token_lora_alpha = float(config.split_feedback_source_token_lora_alpha)
            if self.split_feedback_source_token_lora_rank > 0:
                num_sources = len(self.split_feedback_source_names)
                rank = self.split_feedback_source_token_lora_rank
                self.split_feedback_source_token_lora_norm = nn.LayerNorm(self.inner_dim)
                token_lora_down = torch.empty(num_sources, self.inner_dim, rank, dtype=torch.float32)
                token_lora_down.normal_(mean=0.0, std=self.inner_dim**-0.5)
                token_lora_up = torch.zeros(num_sources, rank, self.inner_dim, dtype=torch.float32)
                up_std = float(config.split_feedback_source_token_lora_up_init_std)
                if up_std > 0.0:
                    token_lora_up.normal_(mean=0.0, std=up_std)
                self.split_feedback_source_token_lora_down = nn.Parameter(token_lora_down)
                self.split_feedback_source_token_lora_up = nn.Parameter(token_lora_up)
            else:
                self.split_feedback_source_token_lora_norm = None
                self.split_feedback_source_token_lora_down = None
                self.split_feedback_source_token_lora_up = None
        else:
            self.split_feedback_gain = None
            self.split_feedback_bias = None
            self.split_feedback_source_names = []
            self._source_to_index = {}
            self.split_feedback_source_gain_delta = None
            self.split_feedback_source_bias = None
            self.split_feedback_source_routing_delta = None
            self.split_feedback_source_i2f_gate_delta = None
            self.split_feedback_source_i2f_lora_gain_delta = None
            self.split_feedback_source_token_lora_rank = 0
            self.split_feedback_source_token_lora_alpha = 1.0
            self.split_feedback_source_token_lora_norm = None
            self.split_feedback_source_token_lora_down = None
            self.split_feedback_source_token_lora_up = None
        self._runtime_prompt_sources: str | list[str] | None = None

        adapter_layer_set = set(self.adapter_layer_indices)
        if self.mode in {"feedback_tokens", "split_text", "contrast_readout"}:
            # These modes append extra context tokens, so every layer needs the
            # masked processor to keep base tokens blind to the appended stream.
            processor_indices = list(range(self.num_layers))
        else:
            processor_indices = self.adapter_layer_indices
        processors = []
        for index in processor_indices:
            is_trainable_layer = index in adapter_layer_set
            processor = BridgeCausalJointAttnProcessor2_0(
                mode=self.mode,
                lock_strength=config.lock_strength if is_trainable_layer else 0.0,
                routing_scale_init=config.routing_scale_init,
                gate_init_i2t=config.gate_init_i2t,
                gate_init_t2i=config.gate_init_t2i,
                gate_init_i2f=config.gate_init_i2f,
                inner_dim=self.inner_dim,
                i2f_lora_rank=config.i2f_lora_rank if is_trainable_layer else 0,
                i2f_lora_alpha=config.i2f_lora_alpha,
                i2f_lora_norm=bool(int(getattr(config, "i2f_lora_norm", 1))),
                train_gates=is_trainable_layer,
            )
            processors.append((str(index), processor))
        self.processors = nn.ModuleDict(processors)
        self._installed = False

    @property
    def adapter_layer_set(self) -> set[int]:
        return set(self.adapter_layer_indices)

    def install(self, transformer) -> None:
        for index, block in enumerate(transformer.transformer_blocks):
            key = str(index)
            if key in self.processors:
                processor = self.processors[key]
                if hasattr(block.attn, "set_processor"):
                    block.attn.set_processor(processor)
                else:
                    block.attn.processor = processor
        self._installed = True

    def adapter_config(self) -> dict:
        data = asdict(self.config_obj)
        data["inner_dim"] = self.inner_dim
        data["num_layers"] = self.num_layers
        data["adapter_layer_indices"] = self.adapter_layer_indices
        if self.mode == "readout_lora":
            # Human-readable record of the frozen c_l constants; the canonical
            # values live in the state_dict buffers. Filtered out by the
            # dataclass-field whitelist on reload, so purely informational.
            data["readout_scale_values"] = {
                key: float(processor.readout_scale.detach().float().item())
                for key, processor in self.processors.items()
                if getattr(processor, "readout_scale", None) is not None
            }
        return data

    def set_readout_scales(self, scales: dict) -> None:
        """Assign the frozen per-layer readout_scale constants (readout_lora mode).

        Strict: the provided mapping must cover every adapter layer exactly, so a
        control run can never silently train with default 1.0 scales.
        """
        if self.mode != "readout_lora":
            raise ValueError(f"set_readout_scales requires mode=readout_lora, got {self.mode!r}")
        provided = {str(int(str(key))): float(value) for key, value in scales.items()}
        expected = {str(index) for index in self.adapter_layer_indices}
        if set(provided) != expected:
            missing = sorted(expected - set(provided), key=int)
            extra = sorted(set(provided) - expected, key=int)
            raise ValueError(
                f"readout_scale layer mismatch: missing={missing} extra={extra} expected={sorted(expected, key=int)}"
            )
        for key, value in provided.items():
            if not (value > 0.0):
                raise ValueError(f"readout_scale for layer {key} must be positive, got {value}")
            with torch.no_grad():
                self.processors[key].readout_scale.fill_(value)

    def set_prompt_source(self, prompt_sources: str | Sequence[str] | None) -> None:
        if prompt_sources is None:
            self._runtime_prompt_sources = None
        elif isinstance(prompt_sources, str):
            self._runtime_prompt_sources = _normalize_prompt_source(prompt_sources)
        else:
            self._runtime_prompt_sources = [_normalize_prompt_source(source) for source in prompt_sources]

    def set_runtime_enabled(self, enabled: bool) -> None:
        self._runtime_enabled = bool(enabled)

    def runtime_enabled(self) -> bool:
        return bool(self._runtime_enabled)

    def set_visual_swap(
        self,
        partner_indices: torch.Tensor | None = None,
        scale: float = 0.0,
    ) -> None:
        for processor in self.processors.values():
            processor.set_visual_swap(partner_indices, scale)

    @staticmethod
    def _summarize_tensors(stats: dict[str, float], name: str, tensors: list[torch.Tensor]) -> None:
        if not tensors:
            return
        values = torch.cat([tensor.detach().float().reshape(-1).cpu() for tensor in tensors])
        stats[f"{name}_mean"] = float(values.mean().item())
        stats[f"{name}_min"] = float(values.min().item())
        stats[f"{name}_max"] = float(values.max().item())
        stats[f"{name}_std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0

    def training_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        with torch.no_grad():
            if self.feedback_tokens is not None:
                feedback = self.feedback_tokens.detach().float()
                stats["feedback_tokens_mean"] = float(feedback.mean().item())
                stats["feedback_tokens_std"] = float(feedback.std(unbiased=False).item())
                stats["feedback_tokens_norm"] = float(feedback.norm().item())
            if self.split_feedback_gain is not None:
                stats["split_feedback_gain"] = float(self.split_feedback_gain.detach().float().item())
            if self.split_feedback_bias is not None:
                bias = self.split_feedback_bias.detach().float()
                stats["split_feedback_bias_mean"] = float(bias.mean().item())
                stats["split_feedback_bias_std"] = float(bias.std(unbiased=False).item())
                stats["split_feedback_bias_norm"] = float(bias.norm().item())
            if self.split_feedback_source_gain_delta is not None:
                gain_delta = self.split_feedback_source_gain_delta.detach().float()
                stats["split_feedback_source_gain_delta_mean"] = float(gain_delta.mean().item())
                stats["split_feedback_source_gain_delta_std"] = float(gain_delta.std(unbiased=False).item())
                base_gain = (
                    float(self.split_feedback_gain.detach().float().item())
                    if self.split_feedback_gain is not None
                    else 0.0
                )
                for index, name in enumerate(self.split_feedback_source_names):
                    stats[f"split_feedback_gain_{name}"] = float(base_gain + gain_delta[index].item())
            if self.split_feedback_source_bias is not None:
                source_bias = self.split_feedback_source_bias.detach().float()
                stats["split_feedback_source_bias_mean"] = float(source_bias.mean().item())
                stats["split_feedback_source_bias_std"] = float(source_bias.std(unbiased=False).item())
                for index, name in enumerate(self.split_feedback_source_names):
                    stats[f"split_feedback_source_bias_norm_{name}"] = float(source_bias[index].norm().item())
            if self.split_feedback_source_routing_delta is not None:
                routing_delta = self.split_feedback_source_routing_delta.detach().float()
                stats["split_feedback_source_routing_delta_mean"] = float(routing_delta.mean().item())
                stats["split_feedback_source_routing_delta_std"] = float(routing_delta.std(unbiased=False).item())
                base_routing_values = []
                for processor in self.processors.values():
                    tensor = getattr(processor, "routing_scale", None)
                    if tensor is not None:
                        base_routing_values.append(tensor.detach().float().reshape(1))
                base_routing = (
                    float(torch.cat(base_routing_values).mean().item())
                    if base_routing_values
                    else 0.0
                )
                for index, name in enumerate(self.split_feedback_source_names):
                    stats[f"split_feedback_routing_{name}"] = float(base_routing + routing_delta[index].item())
            if self.split_feedback_source_i2f_gate_delta is not None:
                gate_delta = self.split_feedback_source_i2f_gate_delta.detach().float()
                stats["split_feedback_source_i2f_gate_delta_mean"] = float(gate_delta.mean().item())
                stats["split_feedback_source_i2f_gate_delta_std"] = float(gate_delta.std(unbiased=False).item())
                base_gate_values = []
                for processor in self.processors.values():
                    tensor = getattr(processor, "gate_i2f", None)
                    if tensor is not None:
                        base_gate_values.append(tensor.detach().float().reshape(1))
                base_gate = float(torch.cat(base_gate_values).mean().item()) if base_gate_values else 0.0
                for index, name in enumerate(self.split_feedback_source_names):
                    stats[f"split_feedback_gate_i2f_{name}"] = float(base_gate + gate_delta[index].item())
            if self.split_feedback_source_i2f_lora_gain_delta is not None:
                lora_gain_delta = self.split_feedback_source_i2f_lora_gain_delta.detach().float()
                stats["split_feedback_source_i2f_lora_gain_delta_mean"] = float(lora_gain_delta.mean().item())
                stats["split_feedback_source_i2f_lora_gain_delta_std"] = float(
                    lora_gain_delta.std(unbiased=False).item()
                )
                for index, name in enumerate(self.split_feedback_source_names):
                    stats[f"split_feedback_i2f_lora_gain_{name}"] = float(1.0 + lora_gain_delta[index].item())
            if self.split_feedback_source_token_lora_down is not None:
                down = self.split_feedback_source_token_lora_down.detach().float()
                up = self.split_feedback_source_token_lora_up.detach().float()
                stats["split_feedback_source_token_lora_rank"] = float(self.split_feedback_source_token_lora_rank)
                stats["split_feedback_source_token_lora_alpha"] = float(self.split_feedback_source_token_lora_alpha)
                stats["split_feedback_source_token_lora_down_norm_mean"] = float(
                    down.flatten(1).norm(dim=1).mean().item()
                )
                stats["split_feedback_source_token_lora_up_norm_mean"] = float(
                    up.flatten(1).norm(dim=1).mean().item()
                )
                for index, name in enumerate(self.split_feedback_source_names):
                    stats[f"split_feedback_source_token_lora_down_norm_{name}"] = float(
                        down[index].norm().item()
                    )
                    stats[f"split_feedback_source_token_lora_up_norm_{name}"] = float(up[index].norm().item())

            for attr in ("gate_i2t", "gate_t2i", "gate_i2f", "routing_scale", "readout_scale"):
                values = []
                for processor in self.processors.values():
                    tensor = getattr(processor, attr, None)
                    if tensor is not None:
                        values.append(tensor)
                self._summarize_tensors(stats, attr, values)

            for attr in ("i2f_down", "i2f_up"):
                values = []
                for processor in self.processors.values():
                    module = getattr(processor, attr, None)
                    if module is not None:
                        values.append(module.weight.detach().float().norm().reshape(1))
                self._summarize_tensors(stats, f"{attr}_weight_norm", values)
        return stats

    def save_adapter(self, output_dir: Path | str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "adapter.pt")
        (output / "adapter_config.json").write_text(json.dumps(self.adapter_config(), indent=2) + "\n", encoding="utf-8")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Older checkpoints predate routing_scale. Preserve their old behavior by
        # filling the missing routing gates with the module defaults.
        if strict:
            state_dict = dict(state_dict)
            current = super().state_dict()
            for key, value in current.items():
                if key.endswith(".routing_scale") and key not in state_dict:
                    state_dict[key] = value.detach().clone()
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError as exc:
            if "assign" not in str(exc):
                raise
            return super().load_state_dict(state_dict, strict=strict)

    def _source_indices(
        self,
        prompt_sources: str | Sequence[str] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        sources = prompt_sources if prompt_sources is not None else self._runtime_prompt_sources
        if sources is None:
            source_list = ["unknown"] * batch_size
        elif isinstance(sources, str):
            source_list = [_normalize_prompt_source(sources)] * batch_size
        else:
            source_list = [_normalize_prompt_source(source) for source in sources]
            if len(source_list) == 1:
                source_list = source_list * batch_size
            elif len(source_list) != batch_size and len(source_list) > 0 and batch_size % len(source_list) == 0:
                source_list = source_list * (batch_size // len(source_list))
        if len(source_list) != batch_size:
            raise ValueError(
                f"prompt_sources length {len(source_list)} does not match adapter batch size {batch_size}"
            )
        unknown_index = self._source_to_index.get("unknown", 0)
        indices = [self._source_to_index.get(source, unknown_index) for source in source_list]
        return torch.tensor(indices, device=device, dtype=torch.long)

    def _augment_context(
        self,
        encoder_hidden_states: torch.Tensor,
        prompt_sources: str | Sequence[str] | None = None,
    ):
        semantic_len = encoder_hidden_states.shape[1]
        feedback_len = 0
        if self.mode == "feedback_tokens":
            feedback = self.feedback_tokens.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            feedback = feedback.unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, feedback], dim=1)
            feedback_len = feedback.shape[1]
        elif self.mode == "split_text":
            gain = self.split_feedback_gain.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            if self.split_feedback_source_gain_delta is not None:
                source_indices = self._source_indices(
                    prompt_sources,
                    batch_size=encoder_hidden_states.shape[0],
                    device=encoder_hidden_states.device,
                )
                source_gain = self.split_feedback_source_gain_delta[source_indices].to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                )
                gain = gain.view(1, 1, 1) + source_gain.view(-1, 1, 1)
            feedback = gain * encoder_hidden_states
            source_token_lora = self._source_token_lora_delta(encoder_hidden_states, prompt_sources)
            if source_token_lora is not None:
                feedback = feedback + source_token_lora
            if self.split_feedback_bias is not None:
                bias = self.split_feedback_bias.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                feedback = feedback + bias.view(1, 1, -1)
            if self.split_feedback_source_bias is not None:
                source_indices = self._source_indices(
                    prompt_sources,
                    batch_size=encoder_hidden_states.shape[0],
                    device=encoder_hidden_states.device,
                )
                source_bias = self.split_feedback_source_bias[source_indices].to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                )
                feedback = feedback + source_bias.view(encoder_hidden_states.shape[0], 1, -1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, feedback], dim=1)
            feedback_len = semantic_len
        elif self.mode == "contrast_readout":
            # Exact clean copy of the text stream; kept image-blind by the
            # processor masking so it stays a semantic reference.
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=1)
            feedback_len = semantic_len
        return encoder_hidden_states, semantic_len, feedback_len

    def _source_token_lora_delta(
        self,
        encoder_hidden_states: torch.Tensor,
        prompt_sources: str | Sequence[str] | None = None,
    ) -> torch.Tensor | None:
        if self.split_feedback_source_token_lora_down is None:
            return None
        source_indices = self._source_indices(
            prompt_sources,
            batch_size=encoder_hidden_states.shape[0],
            device=encoder_hidden_states.device,
        )
        down = self.split_feedback_source_token_lora_down[source_indices].to(
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        up = self.split_feedback_source_token_lora_up[source_indices].to(
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        x = self.split_feedback_source_token_lora_norm(encoder_hidden_states)
        x = torch.bmm(x, down)
        x = torch.bmm(x, up)
        x = x * (self.split_feedback_source_token_lora_alpha / max(1, self.split_feedback_source_token_lora_rank))
        return x.to(dtype=encoder_hidden_states.dtype)

    def _source_gate_adjustments(
        self,
        prompt_sources: str | Sequence[str] | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if (
            self.split_feedback_source_routing_delta is None
            and self.split_feedback_source_i2f_gate_delta is None
            and self.split_feedback_source_i2f_lora_gain_delta is None
        ):
            return None, None, None
        source_indices = self._source_indices(prompt_sources, batch_size=batch_size, device=device)
        routing_delta = None
        if self.split_feedback_source_routing_delta is not None:
            routing_delta = self.split_feedback_source_routing_delta[source_indices].to(device=device)
        i2f_gate_delta = None
        if self.split_feedback_source_i2f_gate_delta is not None:
            i2f_gate_delta = self.split_feedback_source_i2f_gate_delta[source_indices].to(device=device)
        i2f_lora_gain_delta = None
        if self.split_feedback_source_i2f_lora_gain_delta is not None:
            i2f_lora_gain_delta = self.split_feedback_source_i2f_lora_gain_delta[source_indices].to(device=device)
        return routing_delta, i2f_gate_delta, i2f_lora_gain_delta

    def _rho_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.detach().float()
        return torch.clamp(timestep / self.timestep_max, min=0.0, max=1.0)

    @staticmethod
    def _add_optional_tensors(
        left: torch.Tensor | None,
        right: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if left is None:
            return right
        if right is None:
            return left
        return left + right.to(device=left.device, dtype=left.dtype)

    def _timestep_gate_adjustments(self, timestep: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        self._last_timestep_gate_values = []
        self._last_rho = self._rho_from_timestep(timestep).to(device=timestep.device)
        if not self.use_timestep_gates or self.timestep_gate_mlp is None or self.timestep_layer_embed is None:
            return {}
        rho = self._last_rho.to(device=timestep.device, dtype=torch.float32)
        middle_weight = torch.clamp(4.0 * rho * (1.0 - rho), min=0.0, max=1.0)
        late_weight = 1.0 - rho
        gate_param = next(self.timestep_gate_mlp.parameters())
        gate_dtype = gate_param.dtype
        adjustments: dict[str, dict[str, torch.Tensor]] = {}
        for layer_index in self.adapter_layer_indices:
            layer_ids = torch.full(
                (rho.shape[0],),
                int(layer_index),
                device=timestep.device,
                dtype=torch.long,
            )
            layer_embed = self.timestep_layer_embed(layer_ids).to(dtype=gate_dtype)
            gate_input = torch.cat([rho.reshape(-1, 1).to(dtype=gate_dtype), layer_embed], dim=-1)
            raw = self.timestep_gate_mlp(gate_input)
            rho_for_raw = rho.to(dtype=raw.dtype)
            middle_for_raw = middle_weight.to(dtype=raw.dtype)
            late_for_raw = late_weight.to(dtype=raw.dtype)
            gate_i2s_delta = self.timestep_gate_scale * raw[:, 0] * (0.5 + rho_for_raw)
            gate_i2f_delta = self.timestep_gate_scale * raw[:, 1] * middle_for_raw
            gate_f2i_delta = self.timestep_gate_scale * raw[:, 2] * middle_for_raw
            semantic_leak = self.max_semantic_leak * (2.0 * torch.sigmoid(raw[:, 3]) - 1.0) * late_for_raw
            values = {
                "gate_i2s_delta": gate_i2s_delta,
                "gate_i2f_delta": gate_i2f_delta,
                "gate_f2i_delta": gate_f2i_delta,
                "semantic_leak": semantic_leak,
            }
            adjustments[str(layer_index)] = values
            self._last_timestep_gate_values.append(values)
        return adjustments

    def _set_processor_layout(
        self,
        semantic_len: int,
        feedback_len: int,
        source_routing_delta: torch.Tensor | None = None,
        source_i2f_gate_delta: torch.Tensor | None = None,
        source_i2f_lora_gain_delta: torch.Tensor | None = None,
        timestep_gate_adjustments: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> None:
        timestep_gate_adjustments = timestep_gate_adjustments or {}
        for key, processor in self.processors.items():
            gate_values = timestep_gate_adjustments.get(key, {})
            gate_i2f_delta = self._add_optional_tensors(
                source_i2f_gate_delta,
                gate_values.get("gate_i2f_delta"),
            )
            processor.set_token_layout(semantic_len, feedback_len)
            processor.set_source_gate_adjustments(
                source_routing_delta,
                gate_i2f_delta,
                source_i2f_lora_gain_delta,
                gate_values.get("gate_i2s_delta"),
                gate_values.get("gate_f2i_delta"),
                gate_values.get("semantic_leak"),
            )

    def regularization_losses(
        self,
        *,
        semantic_weight: float = 0.0,
        leak_weight: float = 0.0,
        gate_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        rho = self._last_rho

        if (
            float(semantic_weight) > 0.0
            and self._last_semantic_reference is not None
            and self._last_semantic_states
        ):
            reference = self._last_semantic_reference.detach()
            per_layer = []
            for semantic_state in self._last_semantic_states:
                cosine = F.cosine_similarity(semantic_state.float(), reference.float(), dim=-1)
                per_layer.append((1.0 - cosine).mean(dim=1))
            per_sample = torch.stack(per_layer, dim=0).mean(dim=0)
            if rho is None:
                weight = float(semantic_weight)
            else:
                weight = float(semantic_weight) * (1.0 + rho.to(device=per_sample.device, dtype=per_sample.dtype))
            weighted = (per_sample * weight).mean()
            losses["semantic_preservation_loss"] = per_sample.mean()
            losses["semantic_preservation_weighted_loss"] = weighted
            total = weighted if total is None else total + weighted

        if float(leak_weight) > 0.0 and self._last_timestep_gate_values:
            leak_values = [values["semantic_leak"].float() for values in self._last_timestep_gate_values]
            leak = torch.stack(leak_values, dim=0)
            per_sample = leak.square().mean(dim=0)
            if rho is None:
                weight = float(leak_weight)
            else:
                weight = float(leak_weight) * (1.0 + 2.0 * rho.to(device=per_sample.device, dtype=per_sample.dtype))
            weighted = (per_sample * weight).mean()
            losses["semantic_leak_penalty"] = per_sample.mean()
            losses["semantic_leak_weighted_penalty"] = weighted
            total = weighted if total is None else total + weighted

        if float(gate_weight) > 0.0 and self._last_timestep_gate_values:
            gate_terms = []
            for values in self._last_timestep_gate_values:
                gate_terms.extend(
                    [
                        values["gate_i2s_delta"].float(),
                        values["gate_i2f_delta"].float(),
                        values["gate_f2i_delta"].float(),
                    ]
                )
            gate = torch.stack(gate_terms, dim=0)
            weighted = float(gate_weight) * gate.square().mean()
            losses["timestep_gate_regularization"] = gate.square().mean()
            losses["timestep_gate_weighted_regularization"] = weighted
            total = weighted if total is None else total + weighted

        if total is not None:
            losses["bridge_causal_regularization_loss"] = total
        return losses

    def forward(
        self,
        transformer,
        *,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        pooled_projections: torch.FloatTensor,
        timestep: torch.LongTensor,
        prompt_sources: str | Sequence[str] | None = None,
        return_dict: bool = False,
    ):
        if not self._installed:
            self.install(transformer)

        height, width = hidden_states.shape[-2:]
        hidden_states = transformer.pos_embed(hidden_states)
        temb = transformer.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)
        semantic_reference = encoder_hidden_states
        encoder_hidden_states, semantic_len, feedback_len = self._augment_context(encoder_hidden_states, prompt_sources)
        source_routing_delta, source_i2f_gate_delta, source_i2f_lora_gain_delta = self._source_gate_adjustments(
            prompt_sources,
            batch_size=encoder_hidden_states.shape[0],
            device=encoder_hidden_states.device,
        )
        timestep_gate_adjustments = self._timestep_gate_adjustments(timestep)
        self._set_processor_layout(
            semantic_len,
            feedback_len,
            source_routing_delta,
            source_i2f_gate_delta,
            source_i2f_lora_gain_delta,
            timestep_gate_adjustments,
        )
        self._last_semantic_reference = semantic_reference
        self._last_semantic_states = []

        for index, block in enumerate(transformer.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )
            if index in self.adapter_layer_set and semantic_len > 0 and encoder_hidden_states is not None:
                self._last_semantic_states.append(encoder_hidden_states[:, :semantic_len])

        hidden_states = transformer.norm_out(hidden_states, temb)
        hidden_states = transformer.proj_out(hidden_states)

        patch_size = transformer.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], transformer.out_channels, height * patch_size, width * patch_size)
        )

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        return (output,)


def mark_only_bridge_causal_trainable(adapter: BridgeCausalSD3Adapter, modules: Iterable[nn.Module]) -> None:
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(False)
    for param in adapter.parameters():
        param.requires_grad_(True)
