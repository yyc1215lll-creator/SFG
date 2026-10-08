import argparse
import importlib.util
import inspect
import math
import os
import re
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import FluxPipeline as DiffusersFluxPipeline
from diffusers import FluxTransformer2DModel
from diffusers.models.attention_processor import Attention
try:
    from diffusers.models.transformers.transformer_flux import FluxAttention
except ImportError:  # Diffusers versions before the dedicated FLUX attention class.
    FluxAttention = Attention

FLUX_ATTENTION_TYPES = (Attention, FluxAttention)
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image


class FluxBridgeAttnProcessor2_0:
    """SFG/DirectSFG processor for FLUX dual-stream joint attention."""

    def __init__(
        self,
        sfg_strength_u=0.2,
        bridge_scale_delta=None,
        sfg_strength_u_t2i=None,
        bridge_scale_delta_t2i=None,
        bridge_direction="image_to_text",
        clean_prefix_batch_size=0,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxBridgeAttnProcessor2_0 requires PyTorch 2.0 scaled dot-product attention.")
        sfg_strength_u = float(sfg_strength_u)
        if sfg_strength_u < -1.0 or sfg_strength_u > 1.0:
            raise ValueError("sfg_strength_u must be in [-1, 1].")
        if sfg_strength_u_t2i is None or sfg_strength_u_t2i == "":
            sfg_strength_u_t2i = sfg_strength_u
        sfg_strength_u_t2i = float(sfg_strength_u_t2i)
        if sfg_strength_u_t2i < -1.0 or sfg_strength_u_t2i > 1.0:
            raise ValueError("sfg_strength_u_t2i must be in [-1, 1].")

        direction = normalize_bridge_direction(bridge_direction)
        primary_delta = -sfg_strength_u if bridge_scale_delta is None else float(bridge_scale_delta)
        t2i_delta = -sfg_strength_u_t2i if bridge_scale_delta_t2i is None else float(bridge_scale_delta_t2i)
        if direction == "image_to_text":
            self.bridge_scale_delta_i2t = primary_delta
            self.bridge_scale_delta_t2i = 0.0
        elif direction == "text_to_image":
            self.bridge_scale_delta_i2t = 0.0
            self.bridge_scale_delta_t2i = primary_delta
        else:
            self.bridge_scale_delta_i2t = primary_delta
            self.bridge_scale_delta_t2i = t2i_delta
        self.clean_prefix_batch_size = max(0, int(clean_prefix_batch_size))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]

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

        text_seq_len = 0
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
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

            # FLUX dual blocks concatenate text first and image second.
            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        if encoder_hidden_states is not None and text_seq_len > 0:
            weak_start = min(self.clean_prefix_batch_size, batch_size)
            if weak_start < batch_size:
                total_seq_len = key.shape[2]
                bridge_specs = []
                if self.bridge_scale_delta_i2t != 0.0:
                    bridge_specs.append(
                        (text_seq_len, total_seq_len, 0, text_seq_len, self.bridge_scale_delta_i2t)
                    )
                if self.bridge_scale_delta_t2i != 0.0:
                    bridge_specs.append(
                        (0, text_seq_len, text_seq_len, total_seq_len, self.bridge_scale_delta_t2i)
                    )
                if bridge_specs:
                    updated = output.clone()
                    for target_start, target_end, source_start, source_end, scale_delta in bridge_specs:
                        contrib = self._value_slice_contribution(
                            query[weak_start:, :, target_start:target_end],
                            key[weak_start:],
                            value[weak_start:],
                            source_start,
                            source_end,
                        )
                        updated[weak_start:, :, target_start:target_end] = (
                            updated[weak_start:, :, target_start:target_end] + scale_delta * contrib
                        )
                    output = updated

        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        output = output.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = output[:, :text_seq_len], output[:, text_seq_len:]
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return output

    @staticmethod
    def _value_slice_contribution(query_target, key_all, value_all, source_start, source_end):
        value_source_only = torch.zeros_like(value_all)
        value_source_only[:, :, source_start:source_end] = value_all[:, :, source_start:source_end]
        return F.scaled_dot_product_attention(
            query_target,
            key_all,
            value_source_only,
            dropout_p=0.0,
            is_causal=False,
        )


def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    if height <= 1:
        return img
    kernel_size = min(int(kernel_size), height - (height % 2 - 1))
    kernel_size = max(kernel_size, 3)
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=img.device, dtype=torch.float32)
    pdf = torch.exp(-0.5 * (x / float(sigma)).pow(2))
    x_kernel = (pdf / pdf.sum()).to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
    img = F.pad(img, [kernel_size // 2] * 4, mode="reflect")
    return F.conv2d(img, kernel2d, groups=img.shape[-3])


def infer_flux_single_text_seq_len(seq_len, expected_text_seq_len=None, max_text_seq_len=512):
    if expected_text_seq_len is not None:
        expected_text_seq_len = int(expected_text_seq_len)
        image_seq_len = int(seq_len) - expected_text_seq_len
        side = math.isqrt(max(0, image_seq_len))
        if 0 < expected_text_seq_len < seq_len and side * side == image_seq_len:
            return expected_text_seq_len

    for side in range(math.isqrt(max(0, int(seq_len) - 1)), 0, -1):
        image_seq_len = side * side
        text_seq_len = int(seq_len) - image_seq_len
        if 0 < text_seq_len <= int(max_text_seq_len):
            return text_seq_len
    return None


class FluxSEGAttnProcessor2_0:
    """SEG processor for FLUX dual-stream and single-stream joint attention."""

    def __init__(self, blur_sigma=9999999.0, inf_blur_threshold=9999.0, single_text_seq_len=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxSEGAttnProcessor2_0 requires PyTorch 2.0 scaled dot-product attention.")
        self.blur_sigma = float(blur_sigma)
        self.inf_blur = self.blur_sigma >= float(inf_blur_threshold) or self.blur_sigma <= 0.0
        self.single_text_seq_len = None if single_text_seq_len is None else int(single_text_seq_len)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]

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

        text_seq_len = 0
        if encoder_hidden_states is not None:
            image_seq_len = hidden_states.shape[1]
            side = math.isqrt(image_seq_len)
            if side * side == image_seq_len:
                query = self._blur_query(query, batch_size, attn.heads, head_dim, side)

            text_seq_len = encoder_hidden_states.shape[1]
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

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)
        else:
            text_seq_len = infer_flux_single_text_seq_len(query.shape[2], self.single_text_seq_len)
            if text_seq_len is not None:
                image_seq_len = query.shape[2] - text_seq_len
                side = math.isqrt(image_seq_len)
                if side * side == image_seq_len:
                    query_text = query[:, :, :text_seq_len]
                    query_img = self._blur_query(
                        query[:, :, text_seq_len:],
                        batch_size,
                        attn.heads,
                        head_dim,
                        side,
                    )
                    query = torch.cat([query_text, query_img], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        output = output.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = output[:, :text_seq_len], output[:, text_seq_len:]
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return output

    def _blur_query(self, query, batch_size, heads, head_dim, side):
        query = query.permute(0, 1, 3, 2).reshape(batch_size, heads * head_dim, side, side)
        if self.inf_blur:
            query = query.mean(dim=(-2, -1), keepdim=True).expand_as(query)
        else:
            kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
            query = gaussian_blur_2d(query, kernel_size, self.blur_sigma)
        return query.reshape(batch_size, heads, head_dim, side * side).permute(0, 1, 3, 2)


class FluxPAGAttnProcessor2_0:
    """PAG processor for FLUX dual-stream and single-stream joint attention."""

    image_query_chunk_size = 256

    def __init__(self, single_text_seq_len=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxPAGAttnProcessor2_0 requires PyTorch 2.0 scaled dot-product attention.")
        self.single_text_seq_len = None if single_text_seq_len is None else int(single_text_seq_len)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]

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

        text_seq_len = 0
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
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

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if encoder_hidden_states is None:
            text_seq_len = infer_flux_single_text_seq_len(query.shape[2], self.single_text_seq_len) or 0

        if text_seq_len > 0:
            query_text = query[:, :, :text_seq_len]
            query_img = query[:, :, text_seq_len:]
            key_text = key[:, :, :text_seq_len]
            key_img = key[:, :, text_seq_len:]
            value_text = value[:, :, :text_seq_len]
            value_img = value[:, :, text_seq_len:]

            text_out = F.scaled_dot_product_attention(query_text, key, value, dropout_p=0.0, is_causal=False)
            image_out = self._chunked_identity_mask_attention(
                query_img,
                key_img,
                value_img,
                key_text,
                value_text,
                head_dim,
            )
            output = torch.cat([text_out, image_out], dim=2)
        else:
            output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        output = output.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = output[:, :text_seq_len], output[:, text_seq_len:]
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return output

    def _chunked_identity_mask_attention(self, query_img, key_img, value_img, key_text, value_text, head_dim):
        scale = head_dim ** -0.5
        chunks = []
        chunk_size = self.image_query_chunk_size

        for start in range(0, query_img.shape[2], chunk_size):
            end = min(start + chunk_size, query_img.shape[2])
            q = query_img[:, :, start:end]
            k_diag = key_img[:, :, start:end]
            v_diag = value_img[:, :, start:end]

            diag_scores = (q.float() * k_diag.float()).sum(dim=-1, keepdim=True) * scale
            text_scores = torch.matmul(q.float(), key_text.float().transpose(-2, -1)) * scale
            scores = torch.cat([diag_scores, text_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1).to(q.dtype)

            diag_out = weights[..., :1] * v_diag
            text_out = torch.matmul(weights[..., 1:], value_text)
            chunks.append(diag_out + text_out)

        return torch.cat(chunks, dim=2)


def normalize_bridge_direction(direction):
    direction = str(direction or "image_to_text").strip().lower().replace("-", "_")
    aliases = {
        "i2t": "image_to_text",
        "img2txt": "image_to_text",
        "image2text": "image_to_text",
        "image_to_text": "image_to_text",
        "t2i": "text_to_image",
        "txt2img": "text_to_image",
        "text2image": "text_to_image",
        "text_to_image": "text_to_image",
        "both": "both",
        "bidirectional": "both",
        "bi_directional": "both",
        "bidir": "both",
        "i2t_t2i": "both",
        "image_to_text_and_text_to_image": "both",
    }
    direction = aliases.get(direction, direction)
    if direction not in {"image_to_text", "text_to_image", "both"}:
        raise ValueError(f"Unsupported bridge_direction: {direction}")
    return direction


def normalize_bridge_variant(sfg_mode, bridge_variant):
    mode = str(sfg_mode or "none").strip().lower()
    variant = str(bridge_variant or "explicit").strip().lower()
    if mode in {"direct_sfg", "direct_sfg_alt"}:
        variant = "direct_sfg"
    aliases = {
        "explicit_sfg": "explicit",
        "sfg": "explicit",
        "bridge": "explicit",
        "direct_sfg_alt": "direct_sfg",
    }
    variant = aliases.get(variant, variant)
    if variant not in {"explicit", "direct_sfg"}:
        raise ValueError(f"Unsupported bridge_variant: {bridge_variant}")
    return variant


def normalize_cfgzero_mode(mode):
    mode = str(mode or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "0": "none",
        "false": "none",
        "zero": "zero_init",
        "zero_init": "zero_init",
        "zeroinit": "zero_init",
        "cfg_zero": "cfg_zero_star",
        "cfgzero": "cfg_zero_star",
        "cfg_zero_star": "cfg_zero_star",
        "cfgzero_star": "cfg_zero_star",
        "cfg_zero*": "cfg_zero_star",
        "cfgzero*": "cfg_zero_star",
        "full": "cfg_zero_star",
        "star": "cfg_zero_star",
        "optimized": "optimized_scale",
        "optimized_scale": "optimized_scale",
        "opt_scale": "optimized_scale",
        "scaler": "optimized_scale",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "zero_init", "optimized_scale", "cfg_zero_star"}:
        raise ValueError(f"Unsupported cfgzero_mode: {mode}")
    return mode


def cfgzero_uses_zero_init(mode):
    return mode in {"zero_init", "cfg_zero_star"}


def cfgzero_uses_optimized_scale(mode):
    return mode in {"optimized_scale", "cfg_zero_star"}


def resolve_cfgzero_zero_steps(mode, requested_steps, total_steps):
    requested_steps = int(requested_steps)
    if requested_steps < 0:
        raise ValueError("--cfgzero-zero-steps must be >= 0")
    if not cfgzero_uses_zero_init(mode):
        return requested_steps
    if requested_steps > 0:
        return requested_steps
    return max(1, int(round(0.04 * int(total_steps))))


def optimized_cfgzero_scale(noise_text, noise_uncond):
    positive = noise_text.float().flatten(1)
    negative = noise_uncond.float().flatten(1)
    dot_product = torch.sum(positive * negative, dim=1, keepdim=True)
    squared_norm = torch.sum(negative.square(), dim=1, keepdim=True).clamp_min(1e-8)
    scale = dot_product / squared_norm
    view_shape = [noise_text.shape[0]] + [1] * (noise_text.ndim - 1)
    return scale.reshape(view_shape).to(dtype=noise_text.dtype, device=noise_text.device)


def use_bridge_step(step_index, start_step, end_step):
    step_number = int(step_index) + 1
    if step_number < int(start_step):
        return False
    end_step = int(end_step)
    if end_step > 0 and step_number > end_step:
        return False
    return True


def flux_s2_candidate_blocks(pipe, block_start, block_end, block_family):
    block_family = str(block_family).lower()
    num_dual_blocks = len(pipe.transformer.transformer_blocks)
    num_single_blocks = len(getattr(pipe.transformer, "single_transformer_blocks", []))
    if block_family == "dual":
        num_blocks = num_dual_blocks
    elif block_family == "single":
        num_blocks = num_single_blocks
    elif block_family == "all":
        num_blocks = num_dual_blocks + num_single_blocks
    else:
        raise ValueError(f"Unsupported S2 block family: {block_family}")
    start = max(int(block_start), 0)
    end = int(block_end)
    if end <= 0:
        end = int(num_blocks)
    end = min(end, int(num_blocks))
    if start >= end:
        raise ValueError(
            f"S2 block candidate range is empty: start={block_start}, end={block_end}, "
            f"num_blocks={num_blocks}."
        )
    candidates = []
    for global_index in range(start, end):
        if block_family == "dual":
            candidates.append(("dual", global_index))
        elif block_family == "single":
            candidates.append(("single", global_index))
        elif global_index < num_dual_blocks:
            candidates.append(("dual", global_index))
        else:
            candidates.append(("single", global_index - num_dual_blocks))
    return candidates


def sample_s2_block_indices(candidates, num_drop_blocks, device):
    num_drop_blocks = int(num_drop_blocks)
    if num_drop_blocks < 0:
        raise ValueError("--s2-num-drop-blocks must be >= 0")
    if num_drop_blocks == 0:
        return []
    if num_drop_blocks > len(candidates):
        raise ValueError(
            f"s2_num_drop_blocks={num_drop_blocks} exceeds available S2 candidate blocks "
            f"({len(candidates)})."
        )
    perm = torch.randperm(len(candidates), device=device)[:num_drop_blocks]
    selected = sorted(int(index) for index in perm.detach().cpu().tolist())
    return [candidates[index] for index in selected]


@contextmanager
def drop_flux_transformer_blocks(transformer, block_indices):
    saved_forwards = []

    def dropped_dual_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        **_kwargs,
    ):
        return encoder_hidden_states, hidden_states

    def dropped_single_forward(
        self,
        hidden_states,
        temb,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        **_kwargs,
    ):
        return hidden_states

    try:
        normalized = []
        for block_spec in block_indices:
            if isinstance(block_spec, tuple):
                family, index = block_spec
            else:
                family, index = "dual", block_spec
            normalized.append((str(family), int(index)))
        for family, index in sorted(set(normalized)):
            if family == "dual":
                block = transformer.transformer_blocks[index]
                replacement = dropped_dual_forward
            elif family == "single":
                block = transformer.single_transformer_blocks[index]
                replacement = dropped_single_forward
            else:
                raise ValueError(f"Unsupported S2 dropped block family: {family}")
            saved_forwards.append((block, block.forward))
            block.forward = MethodType(replacement, block)
        yield
    finally:
        for block, forward in reversed(saved_forwards):
            block.forward = forward


def normalize_flux_bridge_layers(pipe, layers):
    num_blocks = len(pipe.transformer.transformer_blocks)

    def block_range(start, end):
        return [rf"transformer_blocks\.{index}\.attn$" for index in range(start, end)]

    def normalize_one(layer):
        stripped = str(layer).strip()
        lowered = stripped.lower()
        if lowered in {"", "all", "default"}:
            return [r"transformer_blocks\.\d+\.attn$"]
        if lowered in {"early", "first"}:
            return block_range(0, max(1, num_blocks // 3))
        if lowered in {"mid", "middle"}:
            return block_range(num_blocks // 3, max(num_blocks // 3 + 1, (2 * num_blocks) // 3))
        if lowered in {"late", "last"}:
            return block_range((2 * num_blocks) // 3, num_blocks)
        match = re.fullmatch(r"blocks\.(\d+)", stripped)
        if match:
            return [rf"transformer_blocks\.{match.group(1)}\.attn$"]
        if stripped.isdigit():
            return [rf"transformer_blocks\.{stripped}\.attn$"]
        return [stripped]

    raw_layers = [item for item in re.split(r"[\s,;]+", str(layers or "all")) if item]
    normalized = []
    for layer in raw_layers:
        normalized.extend(normalize_one(layer))
    return normalized


def normalize_flux_self_guidance_layers(pipe, layers):
    num_dual_blocks = len(pipe.transformer.transformer_blocks)
    num_single_blocks = len(getattr(pipe.transformer, "single_transformer_blocks", []))

    def family_block_pattern(family, index):
        if family == "dual":
            return rf"transformer_blocks\.{index}\.attn$"
        if family == "single":
            return rf"single_transformer_blocks\.{index}\.attn$"
        raise ValueError(f"Unsupported FLUX self-guidance block family: {family}")

    def both_family_block(index):
        patterns = []
        if 0 <= index < num_dual_blocks:
            patterns.append(family_block_pattern("dual", index))
        if 0 <= index < num_single_blocks:
            patterns.append(family_block_pattern("single", index))
        if not patterns:
            raise ValueError(
                f"FLUX self-guidance block index {index} is outside dual={num_dual_blocks}, "
                f"single={num_single_blocks}."
            )
        return patterns

    def family_range(family, start, end):
        count = num_dual_blocks if family == "dual" else num_single_blocks
        start = max(0, min(int(start), count))
        end = max(start, min(int(end), count))
        return [family_block_pattern(family, index) for index in range(start, end)]

    def ratio_range(count, start_ratio, end_ratio):
        start = int(count * start_ratio)
        end = int(count * end_ratio)
        return start, max(start + 1, end)

    def normalize_one(layer):
        stripped = str(layer).strip()
        lowered = stripped.lower()
        if lowered in {"", "all", "default"}:
            return [r"(?:transformer_blocks|single_transformer_blocks)\.\d+\.attn$"]
        if lowered in {"early", "first"}:
            dual_start, dual_end = ratio_range(num_dual_blocks, 0.0, 1.0 / 3.0)
            single_start, single_end = ratio_range(num_single_blocks, 0.0, 1.0 / 3.0)
            return family_range("dual", dual_start, dual_end) + family_range("single", single_start, single_end)
        if lowered in {"mid", "middle"}:
            dual_start, dual_end = ratio_range(num_dual_blocks, 1.0 / 3.0, 2.0 / 3.0)
            single_start, single_end = ratio_range(num_single_blocks, 1.0 / 3.0, 2.0 / 3.0)
            return family_range("dual", dual_start, dual_end) + family_range("single", single_start, single_end)
        if lowered in {"late", "last"}:
            dual_start, dual_end = ratio_range(num_dual_blocks, 2.0 / 3.0, 1.0)
            single_start, single_end = ratio_range(num_single_blocks, 2.0 / 3.0, 1.0)
            return family_range("dual", dual_start, dual_end) + family_range("single", single_start, single_end)

        match = re.fullmatch(r"(?:blocks|block)\.(\d+)", lowered)
        if match:
            return both_family_block(int(match.group(1)))
        match = re.fullmatch(r"(?:dual|transformer_blocks)\.?(\d+)(?:\.attn)?", lowered)
        if match:
            return [family_block_pattern("dual", int(match.group(1)))]
        match = re.fullmatch(r"(?:single|single_transformer_blocks)\.?(\d+)(?:\.attn)?", lowered)
        if match:
            return [family_block_pattern("single", int(match.group(1)))]
        match = re.fullmatch(r"global\.(\d+)", lowered)
        if match:
            index = int(match.group(1))
            if index < num_dual_blocks:
                return [family_block_pattern("dual", index)]
            return [family_block_pattern("single", index - num_dual_blocks)]
        if stripped.isdigit():
            return both_family_block(int(stripped))
        return [stripped]

    raw_layers = [item for item in re.split(r"[\s,;]+", str(layers or "all")) if item]
    normalized = []
    for layer in raw_layers:
        normalized.extend(normalize_one(layer))
    return normalized


def flux_bridge_target_modules(pipe, layer_patterns):
    targets = []
    for pattern in layer_patterns:
        for name, module in pipe.transformer.named_modules():
            if not (
                isinstance(module, FLUX_ATTENTION_TYPES)
                and name.startswith("transformer_blocks.")
                and name.endswith(".attn")
                and hasattr(module, "add_q_proj")
                and re.search(pattern, name) is not None
            ):
                continue
            if all(existing is not module for _, existing in targets):
                targets.append((name, module))
    if not targets:
        raise ValueError(f"Cannot find FLUX SFG attention layers: {layer_patterns}")
    return targets


def flux_self_guidance_target_modules(pipe, layer_patterns, method_name):
    targets = []
    for pattern in layer_patterns:
        for name, module in pipe.transformer.named_modules():
            if not (
                isinstance(module, FLUX_ATTENTION_TYPES)
                and (name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks."))
                and name.endswith(".attn")
                and re.search(pattern, name) is not None
            ):
                continue
            if all(existing is not module for _, existing in targets):
                targets.append((name, module))
    if not targets:
        raise ValueError(f"Cannot find FLUX {method_name} attention layers: {layer_patterns}")
    return targets


@contextmanager
def flux_bridge_attn_processors(
    target_modules,
    sfg_strength_u,
    clean_prefix_batch_size=0,
    bridge_scale_delta=None,
    sfg_strength_u_t2i=None,
    bridge_scale_delta_t2i=None,
    bridge_direction="image_to_text",
):
    saved_processors = []
    try:
        for _name, module in target_modules:
            saved_processors.append((module, module.processor))
            processor = FluxBridgeAttnProcessor2_0(
                sfg_strength_u=sfg_strength_u,
                bridge_scale_delta=bridge_scale_delta,
                sfg_strength_u_t2i=sfg_strength_u_t2i,
                bridge_scale_delta_t2i=bridge_scale_delta_t2i,
                bridge_direction=bridge_direction,
                clean_prefix_batch_size=clean_prefix_batch_size,
            )
            module.set_processor(processor)
        yield
    finally:
        for module, processor in reversed(saved_processors):
            module.set_processor(processor)


@contextmanager
def flux_self_guidance_attn_processors(target_modules, processor_factory):
    saved_processors = []
    try:
        for _name, module in target_modules:
            saved_processors.append((module, module.processor))
            module.set_processor(processor_factory())
        yield
    finally:
        for module, processor in reversed(saved_processors):
            module.set_processor(processor)


def normclip_residual(residual, reference, tau, eps=1e-6):
    if float(tau) >= 999.0:
        return residual
    residual_flat = residual.float().flatten(1)
    reference_flat = reference.float().flatten(1)
    residual_norm = residual_flat.norm(dim=1)
    reference_norm = reference_flat.norm(dim=1)
    scale = torch.minimum(
        torch.ones_like(residual_norm),
        float(tau) * reference_norm / residual_norm.clamp_min(eps),
    )
    view_shape = [residual.shape[0]] + [1] * (residual.ndim - 1)
    return residual * scale.reshape(view_shape).to(residual.dtype)


def load_pipeline_class(pipeline_file: Path | None, class_name: str):
    if pipeline_file is None:
        return DiffusersFluxPipeline
    pipeline_file = Path(pipeline_file)
    if not pipeline_file.exists():
        raise FileNotFoundError(f"pipeline file not found: {pipeline_file}")
    module_name = f"sfg_flux_pipeline_{abs(hash(str(pipeline_file.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pipeline file: {pipeline_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{pipeline_file} does not define {class_name}") from exc


def auto_pipeline_file(transformer_dir: Path | None, pipeline_file: Path | None) -> Path | None:
    if pipeline_file is not None:
        return pipeline_file
    if transformer_dir is None:
        return None
    candidate = Path(transformer_dir) / "pipeline_flux_de_distill.py"
    return candidate if candidate.exists() else None


def load_flux_pipeline(args, device):
    pipeline_file = auto_pipeline_file(args.transformer_dir, args.pipeline_file)
    pipeline_cls = load_pipeline_class(pipeline_file, args.pipeline_class)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if args.transformer_dir is not None:
        transformer = FluxTransformer2DModel.from_pretrained(
            str(args.transformer_dir),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        kwargs["transformer"] = transformer
    pipe = pipeline_cls.from_pretrained(str(args.model_dir), **kwargs)
    pipe.to(device=device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, pipeline_file


def pipeline_supports_negative_prompt(pipe) -> bool:
    return "negative_prompt" in inspect.signature(pipe.encode_prompt).parameters


def normalize_negative_prompt(negative_prompt: str | None, batch_size: int):
    prompt = "" if negative_prompt is None else str(negative_prompt)
    return [prompt] * batch_size


@torch.no_grad()
def flux_bridge_generate(pipe, prompts, width, height, steps, guidance_scale, generators, args):
    variant = normalize_bridge_variant(args.sfg_mode, args.bridge_variant)
    bridge_direction = normalize_bridge_direction(args.bridge_direction)
    cfgzero_mode = normalize_cfgzero_mode(args.cfgzero_mode)
    cfgzero_zero_steps = resolve_cfgzero_zero_steps(cfgzero_mode, args.cfgzero_zero_steps, steps)
    cfgzero_requested = cfgzero_mode != "none" or cfgzero_zero_steps > 0
    bridge_requested = (
        args.bridge_omega != 0.0
        or args.sfg_mode not in {"none", "off", "0"}
        or args.bridge_variant in {"direct_sfg", "direct_sfg_alt"}
    )
    s2_guidance_scale = float(args.s2_guidance_scale)
    s2_enabled = s2_guidance_scale != 0.0
    pag_scale = float(args.pag_scale)
    seg_scale = float(args.seg_scale)
    pag_enabled = pag_scale != 0.0
    seg_enabled = seg_scale != 0.0
    if s2_enabled and s2_guidance_scale <= 0.0:
        raise ValueError("--s2-guidance-scale must be > 0 when S2Guidance is enabled")
    if s2_enabled and int(args.s2_num_drop_blocks) <= 0:
        raise ValueError("--s2-num-drop-blocks must be > 0 when S2Guidance is enabled")
    if pag_enabled and pag_scale <= 0.0:
        raise ValueError("--pag-scale must be > 0 when PAG is enabled")
    if seg_enabled and seg_scale <= 0.0:
        raise ValueError("--seg-scale must be > 0 when SEG is enabled")
    if args.s2_start_step < 1:
        raise ValueError("--s2-start-step must be >= 1")
    if args.s2_end_step < 0:
        raise ValueError("--s2-end-step must be >= 0")
    if args.s2_end_step > 0 and args.s2_end_step < args.s2_start_step:
        raise ValueError("--s2-end-step must be >= --s2-start-step")
    active_methods = sum([s2_enabled, bridge_requested, pag_enabled, seg_enabled])
    if active_methods > 1:
        raise ValueError("FLUX S2Guidance, PAG, SEG, and SFG/DirectSFG are separate methods.")
    if cfgzero_requested and active_methods:
        raise ValueError("FLUX CFG-Zero baseline is kept separate from other guidance baselines.")
    if args.bridge_start_step < 1:
        raise ValueError("--bridge-start-step must be >= 1")
    if args.bridge_end_step < 0:
        raise ValueError("--bridge-end-step must be >= 0")
    if args.bridge_end_step > 0 and args.bridge_end_step < args.bridge_start_step:
        raise ValueError("--bridge-end-step must be >= --bridge-start-step")
    if variant == "explicit" and args.bridge_omega < 0.0:
        raise ValueError("--bridge-omega must be >= 0 for explicit SFG")

    batch_size = len(prompts)
    device = pipe._execution_device
    pipe.check_inputs(
        prompts,
        None,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=512,
    )
    pipe._guidance_scale = guidance_scale
    pipe._joint_attention_kwargs = None
    pipe._interrupt = False

    supports_negative_prompt = pipeline_supports_negative_prompt(pipe)
    negative_prompts = normalize_negative_prompt(args.negative_prompt, batch_size)
    if supports_negative_prompt:
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            negative_prompt=negative_prompts,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
    else:
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generators,
        None,
    )

    sigmas = np.linspace(1.0, 1 / steps, steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        steps,
        device,
        None,
        sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    transformer_uses_guidance_embed = bool(getattr(pipe.transformer.config, "guidance_embeds", False))
    use_true_cfg = supports_negative_prompt and not transformer_uses_guidance_embed and float(guidance_scale) > 1.0
    if cfgzero_uses_optimized_scale(cfgzero_mode) and not use_true_cfg:
        raise ValueError("CFG-Zero optimized scale requires true conditional/unconditional FLUX predictions.")
    if transformer_uses_guidance_embed:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    def transformer_forward(current_latents, current_prompt_embeds=None, current_pooled_prompt_embeds=None):
        current_prompt_embeds = prompt_embeds if current_prompt_embeds is None else current_prompt_embeds
        current_pooled_prompt_embeds = pooled_prompt_embeds if current_pooled_prompt_embeds is None else current_pooled_prompt_embeds
        timestep = current_timestep.expand(current_latents.shape[0]).to(current_latents.dtype)
        current_guidance = None
        if guidance is not None:
            current_guidance = guidance
            if current_latents.shape[0] != guidance.shape[0]:
                current_guidance = guidance[:1].expand(current_latents.shape[0])
        return pipe.transformer(
            hidden_states=current_latents,
            timestep=timestep / 1000,
            guidance=current_guidance,
            pooled_projections=current_pooled_prompt_embeds,
            encoder_hidden_states=current_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    def clean_predictions():
        if not use_true_cfg:
            noise_text = transformer_forward(latents)
            return None, noise_text, noise_text
        latent_model_input = torch.cat([latents] * 2)
        cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        cfg_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        noise_pred = transformer_forward(
            latent_model_input,
            current_prompt_embeds=cfg_prompt_embeds,
            current_pooled_prompt_embeds=cfg_pooled_prompt_embeds,
        )
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_cfg = noise_uncond + float(guidance_scale) * (noise_text - noise_uncond)
        return noise_uncond, noise_text, noise_cfg

    def apply_cfgzero(step_index, noise_uncond, noise_text, noise_pred):
        if cfgzero_uses_optimized_scale(cfgzero_mode):
            scale = optimized_cfgzero_scale(noise_text, noise_uncond)
            scaled_uncond = noise_uncond * scale
            noise_pred = scaled_uncond + float(guidance_scale) * (noise_text - scaled_uncond)
        if cfgzero_uses_zero_init(cfgzero_mode) and step_index < cfgzero_zero_steps:
            noise_pred = torch.zeros_like(noise_pred)
        return noise_pred

    s2_candidates = None
    if s2_enabled:
        s2_candidates = flux_s2_candidate_blocks(
            pipe,
            args.s2_block_start,
            args.s2_block_end,
            args.s2_block_family,
        )

    pag_target_modules = None
    if pag_enabled:
        pag_target_modules = flux_self_guidance_target_modules(
            pipe,
            normalize_flux_self_guidance_layers(pipe, args.pag_applied_layers),
            "PAG",
        )

    seg_target_modules = None
    if seg_enabled:
        seg_target_modules = flux_self_guidance_target_modules(
            pipe,
            normalize_flux_self_guidance_layers(pipe, args.seg_applied_layers),
            "SEG",
        )

    target_modules = None
    if bridge_requested:
        target_modules = flux_bridge_target_modules(
            pipe,
            normalize_flux_bridge_layers(pipe, args.bridge_layers),
        )

    for step_index, current_timestep in enumerate(timesteps):
        if pipe.interrupt:
            continue

        if s2_enabled:
            noise_uncond, noise_text, noise_pred = clean_predictions()
            if use_bridge_step(step_index, args.s2_start_step, args.s2_end_step):
                dropped_blocks = sample_s2_block_indices(
                    s2_candidates,
                    args.s2_num_drop_blocks,
                    device=latents.device,
                )
                with drop_flux_transformer_blocks(pipe.transformer, dropped_blocks):
                    noise_sub = transformer_forward(latents)
                noise_pred = noise_pred + s2_guidance_scale * (noise_text - noise_sub)
        elif pag_enabled:
            noise_uncond, noise_text, clean_noise_pred = clean_predictions()
            with flux_self_guidance_attn_processors(
                pag_target_modules,
                lambda: FluxPAGAttnProcessor2_0(single_text_seq_len=prompt_embeds.shape[1]),
            ):
                noise_perturb = transformer_forward(latents)
            noise_pred = clean_noise_pred + pag_scale * (noise_text - noise_perturb)
        elif seg_enabled:
            noise_uncond, noise_text, clean_noise_pred = clean_predictions()
            with flux_self_guidance_attn_processors(
                seg_target_modules,
                lambda: FluxSEGAttnProcessor2_0(
                    blur_sigma=args.seg_blur_sigma,
                    single_text_seq_len=prompt_embeds.shape[1],
                ),
            ):
                noise_perturb = transformer_forward(latents)
            noise_pred = clean_noise_pred + seg_scale * (noise_text - noise_perturb)
        elif bridge_requested and use_bridge_step(step_index, args.bridge_start_step, args.bridge_end_step):
            noise_uncond, noise_text, clean_noise_pred = clean_predictions()
            if variant == "direct_sfg":
                direct_sfg_scale_t2i = args.direct_sfg_scale_t2i
                if direct_sfg_scale_t2i is None and bridge_direction == "both":
                    direct_sfg_scale_t2i = args.direct_sfg_scale
                processor_context = flux_bridge_attn_processors(
                    target_modules,
                    sfg_strength_u=0.0,
                    bridge_scale_delta=args.direct_sfg_scale,
                    bridge_scale_delta_t2i=direct_sfg_scale_t2i,
                    bridge_direction=bridge_direction,
                )
                with processor_context:
                    noise_bridge = transformer_forward(latents)
                if use_true_cfg:
                    noise_pred = noise_uncond + float(guidance_scale) * (noise_bridge - noise_uncond)
                else:
                    noise_pred = noise_bridge
            else:
                with flux_bridge_attn_processors(
                    target_modules,
                    sfg_strength_u=args.sfg_strength_u,
                    sfg_strength_u_t2i=args.sfg_strength_u_t2i,
                    bridge_direction=bridge_direction,
                ):
                    noise_bridge = transformer_forward(latents)
                bridge_residual = noise_text - noise_bridge
                bridge_residual = normclip_residual(bridge_residual, noise_text, args.bridge_normclip_tau)
                noise_pred = clean_noise_pred + float(args.bridge_omega) * bridge_residual
        else:
            noise_uncond, noise_text, noise_pred = clean_predictions()
            if cfgzero_requested:
                noise_pred = apply_cfgzero(step_index, noise_uncond, noise_text, noise_pred)

        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, current_timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def setup_dist():
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def valid_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def load_prompts(path: Path, num_images: int):
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= num_images:
                break
    if len(prompts) < num_images:
        raise ValueError(f"prompt file has {len(prompts)} prompts, requested {num_images}")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="FLUX.1-dev prompt-list sampler")
    parser.add_argument("--model-dir", type=Path, default=Path("models/FLUX.1-dev"))
    parser.add_argument("--transformer-dir", type=Path, default=None)
    parser.add_argument("--pipeline-file", type=Path, default=None)
    parser.add_argument("--pipeline-class", type=str, default="FluxPipeline")
    parser.add_argument("--prompt-file", type=Path, default=Path("examples/assets/alignment/hpsv21_1k.txt"))
    parser.add_argument("--workdir", type=Path, default=Path("outputs/flux1dev_baseline_hps1k"))
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--s2-guidance-scale", type=float, default=0.0)
    parser.add_argument("--s2-num-drop-blocks", type=int, default=3)
    parser.add_argument("--s2-block-start", type=int, default=1)
    parser.add_argument("--s2-block-end", type=int, default=0)
    parser.add_argument("--s2-block-family", type=str, default="dual", choices=["dual", "single", "all"])
    parser.add_argument("--s2-start-step", type=int, default=1)
    parser.add_argument("--s2-end-step", type=int, default=0)
    parser.add_argument("--pag-scale", type=float, default=0.0)
    parser.add_argument("--pag-applied-layers", type=str, default="mid")
    parser.add_argument("--seg-scale", type=float, default=0.0)
    parser.add_argument("--seg-blur-sigma", type=float, default=9999999.0)
    parser.add_argument("--seg-applied-layers", type=str, default="mid")
    parser.add_argument(
        "--cfgzero-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "off",
            "zero_init",
            "zeroinit",
            "optimized_scale",
            "opt_scale",
            "cfg_zero_star",
            "cfgzero_star",
            "cfg_zero",
            "cfgzero",
            "full",
        ],
    )
    parser.add_argument("--cfgzero-zero-steps", type=int, default=0)
    parser.add_argument(
        "--sfg-mode",
        type=str,
        default="none",
        choices=["none", "explicit", "explicit_sfg", "sfg", "bridge", "direct_sfg", "direct_sfg_alt"],
    )
    parser.add_argument(
        "--bridge-variant",
        type=str,
        default="explicit",
        choices=["explicit", "explicit_sfg", "direct_sfg", "direct_sfg_alt"],
    )
    parser.add_argument("--direct_sfg-scale", type=float, default=1.0)
    parser.add_argument("--direct_sfg-scale-t2i", type=float, default=None)
    parser.add_argument("--sfg-strength-u", type=float, default=0.2)
    parser.add_argument("--sfg-strength-u-t2i", type=float, default=None)
    parser.add_argument("--bridge-omega", type=float, default=0.0)
    parser.add_argument("--bridge-normclip-tau", type=float, default=999.0)
    parser.add_argument("--bridge-start-step", type=int, default=1)
    parser.add_argument("--bridge-end-step", type=int, default=0)
    parser.add_argument("--bridge-layers", type=str, default="all")
    parser.add_argument(
        "--bridge-direction",
        type=str,
        default="image_to_text",
        choices=["image_to_text", "i2t", "text_to_image", "t2i", "both", "bidirectional", "bidir"],
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    rank, _local_rank, world_size, device = setup_dist()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    np.random.seed(args.seed + rank)
    result_dir = args.workdir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt_file, args.num_images)
    if is_main:
        print(
            "[flux-baseline] "
            f"model_dir={args.model_dir} prompt_file={args.prompt_file} "
            f"transformer_dir={args.transformer_dir} pipeline_file={args.pipeline_file} "
            f"num_images={args.num_images} steps={args.steps} "
            f"guidance_scale={args.guidance_scale} negative_prompt={args.negative_prompt!r} batch_size={args.batch_size} "
            f"size={args.width}x{args.height} "
            f"seed={args.seed} world_size={world_size}",
            flush=True,
        )
        print(
            "[flux-baseline] "
            f"s2_guidance_scale={args.s2_guidance_scale} "
            f"s2_num_drop_blocks={args.s2_num_drop_blocks} "
            f"s2_block_range={args.s2_block_start}:{args.s2_block_end} "
            f"s2_block_family={args.s2_block_family} "
            f"s2_window={args.s2_start_step}to{args.s2_end_step}",
            flush=True,
        )
        print(
            "[flux-baseline] "
            f"pag_scale={args.pag_scale} pag_applied_layers={args.pag_applied_layers} "
            f"seg_scale={args.seg_scale} seg_blur_sigma={args.seg_blur_sigma} "
            f"seg_applied_layers={args.seg_applied_layers}",
            flush=True,
        )
        print(
            "[flux-baseline] "
            f"cfgzero_mode={args.cfgzero_mode} cfgzero_zero_steps={args.cfgzero_zero_steps}",
            flush=True,
        )
        print(
            "[flux-baseline] "
            f"sfg_mode={args.sfg_mode} bridge_variant={args.bridge_variant} "
            f"direct_sfg_scale={args.direct_sfg_scale} direct_sfg_scale_t2i={args.direct_sfg_scale_t2i} "
            f"sfg_strength_u={args.sfg_strength_u} sfg_strength_u_t2i={args.sfg_strength_u_t2i} "
            f"bridge_omega={args.bridge_omega} bridge_normclip_tau={args.bridge_normclip_tau} "
            f"bridge_window={args.bridge_start_step}to{args.bridge_end_step} "
            f"bridge_layers={args.bridge_layers} bridge_direction={args.bridge_direction}",
            flush=True,
        )
        print(f"[flux-baseline] result_dir={result_dir}", flush=True)

    pipe, loaded_pipeline_file = load_flux_pipeline(args, device)
    if is_main:
        print(
            "[flux-baseline] "
            f"pipeline_class={pipe.__class__.__module__}.{pipe.__class__.__name__} "
            f"loaded_pipeline_file={loaded_pipeline_file} "
            f"transformer_guidance_embeds={getattr(pipe.transformer.config, 'guidance_embeds', None)} "
            f"true_cfg_support={pipeline_supports_negative_prompt(pipe)}",
            flush=True,
        )

    generated = 0
    skipped = 0

    def generate_batch(batch_indices):
        batch_prompts = [prompts[idx] for idx in batch_indices]
        generators = [
            torch.Generator(device=device).manual_seed(args.seed + idx)
            for idx in batch_indices
        ]
        bridge_requested = (
            args.bridge_omega != 0.0
            or args.sfg_mode not in {"none", "off", "0"}
            or args.bridge_variant in {"direct_sfg", "direct_sfg_alt"}
        )
        s2_requested = args.s2_guidance_scale != 0.0
        pag_requested = args.pag_scale != 0.0
        seg_requested = args.seg_scale != 0.0
        cfgzero_requested = normalize_cfgzero_mode(args.cfgzero_mode) != "none" or args.cfgzero_zero_steps > 0
        with torch.inference_mode():
            if bridge_requested or s2_requested or pag_requested or seg_requested or cfgzero_requested:
                images = flux_bridge_generate(
                    pipe,
                    batch_prompts,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generators=generators,
                    args=args,
                )
            else:
                pipe_kwargs = {
                    "width": args.width,
                    "height": args.height,
                    "num_inference_steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "generator": generators,
                }
                if pipeline_supports_negative_prompt(pipe):
                    pipe_kwargs["negative_prompt"] = normalize_negative_prompt(args.negative_prompt, len(batch_prompts))
                images = pipe(batch_prompts, **pipe_kwargs).images

        for idx, image in zip(batch_indices, images):
            out_path = result_dir / f"{idx:05d}.png"
            tmp_path = out_path.with_name(f".{out_path.name}.rank{rank}.tmp.png")
            tmp_path.unlink(missing_ok=True)
            image.save(tmp_path)
            tmp_path.replace(out_path)

    pending_indices = []
    for idx in range(rank, args.num_images, world_size):
        out_path = result_dir / f"{idx:05d}.png"
        if not args.overwrite and valid_png(out_path):
            skipped += 1
            local_seen = generated + skipped
            if local_seen == 1 or local_seen % 10 == 0:
                print(
                    f"[rank {rank}] local_seen={local_seen} "
                    f"generated={generated} skipped={skipped} global_idx={idx}",
                    flush=True,
                )
            continue
        if out_path.exists():
            out_path.unlink()

        pending_indices.append(idx)
        if len(pending_indices) >= args.batch_size:
            generate_batch(pending_indices)
            generated += len(pending_indices)
            local_seen = generated + skipped
            print(
                f"[rank {rank}] local_seen={local_seen} "
                f"generated={generated} skipped={skipped} "
                f"global_idx={pending_indices[-1]} batch={len(pending_indices)}",
                flush=True,
            )
            pending_indices = []

    if pending_indices:
        generate_batch(pending_indices)
        generated += len(pending_indices)
        local_seen = generated + skipped
        print(
            f"[rank {rank}] local_seen={local_seen} "
            f"generated={generated} skipped={skipped} "
            f"global_idx={pending_indices[-1]} batch={len(pending_indices)}",
            flush=True,
        )

    barrier()
    if is_main:
        count = len(list(result_dir.glob("*.png")))
        print(f"[flux-baseline] complete png_count={count}/{args.num_images}", flush=True)
    cleanup_dist()


if __name__ == "__main__":
    main()
