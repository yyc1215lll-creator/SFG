from __future__ import annotations

import json
import math
import inspect
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from types import MethodType
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
try:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
except ImportError:
    from diffusers.utils import BaseOutput

    @dataclass
    class Transformer2DModelOutput(BaseOutput):
        sample: Any = None


DEFAULT_SD35M_MODEL = "stabilityai/stable-diffusion-3.5-medium"
DEFAULT_SD3M_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"

SD3_MODEL_ALIASES = {
    "sd35m": DEFAULT_SD35M_MODEL,
    "sd35-medium": DEFAULT_SD35M_MODEL,
    "stable-diffusion-3.5-medium": DEFAULT_SD35M_MODEL,
    "sd3m": DEFAULT_SD3M_MODEL,
    "sd3-medium": DEFAULT_SD3M_MODEL,
    "stable-diffusion-3-medium": DEFAULT_SD3M_MODEL,
    "stable-diffusion-3-medium-diffusers": DEFAULT_SD3M_MODEL,
}

PROMPT_REINJECTION_METHODS = {
    "prompt_reinjection",
    "promptreinjection",
    "reinjection",
    "pr",
}


def normalize_sd35_method(method: Any) -> str:
    return str(method or "cfg").strip().lower().replace("-", "_")


def is_prompt_reinjection_method(method: Any) -> bool:
    return normalize_sd35_method(method) in PROMPT_REINJECTION_METHODS


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


def optimized_cfgzero_scale(noise_text, noise_uncond):
    positive = noise_text.float().flatten(1)
    negative = noise_uncond.float().flatten(1)
    dot_product = torch.sum(positive * negative, dim=1, keepdim=True)
    squared_norm = torch.sum(negative.square(), dim=1, keepdim=True).clamp_min(1e-8)
    scale = dot_product / squared_norm
    view_shape = [noise_text.shape[0]] + [1] * (noise_text.ndim - 1)
    return scale.reshape(view_shape).to(dtype=noise_text.dtype, device=noise_text.device)


def _ssg_candidate_indices(length, max_candidates, device):
    length = int(length)
    max_candidates = int(max_candidates)
    if length <= 1:
        return None
    if max_candidates <= 0 or length <= max_candidates:
        return torch.arange(length, device=device)
    indices = torch.linspace(0, length - 1, steps=max_candidates, device=device)
    return indices.round().long().unique()


def _ssg_dissimilar_pairs(features, ratio, max_pairs):
    num_items = int(features.shape[0])
    if num_items <= 1:
        empty = torch.empty(0, device=features.device, dtype=torch.long)
        empty_scores = torch.empty(0, device=features.device, dtype=torch.float32)
        return empty, empty, empty_scores
    # Official SSG selects the least-similar token pairs from the full pairwise
    # cosine matrix. We keep the same criterion while allowing candidate/pair
    # caps outside this helper for SD3 image-token cost control.
    target_pairs = max(1, int(float(ratio) * num_items))
    if int(max_pairs) > 0:
        target_pairs = min(target_pairs, int(max_pairs))
    target_pairs = min(target_pairs, num_items * (num_items - 1) // 2)
    if target_pairs <= 0:
        empty = torch.empty(0, device=features.device, dtype=torch.long)
        empty_scores = torch.empty(0, device=features.device, dtype=torch.float32)
        return empty, empty, empty_scores

    normalized = F.normalize(features.float(), p=2, dim=-1, eps=1e-6)
    sim_matrix = torch.matmul(normalized, normalized.transpose(0, 1))
    left_all, right_all = torch.triu_indices(num_items, num_items, offset=1, device=features.device)
    sim_values = sim_matrix[left_all, right_all]
    order = torch.topk(sim_values, k=target_pairs, largest=False).indices
    return left_all[order], right_all[order], sim_values[order]


def _ssg_pair_sample(left, right, limit=16):
    if left.numel() == 0:
        return ""
    pairs = torch.stack([left[:limit], right[:limit]], dim=1).detach().cpu().tolist()
    return ";".join(f"{int(a)}-{int(b)}" for a, b in pairs)


def _ssg_write_diag(
    diag_writer,
    *,
    step_index,
    timestep,
    layer_index,
    kind,
    token_space,
    batch_index,
    sequence_length,
    channel_count,
    text_token_count,
    candidate_count,
    left,
    right,
    scores,
):
    if diag_writer is None:
        return
    pair_count = int(left.numel())
    scores_f = scores.detach().float() if scores.numel() else scores
    diag_writer.writerow(
        {
            "step": int(step_index) if step_index is not None else "",
            "timestep": timestep,
            "layer_index": int(layer_index),
            "kind": kind,
            "token_space": token_space,
            "batch_index": int(batch_index),
            "image_token_count": int(sequence_length),
            "text_token_count": int(text_token_count),
            "channel_count": int(channel_count),
            "candidate_count": int(candidate_count),
            "pair_count": pair_count,
            "left_min": int(left.min().item()) if pair_count else "",
            "left_max": int(left.max().item()) if pair_count else "",
            "right_min": int(right.min().item()) if pair_count else "",
            "right_max": int(right.max().item()) if pair_count else "",
            "score_min": float(scores_f.min().item()) if pair_count else "",
            "score_max": float(scores_f.max().item()) if pair_count else "",
            "score_mean": float(scores_f.mean().item()) if pair_count else "",
            "sample_pairs": _ssg_pair_sample(left, right),
        }
    )


def _ssg_spatial_swap(
    hidden_states,
    ratio,
    max_pairs,
    max_candidates,
    *,
    diag_writer=None,
    layer_index=0,
    step_index=None,
    timestep="",
    text_token_count=0,
):
    batch_size, num_tokens, _channels = hidden_states.shape
    candidate_indices = _ssg_candidate_indices(num_tokens, max_candidates, hidden_states.device)
    if candidate_indices is None or candidate_indices.numel() <= 1:
        return hidden_states
    output = hidden_states.clone()
    for batch_index in range(batch_size):
        candidates = hidden_states[batch_index, candidate_indices]
        left_candidates, right_candidates, scores = _ssg_dissimilar_pairs(candidates, ratio, max_pairs)
        if left_candidates.numel() == 0:
            continue
        left = candidate_indices[left_candidates]
        right = candidate_indices[right_candidates]
        tmp = output[batch_index, left].clone()
        output[batch_index, left] = output[batch_index, right]
        output[batch_index, right] = tmp
        _ssg_write_diag(
            diag_writer,
            step_index=step_index,
            timestep=timestep,
            layer_index=layer_index,
            kind="spatial",
            token_space="image_latent_tokens",
            batch_index=batch_index,
            sequence_length=num_tokens,
            channel_count=hidden_states.shape[-1],
            text_token_count=text_token_count,
            candidate_count=candidate_indices.numel(),
            left=left,
            right=right,
            scores=scores,
        )
    return output


def _ssg_channel_swap(
    hidden_states,
    ratio,
    max_pairs,
    max_candidates,
    *,
    diag_writer=None,
    layer_index=0,
    step_index=None,
    timestep="",
    text_token_count=0,
):
    batch_size, _num_tokens, channels = hidden_states.shape
    candidate_indices = _ssg_candidate_indices(channels, max_candidates, hidden_states.device)
    if candidate_indices is None or candidate_indices.numel() <= 1:
        return hidden_states
    output = hidden_states.clone()
    for batch_index in range(batch_size):
        candidates = hidden_states[batch_index].transpose(0, 1)[candidate_indices]
        left_candidates, right_candidates, scores = _ssg_dissimilar_pairs(candidates, ratio, max_pairs)
        if left_candidates.numel() == 0:
            continue
        left = candidate_indices[left_candidates]
        right = candidate_indices[right_candidates]
        tmp = output[batch_index, :, left].clone()
        output[batch_index, :, left] = output[batch_index, :, right]
        output[batch_index, :, right] = tmp
        _ssg_write_diag(
            diag_writer,
            step_index=step_index,
            timestep=timestep,
            layer_index=layer_index,
            kind="channel",
            token_space="channels",
            batch_index=batch_index,
            sequence_length=hidden_states.shape[1],
            channel_count=channels,
            text_token_count=text_token_count,
            candidate_count=candidate_indices.numel(),
            left=left,
            right=right,
            scores=scores,
        )
    return output


def ssg_self_swap(
    hidden_states,
    mode="both",
    ratio=0.10,
    max_pairs=0,
    max_candidates=512,
    layer_index=0,
    *,
    diag_writer=None,
    step_index=None,
    timestep="",
    text_token_count=0,
):
    mode = str(mode or "both").strip().lower().replace("-", "_")
    if float(ratio) <= 0.0:
        return hidden_states
    if mode == "alternate":
        mode = "spatial" if int(layer_index) % 2 == 0 else "channel"
    output = hidden_states
    if mode in {"spatial", "both"}:
        output = _ssg_spatial_swap(
            output,
            ratio,
            max_pairs,
            max_candidates,
            diag_writer=diag_writer,
            layer_index=layer_index,
            step_index=step_index,
            timestep=timestep,
            text_token_count=text_token_count,
        )
    if mode in {"channel", "both"}:
        output = _ssg_channel_swap(
            output,
            ratio,
            max_pairs,
            max_candidates,
            diag_writer=diag_writer,
            layer_index=layer_index,
            step_index=step_index,
            timestep=timestep,
            text_token_count=text_token_count,
        )
    if mode not in {"spatial", "channel", "both"}:
        raise ValueError(f"Unsupported SSG swap mode: {mode}")
    return output


def resolve_sd3_model_key(model_key: str) -> str:
    return SD3_MODEL_ALIASES.get(str(model_key), str(model_key))


def _standardize_tokenwise(x: torch.Tensor, eps: float = 1e-6):
    if x.ndim != 3:
        raise ValueError(f"Prompt Reinjection expects 3D text states, got shape {tuple(x.shape)}.")
    mean = x.mean(dim=-1, keepdim=True)
    std = torch.clamp(x.std(dim=-1, keepdim=True), min=eps)
    return (x - mean) / (std + eps), mean, std


def _prepare_prompt_reinjection_weights(weights, *, device, dtype, expected_count: int) -> torch.Tensor:
    if torch.is_tensor(weights):
        prepared = weights
    elif isinstance(weights, (list, tuple)):
        prepared = torch.tensor([float(value) for value in weights])
    elif isinstance(weights, str):
        parts = [part for part in re.split(r"[,;\s]+", weights.strip()) if part]
        if len(parts) > 1:
            prepared = torch.tensor([float(value) for value in parts])
        elif len(parts) == 1:
            prepared = torch.tensor([float(parts[0])])
        else:
            prepared = torch.tensor([0.0])
    else:
        prepared = torch.tensor([float(weights)])
    prepared = prepared.to(device=device, dtype=dtype).flatten()
    if prepared.numel() == 1 and expected_count > 1:
        prepared = prepared.repeat(expected_count)
    elif prepared.numel() != expected_count:
        raise ValueError(
            "prompt_reinjection_weight must be a scalar or have the same length as target layers "
            f"({expected_count})."
        )
    return prepared


def _torch_load_cpu_compat(path: str):
    load_kwargs = {"map_location": "cpu", "mmap": True}
    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("mmap", None)
        return torch.load(path, **load_kwargs)


def _load_prompt_reinjection_procrustes(path: str):
    path = str(Path(path).expanduser())
    data = _torch_load_cpu_compat(path)
    target_layers = None
    meta = data if isinstance(data, dict) else None
    if isinstance(data, dict):
        if "rotation_matrices" in data:
            rotations = data["rotation_matrices"]
        elif "R" in data:
            rotations = data["R"]
        else:
            raise KeyError(f"PromptReinjection Procrustes file missing rotation_matrices/R key: {path}")
        target_layers = data.get("target_layers")
    else:
        rotations = data
    if not torch.is_tensor(rotations):
        rotations = torch.tensor(rotations)
    if rotations.dim() == 2:
        rotations = rotations.unsqueeze(0)
    if rotations.dim() != 3:
        raise ValueError(
            "prompt_reinjection_procrustes_path must contain rotations with shape (N,D,D) or (D,D); "
            f"got {tuple(rotations.shape)} from {path}."
        )
    return rotations, target_layers, meta


def _select_prompt_reinjection_rotations(
    rotations: torch.Tensor,
    saved_target_layers: Any,
    residual_target_layers: List[int],
) -> torch.Tensor:
    if saved_target_layers is None:
        if rotations.shape[0] != len(residual_target_layers):
            raise ValueError(
                "PromptReinjection rotation count must match target layer count when the Procrustes file "
                f"does not store target_layers: rotations={rotations.shape[0]} targets={len(residual_target_layers)}."
            )
        return rotations

    saved_layers = [int(layer) for layer in saved_target_layers]
    missing = [int(layer) for layer in residual_target_layers if int(layer) not in saved_layers]
    if missing:
        raise ValueError(
            "prompt_reinjection_target_layers must be a subset of target_layers in the Procrustes file. "
            f"Missing: {missing}"
        )
    indices = torch.tensor([saved_layers.index(int(layer)) for layer in residual_target_layers], dtype=torch.long)
    return rotations.index_select(0, indices)


def _prepare_prompt_reinjection_rotations(
    cache_owner: Any,
    rotations: Optional[torch.Tensor],
    *,
    device,
    dtype,
    expected_count: int,
    feature_dim: int,
) -> Optional[torch.Tensor]:
    if rotations is None:
        return None
    if rotations.dim() == 2:
        rotations = rotations.unsqueeze(0)
    if rotations.dim() != 3:
        raise ValueError("PromptReinjection rotations must have shape (N,D,D) or (D,D).")
    if rotations.shape[0] != expected_count:
        raise ValueError(
            f"PromptReinjection rotation count mismatch: rotations={rotations.shape[0]} targets={expected_count}."
        )
    if rotations.shape[-1] != feature_dim or rotations.shape[-2] != feature_dim:
        raise ValueError(
            "PromptReinjection rotation feature dimension mismatch: "
            f"rotation={tuple(rotations.shape[-2:])} feature_dim={feature_dim}."
        )

    cache_key = (id(rotations), tuple(rotations.shape), str(device), dtype)
    cached = getattr(cache_owner, "_sfg_prompt_reinjection_rotation_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rotations"]

    prepared = rotations.to(device=device, dtype=dtype, non_blocking=True)
    try:
        setattr(cache_owner, "_sfg_prompt_reinjection_rotation_cache", {"key": cache_key, "rotations": prepared})
    except Exception:
        pass
    return prepared


def _apply_prompt_reinjection_residual(
    target: torch.Tensor,
    origin: torch.Tensor,
    weight: torch.Tensor,
    *,
    use_anchoring: bool,
    stop_grad: bool,
    rotation_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target_base = target.detach() if stop_grad else target
    origin_base = origin.detach() if stop_grad else origin
    if not use_anchoring:
        if rotation_matrix is not None:
            origin_base = torch.matmul(origin_base, rotation_matrix)
        return target_base + weight * origin_base

    target_norm, target_mean, target_std = _standardize_tokenwise(target_base)
    origin_norm, _, _ = _standardize_tokenwise(origin_base)
    if rotation_matrix is not None:
        origin_norm = torch.matmul(origin_norm, rotation_matrix)
    if bool((weight >= 0).item()):
        mixed = target_norm + weight * origin_norm
    else:
        mixed = target_norm * (1 - weight)
    mixed = torch.nn.functional.layer_norm(mixed, normalized_shape=(mixed.shape[-1],), eps=1e-6)
    return mixed * target_std + target_mean


def _bool_arg(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _prompt_reinjection_layer_token(token: Any, *, num_layers: int) -> int:
    text = str(token).strip().lower().replace("blocks.", "").replace("block.", "").replace("layers.", "")
    if text == "last":
        return int(num_layers) - 1
    return int(text)


def _normalize_prompt_reinjection_target_layers(target_layers: Any, *, origin_layer: int, num_layers: int) -> List[int]:
    origin_layer = int(origin_layer)
    num_layers = int(num_layers)
    if origin_layer < 0 or origin_layer >= num_layers:
        raise ValueError(f"prompt_reinjection_origin_layer must be in [0, {num_layers - 1}], got {origin_layer}.")

    use_default = target_layers is None
    if isinstance(target_layers, str):
        normalized = target_layers.strip().lower()
        use_default = normalized in {"", "default", "auto", "paper", "origin+1-last"}
    if use_default:
        values = list(range(origin_layer + 1, num_layers))
    elif torch.is_tensor(target_layers):
        values = [int(value) for value in target_layers.detach().cpu().flatten().tolist()]
    elif isinstance(target_layers, (list, tuple)):
        values = [int(value) for value in target_layers]
    else:
        values = []
        for part in re.split(r"[,;\s]+", str(target_layers).strip()):
            if not part:
                continue
            if ":" in part:
                start_text, end_text = part.split(":", 1)
                start = _prompt_reinjection_layer_token(start_text, num_layers=num_layers)
                end = _prompt_reinjection_layer_token(end_text, num_layers=num_layers)
                if start > end:
                    raise ValueError(f"Invalid prompt_reinjection_target_layers range: {part}")
                values.extend(range(start, end + 1))
            elif "-" in part:
                start_text, end_text = part.split("-", 1)
                start = _prompt_reinjection_layer_token(start_text, num_layers=num_layers)
                end = _prompt_reinjection_layer_token(end_text, num_layers=num_layers)
                if start > end:
                    raise ValueError(f"Invalid prompt_reinjection_target_layers range: {part}")
                values.extend(range(start, end + 1))
            else:
                values.append(_prompt_reinjection_layer_token(part, num_layers=num_layers))

    deduped = []
    seen = set()
    for layer in values:
        layer = int(layer)
        if layer in seen:
            continue
        if layer < 0 or layer >= num_layers:
            raise ValueError(
                f"prompt_reinjection_target_layers contains {layer}, but transformer has layers 0-{num_layers - 1}."
            )
        if layer <= origin_layer:
            raise ValueError(
                "prompt_reinjection_target_layers must be after prompt_reinjection_origin_layer "
                f"({origin_layer}); got {layer}."
            )
        seen.add(layer)
        deduped.append(layer)
    if not deduped:
        raise ValueError("prompt_reinjection_target_layers resolved to an empty layer list.")
    return deduped


def _sd3_block_accepts_kwarg(block: Any, kwarg_name: str) -> bool:
    cache_name = f"_sfg_accepts_{kwarg_name}"
    cached = getattr(block, cache_name, None)
    if cached is not None:
        return bool(cached)
    try:
        signature = inspect.signature(block.forward)
    except (TypeError, ValueError, AttributeError):
        accepts = False
    else:
        accepts = kwarg_name in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
    try:
        setattr(block, cache_name, bool(accepts))
    except Exception:
        pass
    return bool(accepts)


def _prompt_reinjection_transformer_forward(
    transformer,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    pooled_projections: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    block_controlnet_hidden_states: Optional[List[torch.Tensor]] = None,
    joint_attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,
    skip_layers: Optional[List[int]] = None,
    output_text_inputs: bool = False,
    *,
    residual_origin_layer: int,
    residual_target_layers: List[int],
    residual_weights: Any,
    residual_use_anchoring: bool,
    residual_stop_grad: bool,
    residual_rotation_matrices: Optional[torch.Tensor] = None,
):
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()

    use_residual = encoder_hidden_states is not None and residual_target_layers
    target_layers_set = set(int(layer) for layer in residual_target_layers) if use_residual else set()
    residual_target_to_idx = {
        int(layer): idx for idx, layer in enumerate(residual_target_layers)
    }

    residual_weights_tensor = None
    if use_residual:
        residual_weights_tensor = _prepare_prompt_reinjection_weights(
            residual_weights,
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
            expected_count=len(residual_target_layers),
        )
    else:
        residual_weights_tensor = None
    residual_rotations_tensor = None

    height, width = hidden_states.shape[-2:]
    hidden_states = transformer.pos_embed(hidden_states)
    temb = transformer.time_text_embed(timestep, pooled_projections)

    if encoder_hidden_states is not None:
        encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)
        if use_residual:
            residual_rotations_tensor = _prepare_prompt_reinjection_rotations(
                transformer,
                residual_rotation_matrices,
                device=encoder_hidden_states.device,
                dtype=encoder_hidden_states.dtype,
                expected_count=len(residual_target_layers),
                feature_dim=encoder_hidden_states.shape[-1],
            )

    if joint_attention_kwargs and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states, ip_temb = transformer.image_proj(ip_adapter_image_embeds, timestep)
        joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

    saved_origin_state = None
    txt_input_states_list = [] if output_text_inputs else None

    for index_block, block in enumerate(transformer.transformer_blocks):
        is_skip = skip_layers is not None and index_block in skip_layers

        if output_text_inputs and not is_skip:
            txt_input_states_list.append(encoder_hidden_states)

        if use_residual and index_block == int(residual_origin_layer):
            saved_origin_state = encoder_hidden_states

        if use_residual and index_block in target_layers_set:
            if saved_origin_state is None:
                raise RuntimeError(
                    f"PromptReinjection origin layer {residual_origin_layer} was not cached before target layer "
                    f"{index_block}."
                )
            if saved_origin_state.shape != encoder_hidden_states.shape:
                raise ValueError(
                    "PromptReinjection residual shape mismatch: "
                    f"origin={tuple(saved_origin_state.shape)} target={tuple(encoder_hidden_states.shape)}."
                )
            local_index = residual_target_to_idx[index_block]
            rotation_matrix = (
                residual_rotations_tensor[local_index]
                if residual_rotations_tensor is not None
                else None
            )
            encoder_hidden_states = _apply_prompt_reinjection_residual(
                encoder_hidden_states,
                saved_origin_state,
                residual_weights_tensor[local_index],
                use_anchoring=bool(residual_use_anchoring),
                stop_grad=bool(residual_stop_grad),
                rotation_matrix=rotation_matrix,
            )

        if not is_skip:
            block_accepts_joint_attention_kwargs = _sd3_block_accepts_kwarg(block, "joint_attention_kwargs")
            if torch.is_grad_enabled() and transformer.gradient_checkpointing:
                block_args = [block, hidden_states, encoder_hidden_states, temb]
                if block_accepts_joint_attention_kwargs:
                    block_args.append(joint_attention_kwargs)
                encoder_hidden_states, hidden_states = transformer._gradient_checkpointing_func(*block_args)
            else:
                block_kwargs = {
                    "hidden_states": hidden_states,
                    "encoder_hidden_states": encoder_hidden_states,
                    "temb": temb,
                }
                if block_accepts_joint_attention_kwargs:
                    block_kwargs["joint_attention_kwargs"] = joint_attention_kwargs
                encoder_hidden_states, hidden_states = block(**block_kwargs)

        if block_controlnet_hidden_states is not None and not block.context_pre_only:
            interval_control = len(transformer.transformer_blocks) / len(block_controlnet_hidden_states)
            hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

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

    if not return_dict:
        if output_text_inputs:
            return {"sample": output, "txt_input_states": txt_input_states_list}
        return (output,)
    return Transformer2DModelOutput(sample=output)


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


class SEGJointAttnProcessor2_0:
    """Joint-attention SEG processor for SD3-like transformers."""

    def __init__(self, blur_sigma=10000.0, do_cfg=True, inf_blur_threshold=9999.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SEGJointAttnProcessor2_0 requires PyTorch 2.0 scaled dot-product attention.")
        self.blur_sigma = float(blur_sigma)
        self.do_cfg = bool(do_cfg)
        self.inf_blur = self.blur_sigma >= float(inf_blur_threshold) or self.blur_sigma <= 0.0

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

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        image_seq_len = query.shape[1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        side = math.isqrt(image_seq_len)
        if side * side == image_seq_len:
            if self.do_cfg:
                query_uncond, query_org, query_ptb = query.chunk(3)
                query_ptb = self._blur_query(query_ptb, batch_size // 3, attn.heads, head_dim, side)
                query = torch.cat((query_uncond, query_org, query_ptb), dim=0)
            else:
                query_org, query_ptb = query.chunk(2)
                query_ptb = self._blur_query(query_ptb, batch_size // 2, attn.heads, head_dim, side)
                query = torch.cat((query_org, query_ptb), dim=0)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
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

    def _blur_query(self, query_ptb, batch_size, heads, head_dim, side):
        query_ptb = query_ptb.permute(0, 1, 3, 2).reshape(batch_size, heads * head_dim, side, side)
        if self.inf_blur:
            query_ptb = query_ptb.mean(dim=(-2, -1), keepdim=True).expand_as(query_ptb)
        else:
            kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
            query_ptb = gaussian_blur_2d(query_ptb, kernel_size, self.blur_sigma)
        return query_ptb.reshape(batch_size, heads, head_dim, side * side).permute(0, 1, 3, 2)


class SFGJointAttnProcessor2_0:
    """Scales selected cross-modal value contributions in SD3 joint attention.

    Normal sampling evaluates image and text query rows separately with their
    corresponding value slice already scaled.  The two query slices together
    contain exactly the rows of one ordinary joint-attention evaluation.  This
    is mathematically equivalent to adding an isolated bridge contribution to
    the ordinary attention output, but avoids that extra SDPA work.  The legacy
    decomposition remains available when bridge statistics are requested.
    """

    image_query_chunk_size = 128

    def __init__(
        self,
        sfg_strength_u=0.2,
        layer_name="",
        stats=None,
        clean_prefix_batch_size=0,
        bridge_scale_delta=None,
        sfg_strength_u_t2i=None,
        bridge_scale_delta_t2i=None,
        bridge_direction="image_to_text",
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SFGJointAttnProcessor2_0 requires PyTorch 2.0 scaled dot-product attention.")
        sfg_strength_u = float(sfg_strength_u)
        if sfg_strength_u < -1.0 or sfg_strength_u > 1.0:
            raise ValueError("sfg_strength_u must be in [-1, 1].")
        if sfg_strength_u_t2i is None or sfg_strength_u_t2i == "":
            sfg_strength_u_t2i = sfg_strength_u
        sfg_strength_u_t2i = float(sfg_strength_u_t2i)
        if sfg_strength_u_t2i < -1.0 or sfg_strength_u_t2i > 1.0:
            raise ValueError("sfg_strength_u_t2i must be in [-1, 1].")
        self.sfg_strength_u = sfg_strength_u
        self.layer_name = str(layer_name)
        self.stats = stats
        self.clean_prefix_batch_size = max(0, int(clean_prefix_batch_size))
        self.bridge_direction = self._normalize_bridge_direction(bridge_direction)
        primary_delta = -sfg_strength_u if bridge_scale_delta is None else float(bridge_scale_delta)
        t2i_delta = -sfg_strength_u_t2i if bridge_scale_delta_t2i is None else float(bridge_scale_delta_t2i)
        if self.bridge_direction == "image_to_text":
            self.bridge_scale_delta_i2t = primary_delta
            self.bridge_scale_delta_t2i = 0.0
        elif self.bridge_direction == "text_to_image":
            self.bridge_scale_delta_i2t = 0.0
            self.bridge_scale_delta_t2i = primary_delta
        else:
            self.bridge_scale_delta_i2t = primary_delta
            self.bridge_scale_delta_t2i = t2i_delta

    @staticmethod
    def _normalize_bridge_direction(direction):
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

        bridge_specs = []
        weak_start = batch_size
        if encoder_hidden_states is not None:
            weak_start = min(self.clean_prefix_batch_size, batch_size)
            if weak_start < batch_size:
                total_seq_len = key.shape[2]
                if self.bridge_scale_delta_i2t != 0.0:
                    bridge_specs.append(
                        (
                            "image_to_text",
                            0,
                            image_seq_len,
                            image_seq_len,
                            total_seq_len,
                            self.bridge_scale_delta_i2t,
                        )
                    )
                if self.bridge_scale_delta_t2i != 0.0:
                    bridge_specs.append(
                        (
                            "text_to_image",
                            image_seq_len,
                            total_seq_len,
                            0,
                            image_seq_len,
                            self.bridge_scale_delta_t2i,
                        )
                    )

        if bridge_specs and self.stats is None:
            hidden_states = self._direct_split_bridge_attention(
                query,
                key,
                value,
                image_seq_len=image_seq_len,
                weak_start=weak_start,
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            if bridge_specs:
                updated_hidden_states = hidden_states.clone()
                for direction_name, target_start, target_end, source_start, source_end, scale_delta in bridge_specs:
                    source_value = value[weak_start:, :, source_start:source_end]
                    bridge_contrib, cross_mass = self._chunked_value_slice_contribution(
                        query[weak_start:, :, target_start:target_end],
                        key[weak_start:],
                        source_value,
                        source_start,
                        source_end,
                        head_dim,
                    )
                    updated_hidden_states[weak_start:, :, target_start:target_end] = (
                        updated_hidden_states[weak_start:, :, target_start:target_end]
                        + scale_delta * bridge_contrib
                    )
                    self.stats.append(
                        {
                            "layer": self.layer_name,
                            "bridge_direction": direction_name,
                            "cross_mass": float(cross_mass.detach().float().cpu().item()),
                        }
                    )
                hidden_states = updated_hidden_states

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
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
    def _scaled_value_slice_for_suffix(value, weak_start, source_start, source_end, scale_delta):
        if scale_delta == 0.0 or weak_start >= value.shape[0]:
            return value
        scaled_value = value.clone()
        scaled_value[weak_start:, :, source_start:source_end].mul_(1.0 + float(scale_delta))
        return scaled_value

    def _direct_split_bridge_attention(self, query, key, value, *, image_seq_len, weak_start):
        """Evaluate the modified attention directly using one full set of query rows.

        Image queries use values whose text slice carries the SFGS multiplier;
        text queries use values whose image slice carries the SFGX multiplier.
        Keys are unchanged, so both calls retain the original full-key softmax.
        """

        total_seq_len = key.shape[2]
        image_query_value = self._scaled_value_slice_for_suffix(
            value,
            weak_start,
            image_seq_len,
            total_seq_len,
            self.bridge_scale_delta_i2t,
        )
        image_output = F.scaled_dot_product_attention(
            query[:, :, :image_seq_len],
            key,
            image_query_value,
            dropout_p=0.0,
            is_causal=False,
        )
        del image_query_value

        text_query_value = self._scaled_value_slice_for_suffix(
            value,
            weak_start,
            0,
            image_seq_len,
            self.bridge_scale_delta_t2i,
        )
        text_output = F.scaled_dot_product_attention(
            query[:, :, image_seq_len:],
            key,
            text_query_value,
            dropout_p=0.0,
            is_causal=False,
        )
        return torch.cat((image_output, text_output), dim=2)

    def _sdpa_value_slice_contribution(self, query_target, key_all, value_all, source_start, source_end):
        value_source_only = torch.zeros_like(value_all)
        value_source_only[:, :, source_start:source_end] = value_all[:, :, source_start:source_end]
        return F.scaled_dot_product_attention(
            query_target,
            key_all,
            value_source_only,
            dropout_p=0.0,
            is_causal=False,
        )

    def _chunked_value_slice_contribution(self, query_target, key_all, value_source, source_start, source_end, head_dim):
        scale = head_dim ** -0.5
        chunks = []
        mass_sum = None
        mass_count = 0
        key_all_t = key_all.float().transpose(-2, -1)

        for start in range(0, query_target.shape[2], self.image_query_chunk_size):
            end = min(start + self.image_query_chunk_size, query_target.shape[2])
            q = query_target[:, :, start:end]
            scores = torch.matmul(q.float(), key_all_t) * scale
            weights = torch.softmax(scores, dim=-1).to(q.dtype)
            source_weights = weights[..., source_start:source_end]
            chunks.append(torch.matmul(source_weights, value_source))
            source_mass = source_weights.float().sum(dim=-1)
            batch_mass_sum = source_mass.sum()
            mass_sum = batch_mass_sum if mass_sum is None else mass_sum + batch_mass_sum
            mass_count += source_mass.numel()

        mean_mass = mass_sum / max(mass_count, 1)
        return torch.cat(chunks, dim=2), mean_mass


class TACAJointAttnProcessor2_0:
    """TACA-style temperature scaling for SD3 joint attention.

    SD3 concatenates image tokens first and text tokens second inside each joint
    attention block. TACA multiplies the image-query/text-key logits before the
    softmax, strengthening text grounding without adding an extra model branch.
    """

    image_query_chunk_size = 128

    def __init__(self, taca_scale=1.2, layer_name=""):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("TACAJointAttnProcessor2_0 requires PyTorch 2.0 scaled dot-product attention.")
        taca_scale = float(taca_scale)
        if taca_scale <= 0.0:
            raise ValueError("taca_scale must be > 0.")
        self.taca_scale = taca_scale
        self.layer_name = str(layer_name)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if attention_mask is not None:
            raise ValueError("TACA SD3 processor does not currently support attention_mask.")

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

        if encoder_hidden_states is None or abs(self.taca_scale - 1.0) <= 1e-8:
            attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            text_start = image_seq_len
            text_end = key.shape[2]
            image_out = self._chunked_taca_image_attention(
                query[:, :, :image_seq_len],
                key,
                value,
                text_start=text_start,
                text_end=text_end,
                head_dim=head_dim,
            )
            text_out = F.scaled_dot_product_attention(
                query[:, :, image_seq_len:],
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            attended = torch.cat([image_out, text_out], dim=2)

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

    def _chunked_taca_image_attention(self, query_image, key_all, value_all, text_start, text_end, head_dim):
        scale = head_dim ** -0.5
        chunks = []
        key_all_t = key_all.float().transpose(-2, -1)
        for start in range(0, query_image.shape[2], self.image_query_chunk_size):
            end = min(start + self.image_query_chunk_size, query_image.shape[2])
            q = query_image[:, :, start:end]
            scores = torch.matmul(q.float(), key_all_t) * scale
            scores[..., text_start:text_end] = scores[..., text_start:text_end] * float(self.taca_scale)
            weights = torch.softmax(scores, dim=-1).to(q.dtype)
            chunks.append(torch.matmul(weights, value_all))
        return torch.cat(chunks, dim=2)


class PAGJointAttnProcessor2_0Compat:
    """Memory-efficient SD3 no-CFG PAG processor copied from the SD3.5 SG/PAG path."""

    image_query_chunk_size = 256

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

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        identity_block_size = hidden_states.shape[1]
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)
        encoder_hidden_states_org, encoder_hidden_states_ptb = encoder_hidden_states.chunk(2)

        hidden_states_org, encoder_hidden_states_org = self._normal_joint_attention(
            attn, hidden_states_org, encoder_hidden_states_org, residual.shape[1]
        )
        hidden_states_ptb, encoder_hidden_states_ptb = self._perturbed_joint_attention(
            attn,
            hidden_states_ptb,
            encoder_hidden_states_ptb,
            residual.shape[1],
            identity_block_size,
        )

        batch_size = hidden_states_org.shape[0]
        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_org = encoder_hidden_states_org.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
            encoder_hidden_states_ptb = encoder_hidden_states_ptb.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])
        encoder_hidden_states = torch.cat([encoder_hidden_states_org, encoder_hidden_states_ptb])
        return hidden_states, encoder_hidden_states

    def _project_joint(self, attn, hidden_states, encoder_hidden_states):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

        batch_size = hidden_states.shape[0]
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        return query, key, value, head_dim

    def _split_and_project_out(self, attn, hidden_states, residual_seq_len):
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :residual_seq_len],
            hidden_states[:, residual_seq_len:],
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

    def _normal_joint_attention(self, attn, hidden_states, encoder_hidden_states, residual_seq_len):
        query, key, value, head_dim = self._project_joint(attn, hidden_states, encoder_hidden_states)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            hidden_states.shape[0], -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        return self._split_and_project_out(attn, hidden_states, residual_seq_len)

    def _perturbed_joint_attention(
        self, attn, hidden_states, encoder_hidden_states, residual_seq_len, identity_block_size
    ):
        query, key, value, head_dim = self._project_joint(attn, hidden_states, encoder_hidden_states)

        query_img = query[:, :, :identity_block_size]
        query_txt = query[:, :, identity_block_size:]
        key_img = key[:, :, :identity_block_size]
        key_txt = key[:, :, identity_block_size:]
        value_img = value[:, :, :identity_block_size]
        value_txt = value[:, :, identity_block_size:]

        text_out = F.scaled_dot_product_attention(query_txt, key, value, dropout_p=0.0, is_causal=False)
        image_out = self._chunked_identity_mask_attention(
            query_img, key_img, value_img, key_txt, value_txt, head_dim
        )

        hidden_states = torch.cat([image_out, text_out], dim=2)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            hidden_states.shape[0], -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        return self._split_and_project_out(attn, hidden_states, residual_seq_len)

    def _chunked_identity_mask_attention(self, query_img, key_img, value_img, key_txt, value_txt, head_dim):
        scale = head_dim ** -0.5
        chunks = []
        chunk_size = self.image_query_chunk_size

        for start in range(0, query_img.shape[2], chunk_size):
            end = min(start + chunk_size, query_img.shape[2])
            q = query_img[:, :, start:end]
            k_diag = key_img[:, :, start:end]
            v_diag = value_img[:, :, start:end]

            diag_scores = (q.float() * k_diag.float()).sum(dim=-1, keepdim=True) * scale
            text_scores = torch.matmul(q.float(), key_txt.float().transpose(-2, -1)) * scale
            scores = torch.cat([diag_scores, text_scores], dim=-1)
            weights = torch.softmax(scores, dim=-1).to(q.dtype)

            diag_out = weights[..., :1] * v_diag
            text_out = torch.matmul(weights[..., 1:], value_txt)
            chunks.append(diag_out + text_out)

        return torch.cat(chunks, dim=2)


def _patch_accelerate_strict_kwarg() -> None:
    try:
        import inspect
        import accelerate
    except Exception:
        return
    fn = getattr(accelerate, "load_checkpoint_and_dispatch", None)
    if fn is None or "strict" in inspect.signature(fn).parameters:
        return

    def wrapper(*args, **kwargs):
        kwargs.pop("strict", None)
        return fn(*args, **kwargs)

    accelerate.load_checkpoint_and_dispatch = wrapper


def _patch_sd3_pag_joint_processor_kwarg() -> None:
    try:
        import inspect
        from diffusers.models import attention_processor
    except Exception:
        return

    cls = getattr(attention_processor, "PAGJointAttnProcessor2_0", None)
    if cls is None or getattr(cls, "_sfg_accepts_attention_mask", False):
        return
    if "attention_mask" in inspect.signature(cls.__call__).parameters:
        cls._sfg_accepts_attention_mask = True
        return

    original_call = cls.__call__

    def wrapper(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        return original_call(self, attn, hidden_states, encoder_hidden_states)

    cls.__call__ = wrapper
    cls._sfg_accepts_attention_mask = True


def _positive_prompts(prompt: Any) -> List[str]:
    if isinstance(prompt, (list, tuple)) and len(prompt) == 2:
        positive = prompt[1]
    else:
        positive = prompt
    if isinstance(positive, str):
        return [positive]
    return list(positive)


def _negative_prompts(prompt: Any, batch_size: int) -> Optional[List[str]]:
    if isinstance(prompt, (list, tuple)) and len(prompt) == 2:
        negative = prompt[0]
        if isinstance(negative, str):
            return [negative] * batch_size
        return list(negative)
    return [""] * batch_size


class SD35M:
    def __init__(
        self,
        solver_config=None,
        device: str = "cuda",
        method: str = "cfg",
        model_key: str = DEFAULT_SD35M_MODEL,
        dtype: torch.dtype = torch.bfloat16,
        bridge_causal_adapter: str = "",
        bridge_causal_prompt_source: str = "",
        sd3_lora_adapter: str = "",
        sd3_diffusers_lora: str = "",
        sd3_diffusers_lora_weight_name: str = "",
    ):
        from diffusers import StableDiffusion3PAGPipeline

        model_key = resolve_sd3_model_key(model_key)
        _patch_accelerate_strict_kwarg()
        _patch_sd3_pag_joint_processor_kwarg()
        self.num_sampling = int(getattr(solver_config, "num_sampling", 28) if solver_config is not None else 28)
        self.device = device
        self.method = normalize_sd35_method(method)
        self.pipe = StableDiffusion3PAGPipeline.from_pretrained(
            model_key,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self._pag_layers = "blocks.13"
        self.bridge_causal_adapter = None
        if bridge_causal_adapter:
            self._load_bridge_causal_adapter(bridge_causal_adapter, prompt_source=bridge_causal_prompt_source)
        self.sd3_lora_adapter = None
        if sd3_lora_adapter:
            self._load_sd3_lora_adapter(sd3_lora_adapter)
        self.sd3_diffusers_lora = str(sd3_diffusers_lora or "")
        self.sd3_diffusers_lora_weight_name = str(sd3_diffusers_lora_weight_name or "")
        self.sd3_diffusers_lora_loaded = False
        self.sd3_diffusers_lora_enabled = False
        self._prompt_reinjection_procrustes_cache = {}
        if self.sd3_diffusers_lora:
            self._load_sd3_diffusers_lora()

    def _load_sd3_lora_adapter(self, adapter_path: str) -> None:
        from sd3_lora_adapter import SD3LoRAAdapter

        transformer_dtype = next(self.pipe.transformer.parameters()).dtype
        adapter = SD3LoRAAdapter.load_adapter(
            self.pipe.transformer,
            adapter_path,
            device=self.device,
            dtype=transformer_dtype,
        )
        adapter.set_scale(1.0)
        adapter.eval()
        self.sd3_lora_adapter = adapter

    def _load_sd3_diffusers_lora(self) -> None:
        if not self.sd3_diffusers_lora or self.sd3_diffusers_lora_loaded:
            return
        kwargs = {}
        if self.sd3_diffusers_lora_weight_name:
            kwargs["weight_name"] = self.sd3_diffusers_lora_weight_name
        self.pipe.load_lora_weights(self.sd3_diffusers_lora, **kwargs)
        self.sd3_diffusers_lora_loaded = True
        self.sd3_diffusers_lora_enabled = True

    def _set_sd3_diffusers_lora_enabled(self, enabled: bool) -> None:
        if not self.sd3_diffusers_lora:
            return
        if not self.sd3_diffusers_lora_loaded:
            self._load_sd3_diffusers_lora()
        enabled = bool(enabled)
        if enabled == self.sd3_diffusers_lora_enabled:
            return
        if enabled:
            self.pipe.enable_lora()
        else:
            self.pipe.disable_lora()
        self.sd3_diffusers_lora_enabled = enabled

    @contextmanager
    def _prompt_reinjection_transformer(
        self,
        *,
        origin_layer: int,
        target_layers: Any,
        weight: Any,
        use_anchoring: bool,
        stop_grad: bool,
        procrustes_path: str = "",
    ):
        transformer = self.pipe.transformer
        num_layers = len(transformer.transformer_blocks)
        resolved_targets = _normalize_prompt_reinjection_target_layers(
            target_layers,
            origin_layer=int(origin_layer),
            num_layers=num_layers,
        )
        residual_rotation_matrices = None
        if procrustes_path:
            procrustes_path = str(Path(procrustes_path).expanduser())
            cached = self._prompt_reinjection_procrustes_cache.get(procrustes_path)
            if cached is None:
                cached = _load_prompt_reinjection_procrustes(procrustes_path)
                self._prompt_reinjection_procrustes_cache[procrustes_path] = cached
            rotations, saved_target_layers, _meta = cached
            residual_rotation_matrices = _select_prompt_reinjection_rotations(
                rotations,
                saved_target_layers,
                resolved_targets,
            )
        original_forward = transformer.forward

        def patched_forward(
            module_self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            pooled_projections: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            block_controlnet_hidden_states: Optional[List[torch.Tensor]] = None,
            joint_attention_kwargs: Optional[dict] = None,
            return_dict: bool = True,
            skip_layers: Optional[List[int]] = None,
            output_text_inputs: bool = False,
            **_,
        ):
            return _prompt_reinjection_transformer_forward(
                module_self,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                block_controlnet_hidden_states=block_controlnet_hidden_states,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
                skip_layers=skip_layers,
                output_text_inputs=output_text_inputs,
                residual_origin_layer=int(origin_layer),
                residual_target_layers=resolved_targets,
                residual_weights=weight,
                residual_use_anchoring=bool(use_anchoring),
                residual_stop_grad=bool(stop_grad),
                residual_rotation_matrices=residual_rotation_matrices,
            )

        transformer.forward = MethodType(patched_forward, transformer)
        try:
            yield resolved_targets
        finally:
            transformer.forward = original_forward

    @torch.no_grad()
    def _sample_prompt_reinjection(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        *,
        origin_layer: int = 1,
        target_layers: Any = "2-23",
        weight: Any = 0.025,
        use_anchoring: bool = True,
        stop_grad: bool = True,
        procrustes_path: str = "",
        generator=None,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0
        use_cfg_reference = float(cfg_guidance) > 1.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=use_cfg_reference,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if use_cfg_reference:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, _num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        latents = pipe.prepare_latents(
            len(prompts),
            pipe.transformer.config.in_channels,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        with self._prompt_reinjection_transformer(
            origin_layer=int(origin_layer),
            target_layers=target_layers,
            weight=weight,
            use_anchoring=bool(use_anchoring),
            stop_grad=bool(stop_grad),
            procrustes_path=procrustes_path,
        ):
            for timestep in timesteps:
                if use_cfg_reference:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    timestep_input = timestep.expand(latent_model_input.shape[0])
                    noise_pred_all = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_input,
                        encoder_hidden_states=cfg_prompt_embeds,
                        pooled_projections=cfg_pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_uncond, noise_text = noise_pred_all.chunk(2)
                    noise_pred = noise_uncond + float(cfg_guidance) * (noise_text - noise_uncond)
                else:
                    noise_pred = pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep.expand(latents.shape[0]),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_lora_window(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        *,
        active_steps: int,
        lora_scale: float = 1.0,
        inactive_scale: float = 0.0,
        bridge_active_steps: int | None = None,
        generator=None,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        if self.sd3_lora_adapter is None:
            raise RuntimeError("SD3 LoRA active-step sampling requires a loaded SD3 LoRA adapter.")
        active_steps = int(active_steps)
        if active_steps < 0:
            raise ValueError("sd3_lora_active_steps must be >= 0.")
        if bridge_active_steps is not None:
            bridge_active_steps = int(bridge_active_steps)
            if bridge_active_steps < -1:
                raise ValueError("bridge_causal_active_steps must be -1 or >= 0.")

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0
        use_cfg_reference = float(cfg_guidance) > 1.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=use_cfg_reference,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if use_cfg_reference:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, _num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        latents = pipe.prepare_latents(
            len(prompts),
            pipe.transformer.config.in_channels,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        bridge_adapter = self.bridge_causal_adapter
        bridge_was_enabled = None
        if bridge_adapter is not None and bridge_active_steps is not None and bridge_active_steps >= 0:
            if hasattr(bridge_adapter, "runtime_enabled"):
                bridge_was_enabled = bool(bridge_adapter.runtime_enabled())

        try:
            for step, timestep in enumerate(timesteps):
                if active_steps == 0 or step < active_steps:
                    self.sd3_lora_adapter.set_scale(float(lora_scale))
                else:
                    self.sd3_lora_adapter.set_scale(float(inactive_scale))
                if bridge_adapter is not None and bridge_active_steps is not None and bridge_active_steps >= 0:
                    bridge_enabled = bridge_active_steps == 0 or step < bridge_active_steps
                    if hasattr(bridge_adapter, "set_runtime_enabled"):
                        bridge_adapter.set_runtime_enabled(bridge_enabled)

                if use_cfg_reference:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    timestep_input = timestep.expand(latent_model_input.shape[0])
                    noise_pred_all = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_input,
                        encoder_hidden_states=cfg_prompt_embeds,
                        pooled_projections=cfg_pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_uncond, noise_text = noise_pred_all.chunk(2)
                    noise_pred = noise_uncond + float(cfg_guidance) * (noise_text - noise_uncond)
                else:
                    noise_pred = pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep.expand(latents.shape[0]),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
        finally:
            self.sd3_lora_adapter.set_scale(1.0)
            if (
                bridge_adapter is not None
                and bridge_was_enabled is not None
                and hasattr(bridge_adapter, "set_runtime_enabled")
            ):
                bridge_adapter.set_runtime_enabled(bridge_was_enabled)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_taca(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        *,
        taca_scale: float,
        taca_active_steps: int,
        taca_layers: Any = "all",
        sd3_lora_active_steps: int = 0,
        sd3_lora_scale: float = 1.0,
        generator=None,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        taca_scale = float(taca_scale)
        if taca_scale <= 0.0:
            raise ValueError("taca_scale must be > 0.")
        taca_active_steps = int(taca_active_steps)
        if taca_active_steps < 0:
            raise ValueError("taca_active_steps must be >= 0.")
        sd3_lora_active_steps = int(sd3_lora_active_steps or 0)
        if sd3_lora_active_steps < 0:
            raise ValueError("sd3_lora_active_steps must be >= 0.")
        if self.sd3_diffusers_lora:
            self._load_sd3_diffusers_lora()

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0
        use_cfg_reference = float(cfg_guidance) > 1.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=use_cfg_reference,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if use_cfg_reference:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, _num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        latents = pipe.prepare_latents(
            len(prompts),
            pipe.transformer.config.in_channels,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        taca_target_modules = self._sfg_target_modules(self._normalize_sfg_layers(taca_layers))

        try:
            for step, timestep in enumerate(timesteps):
                lora_enabled = sd3_lora_active_steps == 0 or step < sd3_lora_active_steps
                if self.sd3_lora_adapter is not None:
                    self.sd3_lora_adapter.set_scale(float(sd3_lora_scale) if lora_enabled else 0.0)
                self._set_sd3_diffusers_lora_enabled(lora_enabled)

                use_taca = self._use_taca_this_step(step, taca_active_steps)
                taca_context = (
                    self._taca_attn_processors(taca_target_modules, taca_scale)
                    if use_taca
                    else nullcontext()
                )
                with taca_context:
                    if use_cfg_reference:
                        latent_model_input = torch.cat([latents, latents], dim=0)
                        timestep_input = timestep.expand(latent_model_input.shape[0])
                        noise_pred_all = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep_input,
                            encoder_hidden_states=cfg_prompt_embeds,
                            pooled_projections=cfg_pooled_prompt_embeds,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        noise_uncond, noise_text = noise_pred_all.chunk(2)
                        noise_pred = noise_uncond + float(cfg_guidance) * (noise_text - noise_uncond)
                    else:
                        noise_pred = pipe.transformer(
                            hidden_states=latents,
                            timestep=timestep.expand(latents.shape[0]),
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
        finally:
            if self.sd3_lora_adapter is not None:
                self.sd3_lora_adapter.set_scale(1.0)
            self._set_sd3_diffusers_lora_enabled(True)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    def _load_bridge_causal_adapter(self, adapter_path: str, prompt_source: str = "") -> None:
        from bridge_causal_sd3 import BridgeCausalConfig, BridgeCausalSD3Adapter

        path = Path(adapter_path)
        if path.is_dir():
            checkpoint_path = path / "adapter.pt"
            config_path = path / "adapter_config.json"
        else:
            checkpoint_path = path
            config_path = path.with_name("adapter_config.json")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing Bridge-Causal adapter checkpoint: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing Bridge-Causal adapter config: {config_path}")

        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        valid_keys = {field.name for field in fields(BridgeCausalConfig)}
        config = BridgeCausalConfig(**{key: value for key, value in raw_config.items() if key in valid_keys})
        adapter = BridgeCausalSD3Adapter(self.pipe.transformer, config).to(self.device)
        state = torch.load(checkpoint_path, map_location="cpu")
        adapter.load_state_dict(state, strict=True)
        transformer_dtype = next(self.pipe.transformer.parameters()).dtype
        adapter.to(device=self.device, dtype=transformer_dtype)
        adapter.eval()
        if prompt_source:
            adapter.set_prompt_source(prompt_source)
        adapter.install(self.pipe.transformer)

        transformer = self.pipe.transformer
        original_forward = transformer.forward

        def bridge_causal_forward(
            transformer_self,
            hidden_states,
            encoder_hidden_states=None,
            pooled_projections=None,
            timestep=None,
            return_dict=True,
            **kwargs,
        ):
            if (
                not bool(adapter.runtime_enabled())
                or encoder_hidden_states is None
                or pooled_projections is None
                or timestep is None
            ):
                return original_forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    timestep=timestep,
                    return_dict=return_dict,
                    **kwargs,
                )
            return adapter(
                transformer_self,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                return_dict=return_dict,
            )

        transformer.forward = MethodType(bridge_causal_forward, transformer)
        self.bridge_causal_adapter = adapter

    @staticmethod
    def _normalize_pag_layers(layers: Any) -> Any:
        def normalize_one(layer: str) -> str:
            stripped = layer.strip()
            lowered = stripped.lower()
            if lowered in {"", "mid", "default"}:
                return r"transformer_blocks\.13\.attn$"
            match = re.fullmatch(r"blocks\.(\d+)", stripped)
            if match:
                return rf"transformer_blocks\.{match.group(1)}\.attn$"
            return stripped

        if layers is None:
            return r"transformer_blocks\.13\.attn$"
        if isinstance(layers, str):
            return normalize_one(layers)
        if isinstance(layers, (list, tuple)):
            return [normalize_one(str(layer)) for layer in layers]
        return layers

    @staticmethod
    def _s2_candidate_block_indices(num_blocks: int, block_start: int, block_end: int) -> List[int]:
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
        return list(range(start, end))

    @staticmethod
    def _sample_s2_block_indices(candidates: List[int], num_drop_blocks: int, device) -> List[int]:
        num_drop_blocks = int(num_drop_blocks)
        if num_drop_blocks < 0:
            raise ValueError("s2_num_drop_blocks must be >= 0.")
        if num_drop_blocks == 0:
            return []
        if num_drop_blocks > len(candidates):
            raise ValueError(
                f"s2_num_drop_blocks={num_drop_blocks} exceeds available S2 candidate blocks "
                f"({len(candidates)})."
            )
        candidate_tensor = torch.tensor(candidates, device=device, dtype=torch.long)
        perm = torch.randperm(candidate_tensor.numel(), device=device)[:num_drop_blocks]
        return sorted(int(i) for i in candidate_tensor[perm].detach().cpu().tolist())

    @staticmethod
    @contextmanager
    def _drop_sd3_transformer_blocks(transformer, block_indices: List[int]):
        saved_forwards = []

        def dropped_forward(self, hidden_states, encoder_hidden_states, temb):
            return encoder_hidden_states, hidden_states

        try:
            for index in sorted(set(int(i) for i in block_indices)):
                block = transformer.transformer_blocks[index]
                saved_forwards.append((block, block.forward))
                block.forward = MethodType(dropped_forward, block)
            yield
        finally:
            for block, forward in reversed(saved_forwards):
                block.forward = forward

    @staticmethod
    @contextmanager
    def _ssg_transformer_blocks(
        transformer,
        target_layers=None,
        mode="both",
        ratio=0.10,
        max_pairs=0,
        max_candidates=512,
        *,
        step_index=None,
        timestep="",
        diag_writer=None,
    ):
        saved_forwards = []
        target_layer_set = None if target_layers is None else {int(index) for index in target_layers}

        try:
            for layer_index, block in enumerate(getattr(transformer, "transformer_blocks", [])):
                if target_layer_set is not None and layer_index not in target_layer_set:
                    continue
                original_forward = block.forward

                def wrapped_forward(
                    self_block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    _original_forward=original_forward,
                    _layer_index=layer_index,
                ):
                    text_token_count = 0
                    if encoder_hidden_states is not None and hasattr(encoder_hidden_states, "shape"):
                        text_token_count = int(encoder_hidden_states.shape[1])
                    hidden_states = ssg_self_swap(
                        hidden_states,
                        mode=mode,
                        ratio=ratio,
                        max_pairs=max_pairs,
                        max_candidates=max_candidates,
                        layer_index=_layer_index,
                        diag_writer=diag_writer,
                        step_index=step_index,
                        timestep=timestep,
                        text_token_count=text_token_count,
                    )
                    return _original_forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                    )

                saved_forwards.append((block, original_forward))
                block.forward = MethodType(wrapped_forward, block)
            yield
        finally:
            for block, forward in reversed(saved_forwards):
                block.forward = forward

    @staticmethod
    def _use_s2_this_step(step_index: int, s2_start_step: int, s2_end_step: int) -> bool:
        step_number = int(step_index) + 1
        if step_number < int(s2_start_step):
            return False
        s2_end_step = int(s2_end_step)
        if s2_end_step > 0 and step_number > s2_end_step:
            return False
        return True

    @staticmethod
    def _use_ssg_this_step(step_index: int, ssg_start_step: int, ssg_end_step: int) -> bool:
        step_number = int(step_index) + 1
        if step_number < int(ssg_start_step):
            return False
        ssg_end_step = int(ssg_end_step)
        if ssg_end_step > 0 and step_number > ssg_end_step:
            return False
        return True

    def _normalize_ssg_layers(self, layers: Any) -> List[int]:
        num_blocks = len(self.pipe.transformer.transformer_blocks)

        def layer_range(start: int, end: int) -> List[int]:
            return list(range(max(0, start), min(num_blocks, end)))

        def normalize_one(layer: str) -> List[int]:
            stripped = layer.strip()
            lowered = stripped.lower()
            if lowered in {"", "all", "default"}:
                return list(range(num_blocks))
            if lowered in {"early", "first"}:
                return layer_range(0, max(1, num_blocks // 3))
            if lowered in {"mid", "middle"}:
                return layer_range(num_blocks // 3, max(num_blocks // 3 + 1, (2 * num_blocks) // 3))
            if lowered in {"late", "last"}:
                return layer_range((2 * num_blocks) // 3, num_blocks)
            match = re.fullmatch(r"blocks\.(\d+)", stripped)
            if match:
                return [int(match.group(1))]
            range_match = re.fullmatch(r"(?:blocks\.)?(\d+)\s*[-:]\s*(?:blocks\.)?(\d+)", stripped)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if end < start:
                    start, end = end, start
                return layer_range(start, end + 1)
            if stripped.isdigit():
                return [int(stripped)]
            raise ValueError(f"Unsupported SD3 SSG layer selector: {layer!r}")

        if layers is None:
            raw_layers = ["all"]
        elif isinstance(layers, (list, tuple)):
            raw_layers = [str(layer) for layer in layers]
        else:
            raw_layers = [layer for layer in re.split(r"[\s,;]+", str(layers)) if layer]

        normalized: List[int] = []
        seen = set()
        for layer in raw_layers:
            for index in normalize_one(str(layer)):
                if index < 0 or index >= num_blocks:
                    raise ValueError(f"SD3 SSG layer index out of range: {index}; num_blocks={num_blocks}")
                if index not in seen:
                    normalized.append(index)
                    seen.add(index)
        if not normalized:
            raise ValueError(f"Cannot find SD3 SSG layers from selector: {layers!r}")
        return normalized

    def _normalize_sfg_layers(self, layers: Any) -> List[str]:
        num_blocks = len(self.pipe.transformer.transformer_blocks)

        def block_range(start: int, end: int) -> List[str]:
            return [rf"transformer_blocks\.{index}\.attn$" for index in range(start, end)]

        def normalize_one(layer: str) -> List[str]:
            stripped = layer.strip()
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

        if layers is None:
            return [r"transformer_blocks\.\d+\.attn$"]
        if isinstance(layers, (list, tuple)):
            raw_layers = [str(layer) for layer in layers]
        else:
            raw_layers = [layer for layer in re.split(r"[\s,;]+", str(layers)) if layer]

        normalized = []
        for layer in raw_layers:
            normalized.extend(normalize_one(str(layer)))
        return normalized

    @staticmethod
    def _use_sfg_this_step(step_index: int, bridge_start_step: int, bridge_end_step: int) -> bool:
        step_number = int(step_index) + 1
        if step_number < int(bridge_start_step):
            return False
        bridge_end_step = int(bridge_end_step)
        if bridge_end_step > 0 and step_number > bridge_end_step:
            return False
        return True

    def _sfg_target_modules(self, bridge_layers: List[str]):
        def is_self_attn(module):
            return isinstance(module, Attention) and not module.is_cross_attention

        def is_fake_integral_match(layer_id, name):
            layer_id = layer_id.split(".")[-1]
            name = name.split(".")[-1]
            return layer_id.isnumeric() and name.isnumeric() and layer_id == name

        targets = []
        for layer_id in bridge_layers:
            for name, module in self.pipe.transformer.named_modules():
                if (
                    is_self_attn(module)
                    and re.search(layer_id, name) is not None
                    and not is_fake_integral_match(layer_id, name)
                    and all(existing_module is not module for _, existing_module in targets)
                ):
                    targets.append((name, module))
        if not targets:
            raise ValueError(f"Cannot find SD3.5M SFG attention layers: {bridge_layers}")
        return targets

    @staticmethod
    @contextmanager
    def _taca_attn_processors(target_modules, taca_scale: float):
        saved_processors = []
        try:
            for name, module in target_modules:
                saved_processors.append((module, module.processor))
                processor = TACAJointAttnProcessor2_0(
                    taca_scale=taca_scale,
                    layer_name=name,
                )
                if hasattr(module, "set_processor"):
                    module.set_processor(processor)
                else:
                    module.processor = processor
            yield
        finally:
            for module, processor in reversed(saved_processors):
                if hasattr(module, "set_processor"):
                    module.set_processor(processor)
                else:
                    module.processor = processor

    @staticmethod
    def _use_taca_this_step(step_index: int, taca_active_steps: int) -> bool:
        taca_active_steps = int(taca_active_steps)
        return taca_active_steps == 0 or int(step_index) < taca_active_steps

    @staticmethod
    @contextmanager
    def _sfg_attn_processors(
        target_modules,
        sfg_strength_u: float,
        stats=None,
        clean_prefix_batch_size: int = 0,
        bridge_scale_delta=None,
        sfg_strength_u_t2i=None,
        bridge_scale_delta_t2i=None,
        bridge_direction="image_to_text",
    ):
        saved_processors = []
        try:
            for name, module in target_modules:
                saved_processors.append((module, module.processor))
                processor = SFGJointAttnProcessor2_0(
                    sfg_strength_u=sfg_strength_u,
                    layer_name=name,
                    stats=stats,
                    clean_prefix_batch_size=clean_prefix_batch_size,
                    bridge_scale_delta=bridge_scale_delta,
                    sfg_strength_u_t2i=sfg_strength_u_t2i,
                    bridge_scale_delta_t2i=bridge_scale_delta_t2i,
                    bridge_direction=bridge_direction,
                )
                if hasattr(module, "set_processor"):
                    module.set_processor(processor)
                else:
                    module.processor = processor
            yield
        finally:
            for module, processor in reversed(saved_processors):
                if hasattr(module, "set_processor"):
                    module.set_processor(processor)
                else:
                    module.processor = processor

    @staticmethod
    def _normalize_bridge_variant(sfg_mode: Any = "none", bridge_variant: Any = "explicit") -> str:
        mode = str(sfg_mode or "none").strip().lower()
        variant = str(bridge_variant or "explicit").strip().lower()
        if mode in {"direct_sfg", "direct_sfg_alt"}:
            variant = "direct_sfg"
        if mode in {"sparse_sfg", "sparse"}:
            variant = "sparse"
        aliases = {
            "explicit_sfg": "explicit",
            "sfg": "explicit",
            "bridge": "explicit",
            "direct_sfg_alt": "direct_sfg",
        }
        variant = aliases.get(variant, variant)
        if variant not in {"explicit", "direct_sfg", "sparse"}:
            raise ValueError(f"Unsupported SFG variant: {bridge_variant}")
        return variant

    @staticmethod
    def _flatten_prediction(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float().flatten(1)

    @classmethod
    def _orthogonalize_residual(cls, residual: torch.Tensor, reference: torch.Tensor, eps: float = 1e-6):
        residual_flat = cls._flatten_prediction(residual)
        reference_flat = cls._flatten_prediction(reference)
        denom = reference_flat.pow(2).sum(dim=1).clamp_min(eps)
        coeff = (residual_flat * reference_flat).sum(dim=1) / denom
        ortho = residual_flat - coeff[:, None] * reference_flat
        return ortho.reshape_as(residual).to(residual.dtype)

    @classmethod
    def _normclip_residual(
        cls,
        residual: torch.Tensor,
        reference: torch.Tensor,
        tau: float,
        eps: float = 1e-6,
    ):
        residual_flat = cls._flatten_prediction(residual)
        reference_flat = cls._flatten_prediction(reference)
        residual_norm = residual_flat.norm(dim=1)
        reference_norm = reference_flat.norm(dim=1)
        scale = torch.minimum(
            torch.ones_like(residual_norm),
            float(tau) * reference_norm / residual_norm.clamp_min(eps),
        )
        view_shape = [residual.shape[0]] + [1] * (residual.ndim - 1)
        clipped = residual * scale.reshape(view_shape).to(residual.dtype)
        cosine = F.cosine_similarity(residual_flat, reference_flat, dim=1)
        ratio = residual_norm / reference_norm.clamp_min(eps)
        return clipped, {
            "residual_norm": residual_norm,
            "reference_norm": reference_norm,
            "ratio": ratio,
            "cosine": cosine,
            "clip_scale": scale,
        }

    @classmethod
    def _bridge_geometry_metrics(
        cls,
        raw_residual: torch.Tensor,
        effective_residual: torch.Tensor,
        reference: torch.Tensor,
        safe_residual: torch.Tensor,
        cfg_guidance: float,
        bridge_omega: float,
        eps: float = 1e-6,
    ):
        raw_flat = cls._flatten_prediction(raw_residual)
        effective_flat = cls._flatten_prediction(effective_residual)
        reference_flat = cls._flatten_prediction(reference)
        safe_flat = cls._flatten_prediction(safe_residual)

        raw_norm = raw_flat.norm(dim=1)
        effective_norm = effective_flat.norm(dim=1)
        reference_norm = reference_flat.norm(dim=1)
        safe_norm = safe_flat.norm(dim=1)

        reference_sq = reference_flat.pow(2).sum(dim=1).clamp_min(eps)
        raw_dot_reference = (raw_flat * reference_flat).sum(dim=1)
        parallel_coeff = raw_dot_reference / reference_sq
        parallel_flat = parallel_coeff[:, None] * reference_flat
        parallel_norm = parallel_flat.norm(dim=1)
        orth_norm = (raw_flat - parallel_flat).norm(dim=1)

        cfg_scale = max(abs(float(cfg_guidance)), eps)
        return {
            "ratio_raw": raw_norm / reference_norm.clamp_min(eps),
            "ratio_effective": effective_norm / reference_norm.clamp_min(eps),
            "cos_raw": F.cosine_similarity(raw_flat, reference_flat, dim=1),
            "cos_effective": F.cosine_similarity(effective_flat, reference_flat, dim=1),
            "parallel_coeff": parallel_coeff,
            "parallel_norm_frac": parallel_norm / raw_norm.clamp_min(eps),
            "orth_frac": orth_norm / raw_norm.clamp_min(eps),
            "norm_cfg": reference_norm,
            "norm_sfg_s_raw": raw_norm,
            "norm_sfg_s_effective": effective_norm,
            "norm_sfg_s_safe": safe_norm,
            "update_ratio": float(bridge_omega) * safe_norm / (cfg_scale * reference_norm.clamp_min(eps)),
        }

    @staticmethod
    def _normalize_seg_layers(layers: Any) -> List[str]:
        if layers is None:
            return ["blocks.7", "blocks.8", "blocks.9"]
        if isinstance(layers, (list, tuple)):
            parsed = [str(layer).strip() for layer in layers if str(layer).strip()]
        else:
            stripped = str(layers).strip()
            if stripped.lower() in {"", "mid", "default"}:
                return ["blocks.7", "blocks.8", "blocks.9"]
            if stripped.lower() in {"none", "null", "[]"}:
                return []
            parsed = [layer for layer in re.split(r"[\s,;]+", stripped) if layer]
        return parsed

    def _set_seg_attn_processor(self, seg_applied_layers, do_classifier_free_guidance, seg_blur_sigma):
        processor = SEGJointAttnProcessor2_0(
            blur_sigma=seg_blur_sigma,
            do_cfg=do_classifier_free_guidance,
        )
        saved_processors = {}

        def is_self_attn(module):
            return isinstance(module, Attention) and not module.is_cross_attention

        def is_fake_integral_match(layer_id, name):
            layer_id = layer_id.split(".")[-1]
            name = name.split(".")[-1]
            return layer_id.isnumeric() and name.isnumeric() and layer_id == name

        for layer_id in seg_applied_layers:
            target_modules = []
            for name, module in self.pipe.transformer.named_modules():
                if (
                    is_self_attn(module)
                    and re.search(layer_id, name) is not None
                    and not is_fake_integral_match(layer_id, name)
                ):
                    target_modules.append(module)
            if not target_modules:
                raise ValueError(f"Cannot find SD3.5M SEG attention layer: {layer_id}")
            for module in target_modules:
                if module not in saved_processors:
                    saved_processors[module] = module.processor
                    if hasattr(module, "set_processor"):
                        module.set_processor(processor)
                    else:
                        module.processor = processor
        return saved_processors

    @staticmethod
    def _restore_attn_processors(saved_processors):
        for module, processor in saved_processors.items():
            if hasattr(module, "set_processor"):
                module.set_processor(processor)
            else:
                module.processor = processor

    def _set_pag_attn_processor(self, pag_applied_layers, do_classifier_free_guidance=True):
        from diffusers.models.attention_processor import PAGCFGJointAttnProcessor2_0

        processor = (
            PAGCFGJointAttnProcessor2_0()
            if do_classifier_free_guidance
            else PAGJointAttnProcessor2_0Compat()
        )
        saved_processors = {}

        def is_self_attn(module):
            return isinstance(module, Attention) and not module.is_cross_attention

        def is_fake_integral_match(layer_id, name):
            layer_id = layer_id.split(".")[-1]
            name = name.split(".")[-1]
            return layer_id.isnumeric() and name.isnumeric() and layer_id == name

        for layer_id in pag_applied_layers:
            target_modules = []
            for name, module in self.pipe.transformer.named_modules():
                if (
                    is_self_attn(module)
                    and re.search(layer_id, name) is not None
                    and not is_fake_integral_match(layer_id, name)
                ):
                    target_modules.append(module)
            if not target_modules:
                raise ValueError(f"Cannot find SD3.5M PAG attention layer: {layer_id}")
            for module in target_modules:
                if module not in saved_processors:
                    saved_processors[module] = module.processor
                    if hasattr(module, "set_processor"):
                        module.set_processor(processor)
                    else:
                        module.processor = processor
        return saved_processors

    def _set_pag_cfg_attn_processor(self, pag_applied_layers):
        return self._set_pag_attn_processor(
            pag_applied_layers,
            do_classifier_free_guidance=True,
        )

    @staticmethod
    def _pg_corrupt_scale(pg_noise_scale=0.0, pg_bad_xt_corrupt_scale=None) -> float:
        if pg_bad_xt_corrupt_scale is None:
            return float(pg_noise_scale)
        return float(pg_bad_xt_corrupt_scale)

    @staticmethod
    def _pg_is_xtmix_corruption(pg_bad_xt_corrupt_type) -> bool:
        corrupt_type = str(pg_bad_xt_corrupt_type).lower()
        return corrupt_type in {"xtmix", "xt_mix", "x_t_mix", "refmix", "ref_mix"}

    @staticmethod
    def _normalize_pg_noise_mode(pg_noise_mode) -> str:
        return str(pg_noise_mode).lower().replace("-", "_")

    @classmethod
    def _pg_is_step_noise_mode(cls, pg_noise_mode) -> bool:
        return cls._normalize_pg_noise_mode(pg_noise_mode) in {
            "step_uncond",
            "step_cfg",
            "step_size_uncond",
            "step_size_cfg",
            "stepsize_uncond",
            "stepsize_cfg",
        }

    @classmethod
    def _pg_step_noise_reference(cls, pg_noise_mode) -> str:
        if cls._normalize_pg_noise_mode(pg_noise_mode).endswith("_cfg"):
            return "cfg"
        return "uncond"

    @classmethod
    def _pg_is_flow_noise_mode(cls, pg_noise_mode) -> bool:
        return cls._normalize_pg_noise_mode(pg_noise_mode) in {
            "flow_prev_uncond",
            "flow_prev_cfg",
            "flow_uncond",
            "flow_cfg",
        }

    @classmethod
    def _pg_flow_reference(cls, pg_noise_mode) -> str:
        if cls._normalize_pg_noise_mode(pg_noise_mode).endswith("_cfg"):
            return "cfg"
        return "uncond"

    @staticmethod
    def _rms(x):
        reduce_dims = tuple(range(1, x.ndim))
        return x.float().pow(2).mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(1e-6)

    @classmethod
    def _clamp_delta_rms(cls, delta, latents, pg_noise_tau=0.0):
        tau = float(pg_noise_tau)
        if tau <= 0.0:
            return delta
        delta_rms = cls._rms(delta)
        latent_rms = cls._rms(latents)
        max_rms = latent_rms * tau
        scale = torch.minimum(torch.ones_like(delta_rms), max_rms / delta_rms.clamp_min(1e-6))
        return (delta.float() * scale).to(dtype=delta.dtype)

    @staticmethod
    def _scheduler_delta_sigma(scheduler, timesteps, step, device=None) -> float:
        sigmas = getattr(scheduler, "sigmas", None)
        try:
            if sigmas is not None and len(sigmas) > step + 1:
                return float((sigmas[step] - sigmas[step + 1]).abs().item())
        except Exception:
            pass
        current = timesteps[step]
        if step + 1 < len(timesteps):
            nxt = timesteps[step + 1]
        else:
            nxt = torch.as_tensor(0.0, device=device or current.device, dtype=current.dtype)
        current_value = float(current.item() if hasattr(current, "item") else current)
        next_value = float(nxt.item() if hasattr(nxt, "item") else nxt)
        return abs(current_value - next_value) / 1000.0

    @classmethod
    def _pg_corruption_enabled(
        cls,
        pg_bad_xt_corrupt_type="noise",
        pg_noise_scale=0.0,
        pg_bad_xt_corrupt_scale=None,
    ) -> bool:
        corrupt_type = str(pg_bad_xt_corrupt_type).lower()
        if corrupt_type in {"none", "identity"}:
            return False
        return abs(cls._pg_corrupt_scale(pg_noise_scale, pg_bad_xt_corrupt_scale)) > 0.0

    @staticmethod
    def _make_pg_spatial_noise(
        latents,
        noise_scale=0.0,
        pg_noise_mean=-3.0,
        pg_noise_std=0.5,
        pg_noise_mode="logspatial",
        reference_pred=None,
        delta_sigma=1.0,
        pg_noise_tau=0.0,
    ):
        batch_size, _, height, width = latents.shape
        noise_mode = SD35M._normalize_pg_noise_mode(pg_noise_mode)
        if noise_mode in {"rms", "rms_spatial", "rms-normalized", "rms_normalized"}:
            noise = torch.randn((batch_size, 1, height, width), device=latents.device, dtype=latents.dtype)
            noise_rms = SD35M._rms(noise)
            latent_rms = SD35M._rms(latents)
            noise = noise.float() / noise_rms * latent_rms * float(noise_scale)
            return SD35M._clamp_delta_rms(noise.to(dtype=latents.dtype), latents, pg_noise_tau)
        if SD35M._pg_is_step_noise_mode(noise_mode):
            if reference_pred is None:
                raise ValueError("step-size PG noise requires a reference prediction.")
            noise = torch.randn((batch_size, 1, height, width), device=latents.device, dtype=latents.dtype)
            noise_rms = SD35M._rms(noise)
            pred_rms = SD35M._rms(reference_pred)
            noise = noise.float() / noise_rms * pred_rms * float(delta_sigma) * float(noise_scale)
            return SD35M._clamp_delta_rms(noise.to(dtype=latents.dtype), latents, pg_noise_tau)
        if noise_mode not in {"logspatial", "log_spatial", "legacy"}:
            raise ValueError(
                "pg_noise_mode must be one of logspatial, rms, step_uncond, step_cfg, "
                "flow_prev_uncond, or flow_prev_cfg."
            )

        noises = []
        for _ in range(batch_size):
            noise = torch.randn((1, 1, height, width), device=latents.device, dtype=latents.dtype)
            log_sigma = float(pg_noise_mean) + float(pg_noise_std) * torch.randn(
                (1,), device=latents.device, dtype=torch.float32
            )
            sigma = torch.exp(log_sigma).to(dtype=latents.dtype) * float(noise_scale)
            noises.append(noise * sigma.view(1, 1, 1, 1))
        return torch.cat(noises, dim=0)

    @classmethod
    def _make_pg_bad_latent_input(
        cls,
        clean_model_input,
        pg_bad_xt_corrupt_type="noise",
        pg_noise_scale=0.0,
        pg_noise_mean=-3.0,
        pg_noise_std=0.5,
        pg_noise_mode="logspatial",
        pg_bad_xt_corrupt_scale=None,
        reference_model_input=None,
        reference_pred=None,
        prev_velocity=None,
        delta_sigma=1.0,
        pg_noise_tau=0.0,
    ):
        corrupt_type = str(pg_bad_xt_corrupt_type).lower()
        corrupt_scale = cls._pg_corrupt_scale(pg_noise_scale, pg_bad_xt_corrupt_scale)
        if corrupt_type == "noise":
            if cls._pg_is_flow_noise_mode(pg_noise_mode):
                if prev_velocity is None or abs(corrupt_scale) == 0.0:
                    return clean_model_input
                delta = prev_velocity.detach().to(dtype=clean_model_input.dtype)
                delta = delta * float(delta_sigma) * float(corrupt_scale)
                delta = cls._clamp_delta_rms(delta, clean_model_input, pg_noise_tau)
                return clean_model_input + delta
            if abs(corrupt_scale) == 0.0:
                return clean_model_input
            return clean_model_input + cls._make_pg_spatial_noise(
                clean_model_input,
                corrupt_scale,
                pg_noise_mean,
                pg_noise_std,
                pg_noise_mode,
                reference_pred=reference_pred,
                delta_sigma=delta_sigma,
                pg_noise_tau=pg_noise_tau,
            )
        if corrupt_type in {"none", "identity"}:
            return clean_model_input
        if cls._pg_is_xtmix_corruption(corrupt_type):
            if reference_model_input is None:
                raise ValueError("PG xtmix corruption requires a reference latent input.")
            ref_weight = float(corrupt_scale)
            if not 0.0 <= ref_weight <= 1.0:
                raise ValueError("PG xtmix corrupt scale must be in [0, 1].")
            return ref_weight * reference_model_input + (1.0 - ref_weight) * clean_model_input
        raise ValueError("pg_bad_xt_corrupt_type must be one of noise, xtmix, none.")

    @classmethod
    def _use_pg_this_step(
        cls,
        step_index,
        num_steps,
        timestep,
        pg_bad_condition="same",
        pg_bad_xt_corrupt_type="noise",
        pg_noise_scale=0.0,
        pg_bad_xt_corrupt_scale=None,
        pg_max_t=1000.0,
        pg_disable_last_steps=0,
        pg_start_step=1,
        pg_end_step=0,
    ) -> bool:
        if str(pg_bad_condition).lower() != "uncond":
            return False
        if not cls._pg_corruption_enabled(pg_bad_xt_corrupt_type, pg_noise_scale, pg_bad_xt_corrupt_scale):
            return False
        current_step = int(step_index) + 1
        if current_step < int(pg_start_step):
            return False
        pg_end_step = int(pg_end_step)
        if pg_end_step > 0 and current_step > pg_end_step:
            return False
        timestep_value = float(timestep.item() if hasattr(timestep, "item") else timestep)
        if timestep_value >= float(pg_max_t):
            return False
        pg_disable_last_steps = int(pg_disable_last_steps)
        if pg_disable_last_steps > 0 and int(step_index) >= max(int(num_steps) - pg_disable_last_steps, 0):
            return False
        return True

    @torch.no_grad()
    def _sample_pg(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        pg_bad_condition = str(kwargs.get("pg_bad_condition", "same")).lower()
        pg_bad_xt_corrupt_type = str(kwargs.get("pg_bad_xt_corrupt_type", "noise")).lower()
        pg_noise_scale = float(kwargs.get("pg_noise_scale", 0.0))
        pg_noise_mean = float(kwargs.get("pg_noise_mean", -3.0))
        pg_noise_std = float(kwargs.get("pg_noise_std", 0.5))
        pg_noise_mode = str(kwargs.get("pg_noise_mode", "logspatial")).lower()
        pg_noise_tau = float(kwargs.get("pg_noise_tau", 0.0))
        pg_bad_xt_corrupt_scale = kwargs.get("pg_bad_xt_corrupt_scale", None)
        pg_bad_xt_corrupt_scale = (
            None if pg_bad_xt_corrupt_scale is None else float(pg_bad_xt_corrupt_scale)
        )
        pg_bad_xt_ref_refresh_steps = int(kwargs.get("pg_bad_xt_ref_refresh_steps", 0))
        pg_max_t = float(kwargs.get("pg_max_t", 1000.0))
        pg_disable_last_steps = int(kwargs.get("pg_disable_last_steps", 0))
        pg_start_step = int(kwargs.get("pg_start_step", 1))
        pg_end_step = int(kwargs.get("pg_end_step", 0))

        if pg_bad_condition != "uncond":
            raise ValueError("SD3.5M PG currently supports pg_bad_condition='uncond' only.")
        if float(cfg_guidance) <= 1.0:
            raise ValueError("SD3.5M PG requires cfg_guidance > 1.0.")
        if pg_noise_scale < 0.0:
            raise ValueError("pg_noise_scale must be >= 0.")
        if pg_noise_tau < 0.0:
            raise ValueError("pg_noise_tau must be >= 0.")
        if pg_bad_xt_corrupt_scale is not None and pg_bad_xt_corrupt_scale < 0.0:
            raise ValueError("pg_bad_xt_corrupt_scale must be >= 0 when provided.")
        if pg_bad_xt_ref_refresh_steps < 0:
            raise ValueError("pg_bad_xt_ref_refresh_steps must be >= 0.")
        if pg_start_step < 1:
            raise ValueError("pg_start_step must be >= 1.")
        if pg_end_step < 0:
            raise ValueError("pg_end_step must be >= 0, where 0 means no explicit end.")
        if pg_end_step > 0 and pg_end_step < pg_start_step:
            raise ValueError("pg_end_step must be >= pg_start_step when set.")

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False

        (
            positive_prompt_embeds,
            negative_prompt_embeds,
            positive_pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, positive_pooled_prompt_embeds], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        pg_initial_latents = latents.detach().clone()
        pg_refresh_latents = pg_initial_latents
        pg_diag_dir = kwargs.get("pg_diag_dir", None)
        pg_diag_prefix = str(kwargs.get("pg_diag_prefix", "") or "")
        diag_writer = None
        diag_file = None
        if pg_diag_dir:
            import csv
            import os
            from pathlib import Path

            diag_dir = Path(pg_diag_dir)
            diag_dir.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", pg_diag_prefix).strip("_") or "pg_diag"
            diag_file = (diag_dir / f"{safe_prefix}_rank{rank}.csv").open("a", newline="")
            diag_writer = csv.writer(diag_file)
            if diag_file.tell() == 0:
                diag_writer.writerow(
                    [
                        "step",
                        "timestep",
                        "batch_size",
                        "ratio_mean",
                        "ratio_median",
                        "cos_mean",
                        "cos_median",
                        "cos_neg_frac",
                        "norm_cfg_mean",
                        "norm_pg_mean",
                        "pg_noise_scale",
                        "pg_noise_mode",
                        "pg_noise_tau",
                        "delta_sigma",
                        "pg_start_step",
                        "pg_end_step",
                        "norm_uncond_v_mean",
                        "norm_bad_v_mean",
                        "norm_cond_v_mean",
                        "norm_cfg_v_mean",
                        "norm_pg_v_mean",
                        "cos_bad_uncond_mean",
                        "cos_bad_cond_mean",
                        "cos_dpg_dcfg_mean",
                        "cos_dpg_dcfg_median",
                        "cos_dpg_dcfg_neg_frac",
                        "ratio_dpg_dcfg_mean",
                        "ratio_dpg_dcfg_median",
                        "cos_update_pg_cfg_mean",
                        "cos_update_pg_cfg_median",
                        "update_delta_ratio_mean",
                        "update_delta_ratio_median",
                        "update_delta_cos_cfg_mean",
                        "update_delta_cos_cfg_median",
                        "update_delta_cos_cfg_neg_frac",
                        "r_pg_proj_on_dcfg_mean",
                        "r_pg_proj_on_dcfg_median",
                        "r_pg_proj_on_dcfg_neg_frac",
                        "bad_uncond_delta_norm_ratio_mean",
                        "bad_uncond_delta_norm_ratio_median",
                        "cond_repeat_delta_norm_mean",
                    ]
                )

        prev_uncond_velocity = None
        prev_cfg_velocity = None
        for step, timestep in enumerate(timesteps):
            clean_uncond = None
            clean_text = None
            if (
                self._pg_is_xtmix_corruption(pg_bad_xt_corrupt_type)
                and pg_bad_xt_ref_refresh_steps > 0
                and step > 0
                and step % pg_bad_xt_ref_refresh_steps == 0
            ):
                pg_refresh_latents = latents.detach().clone()

            do_pg_this_step = self._use_pg_this_step(
                step,
                len(timesteps),
                timestep,
                pg_bad_condition,
                pg_bad_xt_corrupt_type,
                pg_noise_scale,
                pg_bad_xt_corrupt_scale,
                pg_max_t,
                pg_disable_last_steps,
                pg_start_step,
                pg_end_step,
            )
            do_pg_diag = do_pg_this_step and diag_writer is not None
            if do_pg_this_step:
                reference_latents = None
                if self._pg_is_xtmix_corruption(pg_bad_xt_corrupt_type):
                    reference_latents = pg_refresh_latents
                delta_sigma = self._scheduler_delta_sigma(pipe.scheduler, timesteps, step, device=device)
                needs_clean_uncond = (
                    do_pg_diag
                    or self._pg_is_step_noise_mode(pg_noise_mode)
                    or self._pg_is_flow_noise_mode(pg_noise_mode)
                )
                if needs_clean_uncond:
                    clean_model_input = torch.cat([latents, latents], dim=0)
                    clean_timestep_input = timestep.expand(clean_model_input.shape[0])
                    clean_pred = pipe.transformer(
                        hidden_states=clean_model_input,
                        timestep=clean_timestep_input,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    clean_uncond, clean_text = clean_pred.chunk(2)
                    clean_cfg = clean_uncond + float(cfg_guidance) * (clean_text - clean_uncond)
                    if self._pg_is_step_noise_mode(pg_noise_mode):
                        if self._pg_step_noise_reference(pg_noise_mode) == "cfg":
                            reference_pred = clean_cfg
                        else:
                            reference_pred = clean_uncond
                    else:
                        reference_pred = None
                    if self._pg_is_flow_noise_mode(pg_noise_mode):
                        if self._pg_flow_reference(pg_noise_mode) == "cfg":
                            flow_velocity = prev_cfg_velocity if prev_cfg_velocity is not None else clean_cfg
                        else:
                            flow_velocity = prev_uncond_velocity if prev_uncond_velocity is not None else clean_uncond
                    else:
                        flow_velocity = None
                else:
                    clean_uncond = None
                    clean_text = None
                    reference_pred = None
                    if self._pg_is_flow_noise_mode(pg_noise_mode):
                        flow_velocity = prev_cfg_velocity if self._pg_flow_reference(pg_noise_mode) == "cfg" else prev_uncond_velocity
                    else:
                        flow_velocity = None

                latents_bad = self._make_pg_bad_latent_input(
                    latents,
                    pg_bad_xt_corrupt_type,
                    pg_noise_scale,
                    pg_noise_mean,
                    pg_noise_std,
                    pg_noise_mode,
                    pg_bad_xt_corrupt_scale,
                    reference_latents,
                    reference_pred=reference_pred,
                    prev_velocity=flow_velocity,
                    delta_sigma=delta_sigma,
                    pg_noise_tau=pg_noise_tau,
                )
                latent_model_input = torch.cat([latents_bad, latents], dim=0)
                model_prompt_embeds = prompt_embeds
                model_pooled_prompt_embeds = pooled_prompt_embeds
            else:
                delta_sigma = self._scheduler_delta_sigma(pipe.scheduler, timesteps, step, device=device)
                latent_model_input = torch.cat([latents, latents], dim=0)
                model_prompt_embeds = prompt_embeds
                model_pooled_prompt_embeds = pooled_prompt_embeds

            timestep_input = timestep.expand(latent_model_input.shape[0])
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=model_prompt_embeds,
                pooled_projections=model_pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_bad, noise_text = noise_pred.chunk(2)
            if do_pg_diag:
                if clean_uncond is None:
                    raise RuntimeError("PG diagnostics require clean uncond prediction.")
                noise_uncond = clean_uncond
                v_uncond = noise_uncond.float().flatten(1)
                v_bad = noise_bad.float().flatten(1)
                v_cond = noise_text.float().flatten(1)
                v_cond_clean = clean_text.float().flatten(1)
                d_cfg = v_cond - v_uncond
                d_pg = v_cond - v_bad
                r_pg = v_uncond - v_bad
                v_cfg = v_uncond + float(cfg_guidance) * d_cfg
                v_pg = v_bad + float(cfg_guidance) * d_pg
                update_delta = v_pg - v_cfg
                norm_cfg = d_cfg.norm(dim=1)
                norm_pg = r_pg.norm(dim=1)
                ratio = norm_pg / norm_cfg.clamp_min(1e-6)
                cos = F.cosine_similarity(r_pg, d_cfg, dim=1)
                norm_uncond_v = v_uncond.norm(dim=1)
                norm_bad_v = v_bad.norm(dim=1)
                norm_cond_v = v_cond.norm(dim=1)
                norm_cfg_v = v_cfg.norm(dim=1)
                norm_pg_v = v_pg.norm(dim=1)
                norm_d_pg = d_pg.norm(dim=1)
                cos_bad_uncond = F.cosine_similarity(v_bad, v_uncond, dim=1)
                cos_bad_cond = F.cosine_similarity(v_bad, v_cond, dim=1)
                cos_dpg_dcfg = F.cosine_similarity(d_pg, d_cfg, dim=1)
                ratio_dpg_dcfg = norm_d_pg / norm_cfg.clamp_min(1e-6)
                cos_update_pg_cfg = F.cosine_similarity(v_pg, v_cfg, dim=1)
                update_delta_norm = update_delta.norm(dim=1)
                update_delta_ratio = update_delta_norm / norm_cfg_v.clamp_min(1e-6)
                update_delta_cos_cfg = F.cosine_similarity(update_delta, v_cfg, dim=1)
                r_pg_proj_on_dcfg = (r_pg * d_cfg).sum(dim=1) / norm_cfg.pow(2).clamp_min(1e-6)
                bad_uncond_delta_norm_ratio = norm_pg / norm_uncond_v.clamp_min(1e-6)
                cond_repeat_delta_norm = (v_cond - v_cond_clean).norm(dim=1)
                diag_writer.writerow(
                    [
                        int(step) + 1,
                        float(timestep.item() if hasattr(timestep, "item") else timestep),
                        int(ratio.numel()),
                        float(ratio.mean().item()),
                        float(ratio.median().item()),
                        float(cos.mean().item()),
                        float(cos.median().item()),
                        float((cos < 0).float().mean().item()),
                        float(norm_cfg.mean().item()),
                        float(norm_pg.mean().item()),
                        float(pg_noise_scale),
                        str(pg_noise_mode),
                        float(pg_noise_tau),
                        float(delta_sigma),
                        int(pg_start_step),
                        int(pg_end_step),
                        float(norm_uncond_v.mean().item()),
                        float(norm_bad_v.mean().item()),
                        float(norm_cond_v.mean().item()),
                        float(norm_cfg_v.mean().item()),
                        float(norm_pg_v.mean().item()),
                        float(cos_bad_uncond.mean().item()),
                        float(cos_bad_cond.mean().item()),
                        float(cos_dpg_dcfg.mean().item()),
                        float(cos_dpg_dcfg.median().item()),
                        float((cos_dpg_dcfg < 0).float().mean().item()),
                        float(ratio_dpg_dcfg.mean().item()),
                        float(ratio_dpg_dcfg.median().item()),
                        float(cos_update_pg_cfg.mean().item()),
                        float(cos_update_pg_cfg.median().item()),
                        float(update_delta_ratio.mean().item()),
                        float(update_delta_ratio.median().item()),
                        float(update_delta_cos_cfg.mean().item()),
                        float(update_delta_cos_cfg.median().item()),
                        float((update_delta_cos_cfg < 0).float().mean().item()),
                        float(r_pg_proj_on_dcfg.mean().item()),
                        float(r_pg_proj_on_dcfg.median().item()),
                        float((r_pg_proj_on_dcfg < 0).float().mean().item()),
                        float(bad_uncond_delta_norm_ratio.mean().item()),
                        float(bad_uncond_delta_norm_ratio.median().item()),
                        float(cond_repeat_delta_norm.mean().item()),
                    ]
                )
                diag_file.flush()
            noise_pred = noise_bad + float(cfg_guidance) * (noise_text - noise_bad)
            if clean_uncond is not None and clean_text is not None:
                prev_uncond_velocity = clean_uncond.detach()
                prev_cfg_velocity = (clean_uncond + float(cfg_guidance) * (clean_text - clean_uncond)).detach()
            else:
                prev_uncond_velocity = noise_bad.detach()
                prev_cfg_velocity = noise_pred.detach()
            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        if diag_file is not None:
            diag_file.close()

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_direct_sfg(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        direct_sfg_scale = float(kwargs.get("direct_sfg_scale", 1.0))
        direct_sfg_scale_t2i_raw = kwargs.get("direct_sfg_scale_t2i", None)
        bridge_start_step = int(kwargs.get("bridge_start_step", 1))
        bridge_end_step = int(kwargs.get("bridge_end_step", 0))
        bridge_layers = self._normalize_sfg_layers(kwargs.get("bridge_layers", "all"))
        bridge_direction = SFGJointAttnProcessor2_0._normalize_bridge_direction(
            kwargs.get("bridge_direction", "image_to_text")
        )
        if direct_sfg_scale_t2i_raw is None or direct_sfg_scale_t2i_raw == "":
            direct_sfg_scale_t2i = direct_sfg_scale if bridge_direction == "both" else None
        else:
            direct_sfg_scale_t2i = float(direct_sfg_scale_t2i_raw)
        bridge_orthogonal_raw = kwargs.get("bridge_orthogonal", 0)
        if isinstance(bridge_orthogonal_raw, str):
            bridge_orthogonal = bridge_orthogonal_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            bridge_orthogonal = bool(bridge_orthogonal_raw)

        # Signed coefficient: positive reinforces the selected bridge, negative
        # weakens/reverses it. SFGX uses the negative side.
        if bridge_start_step < 1:
            raise ValueError("bridge_start_step must be >= 1.")
        if bridge_end_step < 0:
            raise ValueError("bridge_end_step must be >= 0, where 0 means no explicit end.")
        if bridge_end_step > 0 and bridge_end_step < bridge_start_step:
            raise ValueError("bridge_end_step must be >= bridge_start_step when set.")
        if bridge_orthogonal:
            raise ValueError("DirectSFG does not support bridge_orthogonal; use bridge_variant=explicit.")

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0
        use_cfg_reference = float(cfg_guidance) > 1.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=use_cfg_reference,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if use_cfg_reference:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, _num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        target_modules = self._sfg_target_modules(bridge_layers)

        for step, timestep in enumerate(timesteps):
            use_bridge_step = self._use_sfg_this_step(step, bridge_start_step, bridge_end_step)
            processor_context = (
                self._sfg_attn_processors(
                    target_modules,
                    sfg_strength_u=0.0,
                    clean_prefix_batch_size=latents.shape[0] if use_cfg_reference else 0,
                    bridge_scale_delta=direct_sfg_scale,
                    bridge_scale_delta_t2i=direct_sfg_scale_t2i,
                    bridge_direction=bridge_direction,
                )
                if use_bridge_step
                else nullcontext()
            )
            with processor_context:
                if use_cfg_reference:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    timestep_input = timestep.expand(latent_model_input.shape[0])
                    noise_pred_all = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_input,
                        encoder_hidden_states=cfg_prompt_embeds,
                        pooled_projections=cfg_pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_uncond, noise_text = noise_pred_all.chunk(2)
                    noise_pred = noise_uncond + float(cfg_guidance) * (noise_text - noise_uncond)
                else:
                    noise_pred = pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep.expand(latents.shape[0]),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_sfg(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        bridge_variant = self._normalize_bridge_variant(
            kwargs.get("sfg_mode", "none"),
            kwargs.get("bridge_variant", "explicit"),
        )
        if bridge_variant == "direct_sfg":
            return self._sample_direct_sfg(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                generator=generator,
                **kwargs,
            )
        if bridge_variant != "explicit":
            raise ValueError(f"SFG variant {bridge_variant!r} is not implemented yet.")

        sfg_strength_u = float(kwargs.get("sfg_strength_u", 0.2))
        sfg_strength_u_t2i_raw = kwargs.get("sfg_strength_u_t2i", sfg_strength_u)
        if sfg_strength_u_t2i_raw is None or sfg_strength_u_t2i_raw == "":
            sfg_strength_u_t2i_raw = sfg_strength_u
        sfg_strength_u_t2i = float(sfg_strength_u_t2i_raw)
        bridge_omega = float(kwargs.get("bridge_omega", 0.0))
        bridge_normclip_tau = float(kwargs.get("bridge_normclip_tau", 0.3))
        bridge_start_step = int(kwargs.get("bridge_start_step", 1))
        bridge_end_step = int(kwargs.get("bridge_end_step", 0))
        bridge_layers = self._normalize_sfg_layers(kwargs.get("bridge_layers", "all"))
        bridge_direction = SFGJointAttnProcessor2_0._normalize_bridge_direction(
            kwargs.get("bridge_direction", "image_to_text")
        )
        bridge_orthogonal_raw = kwargs.get("bridge_orthogonal", 0)
        if isinstance(bridge_orthogonal_raw, str):
            bridge_orthogonal = bridge_orthogonal_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            bridge_orthogonal = bool(bridge_orthogonal_raw)

        if bridge_omega < 0.0:
            raise ValueError("Explicit SFG requires bridge_omega >= 0.")
        if sfg_strength_u == 0.0 or sfg_strength_u < -1.0 or sfg_strength_u > 1.0:
            raise ValueError("sfg_strength_u must be non-zero and in [-1, 1].")
        if sfg_strength_u_t2i < -1.0 or sfg_strength_u_t2i > 1.0:
            raise ValueError("sfg_strength_u_t2i must be in [-1, 1].")
        if bridge_normclip_tau < 0.0:
            raise ValueError("bridge_normclip_tau must be >= 0.")
        if bridge_start_step < 1:
            raise ValueError("bridge_start_step must be >= 1.")
        if bridge_end_step < 0:
            raise ValueError("bridge_end_step must be >= 0, where 0 means no explicit end.")
        if bridge_end_step > 0 and bridge_end_step < bridge_start_step:
            raise ValueError("bridge_end_step must be >= bridge_start_step when set.")

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0
        use_cfg_reference = (
            float(cfg_guidance) > 1.0
            or bridge_orthogonal
            or bridge_normclip_tau < 999.0
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=use_cfg_reference,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if use_cfg_reference:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        target_modules = self._sfg_target_modules(bridge_layers)

        bridge_diag_dir = kwargs.get("bridge_diag_dir", None)
        bridge_diag_prefix = str(kwargs.get("bridge_diag_prefix", "") or "")
        image_indices = kwargs.get("image_indices", None)
        if image_indices is None or len(image_indices) != len(prompts):
            image_indices = list(range(len(prompts)))
        diag_writer = None
        diag_file = None
        per_image_diag_writer = None
        per_image_diag_file = None
        bridge_stats = [] if bridge_diag_dir else None
        if bridge_diag_dir:
            import csv
            import os
            from pathlib import Path

            diag_dir = Path(bridge_diag_dir)
            diag_dir.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", bridge_diag_prefix).strip("_") or "sfg_diag"
            diag_file = (diag_dir / f"{safe_prefix}_rank{rank}.csv").open("a", newline="")
            diag_writer = csv.writer(diag_file)
            if diag_file.tell() == 0:
                diag_writer.writerow(
                    [
                        "step",
                        "timestep",
                        "batch_size",
                        "sfg_strength_u",
                        "bridge_omega",
                        "bridge_normclip_tau",
                        "bridge_start_step",
                        "bridge_end_step",
                        "bridge_orthogonal",
                        "bridge_direction",
                        "num_bridge_layers",
                        "ratio_mean",
                        "ratio_median",
                        "cos_mean",
                        "cos_median",
                        "cos_neg_frac",
                        "norm_cfg_mean",
                        "norm_bridge_mean",
                        "clip_scale_mean",
                        "clip_scale_min",
                        "clip_scale_max",
                        "text_attention_mass_mean",
                        "text_attention_mass_min",
                        "text_attention_mass_max",
                        "bridge_layers",
                    ]
                )
            per_image_diag_file = (diag_dir / f"{safe_prefix}_per_image_rank{rank}.csv").open("a", newline="")
            per_image_diag_writer = csv.writer(per_image_diag_file)
            if per_image_diag_file.tell() == 0:
                per_image_diag_writer.writerow(
                    [
                        "image_index",
                        "step",
                        "timestep",
                        "sfg_strength_u",
                        "bridge_omega",
                        "cfg_guidance",
                        "bridge_normclip_tau",
                        "bridge_start_step",
                        "bridge_end_step",
                        "bridge_orthogonal",
                        "bridge_direction",
                        "ratio_raw",
                        "ratio_effective",
                        "cos_raw",
                        "cos_effective",
                        "parallel_coeff",
                        "parallel_norm_frac",
                        "orth_frac",
                        "norm_cfg",
                        "norm_sfg_s_raw",
                        "norm_sfg_s_effective",
                        "norm_sfg_s_safe",
                        "update_ratio",
                        "clip_scale",
                    ]
                )

        try:
            for step, timestep in enumerate(timesteps):
                use_bridge_step = self._use_sfg_this_step(step, bridge_start_step, bridge_end_step)
                bridge_diag = None
                if use_bridge_step and not use_cfg_reference:
                    if bridge_stats is not None:
                        bridge_stats.clear()
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    timestep_input = timestep.expand(latent_model_input.shape[0])
                    prompt_model_input = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                    pooled_model_input = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                    with self._sfg_attn_processors(
                        target_modules,
                        sfg_strength_u=sfg_strength_u,
                        sfg_strength_u_t2i=sfg_strength_u_t2i,
                        stats=bridge_stats,
                        clean_prefix_batch_size=latents.shape[0],
                        bridge_direction=bridge_direction,
                    ):
                        fused_pred = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep_input,
                            encoder_hidden_states=prompt_model_input,
                            pooled_projections=pooled_model_input,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    noise_text, noise_bridge = fused_pred.chunk(2)
                    bridge_residual = noise_text - noise_bridge
                    residual_flat = self._flatten_prediction(bridge_residual)
                    residual_norm = residual_flat.norm(dim=1)
                    bridge_safe = bridge_residual
                    bridge_diag = {
                        "ratio": torch.full_like(residual_norm, float("nan")),
                        "cosine": torch.full_like(residual_norm, float("nan")),
                        "reference_norm": torch.full_like(residual_norm, float("nan")),
                        "residual_norm": residual_norm,
                        "clip_scale": torch.ones_like(residual_norm),
                    }
                    noise_pred = noise_text + bridge_omega * bridge_safe
                elif use_cfg_reference:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    timestep_input = timestep.expand(latent_model_input.shape[0])
                    clean_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_input,
                        encoder_hidden_states=cfg_prompt_embeds,
                        pooled_projections=cfg_pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_uncond, noise_text = clean_pred.chunk(2)
                    d_cfg = noise_text - noise_uncond
                    noise_pred = noise_uncond + float(cfg_guidance) * d_cfg
                else:
                    noise_text = pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep.expand(latents.shape[0]),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    d_cfg = None
                    noise_pred = noise_text

                if use_bridge_step and bridge_diag is None:
                    if bridge_stats is not None:
                        bridge_stats.clear()
                    with self._sfg_attn_processors(
                        target_modules,
                        sfg_strength_u=sfg_strength_u,
                        sfg_strength_u_t2i=sfg_strength_u_t2i,
                        stats=bridge_stats,
                        bridge_direction=bridge_direction,
                    ):
                        noise_bridge = pipe.transformer(
                            hidden_states=latents,
                            timestep=timestep.expand(latents.shape[0]),
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                    raw_bridge_residual = noise_text - noise_bridge
                    bridge_residual = raw_bridge_residual
                    if bridge_orthogonal:
                        bridge_residual = self._orthogonalize_residual(raw_bridge_residual, d_cfg)
                    if d_cfg is None:
                        residual_flat = self._flatten_prediction(bridge_residual)
                        residual_norm = residual_flat.norm(dim=1)
                        bridge_safe = bridge_residual
                        bridge_diag = {
                            "ratio": torch.full_like(residual_norm, float("nan")),
                            "cosine": torch.full_like(residual_norm, float("nan")),
                            "reference_norm": torch.full_like(residual_norm, float("nan")),
                            "residual_norm": residual_norm,
                            "clip_scale": torch.ones_like(residual_norm),
                        }
                    else:
                        bridge_safe, bridge_diag = self._normclip_residual(
                            bridge_residual,
                            d_cfg,
                            tau=bridge_normclip_tau,
                        )
                    noise_pred = noise_pred + bridge_omega * bridge_safe

                    if per_image_diag_writer is not None and d_cfg is not None:
                        geom = self._bridge_geometry_metrics(
                            raw_bridge_residual,
                            bridge_residual,
                            d_cfg,
                            bridge_safe,
                            cfg_guidance=float(cfg_guidance),
                            bridge_omega=float(bridge_omega),
                        )
                        clip_scale = bridge_diag["clip_scale"]
                        timestep_value = float(timestep.item() if hasattr(timestep, "item") else timestep)
                        for local_idx, image_index in enumerate(image_indices):
                            per_image_diag_writer.writerow(
                                [
                                    int(image_index),
                                    int(step) + 1,
                                    timestep_value,
                                    float(sfg_strength_u),
                                    float(bridge_omega),
                                    float(cfg_guidance),
                                    float(bridge_normclip_tau),
                                    int(bridge_start_step),
                                    int(bridge_end_step),
                                    int(bridge_orthogonal),
                                    bridge_direction,
                                    float(geom["ratio_raw"][local_idx].item()),
                                    float(geom["ratio_effective"][local_idx].item()),
                                    float(geom["cos_raw"][local_idx].item()),
                                    float(geom["cos_effective"][local_idx].item()),
                                    float(geom["parallel_coeff"][local_idx].item()),
                                    float(geom["parallel_norm_frac"][local_idx].item()),
                                    float(geom["orth_frac"][local_idx].item()),
                                    float(geom["norm_cfg"][local_idx].item()),
                                    float(geom["norm_sfg_s_raw"][local_idx].item()),
                                    float(geom["norm_sfg_s_effective"][local_idx].item()),
                                    float(geom["norm_sfg_s_safe"][local_idx].item()),
                                    float(geom["update_ratio"][local_idx].item()),
                                    float(clip_scale[local_idx].item()),
                                ]
                            )
                        per_image_diag_file.flush()

                if bridge_diag is not None and diag_writer is not None:
                    masses = torch.tensor(
                        [item["cross_mass"] for item in bridge_stats],
                        dtype=torch.float32,
                    ) if bridge_stats else torch.empty(0, dtype=torch.float32)
                    ratio = bridge_diag["ratio"]
                    cosine = bridge_diag["cosine"]
                    clip_scale = bridge_diag["clip_scale"]
                    diag_writer.writerow(
                        [
                            int(step) + 1,
                            float(timestep.item() if hasattr(timestep, "item") else timestep),
                            int(ratio.numel()),
                            float(sfg_strength_u),
                            float(bridge_omega),
                            float(bridge_normclip_tau),
                            int(bridge_start_step),
                            int(bridge_end_step),
                            int(bridge_orthogonal),
                            bridge_direction,
                            int(len(target_modules)),
                            float(ratio.mean().item()),
                            float(ratio.median().item()),
                            float(cosine.mean().item()),
                            float(cosine.median().item()),
                            float((cosine < 0).float().mean().item()),
                            float(bridge_diag["reference_norm"].mean().item()),
                            float(bridge_diag["residual_norm"].mean().item()),
                            float(clip_scale.mean().item()),
                            float(clip_scale.min().item()),
                            float(clip_scale.max().item()),
                            float(masses.mean().item()) if masses.numel() else "",
                            float(masses.min().item()) if masses.numel() else "",
                            float(masses.max().item()) if masses.numel() else "",
                            ",".join(name for name, _ in target_modules),
                        ]
                    )
                    diag_file.flush()

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
        finally:
            if diag_file is not None:
                diag_file.close()

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_s2(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        s2_guidance_scale = float(kwargs.get("s2_guidance_scale", 0.0))
        s2_num_drop_blocks = int(kwargs.get("s2_num_drop_blocks", 3))
        s2_block_start = int(kwargs.get("s2_block_start", 1))
        s2_block_end = int(kwargs.get("s2_block_end", 0))
        s2_start_step = int(kwargs.get("s2_start_step", 1))
        s2_end_step = int(kwargs.get("s2_end_step", 0))

        if s2_guidance_scale <= 0.0:
            raise ValueError("SD3.5M S2Guidance requires s2_guidance_scale > 0.")
        if s2_num_drop_blocks <= 0:
            raise ValueError("SD3.5M S2Guidance requires s2_num_drop_blocks > 0.")
        if s2_start_step < 1:
            raise ValueError("s2_start_step must be >= 1.")
        if s2_end_step < 0:
            raise ValueError("s2_end_step must be >= 0, where 0 means no explicit end.")
        if s2_end_step > 0 and s2_end_step < s2_start_step:
            raise ValueError("s2_end_step must be >= s2_start_step when set.")

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0
        use_cfg_reference = abs(float(cfg_guidance) - 1.0) > 1e-6

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=use_cfg_reference,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if use_cfg_reference:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )
        candidates = self._s2_candidate_block_indices(
            len(pipe.transformer.transformer_blocks),
            s2_block_start,
            s2_block_end,
        )

        for step, timestep in enumerate(timesteps):
            if use_cfg_reference:
                latent_model_input = torch.cat([latents, latents], dim=0)
                timestep_input = timestep.expand(latent_model_input.shape[0])
                clean_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    encoder_hidden_states=cfg_prompt_embeds,
                    pooled_projections=cfg_pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_uncond, noise_text = clean_pred.chunk(2)
                noise_pred = noise_uncond + float(cfg_guidance) * (noise_text - noise_uncond)
            else:
                noise_text = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred = noise_text

            if self._use_s2_this_step(step, s2_start_step, s2_end_step):
                dropped_blocks = self._sample_s2_block_indices(
                    candidates,
                    s2_num_drop_blocks,
                    device=latents.device,
                )
                timestep_sub = timestep.expand(latents.shape[0])
                with self._drop_sd3_transformer_blocks(pipe.transformer, dropped_blocks):
                    noise_sub = pipe.transformer(
                        hidden_states=latents,
                        timestep=timestep_sub,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred + s2_guidance_scale * (noise_text - noise_sub)

            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_ssg(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        ssg_scale = float(kwargs.get("ssg_scale", 0.0))
        ssg_mode = str(kwargs.get("ssg_mode", "both"))
        ssg_ratio = float(kwargs.get("ssg_ratio", 0.10))
        ssg_max_pairs = int(kwargs.get("ssg_max_pairs", 0))
        ssg_max_candidates = int(kwargs.get("ssg_max_candidates", 512))
        ssg_start_step = int(kwargs.get("ssg_start_step", 1))
        ssg_end_step = int(kwargs.get("ssg_end_step", 0))
        ssg_applied_layers = self._normalize_ssg_layers(kwargs.get("ssg_applied_layers", "all"))
        ssg_diag_dir = kwargs.get("ssg_diag_dir", None)
        ssg_diag_prefix = str(kwargs.get("ssg_diag_prefix", "") or "")
        if ssg_scale <= 0.0:
            raise ValueError("SD3 SSG requires ssg_scale > 0.")
        if ssg_ratio <= 0.0:
            raise ValueError("ssg_ratio must be > 0.")
        if ssg_max_pairs < 0:
            raise ValueError("ssg_max_pairs must be >= 0.")
        if ssg_max_candidates < 0:
            raise ValueError("ssg_max_candidates must be >= 0.")
        if ssg_start_step < 1:
            raise ValueError("ssg_start_step must be >= 1.")
        if ssg_end_step < 0:
            raise ValueError("ssg_end_step must be >= 0.")
        if ssg_end_step > 0 and ssg_end_step < ssg_start_step:
            raise ValueError("ssg_end_step must be >= ssg_start_step.")

        pipe = self.pipe
        device = pipe._execution_device
        do_cfg = abs(float(cfg_guidance) - 1.0) > 1e-6
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if do_cfg:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, _num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        ssg_diag_file = None
        ssg_diag_writer = None
        if ssg_diag_dir:
            import csv
            import os

            diag_dir = Path(ssg_diag_dir)
            diag_dir.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", ssg_diag_prefix).strip("_") or "ssg_diag"
            ssg_diag_file = (diag_dir / f"{safe_prefix}_rank{rank}.csv").open("a", newline="")
            fieldnames = [
                "step",
                "timestep",
                "layer_index",
                "kind",
                "token_space",
                "batch_index",
                "image_token_count",
                "text_token_count",
                "channel_count",
                "candidate_count",
                "pair_count",
                "left_min",
                "left_max",
                "right_min",
                "right_max",
                "score_min",
                "score_max",
                "score_mean",
                "sample_pairs",
                "ssg_scale",
                "clean_norm_mean",
                "clean_norm_max",
                "perturbed_norm_mean",
                "residual_norm_mean",
                "residual_norm_max",
                "residual_to_clean_mean",
                "residual_to_clean_max",
                "residual_cosine_mean",
                "update_norm_mean",
                "update_to_clean_mean",
            ]
            ssg_diag_writer = csv.DictWriter(ssg_diag_file, fieldnames=fieldnames)
            if ssg_diag_file.tell() == 0:
                ssg_diag_writer.writeheader()

        def predict_cond(current_latents, timestep, use_ssg=False, step_index=None):
            timestep_value = ""
            if use_ssg and hasattr(timestep, "detach"):
                timestep_value = float(timestep.detach().float().cpu().item())
            context = self._ssg_transformer_blocks(
                pipe.transformer,
                target_layers=ssg_applied_layers,
                mode=ssg_mode,
                ratio=ssg_ratio,
                max_pairs=ssg_max_pairs,
                max_candidates=ssg_max_candidates,
                step_index=step_index,
                timestep=timestep_value,
                diag_writer=ssg_diag_writer,
            ) if use_ssg else nullcontext()
            with context:
                return pipe.transformer(
                    hidden_states=current_latents,
                    timestep=timestep.expand(current_latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

        def predict_clean_parts(current_latents, timestep):
            if do_cfg:
                latent_model_input = torch.cat([current_latents, current_latents], dim=0)
                noise_pred_all = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep.expand(latent_model_input.shape[0]),
                    encoder_hidden_states=cfg_prompt_embeds,
                    pooled_projections=cfg_pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_uncond, noise_text = noise_pred_all.chunk(2)
                return noise_uncond, noise_text
            noise_text = predict_cond(current_latents, timestep, use_ssg=False)
            return None, noise_text

        try:
            for step, timestep in enumerate(timesteps):
                if pipe.interrupt:
                    continue
                noise_uncond, clean_pred = predict_clean_parts(latents, timestep)
                if do_cfg:
                    noise_pred = noise_uncond + float(cfg_guidance) * (clean_pred - noise_uncond)
                else:
                    noise_pred = clean_pred
                if self._use_ssg_this_step(step, ssg_start_step, ssg_end_step):
                    perturbed_pred = predict_cond(latents, timestep, use_ssg=True, step_index=step)
                    ssg_residual = clean_pred - perturbed_pred
                    if ssg_diag_writer is not None:
                        clean_flat = clean_pred.float().flatten(1)
                        perturbed_flat = perturbed_pred.float().flatten(1)
                        residual_flat = ssg_residual.float().flatten(1)
                        clean_norm = clean_flat.norm(dim=1).clamp_min(1e-8)
                        perturbed_norm = perturbed_flat.norm(dim=1)
                        residual_norm = residual_flat.norm(dim=1)
                        update_norm = float(ssg_scale) * residual_norm
                        residual_cosine = F.cosine_similarity(clean_flat, residual_flat, dim=1, eps=1e-8)
                        ssg_diag_writer.writerow(
                            {
                                "step": int(step) + 1,
                                "timestep": float(timestep.detach().float().cpu().item())
                                if hasattr(timestep, "detach")
                                else timestep,
                                "layer_index": "",
                                "kind": "residual",
                                "token_space": "noise_prediction",
                                "batch_index": "all",
                                "image_token_count": int(clean_pred.shape[1])
                                if clean_pred.ndim > 2
                                else "",
                                "channel_count": int(clean_pred.shape[-1])
                                if clean_pred.ndim > 1
                                else "",
                                "ssg_scale": float(ssg_scale),
                                "clean_norm_mean": float(clean_norm.mean().item()),
                                "clean_norm_max": float(clean_norm.max().item()),
                                "perturbed_norm_mean": float(perturbed_norm.mean().item()),
                                "residual_norm_mean": float(residual_norm.mean().item()),
                                "residual_norm_max": float(residual_norm.max().item()),
                                "residual_to_clean_mean": float((residual_norm / clean_norm).mean().item()),
                                "residual_to_clean_max": float((residual_norm / clean_norm).max().item()),
                                "residual_cosine_mean": float(residual_cosine.mean().item()),
                                "update_norm_mean": float(update_norm.mean().item()),
                                "update_to_clean_mean": float((update_norm / clean_norm).mean().item()),
                            }
                        )
                    if ssg_diag_file is not None:
                        ssg_diag_file.flush()
                    noise_pred = noise_pred + float(ssg_scale) * ssg_residual

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
        finally:
            if ssg_diag_file is not None:
                ssg_diag_file.close()

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_zero_init(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        cfgzero_mode = normalize_cfgzero_mode(kwargs.get("cfgzero_mode", "none"))
        zero_steps = int(kwargs.get("cfgzero_zero_steps", 1))
        if cfgzero_mode == "none" and zero_steps > 0:
            cfgzero_mode = "zero_init"
        if cfgzero_uses_zero_init(cfgzero_mode):
            if zero_steps <= 0:
                zero_steps = max(1, int(round(0.04 * int(self.num_sampling))))
        elif zero_steps < 0:
            raise ValueError("cfgzero_zero_steps must be >= 0.")

        pipe = self.pipe
        device = pipe._execution_device
        do_cfg = abs(float(cfg_guidance) - 1.0) > 1e-6
        if cfgzero_uses_optimized_scale(cfgzero_mode) and not do_cfg:
            raise ValueError("SD3 CFG-Zero optimized scale requires classifier-free guidance.")
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if do_cfg:
            cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            cfg_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )

        timesteps, _num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        for step, timestep in enumerate(timesteps):
            if pipe.interrupt:
                continue
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                noise_pred_all = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep.expand(latent_model_input.shape[0]),
                    encoder_hidden_states=cfg_prompt_embeds,
                    pooled_projections=cfg_pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_uncond, noise_text = noise_pred_all.chunk(2)
                if cfgzero_uses_optimized_scale(cfgzero_mode):
                    scale = optimized_cfgzero_scale(noise_text, noise_uncond)
                    scaled_uncond = noise_uncond * scale
                    noise_pred = scaled_uncond + float(cfg_guidance) * (noise_text - scaled_uncond)
                else:
                    noise_pred = noise_uncond + float(cfg_guidance) * (noise_text - noise_uncond)
            else:
                noise_pred = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep.expand(latents.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            if cfgzero_uses_zero_init(cfgzero_mode) and step < zero_steps:
                noise_pred = torch.zeros_like(noise_pred)

            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_pag(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        pag_layers,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        pag_scale = float(kwargs.get("pag_scale", 0.0))
        if pag_scale <= 0.0 or not pag_layers:
            raise ValueError("SD3.5M PAG requires pag_scale > 0 and at least one PAG layer.")
        if pag_scale < 0.0:
            raise ValueError("pag_scale must be >= 0.")

        pipe = self.pipe
        device = pipe._execution_device
        do_cfg = float(cfg_guidance) > 1.0
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds, pooled_prompt_embeds],
                dim=0,
            )
        else:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        saved_processors = self._set_pag_attn_processor(
            pag_layers,
            do_classifier_free_guidance=do_cfg,
        )
        try:
            for timestep in timesteps:
                latent_repeat_count = prompt_embeds.shape[0] // latents.shape[0]
                latent_model_input = torch.cat([latents] * latent_repeat_count, dim=0)
                timestep_input = timestep.expand(latent_model_input.shape[0])
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                if do_cfg:
                    noise_uncond, noise_text, noise_perturb = noise_pred.chunk(3)
                    noise_pred = (
                        noise_uncond
                        + float(cfg_guidance) * (noise_text - noise_uncond)
                        + pag_scale * (noise_text - noise_perturb)
                    )
                else:
                    noise_text, noise_perturb = noise_pred.chunk(2)
                    noise_pred = noise_text + pag_scale * (noise_text - noise_perturb)

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
        finally:
            self._restore_attn_processors(saved_processors)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def _sample_seg(
        self,
        prompts,
        negative_prompts,
        height,
        width,
        cfg_guidance,
        generator=None,
        **kwargs,
    ):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

        seg_scale = float(kwargs.get("seg_scale", 0.0))
        seg_blur_sigma = float(kwargs.get("seg_blur_sigma", 10000.0))
        seg_layers = self._normalize_seg_layers(kwargs.get("seg_applied_layers", "mid"))
        do_cfg = float(cfg_guidance) > 1.0

        if seg_scale <= 0.0 or not seg_layers:
            raise ValueError("SD3.5M SEG requires seg_scale > 0 and at least one SEG layer.")
        if seg_scale < 0.0:
            raise ValueError("seg_scale must be >= 0.")

        pipe = self.pipe
        device = pipe._execution_device
        pipe._guidance_scale = float(cfg_guidance)
        pipe._clip_skip = None
        pipe._joint_attention_kwargs = None
        pipe._interrupt = False
        pipe._pag_scale = 0.0
        pipe._pag_adaptive_scale = 0.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, self.num_sampling, device, None)
        pipe._num_timesteps = len(timesteps)
        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            len(prompts),
            num_channels_latents,
            int(height),
            int(width),
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        saved_processors = self._set_seg_attn_processor(
            seg_layers,
            do_classifier_free_guidance=do_cfg,
            seg_blur_sigma=seg_blur_sigma,
        )
        try:
            for timestep in timesteps:
                latent_repeat_count = prompt_embeds.shape[0] // latents.shape[0]
                latent_model_input = torch.cat([latents] * latent_repeat_count, dim=0)
                timestep_input = timestep.expand(latent_model_input.shape[0])
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                if do_cfg:
                    noise_uncond, noise_text, noise_text_perturb = noise_pred.chunk(3)
                    noise_pred = (
                        noise_uncond
                        + float(cfg_guidance) * (noise_text - noise_uncond)
                        + seg_scale * (noise_text - noise_text_perturb)
                    )
                else:
                    noise_text, noise_text_perturb = noise_pred.chunk(2)
                    noise_pred = noise_text + seg_scale * (noise_text - noise_text_perturb)

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
        finally:
            self._restore_attn_processors(saved_processors)

        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.no_grad()
    def sample(
        self,
        prompt=None,
        prompt1=None,
        prompt2=None,
        cfg_guidance: float = 3.5,
        target_size=(1024, 1024),
        generator=None,
        **kwargs,
    ):
        prompt_source = prompt if prompt is not None else prompt1
        prompts = _positive_prompts(prompt_source)
        negative_prompts = _negative_prompts(prompt_source, len(prompts))
        height, width = target_size
        pag_scale = float(kwargs.get("pag_scale", 0.0))
        seg_scale = float(kwargs.get("seg_scale", 0.0))
        s2_guidance_scale = float(kwargs.get("s2_guidance_scale", 0.0))
        ssg_scale = float(kwargs.get("ssg_scale", 0.0))
        cfgzero_mode = normalize_cfgzero_mode(kwargs.get("cfgzero_mode", "none"))
        cfgzero_zero_steps = int(kwargs.get("cfgzero_zero_steps", 0))
        if cfgzero_mode == "none" and cfgzero_zero_steps > 0:
            cfgzero_mode = "zero_init"
        if cfgzero_uses_zero_init(cfgzero_mode) and cfgzero_zero_steps == 0:
            cfgzero_zero_steps = max(1, int(round(0.04 * int(self.num_sampling))))
        sfg_mode = str(kwargs.get("sfg_mode", "none")).lower()
        bridge_variant = self._normalize_bridge_variant(sfg_mode, kwargs.get("bridge_variant", "explicit"))
        bridge_omega = float(kwargs.get("bridge_omega", 0.0))
        taca_scale = float(kwargs.get("taca_scale", 1.0))
        taca_active_steps = int(kwargs.get("taca_active_steps", 3) or 0)
        taca_enabled = abs(taca_scale - 1.0) > 1e-8
        prompt_reinjection_enabled = is_prompt_reinjection_method(self.method)
        seg_layers = self._normalize_seg_layers(kwargs.get("seg_applied_layers", "mid"))
        pg_enabled = (
            str(kwargs.get("pg_bad_condition", "same")).lower() == "uncond"
            and self._pg_corruption_enabled(
                kwargs.get("pg_bad_xt_corrupt_type", "noise"),
                kwargs.get("pg_noise_scale", 0.0),
                kwargs.get("pg_bad_xt_corrupt_scale", None),
            )
        )
        if seg_scale < 0.0:
            raise ValueError("seg_scale must be >= 0.")
        seg_enabled = seg_scale > 0.0
        if seg_enabled and not seg_layers:
            raise ValueError("SD3.5M SEG requires at least one SEG layer.")
        if pg_enabled and (pag_scale > 0.0 or seg_enabled):
            raise ValueError("SD3.5M PG, PAG, and SEG are separate methods.")
        if seg_enabled and pag_scale > 0.0:
            raise ValueError("SD3.5M SEG and PAG are separate methods.")
        if s2_guidance_scale < 0.0:
            raise ValueError("s2_guidance_scale must be >= 0.")
        s2_enabled = s2_guidance_scale > 0.0
        if ssg_scale < 0.0:
            raise ValueError("ssg_scale must be >= 0.")
        ssg_enabled = ssg_scale > 0.0
        if bridge_omega < 0.0:
            raise ValueError("bridge_omega must be >= 0.")
        sfg_enabled = sfg_mode in {
            "explicit",
            "explicit_sfg",
            "sfg",
            "bridge",
            "direct_sfg",
            "direct_sfg_alt",
        } or bridge_variant == "direct_sfg" or bridge_omega > 0.0
        if sfg_mode not in {
            "none",
            "off",
            "0",
            "explicit",
            "explicit_sfg",
            "sfg",
            "bridge",
            "direct_sfg",
            "direct_sfg_alt",
        }:
            raise ValueError(f"Unsupported SFG mode: {sfg_mode}")
        if bridge_variant == "sparse":
            raise ValueError("Sparse SFG is not implemented yet; use bridge_start_step/bridge_end_step windows.")
        if s2_enabled and (pg_enabled or pag_scale > 0.0 or seg_enabled or sfg_enabled):
            raise ValueError("SD3.5M S2Guidance, SFG, PG, PAG, and SEG are separate methods.")
        if ssg_enabled and (pg_enabled or pag_scale > 0.0 or seg_enabled or s2_enabled or sfg_enabled):
            raise ValueError("SD3 SSG must be run separately from PG, PAG, SEG, S2, and SFG.")
        if sfg_enabled and (pg_enabled or pag_scale > 0.0 or seg_enabled):
            raise ValueError("SD3.5M SFG, PG, PAG, and SEG are separate methods.")
        if cfgzero_zero_steps < 0:
            raise ValueError("cfgzero_zero_steps must be >= 0.")
        if taca_active_steps < 0:
            raise ValueError("taca_active_steps must be >= 0.")
        cfgzero_enabled = cfgzero_mode != "none" or cfgzero_zero_steps > 0
        if ssg_enabled and cfgzero_enabled:
            raise ValueError("SD3 SSG and CFG-Zero must be run separately.")
        if cfgzero_enabled and (pg_enabled or pag_scale > 0.0 or seg_enabled or s2_enabled or sfg_enabled):
            raise ValueError("SD3 CFG-Zero must be run separately from PG, PAG, SEG, S2, and SFG.")
        if taca_enabled and (
            pg_enabled
            or pag_scale > 0.0
            or seg_enabled
            or s2_enabled
            or ssg_enabled
            or sfg_enabled
            or cfgzero_enabled
        ):
            raise ValueError("SD3 TACA must be run separately from PG, PAG, SEG, S2, SSG, CFG-Zero, and SFG.")
        if taca_enabled and self.bridge_causal_adapter is not None:
            raise ValueError("SD3 TACA is not currently supported together with a Bridge-Causal adapter.")
        if prompt_reinjection_enabled and (
            pg_enabled
            or pag_scale > 0.0
            or seg_enabled
            or s2_enabled
            or ssg_enabled
            or sfg_enabled
            or cfgzero_enabled
            or taca_enabled
        ):
            raise ValueError(
                "SD3 PromptReinjection must be run as a separate method, not combined with PG, PAG, SEG, "
                "S2, SSG, CFG-Zero, SFG/SFG, or TACA."
            )
        if prompt_reinjection_enabled and (
            self.bridge_causal_adapter is not None or self.sd3_lora_adapter is not None or self.sd3_diffusers_lora
        ):
            raise ValueError("SD3 PromptReinjection is a standalone method here; do not combine it with adapters/LoRA.")
        if prompt_reinjection_enabled:
            return self._sample_prompt_reinjection(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                origin_layer=int(kwargs.get("prompt_reinjection_origin_layer", 1)),
                target_layers=kwargs.get("prompt_reinjection_target_layers", "2-23"),
                weight=kwargs.get("prompt_reinjection_weight", 0.025),
                use_anchoring=_bool_arg(kwargs.get("prompt_reinjection_use_anchoring", 1), default=True),
                stop_grad=_bool_arg(kwargs.get("prompt_reinjection_stop_grad", 1), default=True),
                procrustes_path=str(kwargs.get("prompt_reinjection_procrustes_path", "") or ""),
                generator=generator,
            )
        if pg_enabled:
            return self._sample_pg(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                generator=generator,
                **kwargs,
            )
        if s2_enabled:
            return self._sample_s2(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                generator=generator,
                **kwargs,
            )
        if ssg_enabled:
            return self._sample_ssg(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                generator=generator,
                **kwargs,
            )
        if sfg_enabled:
            return self._sample_sfg(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                generator=generator,
                **kwargs,
            )
        pag_layers = self._normalize_pag_layers(kwargs.get("pag_applied_layers", "blocks.1"))
        if isinstance(pag_layers, str):
            pag_layers = [pag_layers]
        if pag_scale > 0.0:
            return self._sample_pag(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                pag_layers,
                generator=generator,
                **kwargs,
            )
        if seg_enabled:
            return self._sample_seg(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                generator=generator,
                **{**kwargs, "seg_applied_layers": seg_layers},
            )
        if cfgzero_enabled:
            return self._sample_zero_init(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                generator=generator,
                **{**kwargs, "cfgzero_mode": cfgzero_mode, "cfgzero_zero_steps": cfgzero_zero_steps},
            )
        sd3_lora_active_steps = int(kwargs.get("sd3_lora_active_steps", 0) or 0)
        sd3_lora_scale = float(kwargs.get("sd3_lora_scale", 1.0))
        bridge_causal_active_steps = kwargs.get("bridge_causal_active_steps", -1)
        bridge_causal_active_steps = int(bridge_causal_active_steps if bridge_causal_active_steps is not None else -1)
        if taca_enabled or (
            self.sd3_diffusers_lora
            and (sd3_lora_active_steps > 0 or abs(sd3_lora_scale - 1.0) > 1e-8)
        ):
            return self._sample_taca(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                taca_scale=taca_scale,
                taca_active_steps=taca_active_steps,
                taca_layers=kwargs.get("taca_layers", "all"),
                sd3_lora_active_steps=sd3_lora_active_steps,
                sd3_lora_scale=sd3_lora_scale,
                generator=generator,
            )
        if self.sd3_lora_adapter is not None and (
            sd3_lora_active_steps > 0 or abs(sd3_lora_scale - 1.0) > 1e-8
        ):
            return self._sample_lora_window(
                prompts,
                negative_prompts,
                height,
                width,
                cfg_guidance,
                active_steps=sd3_lora_active_steps,
                lora_scale=sd3_lora_scale,
                bridge_active_steps=bridge_causal_active_steps,
                generator=generator,
            )
        output = self.pipe(
            prompts,
            negative_prompt=negative_prompts,
            width=int(width),
            height=int(height),
            num_inference_steps=self.num_sampling,
            guidance_scale=float(cfg_guidance),
            pag_scale=0.0,
            generator=generator,
        )
        return output.images


def get_solver(method, solver_config=None, device=None, **kwargs):
    method = normalize_sd35_method(method)
    if method not in {"cfg", "ddim", "sd35", "sd35m", "flowmatch", *PROMPT_REINJECTION_METHODS}:
        raise ValueError(f"Unsupported SD3.5M method: {method}")
    return SD35M(solver_config=solver_config, device=device or "cuda", method=method, **kwargs)
