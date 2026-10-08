from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


def _safe_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "__", name).strip("_")


def _resolve_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = 1.0
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1.0 / max(1, base.in_features))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        if float(self.scale) == 0.0:
            return base
        delta = self.lora_up(self.lora_down(x)) * (self.alpha / max(1, self.rank))
        return base + float(self.scale) * delta


class SD3LoRAAdapter(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        target_regex: str = r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_out\.0|to_add_out)$",
    ):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.target_regex = str(target_regex)
        object.__setattr__(self, "transformer", transformer)
        self.name_map: dict[str, str] = {}
        modules: dict[str, LoRALinear] = {}
        pattern = re.compile(self.target_regex)
        for name, module in list(transformer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not pattern.search(name):
                continue
            parent, attr = _resolve_parent(transformer, name)
            wrapped = LoRALinear(module, rank=self.rank, alpha=self.alpha)
            setattr(parent, attr, wrapped)
            safe = _safe_key(name)
            if safe in modules:
                raise ValueError(f"duplicate safe LoRA key for {name}: {safe}")
            modules[safe] = wrapped
            self.name_map[safe] = name
        if not modules:
            raise ValueError(f"No Linear modules matched LoRA target regex: {self.target_regex}")
        self.lora_modules = nn.ModuleDict(modules)

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.lora_modules.values():
            yield module.lora_down.weight
            yield module.lora_up.weight

    def set_scale(self, scale: float) -> None:
        for module in self.lora_modules.values():
            module.scale = float(scale)

    def lora_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            key: {
                "lora_down.weight": module.lora_down.weight.detach().cpu(),
                "lora_up.weight": module.lora_up.weight.detach().cpu(),
            }
            for key, module in self.lora_modules.items()
        }

    def load_lora_state_dict(self, state: dict, strict: bool = True) -> None:
        missing = []
        for key, module in self.lora_modules.items():
            item = state.get(key)
            if item is None:
                missing.append(key)
                continue
            module.lora_down.weight.data.copy_(item["lora_down.weight"].to(module.lora_down.weight.device))
            module.lora_up.weight.data.copy_(item["lora_up.weight"].to(module.lora_up.weight.device))
        extra = sorted(set(state) - set(self.lora_modules))
        if strict and (missing or extra):
            raise RuntimeError(f"LoRA state mismatch: missing={missing} extra={extra}")

    def save_adapter(self, output_dir: Path | str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.lora_state_dict(), output_dir / "adapter.pt")
        config = {
            "adapter_type": "sd3_lora",
            "rank": self.rank,
            "alpha": self.alpha,
            "target_regex": self.target_regex,
            "name_map": self.name_map,
        }
        (output_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_adapter(cls, transformer: nn.Module, adapter_path: Path | str, device=None, dtype=None) -> "SD3LoRAAdapter":
        path = Path(adapter_path)
        if path.is_dir():
            checkpoint_path = path / "adapter.pt"
            config_path = path / "adapter_config.json"
        else:
            checkpoint_path = path
            config_path = path.with_name("adapter_config.json")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing SD3 LoRA adapter checkpoint: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing SD3 LoRA adapter config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        adapter = cls(
            transformer,
            rank=int(config["rank"]),
            alpha=float(config["alpha"]),
            target_regex=str(config["target_regex"]),
        )
        state = torch.load(checkpoint_path, map_location=device or "cpu")
        adapter.load_lora_state_dict(state, strict=True)
        if device is not None or dtype is not None:
            adapter.to(device=device, dtype=dtype)
        adapter.eval()
        return adapter

    def training_stats(self) -> dict[str, float]:
        with torch.no_grad():
            down_norms = [module.lora_down.weight.detach().float().norm().reshape(1) for module in self.lora_modules.values()]
            up_norms = [module.lora_up.weight.detach().float().norm().reshape(1) for module in self.lora_modules.values()]
            stats = {
                "lora_modules": float(len(self.lora_modules)),
                "lora_rank": float(self.rank),
                "lora_alpha": float(self.alpha),
            }
            if down_norms:
                down = torch.cat(down_norms)
                stats["lora_down_norm_mean"] = float(down.mean().item())
                stats["lora_down_norm_max"] = float(down.max().item())
            if up_norms:
                up = torch.cat(up_norms)
                stats["lora_up_norm_mean"] = float(up.mean().item())
                stats["lora_up_norm_max"] = float(up.max().item())
            return stats


class VisualSwapJointAttnProcessor2_0(nn.Module):
    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("VisualSwapJointAttnProcessor2_0 requires PyTorch SDPA.")
        self.partner_indices: torch.Tensor | None = None
        self.scale: float = 0.0
        self.text_anchor_tag: str | None = None
        self.text_anchor_detach: bool = True
        self.text_anchor_records: dict[str, torch.Tensor] = {}

    def set_visual_swap(self, partner_indices: torch.Tensor | None, scale: float = 0.0) -> None:
        self.partner_indices = partner_indices
        self.scale = float(scale)

    def set_text_anchor_record(self, tag: str | None, *, detach: bool = True) -> None:
        self.text_anchor_tag = None if tag is None else str(tag)
        self.text_anchor_detach = bool(detach)
        if self.text_anchor_tag is not None:
            self.text_anchor_records.pop(self.text_anchor_tag, None)

    def clear_text_anchor_records(self) -> None:
        self.text_anchor_records.clear()
        self.text_anchor_tag = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
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

        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        if (
            encoder_hidden_states is not None
            and self.partner_indices is not None
            and abs(float(self.scale)) > 0.0
            and image_seq_len > 0
        ):
            attended = self._apply_text_image_visual_swap(
                attended,
                query,
                key,
                value,
                image_seq_len=image_seq_len,
                target_start=image_seq_len,
                target_end=key.shape[2],
                partner_indices=self.partner_indices,
                scale=float(self.scale),
            )

        hidden_states = attended.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            if self.text_anchor_tag is not None:
                record = encoder_hidden_states.detach() if self.text_anchor_detach else encoder_hidden_states
                self.text_anchor_records[self.text_anchor_tag] = record
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
        partner_indices: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        partner = partner_indices.to(device=query.device, dtype=torch.long)
        if partner.numel() != query.shape[0]:
            raise ValueError(f"visual-swap partner count {partner.numel()} does not match batch size {query.shape[0]}")
        key_swapped = key.clone()
        value_swapped = value.clone()
        key_swapped[:, :, :image_seq_len] = key.index_select(0, partner)[:, :, :image_seq_len]
        value_swapped[:, :, :image_seq_len] = value.index_select(0, partner)[:, :, :image_seq_len]
        query_target = query[:, :, target_start:target_end]
        original = self._value_slice_contribution(query_target, key, value, 0, image_seq_len)
        swapped = self._value_slice_contribution(query_target, key_swapped, value_swapped, 0, image_seq_len)
        updated = attended.clone()
        updated[:, :, target_start:target_end] = updated[:, :, target_start:target_end] + float(scale) * (swapped - original)
        return updated


def install_visual_swap_processors(transformer: nn.Module) -> list[VisualSwapJointAttnProcessor2_0]:
    processors: list[VisualSwapJointAttnProcessor2_0] = []
    for block in transformer.transformer_blocks:
        processor = VisualSwapJointAttnProcessor2_0()
        if hasattr(block.attn, "set_processor"):
            block.attn.set_processor(processor)
        else:
            block.attn.processor = processor
        processors.append(processor)
    return processors


def set_visual_swap(processors: Iterable[VisualSwapJointAttnProcessor2_0], partner_indices, scale: float) -> None:
    for processor in processors:
        processor.set_visual_swap(partner_indices, scale)


def set_text_anchor_record(
    processors: Iterable[VisualSwapJointAttnProcessor2_0],
    tag: str | None,
    *,
    detach: bool = True,
) -> None:
    for processor in processors:
        processor.set_text_anchor_record(tag, detach=detach)


def clear_text_anchor_records(processors: Iterable[VisualSwapJointAttnProcessor2_0]) -> None:
    for processor in processors:
        processor.clear_text_anchor_records()
