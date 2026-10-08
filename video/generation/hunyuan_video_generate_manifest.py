#!/usr/bin/env python3
"""Generate a deterministic HunyuanVideo-1.5 T2V or I2V manifest.

Two torchrun layouts are supported:

``sequence``
    All ranks cooperate on the same prompt with sequence parallelism.  Rank
    zero writes the video and progress metadata.

``data``
    Sequence parallelism is disabled.  Every rank owns one GPU, loads one
    complete pipeline, and generates the manifest rows whose global index is
    congruent to that rank modulo world size.  Sampling then has no cross-GPU
    attention communication; ranks synchronize only around setup/finalization.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(
    os.environ.get("SFG_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
HUNYUAN_ROOT = Path(
    os.environ.get(
        "HUNYUAN_SOURCE_ROOT", REPO_ROOT / "third_party" / "HunyuanVideo-1.5"
    )
).resolve()
sys.path.insert(0, str(HUNYUAN_ROOT))

import einops
import imageio
import torch
import torch.distributed as dist

from hyvideo.commons import maybe_fallback_attn_mode
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline

import hunyuan_video_sfg as hunyuan_video_sfg_module
import hunyuan_video_cfg_zero_init as hunyuan_video_cfg_zero_init_module
import hunyuan_video_cfg_zero_star as hunyuan_video_cfg_zero_star_module
import hunyuan_video_s2 as hunyuan_video_s2_module
import hunyuan_video_stg as hunyuan_video_stg_module
from hunyuan_video_cfg_zero_init import (
    cfg_zero_init_runtime_stats,
    install_hunyuan_cfg_zero_init,
)
from hunyuan_video_cfg_zero_star import (
    cfg_zero_star_runtime_stats,
    install_hunyuan_cfg_zero_star,
)
from hunyuan_video_sfg import (
    install_hunyuan_explicit_sfg,
    sfg_runtime_stats,
)
from hunyuan_video_s2 import install_hunyuan_s2_guidance, s2_runtime_stats
from hunyuan_video_stg import install_hunyuan_stg, stg_runtime_stats


def iter_jsonl(path: Path, *, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("prompt", "seed", "relative_output"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number}: missing {key!r}")
            if task == "i2v":
                for key in ("reference_image", "reference_image_sha256"):
                    if key not in row:
                        raise ValueError(f"{path}:{line_number}: missing {key!r}")
                digest = str(row["reference_image_sha256"]).lower()
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError(
                        f"{path}:{line_number}: invalid reference_image_sha256"
                    )
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transformer_weight_inventory(transformer_dir: Path) -> list[dict[str, Any]]:
    weight_paths = sorted(transformer_dir.glob("*.safetensors"))
    if not weight_paths:
        raise FileNotFoundError(f"No safetensors weights found in {transformer_dir}")
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in weight_paths
    ]


def save_video(video: torch.Tensor, path: Path, fps: int) -> dict[str, int]:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size one, got {tuple(video.shape)}")
        video = video[0]
    frames = (video * 255).clamp(0, 255).to(torch.uint8)
    frames = einops.rearrange(frames, "c f h w -> f h w c").cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)
    return {
        "frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "fps": int(fps),
    }


def resolve_reference_image(
    raw_path: str,
    *,
    manifest_path: Path,
    image_root: Path | None,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        base = image_root.resolve() if image_root is not None else manifest_path.parent
        path = base / path
    return path.resolve()


def validate_i2v_references(
    rows: list[dict[str, Any]],
    *,
    manifest_path: Path,
    image_root: Path | None,
) -> tuple[list[Path], int]:
    resolved: list[Path] = []
    digest_cache: dict[Path, str] = {}
    for index, row in enumerate(rows):
        path = resolve_reference_image(
            str(row["reference_image"]),
            manifest_path=manifest_path,
            image_root=image_root,
        )
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"I2V manifest row {index}: {path}")
        actual_digest = digest_cache.get(path)
        if actual_digest is None:
            actual_digest = sha256_file(path)
            digest_cache[path] = actual_digest
        expected_digest = str(row["reference_image_sha256"]).lower()
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"I2V reference hash mismatch at row {index}: {path}; "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        resolved.append(path)
    return resolved, len(digest_cache)


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def broadcast_bool(value: bool) -> bool:
    if not dist.is_initialized():
        return value
    payload = [value if rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return bool(payload[0])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic HunyuanVideo-1.5 T2V or I2V videos from JSONL"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("t2v", "i2v"),
        default="t2v",
        help="Generation task. I2V requires reference_image and its SHA256 per row.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help=(
            "Optional root for relative I2V reference_image paths. Without it, "
            "paths are resolved relative to the manifest."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models" / "HunyuanVideo-1.5",
    )
    parser.add_argument(
        "--precomputed-transformer-sha256",
        help=(
            "Optional SHA256 computed by the audited launcher for the single I2V "
            "safetensors file. Avoids hashing the same local model again in smoke/main runs."
        ),
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument(
        "--guidance-variant",
        choices=(
            "standard",
            "cfg_zero_init",
            "cfg_zero_star",
            "s2_noguidance",
            "stg_noguidance",
            "stg_hv15_noguidance",
        ),
        default="standard",
        help=(
            "standard preserves the native pipeline; cfg_zero_init zeros only "
            "the first raw CFG velocity; cfg_zero_star applies the complete "
            "official optimized-scale plus zero-init method; s2_noguidance "
            "applies stochastic block-dropping self-guidance; stg_noguidance "
            "preserves the historical literal STG-default port; "
            "stg_hv15_noguidance is the explicitly adapted HV1.5 Base arm; "
            "all self-guidance variants disable native CFG"
        ),
    )
    parser.add_argument(
        "--zero-init-steps",
        type=int,
        default=0,
        help="Number of initial denoising velocities to set exactly to zero",
    )
    parser.add_argument("--s2-omega", type=float)
    parser.add_argument("--s2-drop-count", type=int)
    parser.add_argument("--s2-active-start", type=int)
    parser.add_argument("--s2-active-end", type=int)
    parser.add_argument("--s2-controller-seed", type=int)
    parser.add_argument("--stg-scale", type=float)
    parser.add_argument("--stg-block-index", type=int)
    parser.add_argument("--stg-active-start", type=int)
    parser.add_argument("--stg-active-end", type=int)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        help="Sampling steps. Defaults to 25 for I2V and 50 for legacy T2V calls.",
    )
    parser.add_argument("--video-length", type=int, default=121)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--parallel-mode",
        choices=("sequence", "data"),
        default="sequence",
        help=(
            "sequence: all torchrun ranks cooperate on each video; "
            "data: one independent single-GPU pipeline per rank"
        ),
    )
    parser.add_argument(
        "--sfg-s-u",
        type=float,
        default=0.0,
        help=(
            "Enable explicit SFGS with positive u on the image-query <- text-value "
            "bridge. For example, 0.1 constructs a 0.9x bridge branch."
        ),
    )
    parser.add_argument(
        "--sfg-x-u",
        type=float,
        default=0.0,
        help=(
            "Enable explicit SFGX on the text-query <- image-value bridge. "
            "For example, u=-0.1 constructs a 1.1x branch and u=+0.1 "
            "constructs a 0.9x branch."
        ),
    )
    parser.add_argument(
        "--sfg_x-omega",
        type=float,
        default=1.0,
        help="Multiplier in D_clean + omega * (D_clean - D_counterfactual)",
    )
    parser.add_argument(
        "--sfg_x-layers",
        default="all",
        help="Comma-separated zero-based double-block indices, or all",
    )
    parser.add_argument(
        "--bridge-active-steps",
        "--sfg_x-active-steps",
        dest="bridge_active_steps",
        type=int,
        default=0,
        help=(
            "Apply the SFGS/SFGX counterfactual only on the first N denoising "
            "steps; 0 applies it on every step."
        ),
    )
    parser.add_argument(
        "--sfg_x-condition-scope",
        choices=("joint_encoder", "text_only"),
        default="joint_encoder",
        help=(
            "joint_encoder preserves full I2V SFG over image-semantic+prompt "
            "encoder tokens; text_only scales only video<->prompt-text edges "
            "and leaves the leading image-semantic tokens unscaled"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offloading", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_inference_steps is None:
        args.num_inference_steps = 25 if args.task == "i2v" else 50
    if args.task == "i2v" and args.num_inference_steps != 25:
        raise ValueError(
            "the fixed HunyuanVideo-1.5 I2V protocol requires exactly 25 sampling steps"
        )
    if args.guidance_scale < 1:
        raise ValueError("guidance scale must be >= 1; use 1 for no-CFG")
    cfg_enabled = args.guidance_scale > 1
    # This fixed manifest runner only exposes full-run native CFG.  Keeping the
    # active window explicit prevents a future pipeline default change from
    # silently turning a 25-step CFG baseline into a partial-CFG run.
    cfg_active_steps = args.num_inference_steps if cfg_enabled else 0
    sfg_x_enabled = (
        args.sfg_s_u != 0.0 or args.sfg_x_u != 0.0
    )
    bridge_active_steps = (
        (args.bridge_active_steps or args.num_inference_steps) if sfg_x_enabled else 0
    )
    if sfg_x_enabled:
        if args.guidance_scale != 1.0:
            raise ValueError(
                "the Hunyuan SFGS/SFGX arm is standalone and requires guidance_scale=1"
            )
        if not 0.0 <= args.sfg_s_u <= 1.0:
            raise ValueError("SFGS requires --sfg-s-u in [0, 1]")
        if not -1.0 <= args.sfg_x_u <= 1.0:
            raise ValueError("SFGX requires --sfg-x-u in [-1, 1]")
        if args.sfg_x_omega < 0.0:
            raise ValueError("--sfg_x-omega must be >= 0")
        if not 1 <= bridge_active_steps <= args.num_inference_steps:
            raise ValueError(
                "--bridge-active-steps must be in [1, num_inference_steps]"
            )
        if args.sfg_x_condition_scope == "text_only" and args.task != "i2v":
            raise ValueError("--sfg_x-condition-scope text_only is only valid for I2V")
    elif args.sfg_x_condition_scope != "joint_encoder":
        raise ValueError(
            "--sfg_x-condition-scope is meaningful only when SFGS/SFGX is enabled"
        )

    cfg_zero_init_enabled = args.guidance_variant == "cfg_zero_init"
    cfg_zero_star_enabled = args.guidance_variant == "cfg_zero_star"
    s2_enabled = args.guidance_variant == "s2_noguidance"
    stg_literal_enabled = args.guidance_variant == "stg_noguidance"
    stg_hv15_enabled = args.guidance_variant == "stg_hv15_noguidance"
    stg_enabled = stg_literal_enabled or stg_hv15_enabled
    if cfg_zero_init_enabled or cfg_zero_star_enabled:
        if not cfg_enabled:
            raise ValueError(
                "CFG zero-init/CFG-Zero* requires native CFG guidance_scale > 1"
            )
        if sfg_x_enabled:
            raise ValueError("CFG zero-init/CFG-Zero* cannot be combined with SFGS/SFGX")
        if not 1 <= args.zero_init_steps <= args.num_inference_steps:
            raise ValueError(
                "--zero-init-steps must be in [1, num_inference_steps]"
            )
    elif args.zero_init_steps != 0:
        raise ValueError(
            "--zero-init-steps is valid only for cfg_zero_init or cfg_zero_star"
        )

    s2_values = (
        args.s2_omega,
        args.s2_drop_count,
        args.s2_active_start,
        args.s2_active_end,
        args.s2_controller_seed,
    )
    if s2_enabled:
        if cfg_enabled or args.guidance_scale != 1.0:
            raise ValueError("S2 comparison is pinned to NoGuidance (scale=1)")
        if sfg_x_enabled:
            raise ValueError("S2 cannot be combined with SFGS/SFGX")
        args.s2_omega = 0.25 if args.s2_omega is None else args.s2_omega
        args.s2_drop_count = (
            5 if args.s2_drop_count is None else args.s2_drop_count
        )
        # Exactly 20/25 central steps.  With an odd number of total steps, the
        # unavoidable extra excluded step is placed at the high-noise start.
        args.s2_active_start = (
            3 if args.s2_active_start is None else args.s2_active_start
        )
        args.s2_active_end = (
            23 if args.s2_active_end is None else args.s2_active_end
        )
        args.s2_controller_seed = (
            20250818
            if args.s2_controller_seed is None
            else args.s2_controller_seed
        )
        if args.s2_omega < 0.0:
            raise ValueError("--s2-omega must be non-negative")
        if not 1 <= args.s2_drop_count <= 53:
            raise ValueError("--s2-drop-count must be in [1, 53]")
        if not (
            0
            <= args.s2_active_start
            < args.s2_active_end
            <= args.num_inference_steps
        ):
            raise ValueError("invalid S2 active-step interval")
    elif any(value is not None for value in s2_values):
        raise ValueError("--s2-* arguments are valid only for s2_noguidance")

    stg_values = (
        args.stg_scale,
        args.stg_block_index,
        args.stg_active_start,
        args.stg_active_end,
    )
    if stg_enabled:
        if args.task != "t2v":
            raise ValueError("the Hunyuan STG ports are pinned to T2V")
        if cfg_enabled or args.guidance_scale != 1.0:
            raise ValueError("STG comparison is pinned to NoGuidance (scale=1)")
        if sfg_x_enabled:
            raise ValueError("STG cannot be combined with SFGS/SFGX")
        if stg_literal_enabled:
            args.stg_scale = 1.0 if args.stg_scale is None else args.stg_scale
            args.stg_block_index = (
                2 if args.stg_block_index is None else args.stg_block_index
            )
            args.stg_active_start = (
                0 if args.stg_active_start is None else args.stg_active_start
            )
            args.stg_active_end = (
                args.num_inference_steps
                if args.stg_active_end is None
                else args.stg_active_end
            )
            if (
                args.stg_scale != 1.0
                or args.stg_block_index != 2
                or args.stg_active_start != 0
                or args.stg_active_end != args.num_inference_steps
            ):
                raise ValueError(
                    "the historical literal STG port requires scale=1.0, "
                    "block index 2, and all denoising steps"
                )
        else:
            # The literal scale=1 port produced a skip residual whose RMS was
            # 0.97--1.40x the clean prediction on the frozen Quality7 run.
            # Scale 0.15 bounds the observed correction near 0.15--0.21x and
            # the early seven-step window avoids corrupting late refinement.
            args.stg_scale = 0.15 if args.stg_scale is None else args.stg_scale
            args.stg_block_index = (
                2 if args.stg_block_index is None else args.stg_block_index
            )
            args.stg_active_start = (
                0 if args.stg_active_start is None else args.stg_active_start
            )
            args.stg_active_end = (
                7 if args.stg_active_end is None else args.stg_active_end
            )
        if args.stg_scale < 0.0:
            raise ValueError("--stg-scale must be non-negative")
        if not 0 <= args.stg_block_index < 54:
            raise ValueError("--stg-block-index must be in [0, 53]")
        if not (
            0
            <= args.stg_active_start
            < args.stg_active_end
            <= args.num_inference_steps
        ):
            raise ValueError("invalid STG active-step interval")
    elif any(value is not None for value in stg_values):
        raise ValueError(
            "--stg-* arguments are valid only for stg_noguidance or "
            "stg_hv15_noguidance"
        )
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    sequence_parallel_size = world_size if args.parallel_mode == "sequence" else 1
    initialize_parallel_state(sp=sequence_parallel_size)

    manifest_path = args.manifest.resolve()
    rows = iter_jsonl(manifest_path, task=args.task)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    indexed_rows = list(enumerate(rows))
    if args.parallel_mode == "data":
        owned_rows = [item for item in indexed_rows if item[0] % world_size == rank()]
    else:
        owned_rows = indexed_rows

    reference_paths: list[Path | None] = [None] * len(rows)
    unique_reference_image_count = 0
    if args.task == "i2v":
        validation_payload: list[Any] = [None, None, None]
        if rank() == 0:
            try:
                checked_paths, unique_reference_image_count = validate_i2v_references(
                    rows,
                    manifest_path=manifest_path,
                    image_root=args.image_root,
                )
                validation_payload = [
                    [str(path) for path in checked_paths],
                    unique_reference_image_count,
                    None,
                ]
            except Exception as error:  # broadcast the same fatal error to every rank
                validation_payload = [None, None, f"{type(error).__name__}: {error}"]
        if dist.is_initialized():
            dist.broadcast_object_list(validation_payload, src=0)
        if validation_payload[2] is not None:
            raise RuntimeError(str(validation_payload[2]))
        reference_paths = [Path(value) for value in validation_payload[0]]
        unique_reference_image_count = int(validation_payload[1])

    transformer_version = f"480p_{args.task}"
    transformer_dir = args.model_path.resolve() / "transformer" / transformer_version
    if not transformer_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {transformer_version} weights: {transformer_dir}"
        )
    transformer_config_path = transformer_dir / "config.json"
    transformer_config = json.loads(transformer_config_path.read_text(encoding="utf-8"))
    if transformer_config.get("ideal_task") != args.task:
        raise RuntimeError(
            f"Transformer task mismatch in {transformer_config_path}: "
            f"expected={args.task!r}, actual={transformer_config.get('ideal_task')!r}"
        )
    if int(transformer_config.get("mm_double_blocks_depth", -1)) != 54 or int(
        transformer_config.get("mm_single_blocks_depth", -1)
    ) != 0:
        raise RuntimeError(
            "Joint SFG is pinned to the 54-double/0-single HunyuanVideo-1.5 architecture"
        )
    model_config_path = args.model_path.resolve() / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    vision_num_semantic_tokens = int(model_config["vision_num_semantic_tokens"])
    if args.task == "i2v" and vision_num_semantic_tokens != 729:
        raise RuntimeError(
            "I2V Joint SFG is pinned to 729 reference-image semantic tokens; "
            f"found {vision_num_semantic_tokens}"
        )

    if args.task == "t2v":
        t2v_weight_path = transformer_dir / "diffusion_pytorch_model.safetensors"
        if not t2v_weight_path.is_file():
            raise FileNotFoundError(t2v_weight_path)
        transformer_weights = [
            {
                "name": t2v_weight_path.name,
                "bytes": t2v_weight_path.stat().st_size,
                "sha256": "71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86",
            }
        ]
    else:
        precomputed_digest = args.precomputed_transformer_sha256
        if precomputed_digest is not None:
            precomputed_digest = precomputed_digest.strip().lower()
            if len(precomputed_digest) != 64 or any(
                char not in "0123456789abcdef" for char in precomputed_digest
            ):
                raise ValueError("Invalid --precomputed-transformer-sha256")
            weight_paths = sorted(transformer_dir.glob("*.safetensors"))
            if len(weight_paths) != 1:
                raise RuntimeError(
                    "Precomputed transformer SHA256 requires exactly one safetensors file; "
                    f"found {len(weight_paths)}"
                )
            transformer_weights = [
                {
                    "name": weight_paths[0].name,
                    "bytes": weight_paths[0].stat().st_size,
                    "sha256": precomputed_digest,
                }
            ]
        else:
            weight_inventory_payload: list[Any] = [None, None]
            if rank() == 0:
                try:
                    weight_inventory_payload = [
                        transformer_weight_inventory(transformer_dir),
                        None,
                    ]
                except Exception as error:
                    weight_inventory_payload = [None, f"{type(error).__name__}: {error}"]
            if dist.is_initialized():
                dist.broadcast_object_list(weight_inventory_payload, src=0)
            if weight_inventory_payload[1] is not None:
                raise RuntimeError(str(weight_inventory_payload[1]))
            transformer_weights = list(weight_inventory_payload[0])

    output_dir = args.output_dir.resolve()
    progress_path = (
        output_dir / f"generation_progress.rank{rank():02d}.jsonl"
        if args.parallel_mode == "data"
        else output_dir / "generation_progress.jsonl"
    )
    try:
        flash_attn_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        flash_attn_version = None
    run_config = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": args.method,
        "task": args.task,
        "guidance_variant": args.guidance_variant,
        "guidance_scale": args.guidance_scale,
        "cfg_enabled": cfg_enabled,
        "cfg_active_steps": cfg_active_steps if cfg_enabled else None,
        "cfg_active_fraction": (
            cfg_active_steps / args.num_inference_steps if cfg_enabled else None
        ),
        "cfg_condition_scope": (
            "text_only_shared_reference" if cfg_enabled and args.task == "i2v" else None
        ),
        "cfg_negative_prompt": "" if cfg_enabled else None,
        "cfg_zero_init_enabled": cfg_zero_init_enabled,
        "cfg_zero_star_enabled": cfg_zero_star_enabled,
        "zero_init_steps": (
            args.zero_init_steps
            if cfg_zero_init_enabled or cfg_zero_star_enabled
            else None
        ),
        "zero_init_fraction": (
            args.zero_init_steps / args.num_inference_steps
            if cfg_zero_init_enabled or cfg_zero_star_enabled
            else None
        ),
        "cfg_zero_optimized_scale_projection_enabled": cfg_zero_star_enabled,
        "cfg_zero_star_formula": (
            "alpha*v_uncond+w*(v_cond-alpha*v_uncond)"
            if cfg_zero_star_enabled
            else None
        ),
        "cfg_zero_star_alpha_formula": (
            "<v_cond,v_uncond>/(||v_uncond||^2+1e-8)"
            if cfg_zero_star_enabled
            else None
        ),
        "cfg_zero_star_epsilon": 1e-8 if cfg_zero_star_enabled else None,
        "cfg_zero_star_official_repository": (
            "https://github.com/WeichenFan/CFG-Zero-star"
            if cfg_zero_star_enabled
            else None
        ),
        "cfg_zero_star_official_commit": (
            "3162be1fba5dd0129ac8423ad6919d928f420a8d"
            if cfg_zero_star_enabled
            else None
        ),
        "s2_enabled": s2_enabled,
        "s2_formula": (
            "D_clean + omega * (D_clean - D_sub)" if s2_enabled else None
        ),
        "s2_cfg_lambda": 1.0 if s2_enabled else None,
        "s2_unconditional_branch_enabled": False if s2_enabled else None,
        "s2_omega": args.s2_omega if s2_enabled else None,
        "s2_drop_count": args.s2_drop_count if s2_enabled else None,
        "s2_excluded_block_indices": [0] if s2_enabled else None,
        "s2_active_start": args.s2_active_start if s2_enabled else None,
        "s2_active_end_exclusive": args.s2_active_end if s2_enabled else None,
        "s2_active_steps": (
            args.s2_active_end - args.s2_active_start if s2_enabled else None
        ),
        "s2_controller_seed": args.s2_controller_seed if s2_enabled else None,
        "stg_enabled": stg_enabled,
        "stg_formula": (
            "D_clean + scale * (D_clean - D_perturbed)"
            if stg_enabled
            else None
        ),
        "stg_unconditional_branch_enabled": False if stg_enabled else None,
        "stg_guidance_rescaling_enabled": False if stg_enabled else None,
        "stg_scale": args.stg_scale if stg_enabled else None,
        "stg_block_indices": [args.stg_block_index] if stg_enabled else None,
        "stg_active_start": args.stg_active_start if stg_enabled else None,
        "stg_active_end_exclusive": args.stg_active_end if stg_enabled else None,
        "stg_active_steps": (
            args.stg_active_end - args.stg_active_start if stg_enabled else None
        ),
        "stg_port_kind": (
            "hv15_base_adapted"
            if stg_hv15_enabled
            else "literal_default_port"
            if stg_enabled
            else None
        ),
        "stg_reference_architecture": (
            "20_dual_plus_40_single_cfg_distilled" if stg_enabled else None
        ),
        "stg_target_architecture": "54_double_base" if stg_enabled else None,
        "stg_block_mapping": (
            "zero_based_global_depth_position" if stg_enabled else None
        ),
        "stg_official_repository": (
            "https://github.com/junhahyung/STGuidance"
            if stg_enabled
            else None
        ),
        "stg_official_commit": (
            "d9e7be5dadfc8d0f53855b2a56e478a5e64b2ca4"
            if stg_enabled
            else None
        ),
        "stg_official_hunyuan_source_sha256": (
            "f0c911bddb818d050b45e801f2a14b432ac7b588ed5af3b3c2c2ede2b1715c99"
            if stg_enabled
            else None
        ),
        "bridge_guidance_enabled": sfg_x_enabled,
        "sfg_x_enabled": sfg_x_enabled,
        "sfg_x_mode": "explicit" if sfg_x_enabled else "none",
        "sfg_x_direction": (
            "both"
            if args.sfg_s_u and args.sfg_x_u
            else "image_to_text"
            if args.sfg_s_u
            else "text_to_image"
            if args.sfg_x_u
            else None
        ),
        "sfg_s_u": args.sfg_s_u if sfg_x_enabled else None,
        "sfg_s_bridge_multiplier": (
            1.0 - args.sfg_s_u if sfg_x_enabled else None
        ),
        "sfg_x_u": args.sfg_x_u if sfg_x_enabled else None,
        "sfg_x_strong_bridge_multiplier": (
            1.0 - args.sfg_x_u if sfg_x_enabled else None
        ),
        "sfg_x_omega": args.sfg_x_omega if sfg_x_enabled else None,
        "sfg_x_layers": args.sfg_x_layers if sfg_x_enabled else None,
        "bridge_active_steps": bridge_active_steps if sfg_x_enabled else None,
        "bridge_active_fraction": (
            bridge_active_steps / args.num_inference_steps if sfg_x_enabled else None
        ),
        "sfg_x_branch_mode": "sequential" if sfg_x_enabled else None,
        "sfg_x_formula": (
            "D_clean + omega * (D_clean - D_counterfactual)" if sfg_x_enabled else None
        ),
        "condition_scope": args.sfg_x_condition_scope if sfg_x_enabled else None,
        "joint_condition_definition": (
            "reference-image semantic tokens and prompt tokens receive identical "
            "SFGS/SFGX value scaling"
            if sfg_x_enabled
            and args.task == "i2v"
            and args.sfg_x_condition_scope == "joint_encoder"
            else None
        ),
        "text_only_condition_definition": (
            "only video<->prompt-text value bridges are scaled; the leading "
            "reference-image semantic tokens remain outside attention scaling"
            if sfg_x_enabled
            and args.task == "i2v"
            and args.sfg_x_condition_scope == "text_only"
            else None
        ),
        "vision_num_semantic_tokens": (
            vision_num_semantic_tokens if args.task == "i2v" else None
        ),
        "flow_shift": args.flow_shift,
        "num_inference_steps": args.num_inference_steps,
        "video_length": args.video_length,
        "aspect_ratio": args.aspect_ratio,
        "fps": args.fps,
        "prompt_rewrite": False,
        "super_resolution": False,
        "vae_tile_parallelism": args.parallel_mode == "sequence",
        "cache": False,
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "flash_attn_version": flash_attn_version,
        "attention_mode_effective": maybe_fallback_attn_mode("flash"),
        "transformer_version": transformer_version,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "model_path": str(args.model_path.resolve()),
        "hunyuan_source_commit": "60783e704160023913bee78f0b47036d393d4dfa",
        "hunyuan_pipeline_sha256": sha256_file(
            HUNYUAN_ROOT / "hyvideo" / "pipelines" / "hunyuan_video_pipeline.py"
        ),
        "hunyuan_transformer_source_sha256": sha256_file(
            HUNYUAN_ROOT
            / "hyvideo"
            / "models"
            / "transformers"
            / "hunyuanvideo_1_5_transformer.py"
        ),
        "sfg_runtime_sha256": sha256_file(
            Path(hunyuan_video_sfg_module.__file__).resolve()
        ),
        "cfg_zero_init_runtime_sha256": (
            sha256_file(Path(hunyuan_video_cfg_zero_init_module.__file__).resolve())
            if cfg_zero_init_enabled
            else None
        ),
        "cfg_zero_star_runtime_sha256": (
            sha256_file(Path(hunyuan_video_cfg_zero_star_module.__file__).resolve())
            if cfg_zero_star_enabled
            else None
        ),
        "s2_runtime_sha256": (
            sha256_file(Path(hunyuan_video_s2_module.__file__).resolve())
            if s2_enabled
            else None
        ),
        "stg_runtime_sha256": (
            sha256_file(Path(hunyuan_video_stg_module.__file__).resolve())
            if stg_enabled
            else None
        ),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "transformer_sha256": (
            transformer_weights[0]["sha256"]
            if len(transformer_weights) == 1
            else None
        ),
        "transformer_weight_files": transformer_weights,
        "transformer_config_sha256": sha256_file(transformer_config_path),
        "reference_image_root": (
            str(args.image_root.resolve()) if args.image_root is not None else None
        ),
        "unique_reference_image_count": unique_reference_image_count,
        "world_size": world_size,
        "parallel_mode": args.parallel_mode,
        "sequence_parallel_size": sequence_parallel_size,
        "data_parallel_size": world_size if args.parallel_mode == "data" else 1,
        "requested_video_count": len(rows),
    }
    if rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "generation_config.json", run_config)
    if dist.is_initialized():
        dist.barrier()

    device = torch.device("cpu" if args.offloading else "cuda")
    transformer_init_device = device
    pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=str(args.model_path.resolve()),
        transformer_version=transformer_version,
        create_sr_pipeline=False,
        transformer_dtype=torch.bfloat16,
        device=device,
        transformer_init_device=transformer_init_device,
    )
    pipeline.apply_infer_optimization(
        infer_state=None,
        enable_offloading=args.offloading,
        enable_group_offloading=args.offloading,
        overlap_group_offloading=False,
    )

    # Count logical denoising-step invocations independently of SFGX's internal
    # branch forwards.  With guidance_scale=1 and SFGX disabled this must be
    # exactly one transformer evaluation per sampling step.
    transformer_step_invocations = 0
    i2v_task_validation_calls = 0
    i2v_observed_vision_tokens: int | None = None
    i2v_nonzero_reference_semantics = False
    i2v_cfg_shared_reference_semantics = True
    i2v_cfg_shared_model_input = True

    def count_transformer_step(_module, _inputs, forward_kwargs):
        nonlocal transformer_step_invocations
        nonlocal i2v_task_validation_calls
        nonlocal i2v_observed_vision_tokens
        nonlocal i2v_nonzero_reference_semantics
        nonlocal i2v_cfg_shared_reference_semantics
        nonlocal i2v_cfg_shared_model_input
        transformer_step_invocations += 1
        if args.task != "i2v":
            return
        actual_task = str(forward_kwargs.get("mask_type", "")).strip().lower()
        if actual_task != "i2v":
            raise RuntimeError(
                f"I2V generator invoked transformer with mask_type={actual_task!r}"
            )
        vision_states = forward_kwargs.get("vision_states")
        if not isinstance(vision_states, torch.Tensor):
            raise RuntimeError("I2V generator requires tensor vision_states")
        token_count = int(vision_states.shape[1])
        if token_count != vision_num_semantic_tokens:
            raise RuntimeError(
                "I2V reference semantic-token mismatch: "
                f"expected={vision_num_semantic_tokens}, actual={token_count}"
            )
        # The pipeline source is pinned by SHA256; one exact equality check per
        # rank proves the packed I2V CFG layout without synchronizing every step.
        if cfg_enabled and i2v_task_validation_calls == 0:
            hidden_states = _inputs[0] if _inputs else None
            if (
                vision_states.shape[0] != 2
                or not torch.equal(vision_states[:1], vision_states[1:])
            ):
                i2v_cfg_shared_reference_semantics = False
                raise RuntimeError(
                    "I2V CFG unconditional/conditional branches must share the exact "
                    "same reference-image semantic tokens"
                )
            if (
                not isinstance(hidden_states, torch.Tensor)
                or hidden_states.shape[0] != 2
                or not torch.equal(hidden_states[:1], hidden_states[1:])
            ):
                i2v_cfg_shared_model_input = False
                raise RuntimeError(
                    "I2V CFG unconditional/conditional branches must share the exact "
                    "same noisy latent and reference-image latent condition"
                )
        if i2v_task_validation_calls == 0:
            if bool(torch.all(vision_states == 0).item()):
                raise RuntimeError("I2V reference-image semantic stream is all zero")
            i2v_nonzero_reference_semantics = True
        i2v_observed_vision_tokens = token_count
        i2v_task_validation_calls += 1

    nfe_hook = pipeline.transformer.register_forward_pre_hook(
        count_transformer_step, with_kwargs=True
    )
    guidance_controller = None
    if sfg_x_enabled:
        guidance_controller = install_hunyuan_explicit_sfg(
            pipeline.transformer,
            sfg_s_u=args.sfg_s_u,
            sfg_strength_u=args.sfg_x_u,
            omega=args.sfg_x_omega,
            layers=args.sfg_x_layers,
            total_steps=args.num_inference_steps,
            active_steps=bridge_active_steps,
            expected_task=args.task,
            expected_vision_tokens=(
                vision_num_semantic_tokens if args.task == "i2v" else None
            ),
            condition_scope=args.sfg_x_condition_scope,
        )
        if rank() == 0:
            print(
                "SFGX installed: "
                + json.dumps(guidance_controller.config.to_dict(), sort_keys=True),
                flush=True,
            )
    elif cfg_zero_init_enabled:
        guidance_controller = install_hunyuan_cfg_zero_init(
            pipeline.transformer,
            total_steps=args.num_inference_steps,
            zero_init_steps=args.zero_init_steps,
            expected_task=args.task,
        )
        if rank() == 0:
            print(
                "CFG zero-init installed: "
                + json.dumps(guidance_controller.config.to_dict(), sort_keys=True),
                flush=True,
            )
    elif cfg_zero_star_enabled:
        guidance_controller = install_hunyuan_cfg_zero_star(
            pipeline.transformer,
            total_steps=args.num_inference_steps,
            zero_init_steps=args.zero_init_steps,
            guidance_scale=args.guidance_scale,
            expected_task=args.task,
        )
        if rank() == 0:
            print(
                "Official CFG-Zero* installed: "
                + json.dumps(guidance_controller.config.to_dict(), sort_keys=True),
                flush=True,
            )
    elif s2_enabled:
        guidance_controller = install_hunyuan_s2_guidance(
            pipeline.transformer,
            omega=args.s2_omega,
            total_steps=args.num_inference_steps,
            active_start=args.s2_active_start,
            active_end=args.s2_active_end,
            drop_count=args.s2_drop_count,
            controller_seed=args.s2_controller_seed,
            expected_task=args.task,
        )
        if rank() == 0:
            print(
                "S2 NoGuidance installed: "
                + json.dumps(guidance_controller.config.to_dict(), sort_keys=True),
                flush=True,
            )
    elif stg_enabled:
        guidance_controller = install_hunyuan_stg(
            pipeline.transformer,
            scale=args.stg_scale,
            total_steps=args.num_inference_steps,
            active_start=args.stg_active_start,
            active_end=args.stg_active_end,
            block_indices=(args.stg_block_index,),
            port_kind=(
                "hv15_base_adapted"
                if stg_hv15_enabled
                else "literal_default_port"
            ),
            expected_task=args.task,
        )
        if rank() == 0:
            print(
                "Auditable STG installed: "
                + json.dumps(guidance_controller.config.to_dict(), sort_keys=True),
                flush=True,
            )

    completed = 0
    skipped = 0
    writes_outputs = args.parallel_mode == "data" or rank() == 0
    for local_index, (index, row) in enumerate(owned_rows):
        output_path = output_dir / row["relative_output"]
        should_skip = False
        if writes_outputs:
            should_skip = (
                args.resume and output_path.is_file() and output_path.stat().st_size > 0
            )
        if args.parallel_mode == "sequence":
            should_skip = broadcast_bool(should_skip)
        if should_skip:
            skipped += 1
            if writes_outputs:
                print(f"[{index + 1}/{len(rows)}] skip {output_path}", flush=True)
            continue

        started = time.monotonic()
        if writes_outputs:
            rank_prefix = f"rank={rank()} " if args.parallel_mode == "data" else ""
            print(
                f"[{index + 1}/{len(rows)}] {rank_prefix}method={args.method} "
                f"cfg={args.guidance_scale:g} seed={row['seed']} "
                f"output={output_path}",
                flush=True,
            )

        reference_image = reference_paths[index]
        if guidance_controller is not None and not sfg_x_enabled:
            guidance_controller.begin_sample(int(row["seed"]))
        try:
            result = pipeline(
                enable_sr=False,
                prompt=str(row["prompt"]),
                aspect_ratio=args.aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                video_length=args.video_length,
                guidance_scale=args.guidance_scale,
                cfg_active_steps=cfg_active_steps,
                flow_shift=args.flow_shift,
                negative_prompt="",
                seed=int(row["seed"]),
                output_type="pt",
                prompt_rewrite=False,
                return_pre_sr_video=False,
                enable_vae_tile_parallelism=args.parallel_mode == "sequence",
                reference_image=(
                    str(reference_image) if reference_image is not None else None
                ),
            )
        except Exception:
            if guidance_controller is not None and not sfg_x_enabled:
                guidance_controller.abort_sample()
            raise
        if guidance_controller is not None and not sfg_x_enabled:
            guidance_controller.finish_sample()

        if writes_outputs:
            video_meta = save_video(result.videos, output_path, args.fps)
            elapsed = time.monotonic() - started
            append_jsonl(
                progress_path,
                {
                    "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "index": index,
                    "method": args.method,
                    "prompt": row["prompt"],
                    "seed": row["seed"],
                    "relative_output": row["relative_output"],
                    "reference_image": row.get("reference_image"),
                    "reference_image_sha256": row.get("reference_image_sha256"),
                    "bytes": output_path.stat().st_size,
                    "elapsed_seconds": elapsed,
                    **video_meta,
                },
            )
            print(f"saved {output_path} ({elapsed:.1f}s)", flush=True)
        completed += 1
        del result
        if dist.is_initialized() and args.parallel_mode == "sequence":
            dist.barrier()

    nfe_hook.remove()
    expected_step_invocations = completed * args.num_inference_steps
    if transformer_step_invocations != expected_step_invocations:
        raise RuntimeError(
            "unexpected transformer invocation count: "
            f"actual={transformer_step_invocations} "
            f"expected={expected_step_invocations}"
        )

    i2v_input_runtime = None
    if args.task == "i2v":
        if i2v_task_validation_calls != expected_step_invocations:
            raise RuntimeError(
                "I2V task validation count mismatch: "
                f"actual={i2v_task_validation_calls} expected={expected_step_invocations}"
            )
        if completed and not i2v_nonzero_reference_semantics:
            raise RuntimeError("I2V reference-image semantic stream was never validated")
        i2v_input_runtime = {
            "task_validation_calls": i2v_task_validation_calls,
            "expected_task_validation_calls": expected_step_invocations,
            "observed_vision_tokens": i2v_observed_vision_tokens,
            "expected_vision_tokens": vision_num_semantic_tokens,
            "nonzero_reference_semantics": i2v_nonzero_reference_semantics,
            "cfg_shared_reference_semantics": (
                i2v_cfg_shared_reference_semantics if cfg_enabled else None
            ),
            "cfg_shared_model_input": (
                i2v_cfg_shared_model_input if cfg_enabled else None
            ),
        }

    runtime_sfg_x = sfg_runtime_stats(pipeline.transformer)
    runtime_cfg_zero_init = cfg_zero_init_runtime_stats(pipeline.transformer)
    runtime_cfg_zero_star = cfg_zero_star_runtime_stats(pipeline.transformer)
    runtime_s2 = s2_runtime_stats(pipeline.transformer)
    runtime_stg = stg_runtime_stats(pipeline.transformer)
    active_runtime_count = sum(
        runtime is not None
        for runtime in (
            runtime_sfg_x,
            runtime_cfg_zero_init,
            runtime_cfg_zero_star,
            runtime_s2,
            runtime_stg,
        )
    )
    if active_runtime_count > 1:
        raise RuntimeError("multiple guidance controllers were active")
    if runtime_sfg_x is not None:
        expected_strong_forward_calls = completed * bridge_active_steps
        if int(runtime_sfg_x["logical_forward_calls"]) != expected_step_invocations:
            raise RuntimeError(
                "SFGS/SFGX logical forward schedule mismatch: "
                f"actual={runtime_sfg_x['logical_forward_calls']} "
                f"expected={expected_step_invocations}"
            )
        if int(runtime_sfg_x["strong_forward_calls"]) != expected_strong_forward_calls:
            raise RuntimeError(
                "SFGS/SFGX active-step count mismatch: "
                f"actual={runtime_sfg_x['strong_forward_calls']} "
                f"expected={expected_strong_forward_calls}"
            )
        actual_calls = int(runtime_sfg_x["bridge_attention_calls"])
        expected_calls = int(runtime_sfg_x["expected_bridge_attention_calls"])
        if actual_calls != expected_calls:
            raise RuntimeError(
                f"SFGS/SFGX did not cover every selected block: actual={actual_calls} "
                f"expected={expected_calls}"
            )
        for label in ("sfg_s", "sfg_x"):
            actual_direction_calls = int(runtime_sfg_x[f"{label}_attention_calls"])
            expected_direction_calls = int(
                runtime_sfg_x[f"expected_{label}_attention_calls"]
            )
            if actual_direction_calls != expected_direction_calls:
                raise RuntimeError(
                    f"{label.upper()} directional bridge call mismatch: "
                    f"actual={actual_direction_calls} "
                    f"expected={expected_direction_calls}"
                )
        if int(runtime_sfg_x["strong_forward_calls"]) > 0:
            residual_ratio = runtime_sfg_x["first_residual_relative_rms"]
            if residual_ratio is None or float(residual_ratio) <= 0.0:
                raise RuntimeError(f"SFGX counterfactual residual is not active: {residual_ratio}")
        if args.task == "i2v" and expected_step_invocations > 0:
            if int(runtime_sfg_x["task_validation_calls"]) != expected_step_invocations:
                raise RuntimeError(
                    "I2V task validation did not cover every denoising step: "
                    f"actual={runtime_sfg_x['task_validation_calls']} "
                    f"expected={expected_step_invocations}"
                )
            if int(runtime_sfg_x["observed_vision_tokens"] or -1) != vision_num_semantic_tokens:
                raise RuntimeError(
                    "I2V reference-image semantic-token audit failed: "
                    f"{runtime_sfg_x['observed_vision_tokens']}"
                )
            if not bool(runtime_sfg_x["joint_attention_layout_validated"]):
                raise RuntimeError(
                    "I2V SFG never validated the reference+prompt encoder layout"
                )
            if str(runtime_sfg_x["condition_scope"]) != args.sfg_x_condition_scope:
                raise RuntimeError(
                    f"unexpected I2V SFG condition scope: {runtime_sfg_x['condition_scope']}"
                )
            if args.sfg_x_condition_scope == "text_only":
                if not bool(runtime_sfg_x["vision_attention_scaling_excluded"]):
                    raise RuntimeError(
                        "I2V text-only SFG did not exclude image-semantic tokens"
                    )
                if int(runtime_sfg_x["unscaled_leading_encoder_tokens"] or -1) != (
                    vision_num_semantic_tokens
                ):
                    raise RuntimeError(
                        "I2V text-only SFG has the wrong unscaled token boundary"
                    )
                if int(runtime_sfg_x["observed_valid_prompt_tokens"] or 0) < 1:
                    raise RuntimeError(
                        "I2V text-only SFG did not observe a valid prompt segment"
                    )

    if runtime_cfg_zero_init is not None:
        expected_zeroed_calls = completed * args.zero_init_steps
        zero_checks = {
            "logical_forward_calls": int(
                runtime_cfg_zero_init["logical_forward_calls"]
            )
            == expected_step_invocations,
            "zeroed_forward_calls": int(
                runtime_cfg_zero_init["zeroed_forward_calls"]
            )
            == expected_zeroed_calls,
            "expected_zeroed_forward_calls": int(
                runtime_cfg_zero_init["expected_zeroed_forward_calls"]
            )
            == expected_zeroed_calls,
            "completed_samples": int(runtime_cfg_zero_init["completed_samples"])
            == completed,
            "aborted_samples": int(runtime_cfg_zero_init["aborted_samples"]) == 0,
            "sample_inactive": not bool(runtime_cfg_zero_init["sample_active"]),
            "exact_zero_return": float(
                runtime_cfg_zero_init["maximum_returned_abs_on_zero_steps"]
            )
            == 0.0,
        }
        failed_zero_checks = [name for name, ok in zero_checks.items() if not ok]
        if failed_zero_checks:
            raise RuntimeError(
                "CFG zero-init runtime audit failed: "
                f"{failed_zero_checks}; runtime={runtime_cfg_zero_init}"
            )

    if runtime_cfg_zero_star is not None:
        expected_zeroed_calls = completed * args.zero_init_steps
        expected_optimized_calls = completed * (
            args.num_inference_steps - args.zero_init_steps
        )
        star_checks = {
            "logical_forward_calls": int(
                runtime_cfg_zero_star["logical_forward_calls"]
            )
            == expected_step_invocations,
            "packed_cfg_validation_calls": int(
                runtime_cfg_zero_star["packed_cfg_validation_calls"]
            )
            == expected_step_invocations,
            "zeroed_forward_calls": int(
                runtime_cfg_zero_star["zeroed_forward_calls"]
            )
            == expected_zeroed_calls,
            "expected_zeroed_forward_calls": int(
                runtime_cfg_zero_star["expected_zeroed_forward_calls"]
            )
            == expected_zeroed_calls,
            "optimized_scale_calls": int(
                runtime_cfg_zero_star["optimized_scale_calls"]
            )
            == expected_optimized_calls,
            "expected_optimized_scale_calls": int(
                runtime_cfg_zero_star["expected_optimized_scale_calls"]
            )
            == expected_optimized_calls,
            "completed_samples": int(runtime_cfg_zero_star["completed_samples"])
            == completed,
            "aborted_samples": int(runtime_cfg_zero_star["aborted_samples"]) == 0,
            "sample_inactive": not bool(runtime_cfg_zero_star["sample_active"]),
            "projection_enabled": bool(
                runtime_cfg_zero_star["optimized_scale_projection_enabled"]
            ),
            "exact_zero_return": float(
                runtime_cfg_zero_star["maximum_returned_abs_on_zero_steps"]
            )
            == 0.0,
            "finite_alpha": int(runtime_cfg_zero_star["nonfinite_alpha_count"])
            == 0,
            "native_formula_equivalent": float(
                runtime_cfg_zero_star["maximum_native_equivalence_error"]
            )
            == 0.0,
            "official_commit": runtime_cfg_zero_star["official_commit"]
            == "3162be1fba5dd0129ac8423ad6919d928f420a8d",
        }
        if expected_optimized_calls:
            star_checks.update(
                {
                    "alpha_observed": runtime_cfg_zero_star["first_alpha"] is not None,
                    "projection_changed_unconditional": float(
                        runtime_cfg_zero_star[
                            "maximum_scaled_unconditional_relative_change_rms"
                        ]
                        or 0.0
                    )
                    > 0.0,
                }
            )
        failed_star_checks = [name for name, ok in star_checks.items() if not ok]
        if failed_star_checks:
            raise RuntimeError(
                "official CFG-Zero* runtime audit failed: "
                f"{failed_star_checks}; runtime={runtime_cfg_zero_star}"
            )

    if runtime_s2 is not None:
        s2_active_steps = args.s2_active_end - args.s2_active_start
        expected_subnetwork_calls = completed * s2_active_steps
        expected_dropped_block_calls = expected_subnetwork_calls * args.s2_drop_count
        s2_checks = {
            "logical_forward_calls": int(runtime_s2["logical_forward_calls"])
            == expected_step_invocations,
            "clean_forward_calls": int(runtime_s2["clean_forward_calls"])
            == expected_step_invocations,
            "subnetwork_forward_calls": int(runtime_s2["subnetwork_forward_calls"])
            == expected_subnetwork_calls,
            "dropped_block_calls": int(runtime_s2["dropped_block_calls"])
            == expected_dropped_block_calls,
            "expected_dropped_block_calls": int(
                runtime_s2["expected_dropped_block_calls"]
            )
            == expected_dropped_block_calls,
            "completed_samples": int(runtime_s2["completed_samples"]) == completed,
            "aborted_samples": int(runtime_s2["aborted_samples"]) == 0,
            "sample_inactive": not bool(runtime_s2["sample_active"]),
            "block_zero_excluded": 0
            not in set(runtime_s2["candidate_block_indices"]),
        }
        if completed:
            s2_checks["residual_nonzero"] = (
                float(runtime_s2["first_residual_relative_rms"] or 0.0) > 0.0
            )
        failed_s2_checks = [name for name, ok in s2_checks.items() if not ok]
        if failed_s2_checks:
            raise RuntimeError(
                f"S2 runtime audit failed: {failed_s2_checks}; runtime={runtime_s2}"
            )

    if runtime_stg is not None:
        stg_active_steps = args.stg_active_end - args.stg_active_start
        expected_perturbed_calls = (
            completed * stg_active_steps if args.stg_scale > 0.0 else 0
        )
        expected_identity_calls = expected_perturbed_calls
        stg_checks = {
            "logical_forward_calls": int(runtime_stg["logical_forward_calls"])
            == expected_step_invocations,
            "clean_forward_calls": int(runtime_stg["clean_forward_calls"])
            == expected_step_invocations,
            "perturbed_forward_calls": int(runtime_stg["perturbed_forward_calls"])
            == expected_perturbed_calls,
            "identity_block_calls": int(runtime_stg["identity_block_calls"])
            == expected_identity_calls,
            "expected_identity_block_calls": int(
                runtime_stg["expected_identity_block_calls"]
            )
            == expected_identity_calls,
            "restored_block_calls": int(runtime_stg["restored_block_calls"])
            == expected_identity_calls,
            "completed_samples": int(runtime_stg["completed_samples"]) == completed,
            "aborted_samples": int(runtime_stg["aborted_samples"]) == 0,
            "sample_inactive": not bool(runtime_stg["sample_active"]),
            "official_commit": runtime_stg["official_commit"]
            == "d9e7be5dadfc8d0f53855b2a56e478a5e64b2ca4",
            "official_source": runtime_stg["official_hunyuan_source_sha256"]
            == "f0c911bddb818d050b45e801f2a14b432ac7b588ed5af3b3c2c2ede2b1715c99",
            "configured_block": runtime_stg["block_indices"]
            == [args.stg_block_index],
            "configured_scale": float(runtime_stg["scale"])
            == args.stg_scale,
            "configured_window": runtime_stg["active_start"]
            == args.stg_active_start
            and runtime_stg["active_end_exclusive"] == args.stg_active_end,
            "configured_port": runtime_stg["port_kind"]
            == (
                "hv15_base_adapted"
                if stg_hv15_enabled
                else "literal_default_port"
            ),
        }
        if completed:
            stg_checks["residual_nonzero"] = (
                float(runtime_stg["maximum_residual_relative_rms"] or 0.0) > 0.0
            )
        failed_stg_checks = [name for name, ok in stg_checks.items() if not ok]
        if failed_stg_checks:
            raise RuntimeError(
                f"STG runtime audit failed: {failed_stg_checks}; runtime={runtime_stg}"
            )

    if runtime_sfg_x is not None:
        branch_model_evaluations = int(runtime_sfg_x["clean_forward_calls"]) + int(
            runtime_sfg_x["strong_forward_calls"]
        )
        expected_branch_model_evaluations = (
            expected_step_invocations + completed * bridge_active_steps
        )
    elif runtime_s2 is not None:
        branch_model_evaluations = int(runtime_s2["clean_forward_calls"]) + int(
            runtime_s2["subnetwork_forward_calls"]
        )
        expected_branch_model_evaluations = expected_step_invocations + (
            completed * (args.s2_active_end - args.s2_active_start)
        )
    elif runtime_stg is not None:
        branch_model_evaluations = int(runtime_stg["clean_forward_calls"]) + int(
            runtime_stg["perturbed_forward_calls"]
        )
        expected_branch_model_evaluations = expected_step_invocations + (
            completed * (args.stg_active_end - args.stg_active_start)
            if args.stg_scale > 0.0
            else 0
        )
    else:
        # CFG packs conditional/unconditional branches in one batched module
        # invocation but still costs two sample-equivalent model evaluations.
        branch_multiplier = 2 if cfg_enabled else 1
        branch_model_evaluations = transformer_step_invocations * branch_multiplier
        expected_branch_model_evaluations = expected_step_invocations * branch_multiplier
    if branch_model_evaluations != expected_branch_model_evaluations:
        raise RuntimeError(
            "unexpected branch-model evaluation count: "
            f"actual={branch_model_evaluations} "
            f"expected={expected_branch_model_evaluations}"
        )
    nfe_runtime = {
        "transformer_step_invocations": transformer_step_invocations,
        "expected_transformer_step_invocations": expected_step_invocations,
        "branch_model_evaluations": branch_model_evaluations,
        "expected_branch_model_evaluations": expected_branch_model_evaluations,
        "generated_video_count": completed,
        "sampling_steps_per_video": args.num_inference_steps,
        "cfg_active_steps_per_video": cfg_active_steps,
        "bridge_active_steps_per_video": bridge_active_steps if sfg_x_enabled else 0,
        "stg_active_steps_per_video": (
            args.stg_active_end - args.stg_active_start if stg_enabled else 0
        ),
        "nfe_per_step": (
            branch_model_evaluations / expected_step_invocations
            if expected_step_invocations
            else None
        ),
        "sfg_x_enabled": sfg_x_enabled,
        "cfg_enabled": cfg_enabled,
        "cfg_zero_init_enabled": cfg_zero_init_enabled,
        "cfg_zero_star_enabled": cfg_zero_star_enabled,
        "s2_enabled": s2_enabled,
        "stg_enabled": stg_enabled,
    }
    if (
        not sfg_x_enabled
        and not s2_enabled
        and not stg_enabled
        and args.guidance_scale == 1.0
        and completed
    ):
        if nfe_runtime["nfe_per_step"] != 1.0:
            raise RuntimeError(
                f"NoGuidance must use exactly 1 NFE/step: {nfe_runtime}"
            )

    local_summary = {
        "rank": rank(),
        "generated_this_run": completed,
        "skipped_existing": skipped,
        "sfg_x_runtime": runtime_sfg_x,
        "cfg_zero_init_runtime": runtime_cfg_zero_init,
        "cfg_zero_star_runtime": runtime_cfg_zero_star,
        "s2_runtime": runtime_s2,
        "stg_runtime": runtime_stg,
        "i2v_input_runtime": i2v_input_runtime,
        "nfe_runtime": nfe_runtime,
    }
    if args.parallel_mode == "data" and dist.is_initialized():
        rank_summaries: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_summaries, local_summary)
    else:
        rank_summaries = [local_summary]

    if rank() == 0:
        bad_outputs = [
            row["relative_output"]
            for row in rows
            if not (output_dir / row["relative_output"]).is_file()
            or (output_dir / row["relative_output"]).stat().st_size <= 0
        ]
        if bad_outputs:
            raise RuntimeError(
                f"generation incomplete; invalid outputs include {bad_outputs[:5]}"
            )

        if args.parallel_mode == "data":
            progress_by_index: dict[int, dict[str, Any]] = {}
            for worker_rank in range(world_size):
                shard_path = output_dir / f"generation_progress.rank{worker_rank:02d}.jsonl"
                if not shard_path.is_file():
                    continue
                for line in shard_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    progress_by_index[int(record["index"])] = record
            for index, row in indexed_rows:
                if index not in progress_by_index:
                    output_path = output_dir / row["relative_output"]
                    progress_by_index[index] = {
                        "completed_at_utc": None,
                        "elapsed_seconds": None,
                        "index": index,
                        "method": args.method,
                        "prompt": row["prompt"],
                        "seed": row["seed"],
                        "relative_output": row["relative_output"],
                        "bytes": output_path.stat().st_size,
                        "recovered_from_existing_output": True,
                    }
            write_jsonl(
                output_dir / "generation_progress.jsonl",
                [progress_by_index[index] for index, _row in indexed_rows],
            )

        if sfg_x_enabled and args.parallel_mode == "data":
            per_rank_runtime = [
                {
                    "rank": int(summary["rank"]),
                    **summary["sfg_x_runtime"],
                }
                for summary in rank_summaries
                if summary is not None and summary["sfg_x_runtime"] is not None
            ]
            residuals = [
                float(item["first_residual_relative_rms"])
                for item in per_rank_runtime
                if item.get("first_residual_relative_rms") is not None
            ]
            runtime_sfg_x = {
                **per_rank_runtime[0],
                "logical_forward_calls": sum(
                    int(item["logical_forward_calls"]) for item in per_rank_runtime
                ),
                "clean_forward_calls": sum(
                    int(item["clean_forward_calls"]) for item in per_rank_runtime
                ),
                "strong_forward_calls": sum(
                    int(item["strong_forward_calls"]) for item in per_rank_runtime
                ),
                "bridge_attention_calls": sum(
                    int(item["bridge_attention_calls"]) for item in per_rank_runtime
                ),
                "expected_bridge_attention_calls": sum(
                    int(item["expected_bridge_attention_calls"])
                    for item in per_rank_runtime
                ),
                "sfg_s_attention_calls": sum(
                    int(item["sfg_s_attention_calls"]) for item in per_rank_runtime
                ),
                "expected_sfg_s_attention_calls": sum(
                    int(item["expected_sfg_s_attention_calls"])
                    for item in per_rank_runtime
                ),
                "sfg_x_attention_calls": sum(
                    int(item["sfg_x_attention_calls"]) for item in per_rank_runtime
                ),
                "expected_sfg_x_attention_calls": sum(
                    int(item["expected_sfg_x_attention_calls"])
                    for item in per_rank_runtime
                ),
                "first_residual_relative_rms": min(residuals) if residuals else None,
                "first_residual_relative_rms_max": max(residuals) if residuals else None,
                "rank_count": len(per_rank_runtime),
                "per_rank": per_rank_runtime,
            }

        if cfg_zero_init_enabled and args.parallel_mode == "data":
            per_rank_zero_runtime = [
                {
                    "rank": int(summary["rank"]),
                    **summary["cfg_zero_init_runtime"],
                }
                for summary in rank_summaries
                if summary is not None
                and summary["cfg_zero_init_runtime"] is not None
            ]
            pre_zero_rms = [
                float(item["first_pre_zero_prediction_rms"])
                for item in per_rank_zero_runtime
                if item.get("first_pre_zero_prediction_rms") is not None
            ]
            runtime_cfg_zero_init = {
                **per_rank_zero_runtime[0],
                "logical_forward_calls": sum(
                    int(item["logical_forward_calls"])
                    for item in per_rank_zero_runtime
                ),
                "zeroed_forward_calls": sum(
                    int(item["zeroed_forward_calls"])
                    for item in per_rank_zero_runtime
                ),
                "expected_zeroed_forward_calls": sum(
                    int(item["expected_zeroed_forward_calls"])
                    for item in per_rank_zero_runtime
                ),
                "completed_samples": sum(
                    int(item["completed_samples"]) for item in per_rank_zero_runtime
                ),
                "aborted_samples": sum(
                    int(item["aborted_samples"]) for item in per_rank_zero_runtime
                ),
                "sample_active": any(
                    bool(item["sample_active"]) for item in per_rank_zero_runtime
                ),
                "first_pre_zero_prediction_rms": (
                    min(pre_zero_rms) if pre_zero_rms else None
                ),
                "first_pre_zero_prediction_rms_max": (
                    max(pre_zero_rms) if pre_zero_rms else None
                ),
                "maximum_returned_abs_on_zero_steps": max(
                    float(item["maximum_returned_abs_on_zero_steps"])
                    for item in per_rank_zero_runtime
                ),
                "rank_count": len(per_rank_zero_runtime),
                "per_rank": per_rank_zero_runtime,
            }

        if cfg_zero_star_enabled and args.parallel_mode == "data":
            per_rank_star_runtime = [
                {
                    "rank": int(summary["rank"]),
                    **summary["cfg_zero_star_runtime"],
                }
                for summary in rank_summaries
                if summary is not None
                and summary["cfg_zero_star_runtime"] is not None
            ]
            pre_zero_rms = [
                float(item["first_pre_zero_prediction_rms"])
                for item in per_rank_star_runtime
                if item.get("first_pre_zero_prediction_rms") is not None
            ]
            first_alphas = [
                float(item["first_alpha"])
                for item in per_rank_star_runtime
                if item.get("first_alpha") is not None
            ]
            projection_changes = [
                float(item["first_scaled_unconditional_relative_change_rms"])
                for item in per_rank_star_runtime
                if item.get("first_scaled_unconditional_relative_change_rms")
                is not None
            ]
            maximum_projection_changes = [
                float(item["maximum_scaled_unconditional_relative_change_rms"])
                for item in per_rank_star_runtime
                if item.get("maximum_scaled_unconditional_relative_change_rms")
                is not None
            ]
            runtime_cfg_zero_star = {
                **per_rank_star_runtime[0],
                "logical_forward_calls": sum(
                    int(item["logical_forward_calls"])
                    for item in per_rank_star_runtime
                ),
                "packed_cfg_validation_calls": sum(
                    int(item["packed_cfg_validation_calls"])
                    for item in per_rank_star_runtime
                ),
                "zeroed_forward_calls": sum(
                    int(item["zeroed_forward_calls"])
                    for item in per_rank_star_runtime
                ),
                "expected_zeroed_forward_calls": sum(
                    int(item["expected_zeroed_forward_calls"])
                    for item in per_rank_star_runtime
                ),
                "optimized_scale_calls": sum(
                    int(item["optimized_scale_calls"])
                    for item in per_rank_star_runtime
                ),
                "expected_optimized_scale_calls": sum(
                    int(item["expected_optimized_scale_calls"])
                    for item in per_rank_star_runtime
                ),
                "completed_samples": sum(
                    int(item["completed_samples"])
                    for item in per_rank_star_runtime
                ),
                "aborted_samples": sum(
                    int(item["aborted_samples"])
                    for item in per_rank_star_runtime
                ),
                "sample_active": any(
                    bool(item["sample_active"]) for item in per_rank_star_runtime
                ),
                "first_pre_zero_prediction_rms": (
                    min(pre_zero_rms) if pre_zero_rms else None
                ),
                "first_pre_zero_prediction_rms_max": (
                    max(pre_zero_rms) if pre_zero_rms else None
                ),
                "maximum_returned_abs_on_zero_steps": max(
                    float(item["maximum_returned_abs_on_zero_steps"])
                    for item in per_rank_star_runtime
                ),
                "first_alpha": first_alphas[0] if first_alphas else None,
                "minimum_alpha": min(
                    float(item["minimum_alpha"])
                    for item in per_rank_star_runtime
                    if item.get("minimum_alpha") is not None
                ),
                "maximum_alpha": max(
                    float(item["maximum_alpha"])
                    for item in per_rank_star_runtime
                    if item.get("maximum_alpha") is not None
                ),
                "nonfinite_alpha_count": sum(
                    int(item["nonfinite_alpha_count"])
                    for item in per_rank_star_runtime
                ),
                "first_scaled_unconditional_relative_change_rms": (
                    min(projection_changes) if projection_changes else None
                ),
                "first_scaled_unconditional_relative_change_rms_max": (
                    max(projection_changes) if projection_changes else None
                ),
                "maximum_scaled_unconditional_relative_change_rms": (
                    max(maximum_projection_changes)
                    if maximum_projection_changes
                    else 0.0
                ),
                "maximum_native_equivalence_error": max(
                    float(item["maximum_native_equivalence_error"])
                    for item in per_rank_star_runtime
                ),
                "rank_count": len(per_rank_star_runtime),
                "per_rank": per_rank_star_runtime,
            }

        if s2_enabled and args.parallel_mode == "data":
            per_rank_s2_runtime = [
                {"rank": int(summary["rank"]), **summary["s2_runtime"]}
                for summary in rank_summaries
                if summary is not None and summary["s2_runtime"] is not None
            ]
            s2_residuals = [
                float(item["first_residual_relative_rms"])
                for item in per_rank_s2_runtime
                if item.get("first_residual_relative_rms") is not None
            ]
            runtime_s2 = {
                **per_rank_s2_runtime[0],
                "logical_forward_calls": sum(
                    int(item["logical_forward_calls"]) for item in per_rank_s2_runtime
                ),
                "clean_forward_calls": sum(
                    int(item["clean_forward_calls"]) for item in per_rank_s2_runtime
                ),
                "subnetwork_forward_calls": sum(
                    int(item["subnetwork_forward_calls"])
                    for item in per_rank_s2_runtime
                ),
                "dropped_block_calls": sum(
                    int(item["dropped_block_calls"])
                    for item in per_rank_s2_runtime
                ),
                "expected_dropped_block_calls": sum(
                    int(item["expected_dropped_block_calls"])
                    for item in per_rank_s2_runtime
                ),
                "completed_samples": sum(
                    int(item["completed_samples"]) for item in per_rank_s2_runtime
                ),
                "aborted_samples": sum(
                    int(item["aborted_samples"]) for item in per_rank_s2_runtime
                ),
                "sample_active": any(
                    bool(item["sample_active"]) for item in per_rank_s2_runtime
                ),
                "first_residual_relative_rms": (
                    min(s2_residuals) if s2_residuals else None
                ),
                "first_residual_relative_rms_max": (
                    max(s2_residuals) if s2_residuals else None
                ),
                "rank_count": len(per_rank_s2_runtime),
                "per_rank_mask_schedule_sha256": [
                    {
                        "rank": item["rank"],
                        "sha256": item["mask_schedule_sha256"],
                    }
                    for item in per_rank_s2_runtime
                ],
                "per_rank": per_rank_s2_runtime,
            }

        if stg_enabled and args.parallel_mode == "data":
            per_rank_stg_runtime = [
                {"rank": int(summary["rank"]), **summary["stg_runtime"]}
                for summary in rank_summaries
                if summary is not None and summary["stg_runtime"] is not None
            ]
            stg_residuals = [
                float(item["first_residual_relative_rms"])
                for item in per_rank_stg_runtime
                if item.get("first_residual_relative_rms") is not None
            ]
            stg_corrections = [
                float(item["first_guidance_correction_relative_rms"])
                for item in per_rank_stg_runtime
                if item.get("first_guidance_correction_relative_rms") is not None
            ]
            runtime_stg = {
                **per_rank_stg_runtime[0],
                "logical_forward_calls": sum(
                    int(item["logical_forward_calls"])
                    for item in per_rank_stg_runtime
                ),
                "clean_forward_calls": sum(
                    int(item["clean_forward_calls"])
                    for item in per_rank_stg_runtime
                ),
                "perturbed_forward_calls": sum(
                    int(item["perturbed_forward_calls"])
                    for item in per_rank_stg_runtime
                ),
                "identity_block_calls": sum(
                    int(item["identity_block_calls"])
                    for item in per_rank_stg_runtime
                ),
                "expected_identity_block_calls": sum(
                    int(item["expected_identity_block_calls"])
                    for item in per_rank_stg_runtime
                ),
                "restored_block_calls": sum(
                    int(item["restored_block_calls"])
                    for item in per_rank_stg_runtime
                ),
                "completed_samples": sum(
                    int(item["completed_samples"])
                    for item in per_rank_stg_runtime
                ),
                "aborted_samples": sum(
                    int(item["aborted_samples"])
                    for item in per_rank_stg_runtime
                ),
                "sample_active": any(
                    bool(item["sample_active"]) for item in per_rank_stg_runtime
                ),
                "first_residual_relative_rms": (
                    min(stg_residuals) if stg_residuals else None
                ),
                "first_residual_relative_rms_max": (
                    max(stg_residuals) if stg_residuals else None
                ),
                "maximum_residual_relative_rms": max(
                    float(item["maximum_residual_relative_rms"])
                    for item in per_rank_stg_runtime
                ),
                "first_guidance_correction_relative_rms": (
                    min(stg_corrections) if stg_corrections else None
                ),
                "first_guidance_correction_relative_rms_max": (
                    max(stg_corrections) if stg_corrections else None
                ),
                "maximum_guidance_correction_relative_rms": max(
                    float(item["maximum_guidance_correction_relative_rms"])
                    for item in per_rank_stg_runtime
                ),
                "rank_count": len(per_rank_stg_runtime),
                "per_rank": per_rank_stg_runtime,
            }

        generated_total = sum(
            int(summary["generated_this_run"])
            for summary in rank_summaries
            if summary is not None
        )
        skipped_total = sum(
            int(summary["skipped_existing"])
            for summary in rank_summaries
            if summary is not None
        )
        per_rank_nfe = [
            {"rank": int(summary["rank"]), **summary["nfe_runtime"]}
            for summary in rank_summaries
            if summary is not None
        ]
        total_step_invocations = sum(
            int(item["transformer_step_invocations"]) for item in per_rank_nfe
        )
        total_expected_step_invocations = sum(
            int(item["expected_transformer_step_invocations"])
            for item in per_rank_nfe
        )
        total_branch_evaluations = sum(
            int(item["branch_model_evaluations"]) for item in per_rank_nfe
        )
        total_expected_branch_evaluations = sum(
            int(item["expected_branch_model_evaluations"])
            for item in per_rank_nfe
        )
        nfe_runtime = {
            "transformer_step_invocations": total_step_invocations,
            "expected_transformer_step_invocations": total_expected_step_invocations,
            "branch_model_evaluations": total_branch_evaluations,
            "expected_branch_model_evaluations": total_expected_branch_evaluations,
            "generated_video_count": generated_total,
            "sampling_steps_per_video": args.num_inference_steps,
            "cfg_active_steps_per_video": cfg_active_steps,
            "bridge_active_steps_per_video": (
                bridge_active_steps if sfg_x_enabled else 0
            ),
            "stg_active_steps_per_video": (
                args.stg_active_end - args.stg_active_start if stg_enabled else 0
            ),
            "nfe_per_step": (
                total_branch_evaluations / total_expected_step_invocations
                if total_expected_step_invocations
                else None
            ),
            "sfg_x_enabled": sfg_x_enabled,
            "cfg_enabled": cfg_enabled,
            "cfg_zero_init_enabled": cfg_zero_init_enabled,
            "cfg_zero_star_enabled": cfg_zero_star_enabled,
            "s2_enabled": s2_enabled,
            "stg_enabled": stg_enabled,
            "rank_count": len(per_rank_nfe),
            "per_rank": per_rank_nfe,
        }
        per_rank_i2v_input = [
            {"rank": int(summary["rank"]), **summary["i2v_input_runtime"]}
            for summary in rank_summaries
            if summary is not None and summary["i2v_input_runtime"] is not None
        ]
        active_i2v_input = [
            item
            for item in per_rank_i2v_input
            if int(item["task_validation_calls"]) > 0
        ]
        i2v_input_runtime = (
            {
                "task_validation_calls": sum(
                    int(item["task_validation_calls"]) for item in per_rank_i2v_input
                ),
                "expected_task_validation_calls": sum(
                    int(item["expected_task_validation_calls"])
                    for item in per_rank_i2v_input
                ),
                "observed_vision_tokens": (
                    int(active_i2v_input[0]["observed_vision_tokens"])
                    if active_i2v_input
                    else None
                ),
                "expected_vision_tokens": vision_num_semantic_tokens,
                "nonzero_reference_semantics": bool(active_i2v_input)
                and all(
                    bool(item["nonzero_reference_semantics"])
                    for item in active_i2v_input
                ),
                "cfg_shared_reference_semantics": (
                    bool(active_i2v_input)
                    and all(
                        bool(item["cfg_shared_reference_semantics"])
                        for item in active_i2v_input
                    )
                    if cfg_enabled
                    else None
                ),
                "cfg_shared_model_input": (
                    bool(active_i2v_input)
                    and all(
                        bool(item["cfg_shared_model_input"])
                        for item in active_i2v_input
                    )
                    if cfg_enabled
                    else None
                ),
                "rank_count": len(per_rank_i2v_input),
                "active_rank_count": len(active_i2v_input),
                "per_rank": per_rank_i2v_input,
            }
            if args.task == "i2v"
            else None
        )
        write_json(
            output_dir / "generation_complete.json",
            {
                **run_config,
                "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "generated_this_run": generated_total,
                "skipped_existing": skipped_total,
                "sfg_x_runtime": runtime_sfg_x,
                "cfg_zero_init_runtime": runtime_cfg_zero_init,
                "cfg_zero_star_runtime": runtime_cfg_zero_star,
                "s2_runtime": runtime_s2,
                "stg_runtime": runtime_stg,
                "i2v_input_runtime": i2v_input_runtime,
                "nfe_runtime": nfe_runtime,
            },
        )
        print(
            f"COMPLETE method={args.method} generated={generated_total} "
            f"skipped={skipped_total}",
            flush=True,
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
