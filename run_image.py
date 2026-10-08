#!/usr/bin/env python3
"""Generate images with SFG using model-specific default parameters."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

from defaults import IMAGE_DEFAULTS

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=IMAGE_DEFAULTS, default="sd35m")
    parser.add_argument("--weights", type=Path, help="Local complete Diffusers model directory")
    parser.add_argument("--transformer", type=Path, help="Required de-distilled transformer directory")
    parser.add_argument("--prompt-file", type=Path, help="One prompt per image, repeated for multiple seeds")
    parser.add_argument("--output", type=Path, default=Path("outputs/images"))
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--baseline", action="store_true", help="Disable SFG; retain the selected guidance")
    for key in ("steps", "start-step", "end-step"):
        parser.add_argument(f"--{key}", type=int)
    for key in ("guidance", "u-s", "u-x", "omega"):
        parser.add_argument(f"--{key}", type=float)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--print-config", action="store_true", help="No models/GPU/dependencies loaded")
    args = parser.parse_args()
    config = dict(IMAGE_DEFAULTS[args.model])
    for key in config:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.print_config:
        print(json.dumps(dict(model=args.model, baseline=args.baseline, **config), indent=2))
        return
    if args.weights is None or args.prompt_file is None:
        parser.error("--weights and --prompt-file are required for generation")
    if args.model == "flux-de-distill" and args.transformer is None:
        parser.error("--transformer must point to the de-distilled transformer, not the original dev weights")
    if args.model != "flux-de-distill" and args.transformer is not None:
        parser.error("--transformer is only valid for flux-de-distill")
    prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
    count = len(prompts) if args.num_images is None else args.num_images
    if count < 1 or count > len(prompts) or args.batch_size < 1:
        parser.error("Require 1 <= num-images <= number of prompt lines, and batch-size >= 1")
    prompts = prompts[:count]
    if not args.weights.is_dir():
        parser.error(f"Model directory does not exist: {args.weights}")
    if config["steps"] < 1 or config["start_step"] < 1 or config["end_step"] < 0:
        parser.error("Invalid sampling steps or SFG window")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.model.startswith("flux"):
        sampler = ROOT / "image" / ("flux_dev.py" if args.model == "flux-dev" else "flux_dedistill.py")
        forwarded = [str(sampler), "--model-dir", str(args.weights.resolve()),
                     "--prompt-file", str(args.prompt_file.resolve()),
                     "--workdir", str(args.output.resolve()), "--num-images", str(count),
                     "--batch-size", str(args.batch_size), "--seed", str(args.seed),
                     "--width", str(args.width), "--height", str(args.height),
                     "--steps", str(config["steps"]), "--guidance-scale", str(config["guidance"]),
                     "--negative-prompt", args.negative_prompt]
        if args.model == "flux-de-distill":
            forwarded += ["--transformer-dir", str(args.transformer.resolve()),
                          "--pipeline-file", str(ROOT / "image/pipeline_flux_de_distill.py")]
        if not args.baseline:
            forwarded += ["--sfg-mode", "explicit", "--bridge-direction", "both",
                          "--sfg-strength-u", str(config["u_s"]), "--sfg-strength-u-t2i", str(config["u_x"]),
                          "--bridge-omega", str(config["omega"]), "--bridge-normclip-tau", "999",
                          "--bridge-start-step", str(config["start_step"]),
                          "--bridge-end-step", str(config["end_step"]), "--bridge-layers", args.layers]
        sys.argv = forwarded
        runpy.run_path(str(sampler), run_name="__main__")
        return

    import torch
    import torch.distributed as dist
    sys.path.insert(0, str(ROOT / "image/sd3"))
    from latent_sd35 import SD35M

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        solver = SD35M(solver_config=SimpleNamespace(num_sampling=config["steps"]),
                       model_key=str(args.weights.resolve()), device=device, dtype=torch.bfloat16)
        destination = args.output / "result"
        destination.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            (args.output / "run_config.json").write_text(json.dumps(
                dict(model=args.model, baseline=args.baseline, seed=args.seed,
                     seed_rule="seed + flat image index", world_size=world,
                     width=args.width, height=args.height, **config), indent=2))
        indices = list(range(rank, count, world))
        for offset in range(0, len(indices), args.batch_size):
            batch = indices[offset:offset + args.batch_size]
            extra = {} if args.baseline else dict(
                sfg_mode="explicit", bridge_variant="explicit", bridge_direction="both",
                sfg_strength_u=config["u_s"], sfg_strength_u_t2i=config["u_x"],
                bridge_omega=config["omega"], bridge_normclip_tau=999.0, bridge_orthogonal=False,
                bridge_start_step=config["start_step"], bridge_end_step=config["end_step"],
                bridge_layers=args.layers)
            images = solver.sample(prompt=[[args.negative_prompt] * len(batch), [prompts[i] for i in batch]],
                                   cfg_guidance=config["guidance"], target_size=(args.height, args.width),
                                   generator=[torch.Generator(device=device).manual_seed(args.seed+i) for i in batch],
                                   **extra)
            for index, image in zip(batch, images):
                target = destination / f"{index:05d}.png"
                if target.exists():
                    raise FileExistsError(f"Use a fresh output directory: {target}")
                image.save(target)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
