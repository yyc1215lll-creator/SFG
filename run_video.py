#!/usr/bin/env python3
"""Generate HunyuanVideo T2V or I2V samples with SFG.

Use torchrun --standalone --nproc_per_node=1 (or 8 for data parallelism).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

from defaults import VIDEO_DEFAULTS

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=VIDEO_DEFAULTS, default="t2v")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--offloading", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    for key in ("steps", "active-steps"):
        parser.add_argument(f"--{key}", type=int)
    for key in ("guidance", "u-s", "u-x", "omega"):
        parser.add_argument(f"--{key}", type=float)
    args = parser.parse_args()
    config = dict(VIDEO_DEFAULTS[args.task])
    for key in config:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if args.baseline:
        config.update(u_s=0.0, u_x=0.0, omega=0.0, active_steps=0)
    if args.print_config:
        print(json.dumps(dict(task=args.task, **config), indent=2))
        return
    if args.weights is None or args.manifest is None:
        parser.error("--weights and --manifest are required for generation")
    sampler = ROOT / "video/generation/hunyuan_video_generate_manifest.py"
    argv = [str(sampler), "--task", "i2v" if args.task.startswith("i2v") else "t2v",
            "--model-path", str(args.weights.resolve()), "--manifest", str(args.manifest.resolve()),
            "--output-dir", str(args.output.resolve()), "--method", f"{args.task}_{'baseline' if args.baseline else 'sfg'}",
            "--guidance-scale", str(config["guidance"]), "--num-inference-steps", str(config["steps"]),
            "--sfg-s-u", str(config["u_s"]), "--sfg-x-u", str(config["u_x"]),
            "--sfg_x-omega", str(config["omega"]), "--bridge-active-steps", str(config["active_steps"]),
            "--sfg_x-condition-scope", config["condition_scope"], "--parallel-mode", "data",
            "--flow-shift", "5", "--video-length", "121", "--fps", "24", "--aspect-ratio", "16:9"]
    if args.image_root:
        argv += ["--image-root", str(args.image_root.resolve())]
    if args.offloading:
        argv += ["--offloading"]
    if args.resume:
        argv += ["--resume"]
    sys.path.insert(0, str(sampler.parent))
    sys.argv = argv
    runpy.run_path(str(sampler), run_name="__main__")


if __name__ == "__main__":
    main()
