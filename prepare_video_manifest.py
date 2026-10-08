#!/usr/bin/env python3
"""Write one user-supplied T2V/I2V sample; no benchmark dataset is bundled."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("manifest.jsonl"))
    args = parser.parse_args()
    row = dict(prompt=args.prompt, seed=args.seed, relative_output="sample-0.mp4")
    if args.reference_image:
        path = args.reference_image.resolve()
        row.update(reference_image=str(path), reference_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
