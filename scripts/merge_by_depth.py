#!/usr/bin/env python3
"""Write a depth-graded merge of two fine-tunes of one model to a new .safetensors file.

The same blend as the Model Merge by Depth node (nodes/merge/_depth.py), done on the CPU into a
real checkpoint, so ComfyUI loads it like any other model — no patches applied at load time.

    python -I scripts/merge_by_depth.py SFT.safetensors BASE.safetensors OUT.safetensors \\
        --early 1 --late 0 --crossover 21 --softness 8

Model A (the first file) takes `early` of the first blocks and `late` of the last. Each weight is
blended in float32 and stored in model A's dtype. Needs torch and safetensors; about one model's
size in RAM.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_spec = importlib.util.spec_from_file_location(
    "depth", Path(__file__).resolve().parents[1] / "nodes" / "merge" / "_depth.py"
)
depth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(depth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_a", type=Path, help="Early blocks' model, e.g. MOSS SFT.")
    parser.add_argument("model_b", type=Path, help="Late blocks' model, e.g. MOSS Base.")
    parser.add_argument("out", type=Path)
    parser.add_argument("--early", type=float, default=1.0)
    parser.add_argument("--late", type=float, default=0.0)
    parser.add_argument("--crossover", type=float, default=21)
    parser.add_argument("--softness", type=float, default=8)
    args = parser.parse_args()

    if args.out.exists():
        print(f"{args.out} exists; not overwriting.", file=sys.stderr)
        return 1
    a = safe_open(args.model_a, "pt", device="cpu")
    b = safe_open(args.model_b, "pt", device="cpu")
    keys_a, keys_b = set(a.keys()), set(b.keys())
    if keys_a != keys_b:
        diff = sorted(keys_a ^ keys_b)
        print(f"The models' weights differ in {len(diff)} names, e.g. {diff[0]}.", file=sys.stderr)
        return 1
    blocks = depth.depth(keys_a)
    shares = depth.curve(blocks, args.early, args.late, args.crossover, args.softness)
    print(f"{blocks} blocks; model A's share by block:")
    print(" ".join(f"{i}:{s:.2f}" for i, s in enumerate(shares)))

    merged = {}
    for n, key in enumerate(sorted(keys_a)):
        share = depth.ratio_for(key, blocks, args.early, args.late, args.crossover, args.softness)
        ta = a.get_tensor(key)
        if share >= 1.0:
            merged[key] = ta
        elif share <= 0.0:
            merged[key] = b.get_tensor(key).to(ta.dtype)
        else:
            tb = b.get_tensor(key)
            merged[key] = torch.lerp(tb.float(), ta.float(), share).to(ta.dtype)
        if n % 50 == 0:
            print(f"  {n}/{len(keys_a)} {key} share {share:.3f}", flush=True)

    metadata = dict(a.metadata() or {})
    metadata["nynxz_merge"] = (
        f"depth: A={args.model_a.name} B={args.model_b.name} early={args.early} "
        f"late={args.late} crossover={args.crossover} softness={args.softness}"
    )
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    save_file(merged, temp, metadata=metadata)
    os.replace(temp, args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 2**30:.2f} GiB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
