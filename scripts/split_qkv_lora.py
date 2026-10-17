#!/usr/bin/env python3
"""Rewrite a LoRA trained on fused q/k/v weights so it also fits checkpoints with separate q, k, v.

ComfyUI places a LoRA's separate q/k/v keys onto a fused `qkv` weight, but not the other way:
a LoRA with `...proj.qkv` keys loses its whole attention update on a split checkpoint ("lora key
not loaded"). This writes a copy whose `qkv` entries are split into `q`, `k` and `v`, exactly:

    LoRA / DoRA   the up matrix (and DoRA's scale) cut into thirds by row; down and alpha shared
    LoHa          each Hadamard factor's left matrix cut into thirds by row
    LoKr          the first Kronecker factor cut into thirds by row — exact when its row count
                  divides by 3, which is the usual case; otherwise the file is left alone

    python -I scripts/split_qkv_lora.py FUSED_LORA.safetensors [OUT.safetensors]

Everything else is copied as is. Runs on the CPU.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FUSED = re.compile(r"^(?P<stem>.*\.proj)\.qkv\.(?P<rest>.+)$")
PARTS = ("q", "k", "v")

#: Tensors cut into thirds by row. Everything else under a qkv prefix is shared by all three.
ROW_SPLIT = (
    "lora_up.weight",
    "lora_B.weight",
    "lora.up.weight",
    "dora_scale",
    "hada_w1_a",
    "hada_w2_a",
)


def thirds(tensor: torch.Tensor, name: str) -> list[torch.Tensor]:
    rows = tensor.shape[0]
    if rows % 3:
        raise ValueError(f"{name}: {rows} rows don't split into thirds")
    third = rows // 3
    return [tensor[i * third : (i + 1) * third].contiguous() for i in range(3)]


def split_lokr(prefix: str, tensors: dict, out: dict) -> None:
    """LoKr's update is kron(w1, w2): a run of a third of its rows is a third of w1's rows."""
    if f"{prefix}.lokr_w1" in tensors:
        w1_parts = thirds(tensors[f"{prefix}.lokr_w1"], f"{prefix}.lokr_w1")
        w1_key = "lokr_w1"
    elif f"{prefix}.lokr_w1_a" in tensors:
        w1_parts = thirds(tensors[f"{prefix}.lokr_w1_a"], f"{prefix}.lokr_w1_a")
        w1_key = "lokr_w1_a"
    else:
        raise ValueError(f"{prefix}: a LoKr without w1")
    for part, w1 in zip(PARTS, w1_parts, strict=True):
        target = f"{prefix[: -len('.qkv')]}.{part}"
        for key, value in tensors.items():
            if not key.startswith(prefix + "."):
                continue
            rest = key[len(prefix) + 1 :]
            out[f"{target}.{rest}"] = w1 if rest == w1_key else value


def convert(tensors: dict) -> tuple[dict, int]:
    """The LoRA with every fused qkv entry split, and how many weights were split."""
    prefixes = sorted({m["stem"] + ".qkv" for k in tensors if (m := FUSED.match(k))})
    out = {k: v for k, v in tensors.items() if not any(k.startswith(p + ".") for p in prefixes)}
    for prefix in prefixes:
        mine = {k: v for k, v in tensors.items() if k.startswith(prefix + ".")}
        if any(".lokr_" in k for k in mine):
            split_lokr(prefix, mine, out)
            continue
        for key, value in mine.items():
            rest = key[len(prefix) + 1 :]
            pieces = thirds(value, key) if rest in ROW_SPLIT else [value] * 3
            for part, piece in zip(PARTS, pieces, strict=True):
                out[f"{prefix[: -len('.qkv')]}.{part}.{rest}"] = piece
    return out, len(prefixes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("lora", type=Path)
    parser.add_argument("out", type=Path, nargs="?")
    args = parser.parse_args()
    out_path = args.out or args.lora.with_name(f"{args.lora.stem}_splitqkv.safetensors")
    if out_path.exists():
        print(f"{out_path} exists; not overwriting.", file=sys.stderr)
        return 1
    source = safe_open(args.lora, "pt", device="cpu")
    names = source.keys()
    tensors = {k: source.get_tensor(k) for k in names}
    try:
        converted, count = convert(tensors)
    except ValueError as e:
        print(f"Can't split this LoRA exactly: {e}", file=sys.stderr)
        return 1
    if not count:
        print("No fused qkv keys in this LoRA; nothing to do.")
        return 0
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    save_file(converted, temp, metadata=dict(source.metadata() or {}))
    os.replace(temp, out_path)
    print(f"Split {count} fused qkv weights into q, k and v → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
