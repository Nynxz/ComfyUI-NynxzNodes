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

Everything else is copied as is. Runs on the CPU. The Nynxz LoRA loaders do the same split on
the fly (nodes/lora/_qkv.py), so this is only for other loaders.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

_spec = importlib.util.spec_from_file_location(
    "qkv", Path(__file__).resolve().parents[1] / "nodes" / "lora" / "_qkv.py"
)
qkv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qkv)


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
        converted, count = qkv.split(tensors)
    except qkv.CannotSplit as e:
        print(f"Can't split this LoRA exactly: {e}", file=sys.stderr)
        return 1
    if not count:
        print("No fused qkv keys in this LoRA; nothing to do.")
        return 0
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    owned = {k: v.clone() for k, v in converted.items()}
    save_file(owned, temp, metadata=dict(source.metadata() or {}))
    os.replace(temp, out_path)
    print(f"Split {count} fused qkv weights into q, k and v → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
