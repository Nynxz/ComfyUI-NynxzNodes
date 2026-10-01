#!/usr/bin/env python3
"""Convert a δ-Vision embedding adapter checkpoint (.pt) to safetensors, and prove nothing changed.

The release is a pickled dict: `state_dict` (72 bf16 tensors), `adapter_config`, `global_step` and
the training `args`. Safetensors keeps the tensors under the same keys and carries the rest as
string metadata — only what `DeltaVisionAdapter.load` reads, plus the config for provenance. The
training `args` are dropped: they are cluster paths and hyperparameters, and the base model's name
is the only part anything downstream needs.

    python scripts/convert_delta_vision.py qwen_embedding_adapter_step2000.pt [-o out.safetensors]

After writing, the file is read back and every tensor compared bit for bit (dtype, shape, and the
raw bytes — `torch.equal` would call -0.0 == 0.0 and NaN != NaN). Exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's raw bytes, so equality is bitwise rather than numeric."""
    return tensor.contiguous().view(torch.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="δ-Vision .pt checkpoint")
    parser.add_argument("-o", "--out", default=None, help="default: next to the input, .safetensors")
    args = parser.parse_args()

    src = Path(args.checkpoint)
    dst = Path(args.out) if args.out else src.with_suffix(".safetensors")

    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    config = checkpoint.get("adapter_config", {})
    if config.get("output_mode") != "embedding_adapter":
        print(f"{src}: output_mode is {config.get('output_mode')!r}, not an embedding adapter")
        return 1
    extra = sorted(k for k in state if not k.startswith(("visual_adapter_down.", "visual_adapter_up.")))
    if extra:
        print(f"{src}: unexpected keys {extra[:5]} — refusing to drop them silently")
        return 1

    base = (checkpoint.get("args") or {}).get("model_path", "")
    metadata = {
        "format": "pt",
        "delta_vision.adapter_config": json.dumps(config, sort_keys=True),
        "delta_vision.global_step": str(checkpoint.get("global_step", "")),
        "delta_vision.base_model": Path(base).name if base else "",
        "delta_vision.source": src.name,
    }
    save_file({k: v.contiguous() for k, v in state.items()}, str(dst), metadata=metadata)

    bad: list[str] = []
    with safe_open(str(dst), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if keys != set(state):
            bad.append(f"key sets differ: missing {sorted(set(state) - keys)[:5]}, extra {sorted(keys - set(state))[:5]}")
        for key in sorted(keys & set(state)):
            a, b = state[key], f.get_tensor(key)
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(_bits(a), _bits(b)):
                bad.append(f"{key}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
        stored = f.metadata()

    dtypes = sorted({str(v.dtype) for v in state.values()})
    print(f"{src.name} -> {dst}  ({dst.stat().st_size / 2**20:.1f} MiB)")
    print(f"  {len(state)} tensors, {', '.join(dtypes)}; step {stored['delta_vision.global_step']}, base {stored['delta_vision.base_model']}")
    if bad:
        print("  MISMATCH:\n    " + "\n    ".join(bad))
        return 1
    print(f"  all {len(state)} tensors bit-identical to the .pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
