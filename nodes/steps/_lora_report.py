"""What a LoRA actually did to a model: which weights it patched, with what kind of adapter, onto
which slice of a fused weight, and how far it moved each one.

Fused q/k/v checkpoints (MOSS-Image, Flux 3) take a LoRA trained on separate q, k and v projections
by patching thirds of the fused `qkv` weight — an offset on the patch. Whether that happens depends
on the LoRA's key names matching ComfyUI's map, and bypass mode can't apply offset patches at all;
either way the q/k/v part of the LoRA silently goes missing. This reads the model's own patch
list, so it shows what will run, not what was meant to.

The grouping is pure; the delta measurement needs torch and ComfyUI.
"""

from __future__ import annotations

import re
from collections import defaultdict

_BLOCK = re.compile(
    r"^(?:diffusion_model\.)?(?:layers|blocks|single_blocks|double_blocks|transformer_blocks)\.(\d+)\.(.+?)(?:\.weight|\.bias)?$"
)
QKV = ("q", "k", "v")


def target(key: str, offset) -> tuple[int | None, str]:
    """`(block, part)` a patch lands on: `(12, "proj.q")`, or `(12, "proj.qkv[k]")` for the k
    third of a fused weight. Weights outside the block stack give `(None, name)`."""
    found = _BLOCK.match(key)
    block, part = (
        (int(found.group(1)), found.group(2))
        if found
        else (None, key.removeprefix("diffusion_model.").removesuffix(".weight"))
    )
    if offset is not None and len(offset) == 3 and offset[2]:
        index = offset[1] // offset[2]
        part = f"{part}[{QKV[index] if part.endswith('qkv') and index < 3 else index}]"
    return block, part


def adapter_name(patch) -> str:
    value = patch[1]
    name = getattr(value, "name", None)
    if name:
        return str(name)
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return value[0]
    return type(value).__name__


def summarize(patches: dict) -> dict:
    """The model's patch list grouped: per part, which blocks it reaches; per adapter type and
    strength, how many patches; and how many land on a slice of a fused weight."""
    parts: dict[str, set[int]] = defaultdict(set)
    outside: list[str] = []
    kinds: dict[str, int] = defaultdict(int)
    sliced = 0
    for key, stack in patches.items():
        for patch in stack:
            offset = patch[3] if len(patch) > 3 else None
            block, part = target(key, offset)
            if block is None:
                outside.append(part)
            else:
                parts[part].add(block)
            kinds[f"{adapter_name(patch)} x{patch[0]:g}"] += 1
            sliced += offset is not None
    return {
        "parts": dict(parts),
        "outside": sorted(set(outside)),
        "kinds": dict(kinds),
        "sliced": sliced,
    }


def ranges(numbers) -> str:
    """`[0, 1, 2, 5]` → `"0-2, 5"`."""
    out, run = [], []
    for n in sorted(numbers):
        if run and n == run[-1] + 1:
            run.append(n)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [n]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


def coverage(summary: dict, blocks: int) -> list[str]:
    """One line per part: the blocks it reaches, and which it misses."""
    lines = []
    every = set(range(blocks))
    for part in sorted(summary["parts"]):
        reached = summary["parts"][part]
        missing = every - reached if blocks else set()
        line = f"  {part:24s} {len(reached):3d} blocks"
        if missing and len(missing) < blocks:
            line += f"  (missing {ranges(missing)})"
        lines.append(line)
    if summary["outside"]:
        lines.append("  outside the blocks: " + ", ".join(summary["outside"]))
    return lines
