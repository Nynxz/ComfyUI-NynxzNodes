"""A LoRA trained on fused q/k/v weights, fitted to a model whose q, k and v are separate.

ComfyUI places a LoRA's separate q/k/v keys onto a fused `qkv` weight (MOSS-Image, Flux 3) but not
the other way: on a split checkpoint a fused-qkv LoRA's whole attention update is dropped as "lora
key not loaded". This splits each such entry into `q`, `k` and `v`, exactly:

    LoRA / DoRA   the up matrix (and DoRA's scale) cut into thirds by row; down and alpha shared
    LoHa          each Hadamard factor's left matrix cut into thirds by row
    LoKr          the first Kronecker factor cut into thirds by row (needs its rows to divide by 3)

Only entries whose target the model lacks as `qkv` but has as `q` are split, so a fused model, or
a LoRA with no fused keys, passes through untouched. Pure torch; no ComfyUI.
"""

from __future__ import annotations

import re

#: `...proj.qkv.lora_up.weight` or `lora_unet_..._proj_qkv.lora_up.weight`.
FUSED = re.compile(r"^(?P<stem>.*?[._]proj)(?P<sep>[._])qkv\.(?P<rest>.+)$")
PARTS = ("q", "k", "v")

#: Tensors cut into thirds by row. Everything else under a qkv prefix is shared by the three.
ROW_SPLIT = (
    "lora_up.weight",
    "lora_B.weight",
    "lora.up.weight",
    "dora_scale",
    "hada_w1_a",
    "hada_w2_a",
)


class CannotSplit(ValueError):
    """A fused entry that can't be split exactly."""


def thirds(tensor, name: str) -> list:
    rows = tensor.shape[0]
    if rows % 3:
        raise CannotSplit(f"{name}: {rows} rows don't split into thirds")
    third = rows // 3
    return [tensor[i * third : (i + 1) * third].contiguous() for i in range(3)]


def _targets(prefix: str, sep: str) -> list[str]:
    return [f"{prefix[: -len(sep + 'qkv')]}{sep}{part}" for part in PARTS]


def _split_lokr(prefix: str, sep: str, mine: dict, out: dict) -> None:
    """LoKr's update is kron(w1, w2): a third of its rows is a third of w1's rows."""
    for name in ("lokr_w1", "lokr_w1_a"):
        if f"{prefix}.{name}" in mine:
            pieces, key = thirds(mine[f"{prefix}.{name}"], f"{prefix}.{name}"), name
            break
    else:
        raise CannotSplit(f"{prefix}: a LoKr without w1")
    for target, w1 in zip(_targets(prefix, sep), pieces, strict=True):
        for full, value in mine.items():
            rest = full[len(prefix) + 1 :]
            out[f"{target}.{rest}"] = w1 if rest == key else value


def fused_prefixes(state: dict) -> dict[str, str]:
    """`{prefix: separator}` for every fused qkv entry in a LoRA's state dict."""
    found = {}
    for key in state:
        match = FUSED.match(key)
        if match:
            found[f"{match['stem']}{match['sep']}qkv"] = match["sep"]
    return found


def split(state: dict, wanted=lambda prefix: True) -> tuple[dict, int]:
    """`state` with every fused qkv entry `wanted(prefix)` approves split into q, k and v, and how
    many were split. Raises CannotSplit when one can't be split exactly."""
    prefixes = {p: s for p, s in fused_prefixes(state).items() if wanted(p)}
    if not prefixes:
        return state, 0
    out = {k: v for k, v in state.items() if not any(k.startswith(p + ".") for p in prefixes)}
    for prefix, sep in prefixes.items():
        mine = {k: v for k, v in state.items() if k.startswith(prefix + ".")}
        if any(".lokr_" in k for k in mine):
            _split_lokr(prefix, sep, mine, out)
            continue
        for full, value in mine.items():
            rest = full[len(prefix) + 1 :]
            pieces = thirds(value, full) if rest in ROW_SPLIT else [value] * 3
            for target, piece in zip(_targets(prefix, sep), pieces, strict=True):
                out[f"{target}.{rest}"] = piece
    return out, len(prefixes)


def for_model(state: dict, key_map: dict) -> tuple[dict, int]:
    """`state` fitted to the model behind `key_map` (ComfyUI's LoRA key map): a fused entry is
    split when the model has no place for it but has one for its `q`."""

    def wanted(prefix: str) -> bool:
        sep = prefix[-4]
        return prefix not in key_map and _targets(prefix, sep)[0] in key_map

    return split(state, wanted)
