"""Reference images a region reads on its own: clean latent tokens, seen only by that region.

Krea 2 reads a reference image the way the in-context edit LoRAs were trained (krea2edit, ostris'
ai-toolkit): the image VAE-encoded at the output's grid, appended as clean tokens at RoPE frame
1 (a second one at frame 2), with no change to the timestep. ComfyUI does exactly that for the
`reference_latents` of a conditioning with the `index` method, so a region's references travel
that way and this module only decides who may read them:

  * a region's references are read by its own canvas tokens, weighted by its claim on each token
    (as its copy of the prompt is), and by its copy of the prompt, and by nothing else — not the
    other regions, not the background, not the plain prompt everyone reads;
  * they read themselves, their region's copy of the prompt, the prompt and the canvas inside
    their region;
  * the region's LoRAs run on them too (`GateState.build_gates`), as an edit LoRA was trained to;
  * every region's subject sits at the same frame, so each looks like "the reference" to its own
    region — they never see each other, so sharing positions is safe.

A source reference (`region` None) is everyone's: an edit's source photo at frame 1, which every
token may read and no region's LoRA touches.

Positions follow the trained "fit" geometry: a reference smaller than the output grid along an
axis (a different aspect) sits centred at a fractional offset with stride 1, which is where the
trainer put it. ComfyUI starts every reference at 0, so `post_input` moves them.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

#: Krea 2's patch size: a latent cell pair per token along each axis.
PATCH = 2


class RefSlot(NamedTuple):
    #: The region that reads it, or None for a reference every token reads.
    region: int | None
    #: RoPE frame index. The target is frame 0.
    frame: int
    #: `(rows, cols)` in tokens.
    grid: tuple[int, int]


class RefSpan(NamedTuple):
    slot: RefSlot
    #: Where its tokens start in the full sequence (text first), and how many there are.
    start: int
    length: int


def token_grid(latent: torch.Tensor) -> tuple[int, int]:
    """`(rows, cols)` of tokens a latent `[B, C, H, W]` becomes, padded to the patch."""
    return math.ceil(latent.shape[-2] / PATCH), math.ceil(latent.shape[-1] / PATCH)


def layout(regions, sources=()) -> list[RefSlot]:
    """The references these regions add, in the order they join the conditioning: the sources
    (frame 1, one each), then each region's own (`ref_latents`, at its `ref_frame`)."""
    slots = [RefSlot(None, i + 1, token_grid(lat)) for i, lat in enumerate(sources)]
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        frame = int(region.get("ref_frame", 1))
        for lat in region.get("ref_latents") or []:
            slots.append(RefSlot(index, frame, token_grid(lat)))
    return slots


def latents(regions, sources=()) -> list[torch.Tensor]:
    """The latents for `layout`, in the same order."""
    out = list(sources)
    for region in regions:
        if isinstance(region, dict):
            out.extend(region.get("ref_latents") or [])
    return out


def spans(
    slots: list[RefSlot], entries: list[int], txtlen: int, canvas: int, copies: int
) -> list[RefSpan]:
    """Where `slots` sit in this forward's sequence, given every reference's token count
    (`reference_image_num_tokens`). Ours are the last ones. Empty when this forward doesn't carry
    them (the unconditional pass, a cached model, a conditioning that lost them)."""
    if not slots or len(entries) < len(slots):
        return []
    tail = entries[len(entries) - len(slots) :]
    if any(n != s.grid[0] * s.grid[1] for n, s in zip(tail, slots, strict=True)):
        return []
    start = txtlen + canvas + copies + sum(entries[: len(entries) - len(slots)])
    out = []
    for slot, n in zip(slots, tail, strict=True):
        out.append(RefSpan(slot, start, n))
        start += n
    return out


def place(img_ids: torch.Tensor, found: list[RefSpan], txtlen: int, target: tuple[int, int]):
    """`img_ids` with each of `found` at its frame and centred on the target grid. Indices in
    `found` are into the full sequence; `img_ids` is the image part, which starts at `txtlen`."""
    if not found:
        return img_ids
    out = img_ids.clone()
    rows, cols = target
    for span in found:
        gh, gw = span.slot.grid
        first = span.start - txtlen
        ids = out[:, first : first + span.length]
        if ids.shape[1] != gh * gw:
            continue
        h = torch.arange(gh, device=ids.device, dtype=ids.dtype) + max(0.0, (rows - gh) / 2)
        w = torch.arange(gw, device=ids.device, dtype=ids.dtype) + max(0.0, (cols - gw) / 2)
        ids[..., 0] = float(span.slot.frame)
        ids[..., 1] = h[:, None].expand(gh, gw).reshape(-1)
        ids[..., 2] = w[None, :].expand(gh, gw).reshape(-1)
    return out


def route(
    bias: torch.Tensor,
    found: list[RefSpan],
    claim: torch.Tensor,
    rows: slice,
    never: float,
    least: float,
    influences: list[float],
    copy_rows,
) -> None:
    """Fill `bias` `[B, L, L]` so each region's references are read by that region alone (see the
    module docstring). `claim` is `[canvas, regions]`; `copy_rows(region)` gives the rows of that
    region's copy of the prompt, or None."""
    own = [s for s in found if s.slot.region is not None]
    for span in own:
        bias[:, :, span.start : span.start + span.length] = never
    for span in own:
        cols = slice(span.start, span.start + span.length)
        region = span.slot.region
        weight = claim[:, region].clamp(min=least)
        logits = weight.log()
        influence = influences[region] if region < len(influences) else 1.0
        if influence != 1.0:
            logits = logits + torch.where(claim[:, region] > least, math.log(influence), 0.0)
        bias[:, rows, cols] = logits.unsqueeze(-1).to(bias.dtype)
        bias[:, cols, cols] = 0.0
        bias[:, cols, rows] = weight.log().unsqueeze(0).to(bias.dtype)
        copy = copy_rows(region)
        if copy is not None:
            bias[:, cols, copy] = 0.0
            bias[:, copy, cols] = 0.0


def attach(conditioning, refs: list[torch.Tensor]):
    """`conditioning` with `refs` appended to its reference latents, read the trained way
    (`index`) unless it already names a method."""
    if not refs:
        return conditioning
    out = []
    for cond, extra in conditioning:
        extra = dict(extra)
        extra["reference_latents"] = list(extra.get("reference_latents") or []) + list(refs)
        extra.setdefault("reference_latents_method", "index")
        out.append([cond, extra])
    return out
