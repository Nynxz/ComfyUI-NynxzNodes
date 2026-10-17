"""How much of model A each weight takes, by how deep its block sits.

Two fine-tunes of one base (MOSS-Image's Base and SFT, say) can be blended weight by weight. In a
diffusion transformer the early blocks mostly settle the layout and the late ones the texture and
fine detail, so a graft can take one model's early blocks and the other's late ones, crossing over
smoothly rather than at a hard cut. Pure — no torch, no ComfyUI — so it's checked anywhere.
"""

from __future__ import annotations

import math
import re

STACKS = ("layers", "blocks", "single_blocks", "transformer_blocks", "double_blocks")
_BLOCK = re.compile(r"^(?:diffusion_model\.)?(" + "|".join(STACKS) + r")\.(\d+)\.")


def block_of(key: str) -> int | None:
    """The block a weight belongs to, or None when it's outside the block stack."""
    found = _BLOCK.match(key)
    return int(found.group(2)) if found else None


def depth(keys) -> int:
    """How many blocks the stack has."""
    numbers = [n for n in (block_of(k) for k in keys) if n is not None]
    return max(numbers) + 1 if numbers else 0


def ratio_at(
    block: float, blocks: int, early: float, late: float, crossover: float, softness: float
):
    """Model A's share in `block`: `early` at the top of the stack, `late` at the bottom, changing
    around `crossover` (a block number) over about `softness` blocks either side. Softness 0 is a
    hard cut."""
    if blocks <= 1:
        return early
    if softness <= 0:
        t = 0.0 if block < crossover else 1.0
    else:
        t = 1.0 / (1.0 + math.exp(-(block - crossover) / (softness / 3.0)))
    return early + (late - early) * t


_HEAD = re.compile(r"(final|head|norm_out|proj_out|out_layer)")
_INPUT = re.compile(r"^(video_embedder|x_embedder|img_in|patch_embed|pos_embed)")


def ratio_for(key: str, blocks: int, early: float, late: float, crossover: float, softness: float):
    """Model A's share of weight `key`, where weights outside the stack go with what they feed:
    the output head (its norm, projection and modulation together) with the last block, the
    input embedder with the first, and what every block reads — a shared modulation, the time
    and text embedders — at the stack's average share, so no block gets the other model's."""
    block = block_of(key)
    if block is not None:
        return ratio_at(block, blocks, early, late, crossover, softness)
    if _HEAD.search(key):
        return late
    if _INPUT.match(key):
        return early
    shares = curve(blocks, early, late, crossover, softness)
    return sum(shares) / len(shares) if shares else early


def curve(blocks: int, early: float, late: float, crossover: float, softness: float) -> list[float]:
    return [ratio_at(i, blocks, early, late, crossover, softness) for i in range(blocks)]
