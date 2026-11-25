"""The attention-bias pieces every family's routing is built from: who reads which prompt copy, and
isolation. Each family lays its sequence out differently and says where things are; what goes
there is the same on all of them."""

from __future__ import annotations

import math

import torch

#: A logit bias that is "never": far below any real logit, and still finite in fp16.
NEVER = -1e4
#: The smallest weight a canvas token gives a prompt, so `log` stays finite.
LEAST = 1e-4


def isolation(claim: torch.Tensor, groups: torch.Tensor, strength: float) -> torch.Tensor:
    """`[canvas, canvas]` — the logit penalty keeping regions' canvas tokens off each other's.

    `claim` is `[canvas, regions]`. Renormalized over regions only: strengths and the schedule are
    about how hard a LoRA fires, the bias only wants to know which region a token belongs to.
    Regions in one group (`groups`, `[regions, groups]`) are one subject and agree. Only region
    against region: a background token belongs to no region, so by agreement alone it would be
    pushed away from every token, itself included — which takes the scene's coherence (lighting,
    contact, perspective) with it.
    """
    share = claim / claim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    share = share @ groups.to(share.device, share.dtype)
    agreement = (share @ share.transpose(0, 1)).clamp(0.0, 1.0)
    claimed = claim.sum(dim=-1).clamp(0.0, 1.0)
    both = claimed.unsqueeze(1) * claimed.unsqueeze(0)
    return -strength * (1.0 - agreement) * both


def route_copies(bias, claim, rows, slots, hidden, shared, influences, also_read=(), scale=1.0):
    """Who reads which prompt, written into `bias` (`[batch, seq, seq]`).

    `slots` is `[(region, columns)]`, one per prompt copy. Each copy reads itself, the canvas and
    `also_read`, never `hidden` (the rest of the text). The canvas tokens (`rows`) read each copy by
    their region's claim on them times `scale`, as `log(weight)` — so a token wholly inside one
    region reads only its copy — and the `shared` spans (the main prompt) for whatever no copied
    region claims. A region's `influences` entry multiplies the attention its tokens give its copy.
    """
    taken = torch.zeros_like(claim[:, 0])
    for region, copy in slots:
        bias[:, copy, hidden] = NEVER
        bias[:, copy, copy] = 0.0
        for span in also_read:
            bias[:, copy, span] = 0.0
        weight = claim[:, region] * scale
        taken = taken + weight
        logits = weight.clamp(min=LEAST).log()
        influence = influences[region] if region < len(influences) else 1.0
        if influence != 1.0:
            # Against the image and every other key, not just the main prompt — and only where
            # the region holds.
            logits = logits + torch.where(weight > LEAST, math.log(influence), 0.0)
        bias[:, rows, copy] = logits.unsqueeze(-1).to(bias.dtype)
    rest = (1.0 - taken).clamp(min=LEAST).log().unsqueeze(-1).to(bias.dtype)
    for span in shared:
        bias[:, rows, span] = rest


def hide_padding(bias: torch.Tensor, valid: torch.Tensor | None, start: int) -> None:
    """Nobody reads a copy's padding: `valid` (`[batch, copy tokens]`, False on padding) covers the
    copies, which start at column `start`."""
    if valid is not None:
        copies = bias[:, :, start : start + valid.shape[1]]
        copies.masked_fill_(~valid.to(bias.device).unsqueeze(1), NEVER)
