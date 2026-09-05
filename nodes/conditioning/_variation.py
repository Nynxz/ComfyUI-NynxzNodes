"""Nudging a conditioning with seeded noise. Pure torch, no ComfyUI.

A "variation seed" for conditioning: perturb the token (and pooled) vectors by a seeded noise field
so you get neighbouring variations of the same prompt without touching the sampler seed.

Two normalizations, and both are load-bearing:

  * **The noise is scaled to each token's own magnitude** before it is added, so `strength` means
    the same thing regardless of how loud a given conditioning happens to be. Without it the same
    number is a whisper on one text encoder and a different prompt entirely on another.
  * **The result is rescaled back to the original magnitude.** Adding noise to a vector makes it
    longer on average — `|a + b|` exceeds `|a|` whenever they are not opposed — so an unnormalized
    nudge raises activation energy along with changing direction, and the drift you see is partly
    "louder" rather than "different". Renormalizing keeps the change purely directional, which is
    what makes a variation read as a variation of the same prompt instead of a stronger one.
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def _perturb(tensor: torch.Tensor, noise: torch.Tensor, strength: float) -> torch.Tensor:
    """`tensor` nudged toward `noise` by `strength`, at its original per-token magnitude."""
    norm = tensor.norm(dim=-1, keepdim=True)
    scaled = noise * (norm / (noise.norm(dim=-1, keepdim=True) + _EPS))
    varied = tensor + strength * scaled
    return varied / (varied.norm(dim=-1, keepdim=True) + _EPS) * norm


def vary_conditioning(conditioning, strength: float, seed: int):
    """Return a seeded variation of `conditioning`. `strength` 0 returns it unchanged."""
    if strength == 0.0:
        return conditioning
    # CPU generator so a seed reproduces regardless of the tensor's device.
    generator = torch.Generator().manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    out = []
    for tensor, meta in conditioning:
        noise = torch.randn(tensor.shape, generator=generator).to(tensor)
        varied = _perturb(tensor, noise, strength)
        new_meta = dict(meta) if meta else {}
        pooled = new_meta.get("pooled_output")
        if pooled is not None:
            pnoise = torch.randn(pooled.shape, generator=generator).to(pooled)
            new_meta["pooled_output"] = _perturb(pooled, pnoise, strength)
        out.append([varied, new_meta])
    return out
