"""Structure canvas — a seeded, smooth random color field to start a sampler from. Pure torch.

The idea is the "draft → partial denoise" trick with the draft model swapped for randomness: a
few-step model started at denoise ~0.8–0.9 treats the start latent's large-scale structure as
IMAGE and builds on it, where at σ = 1 it treats it as noise and falls back to its usual layout. So
a different canvas per seed gives a different composition per seed — where light, dark, mass and
color sit — while the model still decides what goes there.

The canvas is made in pixel space and VAE-encoded by the node, so the latent has real image
statistics rather than a guess at them.

Construction: octaves of bicubically upsampled random grids (value noise), in a luma/chroma space:
  * luma carries the composition (light vs dark masses), scaled by `contrast` around mid-grey;
  * two chroma channels carry the palette, scaled by `color` (0 = greyscale canvas);
  * `scale` is the size of the largest blobs as a fraction of the short side; each further octave
    halves the size and multiplies the amplitude by `detail`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# BT.601 YCbCr → RGB, with chroma centred on 0.
_YCC_TO_RGB = torch.tensor(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ]
)


def _octave(generator, channels, height, width, cell):
    """One layer of value noise: a random grid with `cell`-px spacing, upsampled bicubically."""
    grid_h = max(2, math.ceil(height / cell) + 1)
    grid_w = max(2, math.ceil(width / cell) + 1)
    grid = torch.randn((1, channels, grid_h, grid_w), generator=generator)
    up = F.interpolate(
        grid, size=(grid_h * cell, grid_w * cell), mode="bicubic", align_corners=False
    )
    return up[..., :height, :width]


def _field(generator, channels, height, width, scale, octaves, detail):
    """Sum of octaves, normalized to zero mean / unit std per channel."""
    cell = max(2, round(scale * min(height, width)))
    total = torch.zeros((1, channels, height, width))
    amplitude = 1.0
    for _ in range(max(1, octaves)):
        total += amplitude * _octave(generator, channels, height, width, cell)
        amplitude *= detail
        cell = max(2, cell // 2)
    total -= total.mean(dim=(2, 3), keepdim=True)
    return total / (total.std(dim=(2, 3), keepdim=True) + 1e-8)


def structure_canvas(
    width: int,
    height: int,
    seed: int,
    scale: float = 0.35,
    octaves: int = 2,
    detail: float = 0.4,
    contrast: float = 0.12,
    color: float = 0.05,
    brightness: float = 0.5,
    batch_size: int = 1,
) -> torch.Tensor:
    """`[B, H, W, 3]` float image in 0..1 (ComfyUI IMAGE layout). Item `i` uses seed `seed + i`."""
    images = []
    for index in range(batch_size):
        generator = torch.Generator().manual_seed((int(seed) + index) & 0xFFFFFFFFFFFFFFFF)
        field = _field(generator, 3, height, width, scale, octaves, detail)[0]  # [3, H, W]
        luma = brightness + contrast * field[0]
        cb, cr = color * field[1], color * field[2]
        rgb = torch.einsum("ij,jhw->ihw", _YCC_TO_RGB, torch.stack([luma, cb, cr]))
        images.append(rgb.clamp(0.0, 1.0).permute(1, 2, 0))
    return torch.stack(images)
