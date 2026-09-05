"""Drawing a region map as a picture. Shared by the two diagnostic nodes so they read alike.

One palette and one upscale, used before sampling by Region Inspect and after it by Regions
Preview — a region has to be the same colour in both, or comparing them means nothing.
"""

from __future__ import annotations

import torch

#: Hues assigned to regions in order, as RGB. Distinct at a glance and distinguishable in the
#: common forms of colour blindness — the map is read by shape, but the labels are read by colour.
HUES = (
    (0.95, 0.35, 0.20),  # orange-red
    (0.20, 0.55, 0.95),  # blue
    (0.30, 0.75, 0.35),  # green
    (0.85, 0.75, 0.20),  # yellow
    (0.75, 0.35, 0.85),  # purple
    (0.25, 0.80, 0.80),  # cyan
)


def hue(index: int) -> torch.Tensor:
    return torch.tensor(HUES[index % len(HUES)])


def upscale(grid: torch.Tensor, cell: int) -> torch.Tensor:
    """`[h, w, 3]` -> `[h*cell, w*cell, 3]` by nearest neighbour.

    Nearest rather than bilinear on purpose: a canvas token IS a block, and smoothing the render
    would suggest a spatial precision the map does not have.
    """
    return grid.repeat_interleave(cell, dim=0).repeat_interleave(cell, dim=1)


def overlay(claim: torch.Tensor) -> torch.Tensor:
    """`[h, w, regions]` -> an RGB overlay: hue says who claims a cell, brightness says how hard.

    Both facts matter and they are independent — a cell can be confidently region 1's and barely
    claimed by anybody, and showing only the hue would make those look the same.
    """
    height, width, count = claim.shape
    share = claim / claim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    out = torch.zeros(height, width, 3)
    for index in range(count):
        out += share[..., index : index + 1] * hue(index)
    brightness = claim.amax(dim=-1, keepdim=True).clamp(0.0, 1.0)
    return (out * (0.15 + 0.85 * brightness)).clamp(0.0, 1.0)
