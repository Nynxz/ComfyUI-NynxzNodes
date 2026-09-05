"""Render the fusion weight field to an image — see which source owns which token-grid cell.

The encode captures the exact `[num_sources, tokens]` field it blended with (via `_compute_weights`,
`debug["weights"]`); this turns it into a picture.

One picture can't answer every question, so there are three views (`weight_map_view` on the encode):

* `dominant` (default) — each cell takes its winning source's colour, dimmed by how *contested* it
  is (top weight minus runner-up). Decisive cells are vivid, near-ties go dark, so boundaries pop.
  Read it as "who owns what".
* `per-source` — small multiples: one panel per source, each showing only that source's weight as a
  black→colour ramp. Nothing is ever mixed, so you can see exactly where one source's tokens land.
* `blend` — the legacy convex mix (every cell is the weighted *average* of all source colours). Kept
  for continuity, but it's ambiguous by construction: different weight splits can land on the same
  colour, and heavy mixes wash out toward grey. That's why it isn't the default any more.

The `field_to_*` functions (weights -> arrays) are pure, so the colour mapping is unit-tested; the
PIL upscale, cell separators and legend around them are presentation only.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

# Per-source colours (0..1), matching the Fusion Regions Preview palette.
PALETTE = [
    (1.00, 0.35, 0.35),
    (0.35, 0.78, 1.00),
    (0.55, 1.00, 0.47),
    (1.00, 0.82, 0.31),
    (0.86, 0.51, 1.00),
    (1.00, 0.59, 0.78),
]

WEIGHT_MAP_VIEWS = ["dominant", "per-source", "blend"]
DEFAULT_VIEW = "dominant"

_BG = (24, 24, 24)
_TEXT = (230, 230, 230)


def field_to_rgb(weights, visual_height, visual_width, palette=PALETTE) -> np.ndarray:
    """Convex-combine per-source colours by their weights -> an [H, W, 3] float array in [0,1].

    `weights` is [num_sources, tokens] with columns summing to 1, so the result stays in gamut.
    This is the legacy `blend` view: compact, but a mix is not invertible back to its sources.
    """
    weights = np.asarray(weights, dtype=np.float32)
    num_sources = weights.shape[0]
    colors = np.array([palette[i % len(palette)] for i in range(num_sources)], dtype=np.float32)
    rgb = weights.T @ colors  # [tokens, 3]
    return rgb.reshape(visual_height, visual_width, 3).clip(0.0, 1.0)


def field_to_dominant_rgb(
    weights, visual_height, visual_width, palette=PALETTE, floor=0.22
) -> np.ndarray:
    """Winner-take-all -> an [H, W, 3] float array in [0,1]. No colour mixing at all.

    Each cell takes its top source's colour, scaled by that source's actual share of the cell. A
    cell one source owns outright renders at full colour; a contested cell is necessarily dimmer,
    because the winner's share shrinks the more sources fight over it. So hue answers "who owns
    this" and brightness answers "how much", which the convex mix in `field_to_rgb` collapses into
    a single ambiguous colour.

    (Scaling by the *margin* to the runner-up instead reads as a black halo around every soft edge,
    which looks like a rendering artifact — the winner's share carries the same signal cleanly.)
    """
    w = np.asarray(weights, dtype=np.float32)
    num_sources = w.shape[0]
    colors = np.array([palette[i % len(palette)] for i in range(num_sources)], dtype=np.float32)
    owner = w.argmax(axis=0)  # [tokens]
    share = np.clip(w.max(axis=0), 0.0, 1.0)
    conf = floor + (1.0 - floor) * share
    rgb = colors[owner] * conf[:, None]
    return rgb.reshape(visual_height, visual_width, 3).clip(0.0, 1.0)


def field_to_panels(weights, visual_height, visual_width, palette=PALETTE) -> list[np.ndarray]:
    """One [H, W, 3] float array per source: that source's own weight as a black→colour ramp.

    Small multiples — the overplotting-free view. Weight 0 is black, weight 1 is the source's full
    colour, and no cell ever carries more than one source, so nothing can be misread as a mixture.
    """
    w = np.asarray(weights, dtype=np.float32)
    panels = []
    for i in range(w.shape[0]):
        color = np.array(palette[i % len(palette)], dtype=np.float32)
        amount = w[i].reshape(visual_height, visual_width, 1).clip(0.0, 1.0)
        panels.append((amount * color).clip(0.0, 1.0))
    return panels


def _label_for(labels, i: int) -> str:
    label = str(labels[i]) if labels and i < len(labels) and labels[i] else f"src {i}"
    return label[:17] + "…" if len(label) > 18 else label


def _cells_to_image(rgb: np.ndarray, cell: int) -> Image.Image:
    """[H,W,3] float -> nearest-upscaled PIL, with faint separators so cells stay countable."""
    height, width = rgb.shape[0], rgb.shape[1]
    img = Image.fromarray((rgb * 255.0).astype(np.uint8), "RGB")
    img = img.resize((width * cell, height * cell), Image.NEAREST)
    if cell >= 10:
        draw = ImageDraw.Draw(img)
        for gx in range(1, width):
            draw.line((gx * cell, 0, gx * cell, img.height), fill=(0, 0, 0), width=1)
        for gy in range(1, height):
            draw.line((0, gy * cell, img.width, gy * cell), fill=(0, 0, 0), width=1)
    return img


def _draw_legend(canvas: Image.Image, x: int, y: int, num_sources: int, labels, swatch=18) -> None:
    draw = ImageDraw.Draw(canvas)
    for i in range(num_sources):
        color = tuple(int(c * 255) for c in PALETTE[i % len(PALETTE)])
        draw.rectangle((x, y, x + swatch, y + swatch), fill=color)
        label = _label_for(labels, i)
        draw.text((x + swatch + 4, y + 4), label, fill=_TEXT)
        x += swatch + 8 + len(label) * 6 + 10


def _render_single(rgb: np.ndarray, num_sources: int, labels, cell: int) -> Image.Image:
    """One grid + a legend naming each source's colour."""
    grid = _cells_to_image(rgb, cell)
    legend_h = 26
    canvas = Image.new("RGB", (grid.width, grid.height + legend_h), _BG)
    canvas.paste(grid, (0, 0))
    _draw_legend(canvas, 6, grid.height + 4, num_sources, labels)
    return canvas


def _render_panels(panels: list[np.ndarray], labels, cell: int, columns=3) -> Image.Image:
    """Small multiples: each source's panel titled in its own colour, wrapped into a grid."""
    pad, title_h = 8, 16
    # Panels are shown side by side, so shrink the cells to keep the sheet a sane size.
    panel_cell = max(6, cell // 2)
    imgs = [_cells_to_image(p, panel_cell) for p in panels]
    cols = max(1, min(columns, len(imgs)))
    rows = (len(imgs) + cols - 1) // cols
    step_x = imgs[0].width + pad
    step_y = imgs[0].height + title_h + pad
    canvas = Image.new("RGB", (cols * step_x + pad, rows * step_y + pad), _BG)
    draw = ImageDraw.Draw(canvas)
    for i, img in enumerate(imgs):
        row, col = divmod(i, cols)
        x, y = pad + col * step_x, pad + row * step_y
        color = tuple(int(c * 255) for c in PALETTE[i % len(PALETTE)])
        draw.text((x, y + 3), _label_for(labels, i), fill=color)
        canvas.paste(img, (x, y + title_h))
    return canvas


def render_weight_field(
    weights, visual_height, visual_width, labels=None, cell=26, view=DEFAULT_VIEW
) -> torch.Tensor:
    """The weight field as a [1, H, W, 3] IMAGE. See the module docstring for the three views."""
    w = np.asarray(weights, dtype=np.float32)
    num_sources = int(w.shape[0])
    if view == "per-source":
        canvas = _render_panels(field_to_panels(w, visual_height, visual_width), labels, cell)
    else:
        rgb = (
            field_to_rgb(w, visual_height, visual_width)
            if view == "blend"
            else field_to_dominant_rgb(w, visual_height, visual_width)
        )
        canvas = _render_single(rgb, num_sources, labels, cell)
    frame = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0)
    return frame.unsqueeze(0)
