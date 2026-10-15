"""Laying recorded steps out as frames: each run's picture over a strip with its label, its step and
a progress bar. Pure tensor and PIL work, no model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

SIDE_BY_SIDE = "side by side"
STACKED = "stacked"
LAYOUTS = (SIDE_BY_SIDE, STACKED)

STRIP = (17, 17, 19)
TRACK = (52, 52, 58)
FILL = (232, 163, 61)
DONE = (120, 200, 120)
TEXT = (230, 230, 230)
DIM = (150, 150, 158)
GAP = 4


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def strip(width: int, label: str, step: int, total: int, finished: bool) -> torch.Tensor:
    """The strip under one picture, `[h, width, 3]` in 0..1. `step` counts from 1; `finished`
    is the sampler's final output, after the last step."""
    size = max(12, width // 48)
    bar = max(4, size // 3)
    pad = max(4, size // 2)
    height = pad + size + pad + bar + pad
    image = Image.new("RGB", (width, height), STRIP)
    draw = ImageDraw.Draw(image)
    font = _font(size)
    if label:
        draw.text((pad, pad), label, font=font, fill=TEXT)
    status = "result" if finished else f"step {step}/{total}"
    right = draw.textlength(status, font=font)
    draw.text((width - pad - right, pad), status, font=font, fill=DIM)

    top = pad + size + pad
    left, span = pad, max(1, width - 2 * pad)
    draw.rectangle((left, top, left + span, top + bar - 1), fill=TRACK)
    share = 1.0 if finished else (step / total if total else 0.0)
    if share > 0:
        draw.rectangle(
            (left, top, left + round(span * share), top + bar - 1), fill=DONE if finished else FILL
        )
    if 1 < total <= span // 6:
        for k in range(1, total):
            x = left + round(span * k / total)
            draw.line((x, top, x, top + bar - 1), fill=STRIP)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)


def panel(picture: torch.Tensor, label: str, step: int, total: int, finished: bool) -> torch.Tensor:
    """One run's frame: `picture` `[H, W, 3]` over its strip."""
    under = strip(picture.shape[1], label, step, total, finished)
    return torch.cat([picture[..., :3].clamp(0, 1).float(), under], dim=0)


def fit(frame: torch.Tensor, height: int | None = None, width: int | None = None) -> torch.Tensor:
    """`frame` `[H, W, 3]` scaled to the given height or width, keeping its shape."""
    h, w = frame.shape[:2]
    if height is not None and h != height:
        size = (height, max(1, round(w * height / h)))
    elif width is not None and w != width:
        size = (max(1, round(h * width / w)), width)
    else:
        return frame
    scaled = F.interpolate(
        frame.permute(2, 0, 1)[None], size=size, mode="bilinear", align_corners=False
    )
    return scaled[0].permute(1, 2, 0)


def join(panels: list[torch.Tensor], layout: str) -> torch.Tensor:
    """Panels beside or above each other, with a dark gap between, scaled to the first's size."""
    if len(panels) == 1:
        return panels[0]
    first = panels[0]
    if layout == STACKED:
        rows = [first] + [fit(p, width=first.shape[1]) for p in panels[1:]]
        gap = torch.empty(GAP, first.shape[1], 3).copy_(torch.tensor(STRIP) / 255.0)
        parts = [rows[0]]
        for row in rows[1:]:
            parts += [gap, row]
        return torch.cat(parts, dim=0)
    cols = [first] + [fit(p, height=first.shape[0]) for p in panels[1:]]
    gap = torch.empty(first.shape[0], GAP, 3).copy_(torch.tensor(STRIP) / 255.0)
    parts = [cols[0]]
    for col in cols[1:]:
        parts += [gap, col]
    return torch.cat(parts, dim=1)


def even(frame: torch.Tensor) -> torch.Tensor:
    """Cropped to even sides: most video encoders refuse odd ones."""
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]
