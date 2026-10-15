"""The layout image a generation starts from, so it follows the boxes. Pure tensor work.

From noise, the first steps decide where everything goes, and the boxes only nudge that through
attention. Starting instead from a faint picture of the layout — each layer's first reference in
its box, or a soft tonal shape where it has none, over the background's — the model keeps that
coarse arrangement and paints over it. Everything is blurred hard first, so what carries over is
where things are and their rough colour, never a reference's pixels.

Fresco Apply encodes it and sets how much of the run follows it (`layout_start`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Mid-grey, the least assertive start where nothing is said.
BACKGROUND = 0.5
#: A layer with no reference is a slightly lighter shape: "something is here", no colour.
BLOB = 0.62
#: Blur radius as a share of the canvas's short side.
BLUR = 1 / 24


def _cover(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`image` ([1, h, w, C]) centre-cropped and resized to fill `height` x `width`, as `[3, H, W]`."""
    pixels = image[0, :, :, :3].permute(2, 0, 1).float()
    _, h, w = pixels.shape
    scale = max(height / h, width / w)
    resized = F.interpolate(
        pixels[None],
        size=(max(height, round(h * scale)), max(width, round(w * scale))),
        mode="bilinear",
        align_corners=False,
    )[0]
    top = (resized.shape[1] - height) // 2
    left = (resized.shape[2] - width) // 2
    return resized[:, top : top + height, left : left + width]


def _box(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    rows = torch.nonzero(mask.amax(dim=1) > 0.05).flatten()
    cols = torch.nonzero(mask.amax(dim=0) > 0.05).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def _blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    """A separable box blur, three passes (close to gaussian), on `[3, H, W]`."""
    if radius <= 0:
        return image
    size = 2 * radius + 1
    out = image[None]
    for _ in range(3):
        out = F.avg_pool2d(
            F.pad(out, (radius, radius, 0, 0), mode="replicate"), (1, size), stride=1
        )
        out = F.avg_pool2d(
            F.pad(out, (0, 0, radius, radius), mode="replicate"), (size, 1), stride=1
        )
    return out[0]


def compose(
    width: int,
    height: int,
    layers: list[tuple[torch.Tensor, torch.Tensor | None]],
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """The layout image, `[1, height, width, 3]` in 0..1.

    `layers` are `(mask, image)` bottom to top: the mask at any size (it is stretched to the
    canvas, as the regions are), the image the layer's first reference or None.
    """
    if background is not None:
        canvas = _cover(background, height, width)
    else:
        canvas = torch.full((3, height, width), BACKGROUND)
    for mask, image in layers:
        shape = F.interpolate(
            mask.float()[None, None], size=(height, width), mode="bilinear", align_corners=False
        )[0, 0].clamp(0.0, 1.0)
        box = _box(shape)
        if box is None:
            continue
        top, left, bottom, right = box
        fill = torch.full((3, height, width), BLOB)
        if image is not None:
            fill[:, top:bottom, left:right] = _cover(image, bottom - top, right - left)
        canvas = canvas * (1.0 - shape) + fill * shape
    radius = max(1, round(min(width, height) * BLUR))
    return _blur(canvas, radius).clamp(0.0, 1.0).permute(1, 2, 0)[None]
