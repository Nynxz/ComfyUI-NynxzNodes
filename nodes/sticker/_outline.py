"""Print-ready sticker outlines: binary alpha, smoothed edge, optional die-cut border.

Pure math, no ComfyUI imports, so scripts/check_sticker_outline.py can run it directly.
Sizes are given in source pixels and scaled by `upscale` internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label

# An alpha at or above this counts as fully opaque when picking edge colours.
SOLID_ALPHA = 0.99
# How far in from the transparent area (source px) semi-transparent pixels count as fringe.
# Deeper partial alpha is left alone: it is usually real texture, not an edge.
EDGE_BAND = 3.0
# Below this alpha, pixels connected to the transparent area count as edge halo (drawn outline,
# shadow seams in narrow gaps) that close_gaps / fill_holes may paint over.
HALO_ALPHA = 0.9

# What happens to the semi-transparent outline ring: recolour it from the solid pixels inside,
# paint it with fill_color (for art with its own drawn border), or leave it as drawn.
EDGE_MODES = ["blend", "fill", "keep"]

# Blur (px) for the antialiased edge's sub-pixel contour. ~1 gives a clean 1px ramp.
AA_SIGMA = 1.0


@dataclass(frozen=True)
class OutlineSettings:
    threshold: float = 0.5
    smoothness: float = 2.0
    min_speck_size: int = 64
    fill_holes: bool = False
    close_gaps: float = 0.0
    # Solid colour for area close_gaps / fill_holes add; None blends from the surroundings.
    fill_color: tuple[float, float, float] | None = (1.0, 1.0, 1.0)
    keep_largest: bool = False
    upscale: int = 1
    border_width: float = 0.0
    border_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    edge: str = "blend"
    antialias: bool = False


def parse_hex_color(value: str, name: str = "border_color") -> tuple[float, float, float]:
    """`#rgb` or `#rrggbb` (hash optional) to floats in [0, 1]."""
    error = f"{name} must be a hex colour like #ffffff, got {value!r}"
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise ValueError(error)
    try:
        return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        raise ValueError(error) from None


def parse_fill_color(value: str) -> tuple[float, float, float] | None:
    """A hex colour, or "blend" / empty for a blend of the surrounding colours."""
    if value.strip().lower() in ("", "blend"):
        return None
    return parse_hex_color(value, "fill_color")


def _drop_small(shape: np.ndarray, min_area: float) -> np.ndarray:
    """Remove islands smaller than `min_area` pixels, and fill holes smaller than it."""
    if min_area <= 0:
        return shape
    labels, count = label(shape)
    if count:
        areas = np.bincount(labels.ravel())
        keep = areas >= min_area
        keep[0] = False
        shape = keep[labels]
    # Holes: background components that don't touch the canvas edge.
    labels, count = label(~shape)
    if count:
        areas = np.bincount(labels.ravel())
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        fill = areas < min_area
        fill[0] = False
        fill[edge] = False
        shape = shape | fill[labels]
    return shape


def _largest(shape: np.ndarray) -> np.ndarray:
    """Only the biggest connected piece."""
    labels, count = label(shape)
    if count <= 1:
        return shape
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    return labels == areas.argmax()


def _halo(alpha: np.ndarray, threshold: float) -> np.ndarray:
    """The transparent area plus the semi-transparent pixels connected to it.

    That is the drawn outline and any shadow seams in narrow gaps, but not partial alpha
    enclosed inside the art.
    """
    labels, _ = label(alpha < max(HALO_ALPHA, threshold))
    touching = np.unique(labels[alpha < threshold])
    return np.isin(labels, touching[touching > 0])


def _fill_gaps(
    shape: np.ndarray, halo: np.ndarray, s: OutlineSettings, u: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply close_gaps and fill_holes. Returns the new shape and the area they filled.

    Closing runs on the shape minus its edge halo, so a gap lined with a semi-transparent seam
    closes over the seam too, instead of leaving it as a grey line.
    """
    if s.close_gaps <= 0 and not s.fill_holes:
        return shape, np.zeros_like(shape)
    core = shape & ~halo
    closed = _close(core, s.close_gaps * u)
    filled = shape | closed
    if s.fill_holes:
        filled = binary_fill_holes(filled)
    return filled, (closed & ~core) | (filled & ~(shape | closed))


def _smooth(shape: np.ndarray, sigma: float) -> np.ndarray:
    """Blur then re-threshold at 0.5: rounds jaggies and notches, result stays binary."""
    if sigma <= 0:
        return shape
    return gaussian_filter(shape.astype(np.float32), sigma, mode="nearest") >= 0.5


def _close(shape: np.ndarray, radius: float) -> np.ndarray:
    """Round morphological closing: fills bays and gaps narrower than `2 * radius`."""
    if radius <= 0 or not shape.any():
        return shape
    pad = int(np.ceil(radius)) + 1  # room to grow, so the canvas edge doesn't stop the erosion
    padded = np.pad(shape, pad)
    grown = distance_transform_edt(~padded) <= radius
    closed = distance_transform_edt(grown) > radius
    return (closed | padded)[pad:-pad, pad:-pad]


def _push_pull(rgb: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fill where `weight` is 0 with a smooth blend of the weighted colours around it.

    Averages down a pyramid until every cell has some weight, then blends back up. Unlike a
    nearest-pixel fill this gives soft gradients, not streaks, across wide gaps.
    """
    c = (rgb * weight[..., None]).permute(2, 0, 1)[None]
    w = weight[None, None]
    levels = [(c, w)]
    while min(c.shape[-2:]) > 1:
        c = F.avg_pool2d(c, 2, ceil_mode=True)
        w = F.avg_pool2d(w, 2, ceil_mode=True)
        levels.append((c, w))
    filled = levels[-1][0] / levels[-1][1].clamp_min(1e-8)
    for c, w in reversed(levels[:-1]):
        up = F.interpolate(filled, size=c.shape[-2:], mode="bilinear", align_corners=False)
        own = c / w.clamp_min(1e-8)
        k = w.clamp(0.0, 1.0)
        filled = own * k + up * (1.0 - k)
    return filled[0].permute(1, 2, 0)


def _fix_colors(rgb: torch.Tensor, alpha: np.ndarray, s: OutlineSettings) -> torch.Tensor:
    """Recolour untrusted pixels from the trusted colours around them.

    Untrusted: everything below the threshold (smoothing, closing and filling can pull it into
    the shape) and, unless edge is "keep", the semi-transparent fringe near the transparent area,
    whose straight colour is often grey or matted. Interior partial alpha keeps its own colour.
    """
    outside = alpha < s.threshold
    untrusted = outside.copy()
    if s.edge != "keep" and outside.any():
        near_edge = distance_transform_edt(~outside) <= EDGE_BAND
        untrusted |= near_edge & (alpha < SOLID_ALPHA)
    trusted = ~untrusted
    if not trusted.any() or not untrusted.any():
        return rgb
    keep = torch.from_numpy(trusted)
    return torch.where(keep[..., None], rgb, _push_pull(rgb, keep.float()))


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """[H, W, C] resized to `size`, bicubic, clamped to [0, 1]."""
    if x.shape[:2] == size:
        return x
    y = F.interpolate(x.permute(2, 0, 1)[None], size=size, mode="bicubic", align_corners=False)
    return y[0].permute(1, 2, 0).clamp(0.0, 1.0)


def _shape_masks(
    alpha: torch.Tensor, s: OutlineSettings, m: int, pad: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cut shape at `m` times source size, padded by `pad` source px.

    Returns (shape, paint, final): the sticker without border, the area to paint with
    fill_color, and the shape with its border. All binary.
    """
    h, w = alpha.shape
    a = _resize(alpha[..., None], (h * m, w * m))[..., 0].numpy()
    if pad:
        a = np.pad(a, pad * m, mode="constant")

    min_area = s.min_speck_size * m * m
    sigma = s.smoothness * m
    band = EDGE_BAND * m
    shape = a >= s.threshold
    shape = _drop_small(shape, min_area)
    if s.keep_largest:
        shape = _largest(shape)  # before smoothing and closing, so strays can't join on
    shape = _smooth(shape, sigma)
    shape = _drop_small(shape, min_area)  # smoothing can pinch thin parts into islands

    halo = _halo(a, s.threshold)
    before = shape
    shape, added = _fill_gaps(shape, halo, s, m)
    if added.any() and sigma > 0:
        # Smooth again where fills meet the old edge, so the outline flows through the join.
        near = distance_transform_edt(~added) <= 3 * sigma + 1
        shape = np.where(near, _smooth(shape, sigma), shape)
        added = shape & (added | (~before & near))
    if s.keep_largest:
        shape = _largest(shape)
    added &= shape

    paint = np.zeros_like(shape)
    if s.fill_color is not None:
        # The old edge: the outline ring of the shape before any gaps were filled.
        old_edge = shape & (distance_transform_edt(~(halo | ~before)) <= band)
        paint |= added
        if added.any():
            # The old edge lining a filled gap is inside the sticker now; cover it too.
            paint |= old_edge & (distance_transform_edt(~added) <= 2 * band)
        if s.edge == "fill":
            paint |= shape & (halo | (old_edge & (a < SOLID_ALPHA)))

    final = shape
    if s.border_width > 0 and shape.any():
        final = distance_transform_edt(~shape) <= s.border_width * m
    return shape, paint, final


def _soft_edge(mask: np.ndarray) -> np.ndarray:
    """Antialias a binary mask: alpha from each pixel's sub-pixel distance to the 0.5 contour
    of a lightly blurred copy, which runs along the mask's edge without its pixel steps."""
    field = gaussian_filter(mask.astype(np.float32), AA_SIGMA)
    gy, gx = np.gradient(field)
    slope = np.maximum(np.hypot(gx, gy), 1e-3)
    return np.clip(0.5 + (field - 0.5) / slope, 0.0, 1.0).astype(np.float32)


def outline_one(rgb: torch.Tensor, alpha: torch.Tensor, s: OutlineSettings) -> torch.Tensor:
    """One image: `rgb` [H, W, 3], `alpha` [H, W] (1 = opaque). Returns RGBA [H', W', 4].

    Output alpha is strictly 0 or 1, unless antialias is on: then the same shape gets a soft
    edge about 1px wide.
    """
    u = max(1, int(s.upscale))
    h, w = alpha.shape
    rgb = rgb.float().cpu()
    alpha = alpha.float().cpu()

    # Before the upscale, so resampling can't blend fringe colour back in.
    rgb = _fix_colors(rgb, alpha.numpy(), s)
    color = _resize(rgb, (h * u, w * u)).numpy().copy()  # never write into the input tensor

    # Pad so the border is never clipped by the canvas edge.
    pad = int(np.ceil(max(0.0, s.border_width)))
    if pad:
        color = np.pad(color, ((pad * u, pad * u), (pad * u, pad * u), (0, 0)), mode="edge")

    masks = _shape_masks(alpha, s, u, pad)
    soften = _soft_edge if s.antialias else (lambda x: x.astype(np.float32))
    shape, paint, final = (soften(x)[..., None] for x in masks)

    if s.fill_color is not None:
        color = color * (1.0 - paint) + np.asarray(s.fill_color, np.float32) * paint
    if s.border_width > 0:
        # Border share of each pixel's covered area; 1 outside, to keep resampling clean.
        share = np.where(final > 0, (final - shape).clip(0.0) / np.maximum(final, 1e-8), 1.0)
        color = color * (1.0 - share) + np.asarray(s.border_color, np.float32) * share
    return torch.from_numpy(np.concatenate([color, final], axis=-1).astype(np.float32))


def outline_batch(
    image: torch.Tensor, mask: torch.Tensor | None, s: OutlineSettings
) -> torch.Tensor:
    """IMAGE [B, H, W, 3|4] plus optional core-style MASK (1 = transparent). Returns RGBA."""
    if mask is not None:
        if mask.dim() == 2:
            mask = mask[None]
        alphas = 1.0 - mask.float()
        if alphas.shape[1:] != image.shape[1:3]:
            alphas = F.interpolate(alphas[:, None], size=image.shape[1:3], mode="bilinear")[:, 0]
    elif image.shape[-1] == 4:
        alphas = image[..., 3]
    else:
        raise ValueError(
            "Sticker Outline needs transparency: pass an RGBA image, or wire the mask from "
            "Load Image into `mask`."
        )
    out = []
    for i in range(image.shape[0]):
        a = alphas[min(i, alphas.shape[0] - 1)]
        out.append(outline_one(image[i, ..., :3], a, s))
    return torch.stack(out)
