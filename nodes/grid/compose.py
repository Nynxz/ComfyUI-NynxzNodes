"""Compose a batch of images into a screenshot-grid image.

Pure numpy + PIL. Given N images and a style dict it lays out a `columns`-wide grid,
sizes the cells (either fixed cell size → derived canvas, or fixed canvas → derived
cells), and renders each cell with cover/contain fit, rounded corners, an optional
border and an optional soft drop shadow, over the chosen background.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

from ..background.render import render_spec
from .config import clampf, clampi, hex_to_rgb, merge_style


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8))


def _pil_to_tensor(p: Image.Image, alpha: bool = False) -> torch.Tensor:
    """PIL → [H, W, C] float tensor. RGBA (4-channel) when `alpha`, else RGB."""
    return torch.from_numpy(
        np.array(p.convert("RGBA" if alpha else "RGB")).astype(np.float32) / 255.0
    )


def _paste_over(canvas: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
    """Alpha-composite an RGBA `sprite` onto an RGBA `canvas` at (x, y), in place.

    Proper 'over' compositing (unlike Image.paste with a mask, which straight-blends and
    fringes on a transparent ground). Only the overlapping bbox is touched, and negative
    or out-of-bounds offsets are clipped — so shadows can bleed past the canvas edge.
    """
    cw, ch = canvas.size
    sw, sh = sprite.size
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(cw, x + sw), min(ch, y + sh)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    piece = sprite.crop((dx0 - x, dy0 - y, dx1 - x, dy1 - y))
    region = canvas.crop((dx0, dy0, dx1, dy1))
    canvas.paste(Image.alpha_composite(region, piece), (dx0, dy0))


def _resize_cell(img: Image.Image, cw: int, ch: int, fit: str) -> Image.Image:
    """Resize `img` to a cwxch RGBA cell. cover = crop-to-fill; contain = letterbox."""
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    rgba = img.convert("RGBA")
    if fit == "contain":
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        r = rgba.resize((nw, nh), Image.LANCZOS)
        cell = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        cell.paste(r, ((cw - nw) // 2, (ch - nh) // 2), r)
        return cell
    # cover: scale up to fill, centre-crop (ceil so we never fall short of the cell)
    scale = max(cw / iw, ch / ih)
    nw, nh = max(cw, math.ceil(iw * scale)), max(ch, math.ceil(ih * scale))
    r = rgba.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - cw) // 2, (nh - ch) // 2
    return r.crop((left, top, left + cw, top + ch))


def _aa_rounded_mask(cw: int, ch: int, radius: int, inset: int = 0) -> Image.Image:
    """An anti-aliased rounded-rectangle alpha mask (optionally inset on all sides).

    Each pixel's alpha is its coverage, read straight off the signed distance to the rounded
    rect (negative inside): a pixel one half-pixel inside the edge is solid, one half-pixel
    outside is clear, and the boundary ramps linearly between. That is resolution-independent
    — the corners stay equally smooth on a 320px cell and a 4K one — where supersampling had
    to shrink its factor on large cells to bound memory, degrading to hard 1x edges.
    """
    bw_in = max(0, inset)
    iw, ih = cw - 2 * bw_in, ch - 2 * bw_in
    if iw <= 0 or ih <= 0:
        return Image.new("L", (cw, ch), 0)
    hx, hy = iw / 2.0, ih / 2.0
    rad = max(0.0, min(float(radius - bw_in), hx, hy))
    # Offset of each pixel centre from the nearest corner-arc centre, per axis. The distance
    # is separable up to the hypot, so the axes stay 1-D and broadcast — a full 2-D coordinate
    # grid would cost several more buffers the size of the cell. Everything past the hypot is
    # in-place for the same reason.
    qx = (np.abs(np.arange(cw, dtype=np.float32) + 0.5 - cw / 2.0) - (hx - rad))[None, :]
    qy = (np.abs(np.arange(ch, dtype=np.float32) + 0.5 - ch / 2.0) - (hy - rad))[:, None]
    sd = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))  # distance outside the arc centres
    inside = np.minimum(np.maximum(qx, qy), 0.0)  # ...and within them (0 outside)
    sd += inside
    del inside
    sd -= rad  # <0 inside the shape, >0 outside
    np.subtract(0.5, sd, out=sd)  # signed distance → pixel coverage
    np.clip(sd, 0.0, 1.0, out=sd)
    sd *= 255.0
    return Image.fromarray(np.rint(sd, out=sd).astype(np.uint8), "L")


def _linear_gradient(w: int, h: int, c1, c2, angle: float) -> Image.Image:
    """A wxh RGB image: a 2-colour linear gradient blended along `angle` (degrees)."""
    a = np.array(c1, dtype=np.float32)
    b = np.array(c2, dtype=np.float32)
    ang = math.radians(float(angle))
    yy, xx = np.mgrid[0:h, 0:w]
    proj = xx * math.cos(ang) + yy * math.sin(ang)
    lo, hi = float(proj.min()), float(proj.max())
    t = (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)
    t = t.astype(np.float32)[..., None]
    arr = a * (1.0 - t) + b * t
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _dominant_color(img: Image.Image, boost: float = 1.45) -> tuple[int, int, int]:
    """A vivid representative colour for `img` — a saturation-weighted average, boosted.

    Vivid pixels are weighted more than flat/grey ones (so a mostly-grey screenshot with
    a splash of colour borders in that splash), then pushed away from grey for punch.
    """
    small = np.asarray(
        img.convert("RGB").resize((32, 32), Image.LANCZOS), dtype=np.float32
    ).reshape(-1, 3)
    mx, mn = small.max(1), small.min(1)
    sat = (mx - mn) / (mx + 1e-6)  # 0 grey … 1 vivid
    wt = sat + 0.06  # keep a floor so an all-grey image still averages
    avg = (small * wt[:, None]).sum(0) / wt.sum()
    grey = float(avg.mean())
    out = np.clip(grey + (avg - grey) * boost, 0, 255)
    return int(out[0]), int(out[1]), int(out[2])


def _tint_ring(
    ring: Image.Image, cw: int, ch: int, mode: str, cfg: dict, auto_rgb: tuple[int, int, int] | None
) -> Image.Image:
    """Colour a border ring: solid fill, a 2-colour gradient, or a per-cell auto colour."""
    if mode == "gradient":
        c1 = hex_to_rgb(cfg["border_color"], (255, 255, 255))
        c2 = hex_to_rgb(cfg.get("border_color2", "#6366f1"), (99, 102, 241))
        layer = _linear_gradient(cw, ch, c1, c2, cfg.get("border_angle", 45)).convert("RGBA")
        layer.putalpha(ring)
        return layer
    rgb = (
        auto_rgb
        if (mode == "auto" and auto_rgb)
        else hex_to_rgb(cfg["border_color"], (255, 255, 255))
    )
    sprite = Image.new("RGBA", (cw, ch), (rgb[0], rgb[1], rgb[2], 0))
    sprite.putalpha(ring)
    return sprite


def _bloom_alpha(
    mask: Image.Image, pad_s: int, blur: int, opacity: float, mode: str
) -> Image.Image:
    """The (colour-less) alpha of a cell's shadow/glow, padded by `pad_s` on all sides.

    drop = one Gaussian (a soft shadow). glow = a bright, tight core lightened with a wide
    soft halo, so the edge reads as emitted light rather than a flat haze.
    """
    cw, ch = mask.size
    sm = Image.new("L", (cw + 2 * pad_s, ch + 2 * pad_s), 0)
    sm.paste(mask.point(lambda v: int(v * opacity)), (pad_s, pad_s))
    if mode == "glow":
        tight = sm.filter(ImageFilter.GaussianBlur(max(1.0, blur * 0.45)))
        wide = sm.filter(ImageFilter.GaussianBlur(max(1.0, blur * 1.25)))
        return ImageChops.lighter(tight, wide)
    return sm.filter(ImageFilter.GaussianBlur(blur))


def _colorize(alpha: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    """An RGBA sprite: a flat `rgb` fill carrying `alpha` (an L image) as its alpha."""
    sprite = Image.new("RGBA", alpha.size, (rgb[0], rgb[1], rgb[2], 0))
    sprite.putalpha(alpha)
    return sprite


# Plain backdrop used when no Background Maker is wired in.
_DEFAULT_BG = {"bg_mode": "solid", "bg_color": "#16161c"}


def to_preview_rgb(grid: torch.Tensor, cell: int = 16) -> torch.Tensor:
    """Flatten a (possibly RGBA) grid to a [1, H, W, 3] tensor for on-node previews.

    A transparent grid is composited over a checkerboard so the see-through areas read as
    transparency in the preview; an RGB grid passes through untouched. Keeps the preview
    helper on 3-channel input while the node still outputs the real RGBA image.
    """
    t = grid[0] if getattr(grid, "ndim", 0) == 4 else grid
    if t.shape[-1] < 4:
        return t.unsqueeze(0) if getattr(grid, "ndim", 0) != 4 else grid
    rgb = t[..., :3]
    a = t[..., 3:4]
    h, w = t.shape[0], t.shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((xx // cell) + (yy // cell)) % 2 == 0, 0.36, 0.5).astype(np.float32)
    board = torch.from_numpy(checker).unsqueeze(-1)
    out = rgb * a + board * (1.0 - a)
    return out.unsqueeze(0)


def compose_grid(images, style, background=None) -> torch.Tensor:
    """Return a single [1, H, W, 3] image tensor: the composed grid.

    `images` is an iterable of [H,W,3] (or [1,H,W,3]) tensors. `background`, if given, is a
    resolution-independent NYNXZ_BACKGROUND spec (from a Background Maker), rendered to the
    canvas size here; otherwise a plain solid default is used.
    """
    cfg = merge_style(style)
    pil_imgs = (
        [_tensor_to_pil(t.squeeze(0) if getattr(t, "ndim", 0) == 4 else t) for t in images]
        if images is not None
        else []
    )
    n = len(pil_imgs)

    cols = clampi(cfg["columns"], 1, 64, 3)
    rows = max(1, math.ceil(n / cols)) if n else 1
    gap_x = clampi(cfg["gap_x"], 0, 8192, 12)
    gap_y = clampi(cfg["gap_y"], 0, 8192, 12)
    pad = clampi(cfg["padding"], 0, 8192, 24)
    fit = cfg.get("fit") if cfg.get("fit") in ("cover", "contain") else "cover"

    if cfg.get("size_mode") == "canvas":
        out_w = clampi(cfg["canvas_width"], 16, 16384, 1280)
        out_h = clampi(cfg["canvas_height"], 16, 16384, 720)
        cw = max(1, (out_w - 2 * pad - (cols - 1) * gap_x) // cols)
        ch = max(1, (out_h - 2 * pad - (rows - 1) * gap_y) // rows)
    else:
        cw = clampi(cfg["cell_width"], 1, 16384, 320)
        ch = clampi(cfg["cell_height"], 1, 16384, 180)
        out_w = 2 * pad + cols * cw + (cols - 1) * gap_x
        out_h = 2 * pad + rows * ch + (rows - 1) * gap_y

    canvas = render_spec(out_w, out_h, background if isinstance(background, dict) else _DEFAULT_BG)
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    # Keep an alpha channel in the output only when the ground is actually see-through
    # (a transparent Background Maker) — otherwise stay RGB so ordinary consumers of the
    # IMAGE output are unaffected.
    transparent = canvas.getchannel("A").getextrema()[0] < 255
    if n == 0:
        return _pil_to_tensor(canvas, alpha=transparent).unsqueeze(0)

    radius = clampi(cfg["border_radius"], 0, 8192, 10)
    bw = clampi(cfg["border_width"], 0, 512, 0)
    border_mode = (
        cfg.get("border_mode")
        if cfg.get("border_mode") in ("solid", "gradient", "auto")
        else "solid"
    )
    shadow_mode = "glow" if cfg.get("shadow_mode") == "glow" else "drop"

    # The mask geometry is identical for every cell (built once): the anti-aliased outer
    # mask that clips the image to the rounded rect, and the border ring (outer - inner)
    # that sits ON TOP of the image's outer edge — so the image can never poke outside the
    # border, and both edges stay smooth.
    mask = _aa_rounded_mask(cw, ch, radius, inset=0)
    ring = None
    if bw > 0:
        inner = _aa_rounded_mask(cw, ch, radius, inset=bw)
        ring = ImageChops.subtract(mask, inner)
    # Solid/gradient borders are the same for every cell → tint the ring once. An "auto"
    # border is per-cell (sampled from the image), so it's built inside the loop.
    border_sprite = (
        _tint_ring(ring, cw, ch, border_mode, cfg, None)
        if (ring is not None and border_mode != "auto")
        else None
    )

    # Shadow/glow: the blurred ALPHA is geometry (built once); only the colour can vary
    # per cell. "glow" is symmetric so its offset is forced to 0. "match" tints the glow
    # with each cell's auto colour — but only when the border is actually auto.
    glow_alpha = shadow_sprite = None
    pad_s = ox = oy = 0
    shadow_match = bool(cfg.get("shadow_match")) and border_mode == "auto"
    if cfg.get("shadow"):
        blur = clampi(cfg["shadow_blur"], 0, 256, 18)
        sop = clampf(cfg["shadow_opacity"], 0.0, 1.0, 0.45)
        pad_s = blur * 2 + 2
        glow_alpha = _bloom_alpha(mask, pad_s, blur, sop, shadow_mode)
        if shadow_mode == "drop":
            ox = clampi(cfg.get("shadow_offset_x", 0), -8192, 8192, 0)
            oy = clampi(cfg.get("shadow_offset_y", 10), -8192, 8192, 10)
        if not shadow_match:
            shadow_sprite = _colorize(glow_alpha, hex_to_rgb(cfg["shadow_color"], (0, 0, 0)))

    for i, img in enumerate(pil_imgs):
        r, c = divmod(i, cols)
        x = pad + c * (cw + gap_x)
        y = pad + r * (ch + gap_y)

        # One vivid colour sampled from this cell, shared by an auto border and a matched glow.
        auto_rgb = _dominant_color(img) if (border_mode == "auto" or shadow_match) else None

        if glow_alpha is not None:
            sprite = shadow_sprite if shadow_sprite is not None else _colorize(glow_alpha, auto_rgb)
            _paste_over(canvas, sprite, x - pad_s + ox, y - pad_s + oy)

        cell = _resize_cell(img, cw, ch, fit)
        cell.putalpha(ImageChops.multiply(cell.getchannel("A"), mask))
        bsprite = (
            border_sprite
            if border_sprite is not None
            else (
                _tint_ring(ring, cw, ch, border_mode, cfg, auto_rgb) if ring is not None else None
            )
        )
        if bsprite is not None:
            cell = Image.alpha_composite(cell, bsprite)
        _paste_over(canvas, cell, x, y)

    return _pil_to_tensor(canvas, alpha=transparent).unsqueeze(0)
