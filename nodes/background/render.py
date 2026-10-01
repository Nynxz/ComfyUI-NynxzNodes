"""Render a background spec at an arbitrary size — pure numpy + PIL, no dependencies.

A ``NYNXZ_BACKGROUND`` spec is resolution-independent: the Background Maker builds it
(colours, pattern params, and pre-tinted icon stamps), and the consumer (a grid node)
calls ``render_spec(w, h, spec)`` with its own canvas size. Modes: solid / gradient /
pattern (procedural, tinted) / icons (a seeded grid of icon stamps over the base) /
transparent (a fully see-through ground). The renderer always returns RGBA, so a
transparent ground survives all the way through the grid to save.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageOps

from .config import clampf, clampi, hex_to_rgb
from .patterns import pattern_mask


def _base(w: int, h: int, spec: dict) -> np.ndarray:
    """Solid or gradient base as an (h, w, 3) float array."""
    c1 = np.array(hex_to_rgb(spec.get("bg_color", "#000000")), dtype=np.float32)
    if spec.get("bg_mode") == "gradient":
        c2 = np.array(hex_to_rgb(spec.get("bg_color2", "#ffffff")), dtype=np.float32)
        ang = np.deg2rad(float(spec.get("bg_angle", 45)))
        yy, xx = np.mgrid[0:h, 0:w]
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        lo, hi = float(proj.min()), float(proj.max())
        t = (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)
        return c1 * (1.0 - t.astype(np.float32)[..., None]) + c2 * t.astype(np.float32)[..., None]
    arr = np.empty((h, w, 3), dtype=np.float32)
    arr[:] = c1
    return arr


def _pattern_overlay(base: np.ndarray, w: int, h: int, spec: dict) -> np.ndarray:
    tint = np.array(hex_to_rgb(spec.get("pattern_color", "#ffffff")), dtype=np.float32)
    mask = pattern_mask(spec.get("pattern", "dots"), w, h, spec.get("pattern_scale", 26))
    a = (mask * clampf(spec.get("pattern_opacity", 0.10), 0.0, 1.0))[..., None]
    return base * (1.0 - a) + tint * a


def _icon_overlay(canvas: Image.Image, w: int, h: int, spec: dict) -> Image.Image:
    # Stamps are built HERE (at render time) from the spec's raw icons — nothing is rendered
    # in the Background Maker; it only carries data.
    stamps = prepare_stamps(
        spec.get("icons") or [],
        spec.get("masks"),
        bool(spec.get("tint", True)),
        spec.get("tint_color", "#ffffff"),
        bool(spec.get("invert_alpha", False)),
    )
    if not stamps:
        return canvas
    cols = clampi(spec.get("columns", 12), 1, 256, 12)
    scale = clampf(spec.get("scale", 0.7), 0.05, 1.0, 0.7)
    rot_j = clampf(spec.get("rotate_jitter", 0), 0.0, 180.0, 0.0)
    sc_j = clampf(spec.get("scale_jitter", 0), 0.0, 1.0, 0.0)
    op = clampf(spec.get("opacity", 0.16), 0.0, 1.0, 0.16)
    rng = np.random.RandomState(int(spec.get("seed", 0)) & 0x7FFFFFFF)
    n = len(stamps)

    # A SQUARE grid sized to the output: `cols` square cells fill the width exactly
    # (cell = w/cols), whole rows fill the height, and the block is centred vertically. Each
    # icon is fitted (aspect-preserved) inside its square cell — and the size budget already
    # accounts for rotation — so NO icon is ever cut off, at any output size.
    cell = w / cols
    rows = max(1, int(h // cell))
    y_off = (h - rows * cell) / 2.0
    rad = math.radians(rot_j)
    safe = cell / (abs(math.cos(rad)) + abs(math.sin(rad))) if rot_j > 0 else cell
    budget = min(cell * scale, safe)
    for r in range(rows):
        for c in range(cols):
            src = stamps[0 if n == 1 else rng.randint(n)]
            sz = max(2.0, budget * (1.0 - sc_j * rng.rand()))  # jitter only shrinks → never clips
            iw, ih = src.size
            nw, nh = (sz, sz * ih / iw) if iw >= ih else (sz * iw / ih, sz)
            stamp = src.resize((max(2, int(nw)), max(2, int(nh))), Image.LANCZOS)
            if rot_j > 0:
                stamp = stamp.rotate(
                    rot_j * (rng.rand() * 2 - 1), expand=True, resample=Image.BICUBIC
                )
            if op < 1.0:
                stamp.putalpha(stamp.getchannel("A").point(lambda v: int(v * op)))
            cx = (c + 0.5) * cell
            cy = y_off + (r + 0.5) * cell
            canvas.paste(stamp, (int(cx - stamp.width / 2), int(cy - stamp.height / 2)), stamp)
    return canvas


def render_spec(w: int, h: int, spec: dict | None) -> Image.Image:
    """Render a background spec to an RGBA PIL image of exactly wxh.

    Always RGBA: opaque modes come back with a full alpha channel, and "transparent"
    comes back fully see-through — so the grid can keep the transparency instead of
    baking in a colour, and a saved PNG/WebP carries a real alpha channel.
    """
    w = max(1, int(w))
    h = max(1, int(h))
    spec = spec if isinstance(spec, dict) else {"bg_mode": "solid", "bg_color": "#16161c"}
    mode = spec.get("bg_mode", "solid")
    if mode == "transparent":
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base = _base(w, h, spec)
    if mode == "pattern":
        base = _pattern_overlay(base, w, h, spec)
    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    if mode == "icons":
        canvas = _icon_overlay(canvas, w, h, spec)
    # Every non-transparent ground is fully opaque: icon stamps blend their alpha into the
    # canvas as they paste, so pin alpha back to 255 — otherwise the grid would mistake a
    # solid icon backdrop for a see-through one and emit a 4-channel image.
    canvas.putalpha(255)
    return canvas


def prepare_stamps(icons_pil, masks, tint: bool, tint_color, invert: bool) -> list[Image.Image]:
    """Pre-build each icon as an RGBA stamp (alpha applied, optionally tinted to one colour).

    Done once in the Background Maker so the (resolution-independent) spec carries ready
    stamps and the grid just places + scales them.
    """
    tcol = hex_to_rgb(tint_color, (255, 255, 255))
    stamps: list[Image.Image] = []
    for i, rgb in enumerate(icons_pil):
        mask = masks[i] if masks and i < len(masks) else None
        a = mask.convert("L").resize(rgb.size) if mask is not None else rgb.convert("L")
        if invert:
            a = ImageOps.invert(a)
        if tint:
            st = Image.new("RGBA", rgb.size, (tcol[0], tcol[1], tcol[2], 0))
            st.putalpha(a)
        else:
            st = rgb.convert("RGBA")
            st.putalpha(a)
        stamps.append(st)
    return stamps
