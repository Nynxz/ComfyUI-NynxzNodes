"""Procedural background patterns — pure numpy, no dependencies.

Each pattern is a vectorised function of pixel coordinates, so it renders at any size,
tiles seamlessly by construction (modulo arithmetic), and costs one numpy pass. Returns
a float coverage mask in [0, 1] (1 = full pattern colour) that backgrounds.py tints +
composites over the base colour. Add a pattern by adding one branch here and the matching
entry in config.PATTERNS (and an SVG preview in the panel).
"""

from __future__ import annotations

import numpy as np


def pattern_mask(name: str, w: int, h: int, scale: float) -> np.ndarray:
    """Coverage mask (h, w) float32 in [0, 1] for a named, tileable pattern."""
    s = float(max(4, int(scale)))
    t = max(1.0, s * 0.12)  # line thickness for the line-based patterns
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx.astype(np.float32)
    y = yy.astype(np.float32)

    if name == "dots":
        cx = (x % s) - s / 2.0
        cy = (y % s) - s / 2.0
        d = np.sqrt(cx * cx + cy * cy)
        r = s * 0.22
        return np.clip((r + 0.75 - d) / 1.5, 0.0, 1.0).astype(np.float32)  # ~1.5px soft edge

    if name == "grid":
        m = ((x % s) < t) | ((y % s) < t)
        return m.astype(np.float32)

    if name == "checker":
        half = max(1.0, s / 2.0)
        m = ((np.floor(x / half).astype(np.int64) + np.floor(y / half).astype(np.int64)) % 2) == 0
        return m.astype(np.float32)

    if name == "stripes":
        return ((x % s) < t).astype(np.float32)  # vertical lines

    if name == "diagonal":
        return (((x + y) % s) < t).astype(np.float32)

    if name == "crosshatch":
        m = (((x + y) % s) < t) | ((np.mod(x - y, s)) < t)
        return m.astype(np.float32)

    return np.zeros((h, w), dtype=np.float32)  # unknown → no pattern
