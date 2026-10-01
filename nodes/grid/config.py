"""Shared style contract for the Nynxz grid-preview nodes.

Layout + border + shadow only — backgrounds now live entirely in the Background Maker
(wired into the grid's ``background`` input), so there's no background duplicated here.
The Grid Style panel edits this dict; the compose engine reads it (defaulting any missing
key). Colour/clamp helpers are shared from the background package (grid depends on it).
"""

from __future__ import annotations

from typing import Any

from ..background.config import clampf, clampi, hex_to_rgb  # noqa: F401  (re-exported for compose)

DEFAULT_GRID_STYLE: dict[str, Any] = {
    # layout
    "columns": 3,
    "size_mode": "cells",  # "cells" (cell size fixed → canvas derived) | "canvas"
    "cell_width": 320,
    "cell_height": 180,
    "canvas_width": 1280,
    "canvas_height": 720,
    "gap_x": 12,
    "gap_y": 12,
    "padding": 24,
    "fit": "cover",  # "cover" (crop) | "contain" (letterbox)
    # per-cell border
    "border_width": 0,
    "border_mode": "solid",  # "solid" | "gradient" (2-colour ring) | "auto" (from image)
    "border_color": "#ffffff",  # solid colour / gradient start
    "border_color2": "#6366f1",  # gradient end
    "border_angle": 45,  # gradient direction (deg)
    "border_radius": 10,
    # per-cell drop shadow — or a symmetric glow (shadow_mode="glow")
    "shadow": True,
    "shadow_mode": "drop",  # "drop" (offset shadow) | "glow" (all-directions bloom)
    "shadow_blur": 18,
    "shadow_offset_x": 0,
    "shadow_offset_y": 10,  # ignored in glow mode (glow is symmetric)
    "shadow_color": "#000000",
    "shadow_opacity": 0.45,
    "shadow_match": False,  # glow/shadow borrows each cell's auto-border colour
}


def merge_style(style: Any | None) -> dict[str, Any]:
    """DEFAULT_GRID_STYLE overlaid with a (possibly partial) incoming style dict."""
    cfg = dict(DEFAULT_GRID_STYLE)
    if isinstance(style, dict):
        for key in DEFAULT_GRID_STYLE:
            if key in style and style[key] is not None:
                cfg[key] = style[key]
    return cfg
