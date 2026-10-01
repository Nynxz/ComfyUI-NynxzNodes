"""Shared native input groups + chain-merge for the Nynxz grid-style nodes.

One source of truth for the style fields, drawn on by both the all-in-one Grid Style node
and the chainable aspect nodes (Layout / Border / Shadow), so a default or range can never
drift between them. Each group is a small list of ``io.*.Input`` specs; ``patch_style``
overlays a node's own keys onto the upstream style so styles accumulate down a chain
(``Layout → Border → Shadow → Grid``). Colours are plain STRING inputs drawn with the
ColorField picker, so any colour string can still be wired in.

Underscore-prefixed so the node scanner treats it as a helper and never registers it.
"""

from __future__ import annotations

import json
from typing import Any

from comfy_api.latest import io

from .._lib.io_types import color_input

# ── Layout / geometry ────────────────────────────────────────────────────────
LAYOUT_KEYS = [
    "columns",
    "size_mode",
    "cell_width",
    "cell_height",
    "canvas_width",
    "canvas_height",
    "gap_x",
    "gap_y",
    "padding",
    "fit",
]


def layout_inputs() -> list:
    return [
        io.Int.Input("columns", default=3, min=1, max=64, tooltip="Cells per row."),
        io.Combo.Input(
            "size_mode",
            options=["cells", "canvas"],
            default="cells",
            tooltip="cells = fixed cell size, canvas derived · canvas = fixed canvas, cells derived.",
        ),
        io.Int.Input(
            "cell_width",
            default=320,
            min=1,
            max=16384,
            tooltip="Cell width, px (used when size_mode = cells).",
        ),
        io.Int.Input(
            "cell_height",
            default=180,
            min=1,
            max=16384,
            tooltip="Cell height, px (used when size_mode = cells).",
        ),
        io.Int.Input(
            "canvas_width",
            default=1280,
            min=16,
            max=16384,
            tooltip="Canvas width, px (used when size_mode = canvas).",
        ),
        io.Int.Input(
            "canvas_height",
            default=720,
            min=16,
            max=16384,
            tooltip="Canvas height, px (used when size_mode = canvas).",
        ),
        io.Int.Input(
            "gap_x", default=12, min=0, max=8192, tooltip="Horizontal gap between cells, px."
        ),
        io.Int.Input(
            "gap_y", default=12, min=0, max=8192, tooltip="Vertical gap between cells, px."
        ),
        io.Int.Input(
            "padding", default=24, min=0, max=8192, tooltip="Outer padding around the grid, px."
        ),
        io.Combo.Input(
            "fit",
            options=["cover", "contain"],
            default="cover",
            tooltip="cover = crop to fill the cell · contain = letterbox inside it.",
        ),
    ]


# ── Border ───────────────────────────────────────────────────────────────────
BORDER_KEYS = [
    "border_width",
    "border_mode",
    "border_color",
    "border_color2",
    "border_angle",
    "border_radius",
]


def border_inputs() -> list:
    return [
        io.Int.Input(
            "border_width", default=0, min=0, max=512, tooltip="Ring thickness, px (0 = no border)."
        ),
        io.Combo.Input(
            "border_mode",
            options=["solid", "gradient", "auto"],
            default="solid",
            tooltip="solid = one colour · gradient = 2-colour ring · auto = sampled from each image.",
        ),
        color_input(
            "border_color",
            default="#ffffff",
            tooltip="Solid colour / gradient start (#hex, rgb(…) or a colour name).",
        ),
        color_input(
            "border_color2",
            default="#6366f1",
            tooltip="Gradient end colour (gradient mode only).",
        ),
        io.Int.Input(
            "border_angle", default=45, min=0, max=360, tooltip="Gradient direction, degrees."
        ),
        io.Int.Input("border_radius", default=10, min=0, max=8192, tooltip="Corner radius, px."),
    ]


# ── Shadow / glow ────────────────────────────────────────────────────────────
SHADOW_KEYS = [
    "shadow",
    "shadow_mode",
    "shadow_blur",
    "shadow_offset_x",
    "shadow_offset_y",
    "shadow_color",
    "shadow_opacity",
    "shadow_match",
]


def shadow_inputs() -> list:
    return [
        io.Boolean.Input(
            "shadow",
            default=True,
            tooltip="Enable the drop shadow / glow (wire off to cancel an upstream one).",
        ),
        io.Combo.Input(
            "shadow_mode",
            options=["drop", "glow"],
            default="drop",
            tooltip="drop = offset shadow · glow = symmetric bloom (offset ignored).",
        ),
        io.Int.Input(
            "shadow_blur", default=18, min=0, max=256, tooltip="Blur radius / glow size, px."
        ),
        io.Int.Input(
            "shadow_offset_x",
            default=0,
            min=-8192,
            max=8192,
            tooltip="Shadow X offset, px (drop mode only).",
        ),
        io.Int.Input(
            "shadow_offset_y",
            default=10,
            min=-8192,
            max=8192,
            tooltip="Shadow Y offset, px (drop mode only).",
        ),
        color_input("shadow_color", default="#000000", tooltip="Shadow / glow colour."),
        io.Float.Input(
            "shadow_opacity",
            default=0.45,
            min=0.0,
            max=1.0,
            step=0.05,
            tooltip="Shadow / glow strength (0-1).",
        ),
        io.Boolean.Input(
            "shadow_match",
            default=False,
            tooltip="Tint the glow with each cell's auto-border colour (needs border_mode = auto).",
        ),
    ]


# ── Chain merge ──────────────────────────────────────────────────────────────
def merge_incoming(style: Any) -> dict:
    """Upstream NYNXZ_GRID_STYLE (a dict, a JSON string, or None) → a plain dict."""
    if isinstance(style, str):
        try:
            style = json.loads(style)
        except (TypeError, ValueError):
            style = None
    return dict(style) if isinstance(style, dict) else {}


def patch_style(incoming: Any, fields: dict, keys: list[str]) -> dict:
    """Overlay this node's own ``keys`` (from an execute kwargs bag) onto the upstream style.

    Only the aspect's own keys are touched, so a Border node leaves an upstream Layout's keys
    untouched and vice-versa; two nodes of the same aspect make the later one win.
    """
    out = merge_incoming(incoming)
    for k in keys:
        v = fields.get(k)
        if v is not None:
            out[k] = v
    return out
