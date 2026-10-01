"""Style contract for the Nynxz Background generator + shared colour/clamp helpers.

The Background Maker owns ALL background types (solid / gradient / procedural pattern /
icon grid) and emits a *resolution-independent* spec — there is no canvas size here; the
consumer (a grid node) renders it at its own size. The grid imports the helpers + renderer
from this module, so the dependency runs background ← grid (never the reverse).
"""

from __future__ import annotations

from typing import Any

from PIL import ImageColor

# Procedural patterns (rendered in patterns.py; mirrored as SVG in the panel).
PATTERNS = ["dots", "grid", "checker", "stripes", "diagonal", "crosshatch"]

DEFAULT_BG: dict[str, Any] = {
    "bg_mode": "solid",  # "solid" | "gradient" | "pattern" | "icons" | "transparent"
    "bg_color": "#16161c",  # base fill / pattern+icon base
    "bg_color2": "#262633",  # gradient end
    "bg_angle": 45,  # gradient angle (deg)
    # procedural pattern overlay
    "pattern": "dots",
    "pattern_color": "#ffffff",
    "pattern_scale": 26,
    "pattern_opacity": 0.10,
    # icon-grid overlay — a square grid sized to the output (icons never cut off)
    "columns": 12,  # how many icon columns across the width (cell = width/columns)
    "scale": 0.7,  # icon size as a fraction of the (square) cell
    "tint": True,  # recolour icons to one colour
    "tint_color": "#ffffff",
    "invert_alpha": False,
    "rotate_jitter": 0,
    "scale_jitter": 0.0,
    "opacity": 0.16,  # overlay strength over the base
}
# For a glowing border, pick the "transparent" ground here and give each cell a coloured,
# zero-offset drop shadow on the Grid Style node (shadow_color + big blur, offset 0) — a
# blurred halo around each image reads as a glow, and the see-through ground lets it bloom.


def merge_bg(cfg: Any | None) -> dict[str, Any]:
    """DEFAULT_BG overlaid with a (possibly partial) incoming dict."""
    out = dict(DEFAULT_BG)
    if isinstance(cfg, dict):
        for key in DEFAULT_BG:
            if key in cfg and cfg[key] is not None:
                out[key] = cfg[key]
    return out


def hex_to_rgb(value: str, default=(0, 0, 0)) -> tuple[int, int, int]:
    """Any CSS-style colour — '#rrggbb', '#rgb', 'rrggbb', 'rgb(…)', 'hsl(…)', a name — to
    (r, g, b). Falls back to `default` on junk."""
    s = str(value).strip()
    bare = s.lstrip("#")
    if len(bare) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in bare):
        s = "#" + bare
    try:
        return ImageColor.getrgb(s)[:3]
    except (ValueError, TypeError):
        return default


def clampi(v: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        return max(lo, min(hi, round(float(v))))
    except (ValueError, TypeError):
        return max(lo, min(hi, default))


def clampf(v: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (ValueError, TypeError):
        return max(lo, min(hi, default))
