"""IO types for the Nynxz grid-preview nodes.

- ``NYNXZ_GRID_STYLE`` — a plain *socketed* wire. Grid Style outputs a style dict on it and
  the grid nodes take it as an optional input, so one Grid Style drives many grids.
- ``NYNXZ_GRID_STYLE_CONFIG`` — the socketless on-node WIDGET
  (frontend/widgets/GridStyleConfig.vue) that edits the style dict.
"""

from __future__ import annotations

from comfy_api.latest import io

from .._lib.io_types import widget_type

# Grid Style -> grid nodes: the shared style dict.
NynxzGridStyleType = io.Custom("NYNXZ_GRID_STYLE")

# Bound to frontend/widgets/GridStyleConfig.vue -> io_type NYNXZ_GRID_STYLE_CONFIG.
NynxzGridStyleConfigType = widget_type(
    "GridStyleConfig",
    dict,
    doc="On-node Grid Style WIDGET value (socketless): the whole style dict.",
)
