"""IO types for the Nynxz Background Maker.

- ``NYNXZ_BACKGROUND`` — a plain *socketed* wire carrying the resolution-independent spec
  from the Background Maker to a grid's background input.
- ``NYNXZ_BG_CONFIG`` — the socketless on-node WIDGET (frontend/widgets/BgConfig.vue) that
  edits the style dict.
"""

from __future__ import annotations

from comfy_api.latest import io

from .._zenkit.io_types import widget_type

# Background Maker -> grid: a resolution-independent background spec.
NynxzBackgroundType = io.Custom("NYNXZ_BACKGROUND")

# Bound to frontend/widgets/BgConfig.vue -> io_type NYNXZ_BG_CONFIG.
NynxzBgConfigType = widget_type(
    "BgConfig",
    dict,
    doc="On-node Background Maker WIDGET value (socketless): the style dict.",
)
