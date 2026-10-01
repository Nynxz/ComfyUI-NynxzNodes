"""IO types for the Nynxz Save nodes.

- ``NYNXZ_SAVE_INFO`` — a plain *socketed* wire: Save Info emits a config dict on it and
  Save Image takes it as an optional input.
- ``NYNXZ_SAVE_PREVIEW`` — a socketless on-node WIDGET (frontend/widgets/SavePreview.vue)
  showing what was just written. Its value is pushed in by the frontend's onExecuted hook,
  never by the user, so it is always optional.
- ``NYNXZ_SAVE_INFO_CONFIG`` — the socketless WIDGET the Save Info panel
  (frontend/widgets/SaveInfoConfig.vue) edits: one object holding the whole config.
"""

from __future__ import annotations

from comfy_api.latest import io

from .._lib.io_types import widget_type

# A plain socketed link type: Save Info -> Save Image carries a config dict.
NynxzSaveInfoType = io.Custom("NYNXZ_SAVE_INFO")

# Bound to frontend/widgets/SavePreview.vue -> io_type NYNXZ_SAVE_PREVIEW.
NynxzSavePreviewType = widget_type(
    "SavePreview",
    list,
    doc="On-node saved-image preview WIDGET value (socketless): a list of saved records.",
)

# Bound to frontend/widgets/SaveInfoConfig.vue -> io_type NYNXZ_SAVE_INFO_CONFIG.
NynxzSaveInfoConfigType = widget_type(
    "SaveInfoConfig",
    dict,
    doc="On-node Save Info WIDGET value (socketless): the whole config object.",
)
