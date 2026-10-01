"""Custom IO type for the on-node image-compare surface.

The value (`{beforeImages, afterImages}` URL lists) is written by the frontend from the node's
`ui` output on execute, never by the user — so the input is optional, and the widget does not
serialize: those are temp-preview URLs that go stale the moment ComfyUI restarts.

Built through `widget_type()` rather than written out, so the io-type string can only ever agree
with the component filename the frontend registers under. It resolves to the same
`NYNXZ_IMAGE_COMPARE` this node has always used.
"""

from __future__ import annotations

from .._lib.io_types import widget_type

#: Bound to `frontend/widgets/ImageCompare.vue`.
ImageCompareType = widget_type(
    "ImageCompare",
    dict,
    doc="On-node image-compare WIDGET value (socketless): {beforeImages, afterImages}.",
)
