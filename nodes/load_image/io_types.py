"""IO type for the Nynxz Load Image node.

``NYNXZ_IMAGE_PICK`` renders as the on-node picker (frontend/widgets/ImagePick.vue): a
searchable thumbnail dropdown plus a browse dialog. Its value is normally the chosen image's
path relative to the ComfyUI input directory ("sub/photo.png"), or a compound
``{"image": ..., "mask": ...}`` JSON string when a sidecar mask has been drawn.
"""

from __future__ import annotations

from .._lib.io_types import widget_type

# Bound to frontend/widgets/ImagePick.vue -> io_type NYNXZ_IMAGE_PICK.
NynxzImagePickType = widget_type(
    "ImagePick",
    str,
    doc="On-node image-pick WIDGET value (socketless): the input-relative image path.",
)
