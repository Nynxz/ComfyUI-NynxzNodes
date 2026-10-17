"""Nynxz Preview Image — preview images on the node with our custom widget + lightbox.

Like ComfyUI's Preview Image, but it shows the clean Nynxz preview (batch pager + a
fullscreen lightbox) instead of the default bottom strip. Accepts a batch OR a list, so a
dataset/folder loader feeds it directly and you can page/lightbox the whole set.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from .. import _categories
from .._base import NynxzNode
from ..save.io_types import NynxzSavePreviewType


def _flatten(images) -> list[torch.Tensor]:
    """Split batches and lists into a flat list of single-frame tensors."""
    out: list[torch.Tensor] = []
    for item in images if isinstance(images, list) else [images]:
        if item is None:
            continue
        nd = getattr(item, "ndim", 0)
        if nd == 4:
            out.extend(item[i : i + 1] for i in range(item.shape[0]))
        elif nd == 3:
            out.append(item.unsqueeze(0))
    return out


class NynxzPreviewImage(NynxzNode):
    """Preview a batch/list of images on the node (custom widget, no default preview)."""

    CATEGORY = _categories.OUTPUT
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Preview.Image",
            display_name="Preview Image",
            description=(
                "Preview images on the node with a clean custom preview (batch pager + "
                "fullscreen lightbox). Takes a batch or a list."
            ),
            inputs=[
                io.Image.Input("images", tooltip="Images to preview (a batch or a list)."),
                NynxzSavePreviewType.Input("preview", optional=True),
            ],
            outputs=[],
            is_output_node=True,
            is_input_list=True,
        )

    @classmethod
    def execute(cls, images, preview=None) -> io.NodeOutput:
        del preview
        # Save each frame to temp (frames may differ in size, so save individually) and feed
        # the records to the on-node preview via the custom 'preview' key.
        records = []
        for frame in _flatten(images):
            records += ui.ImageSaveHelper.save_images(
                frame,
                "nynxz.preview",
                io.FolderType.temp,
                cls,
                compress_level=2,
            )
        return io.NodeOutput(ui={"preview": records})
