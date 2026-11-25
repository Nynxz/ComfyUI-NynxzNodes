"""Layout Overlay — draws a Layout's boxes and painted shapes over a render, and sends the render
back to the Layout editor as its trace image so you can iterate over the last result."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from .. import _categories
from .._base import NynxzNode
from .._sent_back import sent_back
from ..fresco import _scene
from ..fresco.overlay import draw
from . import _prompt
from .layout import Layout


class NynxzLayoutOverlay(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout.Overlay",
            display_name="Layout Overlay",
            description=(
                "Draws a Layout's boxes and painted shapes over a render to check placement, and "
                "sends the render back to the Layout editor as its trace image."
            ),
            is_experimental=True,
            inputs=[
                io.Image.Input("image", tooltip="The render — VAE Decode's output."),
                Layout.Input("layout", tooltip="The Layout node the render was made from."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="overlay", tooltip="The render with every box/shape outlined."
                )
            ],
        )

    @classmethod
    def execute(cls, image, layout) -> io.NodeOutput:
        parsed = _prompt.parse(layout)
        width, height = parsed["width"], parsed["height"]
        size = _scene.working_size(width, height)
        outlines = [
            {
                "label": box["name"] or box["id"],
                "color": box.get("color", ""),
                "mask": _scene.layer_mask(box, size, (height, width)),
            }
            for box in _prompt.shown(parsed)
        ]
        overlay = torch.stack([draw(frame, outlines) for frame in image], dim=0)
        shown = {
            "images": ui.ImageSaveHelper.save_images(
                overlay, "nynxz.layout.overlay", io.FolderType.temp, cls, compress_level=1
            )
        }
        runtime = layout.get("_runtime") if isinstance(layout, dict) else None
        if isinstance(runtime, dict) and runtime.get("image") is None and runtime.get("node"):
            shown |= sent_back("layout", str(runtime["node"]), image, cls, "nynxz.layout.render")
        return io.NodeOutput(overlay, ui=shown)
