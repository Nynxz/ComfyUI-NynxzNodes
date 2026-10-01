"""Nynxz Image Compare — draggable before/after reveal of two images.

An output node: it saves both inputs as temp previews and returns their locations
in the ui payload; the on-node Vue widget (fed by the frontend's onExecuted hook)
renders the draggable reveal / fade.
"""

from __future__ import annotations

from typing import Any

from comfy_api.latest import io, ui

from .._base import NynxzNode
from .io_types import ImageCompareType


class NynxzImageCompare(NynxzNode):
    CATEGORY = f"{NynxzNode.CATEGORY}/Image"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Image.Compare",  # -> nynxz.Image.Compare, the id it shipped under
            display_name="Image Compare",
            description="Compare two images with a draggable reveal or fade.",
            is_output_node=True,
            inputs=[
                io.Image.Input("image_a", optional=True, tooltip="First image (the 'before')."),
                io.Image.Input("image_b", optional=True, tooltip="Second image (the 'after')."),
                ImageCompareType.Input("compare_view", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        image_a: Any | None = None,
        image_b: Any | None = None,
        compare_view: Any | None = None,
    ) -> io.NodeOutput:
        # compare_view is set by the frontend; the reveal options live in the widget.
        del compare_view

        result: dict[str, list[Any]] = {"a_images": [], "b_images": []}
        if image_a is not None and len(image_a) > 0:
            result["a_images"] = ui.ImageSaveHelper.save_images(
                image_a,
                "nynxz.compare.a",
                io.FolderType.temp,
                cls,
                compress_level=1,
            )
        if image_b is not None and len(image_b) > 0:
            result["b_images"] = ui.ImageSaveHelper.save_images(
                image_b,
                "nynxz.compare.b",
                io.FolderType.temp,
                cls,
                compress_level=1,
            )
        return io.NodeOutput(ui=result)
