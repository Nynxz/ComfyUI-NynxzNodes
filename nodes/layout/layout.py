"""Layout — draw boxes on a canvas for a model that reads a layout from its prompt.

One editor, a mode per model (see _prompt.py for the formats): Ideogram 4's caption, FLUX 3 Image
Klein's caption and element table, MOSS-Image's layout JSON. The node has one output, `layout`;
Layout Extras breaks it out into the prompt, core's elements, the canvas size and an overlay.
"""

from __future__ import annotations

from comfy_api.latest import io, ui

from .. import _categories
from .._base import NynxzNode
from .._zenkit.io_types import widget_type

#: Bound to frontend/widgets/LayoutScene.vue -> io_type NYNXZ_LAYOUT_SCENE.
LayoutSceneType = widget_type("LayoutScene", dict, doc="A layout: mode, canvas, prompt, boxes.")

#: Layout -> Layout Extras: the layout as the editor stores it (see _prompt.py).
Layout = io.Custom("NYNXZ_LAYOUT")

#: Layout Inputs -> Layout: `{"image"}`. The agent's CLIP stays on Layout Inputs, unloaded.
LayoutInputs = io.Custom("NYNXZ_LAYOUT_INPUTS")


class NynxzLayout(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout",
            display_name="Layout",
            description=(
                "Lay a picture out as boxes on a canvas and get the layout prompt your model was "
                "trained on: Ideogram 4, FLUX 3 Image Klein or MOSS-Image. Switch modes any time; "
                "the boxes stay."
            ),
            is_experimental=True,
            inputs=[
                LayoutSceneType.Input("layout", default={}),
                LayoutInputs.Input(
                    "inputs",
                    optional=True,
                    tooltip="Layout Inputs: a photo to trace and remix, and the agent's model.",
                ),
            ],
            outputs=[Layout.Output(display_name="layout", tooltip="Into Layout Extras.")],
            hidden=[io.Hidden.unique_id],
            # An output node, so a run sends Layout Inputs' photo back to the editor.
            is_output_node=True,
        )

    @classmethod
    def execute(cls, layout=None, inputs=None) -> io.NodeOutput:
        value = layout if isinstance(layout, dict) else {}
        image = inputs.get("image") if isinstance(inputs, dict) else None
        if image is None:
            return io.NodeOutput(value)
        return io.NodeOutput(
            value,
            ui={
                "layout_node": [str(cls.hidden.unique_id)],
                "layout_source": ui.ImageSaveHelper.save_images(
                    image[:1], "nynxz.layout.source", io.FolderType.temp, cls, compress_level=1
                ),
                "layout_size": [int(image.shape[2]), int(image.shape[1])],
            },
        )
