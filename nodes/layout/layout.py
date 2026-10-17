"""Layout — draw boxes on a canvas for a model that reads a layout from its prompt.

One editor, a mode per model (see _prompt.py for the formats): Ideogram 4's caption, FLUX 3 Image
Klein's caption and element table, MOSS-Image's layout JSON. The node has one output, `layout`;
Layout Extras breaks it out into the prompt, core's elements, the canvas size and an overlay.
"""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .._zenkit.io_types import widget_type

#: Bound to frontend/widgets/LayoutScene.vue -> io_type NYNXZ_LAYOUT_SCENE.
LayoutSceneType = widget_type("LayoutScene", dict, doc="A layout: mode, canvas, prompt, boxes.")

#: Layout -> Layout Extras: the layout as the editor stores it (see _prompt.py).
Layout = io.Custom("NYNXZ_LAYOUT")


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
            inputs=[LayoutSceneType.Input("layout", default={})],
            outputs=[Layout.Output(display_name="layout", tooltip="Into Layout Extras.")],
        )

    @classmethod
    def execute(cls, layout=None) -> io.NodeOutput:
        return io.NodeOutput(layout if isinstance(layout, dict) else {})
