"""Layout — draw boxes on a canvas, get the layout prompt your model reads.

One editor, a mode per model (see _prompt.py for the formats): Ideogram 4's caption, FLUX 3 Image
Klein's caption and element table, MOSS-Image's layout JSON. Switching mode keeps every box.

`elements` is the same list core's Create Bounding Boxes outputs, so the layout also drives core's
own layout prompt nodes; `width` and `height` are the canvas, for the latent.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .._zenkit.io_types import widget_type
from . import _prompt

#: Bound to frontend/widgets/LayoutScene.vue -> io_type NYNXZ_LAYOUT_SCENE.
LayoutSceneType = widget_type("LayoutScene", dict, doc="A layout: mode, canvas, prompt, boxes.")


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
            outputs=[
                io.String.Output(display_name="prompt", tooltip="Into your text encoder."),
                io.Array.Output(
                    display_name="elements",
                    tooltip="The boxes as core's prompt elements, for its own layout prompt nodes.",
                ),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, layout=None) -> io.NodeOutput:
        parsed = _prompt.parse(layout)
        for warning in _prompt.warnings(parsed):
            logging.info("Nynxz Layout (%s): %s", parsed["mode"], warning)
        return io.NodeOutput(
            _prompt.COMPILERS[parsed["mode"]](parsed),
            _prompt.elements(parsed),
            parsed["width"],
            parsed["height"],
        )
