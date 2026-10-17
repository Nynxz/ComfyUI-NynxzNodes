"""Layout Inputs — what a Layout node draws on: a photo to trace and remix, and the model its
agent asks for a layout.

`clip` is a vision-language text encoder that can generate (Qwen3-VL through CLIPLoader). Running
the graph never loads it: it's lazy and nothing here asks for it. The editor's agent takes the
nodes feeding it and queues its own small Generate Text job when you ask it for a layout.
`image`, when wired, goes back to the Layout editor after a run as its trace image, which the agent
can read.
"""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .layout import LayoutInputs


class NynxzLayoutInputs(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout.Inputs",
            display_name="Layout Inputs",
            description=(
                "Bring a Layout node a photo to trace and remix, and a vision-language model "
                "(Qwen3-VL) for its agent to draw boxes with. The model only loads when you ask "
                "the agent."
            ),
            is_experimental=True,
            inputs=[
                io.Clip.Input(
                    "clip",
                    optional=True,
                    lazy=True,
                    tooltip="A text encoder that can generate and see, e.g. Qwen3-VL-8B from "
                    "CLIPLoader. Used by the Layout editor's agent only.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Shown in the Layout editor after a run, to trace or remix.",
                ),
            ],
            outputs=[LayoutInputs.Output(display_name="inputs", tooltip="Into Layout.")],
        )

    @classmethod
    def check_lazy_status(cls, clip=None, image=None) -> list[str]:
        return ["image"] if image is None else []

    @classmethod
    def execute(cls, clip=None, image=None) -> io.NodeOutput:
        return io.NodeOutput({"image": image})
