"""Layout Inputs — what a Layout node draws on: a photo to trace, segment and edit, its VAE, the
SAM3 checkpoint, and the vision-language model its agent asks for a layout.

`clip` is a vision-language text encoder that can generate (Qwen3-VL through CLIPLoader). Running
the graph never loads it: it's lazy and nothing here asks for it. The editor's agent takes the
nodes feeding it and queues its own small Generate Text job when you ask it for a layout.
`image`, when wired, goes back to the Layout editor after a run as its trace image, which the agent
and SAM3 can read, and travels on the layout wire to Layout Apply.
"""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from ..fresco import _segment
from .layout import LayoutInputs


class NynxzLayoutInputs(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout.Inputs",
            display_name="Layout Inputs",
            description=(
                "Bring a Layout node a photo to trace, segment and edit, its VAE, a SAM3 "
                "checkpoint for the wand/capture tool, and a vision-language model (Qwen3-VL) for "
                "its agent. Neither the agent's CLIP nor SAM3 loads until asked."
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
                    tooltip="Shown in the Layout editor after a run, to trace, segment with SAM3, "
                    "remix or edit in Layout Apply.",
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Carried to Layout Apply, to encode the photo or the layout start.",
                ),
                io.Combo.Input(
                    "sam3",
                    options=_segment.checkpoints(),
                    default=_segment.NONE,
                    tooltip="The SAM3 checkpoint, for boxes that capture something by text or "
                    "click points. Loaded only when a box has something new to segment.",
                ),
            ],
            outputs=[LayoutInputs.Output(display_name="inputs", tooltip="Into Layout.")],
        )

    @classmethod
    def check_lazy_status(cls, clip=None, image=None, vae=None, sam3=_segment.NONE) -> list[str]:
        """Nothing to ask for: the agent CLIP stays unloaded, and `image`/`vae` aren't lazy."""
        return []

    @classmethod
    def execute(cls, clip=None, image=None, vae=None, sam3=_segment.NONE) -> io.NodeOutput:
        return io.NodeOutput(
            {
                "image": image,
                "vae": vae,
                "sam3": None if sam3 == _segment.NONE else sam3,
            }
        )
