from comfy_api.latest import io

from comfy_api.latest._io import ComfyTypeIO, WidgetInput, comfytype

from custom_nodes.nynxzs_custom_nodes.nodes.preview_maker.PreviewMaker import PreviewMaker
from .utils import hex_to_rgb


@comfytype(io_type="NYNXZ_COMPAREVIDEO_OPTS")
class CompareVideoOptions(ComfyTypeIO):
    Type = object

    class Input(WidgetInput):
        def __init__(self, id: str, display_name: str = None, optional=False, tooltip: str = None, lazy: bool = None,
                     background_color="#00FFFF", default: str = None, dynamic_prompts: bool = None,
                     socketless: bool = None, force_input: bool = None):
            super().__init__(id, display_name, optional, tooltip,
                             lazy, default, socketless, None, force_input)
            self.background_color = background_color


class CompareVideoOptionsObject:
    def __init__(self, background_color=hex_to_rgb("#000000")):
        self.background_color = background_color


class CompareVideoOptionsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.CompareVideo.Options",
            display_name="Compare Video Options",
            category="Nynxz",
            inputs=[
                io.String.Input(
                    "background_color", tooltip="Background color for the preview", default="#000000"),
            ],
            outputs=[
                CompareVideoOptions.Output(
                    "options", tooltip="Compare Video Options Object", display_name="options"),
            ]
        )

    @classmethod
    def execute(cls, background_color):
        options = CompareVideoOptionsObject(
            background_color=background_color,
        )

        return io.NodeOutput(options)
