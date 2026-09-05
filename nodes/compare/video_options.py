from typing import ClassVar

from comfy_api.latest import io
from comfy_api.latest._io import ComfyTypeIO, comfytype

from ._base import CompareNode
from ._enums import ResizeModeEnum
from ._utils import hex_to_rgb


@comfytype(io_type="NYNXZ_COMPAREVIDEO_OPTS")
class CompareVideoOptions(ComfyTypeIO):
    Type = object


# Evaluated once at import; a call in a default argument hides that (B008).
_DEFAULT_BACKGROUND = hex_to_rgb("#000000")


class CompareVideoOptionsObject:
    def __init__(
        self,
        background_color=_DEFAULT_BACKGROUND,
        scale_mode="contain",
        fps=8.0,
        upscale_method="lanczos",
        wipe_angle=None,
        fade_width=20,
    ):
        self.background_color = background_color
        self.scale_mode = scale_mode
        self.fps = fps
        self.upscale_method = upscale_method
        self.wipe_angle = wipe_angle
        self.fade_width = fade_width


class CompareVideoOptionsNode(CompareNode):
    UPSCALE_METHODS: ClassVar[list[str]] = [
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Compare.VideoOptions",
            display_name="Compare Video Options",
            inputs=[
                io.String.Input(
                    "background_color",
                    tooltip="Background color for the preview",
                    default="#000000",
                ),
                io.Combo.Input(
                    "scale_mode",
                    options=[e.value for e in ResizeModeEnum],
                    default="contain",
                    tooltip="Choose how images are scaled to fit the target size",
                    display_name="Scale Mode",
                ),
                io.Float.Input(
                    "fps",
                    default=8.0,
                    min=1,
                    max=60,
                    step=1,
                    tooltip="Frames per second of the compare video",
                    display_name="FPS",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=cls.UPSCALE_METHODS,
                    default="lanczos",
                    tooltip="Resampling method used when resizing images",
                    display_name="Upscale Method",
                ),
                io.Int.Input(
                    "wipe_angle",
                    default=-1,
                    min=-1,
                    max=360,
                    optional=True,
                    tooltip="Angle of the wipe in degrees (-1 for direction-based)",
                    display_name="Wipe Angle",
                ),
                io.Int.Input(
                    "fade_width",
                    default=20,
                    min=0,
                    max=100,
                    tooltip="Width of the fade transition in pixels",
                    display_name="Fade Width",
                ),
            ],
            outputs=[
                CompareVideoOptions.Output(
                    "options", tooltip="Compare Video Options Object", display_name="options"
                ),
            ],
        )

    @classmethod
    def execute(cls, background_color, scale_mode, fps, upscale_method, wipe_angle, fade_width):
        options = CompareVideoOptionsObject(
            background_color=background_color,
            scale_mode=scale_mode,
            fps=fps,
            upscale_method=upscale_method,
            wipe_angle=wipe_angle if wipe_angle >= 0 else None,
            fade_width=fade_width,
        )

        return io.NodeOutput(options)
