import numpy as np
import torch
from comfy_api.latest import io
from comfy_api_nodes.util.conversions import tensor_to_pil

from ._base import CompareNode
from ._compare import CompareVideo
from ._enums import UpscaleToEnum
from ._utils import enforce_target, hex_to_rgb, resize_fit


class CreateFadeCompareVideoNode(CompareNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Compare.VideoFade",
            display_name="Create Fade Compare Video",
            inputs=[*CompareVideo.node_inputs()],
            outputs=[
                io.Image.Output(display_name="images", id="frames"),
                io.Float.Output(display_name="fps", id="fps"),
            ],
        )

    @classmethod
    def execute(cls, start_image, end_image, upscale_to, duration, time_padding, options=None):

        # Extract values from options with fallbacks
        upscale_method = CompareVideo.get_option_value(options, "upscale_method", "lanczos")
        scale_mode = CompareVideo.get_option_value(options, "scale_mode", "contain")
        fps = CompareVideo.get_option_value(options, "fps", 8.0)

        img1 = tensor_to_pil(start_image)
        img2 = tensor_to_pil(end_image)

        resample = CompareVideo.resample_by_name(upscale_method)

        # Compute areas
        area1 = img1.size[0] * img1.size[1]
        area2 = img2.size[0] * img2.size[1]
        # Decide target size
        if upscale_to == UpscaleToEnum.LARGER_IMAGE.value:
            base = img1 if area1 >= area2 else img2
        elif upscale_to == UpscaleToEnum.SMALLER_IMAGE.value:
            base = img1 if area1 <= area2 else img2
        else:
            raise ValueError(f"Unknown upscale_to value: {upscale_to}")

        # Make divisible by 2
        w, h = base.size
        if w % 2 != 0:
            w += 1
        if h % 2 != 0:
            h += 1

        target_size = (w, h)

        # Resize both images to the target
        img1 = enforce_target(
            resize_fit(img1, target_size, resample_method=resample, mode=scale_mode),
            target_size,
            hex_to_rgb(options.background_color if options else "#000000"),
        )
        img2 = enforce_target(
            resize_fit(img2, target_size, resample_method=resample, mode=scale_mode),
            target_size,
            hex_to_rgb(options.background_color if options else "#000000"),
        )

        frames = []

        a = np.array(img1).astype(np.float32)
        b = np.array(img2).astype(np.float32)
        frames = CompareVideo.create_fade_frames_from_arrays(
            a, b, duration, fps, time_padding=time_padding
        )

        stacked = torch.stack(frames, dim=0)

        return io.NodeOutput(stacked, fps)
