import numpy as np
import torch

from comfy_api.latest import io
from comfy_api_nodes.util.conversions import tensor_to_pil
from ..enums import UpscaleToEnum, WipeDirectionsEnum
from .CompareVideo import CompareVideo
from .utils import hex_to_rgb, enforce_target, resize_fit


class CreateWipeCompareVideoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.Create.CompareVideo.Wipe",
            display_name="Create Wipe Compare Video",
            category="Nynxz",
            inputs=[
                *CompareVideo.node_inputs(),
                io.Combo.Input(
                    "wipe_direction",
                    options=[e.value for e in WipeDirectionsEnum],
                    default=WipeDirectionsEnum.LEFT_TO_RIGHT.value,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", id="frames"),
                io.Float.Output(display_name="fps", id="fps"),
            ],
        )

    @classmethod
    def execute(
        cls,
        start_image,
        end_image,
        upscale_to,
        duration,
        time_padding,
        wipe_direction,
        options=None,
    ):
        """
        Creates frames for a comparison wipe video where:
        - Left side shows start_image
        - Right side shows end_image
        - A vertical bar wipes left → right
        - Optional fade_width for smooth transition
        - Includes all your resizing methods for fairness
        """

        # Extract values from options with fallbacks
        upscale_method = CompareVideo.get_option_value(
            options, "upscale_method", "lanczos"
        )
        scale_mode = CompareVideo.get_option_value(options, "scale_mode", "contain")
        fps = CompareVideo.get_option_value(options, "fps", 8.0)
        fade_width = CompareVideo.get_option_value(options, "fade_width", 20)
        wipe_angle = CompareVideo.get_option_value(options, "wipe_angle", None)

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

        # Convert both to float32 numpy arrays
        arr1 = np.array(img1).astype(np.float32)
        arr2 = np.array(img2).astype(np.float32)

        frames = []

        frames = CompareVideo.create_frames_from_arrays(
            arr1,
            arr2,
            duration,
            fps,
            wipe_direction,
            wipe_angle=wipe_angle,
            fade_width=fade_width,
            time_padding=time_padding,
        )

        stacked = torch.stack(frames, dim=0)

        return io.NodeOutput(stacked, fps)
