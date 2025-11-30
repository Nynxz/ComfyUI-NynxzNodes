from comfy_api.latest import io
from comfy_api_nodes.util.conversions import tensor_to_pil
from .CompareVideo import CompareVideo

from ..enums import UpscaleToEnum
from .utils import hex_to_rgb, enforce_target, resize_fit


class CreateFadeCompareVideoNode(io.ComfyNode):
    UPSCALE_METHODS = ["nearest-exact",
                       "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.Create.CompareVideo.Fade",
            display_name="Create Fade Compare Video",
            category="Nynxz",
            inputs=[
                *CompareVideo.node_inputs()
            ],
            outputs=[
                io.Image.Output(id="frames"),
            ]
        )

    @classmethod
    def execute(cls,
                start_image,
                end_image,
                upscale_method,
                scale_mode,
                upscale_to,
                duration,
                fps,
                options=None
                ):
        import numpy as np
        import torch

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
        img1 = enforce_target(resize_fit(
            img1, target_size, resample_method=resample, mode=scale_mode), target_size, hex_to_rgb(options.background_color if options else "#000000"))
        img2 = enforce_target(resize_fit(
            img2, target_size, resample_method=resample, mode=scale_mode), target_size, hex_to_rgb(options.background_color if options else "#000000"))

        frames = []

        a = np.array(img1).astype(np.float32)
        b = np.array(img2).astype(np.float32)
        frames = CompareVideo.create_fade_frames_from_arrays(
            a, b, duration, fps)

        stacked = torch.stack(frames, dim=0)

        return io.NodeOutput(
            stacked,
        )
