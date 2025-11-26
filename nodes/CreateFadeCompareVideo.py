from comfy_api.latest import io, ui
from .base.CompareVideo import CompareVideo


class CreateFadeCompareVideo(io.ComfyNode):
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
    def execute(cls, start_image, end_image, upscale_method, force_upscale, duration, fps):
        import numpy as np
        from PIL import Image
        import torch

        # use CompareVideo helpers
        img1 = CompareVideo.to_pil(start_image)
        img2 = CompareVideo.to_pil(end_image)

        resample = CompareVideo.resample_by_name(upscale_method)

        do_resize = bool(force_upscale) or \
            img1.width < img2.width or img1.height < img2.height

        if do_resize and resample:
            img1 = img1.resize(img2.size, resample=resample)

        frames = []

        a = np.array(img1).astype(np.float32)
        b = np.array(img2).astype(np.float32)
        frames = CompareVideo.create_fade_frames_from_arrays(
            a, b, duration, fps)

        stacked = torch.stack(frames, dim=0)

        return io.NodeOutput(
            stacked,
        )
