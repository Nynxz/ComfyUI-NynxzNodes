from enum import Enum
from comfy_api.latest import io, ui
from .base.CompareVideo import CompareVideo


class CreateWipeCompareVideo(io.ComfyNode):
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
                    options=[e.value for e in CompareVideo.WipeDirectionsEnum],
                    default=CompareVideo.WipeDirectionsEnum.LEFT_TO_RIGHT.value,
                ),
                io.Int.Input("fade_width", default=20,
                             min=0, max=100),
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
                force_upscale,
                duration,
                fps,
                wipe_direction,
                fade_width
                ):
        """
        Creates frames for a comparison wipe video where:
        - Left side shows start_image
        - Right side shows end_image
        - A vertical bar wipes left → right
        - Optional fade_width for smooth transition
        - Includes all your resizing methods for fairness
        """

        import numpy as np
        from PIL import Image
        import torch

        img2 = CompareVideo.to_pil(start_image)
        img1 = CompareVideo.to_pil(end_image)

        resample = CompareVideo.resample_by_name(upscale_method)

        do_resize = bool(force_upscale)
        if not do_resize:
            if img1.size[0] < img2.size[0] or img1.size[1] < img2.size[1]:
                do_resize = True

        if do_resize and resample is not None:
            img1 = img1.resize(img2.size, resample=resample)

        # Ensure same size
        width, height = img2.size
        img1 = img1.resize(
            (width, height), resample=resample) if resample else img1

        # Convert both to float32 numpy arrays
        arr1 = np.array(img1).astype(np.float32)
        arr2 = np.array(img2).astype(np.float32)

        frames = []

        frames = CompareVideo.create_frames_from_arrays(
            arr1, arr2, duration, fps, wipe_direction, fade_width
        )

        stacked = torch.stack(frames, dim=0)

        return io.NodeOutput(
            stacked,
        )
