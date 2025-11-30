from typing import Callable, Iterable, List, Optional

from comfy_api.latest import io
from .CompareVideoOptions import CompareVideoOptions
from ..enums import UpscaleToEnum, ResizeModeEnum, WipeDirectionsEnum


class CompareVideo:
    """Base helper class with reusable inputs and utilities for compare-type nodes.

    This class intentionally provides only helpers (static methods / class
    attributes) — it is not a ComfyUI node itself. Nodes can import and reuse
    these helpers to keep code consistent across multiple node implementations.
    """
    # Common upscale/resampling method options shown in the UI
    UPSCALE_METHODS = ["nearest-exact",
                       "bilinear", "area", "bicubic", "lanczos"]

    @staticmethod
    def node_inputs() -> List[io.Input]:
        """Return a list of io.Input objects reused by compare nodes.

        Accepts a fade_width_default so callers can customize the default.
        """
        return [
            io.Image.Input("start_image", tooltip="Starting image",
                           display_name="Start Image"),
            io.Image.Input("end_image", tooltip="Ending image",
                           display_name="End Image"),
            CompareVideoOptions.Input("options", tooltip="Compare Video Options",
                                      optional=True, display_name="Options"),
            io.Combo.Input(
                "upscale_to", options=[e.value for e in UpscaleToEnum], default="Larger Image", tooltip="Choose whether to upscale the start and end images to the larger or smaller of the two images", display_name="Upscale To"),

            io.Combo.Input(
                "scale_mode", options=[e.value for e in ResizeModeEnum], default="contain", tooltip="Choose whether to upscale the start and end images to the larger or smaller of the two images", display_name="Scale Mode"),
            io.Float.Input("duration", default=1.0,
                           min=0.1, max=10.0, step=0.1, tooltip="Duration of the compare video in seconds", display_name="Duration (s)"),
            io.Float.Input("fps", default=8.0, min=1, max=60, step=1,
                           tooltip="Frames per second of the compare video", display_name="FPS"),

            io.Combo.Input(
                "upscale_method", options=CompareVideo.UPSCALE_METHODS, default="lanczos", tooltip="Resampling method used when resizing images", display_name="Upscale Method"),

        ]

    @staticmethod
    def resample_by_name(method_name: str):
        """Return a Pillow resampling constant for a friendly name."""
        from PIL import Image

        method = (method_name or "lanczos").lower()
        resample_map = {
            "nearest": Image.Resampling.NEAREST,
            "box": Image.Resampling.BOX if hasattr(Image, "BOX") else Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "hamming": Image.Resampling.HAMMING if hasattr(Image, "HAMMING") else Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.BICUBIC,
            "none": None,
        }
        return resample_map.get(method, Image.Resampling.LANCZOS)

    # operations to transform the image so the node can apply a single
    # left-right slicing algorithm for all directions
    @staticmethod
    def _op_flip_lr(x):
        import numpy as np

        return np.fliplr(x)

    @staticmethod
    def _op_transpose_hw(x):
        import numpy as np

        return np.transpose(x, (1, 0, 2))

    @staticmethod
    def ops_for_direction(direction_value: str) -> List[Callable]:
        """Return the list of ops for a wipe direction string."""
        return {
            WipeDirectionsEnum.LEFT_TO_RIGHT.value: [],
            WipeDirectionsEnum.RIGHT_TO_LEFT.value: [CompareVideo._op_flip_lr],
            WipeDirectionsEnum.TOP_TO_BOTTOM.value: [CompareVideo._op_transpose_hw],
            WipeDirectionsEnum.BOTTOM_TO_TOP.value: [CompareVideo._op_transpose_hw, CompareVideo._op_flip_lr],
        }[direction_value]

    @staticmethod
    def compose_ops(arr, ops: Iterable[Callable]):
        for f in ops:
            arr = f(arr)
        return arr

    @staticmethod
    def inverse_ops(arr, ops: Iterable[Callable]):
        # all ops used here are self-inverses; reverse order to undo
        for f in reversed(list(ops)):
            arr = f(arr)
        return arr

    @staticmethod
    def create_frames_from_arrays(arr_start, arr_end, duration: float, fps: float, wipe_direction: str, fade_width: Optional[int] = 20, fade_fraction: float = 0.5):
        """Generate torch tensors frames for the wipe animation from numpy arrays.

        Inputs are HWC numpy arrays (float or uint8). The function returns a
        list of torch tensors HWC float32 in 0..1 range, stacked per-frame.
        """
        import numpy as np
        import torch

        # normalize to float32 in 0..255 range (some inputs are already float)
        a = arr_start.astype(np.float32)
        b = arr_end.astype(np.float32)

        num_frames = max(2, int(duration * fps))

        # pick ops based on requested direction and pretransform
        ops = CompareVideo.ops_for_direction(wipe_direction)
        a_t = CompareVideo.compose_ops(a, ops)
        b_t = CompareVideo.compose_ops(b, ops)

        trans_width = a_t.shape[1]
        frames = []

        # ensure sensible fade_width
        if fade_width is None:
            fade_width = max(1, int(trans_width * fade_fraction))

        # single reusable buffer
        out_t = None
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            bar_x = int(progress * trans_width)

            if out_t is None:
                out_t = np.empty_like(a_t)

            # left portion from a_t, right portion from b_t
            if bar_x > 0:
                out_t[:, :bar_x] = a_t[:, :bar_x]
            if bar_x < trans_width:
                out_t[:, bar_x:] = b_t[:, bar_x:]

            # asymmetric blending forward from the bar across fade_width
            if fade_width > 0 and 0 < bar_x < trans_width:
                x0 = bar_x
                x1 = min(trans_width, bar_x + fade_width)
                blend_a = a_t[:, x0:x1]
                blend_b = b_t[:, x0:x1]
                mask = np.linspace(0, 1, x1 - x0)[None, :, None]
                out_t[:, x0:x1] = blend_a * (1 - mask) + blend_b * mask

                # ensure everything after blend is taken from b_t
                if x1 < trans_width:
                    out_t[:, x1:] = b_t[:, x1:]

            # invert to original orientation and convert to torch [0..1]
            out_buf = CompareVideo.inverse_ops(out_t, ops)
            out = np.clip(out_buf / 255.0, 0, 1).astype(np.float32)
            frames.append(torch.from_numpy(out))

        return frames

    @staticmethod
    def create_fade_frames_from_arrays(arr_start, arr_end, duration: float, fps: float):
        """Create alpha-blended crossfade frames between arr_start and arr_end.

        Inputs are HWC numpy arrays; returns a list of torch tensors HWC float32 in 0..1 range.
        """
        import numpy as np
        import torch

        a = arr_start.astype(np.float32)
        b = arr_end.astype(np.float32)

        num_frames = max(2, int(duration * fps))
        frames = []

        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            out = (1.0 - alpha) * a + alpha * b
            out = np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)
            frames.append(torch.from_numpy(out))

        return frames
