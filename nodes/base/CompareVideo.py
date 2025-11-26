from enum import Enum
from typing import Callable, Iterable, List, Optional

from comfy_api.latest import io


class CompareVideo:
    """Base helper class with reusable inputs and utilities for compare-type nodes.

    This class intentionally provides only helpers (static methods / class
    attributes) — it is not a ComfyUI node itself. Nodes can import and reuse
    these helpers to keep code consistent across multiple node implementations.
    """

    # Common upscale/resampling method options shown in the UI
    UPSCALE_METHODS = ["nearest-exact",
                       "bilinear", "area", "bicubic", "lanczos"]

    class WipeDirectionsEnum(Enum):
        LEFT_TO_RIGHT = "Swipe Left to Right"
        RIGHT_TO_LEFT = "Swipe Right to Left"
        TOP_TO_BOTTOM = "Swipe Top to Bottom"
        BOTTOM_TO_TOP = "Swipe Bottom to Top"

    @staticmethod
    def node_inputs() -> List[io.Input]:
        """Return a list of io.Input objects reused by compare nodes.

        Accepts a fade_width_default so callers can customize the default.
        """
        return [
            io.Image.Input("start_image"),
            io.Image.Input("end_image"),
            io.Boolean.Input("force_upscale", default=False),
            io.Float.Input("duration", default=1.0,
                           min=0.1, max=10.0, step=0.1),
            io.Float.Input("fps", default=8.0, min=1, max=60, step=1),

            io.Combo.Input(
                "upscale_method", options=CompareVideo.UPSCALE_METHODS, default="lanczos"),

        ]

    # ------- small reusable helpers that keep node code tidy -------
    @staticmethod
    def to_pil(img):
        """Convert a comfy node image input (PIL, numpy, or torch) into an RGB PIL.Image

        This function mirrors the existing conversion logic used by the node and
        centralizes it so it can be re-used by any node.
        """
        from PIL import Image
        import numpy as np

        if isinstance(img, Image.Image):
            return img.convert("RGB")

        is_torch = hasattr(img, "cpu") and hasattr(img, "numpy")
        if is_torch:
            arr = img.cpu().numpy()
        else:
            arr = np.asarray(img)

        arr = np.squeeze(arr)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        if arr.ndim == 3:
            # support CHW or HWC
            if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))

        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0, 1) * 255

        arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]

        return Image.fromarray(arr)

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
            CompareVideo.WipeDirectionsEnum.LEFT_TO_RIGHT.value: [],
            CompareVideo.WipeDirectionsEnum.RIGHT_TO_LEFT.value: [CompareVideo._op_flip_lr],
            CompareVideo.WipeDirectionsEnum.TOP_TO_BOTTOM.value: [CompareVideo._op_transpose_hw],
            CompareVideo.WipeDirectionsEnum.BOTTOM_TO_TOP.value: [CompareVideo._op_transpose_hw, CompareVideo._op_flip_lr],
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
