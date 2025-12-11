from typing import Callable, Iterable, List, Optional
import numpy as np
import torch
import math

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
    def get_option_value(options, key, default):
        """Get option value from options object with fallback to default."""
        if options and hasattr(options, key):
            return getattr(options, key)
        return default

    @staticmethod
    def node_inputs() -> List[io.Input]:
        """Return a list of io.Input objects reused by compare nodes.

        Common options now moved to the options node for cleaner interfaces.
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
            io.Float.Input("duration", default=1.0, min=0.1, max=10.0, step=0.1,
                           tooltip="Duration of the compare video in seconds", display_name="Duration (s)"),
            io.Float.Input("time_padding", default=0.0, min=0.0, max=5.0, step=0.1,
                           tooltip="Optional time padding (in seconds) added to start and end of the video", display_name="Time Padding (s)"),
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
        return np.fliplr(x)

    @staticmethod
    def _op_transpose_hw(x):
        return np.transpose(x, (1, 0, 2))

    @staticmethod
    def ops_for_angle(angle_degrees: float) -> List[Callable]:
        """Return the list of ops for a wipe angle in degrees.

        0° = right, 90° = down, 180° = left, 270° = up
        """
        import math

        # Normalize angle to 0-360
        angle = angle_degrees % 360

        # Determine which quadrant and base operations needed
        if 0 <= angle < 90:
            # First quadrant: right to down
            # For now, map to closest cardinal direction
            if angle < 45:
                return []  # LEFT_TO_RIGHT equivalent
            else:
                # TOP_TO_BOTTOM equivalent
                return [CompareVideo._op_transpose_hw]
        elif 90 <= angle < 180:
            # Second quadrant: down to left
            if angle < 135:
                # TOP_TO_BOTTOM equivalent
                return [CompareVideo._op_transpose_hw]
            else:
                return [CompareVideo._op_flip_lr]  # RIGHT_TO_LEFT equivalent
        elif 180 <= angle < 270:
            # Third quadrant: left to up
            if angle < 225:
                return [CompareVideo._op_flip_lr]  # RIGHT_TO_LEFT equivalent
            else:
                # BOTTOM_TO_TOP equivalent
                return [CompareVideo._op_transpose_hw, CompareVideo._op_flip_lr]
        else:
            # Fourth quadrant: up to right
            if angle < 315:
                # BOTTOM_TO_TOP equivalent
                return [CompareVideo._op_transpose_hw, CompareVideo._op_flip_lr]
            else:
                return []  # LEFT_TO_RIGHT equivalent

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
    def create_frames_from_arrays_angle(arr_start, arr_end, duration: float, fps: float, wipe_angle: float, fade_width: Optional[int] = 20, fade_fraction: float = 0.5, time_padding: float = 0.0):
        """Generate torch tensors frames for the wipe animation from any angle.

        Args:
            wipe_angle: Angle in degrees (0° = left to right, 90° = top to bottom)
            time_padding: Time in seconds to add static frames at start and end
        """

        # normalize to float32 in 0..255 range
        a = arr_start.astype(np.float32)
        b = arr_end.astype(np.float32)

        height, width = a.shape[:2]
        num_frames = max(2, int(duration * fps))
        padding_frames = int(time_padding * fps) if time_padding > 0 else 0

        # Convert angle to radians
        angle_rad = math.radians(wipe_angle)

        # Calculate the direction vector for the wipe
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(
            np.arange(height), np.arange(width), indexing='ij')

        # Calculate distance along the wipe direction for each pixel
        # Project each pixel onto the wipe direction vector
        if dx != 0:
            # For non-vertical wipes, use x-projection
            projected = x_coords * dx + y_coords * dy
        else:
            # For vertical wipes, use y-projection
            projected = y_coords

        # Normalize projection to 0-1 range
        proj_min = projected.min()
        proj_max = projected.max()
        if proj_max > proj_min:
            normalized_proj = (projected - proj_min) / (proj_max - proj_min)
        else:
            normalized_proj = np.zeros_like(projected)

        # Ensure sensible fade_width
        if fade_width is None:
            fade_width = max(1, int(min(width, height) * fade_fraction))

        fade_fraction_normalized = fade_width / max(width, height)

        frames = []

        # Add padding frames at start (show start image)
        start_tensor = np.clip(a / 255.0, 0, 1).astype(np.float32)
        for _ in range(padding_frames):
            frames.append(torch.from_numpy(start_tensor))

        # Generate transition frames
        for i in range(num_frames):
            progress = i / (num_frames - 1)

            # Create the output frame
            out = np.copy(a)  # Start with start image

            if fade_width > 0:
                # Create smooth transition
                # Pixels are revealed when their normalized projection < progress
                fade_mask = (normalized_proj > (
                    progress - fade_fraction_normalized)) & (normalized_proj <= progress)

                # For fade region, blend between start and end
                if np.any(fade_mask):
                    # Calculate blend factor for fade region
                    fade_region_proj = normalized_proj[fade_mask]
                    blend_factor = (progress - fade_region_proj) / \
                        fade_fraction_normalized
                    blend_factor = np.clip(blend_factor, 0, 1)

                    # Apply blending
                    out[fade_mask] = (a[fade_mask] * (1 - blend_factor[..., np.newaxis]) +
                                      b[fade_mask] * blend_factor[..., np.newaxis])

                # Set fully revealed pixels to end image
                solid_reveal = normalized_proj < (
                    progress - fade_fraction_normalized)
                if np.any(solid_reveal):
                    out[solid_reveal] = b[solid_reveal]
            else:
                # Sharp cutoff - no fade
                reveal_mask = normalized_proj <= progress
                out[reveal_mask] = b[reveal_mask]

            # Convert to torch tensor
            out_tensor = np.clip(out / 255.0, 0, 1).astype(np.float32)
            frames.append(torch.from_numpy(out_tensor))

        # Add padding frames at end (show end image)
        end_tensor = np.clip(b / 255.0, 0, 1).astype(np.float32)
        for _ in range(padding_frames):
            frames.append(torch.from_numpy(end_tensor))

        return frames

    @staticmethod
    def create_frames_from_arrays(arr_start, arr_end, duration: float, fps: float, wipe_direction: Optional[str] = None, wipe_angle: Optional[float] = None, fade_width: Optional[int] = 20, fade_fraction: float = 0.5, time_padding: float = 0.0):
        """Generate torch tensors frames for the wipe animation from numpy arrays.

        Inputs are HWC numpy arrays (float or uint8). The function returns a
        list of torch tensors HWC float32 in 0..1 range, stacked per-frame.

        Args:
            wipe_direction: String direction (legacy support)
            wipe_angle: Angle in degrees (0-360), takes precedence if provided
            time_padding: Time in seconds to add static frames at start and end
        """

        # Use angle-based approach if angle is provided
        if wipe_angle is not None and wipe_angle >= 0:
            return CompareVideo.create_frames_from_arrays_angle(
                arr_start, arr_end, duration, fps, wipe_angle, fade_width, fade_fraction, time_padding
            )

        # Otherwise use the original transform-based approach

        # normalize to float32 in 0..255 range (some inputs are already float)
        a = arr_start.astype(np.float32)
        b = arr_end.astype(np.float32)

        num_frames = max(2, int(duration * fps))
        padding_frames = int(time_padding * fps) if time_padding > 0 else 0

        # pick ops based on direction
        if wipe_direction:
            ops = CompareVideo.ops_for_direction(wipe_direction)
        else:
            ops = []

        a_t = CompareVideo.compose_ops(a, ops)
        b_t = CompareVideo.compose_ops(b, ops)

        trans_width = a_t.shape[1]
        frames = []

        # ensure sensible fade_width
        if fade_width is None:
            fade_width = max(1, int(trans_width * fade_fraction))

        # Add padding frames at start (show start image)
        start_tensor = np.clip(a / 255.0, 0, 1).astype(np.float32)
        for _ in range(padding_frames):
            frames.append(torch.from_numpy(start_tensor))

        # Generate transition frames
        # single reusable buffer
        out_t = None
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            bar_x = int(progress * trans_width)

            if out_t is None:
                out_t = np.empty_like(a_t)

            # left portion from b_t (end image), right portion from a_t (start image)
            # This creates: start -> end wipe (left to right)
            if bar_x > 0:
                out_t[:, :bar_x] = b_t[:, :bar_x]
            if bar_x < trans_width:
                out_t[:, bar_x:] = a_t[:, bar_x:]

            # asymmetric blending forward from the bar across fade_width
            if fade_width > 0 and 0 < bar_x < trans_width:
                x0 = bar_x
                x1 = min(trans_width, bar_x + fade_width)
                blend_a = a_t[:, x0:x1]  # start image (remaining)
                blend_b = b_t[:, x0:x1]  # end image (revealed)
                # Create gradient from 0 to 1 (left to right in blend zone)
                mask = np.linspace(0, 1, x1 - x0)[None, :, None]
                # At x0 (left): mask=0, show b_t (end image)
                # At x1 (right): mask=1, show a_t (start image)
                out_t[:, x0:x1] = blend_b * (1 - mask) + blend_a * mask

                # ensure everything after blend is taken from end image
                if x1 < trans_width:
                    out_t[:, x1:] = a_t[:, x1:]

            # invert to original orientation and convert to torch [0..1]
            out_buf = CompareVideo.inverse_ops(out_t, ops)
            out = np.clip(out_buf / 255.0, 0, 1).astype(np.float32)
            frames.append(torch.from_numpy(out))

        # Add padding frames at end (show end image)
        end_tensor = np.clip(b / 255.0, 0, 1).astype(np.float32)
        for _ in range(padding_frames):
            frames.append(torch.from_numpy(end_tensor))

        return frames

    @staticmethod
    def create_fade_frames_from_arrays(arr_start, arr_end, duration: float, fps: float, time_padding: float = 0.0):
        """Create alpha-blended crossfade frames between arr_start and arr_end.

        Inputs are HWC numpy arrays; returns a list of torch tensors HWC float32 in 0..1 range.
        """

        a = arr_start.astype(np.float32)
        b = arr_end.astype(np.float32)

        num_frames = max(2, int(duration * fps))
        padding_frames = int(time_padding * fps) if time_padding > 0 else 0
        frames = []

        # Add padding frames at start (show start image)
        start_tensor = np.clip(a / 255.0, 0, 1).astype(np.float32)
        for _ in range(padding_frames):
            frames.append(torch.from_numpy(start_tensor))

        # Generate crossfade transition frames
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            out = (1.0 - alpha) * a + alpha * b
            out = np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)
            frames.append(torch.from_numpy(out))

        # Add padding frames at end (show end image)
        end_tensor = np.clip(b / 255.0, 0, 1).astype(np.float32)
        for _ in range(padding_frames):
            frames.append(torch.from_numpy(end_tensor))

        return frames
