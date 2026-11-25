"""Layout Reference Cutout — isolate a subject from a reference image using SAM3 and save the
clean cutout into `input/fresco/cutouts/` so Mixins and Boxes can reuse it across layouts.

Runs inside ComfyUI's execution queue (queued directly by the Mixin / Box reference cutout dialog
or usable standalone in a graph) so SAM3 loads safely under dynamic VRAM.
"""

from __future__ import annotations

import json

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from ..fresco import _segment
from ..fusion._sources import open_image, resolve_path
from . import _cutout


def _pil_to_tensor(image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


class NynxzLayoutCutout(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout.Cutout",
            display_name="Layout Reference Cutout",
            description=(
                "Segments a subject out of a reference image with SAM3 (by signed text or click "
                "points), removes the background, optionally crops tight, and saves the cutout "
                "into input/fresco/cutouts/ for reuse in Mixins and Boxes."
            ),
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "image_ref",
                    default="",
                    tooltip="Relative path of the reference image (e.g. fresco/photo.png).",
                ),
                io.Combo.Input(
                    "image_type",
                    options=["input", "output", "temp"],
                    default="input",
                ),
                io.Combo.Input(
                    "sam3",
                    options=_segment.checkpoints(),
                    default=_segment.NONE,
                    tooltip="SAM3 checkpoint used to segment the reference image.",
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    tooltip="Signed SAM3 prompt (e.g. 'woman', 'jacket, -hands').",
                ),
                io.String.Input(
                    "points",
                    default="[]",
                    tooltip='Optional JSON click points: [{"x": 0.5, "y": 0.5, "include": true}].',
                ),
                io.Combo.Input(
                    "background",
                    options=list(_cutout.BACKGROUNDS),
                    default="white",
                    tooltip="Background fill outside the cutout. White or grey works best for "
                    "Qwen3-VL / Qwen3.5 vision encoders; transparent writes an RGBA PNG.",
                ),
                io.Boolean.Input(
                    "crop",
                    default=True,
                    tooltip="Crop tightly around the segmented subject (+6% padding).",
                ),
                io.Int.Input("grow", default=0, min=-64, max=64),
                io.Int.Input("feather", default=2, min=0, max=32),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="ref"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_ref="",
        image_type="input",
        sam3=_segment.NONE,
        prompt="",
        points="[]",
        background="white",
        crop=True,
        grow=0,
        feather=2,
    ) -> io.NodeOutput:
        if not sam3 or sam3 == _segment.NONE:
            raise ValueError("Pick a SAM3 checkpoint first.")
        path = resolve_path(image_ref, image_type)
        pil_src = open_image(path) if path else None
        if pil_src is None:
            raise ValueError(f"Could not open reference image `{image_ref}` ({image_type}).")

        parsed_points = _cutout.parse_points(points)
        text = str(prompt or "").strip()
        if not parsed_points and not text:
            raise ValueError("Describe what to cut out (e.g. 'person') or click on the image.")

        frame = _pil_to_tensor(pil_src)
        height, width = frame.shape[1], frame.shape[2]
        model, sam_clip = _segment.load(sam3)

        if parsed_points:
            from comfy_extras.nodes_sam3 import SAM3_Detect

            pos = [
                {"x": p["x"] * width, "y": p["y"] * height} for p in parsed_points if p["include"]
            ]
            neg = [
                {"x": p["x"] * width, "y": p["y"] * height}
                for p in parsed_points
                if not p["include"]
            ]
            masks = SAM3_Detect.execute(
                model,
                frame,
                positive_coords=json.dumps(pos) if pos else None,
                negative_coords=json.dumps(neg) if neg else None,
            ).result[0]
        else:
            if sam_clip is None:
                raise ValueError("This SAM3 checkpoint has no text encoder — click points instead.")
            from ..sam3 import _detect
            from ..sam3 import _prompt as sam3_prompt

            masks, _ = _detect.detect(model, sam_clip, frame, sam3_prompt.parse(text))

        result = _cutout.apply_cutout(
            pil_src,
            masks,
            background=background,
            crop=bool(crop),
            grow=int(grow),
            feather=int(feather),
        )
        if result is None:
            raise ValueError(
                "SAM3 found nothing matching that — try different words or click points."
            )

        out_img, out_mask = result
        rel_ref = _cutout.save_cutout(out_img, folder_paths.get_input_directory(), image_ref)
        w, h = out_img.size
        return io.NodeOutput(
            _pil_to_tensor(out_img),
            out_mask.unsqueeze(0),
            rel_ref,
            ui={"cutout": [{"ref": rel_ref, "type": "input", "width": w, "height": h}]},
        )
