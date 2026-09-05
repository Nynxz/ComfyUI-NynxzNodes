"""Regions from Masks — a MASK batch becomes the regions a LoRA can be gated to.

The intended source is **SAM3 Detect** (ComfyUI core, `comfy_extras/nodes_sam3.py`): give it the
image and a text conditioning naming what to find, turn `individual_masks` on, and it returns one
mask per object. Anything else that emits a MASK works the same way: BiRefNet, a matte, a mask you
painted.

**The SAM3 trap, because it costs an evening and reads as a broken node.** SAM3's text prompt has a
`category:N` syntax, and **N defaults to 1** — `_parse_prompts` in `comfy/text_encoders/sam3_clip.py`
returns `("person", 1)` for a bare `person`, and `nodes_sam3.py` then keeps only the single
highest-scoring detection. So a prompt of `person` returns exactly one mask no matter what
`individual_masks` is set to, and it looks for all the world like per-object masks are broken.
Write `person:2` for two of the same thing. Two *different* subjects need no syntax at all —
`woman, man` is two categories and already gives two masks.

Two knobs that are not cosmetic:

  * **`grow`.** A segmenter traces the silhouette it sees. A LoRA gated to precisely that
    silhouette can restyle the face and not the hair, the jawline or the coat's outline, because
    those pixels are on the far side of the boundary it was handed — and those are exactly the
    pixels that carry a character's identity at a glance. Growing gives the LoRA room to redraw the
    edge. This is the first thing to raise if the output looks like the right face pasted into the
    wrong outline.
  * **`feather`.** A hard boundary means the LoRA switches on across one token, which reads as a
    seam. Softening trades a little bleed for a handover.

Both are in pixels of the mask's own frame, applied before projection. Worth knowing what that
projection costs: Krea 2's canvas grid is `(H/16, W/16)`, so a 1024x1024 generation gates over a
64x64 grid. Mask precision far past that is thrown away, and `grow`/`feather` of a few pixels are
sub-token adjustments. Think in tens.

Regions are numbered, not named. The number is the mask's position in the batch, and it is what
every other node here indexes by — so a Region LoRA row pointing at region 2 is the second mask, in
every report, with nothing to keep in sync.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from .._lib.io_types import advanced
from . import _mask
from ._base import RegionNode
from ._io_types import Regions


class NynxzRegionsFromMasks(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.FromMasks",
            display_name="Regions from Masks",
            description=(
                "Turns a MASK batch — SAM3 Detect with individual_masks on, or any other segmenter "
                "— into regions that LoRAs can be gated to. One region per mask in the batch."
            ),
            is_experimental=True,
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip="One mask per region. SAM3 Detect's `masks` output with "
                    "`individual_masks` on is the intended source: one mask per detected object.",
                ),
                io.Int.Input(
                    "grow",
                    default=12,
                    min=0,
                    max=256,
                    tooltip="Dilate each mask, in pixels. A segmenter hugs the silhouette, and a "
                    "LoRA gated to exactly that cannot change the hair, jaw or clothing outline — "
                    "the parts that carry identity at a glance. Raise this first if the face is "
                    "right but the outline is wrong.",
                ),
                io.Int.Input(
                    "feather",
                    default=8,
                    min=0,
                    max=256,
                    tooltip="Soften each mask's edge, in pixels, so the LoRA hands over gradually "
                    "instead of switching on across a single token.",
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How much of these regions Region Latent may rewrite. 1 = rebuild "
                    "freely, which is right when the masks are subjects you are replacing; lower "
                    "it when they are areas you are only restyling. Region Denoise overrides it "
                    "per region. Only Region Latent reads it.",
                ),
                advanced(
                    io.Combo.Input(
                        "fit",
                        options=list(_mask.FITS),
                        default=_mask.DEFAULT_FIT,
                        tooltip="How the mask's frame maps onto the generated canvas. `stretch` is "
                        "right when the mask was drawn on the image being edited at the framing being "
                        "generated, which is the usual case. The others preserve aspect for a mask "
                        "that came from a differently-shaped source.",
                    )
                ),
                Regions.Input(
                    "regions",
                    optional=True,
                    tooltip="Chain from another regions node to add these to an existing set.",
                ),
            ],
            outputs=[
                Regions.Output(
                    display_name="regions",
                    tooltip="Wire into Region LoRAs to bind a LoRA to each one. Region Inspect "
                    "shows what is on it.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, masks, grow=12, feather=8, denoise=1.0, fit=_mask.DEFAULT_FIT, regions=None
    ) -> io.NodeOutput:
        built = [
            {
                "mask": _mask.shape_mask(raw, grow=grow, feather=feather),
                "fit": fit,
                "denoise": float(denoise),
            }
            for raw in _mask.split_batch(masks)
        ]

        if not built:
            raise ValueError(
                "No masks were found in the MASK input. SAM3 Detect returns a union mask unless "
                "`individual_masks` is on — with it off there is one region covering everything, "
                "which cannot separate anything."
            )

        if len(built) == 1:
            # Overwhelmingly the SAM3 detection cap rather than a genuine one-subject image, and it
            # is invisible from the graph: the node ran, a mask came out, it is just the wrong
            # number of them. Warned here because this is the first place the count is observable.
            logging.warning(
                "Nynxz Regions: only ONE mask arrived. If that is not what you expected, SAM3's "
                "prompt syntax is `category:N` and N defaults to 1 — write `person:2`, or name two "
                "categories (`woman, man`)."
            )
        return io.NodeOutput([*(regions or []), *built])
