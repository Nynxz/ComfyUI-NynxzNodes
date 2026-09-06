"""SAM3 Detect (Signed) — stock SAM3 Detect, plus `-term` to subtract from what the rest found.

`person:2, -head:2` is two people with their heads taken out of the mask. The syntax is core's
`category:N`, unchanged, with one character added; see `_prompt.py` for the parse and for the one
place the semantics deliberately differ.

The node takes `CLIP` and a prompt where stock takes a `CONDITIONING`, because that is what a sign
costs. Core parses `category:N` in the tokenizer and hands the detector anonymous
`(embedding, max_detections)` pairs, so a sign written into a `CLIPTextEncode` upstream would be
gone before this node could read it. Encoding each term here keeps the sign attached to its term.

Everything else matches stock: the same detector calls, core's own mask refinement, and the same
two outputs in the same shapes. `individual_masks` off is the union of the residuals; on, it is one
mask per surviving positive, which is what **Regions from Masks** wants.

Text prompts only. Stock's `bboxes` and `positive_coords`/`negative_coords` inputs go through
SAM3's decoder rather than its detector, and nothing there carries a category to sign — keep using
`SAM3 Detect` for those.
"""

from __future__ import annotations

from comfy_api.latest import io

from .._lib.io_types import advanced
from . import _detect, _prompt
from ._base import SAM3Node


class NynxzSAM3DetectSigned(SAM3Node):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="SAM3.DetectSigned",
            display_name="SAM3 Detect (Signed)",
            description=(
                "SAM3 Detect with subtraction. Prefix a term with '-' to remove it from what the "
                "other terms found: 'person:2, -head:2' masks two people minus their heads. "
                "Unprefixed terms behave exactly as they do in SAM3 Detect."
            ),
            search_aliases=[
                "sam3",
                "segment",
                "segment anything",
                "negative mask",
                "subtract mask",
                "open vocabulary",
            ],
            inputs=[
                io.Model.Input("model", tooltip="The SAM3 model, same as SAM3 Detect takes."),
                io.Clip.Input(
                    "clip",
                    tooltip="The SAM3 CLIP. This node encodes each term itself — a sign written "
                    "into a CLIPTextEncode would be lost before the detector saw it.",
                ),
                io.Image.Input("image"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    placeholder="person:2, -head:2",
                    tooltip="Comma-separated terms, SAM3's own 'category:N' syntax. A leading '-' "
                    "subtracts that term from the rest. N caps how many of a term to keep, "
                    "best-scoring first; it defaults to 1 on a positive (SAM3's own default — "
                    "write 'person:2' for two people) and to all matches on a negative, since "
                    "removing one of two heads leaves the other one in the mask.",
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Detection confidence floor, applied to every term.",
                ),
                io.Int.Input(
                    "refine_iterations",
                    default=2,
                    min=0,
                    max=5,
                    tooltip="SAM decoder refinement passes per detection, positives and negatives "
                    "alike. 0 uses the raw detector masks.",
                ),
                io.Boolean.Input(
                    "individual_masks",
                    default=False,
                    tooltip="One mask per surviving positive instead of their union. Turn this on "
                    "to feed Regions from Masks.",
                ),
                advanced(
                    io.Int.Input(
                        "negative_grow",
                        default=4,
                        min=0,
                        max=256,
                        tooltip="Dilate the negatives, in pixels, before removing them. A negative "
                        "traces its own silhouette, so subtracting it exactly tends to leave a "
                        "ring of its edge behind — a collar of neck where a head was. Raise this "
                        "if the hole has a halo; lower it to 0 if the removal eats too much.",
                    )
                ),
                advanced(
                    io.Boolean.Input(
                        "drop_empty",
                        default=False,
                        tooltip="Drop positives the negatives consumed entirely, rather than "
                        "emitting a blank mask for them. Off by default because dropping "
                        "renumbers everything after it, and region numbers are what Region LoRAs "
                        "bind to.",
                    )
                ),
            ],
            outputs=[
                io.Mask.Output("masks"),
                io.BoundingBox.Output(
                    "bboxes",
                    tooltip="One box per output mask, measured from the mask after subtraction so "
                    "the two always describe the same pixels.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        image,
        prompt="",
        threshold=0.5,
        refine_iterations=2,
        individual_masks=False,
        negative_grow=4,
        drop_empty=False,
    ) -> io.NodeOutput:
        terms = _prompt.parse(prompt)
        if not terms:
            raise ValueError(
                "The prompt is empty. Name at least one thing to find, and prefix anything to "
                "remove from it with '-' — for example 'person:2, -head:2'."
            )
        if all(term.negative for term in terms):
            listed = ", ".join(str(term) for term in terms)
            raise ValueError(
                f"Every term in the prompt is negative ({listed}), so there is nothing to subtract "
                "from. A signed prompt needs at least one unprefixed term: 'person:2, -head:2'."
            )

        masks, bboxes = _detect.detect(
            model,
            clip,
            image,
            terms,
            threshold=threshold,
            refine_iterations=refine_iterations,
            individual_masks=individual_masks,
            negative_grow=negative_grow,
            drop_empty=drop_empty,
        )
        return io.NodeOutput(masks, bboxes)
