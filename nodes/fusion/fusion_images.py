"""Fusion Images — collect several IMAGE sockets into a fusion_input, no chaining.

The plain-wire collector: wire each Load Image (or any upstream IMAGE — a sampler, a mask
composite) into an autogrow socket and this outputs a `fusion_input` for the encode node. It's
the wire-side sibling of the Fusion Input grid, which reaches files on disk instead — pick
whichever matches how your images arrive.

One `strength` and one `fit` apply to every wired image. They're single native widgets rather
than per-socket fields because an autogrow template takes exactly one input per repeat, so
there's no way to pair each `image_N` with its own controls — for per-image strength or fit,
use the grid instead. A batched IMAGE contributes one source per frame at that same strength
and fit, and an optional upstream fusion_input is prepended so this still chains with the grid.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import FusionNode
from ._fusion import DEFAULT_FIT, FIT_MODES
from ._io_types import FusionInput


def _image_sockets() -> io.Autogrow.Input:
    """Sixteen IMAGE sockets. A cap, not an allocation — ComfyUI shows one empty socket past
    the last filled one. Named image_1..image_16; `execute` recovers order from that suffix.
    """
    return io.Autogrow.Input(
        "images",
        template=io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"image_{i}" for i in range(1, 17)],
            min=1,
        ),
    )


class NynxzFusionImages(FusionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Fusion.Images",
            display_name="Fusion Images",
            description="Collect several IMAGE sockets into a fusion_input — wire Load Image "
            "nodes straight in, no chaining. One strength and fit apply to every wired image. "
            "Feeds a fusion encode node.",
            inputs=[
                _image_sockets(),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Relative prevalence applied to every wired image, matching the "
                    "grid's strength. The encode node still normalizes across all sources, so "
                    "this only matters against an upstream fusion_input's own strengths.",
                ),
                io.Combo.Input(
                    "fit",
                    options=FIT_MODES,
                    default=DEFAULT_FIT,
                    tooltip="How every image is framed into the shared grid. contain = whole image, "
                    "letterboxed; cover = center-crop to fill; stretch = distort. The encode "
                    "node's fit override can still force one mode for all sources.",
                ),
                FusionInput.Input(
                    "fusion_input",
                    optional=True,
                    tooltip="Optional upstream Fusion Input / Images — its images come first.",
                ),
            ],
            outputs=[FusionInput.Output(display_name="fusion_input")],
        )

    @classmethod
    def execute(
        cls, images: io.Autogrow.Type, strength=1.0, fit=DEFAULT_FIT, fusion_input=None
    ) -> io.NodeOutput:
        # Copy the upstream list — ComfyUI hands the same object to every consumer, so appending
        # in place would make a fan-out silently accumulate images.
        sources: list[dict] = list(fusion_input or [])
        fit = fit if fit in FIT_MODES else DEFAULT_FIT
        try:
            strength = max(0.0, float(strength))
        except (TypeError, ValueError):
            strength = 1.0

        # One source per frame, in socket order. A batched IMAGE spends the shared strength/fit
        # across each of its frames. Flattened inline (rather than via a helper) so this node
        # stays decoupled from the fusion weight-math module — with strength and fit shared, the
        # per-socket grouping the grid needs buys nothing here.
        for name in sorted(images or {}, key=lambda value: int(value.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            if image.ndim == 3:
                image = image.unsqueeze(0)
            for i in range(image.shape[0]):
                sources.append(
                    {
                        "image": image[i : i + 1].clone(),
                        "strength": strength,
                        "fit": fit,
                        "label": f"image {len(sources) + 1}",
                    }
                )
        return io.NodeOutput(sources)
