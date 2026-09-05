"""Region Denoise — how much of one region Region Latent may rewrite.

One node, one job, no magic numbers. Denoise is a property of the *region*, not of the LoRA you
attached to it, so it lives here rather than as a `-1 means leave it alone` sentinel on every node
that binds something — a slider labelled `denoise` sitting at -1 is a bad thing to hand somebody.

`Regions from Masks` sets the baseline for the regions it makes. Use this node wherever one region
needs something different from the rest.

**Only `Region Latent` reads it.** On a text-to-image graph it does nothing at all.

What the number means, since it decides how a swap looks:

  * **1.0** — rebuild the region from noise. The model has no idea what was there, so the pose, the
    silhouette and the framing are all up for grabs, and the result frequently does not line up
    with its surroundings. This is what a "cut out" looks like.
  * **0.5-0.8** — the region's structure survives and its appearance changes. This is the range a
    character swap usually wants.
  * **below ~0.3** — the source shows through and the change is cosmetic.

If a swap is coming out cut-out-looking at a high denoise, the fix is usually not this number: give
the model the source image as a **reference latent** (a `ReferenceLatent` node plus an
`Edit Model Reference Method` set to `index_timestep_zero`) so it can see the original pose while
it rewrites, rather than inventing one.
"""

from __future__ import annotations

from comfy_api.latest import io

from . import _bind
from ._base import RegionNode
from ._io_types import Regions


class NynxzRegionDenoise(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.Denoise",
            display_name="Region Denoise",
            description=(
                "Sets how much of one region Region Latent may rewrite. Only that node reads it."
            ),
            is_experimental=True,
            inputs=[
                Regions.Input("regions"),
                io.Int.Input(
                    "region",
                    default=1,
                    min=1,
                    max=64,
                    tooltip="Which region, numbered as the reports list them (1 = first).",
                ),
                io.Float.Input(
                    "denoise",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="1 rebuilds the region from noise — pose and silhouette included, "
                    "which is what makes a swap look cut out. 0.5-0.8 keeps the structure and "
                    "changes the appearance. Below 0.3 the source shows through.",
                ),
            ],
            outputs=[Regions.Output(display_name="regions")],
        )

    @classmethod
    def execute(cls, regions, region=1, denoise=0.75) -> io.NodeOutput:
        entries = list(regions or [])
        index = _bind.resolve(entries, region)
        entries[index] = {**entries[index], "denoise": float(denoise)}
        return io.NodeOutput(entries)
