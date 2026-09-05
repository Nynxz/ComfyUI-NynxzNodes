"""Region Inspect — break a region set open and look at it, anywhere in the chain.

The debug node. It takes the `regions` wire itself rather than a dedicated `extras` output, and
that is the better shape for one reason: a wire that already exists can be tapped at *any* point,
so you can inspect what the source node produced and what the LoRA bindings look like just before
Apply, from one node, without every node upstream having to grow a second socket to make it
possible.

What it gives you:

  * **`masks`** — every region as a MASK batch, in order. Feed it to a Mask Preview, or back into
    `Regions from Masks` after editing, or into any mask node at all. This is the output that turns
    the region set from an opaque wire into something you can work with.
  * **`preview`** — all of them at once, each in its own colour, so overlaps and holes are visible
    at a glance rather than one mask at a time.
  * **`report`** — coverage, LoRA bindings, overlap, and how much of the frame nothing claims.
  * **`count`** — how many regions are on the wire, for wiring into anything that needs the number.

This is a *pre-sampling* view: the masks as authored. `Regions Preview` is the *post*-sampling one,
showing the map that actually gated the LoRAs after tracking moved it. Two different questions —
"is my region set right?" and "did it survive the run?" — and it is worth being sure which one you
are asking before turning a knob. Both draw a region in the same colour so they can be compared.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from .._lib.io_types import advanced
from . import _mask, _render
from ._base import RegionNode
from ._io_types import Regions


class NynxzRegionInspect(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.Inspect",
            display_name="Region Inspect",
            description=(
                "Breaks a region set open: every mask as a MASK batch, a colour preview of all of "
                "them at once, and a report. Tap the regions wire anywhere."
            ),
            is_experimental=True,
            inputs=[
                Regions.Input("regions"),
                advanced(
                    io.Int.Input(
                        "size",
                        default=512,
                        min=64,
                        max=2048,
                        step=64,
                        tooltip="Resolution to normalise the masks to. They arrive at whatever size "
                        "their producer used, and a MASK batch has to be one shape.",
                    )
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="masks",
                    tooltip="One mask per region, in order. Wire to a Mask Preview, or edit and "
                    "feed back through Regions from Masks.",
                ),
                io.Image.Output(
                    display_name="preview",
                    tooltip="All regions at once, one colour each. Overlaps read as blends and "
                    "unclaimed area reads as dark.",
                ),
                io.String.Output(display_name="report"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, regions, size=512) -> io.NodeOutput:
        entries = list(regions or [])
        if not entries:
            raise ValueError(
                "No regions arrived on the wire. Put a Regions from Masks node upstream of this "
                "one."
            )
        size = max(int(size), 64)
        stacked = _mask.project_all(entries, (size, size))
        claim = _mask.share(stacked)

        grids = stacked.reshape(len(entries), size, size)
        overlay = torch.zeros(size, size, 3)
        for index in range(len(entries)):
            overlay += grids[index].unsqueeze(-1) * _render.hue(index)
        preview = overlay.clamp(0.0, 1.0).unsqueeze(0)

        return io.NodeOutput(grids, preview, _report(entries, stacked, claim), len(entries))


def _report(entries, stacked, claim) -> str:
    lines = [f"{len(entries)} region(s) on the wire"]
    for index, region in enumerate(entries):
        bound = region.get("loras") or []
        lora = (
            ", ".join(f"{b.get('lora_name')} @ {b.get('strength', 1.0):.2f}" for b in bound)
            or "(no LoRA — territory only)"
        )
        raw = float(stacked[index].mean())
        after = float(claim[index].mean())
        # Two numbers because they answer different questions: how much a region covers, and how
        # much it actually gets to keep once the others have taken their share of the overlap.
        shared = "" if abs(raw - after) < 5e-3 else f"  (keeps {after:.1%} after overlap)"
        denoise = float(region.get("denoise", 1.0))
        lines.append(
            f"  region {index + 1}: {raw:.1%} of frame{shared}, denoise {denoise:.2f} — {lora}"
        )

    free = float(_mask.background(claim).mean())
    lines.append(f"  unclaimed: {free:.1%} of the frame carries no LoRA at all")
    lines.append(f"  overlap: {_mask.overlap(stacked):.1%} claimed by more than one region")

    if not any(r.get("loras") for r in entries):
        lines.append(
            "\n  NO LoRA IS BOUND ANYWHERE. Regions Apply will refuse this — put a Region LoRAs "
            "node between the source and it."
        )
    if free < 0.02 and len(entries) > 1:
        lines.append(
            "\n  NOTHING IS BACKGROUND. Every part of the frame is claimed, so each LoRA fires "
            "over its share of the whole canvas — close to what merging them does. Grow the masks "
            "less."
        )
    if free > 0.9:
        lines.append(
            "\n  ALMOST NOTHING IS CLAIMED. The masks are nearly empty at this resolution — check "
            "the source node before going further; every LoRA would be doing nothing."
        )
    return "\n".join(lines)
