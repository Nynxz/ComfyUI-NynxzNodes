"""Regions Apply — patch the model so each region's LoRA and prompt reach only that region's tokens.

Model in, model out. Nothing else changes: the CONDITIONING goes to the sampler untouched, a stock
`LoraLoader` upstream still merges an overall style into the weights as usual, and the branches
added here sit on top of it additively.

Every knob has an off position that recovers a baseline you can compare against. That is deliberate — this was resolved by bisection and never by tuning:

    track = 0        the mask is a cage. Exactly "regional LoRA with the mask you drew" — the
                     behaviour everything else has to beat.
    release = 0      the gate stops dead at the mask edge, all run.
    smooth = 0       the map is used exactly as tracked, speckle and all.
    preheat = 1      no schedule. LoRAs at full strength from step 0, map free all run.
    isolation = 0    attention untouched. The default; raise it only with a specific leak in hand.
    region_prompt    off: the prompt is everyone's, and a LoRA's text side (its `txtfusion`
                     layers, and what it does to the prompt tokens) is not applied at all.
    layout_lock = 0  no second pass. The default. Above 0, the first steps take their layout from
                     the base model (see `_lock`) — try it with preheat 1.

Set them all and this is the plain, obvious implementation — which is exactly why they exist.

The order things happen in one step, on Krea 2 (Z-Image and MOSS-Image add the prompt copies in a
`DIFFUSION_MODEL` wrapper and route them in the attention override instead; see `_zimage`, `_moss`):

    post_input   grid known -> masks projected -> each region's prompt copy added after the canvas
                 -> map in force -> gates built -> LoRAs live
    block 0..n   hooks add each region's gated branch to every routed Linear
    ~route_at    canvas features clustered, map allowed to move toward them (unless frozen)

`route_at` is where in the stack the tracker looks. Middle blocks by default: early ones are still
mostly patch statistics with no idea what a subject is, and by the very end the tokens have become
predictions of their own output rather than descriptions of a region. Reading a few blocks around
it and averaging is cheap insurance against guessing wrong.
"""

from __future__ import annotations

from comfy_api.latest import io

from .._zenkit.io_types import advanced
from ._apply import apply_regions
from ._base import RegionNode
from ._io_types import Regions
from ._settings import RegionSettings


class NynxzRegionsApply(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.Apply",
            display_name="Regions Apply",
            description=(
                "Runs each region's LoRA as a live per-token branch gated to that region's tokens, "
                "instead of merging every delta into the weights where they collide, and gives "
                "each region its own prompt when it has one. Composes with ordinary merged "
                "LoRAs. Krea 2, Z-Image and MOSS-Image; tracking and layout lock are Krea 2 only for now."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model", tooltip="A Krea 2, Z-Image or MOSS-Image model. Returned patched."
                ),
                Regions.Input(
                    "regions",
                    tooltip="Regions with LoRAs bound. Regions without one still hold territory.",
                ),
                io.Float.Input(
                    "track",
                    default=0.4,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="How far a region may follow its subject off the mask. 0 pins it to "
                    "the mask exactly — the static-mask baseline. 1 lets the image's own "
                    "clustering decide the shape, with the mask only choosing which cluster is "
                    "whose.",
                ),
                io.Float.Input(
                    "commit_at",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Fraction of the sampler's steps after which the map freezes and the "
                    "LoRAs ramp to full. Before it, layout is the base model's to decide.",
                ),
                io.Float.Input(
                    "preheat",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="LoRA strength multiplier before the commit. 1 disables the schedule "
                    "entirely (full strength from step 0, map never freezes). Not 0 by default: "
                    "with nothing applied early the base model commits a generic identity that "
                    "the LoRA then has to overwrite late, which is its own fight.",
                ),
                io.Float.Input(
                    "release",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="How far the gate lets go of the mask after the commit — the LoRA's "
                    "minimum strength OUTSIDE its own region. A hard edge is right while identity "
                    "is being placed and wrong once texture is being written across it, which is "
                    "what makes a gated subject look cut out of the image. 0 keeps the hard mask. "
                    "This is bleed by definition, so keep it small.",
                ),
                io.Int.Input(
                    "smooth",
                    default=1,
                    min=0,
                    max=8,
                    tooltip="Box-blur the map, in canvas tokens. Tracking is speckled, and an "
                    "ungated token inside a subject reads as a HOLE punched in it. This closes "
                    "those and feathers the boundary in the same pass. 0 disables it.",
                ),
                io.Float.Input(
                    "isolation",
                    default=0.0,
                    min=0.0,
                    max=8.0,
                    step=0.1,
                    tooltip="Attention-logit penalty keeping each region's canvas tokens off the "
                    "other regions'. The one leak per-token gating cannot close: A's tokens attend "
                    "B's, whose K/V were computed with B's branch live. Costs shared lighting and "
                    "contact shadows, so keep it low — past ~2 you get a collage.",
                ),
                io.Boolean.Input(
                    "region_prompt",
                    default=True,
                    tooltip="Give each region with a LoRA its own copy of the prompt, made with that "
                    "LoRA's text side (its txtfusion layers) and carrying the LoRA in the blocks, and "
                    "have the region's tokens read it. A LoRA's text side has no place on the canvas, "
                    "so without this it isn't applied at all and many LoRAs come out weak. Costs a "
                    "longer sequence and masked attention.",
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="The text encoder, for regions given a prompt of their own in "
                    "Region LoRAs. A region reads its prompt where it would read the main one "
                    "(on Z-Image, after the main prompt). Not needed otherwise.",
                ),
                advanced(
                    io.Float.Input(
                        "ramp",
                        default=0.2,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="How many steps' worth the ramp from preheat to full takes, starting "
                        "at commit_at. 0 is a hard switch.",
                    )
                ),
                advanced(
                    io.Int.Input(
                        "refine",
                        default=2,
                        min=0,
                        max=8,
                        tooltip="Clustering rounds per read. 0 turns tracking off entirely regardless "
                        "of `track`.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "route_at",
                        default=0.5,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="Where in the block stack the canvas features are read, as a fraction "
                        "of depth. Middle is the useful part.",
                    )
                ),
                advanced(
                    io.Int.Input(
                        "route_blocks",
                        default=3,
                        min=1,
                        max=8,
                        tooltip="How many blocks around route_at to average. One block is a noisy "
                        "sample of where the subjects are.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "sharpness",
                        default=2.0,
                        min=0.1,
                        max=10.0,
                        step=0.1,
                        tooltip="How decisively the clustering commits each token to one region. In "
                        "standard deviations, so it is scale-free.",
                    )
                ),
                io.Float.Input(
                    "layout_lock",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip="Experimental. Fraction of the steps that take their layout from the "
                    "base model: each of those steps first runs without the region LoRAs, and the "
                    "real pass uses that pass's image queries and keys (QK-Edit). The LoRAs can "
                    "then run at full strength from step 0 without moving the composition — try "
                    "0.25 with preheat 1. Costs a second pass on those steps and about 1.75 GB "
                    "per image at 1024². 0 is off.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(
        cls,
        model,
        regions,
        track=0.4,
        commit_at=0.3,
        preheat=0.35,
        release=0.15,
        smooth=1,
        isolation=0.0,
        region_prompt=True,
        clip=None,
        ramp=0.2,
        refine=2,
        route_at=0.5,
        route_blocks=3,
        sharpness=2.0,
        layout_lock=0.0,
    ) -> io.NodeOutput:
        settings = RegionSettings(
            track=track,
            commit_at=commit_at,
            preheat=preheat,
            release=release,
            smooth=smooth,
            isolation=isolation,
            region_prompt=region_prompt,
            ramp=ramp,
            refine=refine,
            route_at=route_at,
            route_blocks=route_blocks,
            sharpness=sharpness,
            layout_lock=layout_lock,
        )
        return io.NodeOutput(apply_regions(model, regions, settings, clip))
