"""Regions Apply — patch the model so each region's LoRA fires only on that region's tokens.

Model in, model out. Nothing else changes: the CONDITIONING goes to the sampler untouched, a stock
`LoraLoader` upstream still merges an overall style into the weights as usual, and the branches
added here sit on top of it additively.

Three mechanisms, and each one has an off position that recovers a baseline you can compare
against. That is deliberate — this was resolved by bisection and never by tuning:

    track = 0        the mask is a cage. Exactly "regional LoRA with the mask you drew" — the
                     behaviour everything else has to beat.
    release = 0      the gate stops dead at the mask edge, all run.
    smooth = 0       the map is used exactly as tracked, speckle and all.
    preheat = 1      no schedule. LoRAs at full strength from step 0, map free all run.
    isolation = 0    attention untouched. The default; raise it only with a specific leak in hand.

Set them all and this is the plain, obvious implementation — which is exactly why they exist.

The order things happen in one step:

    post_input   grid known -> masks projected -> map in force -> gates built -> LoRAs live
    block 0..n   hooks add each region's gated branch to every routed Linear
    ~route_at    canvas features clustered, map allowed to move toward them (unless frozen)

`route_at` is where in the stack the tracker looks. Middle blocks by default: early ones are still
mostly patch statistics with no idea what a subject is, and by the very end the tokens have become
predictions of their own output rather than descriptions of a region. Reading a few blocks around
it and averaging is cheap insurance against guessing wrong.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from .._lib.io_types import advanced
from . import _branch, _capture, _patches, _schedule, _state
from ._base import RegionNode
from ._io_types import Regions
from ._krea import check_krea2
from ._lora import load_branches


class NynxzRegionsApply(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.Apply",
            display_name="Regions Apply",
            description=(
                "Runs each region's LoRA as a live per-token branch gated to that region's tokens, "
                "instead of merging every delta into the weights where they collide. The "
                "conditioning is not touched, so this composes with ordinary merged LoRAs. "
                "Krea 2 only."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="A Krea 2 model. Returned patched."),
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
        ramp=0.2,
        refine=2,
        route_at=0.5,
        route_blocks=3,
        sharpness=2.0,
    ) -> io.NodeOutput:
        total_blocks = check_krea2(model, "Regions Apply")
        entries = list(regions or [])
        if not entries:
            raise ValueError("No regions arrived on the wire — there is nothing to gate.")
        if not any(r.get("loras") for r in entries):
            raise ValueError(
                "No region has a LoRA bound, so there is nothing for this node to do — it exists "
                "to route LoRAs, which is the one thing masked conditioning cannot. Put a Region "
                "LoRAs node between the regions and this one. If you only want to PROMPT each "
                "region differently, you do not need these nodes at all: core's "
                "ConditioningSetMask is the whole job."
            )

        patched = model.clone()
        loaded, warnings = _load(patched, entries)

        schedule = _schedule.Schedule(
            commit_at=commit_at, preheat=preheat, ramp=ramp, release=release
        )
        state = _state.GateState(entries, schedule, track)
        config = _patches.TrackConfig(
            track=track,
            refine=refine,
            route_at=route_at,
            route_blocks=route_blocks,
            sharpness=sharpness,
            smooth=smooth,
            isolation=isolation,
        )

        patched.set_model_patch(_patches.post_input_patch(state), "post_input")
        patched.set_model_patch(_patches.attn1_patch(state, config), "attn1_patch")
        _branch.attach(patched, loaded, state)
        _capture.clear()

        # To the log rather than to a socket. What loaded and what will run where is worth
        # having when something is wrong and worth nothing the rest of the time, which is exactly
        # a log line and not an output every graph has to route somewhere. Region Inspect shows
        # the bindings before the run and Regions Preview shows the map after it.
        logging.info("Nynxz Regions: %s", _summary(entries, loaded, config, schedule, total_blocks))
        for line in warnings:
            logging.warning("Nynxz Regions: %s", line)
        return io.NodeOutput(patched)


def _load(patcher, entries):
    """`([(region_index, branches)], warnings)` — every LoRA on every region, in region order.

    Pairs and not one entry per region, because a region may carry several. Each LoRA's own
    strength is applied when it LOADS rather than to the region's gate: the gate is per REGION and
    several LoRAs share it, so a per-LoRA strength has nowhere else to live — and a DoRA or an OFT
    is not linear in strength, so inside `calculate_weight` is the only faithful place for it.
    """
    loaded, warnings = [], []
    for index, region in enumerate(entries):
        for bound in region.get("loras") or []:
            name = bound.get("lora_name")
            if not name:
                continue
            branches, notes = load_branches(patcher, name, float(bound.get("strength", 1.0)))
            loaded.append((index, branches))
            warnings.extend(notes)
    return loaded, warnings


def _summary(entries, loaded, config, schedule, total_blocks) -> str:
    """What will actually run. Deliberately states the mechanism state, not just the settings."""
    lines = []
    by_region: dict[int, list] = {}
    for index, branches in loaded:
        by_region.setdefault(index, []).append(branches)
    for index, region in enumerate(entries):
        label = f"region {index + 1}"
        bound = region.get("loras") or []
        if not bound:
            lines.append(f"{label}: no LoRA — holds territory only")
            continue
        for slot, spec in enumerate(bound):
            branches = by_region.get(index, [])
            layers = branches[slot] if slot < len(branches) else {}
            mlp = sum(1 for path in layers if ".mlp." in path)
            lines.append(
                f"{label}: {spec.get('lora_name')} @ {spec.get('strength', 1.0):.2f} — "
                f"{len(layers)} layers ({len(layers) - mlp} attn, {mlp} mlp)"
            )

    blocks = _patches.route_block(config.route_at, total_blocks, config.route_blocks)
    lines.append("")
    if config.tracking:
        lines.append(
            f"tracking: {config.track:.2f} toward the image, {config.refine} rounds, "
            f"smooth {config.smooth}, read at blocks {blocks} of {total_blocks}"
        )
    else:
        reason = "track = 0" if config.track <= 0 else "refine = 0"
        lines.append(f"tracking: OFF ({reason}) — the masks are used exactly as given")
    lines.append(f"schedule: {schedule.describe()}")
    lines.append(
        f"isolation: {config.isolation:.2f}"
        + (" — attention untouched" if config.isolation <= 0 else "")
    )
    return "\n".join(lines)
