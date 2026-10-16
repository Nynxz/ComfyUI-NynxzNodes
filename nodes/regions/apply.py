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
    region_prompt    off: the prompt is everyone's, and a LoRA's text side (its `txtfusion`
                     layers, and what it does to the prompt tokens) is not applied at all.
    layout_lock = 0  no second pass. The default. Above 0, the first steps take their layout from
                     the base model (see `_lock`) — try it with preheat 1.

Set them all and this is the plain, obvious implementation — which is exactly why they exist.

The order things happen in one step:

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

import logging

import comfy.patcher_extension
from comfy_api.latest import io

from .._zenkit.io_types import advanced
from . import _branch, _capture, _lock, _patches, _schedule, _state, _text, _zimage
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
                "instead of merging every delta into the weights where they collide, and gives "
                "each region its own prompt when it has one. Composes with ordinary merged "
                "LoRAs. Krea 2 and Z-Image; tracking and layout lock are Krea 2 only for now."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="A Krea 2 or Z-Image model. Returned patched."),
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
                    "Region LoRAs. A region's prompt replaces the main prompt there. Not needed "
                    "otherwise.",
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
        prompt_follows_schedule=True,
        layout_isolation=0.0,
        sources=(),
    ) -> io.NodeOutput:
        zimage = _zimage.is_zimage(model)
        total_blocks = _zimage.block_count(model) if zimage else check_krea2(model, "Regions Apply")
        entries = list(regions or [])
        if not entries:
            raise ValueError("No regions arrived on the wire — there is nothing to gate.")
        # A region may arrive with its prompt already encoded (Fresco does, with its references'
        # vision tokens in it); only the plain prompts are left for `clip`.
        conditioned = {
            i: r["conditioning"] for i, r in enumerate(entries) if r.get("conditioning") is not None
        }
        prompted = {
            i: r["prompt"]
            for i, r in enumerate(entries)
            if r.get("prompt") and i not in conditioned
        }
        if not any(r.get("loras") for r in entries) and not prompted and not conditioned:
            raise ValueError(
                "No region has a LoRA or a prompt, so there is nothing for this node to do. Put a "
                "Region LoRAs node between the regions and this one, and give the regions a LoRA "
                "or a prompt."
            )
        if (prompted or conditioned) and not region_prompt:
            raise ValueError(
                "Some regions have a prompt of their own, which needs region_prompt on — turn it "
                "on, or clear those prompts."
            )
        if prompted and clip is None:
            raise ValueError(
                "Some regions have a prompt of their own: wire the text encoder (the CLIP "
                "you encode the main prompt with) into Regions Apply's `clip`."
            )

        if zimage:
            return _apply_zimage(
                model,
                entries,
                prompted,
                conditioned,
                clip,
                _schedule.Schedule(
                    commit_at=commit_at, preheat=preheat, ramp=ramp, release=release
                ),
                _patches.TrackConfig(
                    track=0.0,
                    refine=refine,
                    route_at=route_at,
                    route_blocks=route_blocks,
                    sharpness=sharpness,
                    smooth=smooth,
                    isolation=isolation,
                    layout_isolation=layout_isolation,
                ),
                copy_full=True,
                track=track,
                layout_lock=layout_lock,
                sources=sources,
            )

        if _already_regioned(model):
            raise ValueError(
                "This model already has regions on it — from Fresco Apply or another Regions "
                "Apply upstream. Applying them twice breaks both (each inserts its own prompt "
                "copies, and the other's gates stop lining up), so use one. Fresco Apply has the "
                "schedule and tracking settings too."
            )
        patched = model.clone()
        loaded, text_loaded, report = _load(patched, entries, region_prompt)

        schedule = _schedule.Schedule(
            commit_at=commit_at, preheat=preheat, ramp=ramp, release=release
        )
        # A region's `ref_latents` and these `sources` are only routed here (see `_refs`): the
        # caller puts the same latents on the positive, in `_refs.latents` order.
        state = _state.GateState(entries, schedule, track, sources)
        # Not a widget: callers that build regions themselves (Fresco Apply) can read the prompt
        # copies at full weight while the LoRAs still follow the schedule.
        state.copy_full = not prompt_follows_schedule
        text = None
        if region_prompt:
            text = _text.TextCopies(
                sorted({index for index, _ in loaded} | set(prompted) | set(conditioned)),
                text_loaded,
                {**{i: _encode(clip, prompt) for i, prompt in prompted.items()}, **conditioned},
            )
        config = _patches.TrackConfig(
            track=track,
            refine=refine,
            route_at=route_at,
            route_blocks=route_blocks,
            sharpness=sharpness,
            smooth=smooth,
            isolation=isolation,
            layout_isolation=layout_isolation,
        )

        patched.set_model_patch(_patches.post_input_patch(state, text), "post_input")
        patched.set_model_patch(_patches.attn1_patch(state, config), "attn1_patch")
        _branch.attach(patched, loaded, state, text)
        if layout_lock > 0.0:
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                _lock.WRAPPER_KEY,
                _lock.LayoutLock(state, layout_lock),
            )
        _capture.clear()

        # To the log rather than to a socket. What loaded and what will run where is worth
        # having when something is wrong and worth nothing the rest of the time, which is exactly
        # a log line and not an output every graph has to route somewhere. Region Inspect shows
        # the bindings before the run and Regions Preview shows the map after it.
        logging.info(
            "Nynxz Regions: %s",
            _summary(report, config, schedule, total_blocks, text, layout_lock),
        )
        return io.NodeOutput(patched)


def _apply_zimage(
    model,
    entries,
    prompted,
    conditioned,
    clip,
    schedule,
    config,
    copy_full,
    track,
    layout_lock,
    sources,
):
    """Regions Apply on Z-Image: each region's own prompt, routed by attention (see `_zimage`)."""
    has_lora = any(r.get("loras") for r in entries)
    if not prompted and not conditioned and not has_lora:
        raise ValueError(
            "On Z-Image, Regions Apply routes each region's own prompt and its LoRAs, and no "
            "region has either. Give the regions a prompt or a LoRA."
        )
    if _zimage.already_regioned(model):
        raise ValueError(
            "This model already has regions on it from another Regions Apply upstream. Use one."
        )
    patched = model.clone()
    state = _zimage.ZGateState(entries, schedule, 0.0, sources)
    # The prompts read at full weight (the schedule is for ramping LoRAs in, and a prompt is not a
    # LoRA); the LoRA branches follow the schedule through the gate. `copy_full` carries this.
    state.copy_full = copy_full
    prompts = dict(conditioned)
    for index, prompt in prompted.items():
        encoded, extra = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0]
        prompts[index] = _zimage.trim(encoded, extra)
    loaded = _zimage.load_region_loras(patched, entries)
    # Every region with a LoRA or a prompt of its own gets a copy of the prompt, so a LoRA's trigger
    # word is read through the LoRA there (see `_zimage`).
    copied = sorted(set(prompts) | {region for region, _ in loaded})
    _zimage.install(patched, state, copied, prompts, config)
    if loaded:
        _branch.attach(patched, loaded, state)
    _capture.clear()

    def _line(i: int) -> str:
        parts = []
        if i in prompts:
            parts.append("own prompt")
        layers = sum(len(b) for region, b in loaded if region == i)
        if layers:
            parts.append(f"{layers} LoRA layers")
        return f"region {i + 1}: " + (", ".join(parts) or "holds territory only")

    lines = [_line(i) for i in range(len(entries))]
    lines.append("")
    lines.append(
        f"Z-Image: {len(copied)} region copy(ies) of the prompt routed by attention"
        + (f", {sum(len(b) for _, b in loaded)} LoRA branch(es)" if loaded else "")
    )
    lines.append(f"schedule: {schedule.describe()}")
    lines.append(f"isolation: {config.isolation:.2f}")
    if track > 0.0:
        lines.append("tracking: not on Z-Image yet — the masks are used exactly as given")
    if layout_lock > 0.0:
        lines.append("layout lock: not on Z-Image yet — ignored")
    logging.info("Nynxz Regions: %s", "\n".join(lines))
    return io.NodeOutput(patched)


def _already_regioned(model) -> bool:
    """Whether `model` already carries this pack's region patches."""
    get = getattr(model, "get_injections", None)
    return bool(get and get(_branch.INJECTION_KEY))


def _encode(clip, prompt: str):
    """`[1, seq, features]` — one region's prompt, encoded as CLIP Text Encode would."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0]


def _load(patcher, entries, text: bool = True):
    """`([(region_index, branches)], {region_index: {path: [text branch]}}, [report line])` — every
    LoRA on every region, in region order.

    Pairs and not one entry per region, because a region may carry several. Each LoRA's own
    strength is applied when it LOADS rather than to the region's gate: the gate is per REGION and
    several LoRAs share it, so a per-LoRA strength has nowhere else to live.

    The report line is built here, where the name, the strength and the branches that came back are
    all in hand. `_summary` used to re-derive it by zipping the loaded list against each region's
    bound specs by position, which silently skewed the moment a spec was skipped for having no
    `lora_name` — every line after it then named the wrong LoRA.
    """
    loaded, text_loaded, report = [], {}, []
    for index, region in enumerate(entries):
        bound = region.get("loras") or []
        if not bound:
            what = "its own prompt" if region.get("prompt") else "holds territory only"
            report.append(f"region {index + 1}: no LoRA — {what}")
            continue
        for spec in bound:
            name = spec.get("lora_name")
            if not name:
                continue
            strength = float(spec.get("strength", 1.0))
            branches, text_branches = load_branches(patcher, name, strength, text)
            loaded.append((index, branches))
            for path, branch in text_branches.items():
                text_loaded.setdefault(index, {}).setdefault(path, []).append(branch)
            mlp = sum(1 for path in branches if ".mlp." in path)
            report.append(
                f"region {index + 1}: {name} @ {strength:.2f} — "
                f"{len(branches)} layers ({len(branches) - mlp} attn, {mlp} mlp)"
                + (f", {len(text_branches)} text" if text_branches else "")
            )
    return loaded, text_loaded, report


def _summary(report, config, schedule, total_blocks, text=None, layout_lock=0.0) -> str:
    """What will actually run. Deliberately states the mechanism state, not just the settings."""
    lines = list(report)

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
    if layout_lock > 0.0:
        lines.append(
            f"layout lock: the base model's layout for the first {min(layout_lock, 1.0):.0%} of "
            "steps, each run twice"
        )
    if config.layout_isolation > config.isolation:
        lines.append(
            f"layout isolation: {config.layout_isolation:.2f} until the commit, easing to "
            f"{config.isolation:.2f}"
        )
    lines.append(
        f"isolation: {config.isolation:.2f}"
        + (" — attention untouched" if config.isolation <= 0 and text is None else "")
    )
    if text is None:
        lines.append("region prompt: OFF — the prompt is everyone's, LoRAs' text side not applied")
    else:
        regions = ", ".join(
            f"{r + 1}" + (" (own prompt)" if r in text.prompts else "") for r in text.regions
        )
        lines.append(f"region prompt: a copy of the prompt each for region(s) {regions}")
    return "\n".join(lines)
