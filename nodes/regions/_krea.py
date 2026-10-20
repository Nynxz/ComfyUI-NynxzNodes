"""Regions on Krea 2: patched through its `post_input` and `attn1_patch` hooks (see `_patches`), with
each region's prompt copy made by its LoRA's text side (see `_text`), tracking, and the layout lock.

The model is recognised by `txtfusion`, the 12-layer Qwen3-VL aggregator no other architecture
has — the class name alone is too generic to trust.
"""

from __future__ import annotations

import comfy.patcher_extension

from . import _branch, _lock, _patches, _state, _text
from ._lora import load_branches


def is_krea2(model) -> bool:
    """Whether `model` (a ModelPatcher) wraps Krea 2's DiT."""
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    return diffusion_model is not None and hasattr(diffusion_model, "txtfusion")


def block_count(model) -> int:
    return len(model.model.diffusion_model.blocks)


def already_regioned(model) -> bool:
    get = getattr(model, "get_injections", None)
    return bool(get and get(_branch.INJECTION_KEY))


def install_regions(patched, entries, prompts, settings, sources) -> list[str]:
    """Put the region patches on `patched` (a clone); returns the log lines."""
    loaded, text_loaded, report = load(patched, entries, settings.region_prompt)
    # A region's `ref_latents` and these `sources` are only routed here (see `_refs`): the
    # caller puts the same latents on the positive, in `_refs.latents` order.
    state = _state.GateState(entries, settings.schedule, settings.track, sources)
    state.copy_full = not settings.prompt_follows_schedule
    text = None
    if settings.region_prompt:
        text = _text.TextCopies(
            sorted({index for index, _ in loaded} | set(prompts)), text_loaded, prompts
        )
    patched.set_model_patch(_patches.post_input_patch(state, text), "post_input")
    patched.set_model_patch(_patches.attn1_patch(state, settings), "attn1_patch")
    _branch.attach(patched, loaded, state, text)
    if settings.layout_lock > 0.0:
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _lock.WRAPPER_KEY,
            _lock.LayoutLock(state, settings.layout_lock),
        )
    return summary(report, settings, block_count(patched), text)


def load(patcher, entries, text: bool = True):
    """`([(region_index, branches)], {region_index: {path: [text branch]}}, [report line])` — every
    LoRA on every region, in region order.

    Pairs and not one entry per region, because a region may carry several. Each LoRA's own
    strength is applied when it LOADS rather than to the region's gate: the gate is per REGION and
    several LoRAs share it, so a per-LoRA strength has nowhere else to live.

    The report line is built here, where the name, the strength and the branches that came back
    are all in hand, so a skipped spec can't shift every later line onto the wrong LoRA.
    """
    loaded, text_loaded, report = [], {}, []
    # A LoRA several regions carry at the same strength loads once and its branches are shared, so
    # the hooks compute it once per layer for all of them (see `_branch.by_module`).
    cache: dict[tuple[str, float, bool], tuple[dict, dict]] = {}
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
            key = (name, strength, text)
            if key not in cache:
                cache[key] = load_branches(patcher, name, strength, text)
            branches, text_branches = cache[key]
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


def summary(report, settings, total_blocks: int, text=None) -> list[str]:
    """What will actually run. Deliberately states the mechanism state, not just the settings."""
    lines = list(report)
    blocks = _patches.route_block(settings.route_at, total_blocks, settings.route_blocks)
    lines.append("")
    if settings.tracking:
        lines.append(
            f"tracking: {settings.track:.2f} toward the image, {settings.refine} rounds, "
            f"smooth {settings.smooth}, read at blocks {blocks} of {total_blocks}"
        )
    else:
        reason = "track = 0" if settings.track <= 0 else "refine = 0"
        lines.append(f"tracking: OFF ({reason}) — the masks are used exactly as given")
    lines.append(f"schedule: {settings.schedule.describe()}")
    if settings.layout_lock > 0.0:
        lines.append(
            f"layout lock: the base model's layout for the first "
            f"{min(settings.layout_lock, 1.0):.0%} of steps, each run twice"
        )
    if settings.layout_isolation > settings.isolation:
        lines.append(
            f"layout isolation: {settings.layout_isolation:.2f} until the commit, easing to "
            f"{settings.isolation:.2f}"
        )
    lines.append(
        f"isolation: {settings.isolation:.2f}"
        + (" — attention untouched" if settings.isolation <= 0 and text is None else "")
    )
    if text is None:
        lines.append("region prompt: OFF — the prompt is everyone's, LoRAs' text side not applied")
    else:
        regions = ", ".join(
            f"{r + 1}" + (" (own prompt)" if r in text.prompts else "") for r in text.regions
        )
        lines.append(f"region prompt: a copy of the prompt each for region(s) {regions}")
    return lines
