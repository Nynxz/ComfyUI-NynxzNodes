"""A layout's boxes as regions for Regions Apply and Layout Apply: each box's mask (rectangle or
painted/SAM3 shape), its own prompt, and what it and its mixins bring — LoRAs and reference images
read into its prompt, with per-run image and conditioning caching so shared mixins don't reload
from disk or re-encode duplicates."""

from __future__ import annotations

import logging

from ..fresco import _scene
from . import _prompt

#: How a box's prompt is read next to its references (Text Encode (Fusion)'s prompt_mode):
#: a reading with no image mixed in, so the box's own words aren't drowned out by the picture.
DEFAULT_PROMPT_MODE = "blend clean"


def _fusion_kwargs(
    reference_size: int,
    prompt_mode: str,
    prompt_blend: float,
    blend: str = _prompt.DEFAULT_BLEND,
    softness: float = 0.5,
    style_mode: str = _prompt.DEFAULT_STYLE_MODE,
    style_strength: float = 0.0,
) -> dict:
    """Keyword args for `FusionSettings`, matching Fresco's blend modes while keeping Layout's
    `prompt_mode`/`prompt_blend` defaults and optional mixin style release."""
    settings: dict = {
        "visual_size": reference_size,
        "prompt_mode": prompt_mode,
        "prompt_blend": prompt_blend,
    }
    if blend != _prompt.DEFAULT_BLEND or softness != 0.5:
        settings["blend_strength"] = softness
    if blend == "weighted":
        settings["fusion_method"] = "spatial-strength-random"
    elif blend == "tiled":
        settings["fusion_method"] = "spatial-grid"
    elif blend == "salient":
        settings["content_mode"] = "saliency"
        settings["content_strength"] = 0.7
    if style_mode != _prompt.DEFAULT_STYLE_MODE and style_strength > 0.0:
        settings["style_mode"] = style_mode
        settings["style_strength"] = style_strength
    return settings


def _effective_fusion(layout: dict, box: dict) -> tuple[str, float, str, float]:
    """`(blend, softness, style_mode, style_strength)` for `box`.

    Explicit box blend/softness wins; otherwise inherit from the first mixin on the box that
    carries reference images and sets a non-default fusion mode or style release.
    """
    ref_mixins = [m for m in _prompt.mixins_of(layout, box) if m["refs"]]
    blend = box.get("blend", _prompt.DEFAULT_BLEND)
    softness = float(box.get("softness", 0.5))
    box_explicit = blend != _prompt.DEFAULT_BLEND or softness != 0.5
    if not box_explicit:
        for m in ref_mixins:
            m_blend = m.get("blend", _prompt.DEFAULT_BLEND)
            m_soft = float(m.get("softness", 0.5))
            if m_blend != _prompt.DEFAULT_BLEND or m_soft != 0.5:
                blend, softness = m_blend, m_soft
                break
    style_mode = _prompt.DEFAULT_STYLE_MODE
    style_strength = 0.0
    for m in ref_mixins:
        m_mode = m.get("style_mode", _prompt.DEFAULT_STYLE_MODE)
        m_strength = float(m.get("style_strength", 0.0))
        if m_mode != _prompt.DEFAULT_STYLE_MODE and m_strength > 0.0:
            style_mode, style_strength = m_mode, m_strength
            break
    return blend, softness, style_mode, style_strength


def _refs_key(refs: list[dict]) -> tuple:
    return tuple((r["ref"], r["type"], round(r["strength"], 4), r["fit"]) for r in refs)


def build(
    layout: dict,
    clip=None,
    reference_size: int = 384,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    prompt_blend: float = 0.5,
    *,
    include_backdrop: bool = False,
) -> tuple[list[dict], list[tuple], object | None, list[dict]]:
    """Build `(regions, layout_layers, backdrop_image, outlines)` from a parsed layout.

    - `regions`: ready for `apply_regions` / `Regions Apply`.
    - `layout_layers`: `[(mask, first_ref_tensor | None)]` for `layout_start`.
    - `backdrop_image`: first reference tensor on the backdrop, or None.
    - `outlines`: `[{"label", "color", "mask", "part"}]` for `Layout Overlay`.
    """
    boxes = _prompt.shown(layout)
    width, height = layout["width"], layout["height"]
    size = _scene.working_size(width, height)
    overlap = layout.get("settings", {}).get("overlap", _prompt.DEFAULT_OVERLAP)

    # Expand boxes whose painted shape applies to references only (`shape_for == "refs"`) into
    # two parts in one group: the whole box (prompt + LoRAs) and the painted shape (refs).
    expanded: list[dict] = []
    for box in boxes:
        refs = _prompt.effective_refs(layout, box)
        loras = _prompt.effective_loras(layout, box)
        entry = {**box, "_refs": refs, "_loras": loras, "_group": box["id"], "_part": "layer"}
        if box.get("shape_for") == "refs" and box.get("mask") and refs:
            expanded.append({**entry, "mask": None, "_refs": [], "_part": "box"})
            expanded.append(
                {
                    **entry,
                    "name": f"{box['name'] or box['id']} · refs",
                    "_part": "refs",
                }
            )
        else:
            expanded.append(entry)

    masks, rest = _scene.compose(expanded, size, (height, width), overlap)
    img_cache: dict = {}
    cond_cache: dict = {}

    out: list[dict] = []
    layout_layers: list[tuple] = []
    outlines: list[dict] = []

    for box, mask in zip(expanded, masks, strict=True):
        label = box["name"] or box["desc"][:40] or box["id"]
        region: dict = {
            "mask": mask,
            "fit": "stretch",
            "loras": box["_loras"],
            "strength": 1.0,
            "denoise": float(box.get("denoise", 1.0)),
            "label": label,
            "group": box["_group"],
        }
        influence = float(box.get("influence", 1.0))
        if influence != 1.0:
            region["influence"] = influence

        own = _prompt.region_prompt(layout, box)
        refs = box["_refs"]
        first_img = None
        if refs and clip is None:
            mixins = _prompt.mixins_of(layout, box)
            names = ", ".join(m["name"] or m["id"] for m in mixins if m["refs"])
            source = names or "reference images"
            logging.warning(
                'Nynxz Layout: box "%s" uses %s, whose reference images are not read into its '
                "prompt — wire the CLIP (one that sees images, Qwen3-VL or Qwen3.5) into Layout "
                "Extras or Layout Apply for that.",
                label,
                source,
            )
        elif refs:
            blend, softness, style_mode, style_strength = _effective_fusion(layout, box)
            settings = _fusion_kwargs(
                reference_size,
                prompt_mode,
                prompt_blend,
                blend,
                softness,
                style_mode,
                style_strength,
            )
            conditioning, first_img = _encode_with_refs(
                clip,
                own or layout["prompt"],
                refs,
                label,
                settings,
                img_cache,
                cond_cache,
            )
            if conditioning is not None:
                region["conditioning"] = conditioning
        if own:
            region["prompt"] = own
        elif "conditioning" in region:
            region["prompt"] = layout["prompt"]

        out.append(region)
        layout_layers.append((mask, first_img))
        outlines.append(
            {
                "label": label,
                "color": box.get("color", ""),
                "mask": _scene.layer_mask(box, size, (height, width)),
                "part": box["_part"],
            }
        )

    backdrop_image = None
    backdrop = layout.get("backdrop") or {}
    if backdrop.get("on") and float(rest.max()) > 0.0:
        has_bg_content = bool(
            backdrop.get("prompt") or backdrop.get("refs") or backdrop.get("loras")
        )
        if include_backdrop or has_bg_content:
            bg_region: dict = {
                "mask": rest,
                "fit": "stretch",
                "loras": list(backdrop.get("loras") or []),
                "strength": 1.0,
                "denoise": float(backdrop.get("denoise", _prompt.BACKGROUND_DENOISE)),
                "label": "background",
            }
            bg_influence = float(backdrop.get("influence", 1.0))
            if bg_influence != 1.0:
                bg_region["influence"] = bg_influence
            bg_prompt = backdrop.get("prompt") or ""
            bg_refs = backdrop.get("refs") or []
            if bg_refs and clip is not None:
                bg_settings = _fusion_kwargs(
                    reference_size,
                    prompt_mode,
                    prompt_blend,
                    backdrop.get("blend", _prompt.DEFAULT_BLEND),
                    float(backdrop.get("softness", 0.5)),
                )
                conditioning, backdrop_image = _encode_with_refs(
                    clip,
                    bg_prompt or layout["prompt"],
                    bg_refs,
                    "background",
                    bg_settings,
                    img_cache,
                    cond_cache,
                )
                if conditioning is not None:
                    bg_region["conditioning"] = conditioning
            if bg_prompt:
                bg_region["prompt"] = bg_prompt
            elif "conditioning" in bg_region:
                bg_region["prompt"] = layout["prompt"]
            out.append(bg_region)

    return out, layout_layers, backdrop_image, outlines


def regions(
    layout: dict,
    clip=None,
    reference_size: int = 384,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    prompt_blend: float = 0.5,
) -> list[dict]:
    """The shown boxes (and enabled background region) as regions for Regions Apply, bottom to top."""
    built, _, _, _ = build(layout, clip, reference_size, prompt_mode, prompt_blend)
    return built


def _encode_with_refs(
    clip,
    prompt: str,
    refs: list[dict],
    label: str,
    settings: dict,
    img_cache: dict,
    cond_cache: dict,
):
    """`(conditioning [1, seq, features] | None, first_image | None)` with per-run caches."""
    from ..fusion._fusion import FusionSettings, encode_fusion
    from ..fusion._sources import load_refs

    images, strengths, fits = load_refs(refs, f"Layout: box {label}", cache=img_cache)
    if not images:
        return None, None
    key = (prompt, _refs_key(refs), tuple(sorted(settings.items())))
    if key in cond_cache:
        return cond_cache[key], images[0]
    try:
        encoded = encode_fusion(
            clip,
            prompt,
            sources=images,
            settings=FusionSettings(**settings),
            strengths=strengths,
            fits=fits,
        )
    except ValueError as error:
        raise ValueError(
            f'Box "{label}" has reference images, and the wired text encoder '
            f"can't read them: {error}"
        ) from error
    tensor = encoded[0][0]
    cond_cache[key] = tensor
    return tensor, images[0]
