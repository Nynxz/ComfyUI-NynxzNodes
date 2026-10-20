"""A layout's boxes as regions for Regions Apply: each box's mask, its own prompt, and what the
mixins it names bring — their LoRAs, and their reference images read into its prompt."""

from __future__ import annotations

import logging

from ..fresco import _scene
from . import _prompt


def regions(layout: dict, clip=None, reference_size: int = 384) -> list[dict]:
    """The shown boxes as regions, bottom to top. Where boxes overlap the upper one owns the area,
    as it is drawn in front.

    A box naming mixins with reference images gets its prompt encoded with those images
    (Fusion, as a Fresco layer's references), which needs `clip` to be a text encoder that sees
    images. Their LoRAs ride on the region either way."""
    boxes = _prompt.shown(layout)
    if not boxes:
        return []
    width, height = layout["width"], layout["height"]
    masks, _ = _scene.compose(boxes, _scene.working_size(width, height), (height, width), "top")
    out = []
    for box, mask in zip(boxes, masks, strict=True):
        mixins = _prompt.mixins_of(layout, box)
        label = box["name"] or box["desc"][:40] or box["id"]
        region = {
            "mask": mask,
            "fit": "stretch",
            "loras": [dict(lora) for mixin in mixins for lora in mixin["loras"]],
            "strength": 1.0,
            "denoise": 1.0,
            "label": label,
        }
        own = _prompt.region_prompt(layout, box)
        refs = [ref for mixin in mixins for ref in mixin["refs"]]
        if refs:
            if clip is None:
                names = ", ".join(m["name"] or m["id"] for m in mixins if m["refs"])
                raise ValueError(
                    f'Box "{label}" names {names}, whose reference images are read into its '
                    "prompt by the text encoder. Wire the CLIP (one that can see images, such as "
                    "Qwen3-VL or Qwen3.5) into Layout Extras."
                )
            conditioning = _encode_with_refs(
                clip, own or layout["prompt"], refs, reference_size, label
            )
            if conditioning is not None:
                region["conditioning"] = conditioning
        if own:
            region["prompt"] = own
        elif "conditioning" in region:
            region["prompt"] = layout["prompt"]
        out.append(region)
    return out


def _encode_with_refs(clip, prompt: str, refs: list[dict], size: int, label: str):
    """`prompt` encoded with the reference images, `[1, seq, features]`; None when none load."""
    from ..fusion._fusion import FusionSettings, encode_fusion
    from ..fusion.fusion_input import _load_image, _resolve_path

    images, strengths, fits = [], [], []
    for ref in refs:
        path = _resolve_path(ref["ref"], ref["type"])
        image = _load_image(path) if path else None
        if image is None:
            logging.warning("Nynxz Layout: box %s skips a missing reference: %s", label, ref["ref"])
            continue
        images.append(image)
        strengths.append(ref["strength"])
        fits.append(ref["fit"])
    if not images:
        return None
    try:
        encoded = encode_fusion(
            clip,
            prompt,
            sources=images,
            settings=FusionSettings(visual_size=size),
            strengths=strengths,
            fits=fits,
        )
    except ValueError as error:
        raise ValueError(
            f'Box "{label}" has mixins with reference images, and the wired text encoder '
            f"can't read them: {error}"
        ) from error
    return encoded[0][0]
