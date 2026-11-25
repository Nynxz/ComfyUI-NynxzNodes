"""`apply_regions`: the model patched so each region's LoRAs and prompt reach only its tokens.

Regions Apply, Fresco Apply and Layout all come through here. What is the same for every model
family — the checks on what arrived, encoding the regions' own prompts, the "already regioned"
guard, the log — is done once; the family's `install` does the rest (see `_families`).
"""

from __future__ import annotations

import logging

from . import _capture
from ._families import family_of
from ._settings import RegionSettings

#: What a setting a family doesn't do yet is called in the log, and what happens instead.
_NOT_YET = {
    "track": ("tracking", "the masks are used exactly as given"),
    "layout_lock": ("layout lock", "ignored"),
}


def apply_regions(
    model,
    regions,
    settings: RegionSettings | None = None,
    clip=None,
    sources=(),
    node_name: str = "Regions Apply",
):
    """A patched clone of `model` (a ModelPatcher). `sources` are reference latents the caller puts
    on the positive before the regions' own (see `_refs`)."""
    settings = settings or RegionSettings()
    family = family_of(model, node_name)
    entries = list(regions or [])
    if not entries:
        raise ValueError("No regions arrived on the wire — there is nothing to gate.")
    # A region may arrive with its prompt already encoded (Fresco does, with its references'
    # vision tokens in it); only the plain prompts are left for `clip`.
    conditioned = {
        i: r["conditioning"] for i, r in enumerate(entries) if r.get("conditioning") is not None
    }
    prompted = {
        i: r["prompt"] for i, r in enumerate(entries) if r.get("prompt") and i not in conditioned
    }
    if not any(r.get("loras") for r in entries) and not prompted and not conditioned:
        raise ValueError(
            "No region has a LoRA or a prompt, so there is nothing for this node to do. Put a "
            "Region LoRAs node between the regions and this one, and give the regions a LoRA "
            "or a prompt."
        )
    if (prompted or conditioned) and not settings.region_prompt:
        raise ValueError(
            "Some regions have a prompt of their own, which needs region_prompt on — turn it "
            "on, or clear those prompts."
        )
    if prompted and clip is None:
        raise ValueError(
            f"Some regions have a prompt of their own: wire the text encoder (the CLIP you encode "
            f"the main prompt with) into {node_name}'s `clip`."
        )
    if family.already_regioned(model):
        raise ValueError(
            "This model already has regions on it — from Fresco Apply or another Regions Apply "
            "upstream. Applying them twice breaks both (each inserts its own prompt copies, and "
            "the other's gates stop lining up), so use one."
        )

    prompts = dict(conditioned)
    for index, prompt in prompted.items():
        prompts[index] = encode(clip, prompt, family.trim_prompts)

    patched = model.clone()
    lines = family.install(patched, entries, prompts, settings, sources)
    for name in family.unsupported:
        if getattr(settings, name) > 0.0:
            what, instead = _NOT_YET[name]
            lines.append(f"{what}: not on {family.name} yet — {instead}")
    _capture.clear()
    # To the log rather than to a socket: what loaded and what will run where is worth having when
    # something is wrong and nothing the rest of the time. Region Inspect shows the bindings before
    # the run and Regions Preview the map after it.
    logging.info("Nynxz Regions: %s", "\n".join(lines))
    return patched


def encode(clip, prompt: str, trim: bool = False):
    """`[1, seq, features]` — one region's prompt, encoded as CLIP Text Encode would; with `trim`,
    without the tokens its attention mask leaves out."""
    encoded, extra = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0]
    if not trim:
        return encoded
    mask = (extra or {}).get("attention_mask")
    if mask is None:
        return encoded
    keep = mask.reshape(-1).bool()
    if keep.shape[0] != encoded.shape[1] or bool(keep.all()):
        return encoded
    return encoded[:, keep.to(encoded.device)]
