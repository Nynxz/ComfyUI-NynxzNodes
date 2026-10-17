"""Shared LoRA-stack logic for the loader/apply nodes."""

from __future__ import annotations

import json

import comfy.lora
import comfy.lora_convert
import comfy.sd
import comfy.utils
import folder_paths

from . import _qkv

LOG_PREFIX = "[NynxzNodes]"


def normalize_stack(stack) -> list:
    """Coerce a stack value (list, or JSON string from a widget) into a list of rows."""
    items = stack
    if isinstance(items, str):
        try:
            items = json.loads(items) if items.strip() else []
        except json.JSONDecodeError:
            items = []
    return items if isinstance(items, list) else []


def fit_to_model(model, weights: dict, name: str) -> dict:
    """`weights` with fused q/k/v entries split when `model` has q, k and v separate (see
    _qkv.py); anything else is returned as it came."""
    if model is None:
        return weights
    weights = comfy.lora_convert.convert_lora(weights)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    try:
        fitted, count = _qkv.for_model(weights, key_map)
    except _qkv.CannotSplit as e:
        print(f"{LOG_PREFIX} {name}: fused q/k/v left as is, can't split exactly ({e})")
        return weights
    if count:
        print(f"{LOG_PREFIX} {name}: split {count} fused qkv weights into q, k, v for this model")
    return fitted


def apply_lora_stack(model, clip, stack):
    """Apply each enabled row of `stack` to (model, clip); returns the new (model, clip)."""
    for it in normalize_stack(stack):
        if not isinstance(it, dict) or not it.get("name"):
            continue
        if not it.get("on", True):
            continue

        strength_model = float(it.get("strength", 1.0))
        clip_raw = it.get("clip")
        strength_clip = float(clip_raw) if clip_raw is not None else strength_model
        if strength_model == 0 and strength_clip == 0:
            continue

        path = folder_paths.get_full_path("loras", it["name"])
        if not path:
            print(f"{LOG_PREFIX} LoRA not found, skipping: {it['name']}")
            continue

        weights = fit_to_model(model, comfy.utils.load_torch_file(path, safe_load=True), it["name"])
        model, clip = comfy.sd.load_lora_for_models(
            model, clip, weights, strength_model, strength_clip
        )
    return model, clip
