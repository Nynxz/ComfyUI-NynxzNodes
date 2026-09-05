"""Shared LoRA-stack logic for the loader/apply nodes."""

from __future__ import annotations

import json

import comfy.sd
import comfy.utils
import folder_paths

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

        weights = comfy.utils.load_torch_file(path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(
            model, clip, weights, strength_model, strength_clip
        )
    return model, clip
