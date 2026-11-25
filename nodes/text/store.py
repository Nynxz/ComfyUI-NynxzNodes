"""The saved-text library: named pieces of text in one JSON file in ComfyUI's user directory.

Kept with the user's other settings, not in the pack, so it outlives reinstalling the pack.
Process-cached so reads are free; writes go to a temp file then os.replace, so a crash mid-write
cannot truncate the library.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from .._zenkit.identity import LOG_PREFIX, NAMESPACE

try:
    import folder_paths
except ImportError:  # outside ComfyUI (lint, checks)
    folder_paths = None

#: Longest text kept, in characters. A system prompt is a few thousand; this is a guard against
#: a paste of something that was never meant to be one.
MAX_TEXT = 200_000
MAX_NAME = 120

_cache: list[dict] | None = None


def _store_path() -> str:
    base = folder_paths.get_user_directory() if folder_paths else os.path.dirname(__file__)
    return os.path.join(base, NAMESPACE, "saved_text.json")


def _clean(raw: dict) -> dict | None:
    """One stored entry as the library keeps it, or None for something that isn't one."""
    if not isinstance(raw, dict) or not raw.get("id"):
        return None
    updated = raw.get("updated")
    return {
        "id": str(raw["id"]),
        "name": str(raw.get("name") or "Untitled")[:MAX_NAME],
        "category": str(raw.get("category") or "")[:MAX_NAME],
        "text": str(raw.get("text") or "")[:MAX_TEXT],
        "updated": float(updated) if isinstance(updated, (int, float)) else 0.0,
    }


def _read() -> list[dict]:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_store_path(), encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("texts", []) if isinstance(data, dict) else []
        _cache = [e for e in (_clean(r) for r in raw) if e]
    except (OSError, ValueError):
        _cache = []
    return _cache


def _write(items: list[dict]) -> None:
    global _cache
    _cache = items
    path = _store_path()
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"texts": items}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as e:
        print(f"{LOG_PREFIX} could not save the text library: {e}")


def all_texts() -> list[dict]:
    return [dict(e) for e in _read()]


def save(entry: dict) -> dict:
    """Add `entry`, or replace the one with its id; returns it as stored."""
    items = [dict(e) for e in _read()]
    stored = _clean(
        {**entry, "id": entry.get("id") or uuid.uuid4().hex[:12], "updated": time.time()}
    )
    assert stored is not None  # an id was just given
    for i, existing in enumerate(items):
        if existing["id"] == stored["id"]:
            items[i] = stored
            break
    else:
        items.append(stored)
    _write(items)
    return stored


def delete(entry_id: str) -> bool:
    items = _read()
    kept = [dict(e) for e in items if e["id"] != entry_id]
    if len(kept) == len(items):
        return False
    _write(kept)
    return True
