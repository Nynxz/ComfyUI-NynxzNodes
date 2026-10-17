"""Saved layouts: one JSON file per layout in the user folder, grouped by mode.

    <user>/nynxz_layouts/<mode>/<name>.json  ->  {"name", "mode", "layout", "preview", "saved"}

`layout` is the mode's own value, kept as sent. `preview` is a few boxes the browser draws as a
thumbnail, so listing never has to understand a mode. Names and modes become file names, so both
are checked before they touch the disk.
"""

from __future__ import annotations

import json
import os
import re
import time

FOLDER = "nynxz_layouts"
_MODE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
MAX_NAME = 80
MAX_BYTES = 4 * 1024 * 1024


def valid_mode(mode: object) -> bool:
    return isinstance(mode, str) and bool(_MODE.match(mode))


def clean_name(name: object) -> str | None:
    """`name` as a file stem, or None when it can't be one."""
    if not isinstance(name, str):
        return None
    stem = " ".join(name.split())[:MAX_NAME]
    if not stem or stem in {".", ".."} or stem.startswith(".") or _BAD_NAME.search(stem):
        return None
    return stem


def _folder(root: str, mode: str) -> str:
    return os.path.join(root, FOLDER, mode)


def _file(root: str, mode: str, name: str) -> str | None:
    stem = clean_name(name)
    if not valid_mode(mode) or stem is None:
        return None
    return os.path.join(_folder(root, mode), f"{stem}.json")


def list_layouts(root: str, mode: str) -> list[dict]:
    """Every layout saved for `mode`, newest first: name, saved time and preview, no layout."""
    if not valid_mode(mode):
        return []
    folder = _folder(root, mode)
    if not os.path.isdir(folder):
        return []
    found = []
    for entry in os.scandir(folder):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        try:
            with open(entry.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        found.append(
            {
                "name": entry.name[: -len(".json")],
                "saved": data.get("saved") or entry.stat().st_mtime,
                "preview": data.get("preview"),
            }
        )
    found.sort(key=lambda item: item["saved"], reverse=True)
    return found


def load_layout(root: str, mode: str, name: str) -> dict | None:
    path = _file(root, mode, name)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_layout(root: str, mode: str, name: str, layout: object, preview: object) -> str | None:
    """Write the layout and return the name it was saved under, or None when it can't be."""
    path = _file(root, mode, name)
    if path is None:
        return None
    stem = os.path.basename(path)[: -len(".json")]
    body = json.dumps(
        {"name": stem, "mode": mode, "layout": layout, "preview": preview, "saved": time.time()},
        ensure_ascii=False,
    )
    if len(body.encode("utf-8")) > MAX_BYTES:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as f:
        f.write(body)
    os.replace(temp, path)
    return stem


def delete_layout(root: str, mode: str, name: str) -> bool:
    path = _file(root, mode, name)
    if path is None or not os.path.isfile(path):
        return False
    os.remove(path)
    return True
