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


def modes_saved(root: str) -> list[str]:
    """Every mode that has a folder of saved layouts."""
    base = os.path.join(root, FOLDER)
    if not os.path.isdir(base):
        return []
    return sorted(e.name for e in os.scandir(base) if e.is_dir() and valid_mode(e.name))


def list_all(root: str) -> list[dict]:
    """Every saved layout of every mode, newest first."""
    found = [item for mode in modes_saved(root) for item in list_layouts(root, mode)]
    found.sort(key=lambda item: item["saved"], reverse=True)
    return found


def list_layouts(root: str, mode: str) -> list[dict]:
    """Every layout saved for `mode`, newest first: name, mode, saved time and preview, no
    layout."""
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
                "mode": mode,
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


def rename_layout(root: str, mode: str, name: str, new_name: str) -> str | None:
    """Rename a saved layout and return its new name; None when it's missing, the new name
    can't be a file name, or another layout already has it."""
    old = _file(root, mode, name)
    new = _file(root, mode, new_name)
    if old is None or new is None or not os.path.isfile(old):
        return None
    stem = os.path.basename(new)[: -len(".json")]
    if os.path.exists(new) and os.path.normcase(new) != os.path.normcase(old):
        return None
    data = load_layout(root, mode, name) or {}
    data["name"] = stem
    temp = f"{new}.tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(temp, new)
    if os.path.normcase(new) != os.path.normcase(old):
        os.remove(old)
    return stem
