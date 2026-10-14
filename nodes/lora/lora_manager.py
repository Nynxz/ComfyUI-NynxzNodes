"""Read-only access to ComfyUI-Lora-Manager's downloaded example images.

Lora Manager stores them under its `example_images_path` setting as
`<root>[/<library>]/<sha256>/image_<i>.<ext>` (`i` indexes `civitai.images`) and
`custom_<id>.<ext>` for user imports (`civitai.customImages`). Mirrors
py/utils/settings_paths.py and py/utils/example_images_paths.py upstream.
"""

from __future__ import annotations

import json
import os
import re
import sys

APP_NAME = "ComfyUI-LoRA-Manager"
MEDIA_EXTS = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".gif": "image",
    ".avif": "image",
    ".jxl": "image",
    ".mp4": "video",
    ".webm": "video",
}
_HASH = re.compile(r"[a-f0-9]{64}")
_FILE = re.compile(r"(image_\d+|custom_[A-Za-z0-9_-]+)\.[a-z0-9]+")

_settings_cache: tuple[str, float, dict] | None = None


def _user_config_dir() -> str:
    """platformdirs.user_config_dir(APP_NAME, appauthor=False) at its default location."""
    if sys.platform == "win32":
        base = os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.path.expanduser("~/.config")
    return os.path.join(base, APP_NAME)


def _plugin_dirs() -> list[str]:
    """Lora Manager checkouts beside this pack, for its portable settings.json."""
    custom_nodes = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    try:
        names = os.listdir(custom_nodes)
    except OSError:
        return []
    return [os.path.join(custom_nodes, n) for n in names if "lora-manager" in n.lower()]


def _read(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _settings_path() -> str | None:
    for plugin in _plugin_dirs():
        path = os.path.join(plugin, "settings.json")
        data = _read(path)
        if data is None:
            continue
        if data.get("use_portable_settings"):
            return path
    path = os.path.join(_user_config_dir(), "settings.json")
    return path if os.path.isfile(path) else None


def settings() -> dict:
    """Lora Manager's settings.json, re-read when it changes; {} when not installed."""
    global _settings_cache
    path = _settings_path()
    if not path:
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    if _settings_cache and _settings_cache[0] == path and _settings_cache[1] == mtime:
        return _settings_cache[2]
    data = _read(path) or {}
    _settings_cache = (path, mtime, data)
    return data


def example_folder(sha256: str | None) -> str | None:
    """The folder holding a model's downloaded examples, if it exists."""
    root = settings().get("example_images_path")
    sha = (sha256 or "").lower()
    if not root or not _HASH.fullmatch(sha):
        return None
    root = os.path.abspath(os.path.expanduser(str(root)))
    direct = os.path.join(root, sha)
    if os.path.isdir(direct):
        return direct
    try:
        libraries = [d for d in os.listdir(root) if not _HASH.fullmatch(d)]
    except OSError:
        return None
    for lib in libraries:
        nested = os.path.join(root, lib, sha)
        if os.path.isdir(nested):
            return nested
    return None


def local_files(folder: str | None) -> dict[str, str]:
    """Media files in an example folder, keyed by stem (`image_3`, `custom_ab12`)."""
    if not folder:
        return {}
    try:
        names = os.listdir(folder)
    except OSError:
        return {}
    out = {}
    for n in sorted(names):
        stem, ext = os.path.splitext(n)
        if ext.lower() in MEDIA_EXTS:
            out.setdefault(stem, n)
    return out


def resolve(sha256: str | None, file: str) -> str | None:
    """Absolute path of one example file, or None. `file` must be a bare example name."""
    if not _FILE.fullmatch(file or "") or os.path.splitext(file)[1] not in MEDIA_EXTS:
        return None
    folder = example_folder(sha256)
    if not folder:
        return None
    path = os.path.join(folder, file)
    return path if os.path.isfile(path) else None
