"""Base-directory resolution and project discovery for the Nynxz Save Image node.

Projects are just immediate subfolders of a base directory. The base resolves,
in priority order, from a per-node override, the "Settings → Nynxz" base-directory setting,
and finally the ComfyUI output directory.

The per-node override comes from the workflow, which may not be the user's own (a shared
workflow, or one embedded in a PNG), so it is confined to ComfyUI's output directory: a
relative value is relative to output/, an absolute one must already be inside it, both
checked by realpath (so `..` and symlinks can't step out). Saving anywhere else is the
user's call alone, through the Settings value.
"""

import json
import os

import folder_paths

# Combo sentinels (kept out of the real project list so they can't collide).
ROOT_SENTINEL = "(output root)"
NEW_SENTINEL = "(+ new project)"

# Must match the id registered on the frontend (src/main.ts settings).
_SETTING_BASE_DIR = "nynxz.SaveImage.BaseDirectory"


def _read_user_setting(key: str, default=None):
    """Best-effort read of a persisted ComfyUI frontend setting (default user).

    Frontend settings live in ``<user>/default/comfy.settings.json`` as flat keys.
    Any failure (missing file, multi-user setup, corrupt json) falls back silently.
    """
    try:
        path = os.path.join(folder_paths.get_user_directory(), "default", "comfy.settings.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(key, default)
    except Exception:  # noqa: BLE001 - no server / module absent outside ComfyUI (tests, import-time checks)
        return default


def is_within(root: str, path: str) -> bool:
    """Whether ``path`` is ``root`` or inside it, both compared by realpath."""
    root, path = os.path.realpath(root), os.path.realpath(path)
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:  # different drives (Windows)
        return False


def confine_to_output(value: str) -> str:
    """The realpath of a per-node base directory, which must lie inside ComfyUI's output
    directory. Relative values are relative to it. Raises ValueError otherwise."""
    output = os.path.realpath(folder_paths.get_output_directory())
    raw = os.path.expanduser(str(value).strip())
    if "\0" in raw:
        raise ValueError("Save Image: the base directory contains a null byte.")
    path = os.path.realpath(os.path.join(output, raw))
    if not is_within(output, path):
        raise ValueError(
            f"Save Image: base directory {value.strip()!r} is outside ComfyUI's output folder "
            f"({output}). Use a folder inside it (relative paths are relative to it), or set "
            f"Settings → Nynxz → base directory to save elsewhere."
        )
    return path


def get_base_directory(override: str = "") -> str:
    """Resolve the base output directory.

    Priority: per-node ``override`` (confined to the output directory, see
    ``confine_to_output``; raises ValueError outside it) →
    Settings → Nynxz base directory → ComfyUI output directory.
    """
    override = (override or "").strip()
    if override:
        return confine_to_output(override)
    setting = (_read_user_setting(_SETTING_BASE_DIR, "") or "").strip()
    if setting:
        return os.path.abspath(os.path.expanduser(setting))
    return folder_paths.get_output_directory()


def list_projects(base: str) -> list[str]:
    """Immediate, non-hidden subfolders of ``base`` — the existing projects."""
    try:
        return sorted(
            name
            for name in os.listdir(base)
            if not name.startswith(".") and os.path.isdir(os.path.join(base, name))
        )
    except Exception:  # noqa: BLE001 - no server / module absent outside ComfyUI (tests, import-time checks)
        return []
