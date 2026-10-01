"""Shared save config: the contract between the Save Info hub and Save Image.

`Save Info` emits a plain dict of these keys over the ``NYNXZ_SAVE_INFO`` link;
`Save Image` reads it (falling back to ``DEFAULTS`` when nothing is wired) so a
Save Image works standalone *and* can be driven by a shared hub. Keeping the keys
+ defaults in one place stops the two nodes from drifting apart.
"""

from __future__ import annotations

from typing import Any

from .paths import ROOT_SENTINEL

FORMATS = ["png", "jpg", "webp"]

# Every field a Save Info carries. Save Image merges an incoming dict over these,
# so adding a key here (with a default) is all it takes to extend the contract.
DEFAULTS: dict[str, Any] = {
    "project": ROOT_SENTINEL,  # combo selection (or a sentinel)
    "new_project": "",  # name for a freshly-typed project folder
    "subfolder": "",  # extra subfolder(s) under the project
    "filename_prefix": "",  # shared prefix prepended to each Save Image's name
    "format": "png",
    "quality": 90,
    "embed_metadata": True,  # workflow/prompt in the file (PNG text / EXIF)
    "save_workflow_json": False,  # sidecar <name>.json
    "base_directory": "",  # override; empty = Settings/env/output dir
}


def merge_info(save_info: Any | None) -> dict[str, Any]:
    """Return a full config dict: DEFAULTS overlaid with a wired Save Info dict.

    Unknown keys are ignored and missing keys fall back to defaults, so a partial
    or future-shaped Save Info never breaks Save Image.
    """
    cfg = dict(DEFAULTS)
    if isinstance(save_info, dict):
        for key in DEFAULTS:
            if key in save_info and save_info[key] is not None:
                cfg[key] = save_info[key]
    return cfg
