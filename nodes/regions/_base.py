"""Shared base for regional conditioning and per-region LoRA — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class RegionNode(NynxzNode):
    """Regional conditioning and per-region lora."""

    CATEGORY = f"{NynxzNode.CATEGORY}/Regions"
