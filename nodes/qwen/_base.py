"""Shared base for Qwen3-VL vision-language nodes — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class QwenNode(NynxzNode):
    """Qwen3-vl vision-language nodes."""

    CATEGORY = f"{NynxzNode.CATEGORY}/Qwen3-VL"
