"""Shared base for noise sources — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class NoiseNode(NynxzNode):
    """Noise sources for SamplerCustomAdvanced."""

    CATEGORY = f"{NynxzNode.CATEGORY}/Noise"
