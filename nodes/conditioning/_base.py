"""Shared base for conditioning transforms — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class ConditioningNode(NynxzNode):
    """Conditioning transforms."""

    CATEGORY = f"{NynxzNode.CATEGORY}/Conditioning"
