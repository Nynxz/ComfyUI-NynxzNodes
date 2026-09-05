"""Shared base for the compare-video nodes — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class CompareNode(NynxzNode):
    """Nodes that build side-by-side comparison videos."""

    CATEGORY = f"{NynxzNode.CATEGORY}/Compare"
