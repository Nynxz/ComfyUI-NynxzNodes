"""Shared base for the SAM3 nodes — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class SAM3Node(NynxzNode):
    """Open-vocabulary detection and segmentation with SAM3."""

    CATEGORY = f"{NynxzNode.CATEGORY}/SAM3"
