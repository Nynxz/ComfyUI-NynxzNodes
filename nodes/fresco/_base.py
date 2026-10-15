"""Shared base for Fresco — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class FrescoNode(NynxzNode):
    """Scene composition."""

    CATEGORY = f"{NynxzNode.CATEGORY}/Fresco"
