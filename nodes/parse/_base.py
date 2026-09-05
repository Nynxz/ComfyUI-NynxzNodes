"""Shared base for the parsing nodes — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class ParseNode(NynxzNode):
    """Text and template parsing."""

    CATEGORY = f"{NynxzNode.CATEGORY}/Parse"
