"""Shared base for multi-image fusion conditioning — the group's menu category, in one place."""

from __future__ import annotations

from .. import _categories
from .._base import NynxzNode


class FusionNode(NynxzNode):
    """Multi-image fusion conditioning."""

    CATEGORY = _categories.FUSION
