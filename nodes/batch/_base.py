"""Shared base for the Batch group."""

from __future__ import annotations

from .. import _categories
from .._base import NynxzNode


class BatchNode(NynxzNode):
    CATEGORY = _categories.UTILS
