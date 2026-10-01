"""Shared base for the Batch group."""

from __future__ import annotations

from .._base import NynxzNode


class BatchNode(NynxzNode):
    CATEGORY = f"{NynxzNode.CATEGORY}/Batch"
