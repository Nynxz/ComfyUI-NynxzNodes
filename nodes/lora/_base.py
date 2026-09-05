"""Shared base for the on-node LoRA stack loaders — the group's menu category, in one place."""

from __future__ import annotations

from .._base import NynxzNode


class LoraNode(NynxzNode):
    """The on-node lora stack loaders."""

    CATEGORY = f"{NynxzNode.CATEGORY}/LoRA"
