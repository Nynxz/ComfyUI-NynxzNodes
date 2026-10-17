"""Nynxz Save Info — a reusable "where + how to save" hub.

Set the project folder, format and metadata options once on this node's panel, then wire
its ``save_info`` output into as many Save Image nodes as you like. Change the folder here
and every downstream save follows.

The whole config lives in one socketless ``config`` widget (edited by
frontend/widgets/SaveInfoConfig.vue); this node just normalises it and emits the dict over
the ``NYNXZ_SAVE_INFO`` link. Save Image resolves the base directory and writes the files
(config.py is the shared contract).
"""

from __future__ import annotations

import json

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .config import merge_info
from .io_types import NynxzSaveInfoConfigType, NynxzSaveInfoType


class NynxzSaveInfo(NynxzNode):
    """Shared save settings for one or many Save Image nodes."""

    CATEGORY = _categories.OUTPUT
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.IO.SaveInfo",
            display_name="Save Info",
            description=(
                "Shared save settings (folder, format, metadata) on one smart panel. "
                "Wire its output into one or more Save Image nodes."
            ),
            inputs=[
                NynxzSaveInfoConfigType.Input("config", optional=True),
            ],
            outputs=[
                NynxzSaveInfoType.Output(
                    "save_info",
                    display_name="save_info",
                    tooltip="Wire into one or more Save Image nodes.",
                ),
            ],
        )

    @classmethod
    def execute(cls, config=None) -> io.NodeOutput:
        # The panel serialises one object; tolerate a JSON string too. Fill any missing keys
        # from DEFAULTS so downstream Save Images always get the full contract.
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (TypeError, ValueError):
                config = None
        return io.NodeOutput(merge_info(config if isinstance(config, dict) else None))
