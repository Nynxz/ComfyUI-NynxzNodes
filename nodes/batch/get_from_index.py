"""Get from Index — one item out of a batch of anything."""

from __future__ import annotations

from comfy_api.latest import io

from ._base import BatchNode
from ._index import OUT_OF_RANGE, take


class NynxzGetFromIndex(BatchNode):
    @classmethod
    def define_schema(cls):
        # One template shared by the input and the output, so the output socket reports
        # whatever type was wired in and the graph still type-checks downstream.
        batch = io.MatchType.Template("batch")
        return cls.make_schema(
            node_id="Batch.GetFromIndex",
            display_name="Get from Index",
            description="Takes one item out of a batch of ANY type — images, masks, latents, "
            "audio, conditionings, or any list. Core needs a different node per type; this "
            "works the batch dimension out from the value, and the output socket takes the "
            "type you wired in. Negative indices count from the end, and the result keeps its "
            "batch dimension, so it drops straight back into anything that took the original.",
            inputs=[
                io.MatchType.Input(
                    "batch",
                    template=batch,
                    tooltip="Any batched value: an IMAGE or MASK batch, a LATENT, an AUDIO, a "
                    "CONDITIONING, or a list.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-4096,
                    max=4096,
                    tooltip="Which item, from 0. Negative counts from the end, so -1 is the "
                    "last — the same convention as core's Get Image from Batch.",
                ),
                io.Int.Input(
                    "length",
                    default=1,
                    min=1,
                    max=4096,
                    optional=True,
                    tooltip="How many items to take from `index`. 1 is a single item; more "
                    "returns a smaller batch. Trimmed to what is actually there.",
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=OUT_OF_RANGE,
                    default="clamp",
                    optional=True,
                    tooltip="What an index past the end means. 'clamp' takes the nearest real "
                    "item, matching core's nodes. 'wrap' counts round, for driving this from a "
                    "counter. 'error' stops the run, for when an out-of-range index means "
                    "something upstream is wrong.",
                ),
            ],
            outputs=[
                io.MatchType.Output(template=batch, display_name="item"),
                io.Int.Output(
                    display_name="index",
                    tooltip="The index actually used, after negatives and out_of_range were "
                    "resolved — so a wrapped or clamped index is visible rather than guessed.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many items the incoming batch held.",
                ),
            ],
        )

    @classmethod
    def execute(cls, batch, index=0, length=1, out_of_range="clamp"):
        item, resolved, count = take(batch, index, length, out_of_range)
        return io.NodeOutput(item, resolved, count)
