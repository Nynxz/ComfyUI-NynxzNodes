"""Record Steps — the model, patched to keep what every sampling step predicted."""

from __future__ import annotations

import comfy.patcher_extension
from comfy_api.latest import io

from .._base import NynxzNode
from ._record import Recording, Steps, wrapper

WRAPPER_KEY = "nynxz_record_steps"


class NynxzRecordSteps(NynxzNode):
    CATEGORY = f"{NynxzNode.CATEGORY}/Diagnostics"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Steps.Record",
            display_name="Record Steps",
            description=(
                "Keeps what the sampler predicted at every step, for Step Film. Works with any "
                "sampler on this model, KSampler and SamplerCustom alike, and doesn't change what "
                "it samples. One recording per model: a second sampler on the same patched model "
                "replaces the first's, so give each run its own Record Steps."
            ),
            inputs=[io.Model.Input("model")],
            outputs=[
                io.Model.Output(display_name="model"),
                Steps.Output(display_name="steps"),
            ],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        recording = Recording()
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, WRAPPER_KEY, wrapper(recording)
        )
        return io.NodeOutput(patched, recording)
