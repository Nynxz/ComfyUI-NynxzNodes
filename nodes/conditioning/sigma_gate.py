"""Conditioning Sigma Gate — restrict a conditioning to part of the denoise, in real sigma.

A composable control primitive: drop it on any CONDITIONING wire (a CLIP encode, a Fusion
encode, or anything upstream) to make that conditioning active only within a slice of the
sampling range. The window can be given in denoise percent (0..1, no model needed) or in real
sigma. Conditioning can only store a percent window, so sigma is converted to percent using the
model's own schedule (exact, wire MODEL) or a SIGMAS schedule as a lookup table (approximate).

The gate intersects with any range already on the conditioning, so it stacks on top of an
already-scheduled conditioning instead of overwriting its schedule.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import ConditioningNode
from ._sigma import gate, sigma_to_percent_model, sigma_to_percent_sigmas

MODE_PERCENT = "denoise percent"
MODE_SIGMA = "sigma"


class NynxzConditioningSigmaGate(ConditioningNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Conditioning.SigmaGate",
            display_name="Conditioning Sigma Gate",
            description="Restricts a conditioning to a slice of the denoise so it only applies "
            "for part of sampling. Set the window in denoise percent (0=first step, 1=last) or in "
            "real sigma — sigma needs a MODEL wired for an exact mapping, or a SIGMAS schedule for "
            "an approximate one. Intersects any existing range, so it stacks on other gates.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Combo.Input(
                    "mode",
                    options=[MODE_PERCENT, MODE_SIGMA],
                    default=MODE_PERCENT,
                    tooltip="Interpret start/end as a denoise fraction (0..1) or as sigma values.",
                ),
                io.Float.Input(
                    "start",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip="Where the gate opens. Percent mode: fraction of the denoise (0=first "
                    "step). Sigma mode: the high sigma it becomes active at (early in sampling).",
                ),
                io.Float.Input(
                    "end",
                    default=1.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip="Where the gate closes. Percent mode: fraction of the denoise (1=last "
                    "step). Sigma mode: the low sigma it stays active down to (late in sampling).",
                ),
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip="Optional. In sigma mode, converts sigma → percent exactly using the "
                    "model's own schedule.",
                ),
                io.Sigmas.Input(
                    "sigmas",
                    optional=True,
                    tooltip="Optional. In sigma mode with no MODEL, used as an approximate sigma → "
                    "percent lookup (exact for normal schedulers, soft for Karras).",
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, conditioning, mode, start, end, model=None, sigmas=None) -> io.NodeOutput:
        if not conditioning:
            raise ValueError("Conditioning Sigma Gate needs a conditioning wired in.")

        if mode == MODE_SIGMA:
            if model is not None:
                model_sampling = model.get_model_object("model_sampling")

                def to_percent(sigma):
                    return sigma_to_percent_model(model_sampling.percent_to_sigma, sigma)
            elif sigmas is not None:
                schedule = sigmas.tolist() if hasattr(sigmas, "tolist") else list(sigmas)

                def to_percent(sigma):
                    return sigma_to_percent_sigmas(schedule, sigma)
            else:
                raise ValueError(
                    "Sigma mode needs a MODEL (for an exact sigma→percent mapping) or a SIGMAS "
                    "schedule (approximate) wired in. Switch to 'denoise percent' mode otherwise."
                )
            start_percent = to_percent(float(start))
            end_percent = to_percent(float(end))
        else:
            start_percent, end_percent = float(start), float(end)

        return io.NodeOutput(gate(conditioning, start_percent, end_percent))
