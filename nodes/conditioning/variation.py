"""Conditioning Variation — a variation seed for a prompt, not for the sampler.

Changing the sampler seed rerolls the whole image. This rerolls the *prompt* instead: the
conditioning is nudged in a seeded random direction, so you get neighbours of the same prompt —
the same composition and the same subject, differently — while the sampler seed stays where it is
and keeps everything else comparable.

The nudge is purely directional. Noise is scaled to each token's own magnitude before it is added
and the result is rescaled back afterwards, so `strength` means the same thing on any text encoder
and the variation changes content rather than activation energy. See `_variation.py` for why both
halves of that matter.

Drop it on any CONDITIONING wire, positive or negative.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import ConditioningNode
from ._variation import vary_conditioning


class NynxzConditioningVariation(ConditioningNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Conditioning.Variation",
            display_name="Conditioning Variation",
            description=(
                "Nudges a conditioning with seeded noise for prompt variations that don't touch "
                "the sampler seed. The nudge is direction-only — each token keeps its original "
                "magnitude — so the variation changes content, not loudness."
            ),
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input(
                    "strength",
                    default=0.1,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="How far to nudge. 0 = unchanged; ~0.1 gives close variations; higher "
                    "drifts further from the prompt.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Variation seed — change it to get a different variation of the same "
                    "prompt.",
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(cls, conditioning, strength=0.1, seed=0) -> io.NodeOutput:
        if not conditioning:
            raise ValueError("Conditioning Variation needs a conditioning wired in.")
        return io.NodeOutput(vary_conditioning(conditioning, float(strength), int(seed)))
