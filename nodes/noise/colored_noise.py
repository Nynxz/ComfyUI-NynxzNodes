"""Colored Noise — RandomNoise with a low-frequency tilt, for seed diversity on turbo models.

A drop-in for core's `RandomNoise` on SamplerCustomAdvanced. The seed's white noise is generated
exactly as core does it (same generator, same batch_index handling), then blended toward a
spectrally tilted copy of itself — see `_colored.py`. With strength 0 or color 0 the output is
bit-identical to RandomNoise, so A/B comparisons at one seed are fair.
"""

from __future__ import annotations

import comfy.nested_tensor
import comfy.sample
from comfy_api.latest import io

from ._base import NoiseNode
from ._colored import color_noise


class ColoredNoiseSource:
    """The NOISE object SamplerCustomAdvanced calls: `.seed` + `generate_noise(latent)`."""

    def __init__(self, seed: int, color: float, strength: float, zero_mean: bool):
        self.seed = seed
        self.color = color
        self.strength = strength
        self.zero_mean = zero_mean

    def generate_noise(self, input_latent):
        samples = input_latent["samples"]
        white = comfy.sample.prepare_noise(samples, self.seed, input_latent.get("batch_index"))
        if white.is_nested:
            return comfy.nested_tensor.NestedTensor([self._color(t) for t in white.unbind()])
        return self._color(white)

    def _color(self, white):
        return color_noise(white, self.color, self.strength, self.zero_mean)


class NynxzColoredNoise(NoiseNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Noise.Colored",
            display_name="Colored Noise",
            description=(
                "RandomNoise with its spectrum tilted toward low frequencies, so each seed carries "
                "its own large-scale structure. Aimed at few-step turbo models, whose layout comes "
                "almost entirely from the first step and repeats across white-noise seeds. "
                "Strength 0 is exactly RandomNoise."
            ),
            is_experimental=True,
            inputs=[
                io.Int.Input(
                    "noise_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True
                ),
                io.Float.Input(
                    "color",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    tooltip="Spectral tilt α (power ∝ f^-α): 0 white, 1 pink, 2 red. Higher = "
                    "larger, smoother blobs of structure.",
                ),
                io.Float.Input(
                    "strength",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Blend from the seed's white noise (0) to its tilted version (1). "
                    "The same seed keeps its structure as you raise it.",
                ),
                io.Boolean.Input(
                    "zero_mean",
                    default=True,
                    advanced=True,
                    tooltip="Remove each channel's mean. Off lets a red spectrum's DC offset "
                    "through, which reads as a global color cast.",
                ),
            ],
            outputs=[io.Noise.Output()],
        )

    @classmethod
    def execute(cls, noise_seed, color=1.0, strength=0.5, zero_mean=True) -> io.NodeOutput:
        return io.NodeOutput(ColoredNoiseSource(int(noise_seed), float(color), float(strength), bool(zero_mean)))
