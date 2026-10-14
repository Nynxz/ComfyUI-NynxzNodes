"""Structure Canvas — a random abstract "draft" to start a turbo sampler from, for seed diversity.

Few-step models repeat their composition across seeds because at σ = 1 they read the start latent
as pure noise and fall back to their favourite layout. Starting from a partially-noised image
instead (denoise ~0.8–0.9) makes the model build on that image's large-scale structure. Using a
full-CFG model's low-res draft as that image works well; this node replaces the draft with a
seeded field of soft color blobs, so the variety costs one VAE encode instead of a second model.

Wire LATENT into SamplerCustomAdvanced's latent_image and set the scheduler's denoise below 1:
  ~0.9  the canvas nudges layout and lighting, fidelity barely changes
  ~0.8  the canvas decides where masses, light and color sit
  lower the blobs start to show through as flat, painterly areas
The IMAGE output is the canvas itself, for previewing what each seed will push toward.

The math is in `_canvas.py`.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import NoiseNode
from ._canvas import structure_canvas


class NynxzStructureCanvas(NoiseNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Noise.StructureCanvas",
            display_name="Structure Canvas",
            description=(
                "A seeded field of soft color blobs, VAE-encoded, to start sampling from at "
                "denoise ~0.8–0.9. Each seed gives the model a different large-scale layout to "
                "build on — the 'low-res draft' trick for turbo models, without a draft model."
            ),
            is_experimental=True,
            inputs=[
                io.Vae.Input("vae"),
                io.Int.Input("width", default=1024, min=64, max=8192, step=16),
                io.Int.Input("height", default=1024, min=64, max=8192, step=16),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Canvas seed. Batch item i uses seed + i.",
                ),
                io.Float.Input(
                    "scale",
                    default=0.35,
                    min=0.02,
                    max=1.0,
                    step=0.01,
                    tooltip="Size of the largest blobs, as a fraction of the short side. Larger = "
                    "broader composition masses.",
                ),
                io.Float.Input(
                    "contrast",
                    default=0.12,
                    min=0.0,
                    max=0.6,
                    step=0.01,
                    tooltip="Light/dark swing around the base brightness — how strongly the canvas "
                    "pushes the lighting layout.",
                ),
                io.Float.Input(
                    "color",
                    default=0.05,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip="Palette strength. 0 = greyscale canvas (layout only, no color push).",
                ),
                io.Float.Input(
                    "brightness",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Base brightness the blobs swing around.",
                ),
                io.Int.Input(
                    "octaves",
                    default=2,
                    min=1,
                    max=6,
                    advanced=True,
                    tooltip="Blob layers; each halves the size of the last.",
                ),
                io.Float.Input(
                    "detail",
                    default=0.4,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Amplitude kept per octave. Lower = only the broad masses.",
                ),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="canvas"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae,
        width,
        height,
        seed,
        scale=0.35,
        contrast=0.12,
        color=0.05,
        brightness=0.5,
        octaves=2,
        detail=0.4,
        batch_size=1,
    ) -> io.NodeOutput:
        canvas = structure_canvas(
            int(width),
            int(height),
            int(seed),
            scale=float(scale),
            octaves=int(octaves),
            detail=float(detail),
            contrast=float(contrast),
            color=float(color),
            brightness=float(brightness),
            batch_size=int(batch_size),
        )
        return io.NodeOutput({"samples": vae.encode(canvas)}, canvas)
