"""Text Encode Qwen Image Edit (Fusion) — the Fusion Input consumer.

Same engine as the original visual-fusion node (`_fusion.encode_fusion`), but it takes its
sources from a Fusion Input node instead of its own autogrow sockets. That moves per-image
concerns (strength, fit, mute, order) onto the grid where you can see them, and leaves this
node holding only the prompt and the fusion tuning. Chain Fusion Input nodes together to
group more sources into the one input.

Beyond the original: sources carry a per-image strength (relative prevalence in the blend)
and fit mode, and the visual grid's size/aspect is explicit rather than inherited from
whichever image happened to be first.
"""

import base64
from io import BytesIO

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image

from ._base import FusionNode
from ._fusion import (
    ASPECT_OPTIONS,
    DEFAULT_ALPHA_MODE,
    FIT_OVERRIDE,
    FIT_OVERRIDE_OPTIONS,
    FusionSettings,
    alpha_input,
    encode_fusion,
    flatten_alpha,
    seed_input,
    split_alpha,
    style_inputs,
    tuning_inputs,
    variation_inputs,
)
from ._inspect import build_inspect_payload
from ._io_types import FusionInput, FusionInspect


def _source_thumb(image: torch.Tensor, size: int = 72) -> str:
    """A small aspect-preserving JPEG data-uri of a source image, for the inspector legend.

    Base64-embedded in the payload (which already rides the ui channel), so no temp files or extra
    routes. `image` is a [1, H, W, C] float source; empty string on any failure (thumbs are optional).

    Alpha is composited the same way the encode composites it, so the legend shows the source as
    the blend actually saw it rather than whatever RGB was hiding under the mask.
    """
    try:
        flat = flatten_alpha(*split_alpha(image))
        arr = (flat[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr, "RGB")
        pil.thumbnail((size, size), Image.LANCZOS)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001 - a bad thumbnail must never break the encode
        return ""


class NynxzFusionEncode(FusionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Fusion.Encode",
            display_name="Text Encode (Fusion)",
            description="Fuses the images from a Fusion Input node into one Qwen3-VL conditioning, "
            "blending their visual tokens by each image's strength. Images with an alpha channel "
            "only contribute where they are opaque, so a mask removes an element from the blend.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                FusionInput.Input("fusion_input"),
                io.Combo.Input(
                    "visual_aspect",
                    options=ASPECT_OPTIONS,
                    default="auto",
                    tooltip="Aspect of the shared visual grid every source is fitted into. "
                    "'auto' takes it from the first image.",
                ),
                io.Int.Input(
                    "visual_size",
                    default=384,
                    min=128,
                    max=1024,
                    step=32,
                    tooltip="Square-equivalent side length of the visual grid. Higher = more visual "
                    "tokens = finer fusion detail, at more compute.",
                ),
                io.Combo.Input(
                    "fit",
                    options=FIT_OVERRIDE_OPTIONS,
                    default=FIT_OVERRIDE,
                    tooltip="How every source is framed into the grid. 'per image' honours each image's "
                    "own fit (set on the grid card or the Fusion Images row). cover = center-crop "
                    "to fill; contain = fit whole, letterboxed; stretch = distort to fill. 'cover' "
                    "gives the old center-crop framing.",
                ),
                *tuning_inputs(),
                *style_inputs(),
                *variation_inputs(),
                seed_input(),
                alpha_input(),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
                FusionInspect.Output(
                    display_name="fusion_inspect",
                    tooltip="Wire into a Fusion Inspector node for an interactive view of the blend "
                    "field — hover the token grid, view per-source panels, and see which images win "
                    "where plus the settings that produced it.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        fusion_input,
        visual_aspect="auto",
        visual_size=384,
        fit=FIT_OVERRIDE,
        fusion_method="spatial-checkerboard",
        block_size=2,
        dither_ratio=0.5,
        blend_strength=0.5,
        feather=1.0,
        preserve_norm=True,
        content_mode="none",
        content_strength=0.0,
        content_temperature=1.0,
        style_mode="none",
        style_strength=0.0,
        strength_roll=0.0,
        pattern_jitter=0.0,
        jitter_mode="reassign",
        vae=None,
        seed=0,
        alpha_mode=DEFAULT_ALPHA_MODE,
    ) -> io.NodeOutput:
        # One image is fine: the blend is a passthrough, which is what you want when this node
        # is doing style release or standing in for a plain single-reference encode.
        entries = list(fusion_input or [])
        if not entries:
            raise ValueError(
                "Visual fusion needs at least one image — add one to the Fusion Input grid or "
                "wire one into Fusion Images. Muted rows and missing files don't count."
            )

        settings = FusionSettings(
            fusion_method=fusion_method,
            block_size=block_size,
            dither_ratio=dither_ratio,
            blend_strength=blend_strength,
            feather=feather,
            preserve_norm=preserve_norm,
            content_mode=content_mode,
            content_strength=content_strength,
            content_temperature=content_temperature,
            style_mode=style_mode,
            style_strength=style_strength,
            pattern_jitter=pattern_jitter,
            jitter_mode=jitter_mode,
            strength_roll=strength_roll,
            seed=seed,
            visual_aspect=visual_aspect,
            visual_size=visual_size,
            alpha_mode=alpha_mode,
        )
        # "per image" keeps each source's own fit; any other value forces all of them.
        fits = [entry.get("fit", "contain") if fit == FIT_OVERRIDE else fit for entry in entries]
        debug: dict = {}
        conditioning = encode_fusion(
            clip,
            prompt,
            sources=[entry["image"] for entry in entries],
            settings=settings,
            strengths=[entry.get("strength", 1.0) for entry in entries],
            fits=fits,
            vae=vae,
            debug=debug,
        )
        weights = debug.get("weights")
        labels = [entry.get("label", "") for entry in entries]
        if weights is not None:
            grid_h, grid_w = debug["grid"]
            inspect = build_inspect_payload(
                weights,
                grid_h,
                grid_w,
                labels=labels,
                strengths=[entry.get("strength", 1.0) for entry in entries],
                fits=fits,
                thumbs=[_source_thumb(entry["image"]) for entry in entries],
                settings={
                    "fusion_method": settings.fusion_method,
                    "block_size": settings.block_size,
                    "dither_ratio": settings.dither_ratio,
                    "blend_strength": settings.blend_strength,
                    "feather": settings.feather,
                    "content_mode": settings.content_mode,
                    "content_strength": settings.content_strength,
                    "style_mode": settings.style_mode,
                    "style_strength": settings.style_strength,
                    "pattern_jitter": settings.pattern_jitter,
                    "strength_roll": settings.strength_roll,
                    "preserve_norm": settings.preserve_norm,
                    "seed": settings.seed,
                    "visual_size": settings.visual_size,
                    "visual_aspect": settings.visual_aspect,
                    "alpha_mode": settings.alpha_mode,
                    # How many sources actually carry a mask — the inspector dims the alpha row
                    # when it's none, so the knob reads as inert rather than ignored.
                    "alpha_sources": sum(1 for entry in entries if entry["image"].shape[-1] >= 4),
                },
            )
        else:
            inspect = {}
        return io.NodeOutput(conditioning, inspect)
