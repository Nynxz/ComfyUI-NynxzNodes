"""Layout Apply — turns a Layout into everything a sampler takes in one step: the region-patched
model, the encoded positive prompt, the latent (empty, edited source image, or layout start), the
regions (for Regions Preview / Region Inspect), and the compiled prompt string.

Works with Krea 2, Z-Image and MOSS-Image (everything in `regions/_families`), plus any model when
no regional prompts, references or LoRAs are active (passing `model` through unchanged).
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .._sent_back import sent_back
from .._zenkit.io_types import advanced
from ..fresco import _layout
from ..fusion._fusion import PROMPT_MODES, empty_latent
from ..regions._apply import apply_regions
from ..regions._io_types import Regions
from ..regions._settings import RegionSettings
from ..regions.latent import NynxzRegionLatent
from . import _prompt, _regions
from .layout import Layout


def _active(regions: list[dict]) -> bool:
    """Whether any region gives Regions Apply something to do."""
    return any(
        r.get("loras") or r.get("prompt") or r.get("conditioning") is not None for r in regions
    )


def _layout_latent(
    width: int, height: int, layers: list[tuple], background, vae, start: float
) -> dict:
    """The blurred layout image encoded, with a noise mask holding it for the first `start`."""
    image = _layout.compose(width, height, layers, background)
    samples = vae.encode(image)
    mask = torch.full((1, 1, samples.shape[-2], samples.shape[-1]), 1.0 - float(start))
    return {"samples": samples, "noise_mask": mask}


def _with_differential_diffusion(model):
    """Enable core's Differential Diffusion when a graded noise mask is present, unless already set."""
    if model.model_options.get("denoise_mask_function") is not None:
        return model
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

    return DifferentialDiffusion.execute(model).result[0]


class NynxzLayoutApply(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout.Apply",
            display_name="Layout Apply",
            description=(
                "Puts a Layout on the model in one step: each box's prompt, references, mixins and "
                "LoRAs act only where it sits. Outputs the patched model, positive conditioning, "
                "latent (empty, edited photo, or layout start), regions and prompt string."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="A Krea 2, Z-Image or MOSS-Image model (returned patched when regions "
                    "are active). Other models pass through unchanged when no regions are active.",
                ),
                io.Clip.Input(
                    "clip",
                    tooltip="The text encoder: encodes the main prompt, each box's prompt, and "
                    "any reference images (Qwen3-VL / Qwen3.5).",
                ),
                Layout.Input("layout", tooltip="From Layout."),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional source image to edit (overrides Layout Inputs' image). Each "
                    "box's denoise controls how much of it may change.",
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Encodes the source image or layout_start (overrides Layout Inputs' VAE).",
                ),
                io.Boolean.Input(
                    "edit_image",
                    default=True,
                    tooltip="Edit the photo wired into Layout Inputs (or this node). Off: generate "
                    "from scratch, using the photo only for tracing and SAM3.",
                ),
                io.Float.Input(
                    "layout_isolation",
                    default=0.0,
                    min=0.0,
                    max=8.0,
                    step=0.1,
                    tooltip="Keep regions apart while the layout forms (until commit_at, then "
                    "easing to `isolation`). 0 keeps MOSS's fast split attention; try 1.5-2.0 on "
                    "Krea 2 or when regions ignore their prompts.",
                ),
                io.Float.Input(
                    "layout_start",
                    default=0.0,
                    min=0.0,
                    max=0.6,
                    step=0.05,
                    tooltip="Start generation from a blurred picture of the layout instead of pure "
                    "noise (needs VAE). 0.15-0.25 steers placement. Ignored when editing an image.",
                ),
                advanced(
                    io.Float.Input(
                        "lora_warmup",
                        default=0.35,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="LoRA strength before `commit_at` (Regions Apply's preheat). 1 = "
                        "full strength from step 0.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "commit_at",
                        default=0.3,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="Fraction of steps after which LoRAs ramp to full and tracking freezes.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "track",
                        default=0.0,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="How far a region may follow its subject off the drawn box (Krea 2, "
                        "MOSS-Image).",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "release",
                        default=0.15,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="How far a region's LoRAs reach past its edge after commit_at.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "isolation",
                        default=0.0,
                        min=0.0,
                        max=8.0,
                        step=0.1,
                        tooltip="Attention-logit penalty keeping regions apart all run. Keep low.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "layout_lock",
                        default=0.0,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="Fraction of steps taking layout from the base model (Krea 2 only).",
                    )
                ),
                advanced(
                    io.Int.Input(
                        "reference_size",
                        default=0,
                        min=0,
                        max=1024,
                        step=32,
                        tooltip="Visual grid size for reference images. 0 uses the layout's own "
                        "ref detail setting (default 384).",
                    )
                ),
                advanced(
                    io.Combo.Input(
                        "reference_prompt",
                        options=PROMPT_MODES,
                        default=_regions.DEFAULT_PROMPT_MODE,
                        tooltip="How a box's words balance next to its reference images.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "reference_blend",
                        default=0.5,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="For 'blend clean': 0 = with images only, 1 = clean prompt.",
                    )
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(
                    display_name="positive", tooltip="The compiled layout prompt, encoded."
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Empty at canvas size, or the edited photo / layout start with noise mask.",
                ),
                Regions.Output(
                    display_name="regions",
                    tooltip="The boxes as regions, for Regions Preview or Region Inspect.",
                ),
                io.String.Output(
                    display_name="prompt", tooltip="The compiled prompt text that was encoded."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        layout,
        image=None,
        vae=None,
        edit_image=True,
        layout_isolation=0.0,
        layout_start=0.0,
        lora_warmup=0.35,
        commit_at=0.3,
        track=0.0,
        release=0.15,
        isolation=0.0,
        layout_lock=0.0,
        reference_size=0,
        reference_prompt=_regions.DEFAULT_PROMPT_MODE,
        reference_blend=0.5,
    ) -> io.NodeOutput:
        runtime = layout.get("_runtime") if isinstance(layout, dict) else None
        runtime = runtime if isinstance(runtime, dict) else {}
        if image is None and edit_image:
            image = runtime.get("image")
        if vae is None:
            vae = runtime.get("vae")

        parsed = _prompt.parse(layout)
        for warning in _prompt.warnings(layout):
            logging.info("Nynxz Layout (%s): %s", parsed["mode"], warning)

        width, height = parsed["width"], parsed["height"]
        ref_size = (
            int(reference_size) if int(reference_size) >= 128 else parsed["settings"]["visual_size"]
        )
        img_cache: dict = {}
        regions, layout_layers, backdrop_img, _ = _regions.build(
            parsed,
            clip,
            ref_size,
            reference_prompt,
            reference_blend,
            include_backdrop=image is not None,
            img_cache=img_cache,
        )

        prompt_text = _prompt.COMPILERS[parsed["mode"]](parsed)
        positive, scene_img = _regions.encode_scene_positive(
            parsed,
            prompt_text,
            clip,
            ref_size,
            reference_prompt,
            reference_blend,
            img_cache=img_cache,
        )
        if backdrop_img is None:
            backdrop_img = scene_img

        if _active(regions):
            settings = RegionSettings(
                isolation=isolation,
                layout_lock=layout_lock,
                preheat=lora_warmup,
                commit_at=commit_at,
                track=track,
                release=release,
                prompt_follows_schedule=parsed["mode"] != "regions",
                layout_isolation=layout_isolation,
            )
            patched = apply_regions(model, regions, settings, clip=clip, node_name="Layout Apply")
        else:
            logging.info(
                "Nynxz Layout Apply: no box has a prompt, references or LoRAs — passing model through."
            )
            patched = model

        shown = None
        if image is not None:
            if vae is None:
                raise ValueError("Layout Apply needs the VAE to encode the source image.")
            if not regions:
                raise ValueError(
                    "Editing an image needs at least one visible box or the background region on."
                )
            bg_denoise = float(parsed["backdrop"]["denoise"]) if parsed["backdrop"]["on"] else 0.0
            latent = NynxzRegionLatent.execute(vae, image, regions, background=bg_denoise).result[0]
            if runtime.get("node"):
                shown = sent_back("layout", str(runtime["node"]), image, cls, "nynxz.layout.source")
        elif layout_start > 0.0 and vae is not None:
            latent = _layout_latent(width, height, layout_layers, backdrop_img, vae, layout_start)
        else:
            if layout_start > 0.0:
                logging.warning("Nynxz Layout Apply: layout_start needs the VAE wired.")
            latent = empty_latent(width, height, vae)

        if "noise_mask" in latent:
            patched = _with_differential_diffusion(patched)

        return io.NodeOutput(patched, positive, latent, regions, prompt_text, ui=shown)
