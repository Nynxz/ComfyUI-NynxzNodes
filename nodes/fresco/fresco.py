"""Fresco and Fresco Apply — the scene editor, and the node that puts it on a model.

**Fresco** holds the scene and encodes it. Each layer that is on becomes one region:

    {"mask", "fit": "stretch", "denoise", "loras", "prompt", "label", "conditioning"?}

`conditioning` is the layer's own reading of the prompt: with reference images, a fusion encode
(the layer's prompt read against them, their vision tokens inside the text stream, several
blended by strength the way Text Encode (Fusion) blends); without, a plain encode of the prompt.
Regions Apply puts it in that region's copy of the prompt, so the references are read only by the
layer's own tokens. A layer with references and no prompt reads the scene prompt against them; a
layer with neither still holds its territory and runs its LoRAs there. The background is the last
region, covering whatever no layer claims.

Its one output, `fresco`, carries the regions, the scene prompt encoded, the canvas size and the
node's own id.

**Fresco Apply** turns that into what a sampler takes: the model with every layer's LoRAs and
prompt routed to its tokens (Regions Apply, with Fresco's defaults), the positive, and a latent —
empty at the canvas size, or, with a source image and a VAE, the image encoded with each layer's
denoise as its noise mask (Region Latent), for the SoftPaint Sampler or Differential Diffusion.
With a source image it also sends the image back to the Fresco node it came from, which shows it
under the canvas.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io, ui

from .._zenkit.io_types import advanced
from ..fusion._fusion import FusionSettings, empty_latent, encode_fusion
from ..fusion.fusion_input import _load_image, _resolve_path
from ..regions._io_types import Regions
from ..regions.apply import NynxzRegionsApply
from ..regions.latent import NynxzRegionLatent
from . import _scene
from ._base import FrescoNode
from ._io_types import Fresco, FrescoSceneType


def _encode_text(clip, prompt: str) -> torch.Tensor:
    """`[1, seq, features]` — a prompt encoded as CLIP Text Encode would."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0]


def _references(refs: list[dict], where: str) -> tuple[list, list, list]:
    """The images, strengths and fits of a layer's references that load."""
    images, strengths, fits = [], [], []
    for ref in refs:
        path = _resolve_path(ref["ref"], ref["type"])
        image = _load_image(path) if path else None
        if image is None:
            logging.warning("Nynxz Fresco: %s skips a missing reference: %s", where, ref["ref"])
            continue
        images.append(image)
        strengths.append(ref["strength"])
        fits.append(ref["fit"])
    return images, strengths, fits


def _conditioning(clip, layer: dict, scene_prompt: str, visual_size: int, where: str):
    """A layer's own conditioning, or None when it has nothing to say."""
    images, strengths, fits = _references(layer["refs"], where)
    prompt = layer["prompt"]
    if images:
        encoded = encode_fusion(
            clip,
            prompt or scene_prompt,
            sources=images,
            settings=FusionSettings(visual_size=visual_size),
            strengths=strengths,
            fits=fits,
        )
        return encoded[0][0]
    if prompt:
        return _encode_text(clip, prompt)
    return None


def _region(clip, layer, mask, scene_prompt, visual_size, where) -> dict:
    region = {
        "mask": mask,
        "fit": "stretch",
        "denoise": layer["denoise"],
        "loras": list(layer["loras"]),
        "label": layer["name"],
        "influence": layer["influence"],
    }
    if layer["prompt"]:
        region["prompt"] = layer["prompt"]
    conditioning = _conditioning(clip, layer, scene_prompt, visual_size, where)
    if conditioning is not None:
        region["conditioning"] = conditioning
        region.setdefault("prompt", scene_prompt)
    return region


class NynxzFresco(FrescoNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Fresco",
            display_name="Fresco",
            description=(
                "Compose a scene as layers on a canvas: draw a box (and paint a shape in it), then "
                "give it a prompt, reference images and LoRAs. Wire it into Fresco Apply."
            ),
            is_experimental=True,
            inputs=[
                FrescoSceneType.Input("scene", default={}),
                io.Clip.Input(
                    "clip",
                    tooltip="The text encoder — Krea 2's. Encodes the scene prompt and every "
                    "layer's prompt and references.",
                ),
            ],
            outputs=[Fresco.Output(display_name="fresco", tooltip="Into Fresco Apply.")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, scene=None, clip=None) -> io.NodeOutput:
        parsed = _scene.parse(scene)
        width, height = parsed["width"], parsed["height"]
        settings = parsed["settings"]
        layers = [layer for layer in parsed["layers"] if layer["on"] and layer["w"] and layer["h"]]
        background = parsed["background"]

        size = _scene.working_size(width, height)
        masks, rest = _scene.compose(layers, size, (height, width), settings["overlap"])
        visual_size = settings["visual_size"]
        regions = []
        for index, (layer, mask) in enumerate(zip(layers, masks, strict=True)):
            if float(mask.max()) <= 0.0:
                logging.warning(
                    "Nynxz Fresco: layer '%s' is entirely covered by the layers above it.",
                    layer["name"],
                )
            regions.append(
                _region(clip, layer, mask, parsed["prompt"], visual_size, f"layer {index + 1}")
            )
        if background["on"] and float(rest.max()) > 0.0:
            regions.append(
                _region(clip, background, rest, parsed["prompt"], visual_size, "background")
            )

        logging.info(
            "Nynxz Fresco: %d region(s) at %dx%d — %s",
            len(regions),
            width,
            height,
            ", ".join(
                f"{r['label']}"
                + (f" +{len(r['loras'])} LoRA" if r["loras"] else "")
                + (" +own conditioning" if r.get("conditioning") is not None else "")
                for r in regions
            )
            or "no layers",
        )
        return io.NodeOutput(
            {
                "regions": regions,
                "positive": clip.encode_from_tokens_scheduled(clip.tokenize(parsed["prompt"])),
                "width": width,
                "height": height,
                "background_denoise": background["denoise"] if background["on"] else 0.0,
                "node": str(cls.hidden.unique_id or ""),
            }
        )


def _active(regions: list[dict]) -> bool:
    """Whether any region gives Regions Apply something to do."""
    return any(r.get("loras") or r.get("conditioning") is not None for r in regions)


class NynxzFrescoApply(FrescoNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Fresco.Apply",
            display_name="Fresco Apply",
            description=(
                "Puts a Fresco scene on the model: each layer's LoRAs and prompt act only where "
                "the layer is. Gives the sampler its model, positive and latent. Wire a source "
                "image and a VAE to edit an image instead of generating one. Krea 2 only."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="A Krea 2 model."),
                Fresco.Input("fresco"),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="A source image to edit. The latent becomes this image, with each "
                    "layer's denoise as how much of it may change — sample it with the SoftPaint "
                    "Sampler, or with Differential Diffusion on the model. Shown under Fresco's "
                    "canvas after a run.",
                ),
                io.Vae.Input("vae", optional=True, tooltip="Encodes the source image."),
                advanced(
                    io.Float.Input(
                        "layout_lock",
                        default=0.0,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="Fraction of the steps whose layout comes from the model without "
                        "the layers' LoRAs (see Regions Apply). 0 is off.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "isolation",
                        default=0.0,
                        min=0.0,
                        max=8.0,
                        step=0.1,
                        tooltip="Keep each layer's tokens from attending the others' (see Regions "
                        "Apply). Costs shared lighting; keep it low.",
                    )
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive", tooltip="The scene prompt."),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Empty at the canvas size, or the source image with its noise mask.",
                ),
                Regions.Output(
                    display_name="regions",
                    tooltip="The layers as regions, for Regions Preview or Region Latent.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, model, fresco, image=None, vae=None, layout_lock=0.0, isolation=0.0
    ) -> io.NodeOutput:
        regions = list(fresco.get("regions") or [])
        if _active(regions):
            # The layers' prompt copies at full weight all run: in Fresco the copy IS the content
            # (prompt and references), and the early steps are where it lands.
            patched = NynxzRegionsApply.execute(
                model,
                regions,
                isolation=isolation,
                layout_lock=layout_lock,
                prompt_follows_schedule=False,
            ).result[0]
        else:
            logging.info(
                "Nynxz Fresco: no layer has a prompt, references or LoRAs — the model is passed "
                "through unchanged."
            )
            patched = model

        width, height = int(fresco["width"]), int(fresco["height"])
        shown = None
        if image is not None:
            if vae is None:
                raise ValueError("Fresco Apply needs the VAE to encode the source image.")
            if not regions:
                raise ValueError(
                    "Editing an image needs at least one layer or the background on — every "
                    "pixel would otherwise keep the source."
                )
            latent = NynxzRegionLatent.execute(
                vae, image, regions, background=float(fresco.get("background_denoise", 0.0))
            ).result[0]
            shown = {
                "fresco_source": ui.ImageSaveHelper.save_images(
                    image[:1], "nynxz.fresco.source", io.FolderType.temp, cls, compress_level=1
                ),
                "fresco_node": [fresco.get("node", "")],
                "fresco_size": [int(image.shape[2]), int(image.shape[1])],
            }
        else:
            latent = empty_latent(width, height, vae)
        return io.NodeOutput(patched, fresco["positive"], latent, regions, ui=shown)
