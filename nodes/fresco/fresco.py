"""Fresco — the scene editor node. Layers in, regions out.

Each layer that is on becomes one region on the Regions wire:

    {"mask", "fit": "stretch", "denoise", "loras", "prompt", "label", "conditioning"?}

`conditioning` is the layer's own reading of the prompt, made here so Regions Apply needs no CLIP
for it: with reference images it is a fusion encode (the layer's prompt read against them, their
vision tokens inside the text stream, several blended by strength the way Text Encode (Fusion)
blends); without, a plain encode of the layer's prompt. Regions Apply puts it in that region's copy
of the prompt, so the references are read only by the layer's own tokens.

A layer with references and no prompt of its own reads the scene prompt against them. A layer
with neither still holds its territory, and runs its LoRAs there.

The background is the last region, covering whatever no layer claims, and it carries the same
things a layer does.

Outputs, besides the regions: the scene prompt encoded (wire it to the sampler's positive), an
empty latent at the canvas size, and the size itself.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

from .._zenkit.io_types import advanced
from ..fusion._fusion import FusionSettings, empty_latent, encode_fusion
from ..fusion.fusion_input import _load_image, _resolve_path
from ..regions._io_types import Regions
from . import _scene
from ._base import FrescoNode
from ._io_types import FrescoSceneType


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
        settings = FusionSettings(visual_size=visual_size)
        encoded = encode_fusion(
            clip,
            prompt or scene_prompt,
            sources=images,
            settings=settings,
            strengths=strengths,
            fits=fits,
        )
        return encoded[0][0]
    if prompt:
        return _encode_text(clip, prompt)
    return None


def _needs_clip(layer: dict) -> bool:
    return bool(layer["prompt"] or layer["refs"])


class NynxzFresco(FrescoNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Fresco",
            display_name="Fresco",
            description=(
                "Compose a scene as layers on a canvas: draw a box (and paint a mask in it), then "
                "give it a prompt, reference images and LoRAs. Outputs regions for Regions Apply, "
                "so each layer's LoRAs and references act only where it is."
            ),
            is_experimental=True,
            inputs=[
                FrescoSceneType.Input("scene", default={}),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="The text encoder — Krea 2's for Regions Apply. Needed for the scene "
                    "prompt and for any layer with a prompt or references.",
                ),
                advanced(
                    io.Combo.Input(
                        "overlap",
                        options=list(_scene.OVERLAP),
                        default=_scene.DEFAULT_OVERLAP,
                        tooltip="top: a layer covers the layers under it, like paint. share: "
                        "overlapping layers split the overlap between them.",
                    )
                ),
                advanced(
                    io.Int.Input(
                        "visual_size",
                        default=384,
                        min=128,
                        max=1024,
                        step=32,
                        tooltip="Resolution reference images are read at, as Text Encode "
                        "(Fusion)'s visual_size. Higher reads more detail and costs more tokens "
                        "in every layer's copy of the prompt.",
                    )
                ),
            ],
            outputs=[
                Regions.Output(
                    display_name="regions",
                    tooltip="Into Regions Apply (and Region Latent for an edit).",
                ),
                io.Conditioning.Output(
                    display_name="conditioning",
                    tooltip="The scene prompt, encoded — the sampler's positive.",
                ),
                io.Latent.Output(
                    display_name="latent", tooltip="An empty latent at the canvas size."
                ),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(
        cls, scene=None, clip=None, overlap=_scene.DEFAULT_OVERLAP, visual_size=384
    ) -> io.NodeOutput:
        parsed = _scene.parse(scene)
        width, height = parsed["width"], parsed["height"]
        layers = [layer for layer in parsed["layers"] if layer["on"] and layer["w"] and layer["h"]]
        background = parsed["background"]
        wanted = [*layers, background] if background["on"] else layers
        if clip is None and any(_needs_clip(layer) for layer in wanted):
            raise ValueError(
                "Some Fresco layers have a prompt or reference images: wire the text encoder into "
                "Fresco's `clip`."
            )

        size = _scene.working_size(width, height)
        masks, rest = _scene.compose(layers, size, (height, width), overlap)
        regions = []
        for index, (layer, mask) in enumerate(zip(layers, masks, strict=True)):
            if float(mask.max()) <= 0.0:
                logging.warning(
                    "Nynxz Fresco: layer '%s' is entirely covered by the layers above it.",
                    layer["name"],
                )
            regions.append(
                cls._region(clip, layer, mask, parsed["prompt"], visual_size, f"layer {index + 1}")
            )
        if background["on"] and float(rest.max()) > 0.0:
            regions.append(
                cls._region(clip, background, rest, parsed["prompt"], visual_size, "background")
            )
        if not regions:
            raise ValueError(
                "The Fresco scene has no layers that are on. Draw a box on the canvas."
            )

        conditioning = None
        if clip is not None:
            conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(parsed["prompt"]))
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
            ),
        )
        return io.NodeOutput(regions, conditioning, empty_latent(width, height), width, height)

    @staticmethod
    def _region(clip, layer, mask, scene_prompt, visual_size, where) -> dict:
        region = {
            "mask": mask,
            "fit": "stretch",
            "denoise": layer["denoise"],
            "loras": list(layer["loras"]),
            "label": layer["name"],
        }
        if layer["prompt"]:
            region["prompt"] = layer["prompt"]
        if clip is not None:
            conditioning = _conditioning(clip, layer, scene_prompt, visual_size, where)
            if conditioning is not None:
                region["conditioning"] = conditioning
                region.setdefault("prompt", scene_prompt)
        return region
