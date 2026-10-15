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
from ..regions import _refs
from ..regions._io_types import Regions
from ..regions.apply import NynxzRegionsApply
from ..regions.latent import NynxzRegionLatent
from . import _layout, _look, _scene, _segment
from ._base import FrescoNode
from ._io_types import Fresco, FrescoInputs, FrescoSceneType


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


def _fusion_settings(layer: dict, visual_size: int) -> FusionSettings:
    """The fusion engine's settings for a layer's blend.

    mix       every reference interleaved cell by cell in equal shares
    weighted  cells scattered by strength, so a stronger reference covers more of the reading
    tiled     each reference one contiguous part of the grid
    salient   interleaved, but each cell leans to whichever reference shows something there
    """
    softness = layer.get("softness", 0.5)
    blend = layer.get("blend", _scene.DEFAULT_BLEND)
    settings = FusionSettings(visual_size=visual_size, blend_strength=softness)
    if blend == "weighted":
        settings.fusion_method = "spatial-strength-random"
    elif blend == "tiled":
        settings.fusion_method = "spatial-grid"
    elif blend == "salient":
        settings.content_mode = "saliency"
        settings.content_strength = 0.7
    return settings


def _conditioning(clip, layer: dict, scene_prompt: str, visual_size: int, where: str, images=None):
    """A layer's own conditioning, or None when it has nothing to say."""
    images, strengths, fits = images or _references(layer["refs"], where)
    prompt = layer["prompt"]
    if images:
        encoded = encode_fusion(
            clip,
            prompt or scene_prompt,
            sources=images,
            settings=_fusion_settings(layer, visual_size),
            strengths=strengths,
            fits=fits,
        )
        return encoded[0][0]
    if prompt:
        return _encode_text(clip, prompt)
    return None


def _region(
    clip, layer, mask, scene_prompt, visual_size, where, photo=None
) -> tuple[dict, torch.Tensor | None]:
    """The layer as a region, and its first reference (for the layout image).

    A layer read by look carries its loaded references as `look` for Fresco Apply to encode, and
    its prompt read with them (and the photo being edited, first) in the edit template."""
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
    loaded = _references(layer["refs"], where)
    if layer.get("read") == "look" and loaded[0]:
        region["look"] = loaded[0]
        shown = ([photo] if photo is not None else []) + loaded[0]
        conditioning = _look.encode_prompt(clip, layer["prompt"] or scene_prompt, shown)
    else:
        conditioning = _conditioning(clip, layer, scene_prompt, visual_size, where, loaded)
    if conditioning is not None:
        region["conditioning"] = conditioning
        region.setdefault("prompt", scene_prompt)
    return region, (loaded[0][0] if loaded[0] else None)


class NynxzFrescoInputs(FrescoNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Fresco.Inputs",
            display_name="Fresco Inputs",
            description=(
                "Everything Fresco works with, in one wire: the text encoder, and optionally a "
                "photo to trace, segment and edit, its VAE, and the SAM3 checkpoint for "
                "segmenting layers."
            ),
            is_experimental=True,
            inputs=[
                io.Clip.Input("clip", tooltip="The text encoder — Krea 2's."),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="A photo to work on: shown under the canvas after a run, segmented "
                    "by SAM3 for layers that ask, and edited by Fresco Apply.",
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Carried to Fresco Apply, to encode the photo or the layout start.",
                ),
                io.Combo.Input(
                    "sam3",
                    options=_segment.checkpoints(),
                    default=_segment.NONE,
                    tooltip="The SAM3 checkpoint, for layers that capture something by text or "
                    "click points. Loaded only when a layer has something new to segment.",
                ),
            ],
            outputs=[FrescoInputs.Output(display_name="inputs", tooltip="Into Fresco.")],
        )

    @classmethod
    def execute(cls, clip, image=None, vae=None, sam3=_segment.NONE) -> io.NodeOutput:
        return io.NodeOutput(
            {
                "clip": clip,
                "image": image,
                "vae": vae,
                "sam3": None if sam3 == _segment.NONE else sam3,
            }
        )


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
                io.MultiType.Input(
                    "inputs",
                    types=[io.Clip, FrescoInputs],
                    tooltip="The text encoder (Krea 2's), or Fresco Inputs to bring a photo, its "
                    "VAE and SAM3 along with it.",
                ),
            ],
            outputs=[Fresco.Output(display_name="fresco", tooltip="Into Fresco Apply.")],
            hidden=[io.Hidden.unique_id],
            # An output node, so the editor's Segment button can run Fresco on its own.
            is_output_node=True,
        )

    @classmethod
    def execute(cls, scene=None, inputs=None) -> io.NodeOutput:
        sources = inputs if isinstance(inputs, dict) else {"clip": inputs}
        clip, image = sources.get("clip"), sources.get("image")
        parsed = _scene.parse(scene)
        width, height = parsed["width"], parsed["height"]
        settings = parsed["settings"]
        on = [layer for layer in parsed["layers"] if layer["on"] and layer["w"] and layer["h"]]
        segmented = cls._segment(on, image, sources.get("sam3"))
        layers = _scene.parts(on)
        background = parsed["background"]

        size = _scene.working_size(width, height)
        masks, rest = _scene.compose(layers, size, (height, width), settings["overlap"])
        visual_size = settings["visual_size"]
        regions, layout, outlines = [], [], []
        for index, (layer, mask) in enumerate(zip(layers, masks, strict=True)):
            if float(mask.max()) <= 0.0:
                logging.warning(
                    "Nynxz Fresco: layer '%s' is entirely covered by the layers above it.",
                    layer["name"],
                )
            region, first = _region(
                clip, layer, mask, parsed["prompt"], visual_size, f"layer {index + 1}", image
            )
            region["group"] = layer["group"]
            regions.append(region)
            layout.append((mask, first))
            outlines.append(
                {
                    "label": layer["name"],
                    "color": layer.get("color", ""),
                    "mask": _scene.layer_mask(layer, size, (height, width)),
                    "part": layer["part"],
                }
            )
        backdrop = None
        if background["on"] and float(rest.max()) > 0.0:
            region, backdrop = _region(
                clip, background, rest, parsed["prompt"], visual_size, "background", image
            )
            regions.append(region)

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
        node = str(cls.hidden.unique_id or "")
        look_photo = settings["photo"] == "look" and image is not None
        if look_photo:
            positive = _look.encode(clip, parsed["prompt"], [image])
            negative = _look.encode(clip, "", [image])
        else:
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(parsed["prompt"]))
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        return io.NodeOutput(
            {
                "regions": regions,
                "positive": positive,
                "negative": negative,
                "look_photo": look_photo,
                "width": width,
                "height": height,
                "background_denoise": background["denoise"] if background["on"] else 0.0,
                "node": node,
                "layout": {"layers": layout, "background": backdrop},
                "outlines": outlines,
                "image": image,
                "vae": sources.get("vae"),
            },
            ui=cls._report(node, image, segmented),
        )

    @staticmethod
    def _segment(layers: list[dict], image, checkpoint: str | None) -> list[dict]:
        """Segment every layer that asks and whose shape didn't already come from the same
        request; its mask is replaced for this run. SAM3 is loaded only if one does. Returns what
        changed, for the editor."""
        if image is None:
            stale = []
        else:
            stale = [(layer, src) for layer in layers if (src := _segment.stale(layer, image))]
        if not stale:
            return []
        if checkpoint is None:
            logging.warning(
                "Nynxz Fresco: some layers capture something — pick the SAM3 checkpoint on "
                "Fresco Inputs."
            )
            return []
        model, sam_clip = _segment.load(checkpoint)
        changed = []
        for layer, source in stale:
            mask = _segment.segment(model, sam_clip, image, layer)
            if mask is None:
                continue
            layer["mask"] = _segment.box_local(mask, layer)
            layer["mask_source"] = source
            changed.append({"id": layer["id"], "mask": layer["mask"], "source": source})
        return changed

    @classmethod
    def _report(cls, node: str, image, segmented: list[dict]) -> dict | None:
        """What goes back to the editor: the photo, as the trace image, and new masks."""
        if image is None:
            return None
        return {
            "fresco_node": [node],
            "fresco_source": ui.ImageSaveHelper.save_images(
                image[:1], "nynxz.fresco.source", io.FolderType.temp, cls, compress_level=1
            ),
            "fresco_size": [int(image.shape[2]), int(image.shape[1])],
            "fresco_masks": segmented,
        }


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
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Encodes the source image or the layout start. Not needed when Fresco "
                    "Inputs carries one.",
                ),
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
                        "lora_warmup",
                        default=0.35,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="LoRA strength before `commit_at` (Regions Apply's preheat). 1 = "
                        "full strength from the first step; lower keeps the LoRAs from pushing "
                        "the composition around while it forms.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "commit_at",
                        default=0.3,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="Fraction of the steps after which the LoRAs ramp to full and the "
                        "layer map stops moving.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "track",
                        default=0.0,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="How far a layer may follow its subject off the drawn shape. 0 "
                        "(default) keeps it exactly where you drew it — in Fresco the boxes are the "
                        "layout. Raise it for an edit whose subjects don't sit exactly in the boxes.",
                    )
                ),
                advanced(
                    io.Float.Input(
                        "release",
                        default=0.15,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        tooltip="How far a layer's LoRAs reach past its edge after the commit, so "
                        "subjects don't look cut out. 0 keeps the edge hard.",
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
                io.Float.Input(
                    "layout_isolation",
                    default=2.0,
                    min=0.0,
                    max=8.0,
                    step=0.1,
                    tooltip="How strongly the layers are kept apart while the layout forms, so "
                    "each one follows its own prompt instead of the picture averaging them. In "
                    "force until commit_at, then eases to `isolation` so lighting and edges can "
                    "blend. 0 = off. Raise it if layers ignore their prompts; lower it if the "
                    "result looks like a collage.",
                ),
                io.Float.Input(
                    "layout_start",
                    default=0.0,
                    min=0.0,
                    max=0.6,
                    step=0.05,
                    tooltip="Start from a blurred picture of the layout instead of pure noise, so "
                    "the image follows the boxes: each layer's first reference in its box (or a "
                    "soft shape), over the background's. The share of the run that holds to it — "
                    "0.15-0.25 steers placement, higher starts to keep its colours. Needs the VAE. "
                    "0 = off. Ignored when editing an image.",
                ),
                io.Boolean.Input(
                    "edit_image",
                    default=True,
                    tooltip="Edit the photo wired into Fresco (or into this node). Off: generate "
                    "a new image, using Fresco's photo only to trace and segment over.",
                ),
                io.Boolean.Input(
                    "look_source",
                    default=True,
                    tooltip="With layers that read their references by look, also show the model "
                    "the photo being edited as clean tokens everyone reads — the source half of "
                    "the in-context edit recipe (e.g. the Krea 2 Identity Edit LoRA). Only when "
                    "editing a photo.",
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
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="The empty prompt — read with the photo, carrying the same reference "
                    "tokens, when the photo is read by look (the edit recipe's unconditional). "
                    "For CFG above 1.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        fresco,
        image=None,
        vae=None,
        layout_lock=0.0,
        isolation=0.0,
        lora_warmup=0.35,
        layout_isolation=2.0,
        commit_at=0.3,
        track=0.0,
        release=0.15,
        layout_start=0.0,
        edit_image=True,
        look_source=True,
    ) -> io.NodeOutput:
        if image is None and edit_image:
            image = fresco.get("image")
        vae = vae if vae is not None else fresco.get("vae")
        regions = [dict(r) for r in fresco.get("regions") or []]
        sources = _look_latents(regions, vae, image, fresco, look_source)
        if _active(regions):
            # The layers' prompt copies at full weight all run: in Fresco the copy IS the content
            # (prompt and references), and the early steps are where it lands.
            patched = NynxzRegionsApply.execute(
                model,
                regions,
                isolation=isolation,
                layout_lock=layout_lock,
                preheat=lora_warmup,
                commit_at=commit_at,
                track=track,
                release=release,
                prompt_follows_schedule=False,
                layout_isolation=layout_isolation,
                sources=sources,
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
        elif layout_start > 0.0 and vae is not None:
            latent = _layout_latent(fresco, vae, width, height, layout_start)
        else:
            if layout_start > 0.0:
                logging.warning("Nynxz Fresco: layout_start needs the VAE wired into Fresco Apply.")
            latent = empty_latent(width, height, vae)
        if "noise_mask" in latent:
            patched = _with_differential_diffusion(patched)
        refs = _refs.latents(regions, sources)
        positive = _refs.attach(fresco["positive"], refs)
        negative = _refs.attach(fresco.get("negative") or [], refs)
        return io.NodeOutput(patched, positive, latent, regions, negative, ui=shown)


def _look_latents(regions: list[dict], vae, image, fresco: dict, with_source: bool) -> list:
    """Encode the references of layers read by look onto the output's grid, as `ref_latents` on
    their regions; returns the source latents (the photo being edited) to go before them — when
    the scene reads the photo by look, or layers read by look and `with_source` is on."""
    looking = [r for r in regions if r.get("look")]
    whole = bool(fresco.get("look_photo")) and image is not None
    if not looking and not whole:
        return []
    if vae is None:
        logging.warning(
            "Nynxz Fresco: reading by look needs the VAE (on Fresco Inputs or Fresco Apply) — "
            "without it only the prompts, read with the images, reach the model.",
        )
        return []
    if image is not None:
        width, height = int(image.shape[2]), int(image.shape[1])
    else:
        width, height = int(fresco["width"]), int(fresco["height"])
    sources = (
        [_look.encode_latent(vae, image, width, height)]
        if image is not None and (whole or with_source)
        else []
    )
    for region in looking:
        region["ref_latents"] = [_look.encode_latent(vae, i, width, height) for i in region["look"]]
        region["ref_frame"] = len(sources) + 1
    logging.info(
        "Nynxz Fresco: by look — %sreferences of %d layer(s) (%d latent(s)).",
        "the photo as source tokens, " if sources else "",
        len(looking),
        sum(len(r["ref_latents"]) for r in looking),
    )
    return sources


def _layout_latent(fresco: dict, vae, width: int, height: int, start: float) -> dict:
    """The layout image encoded, with a noise mask that holds it for the first `start` of the run."""
    layout = fresco.get("layout") or {}
    image = _layout.compose(width, height, layout.get("layers") or [], layout.get("background"))
    samples = vae.encode(image)
    mask = torch.full((1, 1, samples.shape[-2], samples.shape[-1]), 1.0 - float(start))
    return {"samples": samples, "noise_mask": mask}


def _with_differential_diffusion(model):
    """`model` reading a graded noise mask as per-pixel start times (core's Differential Diffusion),
    unless something already set a mask function. The SoftPaint Sampler ignores it and reads the
    mask itself."""
    if model.model_options.get("denoise_mask_function") is not None:
        return model
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

    return DifferentialDiffusion.execute(model).result[0]
