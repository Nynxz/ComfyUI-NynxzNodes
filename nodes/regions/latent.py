"""Region Latent — start a remix from the source image, with each region rewritten as much as it
deserves and no more.

The remix problem, stated properly. Swapping two characters in a photo is three different jobs
wearing one denoise slider: the background should barely change, each character should be rebuilt
from scratch, and the pose and framing should survive all of it. One global denoise cannot express
that. Low and the LoRAs cannot overcome the faces already there; high and the composition you were
keeping the image *for* is gone.

So the denoise is per region. Every region carries its own `denoise` — set by Regions from Masks
and overridden by Region Denoise — and this node composites them into one graded mask, using the
same overlap rule the gates use, so the map the LoRA is routed by and the map the pixels are
rewritten by cannot disagree.

**That mask needs Differential Diffusion to mean anything.** Core's node
(`comfy_extras/nodes_differential_diffusion.py`) is what reads a graded mask as a *per-pixel start
time*: a value of 1 is denoised from step 0, 0.4 stays frozen until 60% through the run, 0 never
moves. Without it a mask is binary — inpaint or don't — and every gradation collapses. Wire it on
the model, after Apply::

    Region Latent  ──> latent ──────────────────────────────> KSampler
    Regions Apply  ──> model ──> Differential Diffusion ────> KSampler   (denoise 1.0)

`flatten` is the colour-blob idea. At a denoise around 0.6-0.8 — the band where you want the old
identity gone but the pose kept — the source latent is still largely intact under the noise, and
what it is holding is the *old* character's face. Flattening blends each region toward its own mean
colour, so what survives into the denoise is a correctly-placed, correctly-lit blob of roughly the
right colour with no identity in it. Scaled by each region's own denoise, so a region you are not
rewriting is never touched.

This is the remix path and it wants a source image. Text-to-image needs no latent prior from here —
`Regions Apply` works on an empty latent on its own, with the masks describing where you intend the
characters to be rather than where they already are.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

from . import _mask
from ._base import RegionNode
from ._io_types import Regions


def denoise_field(regions, size: tuple[int, int]):
    """`([H, W] per-pixel denoise, [regions, HW] claim)` — each region's value over its claim.

    A convex combination through the same `share` rule the gates use — a token half-claimed by a
    region at denoise 1 and otherwise background at 0.1 lands at 0.55, not at either end. Using a
    different rule here than the gates use would mean the pixels being rewritten and the LoRA
    rewriting them were routed by two maps that disagree at every boundary.
    """
    masks = _mask.project_all(regions, size)
    claim = _mask.share(masks)
    values = torch.tensor(
        [float(r.get("denoise", 1.0)) for r in regions], dtype=torch.float32
    ).unsqueeze(1)
    field = (claim * values).sum(dim=0)
    return field.reshape(size[0], size[1]), claim


def flatten_latent(
    samples: torch.Tensor, claim: torch.Tensor, regions, amount: float
) -> torch.Tensor:
    """Blend each region toward its own mean colour, by `amount * claim * denoise`.

    Per channel and over the region's own extent, so the result keeps where the subject was and
    what colour it was while losing what it looked like. `amount = 0` returns the samples
    untouched, and that identity is what makes this knob safe to leave alone.
    """
    if amount <= 0.0 or claim.numel() == 0:
        return samples
    height, width = samples.shape[-2], samples.shape[-1]
    out = samples.clone()
    for index, region in enumerate(regions):
        weight = amount * float(region.get("denoise", 1.0))
        if weight <= 0.0:
            continue
        mask = claim[index].reshape(1, 1, height, width).to(samples.device, samples.dtype)
        total = mask.sum()
        if float(total) <= 1e-6:
            continue
        # The region's own mean, per channel: sum(x * m) / sum(m).
        mean = (out * mask).sum(dim=(2, 3), keepdim=True) / total
        blend = (mask * weight).clamp(0.0, 1.0)
        out = out * (1.0 - blend) + mean * blend
    return out


class NynxzRegionLatent(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.Latent",
            display_name="Region Latent",
            description=(
                "Encodes the source image and builds a graded denoise mask from each region's own "
                "denoise, so the background is preserved while the characters are rebuilt. Needs "
                "core's Differential Diffusion on the model to read the gradations."
            ),
            is_experimental=True,
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input(
                    "image",
                    tooltip="The source image being remixed — the one the masks were segmented "
                    "from.",
                ),
                Regions.Input("regions"),
                io.Float.Input(
                    "background",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Denoise for everything no region claims. Low keeps the original "
                    "background, lighting and framing. 0 freezes it exactly, which can leave a "
                    "visible seam where a rebuilt character meets it — a little movement helps "
                    "them agree.",
                ),
                io.Float.Input(
                    "flatten",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Blend each region toward its own mean colour before sampling, so what "
                    "survives the denoise is a correctly-placed blob rather than the old "
                    "character's face. Scaled by each region's denoise. Most useful in the 0.6-0.8 "
                    "denoise band; pointless at 1.0, where the latent is erased anyway.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The encoded source with the graded mask attached as its noise mask.",
                ),
                io.Mask.Output(
                    display_name="denoise_mask",
                    tooltip="The same graded mask, for previewing. Already on the latent — you do "
                    "not need to wire it back through Set Latent Noise Mask.",
                ),
            ],
        )

    @classmethod
    def execute(cls, vae, image, regions, background=0.05, flatten=0.0) -> io.NodeOutput:
        entries = list(regions or [])
        if not entries:
            raise ValueError("No regions arrived on the wire — there is nothing to grade.")

        samples = vae.encode(image[:, :, :, :3])
        height, width = samples.shape[-2], samples.shape[-1]

        # Built on the LATENT grid, where the flatten happens and where the sampler will resize the
        # mask to anyway. One grid, so the mask on the latent and the flatten cannot disagree.
        field, claim = denoise_field(entries, (height, width))
        # Background is a floor, not a competitor: it fills only what no region claimed, so raising
        # a region's denoise never quietly drags the background's up with it.
        residual = _mask.background(claim).reshape(height, width)
        field = field + residual * float(background)

        samples = flatten_latent(samples, claim, entries, float(flatten))

        mask = field.clamp(0.0, 1.0)
        if float(mask.min()) > 0.97:
            logging.warning(
                "Nynxz Regions: every region is at full denoise, so none of the source image "
                "survives and this is a text-to-image run with extra steps. `denoise` defaults to "
                "1.0 on Regions from Masks — lower it there, or per region with Region Denoise."
            )
        latent = {"samples": samples, "noise_mask": mask.reshape(1, 1, height, width)}
        return io.NodeOutput(latent, mask.unsqueeze(0))
