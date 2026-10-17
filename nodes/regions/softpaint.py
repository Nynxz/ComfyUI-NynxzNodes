"""SoftPaint Sampler — Region Latent's graded mask as a soft tether, instead of Differential Diffusion.

Region Latent's mask says how much of each pixel may be rewritten. Differential Diffusion turns that
into a start time per pixel, so a pixel is either pinned to the source or free, and low values
barely edit while full ones can leave seams (the paper behind this measures both). SoftPaint keeps
every pixel tethered to the source by a strength that follows its mask value and loosens as the
noise falls, and runs a short Langevin loop each step so the model can settle inside that tether.
See `_softpaint` for the method.

A SAMPLER, for SamplerCustom or SamplerCustomAdvanced, so it works with any model and any
scheduler, and with or without Regions Apply on the model. Differential Diffusion is not needed and
is ignored if it's on the model: this sampler reads the mask itself.

    Region Latent ──> latent ──────────────────────────> SamplerCustom
    SoftPaint Sampler ──> sampler ─────────────────────────┘

What a mask value means is a choice, `mask_means`:

  * **edit strength** — the paper's: edit strength rises evenly with the value. A middle value
    already rebuilds most of a pixel, and a background at 0.05 still gets its texture redrawn, so
    values tuned for Differential Diffusion edit far more here.
  * **denoise** — Differential Diffusion's: a pixel is held to the source until the noise level DD
    would start it at, and handed over gradually from there instead of all at once. Region
    Latent's values mean what they meant before; this is the one for region swaps.

Pixels whose mask is exactly 0 — a `protect_mask`, or `background = 0` — come back as the source
latent, as they do with an ordinary inpaint.

Cost: `inner_steps + 1` model calls per step, so 3x at the default.
"""

from __future__ import annotations

import logging

import comfy.samplers
import torch
from comfy_api.latest import io
from tqdm.auto import trange

from .. import _categories
from . import _softpaint
from ._base import RegionNode

EDIT_STRENGTH = "edit strength"
DENOISE = "denoise"
MASK_MEANINGS = (EDIT_STRENGTH, DENOISE)


class SoftPaint(comfy.samplers.Sampler):
    """The SAMPLER object: `sample` is what SamplerCustom calls."""

    def __init__(
        self,
        inner_steps: int,
        step_size: float,
        tether: float,
        curve: float,
        warm: bool,
        as_denoise: bool = False,
    ):
        self.as_denoise = bool(as_denoise)
        self.inner_steps = max(1, int(inner_steps))
        self.step_size = float(step_size)
        self.tether = float(tether)
        self.curve = float(curve)
        self.warm = bool(warm)

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        sampling = model_wrap.inner_model.model_sampling
        signal_of = _softpaint.signal_fn(sampling)
        source = latent_image if latent_image is not None else torch.zeros_like(noise)
        x = sampling.noise_scaling(sigmas[0], noise, source, self.max_denoise(model_wrap, sigmas))
        if denoise_mask is None:
            logging.warning(
                "Nynxz SoftPaint: the latent has no noise mask, so nothing is tethered and this is "
                "plain sampling at %dx the cost. Feed it Region Latent's latent.",
                self.inner_steps + 1,
            )
            redo = torch.ones_like(x)
        else:
            redo = denoise_mask.to(x).clamp(0.0, 1.0).expand_as(x)
        if self.as_denoise:
            start = _softpaint.start_sigmas(
                sampling,
                redo,
                float(sigmas[0]),
                max(float(sigmas[-1]), float(sampling.sigma_min)),
            )
            wander = _softpaint.spread_as_denoise(redo, self.tether, start, signal_of)
        else:
            wander = _softpaint.spread(redo, self.tether, self.curve)

        options = extra_args.get("model_options", {})
        seed = extra_args.get("seed")
        generator = torch.Generator(device="cpu").manual_seed(int(seed or 0) + 7)
        ones = x.new_ones([x.shape[0]])

        def denoise(z, sigma):
            return model_wrap(z, sigma * ones, model_options=options, seed=seed)

        def fresh():
            return torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)

        steps = len(sigmas) - 1
        for i in trange(steps, disable=disable_pbar):
            sigma, nxt = float(sigmas[i]), float(sigmas[i + 1])
            if sigma > 0.0:
                signal = signal_of(sigma)
                beta = _softpaint.pull(sigma, signal, wander)
                gamma = (1.0 + self.tether) * beta
                if self.warm:
                    noised = sampling.noise_scaling(sigmas[i], noise, source, False)
                    x = torch.where(gamma > 1.0, noised, x)
                for _ in range(self.inner_steps):
                    target = gamma * source + (1.0 - gamma) * denoise(x, sigmas[i])
                    x = _softpaint.settle(
                        x, target, sigma, signal, self.tether * beta, self.step_size, fresh()
                    )
            x0 = denoise(x, sigmas[i])
            if callback is not None:
                callback(i, x0, x, steps)
            x = _softpaint.step(x, x0, sigma, nxt, signal_of)

        if latent_image is not None and denoise_mask is not None:
            x = torch.where(redo <= 0.0, latent_image.to(x), x)
        return sampling.inverse_noise_scaling(sigmas[-1], x)


class NynxzSoftPaintSampler(RegionNode):
    CATEGORY = _categories.SAMPLING

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.SoftPaint",
            display_name="SoftPaint Sampler",
            description=(
                "A sampler that reads the latent's graded noise mask (Region Latent's) as a soft "
                "per-pixel tether to the source, instead of Differential Diffusion's start times: "
                "weak values still edit, full ones blend in without a seam. For SamplerCustom. "
                "Costs inner_steps + 1 model calls per step."
            ),
            is_experimental=True,
            inputs=[
                io.Int.Input(
                    "inner_steps",
                    default=2,
                    min=1,
                    max=8,
                    tooltip="Langevin updates per step, one model call each. 2 is the paper's "
                    "sweet spot; past 3 buys little.",
                ),
                io.Float.Input(
                    "step_size",
                    default=0.2,
                    min=0.01,
                    max=2.0,
                    step=0.01,
                    tooltip="How far each update moves, in units of the noise level. 0.2 is the "
                    "paper's; higher settles faster and noisier.",
                ),
                io.Float.Input(
                    "tether",
                    default=5.0,
                    min=0.0,
                    max=20.0,
                    step=0.5,
                    tooltip="How hard low-mask pixels are held to the source, like a guidance "
                    "scale for the source image. Higher keeps structure, lower lets weak mask "
                    "values edit more.",
                ),
                io.Float.Input(
                    "curve",
                    default=2.0,
                    min=0.5,
                    max=8.0,
                    step=0.5,
                    tooltip="How mask values map to freedom. 2 makes the edit roughly linear in "
                    "the mask value; higher makes middle values edit more.",
                ),
                io.Boolean.Input(
                    "warm_start",
                    default=True,
                    tooltip="Start each step's loop from the noised source wherever the tether is "
                    "strongest. Speeds settling; without it two inner steps can leave noise at "
                    "mask edges.",
                ),
                io.Combo.Input(
                    "mask_means",
                    options=list(MASK_MEANINGS),
                    default=EDIT_STRENGTH,
                    optional=True,
                    tooltip="edit strength: the paper's reading — edits rise evenly with the mask, "
                    "so middle values rebuild most of a region and a low background still gets "
                    "redrawn. denoise: Differential Diffusion's reading — each pixel is held to the "
                    "source until the noise level DD would start it at, then handed over gradually. "
                    "Use denoise for region swaps on busy scenes. `curve` applies to edit strength "
                    "only.",
                ),
            ],
            outputs=[io.Sampler.Output(display_name="sampler")],
        )

    @classmethod
    def execute(
        cls,
        inner_steps=2,
        step_size=0.2,
        tether=5.0,
        curve=2.0,
        warm_start=True,
        mask_means=EDIT_STRENGTH,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            SoftPaint(inner_steps, step_size, tether, curve, warm_start, mask_means == DENOISE)
        )
