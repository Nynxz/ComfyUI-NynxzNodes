"""Switching a diffusion transformer's blocks off from outside the model.

Core's Skip Layer Guidance asks the model to skip a block through `patches_replace`, which only the
models that read it honour (Flux 1, SD3, Wan). This instead wraps `apply_model` and, for the calls
that ask for it, swaps the chosen blocks' `forward` for one that hands its input straight back. It
needs no support from the model, so Flux 3, Krea 2 and anything else with a list of residual
blocks get it the same way.
"""

from __future__ import annotations

import re
from contextlib import contextmanager

import comfy.patcher_extension
import comfy.samplers

KEY = "nynxz_skip_layers"
STACKS = ("single_blocks", "blocks", "transformer_blocks", "layers")

GUIDANCE = "guidance"
REMOVE = "remove"
MODES = (GUIDANCE, REMOVE)


def parse_blocks(text: str) -> list[int]:
    """`"7, 8, 9"` or `"7-9"` → `[7, 8, 9]`."""
    found = set()
    for start, end in re.findall(r"(\d+)\s*(?:-\s*(\d+))?", text):
        lo, hi = int(start), int(end or start)
        found.update(range(min(lo, hi), max(lo, hi) + 1))
    return sorted(found)


def find_stack(diffusion_model):
    """The model's main list of blocks: the attribute name and the list."""
    for name in STACKS:
        stack = getattr(diffusion_model, name, None)
        if stack is not None and len(stack) > 0:
            return name, stack
    raise ValueError(
        f"Skip Layers: this model has no block list it recognises (looked for {', '.join(STACKS)})."
    )


def _passthrough(x, *args, **kwargs):
    return x


@contextmanager
def skipped(stack, indices):
    """Within the block, the chosen blocks return their input unchanged."""
    blocks = [stack[i] for i in indices if 0 <= i < len(stack)]
    own = [block.__dict__.get("forward") for block in blocks]
    for block in blocks:
        block.forward = _passthrough
    try:
        yield
    finally:
        for block, forward in zip(blocks, own, strict=True):
            if forward is None:
                del block.forward
            else:
                block.forward = forward


def apply_model_wrapper(executor, x, t, *args, **kwargs):
    """Runs the model with the blocks the call's transformer_options name switched off."""
    transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
    wanted = transformer_options.get(KEY)
    if not wanted:
        return executor(x, t, *args, **kwargs)
    indices, sigma_low, sigma_high = wanted
    sigma = float(t.flatten()[0])
    if not sigma_low <= sigma <= sigma_high:
        return executor(x, t, *args, **kwargs)
    _, stack = find_stack(executor.class_obj.diffusion_model)
    with skipped(stack, indices):
        return executor(x, t, *args, **kwargs)


def with_skip(model_options: dict, indices, sigma_low: float, sigma_high: float) -> dict:
    """A copy of `model_options` whose calls skip `indices` between the two sigmas."""
    options = model_options.copy()
    transformer_options = dict(options.get("transformer_options", {}))
    transformer_options[KEY] = (tuple(indices), sigma_low, sigma_high)
    options["transformer_options"] = transformer_options
    return options


def guidance(indices, scale: float, sigma_low: float, sigma_high: float, rescale: float):
    """A post-CFG function: one more pass with the blocks skipped, steered away from.

    The same rule as core's Skip Layer Guidance: result + (cond - cond_skipped) * scale.
    """

    def post_cfg(args):
        result = args["denoised"]
        sigma = float(args["sigma"].flatten()[0])
        if scale == 0 or not sigma_low <= sigma <= sigma_high:
            return result
        options = with_skip(args["model_options"], indices, 0.0, float("inf"))
        (degraded,) = comfy.samplers.calc_cond_batch(
            args["model"], [args["cond"]], args["input"], args["sigma"], options
        )
        guided = result + (args["cond_denoised"] - degraded) * scale
        if rescale:
            factor = args["cond_denoised"].std() / guided.std()
            guided = guided * (rescale * factor + (1 - rescale))
        return guided

    return post_cfg


def install(model, indices, mode: str, scale: float, start: float, end: float, rescale: float):
    """A clone of `model` with the blocks skipped as `mode` says."""
    sampling = model.get_model_object("model_sampling")
    sigma_high = float(sampling.percent_to_sigma(start))
    sigma_low = float(sampling.percent_to_sigma(end))
    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL, KEY, apply_model_wrapper
    )
    if mode == REMOVE:
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        transformer_options[KEY] = (tuple(indices), sigma_low, sigma_high)
    else:
        patched.set_model_sampler_post_cfg_function(
            guidance(indices, scale, sigma_low, sigma_high, rescale)
        )
    return patched
