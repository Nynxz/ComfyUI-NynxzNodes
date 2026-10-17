"""Uncond skip with a strength dial — MOSS-Image's own guidance, made continuous.

MOSS-Image samples with `uncond_skip_layers=[22]`: the negative (uncond) pass runs with block 22
left out, so CFG steers away from a deliberately weakened prediction. That sharpens detail; with it
off the picture looks more real but details can fall apart. Here the chosen blocks' contribution
to the uncond pass is scaled instead of dropped — `strength` 1 is MOSS's skip, 0 is no skip, and
anything between is a dial — and only between two points in sampling, so the last steps can run
plain CFG. The cond pass is never touched, and it costs no extra model call.

ComfyUI batches cond and uncond into one call; `cond_or_uncond` in transformer_options says which
equal chunk of the batch is which (1 = uncond), so the scaling is per batch row.
"""

from __future__ import annotations

from contextlib import contextmanager

from ._skip import find_stack

KEY = "nynxz_uncond_skip"
UNCOND = 1


def row_weights(cond_or_uncond, batch: int, strength: float, like):
    """How much of each batch row's block output to keep: 1 for cond rows, `1 - strength` for
    uncond rows. None when the call has no uncond rows."""
    import torch

    chunks = list(cond_or_uncond or [])
    if UNCOND not in chunks or batch % len(chunks):
        return None
    per = batch // len(chunks)
    keep = [1.0 - strength if kind == UNCOND else 1.0 for kind in chunks for _ in range(per)]
    return torch.tensor(keep, dtype=like.dtype, device=like.device)


def scaled(forward, weights):
    """`forward` with its residual — what the block adds to its input — scaled per batch row.

    Written back into the block's own output, never a new tensor: MOSS's blocks run inside the
    memory compiler region and update the token stream in place, and a tensor allocated inside a
    block is recycled once it ends. The input is copied first because, added to in place, it is
    the very tensor the block returns."""

    def run(x, *args, **kwargs):
        before = x.clone()
        out = forward(x, *args, **kwargs)
        w = weights.to(out.dtype).view(-1, *([1] * (out.dim() - 1)))
        return out.sub_(before).mul_(w).add_(before)

    return run


@contextmanager
def weakened(stack, indices, weights):
    blocks = [stack[i] for i in indices if 0 <= i < len(stack)]
    own = [block.__dict__.get("forward") for block in blocks]
    for block in blocks:
        block.forward = scaled(block.forward, weights)
    try:
        yield
    finally:
        for block, forward in zip(blocks, own, strict=True):
            if forward is None:
                del block.forward
            else:
                block.forward = forward


def apply_model_wrapper(executor, x, t, *args, **kwargs):
    transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
    wanted = transformer_options.get(KEY)
    if not wanted:
        return executor(x, t, *args, **kwargs)
    indices, strength, sigma_low, sigma_high = wanted
    sigma = float(t.flatten()[0])
    if strength == 0 or not sigma_low <= sigma <= sigma_high:
        return executor(x, t, *args, **kwargs)
    weights = row_weights(transformer_options.get("cond_or_uncond"), x.shape[0], strength, x)
    if weights is None:
        return executor(x, t, *args, **kwargs)
    _, stack = find_stack(executor.class_obj.diffusion_model)
    with weakened(stack, indices, weights):
        return executor(x, t, *args, **kwargs)


def install(model, indices, strength: float, start: float, end: float):
    """A clone of `model` whose uncond pass has `indices` weakened by `strength`."""
    import comfy.patcher_extension

    sampling = model.get_model_object("model_sampling")
    sigma_high = float(sampling.percent_to_sigma(start))
    sigma_low = float(sampling.percent_to_sigma(end))
    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL, KEY, apply_model_wrapper
    )
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options[KEY] = (tuple(indices), float(strength), sigma_low, sigma_high)
    return patched
