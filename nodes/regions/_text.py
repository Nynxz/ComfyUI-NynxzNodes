"""Per-region copies of the prompt, so a LoRA's text side can be regional too.

Krea 2 turns the prompt into tokens with `txtfusion`, a small transformer over the Qwen3-VL layers,
and every image token reads those same tokens. A LoRA trained on Krea 2 changes two things there
that the image-side branches never reach: `txtfusion`'s own weights, and what the 28 blocks do to
the text tokens. Neither has a place on the canvas, so neither can be gated by a mask.

What can be done instead is to give each region its own prompt. For every region with a LoRA, the
prompt goes through `txtfusion` again with that region's text-side branches live, and the result
joins the sequence as a copy of the text. In the blocks the copy's tokens carry that region's LoRA
like canvas tokens in the region do. Attention then sends each canvas token to the copy of the
region that claims it, and to the plain prompt for whatever no LoRA claims (see `_patches`).

The cost is a `txtfusion` pass per region — small, and cached, since it doesn't depend on the
timestep — plus the copies' tokens in every block and a masked attention.

Where the copies go. Krea 2 measures the prompt BEFORE `post_input` and cuts the image back out
of the sequence at that offset, so longer text would shift the cut. The copies go after the canvas
instead, ahead of any reference tokens. References are told apart from the canvas by an index
(`timestep_zero_index`, counted the same way), so each block is handed that index moved past the
copies.
"""

from __future__ import annotations

import comfy.utils
import torch

#: Inputs and outputs of the last few `txtfusion` runs. Two, for cond and uncond sampled apart.
_CACHE = 2


def _text_hook(copies, entries):
    """Add the live region's text-side branches to one `txtfusion` Linear's output."""

    def hook(module, args, output):
        branches = entries.get(copies.active)
        if not branches:
            return output
        out = output
        for branch in branches:
            out = out + branch.delta(args[0])
        return out

    return hook


def _stash_hook(copies):
    def hook(module, args):
        if copies.active is None and args:
            copies.stashed = args[0]

    return hook


def _block_hook(state):
    """Move the reference tokens' start past the copies (see the module docstring)."""

    def hook(module, args, kwargs):
        index = kwargs.get("timestep_zero_index")
        if index is None or not state.copy_tokens:
            return None
        return args, {**kwargs, "timestep_zero_index": index + state.copy_tokens}

    return hook


class TextCopies:
    """The regions given their own prompt, their text-side branches, and the hooks that run them."""

    def __init__(self, regions: list[int], text_branches: dict[int, dict[str, list]]):
        #: Region indices with a copy, in sequence order.
        self.regions = regions
        #: `region -> {module path: [branch, ...]}` for the `txtfusion` Linears.
        self.branches = text_branches
        #: The region whose text-side branches are live, or None for the plain prompt.
        self.active: int | None = None
        #: The input of the last plain `txtfusion` run — the prompt the copies are made from.
        self.stashed: torch.Tensor | None = None
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.handles: list = []
        self.diffusion = None

    def inject(self, model, state) -> int:
        """Hook `model` (the `BaseModel`). Returns how many `txtfusion` modules were not found."""
        if self.handles:
            return 0
        diffusion = self.diffusion = model.diffusion_model
        self.handles.append(diffusion.txtfusion.register_forward_pre_hook(_stash_hook(self)))
        for block in diffusion.blocks:
            self.handles.append(
                block.register_forward_pre_hook(_block_hook(state), with_kwargs=True)
            )
        per_module: dict[str, dict[int, list]] = {}
        for region, paths in self.branches.items():
            for path, branches in paths.items():
                per_module.setdefault(path, {})[region] = branches
        missing = 0
        for path, entries in per_module.items():
            try:
                module = comfy.utils.get_attr(model, path)
            except AttributeError:
                missing += 1
                continue
            self.handles.append(module.register_forward_hook(_text_hook(self, entries)))
        return missing

    def eject(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stashed = None
        self.cache.clear()
        self.active = None
        self.diffusion = None

    def make(self, txt: torch.Tensor) -> torch.Tensor | None:
        """`[batch, regions * txtlen, features]`: each region's copy of `txt`, in `regions` order."""
        source, diffusion = self.stashed, self.diffusion
        if (
            not self.regions
            or source is None
            or diffusion is None
            or source.shape[0] != txt.shape[0]
        ):
            return None
        for key, copies in self.cache:
            if (
                key.shape == source.shape
                and key.device == source.device
                and torch.equal(key, source)
            ):
                return copies
        parts = []
        for region in self.regions:
            if not self.branches.get(region):
                parts.append(txt)
                continue
            self.active = region
            try:
                fused = diffusion.txtfusion(source, mask=None, transformer_options={})
            finally:
                self.active = None
            parts.append(diffusion.txtmlp(fused).to(txt.dtype))
        copies = torch.cat(parts, dim=1)
        self.cache = [(source.clone(), copies), *self.cache][:_CACHE]
        return copies
