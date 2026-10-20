"""The live LoRA branches and the lifecycle that attaches them.

`y = W x + s * B(A x)` is applied per token, independently, and nothing in that requires `s` to be a
scalar. Keep `up` and `down` apart instead of folding their product into the weight and `s` becomes
a vector over sequence positions:

    y_i = W x_i  +  SUM_k  gate_k(i) * strength_k * B_k(A_k x_i)

Region B's delta then never touches region A's tokens, and neither of them touches the background.
The cost is two rank-r matmuls per wrapped Linear — about 2% of step time at r=32.

Why forward hooks rather than `add_object_patch`. Replacing a Linear with a wrapper module changes
its path in `named_modules()`, and `ModelPatcher.load` derives every weight key from exactly that
path (`comfy/model_patcher.py:960`) — so a wrapped `...attn.wq` would have its weights looked up at
`...attn.wq.base.weight`, quietly breaking offload, casting, and any normally-loaded LoRA on the
same layer. A forward hook leaves the module tree alone. That last point is load-bearing: these
nodes expect a stock `LoraLoader` upstream for an overall style, and additive hooks over merged
weights is exactly how the two compose.

Hooks do have to be unwound, or a second sampler run stacks a second set. `PatcherInjection` is the
mechanism ComfyUI provides for precisely this: `inject` fires at the end of `patch_model` (after
weights are live) and `eject` at the start of `unpatch_model`, so the hooks exist exactly as long as
the patched model does.
"""

from __future__ import annotations

import contextlib
import logging
import os

import comfy.utils
import torch
from comfy.patcher_extension import PatcherInjection

#: Key under which this pack's injection is registered on the ModelPatcher.
INJECTION_KEY = "nynxz_regions"


def _hook(entries, state):
    """Forward hook adding every region's gated branch to one Linear's output.

    `entries` is `[(regions, Branch), ...]` — only the regions whose LoRA actually touches this
    module, so a layer that only one LoRA trains costs only that one branch.

    Accumulated with `addcmul` into a single tensor we own, because everything here is
    `[batch, seq, out]` and on an MLP projection that is hundreds of MB. The obvious spelling,
    `out = out + branch.delta(x) * gate`, allocates one of those for the gated product and another
    for the sum, per region, per module, every step. Fused and then in-place, only `branch.delta(x)`
    is transient and there is exactly one accumulator. `output` itself is never written to: it is
    the module's own tensor and the caller may still hold it.
    """

    def hook(module, args, output):
        with outside_compiled_memory(), profiled("nynxz/lora"):
            return add(module, args, output)

    def add(module, args, output):
        x = args[0]
        if x.ndim != 3:
            return output
        batch, seqlen, _ = x.shape
        out = None
        for regions, branch in entries:
            gate = None
            for region in regions:
                part = state.for_tokens(batch, seqlen, region, x.device, x.dtype)
                if part is not None:
                    gate = part if gate is None else gate + part
            if gate is None:
                continue
            if out is None:
                out = torch.addcmul(output, branch.delta(x), gate)
            else:
                out.addcmul_(branch.delta(x), gate)
        return output if out is None else out

    return hook


def outside_compiled_memory():
    """Allocate outside ComfyUI's memory compiler while in here.

    With dynamic VRAM, a model's blocks run inside a recorded allocation plan that is replayed every
    step. An allocation the plan doesn't have — any of a hook's — breaks it, and the whole forward
    then allocates the slow way, thousands of driver calls a step. Core's own extra kernels step out
    of the plan for their scratch memory the same way. Nothing on a ComfyUI without the compiler.
    """
    pause = _pause_malloc_graph()
    return pause() if pause else contextlib.nullcontext()


_PAUSE = None


def _pause_malloc_graph():
    """Core's `pause_malloc_graph`, looked up once; False on a ComfyUI without it."""
    global _PAUSE
    if _PAUSE is None:
        try:
            import comfy.model_prefetch

            _PAUSE = getattr(comfy.model_prefetch, "pause_malloc_graph", None) or False
        except ImportError:
            _PAUSE = False
    return _PAUSE


#: `NYNXZ_REGIONS_PROFILE=1` in ComfyUI's environment logs each regioned forward's time and peak
#: memory, and one forward's profile (where the time went, by op, with this pack's hooks labelled).
#: Only the MOSS-Image path times its forwards so far.
PROFILE = os.environ.get("NYNXZ_REGIONS_PROFILE", "") not in ("", "0")
#: The forward profiled in full: a few in, past the first step's setup and the uncond's first run.
PROFILE_AT = 4


def profiled(label: str):
    """A profiler label for `label` when profiling, else nothing."""
    return torch.profiler.record_function(label) if PROFILE else contextlib.nullcontext()


def _by_module(loaded) -> dict[str, list[tuple[tuple[int, ...], object]]]:
    """Invert `[(region, {module: Branch})]` into `module -> [(regions, Branch)]`.

    A branch several regions share (the same LoRA at the same strength, loaded once) is one entry
    with all of them: its delta is computed once per forward and gated by their summed gates,
    instead of once per region. Regions own disjoint tokens after overlap is resolved, so the sum
    is each token's own region's gate.

    Pairs rather than one entry per region, because a region may carry several LoRAs — they share
    its gate and their deltas simply add, which is what stacking LoRAs means anyway, except confined
    to this region instead of applied to the whole image.

    One hook per module rather than one per (module, region): a hook is a Python call on every
    forward of every block, and 28 blocks x 5 projections is already 140 of them.
    """
    grouped: dict[str, dict[int, tuple[list[int], object]]] = {}
    for index, branches in loaded:
        for path, branch in (branches or {}).items():
            regions, _ = grouped.setdefault(path, {}).setdefault(id(branch), ([], branch))
            if index not in regions:
                regions.append(index)
    return {
        path: [(tuple(regions), branch) for regions, branch in by_branch.values()]
        for path, by_branch in grouped.items()
    }


class Injection:
    """Attaches the branch hooks when the model goes live and removes them when it comes back."""

    def __init__(self, loaded, state, text=None):
        self.per_module = _by_module(loaded)
        self.state = state
        self.text = text
        self.handles: list = []

    def inject(self, patcher):
        if self.handles:
            return  # already live; inject_model guards this too, but clones can double up
        missing = 0
        for path, entries in self.per_module.items():
            try:
                module = comfy.utils.get_attr(patcher.model, path)
            except AttributeError:
                missing += 1
                continue
            self.handles.append(module.register_forward_hook(_hook(entries, self.state)))
        if self.text is not None:
            missing += self.text.inject(patcher.model, self.state)
        if missing:
            logging.warning(
                "Nynxz Regions: %d LoRA target modules were not found on the loaded model and are not "
                "being applied. The LoRA may not match this checkpoint.",
                missing,
            )
        logging.info("Nynxz Regions: attached %d branch hooks.", len(self.handles))

    def eject(self, patcher):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.text is not None:
            self.text.eject()
        self.state.reset()


def attach(patcher, loaded, state, text=None) -> None:
    """Register the branch injection on an (already cloned) ModelPatcher; `text` is the regions'
    copies of the prompt (`_text.TextCopies`), if they get them."""
    injection = Injection(loaded, state, text)
    patcher.set_injections(INJECTION_KEY, [PatcherInjection(injection.inject, injection.eject)])
