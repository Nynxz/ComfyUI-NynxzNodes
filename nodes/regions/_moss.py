"""Regions on MOSS-Image: each region's own prompt, routed by attention, and its LoRAs.

MOSS-Image is a single-stream DiT: `[canvas | text]` in one sequence, the canvas first, the text
first refined on its own by a two-block `text_embedder`. Nothing in it takes a mask, but every
attention call goes through `optimized_attention_override`, and a `DIFFUSION_MODEL` wrapper sees
`context` before the model does. So, as on Z-Image (see `_zimage`):

  * the wrapper adds a copy of the text for each region with a prompt of its own, after the main
    prompt: `[canvas | main prompt | region copies]`;
  * the attention override adds the routing as an additive mask, in the joint blocks and in the
    `text_embedder`.

A copy is the region's own prompt alone, not the main prompt followed by it as on Z-Image. On MOSS
the main prompt is often a layout prompt that names every box, and a copy that carried it would
hand each region the others' descriptions. Layout Extras writes each box's prompt with the scene's
caption in front, so the region keeps the scene's style. A region with a LoRA and no prompt of its
own gets a copy of the main prompt, so a trigger word there is read through its LoRA. On an
unconditional row a copy is that row's own (negative) prompt.

Region LoRAs are forward hooks on the Linears (see `_branch`), gated per token as on Z-Image: by
the region's claim on the canvas tokens, and in full on the region's own copy, in the joint blocks
and the `text_embedder` (each sees its own sequence length, so the gates are kept per length).
The joint blocks' `ffn_down` runs through `comfy.ops.linear_input_act`, which fuses the `situ`
activation into the GEMM (into the int8 activation quantization, on an int8 model) and skips the
module's forward, so a hook on it never fires — and forcing its forward costs that fused path. Its
branch is applied around it instead (`FfnDownBranches`): `ffn_up_gate`'s output gives the
activation `ffn_down` reads, and the block's own output takes the branch's delta through the same
`gate_f` the block's residual does.

Who reads what, in the joint blocks:

  * a canvas token reads its region's copy by the region's claim on it, and the main prompt for
    whatever no region claims — `log(weight)`, so a token wholly inside one region reads only its
    copy;
  * a copy reads itself and the canvas, never the main prompt or another copy;
  * the main prompt never reads a copy.

In the `text_embedder` each prompt reads only itself: each is refined as though it were alone.

Positions. MOSS puts the image after the text on its first rope axis, at `f = text length`, so
copies appended to `context` would move the image and every token's distance to it. `rope` is
patched instead: the image and the main prompt keep exactly the positions they have without
regions, and each copy reads from the start of the axis as the main prompt does, as though it were
the prompt for its own region.

MOSS runs its blocks inside a memory compiler region that recycles tensors allocated there between
blocks, so the masks are built in the wrapper, before the blocks, never inside an attention call.

Not here yet: tracking, the layout lock and region references.
"""

from __future__ import annotations

import logging
import math
import re
import time

import comfy.ops
import comfy.patcher_extension
import comfy.utils
import torch

from . import _branch, _capture, _patches, _zimage
from ._branch import PROFILE, PROFILE_AT, profiled

WRAPPER_KEY = "nynxz_regions_moss"


#: Above this share of claimed tokens being part-claimed, a mask is soft by design and the routing
#: keeps it as drawn (masks) rather than giving each token to whoever claims most of it.
SOFT_SHARE = 0.25


def is_moss(model) -> bool:
    """Whether `model` (a ModelPatcher) wraps MOSS-Image's DiT."""
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    return diffusion_model is not None and all(
        hasattr(diffusion_model, name)
        for name in ("text_embedder", "video_embedder", "shared_block_modulation", "layers")
    )


def block_count(model) -> int:
    return len(model.model.diffusion_model.layers)


def already_regioned(model) -> bool:
    get = getattr(model, "get_wrappers", None)
    return bool(get and get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY))


def _routable(path: str) -> bool:
    """The Linears a region LoRA runs on: attention and MLP in the joint blocks and the text
    refiner, as on Z-Image. Not the embedders or the modulation."""
    return (".layers." in path and ".proj." in path) or ".text_embedder.blocks." in path


def load_region_loras(model, entries) -> list[tuple[int, dict]]:
    return _zimage.load_region_loras(model, entries, _routable, "MOSS-Image")


FFN_INJECTION_KEY = "nynxz_regions_moss_ffn"


def branch_footprint(loaded) -> tuple[int, int, set[int]]:
    """`(branches, parameters, ranks)` of the distinct branches in `loaded`."""
    seen, params, ranks = set(), 0, set()

    def walk(branch):
        nonlocal params
        if id(branch) in seen:
            return
        seen.add(id(branch))
        parts = getattr(branch, "parts", None)
        if parts is not None:
            for inner, _, _ in parts:
                walk(inner)
            return
        for name in ("down", "up", "w1", "w2"):
            tensor = getattr(branch, name, None)
            if tensor is not None:
                params += tensor.numel()
        if getattr(branch, "down", None) is not None:
            ranks.add(int(branch.down.shape[0]))

    for _, branches in loaded:
        for branch in branches.values():
            walk(branch)
    return len(seen), params, ranks


_FFN_DOWN = re.compile(r"^(?P<block>.*\.layers\.\d+)\.proj\.ffn_down$")


def split_ffn_down(loaded):
    """`loaded` as `(rest, ffn_down)`: the joint blocks' `ffn_down` branches, which can't hook the
    module itself, apart from every other branch."""
    rest, ffn = [], []
    for region, branches in loaded:
        here = {p: b for p, b in branches.items() if _FFN_DOWN.match(p)}
        rest.append((region, {p: b for p, b in branches.items() if p not in here}))
        if here:
            ffn.append((region, here))
    return rest, ffn


class FfnDownBranches:
    """Region branches for the joint blocks' `ffn_down`, applied around it.

    The block computes `x += gate_f * ffn_down(situ(ffn_up_gate(y)))` with the activation fused into
    `ffn_down`'s GEMM. A LoRA on `ffn_down` adds `delta(situ(...))` to that output, so: a hook on
    `ffn_up_gate` keeps the activation (core's own eager `situ`), and a hook on the block adds each
    branch's gated delta to the block's output, times the block's `gate_f`, in place — the residual
    stream lives outside MOSS's memory compiler region and must not be replaced by a new tensor.
    """

    def __init__(self, loaded, state):
        self.per_module = _branch._by_module(loaded)
        self.state = state
        self.handles: list = []
        self.acts: dict[str, torch.Tensor] = {}

    def inject(self, patcher):
        if self.handles:
            return
        for path, entries in self.per_module.items():
            block_path = _FFN_DOWN.match(path).group("block")
            try:
                block = comfy.utils.get_attr(patcher.model, block_path)
                up = comfy.utils.get_attr(patcher.model, block_path + ".proj.ffn_up_gate")
            except AttributeError:
                logging.warning("Nynxz Regions: %s is not on the loaded model; skipped.", path)
                continue
            self.handles.append(up.register_forward_hook(self._keep(block_path)))
            self.handles.append(block.register_forward_hook(self._add(block_path, entries)))

    def eject(self, patcher):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.acts.clear()
        self.state.reset()

    def _keep(self, key):
        def hook(module, args, output):
            if output.ndim == 3:
                with _branch.outside_compiled_memory():
                    self.acts[key] = _situ(output)
            return output

        return hook

    def _add(self, key, entries):
        state = self.state

        def hook(module, args, output):
            with _branch.outside_compiled_memory(), profiled("nynxz/lora ffn_down"):
                return add(module, args, output)

        def add(module, args, output):
            act = self.acts.pop(key, None)
            if act is None or output.ndim != 3 or len(args) < 2:
                return output
            batch, seqlen, _ = output.shape
            delta = None
            for regions, branch in entries:
                gate = None
                for region in regions:
                    part = state.for_tokens(batch, seqlen, region, act.device, act.dtype)
                    if part is not None:
                        gate = part if gate is None else gate + part
                if gate is None:
                    continue
                if delta is None:
                    delta = branch.delta(act) * gate
                else:
                    delta.addcmul_(branch.delta(act), gate)
            if delta is not None:
                output.addcmul_(delta, args[1][5])
            return output

        return hook


def _situ(up_gate: torch.Tensor) -> torch.Tensor:
    """The activation `ffn_down` reads, as core computes it eagerly."""
    return comfy.ops.INPUT_ACT_EAGER["situ"](up_gate)


def attach_loras(patched, loaded, state) -> None:
    """The region LoRAs' hooks: `_branch`'s for every Linear but the joint blocks' `ffn_down`,
    which `FfnDownBranches` applies around the module."""
    rest, ffn = split_ffn_down(loaded)
    if any(branches for _, branches in rest):
        _branch.attach(patched, rest, state)
    if ffn:
        ffn_branches = FfnDownBranches(ffn, state)
        patched.set_injections(
            FFN_INJECTION_KEY,
            [comfy.patcher_extension.PatcherInjection(ffn_branches.inject, ffn_branches.eject)],
        )


class Layout:
    """Where everything sits in this forward's sequence: `[canvas | main | copies]`."""

    __slots__ = ("canvas", "main", "slots", "valid")

    def __init__(self, canvas, main, slots, valid):
        self.canvas = canvas
        #: Main prompt tokens.
        self.main = main
        #: `(region, start, length)` per copy, `start` counted from the start of the text.
        self.slots = slots
        #: `[batch, copy tokens]` bool, False on a copy's padding; None when there is none.
        self.valid = valid

    @property
    def text(self) -> int:
        """The text length: what the `text_embedder` sees."""
        return self.main + sum(length for _, _, length in self.slots)

    @property
    def joint(self) -> int:
        """The sequence length the joint blocks see."""
        return self.canvas + self.text


class MossRegions:
    """The wrapper, the attention override and the rope patch, sharing the region state."""

    def __init__(
        self,
        state,
        prompts: dict[int, torch.Tensor],
        config,
        rope,
        previous=None,
        lora_regions=(),
    ):
        self.state = state
        #: `region -> [1, seq, features]`, the regions with a prompt of their own.
        self.prompts = prompts
        #: Regions with a LoRA: they get a copy too, and gates for their branches.
        self.lora_regions = set(lora_regions)
        self.regions = sorted(set(prompts) | self.lora_regions)
        self.config = config
        #: The model's own `rope`, to build each part's positions from.
        self.base_rope = rope
        #: An attention override that was on the model before this one, called in place of the
        #: attention itself so both apply.
        self.previous = previous
        self.layout: Layout | None = None
        #: `seqlen -> [batch, 1, seqlen, seqlen]`, this forward's masks.
        self.masks: dict[int, torch.Tensor] = {}
        #: `seqlen -> [(rows, queries, keys)]`: the routing as separate unmasked attention calls,
        #: when it is hard (see `_plans`); None when the masks carry it instead.
        self.plans: dict[int, list] | None = None
        #: Route hard regions by separate attention calls rather than masks (off only to compare).
        self.partition = True
        self.cache: dict[str, tuple] = {}
        self.warned = False

    # -- the wrapper ---------------------------------------------------------------------------

    def wrapper(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        options = transformer_options if transformer_options is not None else {}
        state = self.state
        sigmas = options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas.flatten()[0].item())
            state.note_sigma(sigma)
            state.note_step(sigma, options.get("sample_sigmas"))

        height, width = x.shape[-2], x.shape[-1]
        state.grid = (height, width)
        state.canvas_tokens = height * width
        state.ensure_anchor(x.device)
        _capture.remember(state)

        batch, main, features = context.shape
        cond = _patches._conditional_rows(options, batch) or [True] * batch
        parts, valids, slots, start = [context], [], [], main
        for region in self.regions:
            own = self.prompts.get(region)
            if own is not None:
                own = own[0].to(device=context.device, dtype=context.dtype)
            rows = [own if own is not None and cond[row] else context[row] for row in range(batch)]
            length = max(tokens.shape[0] for tokens in rows)
            block = context.new_zeros((batch, length, features))
            valid = torch.zeros((batch, length), dtype=torch.bool, device=context.device)
            for row, tokens in enumerate(rows):
                block[row, : tokens.shape[0]] = tokens
                valid[row, : tokens.shape[0]] = True
            parts.append(block)
            valids.append(valid)
            slots.append((region, start, length))
            start += length
        extended = torch.cat(parts, dim=1)
        valid = torch.cat(valids, dim=1) if valids else None
        if valid is not None and bool(valid.all()):
            valid = None

        self.layout = Layout(state.canvas_tokens, main, slots, valid)
        self._build_masks(batch, x.device, x.dtype)
        if self.lora_regions:
            self._build_gates(batch, x.device)
        if not PROFILE:
            return executor(x, timestep, extended, transformer_options, **kwargs)
        return self._profiled(executor, x, timestep, extended, transformer_options, kwargs)

    def _profiled(self, executor, x, timestep, context, options, kwargs):
        """A forward, timed — and on the `PROFILE_AT`th, profiled by op into the log."""
        self.forwards = getattr(self, "forwards", 0) + 1
        cuda = x.is_cuda
        if cuda:
            torch.cuda.synchronize(x.device)
            torch.cuda.reset_peak_memory_stats(x.device)
        start = time.perf_counter()
        if self.forwards == PROFILE_AT:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                out = executor(x, timestep, context, options, **kwargs)
                if cuda:
                    torch.cuda.synchronize(x.device)
            sort = "cuda_time_total" if cuda else "cpu_time_total"
            logging.info(
                "Nynxz Regions profile, forward %d, by %s:\n%s",
                self.forwards,
                sort,
                prof.key_averages().table(sort_by=sort, row_limit=30),
            )
            logging.info(
                "Nynxz Regions profile, by CPU time:\n%s",
                prof.key_averages().table(sort_by="cpu_time_total", row_limit=15),
            )
        else:
            out = executor(x, timestep, context, options, **kwargs)
        if cuda:
            torch.cuda.synchronize(x.device)
        layout = self.layout
        logging.info(
            "Nynxz Regions forward %d: %.0f ms, batch %d, %d tokens (%d canvas, %d text), %s, "
            "peak %.1f GB",
            self.forwards,
            (time.perf_counter() - start) * 1000.0,
            x.shape[0],
            layout.joint,
            layout.canvas,
            layout.text,
            "split attention" if self.plans else "masked attention",
            (torch.cuda.max_memory_allocated(x.device) if cuda else 0) / 1e9,
        )
        return out

    def _build_gates(self, batch: int, device) -> None:
        """The LoRA gates for the joint blocks and the `text_embedder`, by sequence length: canvas
        rows by claim (the canvas comes first), each copy in full for its own region."""
        state, layout = self.state, self.layout
        state.txtlen = 0
        joint = state.build_gates(batch, layout.joint, device)
        text = torch.zeros((batch, layout.text, joint.shape[-1]), device=device, dtype=joint.dtype)
        for region, start, length in layout.slots:
            if region not in self.lora_regions:
                continue
            weight = torch.full((batch, length), state.scale, device=device, dtype=joint.dtype)
            if layout.valid is not None:
                first = start - layout.main
                weight = weight * layout.valid[:, first : first + length].to(joint.dtype)
            at = layout.canvas + start
            joint[:, at : at + length, region] = weight
            text[:, start : start + length, region] = weight
        state.by_len = {layout.joint: joint, layout.text: text}

    def _build_masks(self, batch: int, device, dtype) -> None:
        layout = self.layout
        self.plans = self._plans(batch, device) if self.partition else None
        if self.plans is not None:
            self.masks = {}
            return
        masks = {}
        if layout.slots:
            masks[layout.text] = self._text_bias(batch, device, dtype)
        joint = self._joint_bias(batch, device, dtype)
        if joint is not None:
            masks[layout.joint] = joint
        self.masks = masks

    # -- rope ----------------------------------------------------------------------------------

    def rope(self, h, w, text_len, device, transformer_options=None):
        transformer_options = transformer_options if transformer_options is not None else {}
        layout = self.layout
        if layout is None or not layout.slots or text_len != layout.text:
            return self.base_rope(h, w, text_len, device, transformer_options)
        # The image and the main prompt exactly as without regions; each copy from position 0.
        parts = [self.base_rope(h, w, layout.main, device, transformer_options)]
        longest = max(length for _, _, length in layout.slots)
        copy = self.base_rope(1, 1, longest, device, transformer_options)[:, 1:]
        parts += [copy[:, :length] for _, _, length in layout.slots]
        return torch.cat(parts, dim=1)

    # -- the attention override ----------------------------------------------------------------

    def attention(self, func, q, k, v, heads, *args, **kwargs):
        if self.layout is None:
            return self._attention(func, q, k, v, heads, *args, **kwargs)
        with _branch.outside_compiled_memory(), profiled("nynxz/attention"):
            return self._attention(func, q, k, v, heads, *args, **kwargs)

    def _attention(self, func, q, k, v, heads, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        layout = self.layout
        mask = None
        if layout is not None and q.ndim == 3:
            seqlen = q.shape[1]
            joint = options.get("block_index") is not None
            plan = self.plans.get(seqlen) if self.plans else None
            if plan is not None and (seqlen == layout.joint) == joint:
                return self._partitioned(plan, func, q, k, v, heads, args, kwargs)
            if joint and seqlen == layout.joint:
                mask = self.masks.get(seqlen)
            elif joint and not self.warned:
                self.warned = True
                logging.warning(
                    "Nynxz Regions: MOSS-Image's blocks see %d tokens where the region layout "
                    "expects %d — regions are NOT being routed this run. Please report this.",
                    seqlen,
                    layout.joint,
                )
            elif not joint and seqlen == layout.text:
                mask = self.masks.get(seqlen)
        if mask is not None:
            if mask.dtype != q.dtype or mask.device != q.device:
                mask = mask.to(device=q.device, dtype=q.dtype)
            previous = kwargs.get("mask")
            kwargs = {**kwargs, "mask": mask if previous is None else previous + mask}
        if self.previous is not None:
            return self.previous(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    def _partitioned(self, plan, func, q, k, v, heads, args, kwargs):
        """The routing as one unmasked attention call per group of queries that read the same keys
        — so the fast kernels (flash, sage) run, and nothing `[seq, seq]` is ever built."""

        def call(*a, **kw):
            return self.previous(func, *a, **kw) if self.previous is not None else func(*a, **kw)

        out = q.new_empty((*q.shape[:2], v.shape[-1]))
        for rows, queries, keys in plan:
            out[rows, queries] = call(
                q[rows][:, queries],
                k[rows][:, keys],
                v[rows][:, keys],
                heads,
                *args,
                **kwargs,
            )
        return out

    def _plans(self, batch: int, device) -> dict[int, list] | None:
        """The routing as groups of queries and the keys each reads, when every canvas token is in
        one region or none (a token on a box's edge going to whoever claims most of it), nothing is
        isolated and no region reads its prompt more or less strongly than its claim: then the masks
        only ever say "read these, never those", which separate attention calls say exactly. None
        otherwise, and the masks do it.

        Groups, per batch row (or for all rows when every copy has the same length on each):

          * the background (canvas tokens no copied region claims) and the main prompt read the
            canvas and the main prompt;
          * each copied region's canvas tokens and its copy read the canvas and the copy;

        and in the `text_embedder`, each prompt reads itself."""
        state, layout = self.state, self.layout
        if not layout.slots:
            return None
        current = state.map if state.map is not None else state.anchor
        if current is None or current.shape[0] < layout.canvas:
            return None
        isolation = self.config.isolation_at(state.schedule, state.fraction)
        if isolation > 0.0 and len(state.regions) >= 2:
            return None
        copied = [region for region, _, _ in layout.slots]
        if any(state.influences[r] != 1.0 for r in copied if r < len(state.influences)):
            return None
        claim = current[: layout.canvas, copied].float()
        # Box edges rarely fall on the 16px token grid, so the tokens along them are part-claimed.
        # Those go to whoever claims most of them; a mask that is soft through and through (a wide
        # feather) is left to the masks, which read it as drawn.
        soft = ((claim > 1e-3) & (claim < 1.0 - 1e-3)).any(dim=-1)
        claimed = (claim > 1e-3).any(dim=-1)
        if int(soft.sum()) > SOFT_SHARE * max(int(claimed.sum()), 1):
            return None

        key = (batch, device, tuple(layout.slots), self._valid_key(), id(current))
        hit = self.cache.get("plans")
        if hit is not None and hit[0] == key:
            return hit[1]

        strongest, which = claim.max(dim=-1)
        owner = torch.where(strongest >= 0.5, which, torch.full_like(which, -1)).to(device)

        def arange(start, length):
            return torch.arange(start, start + length, device=device)

        if layout.valid is None:
            row_sets = [(slice(0, batch), [length for _, _, length in layout.slots])]
        else:
            valid = layout.valid.to(device)
            row_sets = []
            for row in range(batch):
                lengths = []
                for _, start, length in layout.slots:
                    first = start - layout.main
                    lengths.append(int(valid[row, first : first + length].sum()))
                row_sets.append((slice(row, row + 1), lengths))

        canvas = arange(0, layout.canvas)
        main = arange(layout.canvas, layout.main)
        joint, text = [], []
        for rows, lengths in row_sets:
            joint.append(
                (
                    rows,
                    torch.cat(((owner == -1).nonzero().flatten(), main)),
                    torch.cat((canvas, main)),
                )
            )
            text.append((rows, arange(0, layout.main), arange(0, layout.main)))
            for j, ((_, start, length), used) in enumerate(zip(layout.slots, lengths, strict=True)):
                span = arange(layout.canvas + start, length)
                read = arange(layout.canvas + start, used)
                joint.append(
                    (
                        rows,
                        torch.cat(((owner == j).nonzero().flatten(), span)),
                        torch.cat((canvas, read)),
                    )
                )
                text.append((rows, arange(start, length), arange(start, used)))
        plans = {layout.joint: joint, layout.text: text}
        self.cache["plans"] = (key, plans)
        return plans

    # -- the masks -----------------------------------------------------------------------------

    def _cached(self, name, key, build):
        hit = self.cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        value = build()
        self.cache[name] = (key, value)
        return value

    def _valid_key(self):
        valid = self.layout.valid
        return None if valid is None else (valid.shape, int(valid.sum()), id(valid))

    def _hide_padding(self, bias: torch.Tensor, offset: int) -> None:
        """Nobody reads a copy's padding. `offset` is where the text starts in `bias`."""
        layout = self.layout
        if layout.valid is not None:
            copies = bias[:, :, offset + layout.main : offset + layout.text]
            copies.masked_fill_(~layout.valid.to(bias.device).unsqueeze(1), _patches._NEVER)

    def _text_bias(self, batch, device, dtype):
        """The `text_embedder`: each prompt refined alone."""
        layout = self.layout
        size = layout.text
        key = (batch, size, dtype, device, tuple(layout.slots), self._valid_key())

        def build():
            bias = torch.zeros((batch, size, size), device=device, dtype=torch.float32)
            bias[:, : layout.main, layout.main :] = _patches._NEVER
            for _, start, length in layout.slots:
                copy = slice(start, start + length)
                bias[:, copy, :] = _patches._NEVER
                bias[:, copy, copy] = 0.0
            self._hide_padding(bias, 0)
            return bias.to(dtype).unsqueeze(1)

        return self._cached("text", key, build)

    def _joint_bias(self, batch, device, dtype):
        state, layout = self.state, self.layout
        current = state.map if state.map is not None else state.anchor
        if current is None or current.shape[0] < layout.canvas:
            return None
        regions = len(state.regions)
        isolation = self.config.isolation_at(state.schedule, state.fraction)
        isolate = isolation > 0.0 and regions >= 2
        if not layout.slots and not isolate:
            return None
        size = layout.joint
        key = (
            batch,
            size,
            dtype,
            device,
            tuple(layout.slots),
            self._valid_key(),
            round(isolation, 4),
            id(current),
        )

        def build():
            claim = current[: layout.canvas, :regions].to(device=device, dtype=torch.float32)
            bias = torch.zeros((batch, size, size), device=device, dtype=torch.float32)
            canvas = slice(0, layout.canvas)
            text = slice(layout.canvas, size)
            main = slice(layout.canvas, layout.canvas + layout.main)
            copies = slice(layout.canvas + layout.main, size)

            if isolate:
                share = claim / claim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                share = share @ state.groups.to(share.device, share.dtype)
                agreement = (share @ share.transpose(0, 1)).clamp(0.0, 1.0)
                # Region against region only: a background token belongs to no region, and pushing
                # it away from every token would take the scene's coherence with it.
                claimed = claim.sum(dim=-1).clamp(0.0, 1.0)
                both = claimed.unsqueeze(1) * claimed.unsqueeze(0)
                bias[:, canvas, canvas] = -isolation * (1.0 - agreement) * both

            bias[:, :, copies] = _patches._NEVER
            if layout.slots:
                taken = torch.zeros_like(claim[:, 0])
                for region, start, length in layout.slots:
                    copy = slice(layout.canvas + start, layout.canvas + start + length)
                    bias[:, copy, text] = _patches._NEVER
                    bias[:, copy, copy] = 0.0
                    weight = claim[:, region]
                    taken = taken + weight
                    logits = weight.clamp(min=_patches._LEAST).log()
                    influence = state.influences[region] if region < len(state.influences) else 1.0
                    if influence != 1.0:
                        logits = logits + torch.where(
                            weight > _patches._LEAST, math.log(influence), 0.0
                        )
                    bias[:, canvas, copy] = logits.unsqueeze(-1)
                rest = (1.0 - taken).clamp(min=_patches._LEAST).log().unsqueeze(-1)
                bias[:, canvas, main] = rest
            self._hide_padding(bias, layout.canvas)
            return bias.to(dtype).unsqueeze(1)

        return self._cached("joint", key, build)


def install(patched, state, prompts: dict[int, torch.Tensor], config, loaded=()):
    """Put the wrapper, the attention override and the rope patch on `patched` (a clone); `loaded`
    is the region LoRAs (`load_region_loras`), attached separately with `attach_loras`."""
    options = patched.model_options.setdefault("transformer_options", {})
    routed = MossRegions(
        state,
        prompts,
        config,
        patched.get_model_object("diffusion_model.rope"),
        options.get("optimized_attention_override"),
        lora_regions=[region for region, _ in loaded],
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, routed.wrapper
    )
    patched.add_object_patch("diffusion_model.rope", routed.rope)
    options["optimized_attention_override"] = routed.attention
    return routed
