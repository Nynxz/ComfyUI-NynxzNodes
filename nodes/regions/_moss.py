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

Not here yet: the layout lock and region references.
"""

from __future__ import annotations

import logging
import re
import time

import comfy.ops
import comfy.patcher_extension
import comfy.utils
import torch

from . import _bias, _branch, _capture, _copies, _lora, _patches, _track, _zimage
from ._bias import NEVER
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
    """`[(region, {module_path: branch})]` for every region's LoRAs, on the routable Linears."""
    return [
        (lora.region, lora.branches)
        for lora in _lora.load_region_loras(model, entries, _routable, "MOSS-Image")
    ]


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
        self.per_module = _branch.by_module(loaded)
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
    """Where everything sits in this forward's sequence: `[canvas | main | copies | refs]`."""

    __slots__ = ("canvas", "main", "refs", "slots", "valid")

    def __init__(self, canvas, main, slots, valid, refs=0):
        self.canvas = canvas
        #: Main prompt tokens.
        self.main = main
        #: `(region, start, length)` per copy, `start` counted from the start of the text.
        self.slots = slots
        #: `[batch, copy tokens]` bool, False on a copy's padding; None when there is none.
        self.valid = valid
        #: Reference latent tokens riding in the sequence after the text (the "index" methods).
        #: References MOSS turns into keys and values beforehand aren't counted: they only add keys.
        self.refs = refs

    @property
    def text(self) -> int:
        """The text length: what the `text_embedder` sees."""
        return self.main + sum(length for _, _, length in self.slots)

    @property
    def joint(self) -> int:
        """The sequence length the joint blocks see."""
        return self.canvas + self.text + self.refs


#: The reference methods whose latents MOSS-Image keeps in the sequence; any other turns them into
#: keys and values before the blocks.
IN_SEQUENCE_REFS = ("index", "index_timestep_zero")


def sequence_refs(kwargs) -> int:
    """How many reference latent tokens this forward's blocks carry after the text."""
    latents = kwargs.get("ref_latents") or []
    if not latents or kwargs.get("ref_latents_method") not in IN_SEQUENCE_REFS:
        return 0
    return sum(ref.shape[-2] * ref.shape[-1] for ref in latents)


class MossRegions(_copies.Routing):
    """The wrapper, the attention override and the rope patch, sharing the region state."""

    def __init__(
        self,
        state,
        prompts: dict[int, torch.Tensor],
        settings,
        rope,
        previous=None,
        lora_regions=(),
    ):
        super().__init__(state, prompts, settings, previous)
        #: Regions with a LoRA: they get a copy too, and gates for their branches.
        self.lora_regions = set(lora_regions)
        self.regions = sorted(set(prompts) | self.lora_regions)
        #: The model's own `rope`, to build each part's positions from.
        self.base_rope = rope
        #: `seqlen -> [batch, 1, seqlen, seqlen]`, this forward's masks.
        self.masks: dict[int, torch.Tensor] = {}
        #: `seqlen -> [(rows, queries, keys)]`: the routing as separate unmasked attention calls,
        #: when it is hard (see `_plans`); None when the masks carry it instead.
        self.plans: dict[int, list] | None = None
        #: Route hard regions by separate attention calls rather than masks (off only to compare).
        self.partition = True
        #: True while MOSS encodes the references on their own (`preprocess_refs`), whose attention
        #: is neither the text's nor the blocks' and is left alone.
        self.encoding_refs = False
        #: Whether this forward carries a conditional row (which tracking reads, so CFG shares one
        #: map across both cond and uncond passes).
        self.conditional = True
        #: Canvas `v` features collected across the `route_blocks` of this forward (MOSS applies
        #: RoPE to `q` and `k` before calling the attention override, leaving `v` unrotated).
        self._track_features: list[torch.Tensor] = []

    # -- the wrapper ---------------------------------------------------------------------------

    def wrapper(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        options = transformer_options if transformer_options is not None else {}
        state = self.state
        _copies.begin_forward(state, options, (x.shape[-2], x.shape[-1]), x.device)
        extended, slots, valid = _copies.append_copies(
            context, self.regions, self.prompts, options, with_main=False
        )
        self.layout = Layout(
            state.canvas_tokens, context.shape[1], slots, valid, sequence_refs(kwargs)
        )
        self.encoding_refs = False
        batch = x.shape[0]
        cond = _patches.conditional_rows(options, batch) or [True] * batch
        self.conditional = any(cond)
        self._track_features.clear()
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
            weight = _copies.copy_gate(
                state, layout.valid, start - layout.main, length, batch, device, joint.dtype
            )
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

    def rope(self, h, w, text_len, device, transformer_options=None, ref_shapes=(), f_offset=0):
        transformer_options = transformer_options if transformer_options is not None else {}
        self.encoding_refs = h == 0 and w == 0 and text_len == 0 and bool(ref_shapes)
        layout = self.layout
        if layout is None or not layout.slots or text_len != layout.text:
            return self._base_rope(
                h, w, text_len, device, transformer_options, ref_shapes, f_offset
            )
        # With references MOSS ends the text just before the image (`f_offset = -text_len`): end
        # the main prompt there, as it is without regions.
        if f_offset and f_offset == -text_len:
            f_offset = -layout.main
        # The image, the main prompt and the references exactly as without regions; each copy from
        # where the main prompt starts.
        base = self._base_rope(h, w, layout.main, device, transformer_options, ref_shapes, f_offset)
        split = h * w + layout.main
        longest = max(length for _, _, length in layout.slots)
        copy = self._base_rope(1, 1, longest, device, transformer_options, (), f_offset)[:, 1:]
        parts = [base[:, :split]]
        parts += [copy[:, :length] for _, _, length in layout.slots]
        parts.append(base[:, split:])
        return torch.cat(parts, dim=1)

    def _base_rope(self, h, w, text_len, device, transformer_options, ref_shapes, f_offset):
        # A ComfyUI from before MOSS-Image's references takes no `ref_shapes` or `f_offset`.
        if not ref_shapes and not f_offset:
            return self.base_rope(h, w, text_len, device, transformer_options)
        return self.base_rope(h, w, text_len, device, transformer_options, ref_shapes, f_offset)

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
        if layout is not None and q.ndim == 3 and not self.encoding_refs:
            seqlen = q.shape[1]
            joint = options.get("block_index") is not None
            plan = self.plans.get(seqlen) if self.plans else None
            if plan is not None and (seqlen == layout.joint) == joint:
                return self._partitioned(plan, func, q, k, v, heads, args, kwargs)
            if joint and seqlen == layout.joint:
                self._follow(
                    v,
                    options.get("block_index"),
                    options.get("total_blocks"),
                    q.device,
                    q.dtype,
                )
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
            if k.shape[1] > mask.shape[-1]:
                # References' keys and values computed beforehand: everyone reads them.
                mask = torch.nn.functional.pad(mask, (0, k.shape[1] - mask.shape[-1]))
            previous = kwargs.get("mask")
            kwargs = {**kwargs, "mask": mask if previous is None else previous + mask}
        return self.call(func, q, k, v, heads, *args, **kwargs)

    def _follow(self, v, block, total, device, dtype) -> None:
        """Let the map follow each subject's canvas `v` features at `route_at`, averaged across
        `route_blocks` (on conditional rows only, so cond and uncond share one map)."""
        state, settings, layout = self.state, self.settings, self.layout
        anchor = state.anchor
        if (
            not settings.tracking
            or not self.conditional
            or state.frozen
            or anchor is None
            or block is None
            or not total
        ):
            return
        wanted = _patches.route_block(settings.route_at, int(total), settings.route_blocks)
        if block not in wanted:
            return
        self._track_features.append(_track.canvas_features(v.unsqueeze(1), slice(0, layout.canvas)))
        if block != wanted[-1]:
            return
        features = torch.nn.functional.normalize(
            torch.stack(self._track_features, dim=0).mean(dim=0), dim=-1
        )
        self._track_features.clear()
        anchor = anchor.to(device)
        tracked = _track.refine(anchor, features, settings.refine, settings.sharpness)
        blended = _track.blend(anchor, tracked, settings.track)
        state.map = _track.smooth(blended, state.grid, settings.smooth)
        self._build_masks(v.shape[0], device, dtype)
        if self.lora_regions:
            self._build_gates(v.shape[0], device)
        _capture.remember(state, _track.drift(anchor, state.map))

    def _partitioned(self, plan, func, q, k, v, heads, args, kwargs):
        """The routing as one unmasked attention call per group of queries that read the same keys
        — so the fast kernels (flash, sage) run, and nothing `[seq, seq]` is ever built."""
        out = q.new_empty((*q.shape[:2], v.shape[-1]))
        # References' keys and values computed beforehand sit after the sequence: everyone reads them.
        extra = (
            torch.arange(q.shape[1], k.shape[1], device=q.device)
            if k.shape[1] > q.shape[1]
            else None
        )
        previous = kwargs.get("mask")
        for rows, queries, keys in plan:
            if extra is not None:
                keys = torch.cat((keys, extra))
            call_kwargs = kwargs
            if previous is not None:
                call_kwargs = {**kwargs, "mask": _slice_mask(previous, rows, queries, keys)}
            out[rows, queries] = self.call(
                func,
                q[rows][:, queries],
                k[rows][:, keys],
                v[rows][:, keys],
                heads,
                *args,
                **call_kwargs,
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
        if not layout.slots or self.settings.tracking:
            return None
        current = self.current()
        if current is None or self.isolation() > 0.0:
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

        key = (batch, device, tuple(layout.slots), layout.refs, self.valid_key(), id(current))
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
        refs = arange(layout.canvas + layout.text, layout.refs)
        joint, text = [], []
        for rows, lengths in row_sets:
            joint.append(
                (
                    rows,
                    torch.cat(((owner == -1).nonzero().flatten(), main)),
                    torch.cat((canvas, main, refs)),
                )
            )
            if layout.refs:
                joint.append((rows, refs, torch.cat((canvas, main, refs))))
            text.append((rows, arange(0, layout.main), arange(0, layout.main)))
            for j, ((_, start, length), used) in enumerate(zip(layout.slots, lengths, strict=True)):
                span = arange(layout.canvas + start, length)
                read = arange(layout.canvas + start, used)
                joint.append(
                    (
                        rows,
                        torch.cat(((owner == j).nonzero().flatten(), span)),
                        torch.cat((canvas, read, refs)),
                    )
                )
                text.append((rows, arange(start, length), arange(start, used)))
        plans = {layout.joint: joint, layout.text: text}
        self.cache["plans"] = (key, plans)
        return plans

    # -- the masks -----------------------------------------------------------------------------

    def _text_bias(self, batch, device, dtype):
        """The `text_embedder`: each prompt refined alone."""
        layout = self.layout
        size = layout.text
        key = (batch, size, dtype, device, tuple(layout.slots), self.valid_key())

        def build():
            bias = torch.zeros((batch, size, size), device=device, dtype=torch.float32)
            bias[:, : layout.main, layout.main :] = NEVER
            for _, start, length in layout.slots:
                copy = slice(start, start + length)
                bias[:, copy, :] = NEVER
                bias[:, copy, copy] = 0.0
            _bias.hide_padding(bias, layout.valid, layout.main)
            return bias.to(dtype).unsqueeze(1)

        return self.cached("text", key, build)

    def _joint_bias(self, batch, device, dtype):
        state, layout = self.state, self.layout
        current = self.current()
        if current is None:
            return None
        isolation = self.isolation()
        if not layout.slots and isolation <= 0.0:
            return None
        size = layout.joint
        key = (
            batch,
            size,
            dtype,
            device,
            tuple(layout.slots),
            self.valid_key(),
            round(isolation, 4),
            id(current),
        )

        def build():
            regions = len(state.regions)
            claim = current[: layout.canvas, :regions].to(device=device, dtype=torch.float32)
            bias = torch.zeros((batch, size, size), device=device, dtype=torch.float32)
            canvas = slice(0, layout.canvas)
            if isolation > 0.0:
                bias[:, canvas, canvas] = _bias.isolation(claim, state.groups, isolation)
            text_end = layout.canvas + layout.text
            refs = slice(text_end, size)
            # Nobody but a copy's own tokens and the canvas reads a copy.
            bias[:, :, layout.canvas + layout.main : text_end] = NEVER
            if layout.slots:
                at = layout.canvas
                _bias.route_copies(
                    bias,
                    claim,
                    canvas,
                    [
                        (r, slice(at + start, at + start + length))
                        for r, start, length in layout.slots
                    ],
                    hidden=slice(layout.canvas, text_end),
                    shared=[slice(layout.canvas, layout.canvas + layout.main)],
                    influences=state.influences,
                    also_read=[refs] if layout.refs else (),
                )
            _bias.hide_padding(bias, layout.valid, layout.canvas + layout.main)
            return bias.to(dtype).unsqueeze(1)

        return self.cached("joint", key, build)


def _slice_mask(mask, rows, queries, keys):
    """The model's own attention mask (`[batch or 1, 1, queries or 1, keys]`) for one group."""
    if mask.shape[0] != 1:
        mask = mask[rows]
    if mask.shape[-2] != 1:
        mask = mask[..., queries, :]
    return mask[..., keys]


def install(patched, state, prompts: dict[int, torch.Tensor], settings, loaded=()):
    """Put the wrapper, the attention override and the rope patch on `patched` (a clone); `loaded`
    is the region LoRAs (`load_region_loras`), attached separately with `attach_loras`."""
    options = patched.model_options.setdefault("transformer_options", {})
    routed = MossRegions(
        state,
        prompts,
        settings,
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


def install_regions(patched, entries, prompts, settings, sources) -> list[str]:
    """Regions on `patched` (a clone): each region's own prompt, routed by attention, and its
    LoRAs, gated per token. Returns the log lines."""
    state = _copies.LengthGateState(entries, settings.schedule, settings.track, sources)
    loaded = load_region_loras(patched, entries)
    install(patched, state, prompts, settings, loaded)
    if loaded:
        attach_loras(patched, loaded, state)
    lines = _zimage.region_lines(len(entries), prompts, loaded)
    lines.append("")
    lines.append(
        f"MOSS-Image: {len(prompts)} region prompt(s) routed by attention"
        + (f", {sum(len(b) for _, b in loaded)} LoRA branch(es)" if loaded else "")
    )
    if loaded:
        count, params, ranks = branch_footprint(loaded)
        lines.append(
            f"LoRA branches: {count} distinct, rank {', '.join(map(str, sorted(ranks))) or '-'}, "
            f"{params * 2 / 1e6:.0f} MB at 16-bit on the GPU"
        )
    if settings.tracking:
        lines.append(
            f"tracking: {settings.track:.2f} toward canvas v-features, {settings.refine} rounds, "
            f"smooth {settings.smooth}"
        )
    lines.append(f"schedule: {settings.schedule.describe()}")
    lines.append(f"isolation: {settings.isolation:.2f}")
    return lines
