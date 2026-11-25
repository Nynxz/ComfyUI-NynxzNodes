"""Turning a LoRA into a per-Linear branch that a per-token gate can sit in front of.

`comfy.lora.load_lora` already solves the part that is genuinely hard and boring — mapping a dozen
LoRA key dialects (kohya, diffusers, ai-toolkit, transformers, ...) onto model weight keys. So this
uses it, and then takes the road not taken: instead of handing the result to `add_patches`, which
folds the delta into the weight, it keeps the delta available as a *side branch* so a gate can
multiply it per token.

**Only patches that factor are routed, and that limit is deliberate.** Two shapes factor: a plain
LoRA, and LoKr.

`FactoredBranch` keeps `up` and `down` apart, so the branch is two rank-r matmuls — about 2% of
step time at r=32. The scale convention is copied from `comfy/weight_adapter/lora.py` rather than
guessed::

    alpha = (stored_alpha / rank) if stored_alpha is not None else 1.0
    W += strength * alpha * (up @ down)          up: [out, rank]   down: [rank, in]

`KronBranch` is LoKr: `W += strength * alpha * kron(w1, w2)`, as `comfy/weight_adapter/lokr.py`
merges it, applied to the input as two small matmuls over its reshaped halves the way that file's
bypass `h()` does — no `[out, in]` product is ever built.

LoHa, OFT, BOFT, GLoRA, DoRA, Tucker LoKr and the `diff` / `set` patches `load_lora` also emits do
not factor that way. They are skipped and named, not approximated.

There WAS a fallback that approximated them — materialise `calculate_weight([patch], W) - W` and
fit a rank-32 branch to it with a randomized SVD — and removing it is the point of this note, so
it does not get reinvented. The cost was not where it looked. Per layer it built four full
`[out, in]` fp32 matrices and ran an SVD over them; on CPU, so it never showed up as VRAM. And the
formats that needed it are exactly the ones with NO factorable layers, so it fired on every routed
layer of every LoRA on every region rather than on the occasional straggler — minutes of silent
work before the first step.

Then it succeeded, which was worse. It produced a branch for every layer in the model where a plain
LoRA produces branches for the subset it trained, and every branched module costs full
`[batch, seq, out]` activations per region per step. "Supported" turned into a multi-gigabyte
sampling footprint and a stall with nothing in the log to attribute it to.

Skipping is cheaper and much louder. For those formats a stock `LoraLoader` merges the delta whole
and correctly; it just cannot confine it to a region, which is the trade.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths
import torch
from comfy.weight_adapter import LoKrAdapter, LoRAAdapter

from ..lora import _qkv
from . import _branch


class FactoredBranch:
    """A plain LoRA's delta for one Linear, kept as its two factors instead of their product.

    Cast lazily and cached: the factors land on CPU in whatever dtype they were saved in, and the
    module they attach to may be offloaded, re-cast, or moved between runs. Re-casting per call
    would be a measurable tax at 28 blocks x 5 projections x N concepts, so the cast result is held
    until the target device/dtype actually changes.

    `scale` is folded into the cached `up` rather than applied to `delta(x)`'s result. The result is
    `[batch, seq, out]`, so scaling it afterwards is a second tensor that size — per region, per
    module, every step. `up` is `[out, rank]` and the fold happens once per cast.
    """

    __slots__ = ("_cast", "_cast_key", "down", "scale", "up")

    def __init__(self, down: torch.Tensor, up: torch.Tensor, scale: float):
        self.down = down  # [rank, in]
        self.up = up  # [out, rank]
        self.scale = scale
        self._cast_key = None
        self._cast = None

    def factors(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        if self._cast_key != key:
            # Scale in fp32 before the cast, not after: folding it into an already-narrowed dtype
            # rounds the product at that dtype's precision, and a low-strength LoRA in fp16 is
            # exactly where that shows. `up` is small, so the fp32 temporary costs nothing.
            scaled = self.up.to(dtype=torch.float32) * self.scale
            # These outlive the block that asked for them.
            with _branch.outside_compiled_memory():
                self._cast = (
                    self.down.to(device=device, dtype=dtype),
                    scaled.to(device=device, dtype=dtype),
                )
            self._cast_key = key
        return self._cast

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """`scale * up(down(x))` — the branch's contribution before any gating."""
        down, up = self.factors(x.device, x.dtype)
        return torch.nn.functional.linear(torch.nn.functional.linear(x, down), up)

    def energy(self) -> float:
        """`||scale * up @ down||²`."""
        return delta_energy(self.down, self.up, self.scale)


class KronBranch:
    """A LoKr delta for one Linear, `scale * kron(w1, w2)`, applied without building it.

    `w1: [out_l, in_m]`, `w2: [out_k, in_n]`. The input's last dimension splits as `[in_m, in_n]`,
    `w2` maps the inner half, `w1` the outer, and the result flattens as `[out_l, out_k]` — the row
    order of `torch.kron`. Cast and scaled lazily, as `FactoredBranch` is.
    """

    __slots__ = ("_cast", "_cast_key", "scale", "w1", "w2")

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, scale: float):
        self.w1 = w1
        self.w2 = w2
        self.scale = scale
        self._cast_key = None
        self._cast = None

    def factors(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        if self._cast_key != key:
            scaled = self.w1.to(dtype=torch.float32) * self.scale
            with _branch.outside_compiled_memory():
                self._cast = (
                    scaled.to(device=device, dtype=dtype),
                    self.w2.to(device=device, dtype=dtype),
                )
            self._cast_key = key
        return self._cast

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        w1, w2 = self.factors(x.device, x.dtype)
        grouped = x.reshape(*x.shape[:-1], w1.shape[1], w2.shape[1])
        inner = torch.nn.functional.linear(grouped, w2).transpose(-1, -2)
        outer = torch.nn.functional.linear(inner, w1).transpose(-1, -2)
        return outer.reshape(*x.shape[:-1], -1)

    def energy(self) -> float:
        """`||scale * kron(w1, w2)||²`, which is `scale² ||w1||² ||w2||²`."""
        w1, w2 = self.w1.to(torch.float32), self.w2.to(torch.float32)
        return float(self.scale**2 * w1.pow(2).sum() * w2.pow(2).sum())


class FusedBranch:
    """Several branches writing into disjoint slices of one fused output (Z-Image's `attention.qkv`).

    Lumina fuses q, k and v into one Linear; a LoRA's `to_q`/`to_k`/`to_v` each target one
    `(offset)` slice of it, so three branches land on the same module. They are kept together here:
    `delta` assembles a full-width `[batch, seq, out]` output with each inner branch in its slice,
    zero elsewhere, so the gate — one per token for the whole projection — multiplies it exactly as
    for an unfused Linear. Strength is already folded into each inner branch, so this has no scale.
    """

    __slots__ = ("out", "parts")

    def __init__(self, out: int):
        self.out = out
        #: `(inner, start, length)` per slice.
        self.parts: list[tuple[object, int, int]] = []

    def add(self, inner, offset: tuple[int, int, int]) -> None:
        _, start, length = offset
        self.parts.append((inner, start, length))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for inner, start, length in self.parts:
            piece = inner.delta(x)
            if out is None:
                out = piece.new_zeros(*piece.shape[:-1], self.out)
            out[..., start : start + length] = piece
        return out


def module_path(weight_key: str) -> str:
    """`diffusion_model.blocks.0.attn.wq.weight` -> `diffusion_model.blocks.0.attn.wq`."""
    return weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key


def _factored(adapter) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """`(down, up, alpha)` if this adapter is a plain LoRA, else None.

    `mid` (Tucker/locon) and `reshape` (padded output) both change what the product means, and a
    `dora_scale` needs the merged weight — all three are reported as unroutable rather than
    approximated, because a wrong delta that still renders looks like a tuning problem.
    """
    if not isinstance(adapter, LoRAAdapter):
        return None
    up, down, stored_alpha, mid, dora_scale, reshape = adapter.weights
    if mid is not None or reshape is not None or dora_scale is not None:
        return None
    rank = down.shape[0]
    alpha = (stored_alpha / rank) if stored_alpha is not None else 1.0
    return down, up, float(alpha)


def _kron(adapter) -> KronBranch | None:
    """A LoKr adapter as a `KronBranch` at strength 1, else None.

    The low-rank halves (`w1_a @ w1_b`, `w2_a @ w2_b`) are multiplied out here: each is a fraction
    of the layer's size. `alpha` divides by the rank of the last decomposed half, exactly as core's
    merge does. Tucker (`t2`) and DoRA are left to a stock loader, like the formats above.
    """
    if not isinstance(adapter, LoKrAdapter):
        return None
    w1, w2, stored_alpha, w1_a, w1_b, w2_a, w2_b, t2, dora_scale = adapter.weights
    if t2 is not None or dora_scale is not None:
        return None
    rank = None
    if w1 is None:
        if w1_a is None or w1_b is None:
            return None
        w1, rank = w1_a.float() @ w1_b.float(), w1_b.shape[0]
    if w2 is None:
        if w2_a is None or w2_b is None:
            return None
        w2, rank = w2_a.float() @ w2_b.float(), w2_b.shape[0]
    if w1.ndim != 2 or w2.ndim != 2:
        return None
    alpha = float(stored_alpha) / rank if stored_alpha is not None and rank else 1.0
    return KronBranch(w1, w2, alpha)


def branch_of(adapter):
    """The adapter as a branch at strength 1 (`FactoredBranch` or `KronBranch`), or None."""
    factored = _factored(adapter)
    if factored is not None:
        down, up, alpha = factored
        return FactoredBranch(down, up, alpha)
    return _kron(adapter)


def delta_energy(down: torch.Tensor, up: torch.Tensor, alpha: float) -> float:
    """`||alpha * up @ down||²`, from the rank-sized Gram matrices instead of the full product."""
    up32, down32 = up.to(torch.float32), down.to(torch.float32)
    return float(alpha**2 * ((up32.T @ up32) * (down32 @ down32.T)).sum())


class RegionLora(NamedTuple):
    """One LoRA on one region, loaded."""

    region: int
    name: str
    strength: float
    #: `{module_path: branch}` on the routed Linears.
    branches: dict
    #: `{module_path: branch}` on the text side (`text_marker`), for the region's prompt copy.
    text_branches: dict


def load_region_loras(model, entries, wanted, family: str, text_marker=None, text=True):
    """`[RegionLora]` — every LoRA on every region, in region order, on the Linears `wanted(path)`
    accepts. With a `text_marker`, the Linears it names are kept apart as text-side branches (left
    out, and said so, when `text` is off).

    One entry per LoRA, not per region, because a region may carry several. Each LoRA's own strength
    is applied when it LOADS rather than to the region's gate: the gate is per REGION and several
    LoRAs share it. A LoRA several regions carry at the same strength loads once and its branches
    are shared, so the hooks compute it once per layer for all of them (see `_branch.by_module`).
    """
    out: list[RegionLora] = []
    cache: dict[tuple[str, float], tuple[dict, dict]] = {}
    for index, region in enumerate(entries):
        for spec in region.get("loras") or []:
            name = spec.get("lora_name")
            if not name:
                continue
            strength = float(spec.get("strength", 1.0))
            key = (name, strength)
            if key not in cache:
                cache[key] = load_branches(model, name, strength, wanted, family, text_marker, text)
            out.append(RegionLora(index, name, strength, *cache[key]))
    return out


def load_branches(model, lora_name, strength, wanted, family, text_marker=None, text=True):
    """`({module_path: branch}, {module_path: text branch})` — every routable Linear this LoRA
    touches on `model`, the ModelPatcher the branches will attach to. Its key map is ComfyUI's own
    LoRA Loader's, so a LoRA for another architecture fails here rather than at sample time.

    Diagnostics go to the log **here**, as each LoRA finishes, rather than being held for the caller:
    held to the end they arrive after the part they would have explained, and one LoRA raising
    discards every note collected before it.
    """
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    state = comfy.lora_convert.convert_lora(state)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    try:
        state, split = _qkv.for_model(state, key_map)
    except _qkv.CannotSplit as e:
        split = 0
        logging.warning("Nynxz region LoRA %s: fused q/k/v left as is (%s)", lora_name, e)
    if split:
        logging.info(
            "Nynxz region LoRA %s: split %d fused qkv weights into q, k, v", lora_name, split
        )
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)

    branches: dict = {}
    text_branches: dict = {}
    off_target = 0
    text_side = 0
    #: How much of the LoRA's plain-LoRA weight sits on the text side vs the routed blocks.
    text_energy = 0.0
    block_energy = 0.0
    #: Adapter kinds that do not factor, and how many layers each covered. Reported by NAME rather
    #: than as a guessed list: "LoHa" and "a padded reshape" want different answers from you, and
    #: only the loader knows which one it saw.
    unsupported: dict[str, int] = {}

    for weight_key, patch in loaded.items():
        offset = None
        if isinstance(weight_key, tuple):
            weight_key, offset = weight_key[0], weight_key[1]
        target = module_path(weight_key)
        branch = branch_of(patch)
        if text_marker and text_marker in target:
            if branch is not None:
                text_energy += branch.energy()
            if not text:
                text_side += 1
            elif branch is None:
                kind = patch_kind(patch)
                unsupported[kind] = unsupported.get(kind, 0) + 1
            else:
                branch.scale *= strength
                text_branches[target] = branch
            continue
        if not wanted(target):
            off_target += 1
            continue
        if branch is None:
            kind = patch_kind(patch)
            unsupported[kind] = unsupported.get(kind, 0) + 1
            continue
        block_energy += branch.energy()
        branch.scale *= strength
        if offset is None:
            branches[target] = branch
            continue
        out = _out_features(model, target)
        if out is None:
            continue
        fused = branches.get(target)
        if not isinstance(fused, FusedBranch):
            fused = branches[target] = FusedBranch(out)
        fused.add(branch, offset)

    if not branches and not text_branches:
        raise ValueError(
            _nothing_loaded(lora_name, family, loaded, off_target, text_side, unsupported)
        )
    if unsupported:
        logging.warning(
            "Nynxz Regions: %s — %d layer(s) are NOT applied (%s). Those formats do not factor "
            "into two small matrices, so they cannot become a gated side branch. A stock LoraLoader "
            "merges them whole and correctly; it just cannot confine them to a region.",
            lora_name,
            sum(unsupported.values()),
            _kinds(unsupported),
        )
    if off_target:
        logging.info(
            "Nynxz Regions: %s — %d layer(s) outside the routed Linears (modulation, embedders) are "
            "not applied per region.",
            lora_name,
            off_target,
        )
    if text_side:
        total = text_energy + block_energy
        share = f"{100 * text_energy / total:.0f}%" if total > 0 else "an unknown share"
        logging.warning(
            "Nynxz Regions: %s — skipped %d text-encoder (`%s`) layers, holding %s of its weight, "
            "because the per-region prompt is off. The more weight they hold, the weaker this LoRA "
            "is here than under a normal LoRA loader.",
            lora_name,
            text_side,
            text_marker.rstrip("."),
            share,
        )
    return branches, text_branches


def _out_features(model, path: str) -> int | None:
    try:
        return comfy.utils.get_attr(model.model, path).weight.shape[0]
    except (AttributeError, IndexError):
        return None


def _kinds(unsupported: dict[str, int]) -> str:
    return ", ".join(f"{kind} x{count}" for kind, count in sorted(unsupported.items()))


def patch_kind(patch) -> str:
    """What `load_lora` handed back, by name — `LoRAAdapter`, `LoHaAdapter`, `diff`, ..."""
    if isinstance(patch, tuple):
        return str(patch[0]) if len(patch) == 2 and isinstance(patch[0], str) else "diff"
    return getattr(patch, "name", None) or type(patch).__name__


def _nothing_loaded(lora_name, family, loaded, off_target, text_side, unsupported) -> str:
    """Say which of the several reasons actually applied: a LoRA that matched no model keys at all
    and one whose every layer was skipped want opposite fixes."""
    if not loaded:
        return (
            f"`{lora_name}` matched no weights in this {family} model at all — the same key map "
            "ComfyUI's own LoRA Loader builds. So it is a LoRA for a different architecture, not a "
            "routing problem here."
        )
    parts = []
    if off_target:
        parts.append(f"{off_target} on layers these nodes do not route")
    if text_side:
        parts.append(f"{text_side} on the text encoder, skipped with the per-region prompt off")
    if unsupported:
        parts.append(
            f"{sum(unsupported.values())} in adapter formats that do not factor into two small "
            f"matrices and so cannot be a gated branch ({_kinds(unsupported)}) — load those with a "
            "stock LoraLoader, which merges them whole but cannot confine them to a region"
        )
    return (
        f"`{lora_name}` matched {len(loaded)} weight(s) in this {family} model but none could "
        "become a region branch: " + "; ".join(parts) + "."
    )
