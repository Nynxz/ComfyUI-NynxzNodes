"""Turning a LoRA into a per-Linear branch that a per-token gate can sit in front of.

`comfy.lora.load_lora` already solves the part that is genuinely hard and boring — mapping a dozen
LoRA key dialects (kohya, diffusers, ai-toolkit, transformers, ...) onto model weight keys. So this
uses it, and then takes the road not taken: instead of handing the result to `add_patches`, which
folds the delta into the weight, it keeps the delta available as a *side branch* so a gate can
multiply it per token.

**Only patches that factor are routed, and that limit is deliberate.**

`FactoredBranch` keeps `up` and `down` apart, so the branch is two rank-r matmuls — about 2% of
step time at r=32. The scale convention is copied from `comfy/weight_adapter/lora.py` rather than
guessed::

    alpha = (stored_alpha / rank) if stored_alpha is not None else 1.0
    W += strength * alpha * (up @ down)          up: [out, rank]   down: [rank, in]

LoHa, LoKr, OFT, BOFT, GLoRA, DoRA and the `diff` / `set` patches `load_lora` also emits do not
factor into two small matrices. They are skipped and named, not approximated.

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

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths
import torch
from comfy.weight_adapter import LoRAAdapter

#: Substrings that pick out the attention projections in Krea 2's `SingleStreamBlock`. Identity is
#: mostly written into attention, but most character LoRAs put real weight into the MLP layers too,
#: so both are routed. Attention-only was a knob once and is not one now: with the MLP layers
#: unrouted they are simply not applied, identity comes out weak, and the natural fix — more
#: strength — buys the bleeding straight back. The saving was never worth that trade.
_ATTN_MARKERS = (".attn.wq", ".attn.wk", ".attn.wv", ".attn.wo", ".attn.gate")
_MLP_MARKER = ".mlp."
#: Krea 2's 12-layer Qwen3-VL aggregator. Its blocks are named exactly like the DiT's — `attn.wq`,
#: `mlp.*` — so the markers above match them, but it runs over the TEXT sequence, which has no
#: spatial extent for a region gate to index. Its layers are kept apart, as text-side branches that
#: make each region's own copy of the prompt (see `_text`); with that off they are skipped, and
#: `load_branches` says how much of the LoRA that was.
_TEXT_MARKER = "txtfusion."


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

    @property
    def rank(self) -> int:
        return self.down.shape[0]

    def factors(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        if self._cast_key != key:
            # Scale in fp32 before the cast, not after: folding it into an already-narrowed dtype
            # rounds the product at that dtype's precision, and a low-strength LoRA in fp16 is
            # exactly where that shows. `up` is small, so the fp32 temporary costs nothing.
            scaled = self.up.to(dtype=torch.float32) * self.scale
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


def module_path(weight_key: str) -> str:
    """`diffusion_model.blocks.0.attn.wq.weight` -> `diffusion_model.blocks.0.attn.wq`."""
    return weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key


def wanted(path: str) -> bool:
    """Whether a module path is one these nodes route. See `_TEXT_MARKER` for the exclusion."""
    if _TEXT_MARKER in path:
        return False
    return _MLP_MARKER in path or any(marker in path for marker in _ATTN_MARKERS)


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


def delta_energy(down: torch.Tensor, up: torch.Tensor, alpha: float) -> float:
    """`||alpha * up @ down||²`, from the rank-sized Gram matrices instead of the full product."""
    up32, down32 = up.to(torch.float32), down.to(torch.float32)
    return float(alpha**2 * ((up32.T @ up32) * (down32 @ down32.T)).sum())


def load_branches(
    model, lora_name: str, strength: float = 1.0, text: bool = True
) -> tuple[dict[str, FactoredBranch], dict[str, FactoredBranch]]:
    """Every routable Linear this LoRA touches, as `{module_path: branch}`: the blocks', and the
    `txtfusion` ones for the region's copy of the prompt (empty when `text` is off).

    `model` is the ModelPatcher the branches will attach to; it supplies the key map, so a LoRA that
    does not match the loaded architecture fails here rather than at sample time.

    Diagnostics go to the log **here**, as each LoRA finishes, rather than being returned for the
    caller to emit once every LoRA on every region has loaded. Held to the end they arrive after the
    part they would have explained, and one LoRA raising discards every note collected before it.
    """
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    state = comfy.lora_convert.convert_lora(state)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)

    branches: dict[str, FactoredBranch] = {}
    text_branches: dict[str, FactoredBranch] = {}
    off_target = 0
    text_side = 0
    #: How much of the LoRA's plain-LoRA weight sits on the text encoder vs the routed blocks.
    text_energy = 0.0
    block_energy = 0.0
    #: Adapter kinds that do not factor, and how many layers each covered. Reported by NAME rather
    #: than as a guessed list: "LoHa" and "a padded reshape" want different answers from you, and
    #: only the loader knows which one it saw.
    unsupported: dict[str, int] = {}

    for weight_key, patch in loaded.items():
        target = module_path(weight_key)
        factored = _factored(patch)
        if _TEXT_MARKER in target:
            if factored is not None:
                text_energy += delta_energy(*factored)
            if not text:
                text_side += 1
            elif factored is None:
                kind = _kind(patch)
                unsupported[kind] = unsupported.get(kind, 0) + 1
            else:
                down, up, alpha = factored
                text_branches[target] = FactoredBranch(down, up, alpha * strength)
            continue
        if not wanted(target):
            off_target += 1
            continue

        if factored is None:
            kind = _kind(patch)
            unsupported[kind] = unsupported.get(kind, 0) + 1
            continue
        down, up, alpha = factored
        block_energy += delta_energy(down, up, alpha)
        branches[target] = FactoredBranch(down, up, alpha * strength)

    if not branches and not text_branches:
        raise ValueError(_nothing_loaded(lora_name, loaded, off_target, text_side, unsupported))
    if unsupported:
        detail = ", ".join(f"{kind} x{count}" for kind, count in sorted(unsupported.items()))
        logging.warning(
            "Nynxz Regions: %s — %d layer(s) are NOT applied (%s). Those formats do not factor "
            "into two small matrices, so they cannot become a gated side branch. A stock LoraLoader "
            "merges them whole and correctly; it just cannot confine them to a region.",
            lora_name,
            sum(unsupported.values()),
            detail,
        )
    if text_side:
        total = text_energy + block_energy
        share = f"{100 * text_energy / total:.0f}%" if total > 0 else "an unknown share"
        logging.warning(
            "Nynxz Regions: %s — skipped %d text-encoder (`txtfusion`) layers, holding %s of its "
            "weight, because the per-region prompt is off. The more weight they hold, the weaker "
            "this LoRA is here than under a normal LoRA loader.",
            lora_name,
            text_side,
            share,
        )
    return branches, text_branches


def _kind(patch) -> str:
    """What `load_lora` handed back, by name — `LoRAAdapter`, `LoHaAdapter`, `diff`, ..."""
    if isinstance(patch, tuple):
        return str(patch[0]) if len(patch) == 2 and isinstance(patch[0], str) else "diff"
    return getattr(patch, "name", None) or type(patch).__name__


def _nothing_loaded(lora_name, loaded, off_target, text_side, unsupported) -> str:
    """Say which of the several reasons actually applied.

    Worth the words: an earlier version reported only `off_target`, so a LoRA that matched no model
    keys at all and one whose every layer was skipped both read as "0 such keys were seen", which
    points at the wrong half of the problem.
    """
    if not loaded:
        return (
            f"`{lora_name}` matched no weights in this model at all — `comfy.lora.load_lora` "
            "returned nothing, and that is the same key map ComfyUI's own LoRA Loader builds. So "
            "it is a LoRA for a different architecture, not a routing problem here."
        )
    parts = []
    if off_target:
        parts.append(f"{off_target} on layers these nodes do not route")
    if text_side:
        parts.append(f"{text_side} on the text encoder, skipped with the per-region prompt off")
    if unsupported:
        kinds = ", ".join(f"{k} x{v}" for k, v in sorted(unsupported.items()))
        parts.append(
            f"{sum(unsupported.values())} in adapter formats that do not factor into two small "
            f"matrices and so cannot be a gated branch ({kinds}) — load those with a stock "
            "LoraLoader, which merges them whole but cannot confine them to a region"
        )
    return (
        f"`{lora_name}` matched {len(loaded)} weight(s) in this model but none could become a "
        "region branch: " + "; ".join(parts) + "."
    )
