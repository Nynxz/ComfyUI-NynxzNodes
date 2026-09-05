"""Turning a LoRA into a per-Linear branch that a per-token gate can sit in front of.

`comfy.lora.load_lora` already solves the part that is genuinely hard and boring — mapping a dozen
LoRA key dialects (kohya, diffusers, ai-toolkit, transformers, ...) onto model weight keys. So this
uses it, and then takes the road not taken: instead of handing the result to `add_patches`, which
folds the delta into the weight, it keeps the delta available as a *side branch* so a gate can
multiply it per token.

**Two branch shapes, because there are two kinds of patch and only one of them factors.**

`FactoredBranch` is the cheap path and covers plain LoRA: keep `up` and `down` apart and the
branch is two rank-r matmuls, about 2% of step time at r=32. The scale convention is copied from
`comfy/weight_adapter/lora.py` rather than guessed::

    alpha = (stored_alpha / rank) if stored_alpha is not None else 1.0
    W += strength * alpha * (up @ down)          up: [out, rank]   down: [rank, in]

`DenseBranch` covers everything else — LoHa, LoKr, OFT, BOFT, GLoRA, DoRA, and the `diff` / `set`
patches `load_lora` also emits. None of those factor into two small matrices: DoRA rescales by the
merged weight's norms, OFT rotates it, `diff` is a full delta to begin with. But every one of them
CAN be asked what weight it would have produced, so this asks::

    delta = comfy.lora.calculate_weight([patch], W.clone(), key) - W

and runs `x @ delta.T` as the branch. That is exact — at gate 1 it reproduces what the stock loader
merges, byte for byte — and it costs one dense `[out, in]` matrix per patched layer, which is the
memory an ordinary merge would have spent anyway. The trade is deliberate: these nodes are useless
if half your LoRAs will not load, and "supported but heavier" beats "unsupported".

Because `calculate_weight` is the same function `ModelPatcher` merges through, **anything ComfyUI's
own LoRA loader accepts loads here too.** Strength is applied at load rather than folded into the
gate afterwards: for a plain LoRA the two are identical, but DoRA and OFT are not linear in
strength, so the only faithful place to apply it is inside `calculate_weight`.
"""

from __future__ import annotations

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
#: spatial extent for a region gate to index. Applying a branch there could only mean "everywhere,
#: for every region", which is merging, so it is excluded by name rather than left to be caught by a
#: sequence-length mismatch further downstream. A LoRA that trains it loses those layers, and
#: `load_branches` says so instead of quietly dropping them.
_TEXT_MARKER = "txtfusion."


class FactoredBranch:
    """A plain LoRA's delta for one Linear, kept as its two factors instead of their product.

    Cast lazily and cached: the factors land on CPU in whatever dtype they were saved in, and the
    module they attach to may be offloaded, re-cast, or moved between runs. Re-casting per call
    would be a measurable tax at 28 blocks x 5 projections x N concepts, so the cast result is held
    until the target device/dtype actually changes.
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
            self._cast = (
                self.down.to(device=device, dtype=dtype),
                self.up.to(device=device, dtype=dtype),
            )
            self._cast_key = key
        return self._cast

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """`scale * up(down(x))` — the branch's contribution before any gating."""
        down, up = self.factors(x.device, x.dtype)
        return torch.nn.functional.linear(torch.nn.functional.linear(x, down), up) * self.scale


class DenseBranch:
    """A materialised `[out, in]` delta, for every patch shape that does not factor.

    The same lazy cast as `FactoredBranch`, and the same `delta(x)` contract, so the forward hook
    cannot tell them apart. `scale` is 1.0 by construction: strength was already applied inside
    `calculate_weight`, which is the only faithful place for it on a DoRA or an OFT.
    """

    __slots__ = ("_cast", "_cast_key", "scale", "weight")

    def __init__(self, weight: torch.Tensor):
        self.weight = weight  # [out, in]
        self.scale = 1.0
        self._cast_key = None
        self._cast = None

    def cast(self, device, dtype) -> torch.Tensor:
        key = (device, dtype)
        if self._cast_key != key:
            self._cast = self.weight.to(device=device, dtype=dtype)
            self._cast_key = key
        return self._cast

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.cast(x.device, x.dtype))


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
    `dora_scale` needs the merged weight — all three fall through to the dense path rather than
    being approximated, because a wrong delta that still renders looks like a tuning problem.
    """
    if not isinstance(adapter, LoRAAdapter):
        return None
    up, down, stored_alpha, mid, dora_scale, reshape = adapter.weights
    if mid is not None or reshape is not None or dora_scale is not None:
        return None
    rank = down.shape[0]
    alpha = (stored_alpha / rank) if stored_alpha is not None else 1.0
    return down, up, float(alpha)


#: Rank an un-factorable delta is truncated to. 32 is the rank most character LoRAs train at, so a
#: fallback branch costs what an ordinary one costs; the reported error says whether that was
#: enough for a given layer.
_FALLBACK_RANK = 32
#: Extra columns the randomized SVD oversamples by. Cheap, and it is what makes the top singular
#: vectors accurate rather than merely plausible.
_SVD_OVERSAMPLE = 8
#: Residual past which a fitted branch is dropped instead of applied.
#:
#: A rank-32 fit that leaves more than half the delta behind is not a weaker version of that layer,
#: it is a different one — mostly the directions the truncation kept for arithmetic reasons rather
#: than because the LoRA put weight there. Applying it produces something that renders, which is
#: the one failure mode there is no way to debug from the output. Not applying it produces a
#: visibly weaker identity and a line in the report saying exactly that.
_MAX_RESIDUAL = 0.5


def _dense_delta(base: torch.Tensor, patch, weight_key: str, strength: float):
    """`W' - W` for one patch, by asking ComfyUI to merge it exactly as `ModelPatcher` would.

    The patch tuple shape is `add_patches`'s: `(strength, value, strength_model, offset, function)`.

    **On the CPU, deliberately, and this is the whole difference between working and an OOM.** The
    base weight lives on the sampling device, so doing this arithmetic in place would put a full
    `[out, in]` float32 matrix in VRAM for every layer that cannot be factored — on a DiT whose MLPs
    are 4x the hidden size, two dozen of those is several GB, next to the model that has to fit
    beside them. Copying to CPU first costs one transfer per layer at load time and nothing at all
    per step, because what actually reaches the GPU is the truncated factors below.
    """
    reference = base.detach().to(device="cpu", dtype=torch.float32, copy=True)
    merged = comfy.lora.calculate_weight(
        [(strength, patch, 1.0, None, None)], reference.clone(), weight_key
    )
    if merged is None or merged.shape != reference.shape:
        return None
    return merged - reference


def _truncate(delta: torch.Tensor, rank: int = _FALLBACK_RANK):
    """`(down, up, relative_error)` — the best rank-`rank` approximation of a dense delta.

    Why approximate at all, when `_dense_delta` was exact. A dense branch is exact and unusable:
    it costs what the layer's own weights cost, in VRAM, for every step of the run. Truncating puts
    the fallback back on the same footing as an ordinary LoRA — two rank-r matmuls — at the price of
    an error this returns rather than hides, so the report can say how good the approximation
    actually was instead of claiming it was fine.

    Randomized SVD (`svd_lowrank`) rather than a full one: a full decomposition of a 3072x12288
    matrix takes minutes and computes thousands of singular vectors to throw all but 32 away.
    """
    rank = max(1, min(rank, *delta.shape))
    u, s, v = torch.svd_lowrank(delta, q=min(rank + _SVD_OVERSAMPLE, *delta.shape))
    u, s, v = u[:, :rank], s[:rank], v[:, :rank]
    up = (u * s).contiguous()  # [out, rank]
    down = v.transpose(0, 1).contiguous()  # [rank, in]
    total = delta.norm()
    error = float(((up @ down) - delta).norm() / total.clamp(min=1e-12)) if total > 0 else 0.0
    return down, up, error


def load_branches(model, lora_name: str, strength: float = 1.0):
    """Every routable Linear this LoRA touches, as `{module_path: branch}`, plus warnings.

    `model` is the ModelPatcher the branches will attach to: it supplies both the key map (so a
    LoRA that does not match the loaded architecture fails here rather than at sample time) and the
    base weights the fallback path subtracts from.
    """
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    state = comfy.lora_convert.convert_lora(state)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)

    weights = model.model.state_dict()
    branches: dict[str, FactoredBranch] = {}
    warnings: list[str] = []
    off_target = 0
    text_side = 0
    #: Adapter kinds that needed the fallback, and how well it approximated them. Reported by NAME
    #: rather than as a guessed list: "LoHa" and "a padded reshape" want different answers from
    #: you, and only the loader knows which one it saw.
    approximated: dict[str, list[float]] = {}
    dropped: dict[str, list[float]] = {}
    failed: dict[str, str] = {}

    for weight_key, patch in loaded.items():
        target = module_path(weight_key)
        if _TEXT_MARKER in target:
            text_side += 1
            continue
        if not wanted(target):
            off_target += 1
            continue

        kind = _kind(patch)
        factored = _factored(patch)
        if factored is not None:
            down, up, alpha = factored
            branches[target] = FactoredBranch(down, up, alpha * strength)
            continue

        # Everything else: LoHa, LoKr, OFT, BOFT, GLoRA, DoRA, a Tucker `mid`, a padded `reshape`,
        # and the `diff` / `set` patches `load_lora` emits for a full-weight LoRA. None of those
        # factor, so ask ComfyUI what it would have merged and fit a low-rank branch to that.
        base = weights.get(weight_key)
        if base is None:
            failed[kind] = f"no base weight at {weight_key}"
            continue
        delta = _dense_delta(base, patch, weight_key, strength)
        if delta is None or delta.ndim != 2:
            # A conv delta cannot be a Linear side branch. Krea 2's DiT is Linear throughout, so
            # reaching this means the LoRA was not trained for this model.
            failed[kind] = (
                f"delta was {'None' if delta is None else f'{delta.ndim}-D'}, not a matrix"
            )
            continue
        down, up, error = _truncate(delta)
        del delta
        if error > _MAX_RESIDUAL:
            dropped.setdefault(kind, []).append(error)
            continue
        branches[target] = FactoredBranch(down, up, 1.0)  # strength is already inside the delta
        approximated.setdefault(kind, []).append(error)

    if not branches:
        raise ValueError(_nothing_loaded(lora_name, loaded, off_target, text_side, failed, dropped))
    if dropped:
        detail = ", ".join(
            f"{kind} x{len(errs)} (best {min(errs):.0%} off)"
            for kind, errs in sorted(dropped.items())
        )
        warnings.append(
            f"{lora_name}: {sum(len(v) for v in dropped.values())} layer(s) were NOT applied — "
            f"{detail}. They do not factor, and a rank-{_FALLBACK_RANK} fit of their delta keeps "
            "too little of it to be that layer rather than noise. This LoRA will read weaker here "
            "than under the stock loader, which merges the delta whole."
        )
    if approximated:
        worst = max(max(v) for v in approximated.values())
        detail = ", ".join(
            f"{kind} x{len(errs)} (worst {max(errs):.1%})"
            for kind, errs in sorted(approximated.items())
        )
        warnings.append(
            f"{lora_name}: {sum(len(v) for v in approximated.values())} layer(s) do not factor and "
            f"were fitted to rank {_FALLBACK_RANK} instead — {detail}. "
            + (
                "Well within what the layer was doing, so this should be indistinguishable."
                if worst < 0.15
                else "That is a large residual: those layers will not reproduce what the stock "
                "loader merges. Treat this LoRA's identity here as approximate."
            )
        )
    if text_side:
        warnings.append(
            f"{lora_name}: skipped {text_side} text-encoder (`txtfusion`) layers — those run over "
            "the text sequence, which has no spatial extent to gate against, so a region cannot be "
            "expressed there. If this LoRA puts real weight into them its identity will be weaker "
            "here than under a normal LoRA loader."
        )
    if failed:
        detail = "; ".join(f"{kind}: {why}" for kind, why in sorted(failed.items()))
        warnings.append(
            f"{lora_name}: could not build a branch for some layers ({detail}) — those are not "
            "applied. Everything else from this LoRA is."
        )
    return branches, warnings


def _kind(patch) -> str:
    """What `load_lora` handed back, by name — `LoRAAdapter`, `LoHaAdapter`, `diff`, ..."""
    if isinstance(patch, tuple):
        return str(patch[0]) if len(patch) == 2 and isinstance(patch[0], str) else "diff"
    return getattr(patch, "name", None) or type(patch).__name__


def _nothing_loaded(lora_name, loaded, off_target, text_side, failed, dropped) -> str:
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
        parts.append(f"{text_side} on the text encoder, which has no spatial extent to gate")
    if dropped:
        kinds = ", ".join(f"{k} x{len(v)}" for k, v in sorted(dropped.items()))
        parts.append(
            f"{sum(len(v) for v in dropped.values())} that do not factor and lost too much of "
            f"their delta to a rank-{_FALLBACK_RANK} fit ({kinds})"
        )
    if failed:
        parts.append("; ".join(f"{k}: {v}" for k, v in sorted(failed.items())))
    return (
        f"`{lora_name}` matched {len(loaded)} weight(s) in this model but none could become a "
        "region branch: " + "; ".join(parts) + "."
    )
