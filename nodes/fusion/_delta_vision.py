"""δ-Vision visual memory for the Qwen3-VL text encoder — one language pass over a given visual block.

δ-Vision (Lei et al. 2026, arXiv 2609.34972) replaces the layer-by-layer evolution of visual tokens
with a per-layer low-rank residual MLP over the projector output `E`:

    M_l = E + SiLU(E @ D_l.T) @ U_l.T        (embedding adapter, one D/U pair per decoder layer)

Text tokens then read the visual tokens only through `LN_l(M_l)` → frozen K/V. The released
checkpoint was distilled against Qwen3-VL-4B-Instruct, which is the tokenizer key Krea 2 and
Mage-Flow report.

Why this matters for fusion: `M_l` is a function of each token's own `E` and nothing else — no
LM layer mixes visual tokens with each other. So a blend done at `E` reaches every depth
intact, and a single pass lets the prompt tokens read the *fused* image instead of whichever
source happened to be encoded first.

**How the pass is done without reimplementing attention.** comfy's decoder is left exactly as it
is; hooks pin the residual stream at the visual positions to the adapter's memory:

  * a pre-hook on the Qwen3-VL transformer writes `M_0` into the visual span of `embeds` and turns
    DeepStack off (δ-Vision's student runs without it);
  * a forward hook on layer `l` overwrites its output at the visual span with `M_{l+1}`.

So layer `l` always sees `M_l` there, its K/V at those positions are `LN_l(M_l)` projections, and
text queries attend to them causally with the original M-RoPE positions — δ-Vision's equation 9.
The visual tokens' own updates are computed and thrown away, which costs compute but never
changes a text output. Hidden-state taps (comfy records the stream *entering* a layer) read
`M_l` at the visual span, so the conditioning's visual block is the adapter's memory as well.

Only torch is imported, so this file loads outside ComfyUI too (the probe script does).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F


class DeltaVisionAdapter:
    """The 36 down/up projection pairs of a δ-Vision embedding adapter, stacked for one bmm."""

    def __init__(self, down: torch.Tensor, up: torch.Tensor, meta: dict | None = None):
        # down: [L, R, H], up: [L, H, R] — nn.Linear weight layout, as the checkpoint stores them.
        if down.ndim != 3 or up.ndim != 3 or down.shape[0] != up.shape[0]:
            raise ValueError(
                "δ-Vision adapter weights must be [layers, rank, hidden] / [layers, hidden, rank]."
            )
        self.down = down
        self.up = up
        self.meta = meta or {}

    @property
    def num_layers(self) -> int:
        return int(self.down.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.down.shape[2])

    @property
    def rank(self) -> int:
        return int(self.down.shape[1])

    @classmethod
    def load(cls, path: str | Path) -> DeltaVisionAdapter:
        """Load the released `.pt` or its `.safetensors` conversion (scripts/convert_delta_vision.py)."""
        if str(path).lower().endswith((".safetensors", ".sft")):
            state, config, step, base = _read_safetensors(path)
        else:
            # weights_only: the release is a plain dict of tensors and primitives, so there is no
            # reason to let a downloaded checkpoint run pickle code.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint.get("state_dict", checkpoint)
            config = checkpoint.get("adapter_config", {})
            step = checkpoint.get("global_step")
            base = Path((checkpoint.get("args") or {}).get("model_path", "")).name
        mode = config.get("output_mode", "embedding_adapter")
        if mode != "embedding_adapter":
            raise ValueError(
                f"Only the δ-Vision embedding adapter is supported; this checkpoint is '{mode}'."
            )
        count = sum(1 for key in state if key.startswith("visual_adapter_down."))
        if count == 0:
            raise ValueError(
                f"{path} has no visual_adapter_down.* weights — not a δ-Vision adapter."
            )
        down = torch.stack([state[f"visual_adapter_down.{i}.weight"] for i in range(count)])
        up = torch.stack([state[f"visual_adapter_up.{i}.weight"] for i in range(count)])
        return cls(down, up, {"path": str(path), "step": step, "base": base})

    def memories(self, embeddings: torch.Tensor, identity: bool = False) -> torch.Tensor:
        """`[N, H]` projector embeddings → `[L, N, H]` per-layer visual memory.

        Computed in float32: the delta is small next to `E`, and bf16 rounding of the sum would eat
        a visible share of it. `identity` skips the delta (every `M_l = E`) — the no-adapter control.
        """
        e = embeddings.float()
        if identity:
            return e.unsqueeze(0).expand(self.num_layers, -1, -1)
        down = self.down.to(e.device, torch.float32)
        up = self.up.to(e.device, torch.float32)
        delta = torch.bmm(F.silu(torch.einsum("nh,lrh->lnr", e, down)), up.transpose(1, 2))
        return e.unsqueeze(0) + delta


def _read_safetensors(path):
    """Tensors plus the metadata the converter stores: `(state, adapter_config, step, base_model)`.

    safetensors is imported here rather than at the top so the module still loads with torch alone.
    """
    import json

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        state = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118 - a safetensors file, not a dict
        meta = f.metadata() or {}
    config = json.loads(meta.get("delta_vision.adapter_config") or "{}")
    step = meta.get("delta_vision.global_step")
    return state, config, int(step) if step else None, meta.get("delta_vision.base_model", "")


def language_transformer(clip):
    """The Qwen3-VL module inside a CLIP: the one that builds image inputs and owns the decoder."""
    for module in clip.cond_stage_model.modules():
        if hasattr(module, "build_image_inputs") and hasattr(
            getattr(module, "model", None), "layers"
        ):
            return module
    raise ValueError("δ-Vision needs a Qwen3-VL text encoder; the wired CLIP has none.")


def check_compatible(transformer, adapter: DeltaVisionAdapter):
    layers = len(transformer.model.layers)
    hidden = int(transformer.model.config.hidden_size)
    if layers != adapter.num_layers or hidden != adapter.hidden_size:
        raise ValueError(
            f"This δ-Vision adapter is for a {adapter.num_layers}-layer, {adapter.hidden_size}-wide "
            f"language model (Qwen3-VL-4B); the wired encoder has {layers} layers, {hidden} wide. "
            "Krea 2 and Mage-Flow load the 4B; Qwen Image 2.1 and Ideogram 4 load the 8B, which has "
            "no released adapter."
        )


class _Captured(Exception):
    """Raised by the capture hook to stop the encode once the visual embeddings are in hand."""


@contextmanager
def capture_visual(clip):
    """Collect each encode's projector embeddings `E` without running the language model.

    Yields a list; every `clip.encode_from_tokens` inside the block appends a dict with `E`
    (`[N, H]`, float32, on the execution device), `positions` (visual indices in the full,
    untrimmed sequence) and `length` (that sequence's length) — then aborts. The pixels go through
    the exact preprocessing a real encode uses, so `E` is what the language model would have seen.
    """
    transformer = language_transformer(clip)
    records: list[dict] = []

    def hook(module, args, kwargs):
        embeds = kwargs["embeds"]
        _, mask, _ = module.build_image_inputs(embeds, kwargs.get("embeds_info", []))
        if mask is None:
            raise ValueError("δ-Vision capture: the encode carried no image.")
        records.append(
            {
                "E": embeds[0, mask[0]].float().clone(),
                "positions": mask[0].nonzero().flatten().cpu(),
                "length": int(embeds.shape[1]),
            }
        )
        raise _Captured

    handle = transformer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield records
    finally:
        handle.remove()


def encode_capturing(clip, tokens, records):
    """`clip.encode_from_tokens` under `capture_visual`: returns only after the capture hook fired."""
    try:
        clip.encode_from_tokens(tokens)
    except _Captured:
        return records[-1]
    raise RuntimeError("δ-Vision capture: the encode finished without reaching the language model.")


@contextmanager
def delta_vision_pass(clip, adapter: DeltaVisionAdapter, visual=None, identity: bool = False):
    """Every encode inside the block runs the language model on δ-Vision visual memory.

    `visual`, when given, replaces the projector embeddings before the adapter sees them: a
    `[N, H]` tensor, or a callable `E -> E'`. That is the fusion hook — pass the blended `E` and the
    whole stack, prompt included, reads the blend. The token layout must be the one `visual` was
    built for (same grid, same prompt), which the shape check enforces.
    """
    transformer = language_transformer(clip)
    check_compatible(transformer, adapter)
    layers = transformer.model.layers
    state: dict = {}

    def arm(embeds, mask):
        """Build the memory for one pass. Both entry points (encode, generate) land here."""
        if mask is None:
            raise ValueError("δ-Vision needs an image in the encode; this one has none.")
        span = mask[0]
        e = embeds[0, span]
        if visual is not None:
            e = visual(e) if callable(visual) else visual.to(e.device)
            if e.shape != embeds[0, span].shape:
                raise ValueError(
                    f"δ-Vision: replacement visual block is {tuple(e.shape)}, the encode's is "
                    f"{tuple(embeds[0, span].shape)} — they must share the visual grid."
                )
        state.update(
            span=span, length=int(embeds.shape[1]), memory=adapter.memories(e, identity=identity)
        )

    def before_encode(module, args, kwargs):
        embeds = kwargs["embeds"]
        position_ids, mask, _ = module.build_image_inputs(embeds, kwargs.get("embeds_info", []))
        arm(embeds, mask)
        # DeepStack off: δ-Vision's student runs without it, and the overwrite would erase it anyway.
        kwargs.update(position_ids=position_ids, visual_pos_masks=mask, deepstack_embeds=None)
        return args, kwargs

    # generate() is a plain method that calls the decoder's .forward() directly, so no module hook
    # sees its prefill; wrap it instead. Only the prefill carries the image — decode steps are one
    # token long and read the visual K/V the prefill left in the cache, which came from M_l.
    original_generate = transformer.generate

    def generate(embeds=None, *args, visual_pos_masks=None, deepstack_embeds=None, **kwargs):
        arm(embeds, visual_pos_masks)
        return original_generate(
            embeds, *args, visual_pos_masks=None, deepstack_embeds=None, **kwargs
        )

    def prefill(x):
        return state.get("memory") is not None and x.shape[1] == state["length"]

    def before_first(module, args, kwargs):
        x = kwargs["x"]
        if prefill(x):
            x = x.clone()
            x[:, state["span"]] = state["memory"][0].to(x.dtype)
            kwargs["x"] = x
        return args, kwargs

    def after(index):
        def hook(module, args, kwargs, output):
            x = output[0]
            if prefill(x) and index + 1 < state["memory"].shape[0]:
                x[:, state["span"]] = state["memory"][index + 1].to(x.dtype)
            return output

        return hook

    handles = [
        transformer.register_forward_pre_hook(before_encode, with_kwargs=True),
        layers[0].register_forward_pre_hook(before_first, with_kwargs=True),
    ]
    handles += [
        layer.register_forward_hook(after(i), with_kwargs=True) for i, layer in enumerate(layers)
    ]
    transformer.generate = generate
    try:
        yield
    finally:
        del transformer.generate  # drop the instance override; the class method shows through again
        for handle in handles:
            handle.remove()
        state.clear()
