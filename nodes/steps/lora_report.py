"""LoRA Report — how the LoRAs on a model were actually applied.

Put it after the LoRA loader(s). It reads the model's own patch list and reports: which parts of
which blocks are patched, by what kind of adapter and strength, how many patches land on a slice
of a fused weight, whether any are bypass injections instead, and — for a few sample blocks — how
far the LoRA moves q, k and v (the change's size relative to the weight). Run the same LoRA on a
fused and a split checkpoint and those numbers should match; if a part is missing on one, the
LoRA didn't reach it there. Pick the LoRA file too, and it lists the file's keys this model can't
place — what the loader drops with "lora key not loaded".
"""

from __future__ import annotations

import logging

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io, ui

from .. import _categories
from .._base import NynxzNode
from ..lora import _qkv
from . import _lora_report as report

NONE = "(none)"
PREFIX = "diffusion_model."


def _blocks_in(model) -> int:
    keys = model.model.diffusion_model.state_dict().keys()
    numbers = [b for b, _ in (report.target(PREFIX + k, None) for k in keys) if b is not None]
    return max(numbers) + 1 if numbers else 0


def _weight_kind(model) -> str:
    sd = model.model.diffusion_model.state_dict()
    in_blocks = {k: v for k, v in sd.items() if report.target(PREFIX + k, None)[0] is not None}
    fused = any(k.endswith("qkv.weight") for k in in_blocks)
    sample = next(
        (v for k, v in in_blocks.items() if k.endswith((".q.weight", "qkv.weight"))), None
    )
    if sample is None:
        return f"{'fused' if fused else 'split'} q/k/v"
    kind = type(sample).__name__
    layout = getattr(sample, "_layout_type", None) or getattr(sample, "layout_type", None)
    quant = "" if kind == "Tensor" else f" ({kind} {layout})" if layout else f" ({kind})"
    shape = "fused qkv" if fused else "split q/k/v"
    return f"{shape} · weights {sample.dtype}{quant}"


def _change(model, key: str) -> dict[str, float] | str:
    """How far the patches move `key`, relative to the weight — per third when it's fused."""
    entry = model.get_key_patches(key).get(key)
    if not entry or len(entry) < 2:
        return "not patched"
    weight, convert = entry[0]
    try:
        base = convert(
            comfy.model_management.cast_to_device(weight, "cpu", torch.float32, copy=True),
            inplace=True,
        )
        after = comfy.lora.calculate_weight(
            entry[1:], base.clone(), key, intermediate_dtype=torch.float32
        )
    except Exception as e:  # noqa: BLE001 - a weight we can't read is reported, not fatal
        return f"couldn't measure ({e})"
    delta = after - base
    if key.endswith("qkv.weight"):
        third = base.shape[0] // 3
        slices = {p: slice(i * third, (i + 1) * third) for i, p in enumerate(report.QKV)}
    else:
        slices = {"": slice(None)}
    return {p: float(delta[s].norm() / base[s].norm().clamp_min(1e-12)) for p, s in slices.items()}


def _samples(model, blocks: list[int]) -> list[str]:
    lines = []
    keys = model.model.diffusion_model.state_dict().keys()
    for block in blocks:
        found = []
        for k in keys:
            b, part = report.target(PREFIX + k, None)
            if b == block and part.split(".")[-1] in ("q", "k", "v", "qkv"):
                found.append((part, PREFIX + k))
        cells = []
        for part, key in sorted(found):
            change = _change(model, key)
            if isinstance(change, str):
                cells.append(f"{part.split('.')[-1]}: {change}")
            else:
                cells.extend(f"{(p or part.split('.')[-1])}: {v:.5f}" for p, v in change.items())
        lines.append(f"  block {block:2d}  " + ("   ".join(cells) or "no q/k/v weights"))
    return lines


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.missing: list[str] = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("lora key not loaded:"):
            self.missing.append(message.split(":", 1)[1].strip())


def _dropped(lora: dict, key_map: dict) -> list[str]:
    """The LoRA's keys `load_lora` reports as not loaded — collected whatever the log level is."""
    handler = _Collect()
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(min(level or logging.WARNING, logging.WARNING))
    try:
        comfy.lora.load_lora(lora, key_map, log_missing=True)
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
    return handler.missing


def _placement(model, lora_name: str) -> list[str]:
    """What core's loader and the Nynxz loaders each do with the LoRA file on this model."""
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(path, safe_load=True))
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    core = _dropped(lora, key_map)
    lines = [
        f"{lora_name} with core's Load LoRA: "
        + (
            "every key has a place on this model."
            if not core
            else f"{len(core)} keys have no place here (dropped):"
        )
    ]
    lines.extend(f"  {k}" for k in core[:20])
    if len(core) > 20:
        lines.append(f"  … and {len(core) - 20} more")
    try:
        fitted, split = _qkv.for_model(lora, key_map)
    except _qkv.CannotSplit as e:
        lines.append(f"With the Nynxz LoRA loaders: fused q/k/v can't be split exactly ({e}).")
        return lines
    if split:
        left = _dropped(fitted, key_map)
        lines.append(
            f"With the Nynxz LoRA loaders: {split} fused qkv weights are split into q, k, v, and "
            + ("then every key has a place." if not left else f"{len(left)} keys still have none.")
        )
    return lines


class NynxzLoraReport(NynxzNode):
    CATEGORY = _categories.DIAGNOSTICS

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Lora.Report",
            display_name="LoRA Report",
            description=(
                "After the LoRA loader: which weights the LoRA patched, with what adapter, onto "
                "which slice of a fused q/k/v weight, and how far it moved q, k and v in a few "
                "blocks. Compare a fused and a split checkpoint with the same LoRA — the numbers "
                "should match."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The model after the LoRA loader(s)."),
                io.String.Input(
                    "sample_blocks",
                    default="0, 21, 41",
                    tooltip="Blocks to measure q/k/v change in.",
                ),
                io.Combo.Input(
                    "lora_name",
                    options=[NONE, *folder_paths.get_filename_list("loras")],
                    default=NONE,
                    tooltip="Also check this LoRA file: which of its keys this model can't place.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="The model, unchanged."),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, sample_blocks="0, 21, 41", lora_name=NONE) -> io.NodeOutput:
        blocks = _blocks_in(model)
        summary = report.summarize(model.patches)
        lines = [f"Model: {blocks} blocks, {_weight_kind(model)}"]
        if not model.patches:
            lines.append("No weight patches on this model.")
        else:
            total = sum(summary["kinds"].values())
            lines.append(
                f"Patches: {total} on {len(model.patches)} weights, {summary['sliced']} onto a "
                "slice of a fused weight"
            )
            lines.append(
                "  by adapter: "
                + ", ".join(f"{k}: {n}" for k, n in sorted(summary["kinds"].items()))
            )
            lines.append("Parts patched:")
            lines.extend(report.coverage(summary, blocks))
        if model.injections:
            lines.append(
                "Bypass injections (applied at run time, not as weight patches): "
                + ", ".join(f"{k} ({len(v)})" for k, v in model.injections.items())
            )
        wanted = [int(n) for n in sample_blocks.replace(",", " ").split() if n.isdigit()]
        wanted = [n for n in wanted if n < blocks]
        if wanted and model.patches:
            lines.append("q/k/v change relative to the weight (same LoRA → same numbers):")
            lines.extend(_samples(model, wanted))
        if lora_name != NONE:
            lines.extend(_placement(model, lora_name))
        text = "\n".join(lines)
        logging.info("Nynxz LoRA Report\n%s", text)
        return io.NodeOutput(model, text, ui=ui.PreviewText(text))
