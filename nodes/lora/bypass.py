"""Load LoRA (Bypass, Low Memory) — a LoRA run beside the model's weights without the stock
bypass's memory cost (see `_bypass`). For long video sequences, where ComfyUI's own
"Load LoRA (Bypass)" runs out of memory in the widest layers."""

from __future__ import annotations

import logging

import folder_paths
from comfy_api.latest import io

from . import _bypass
from ._base import LoraNode
from .stack import LOG_PREFIX


class NynxzLoraBypassLowMemory(LoraNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Lora.BypassLowMemory",
            display_name="Load LoRA (Bypass, Low Memory)",
            description=(
                "Runs a LoRA beside the model's weights instead of merging it in, like ComfyUI's "
                "Load LoRA (Bypass), but adds it into each layer's output in place. The stock "
                "bypass holds three copies of that output at once, which runs out of memory on "
                "long video sequences; this holds one."
            ),
            search_aliases=["lora bypass", "bypass lora", "low vram lora", "viggle", "h3 lora"],
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(model)
        import comfy.lora
        import comfy.lora_convert
        import comfy.utils
        import comfy.weight_adapter

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(path, safe_load=True))
        loaded = comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(model.model, {}))

        # As comfy.sd.load_bypass_lora_for_models does: adapters run in bypass, anything else
        # (bias and weight diffs) is patched in the ordinary way.
        adapters = {
            key: patch
            for key, patch in loaded.items()
            if isinstance(patch, comfy.weight_adapter.WeightAdapterBase)
        }
        other = {key: patch for key, patch in loaded.items() if key not in adapters}

        patched = model.clone()
        if other:
            patched.add_patches(other, strength)
        manager = comfy.weight_adapter.BypassInjectionManager()
        known = set(patched.model.state_dict().keys())
        lean = 0
        for key, adapter in adapters.items():
            if key not in known:
                continue
            lean += _bypass.make_lean(adapter)
            manager.add_adapter(key, adapter, strength=strength)
        injections = manager.create_injections(patched.model)
        if manager.get_hook_count() > 0:
            # Keyed per LoRA: one shared key would let a second bypass LoRA replace the first.
            patched.set_injections(f"nynxz_bypass_lora:{lora_name}:{strength}", injections)
        logging.info(
            "%s bypass LoRA %s: %d layers, %d in place, %d on the stock path",
            LOG_PREFIX,
            lora_name,
            manager.get_hook_count(),
            lean,
            manager.get_hook_count() - lean,
        )
        return io.NodeOutput(patched)
