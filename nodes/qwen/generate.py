"""Qwen3-VL Describe — run the Qwen3-VL text encoder as an actual LLM: image + prompt -> text.

Uses ComfyUI's native generate path (no llama.cpp / transformers). Wire the *same* Qwen3-VL CLIP
you feed `Text Encode (Fusion)` — captioning a reference and fusing it are the same model doing two
jobs, so there is no second load. Useful on its own for captioning and prompt expansion.
"""

from comfy_api.latest import io

from .._lib.io_types import advanced
from ._base import QwenNode
from ._generate import run_generate


class NynxzQwenDescribe(QwenNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Qwen.Describe",
            display_name="Qwen3-VL Describe",
            description="Runs a Qwen3-VL text encoder generatively (ComfyUI native inference) — an "
            "image plus a prompt in, generated text out. Wire the same CLIP the Fusion encode uses.",
            inputs=[
                io.Clip.Input(
                    "clip", tooltip="A Qwen3-VL 4B/8B text encoder (as loaded for Qwen-Image)."
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="Describe this image in detail.",
                    tooltip="The instruction / question for the model.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional image to look at. Without it this is a plain text prompt.",
                ),
                advanced(
                    io.String.Input(
                        "system",
                        multiline=True,
                        default="You are a helpful assistant.",
                        tooltip="System prompt.",
                    )
                ),
                io.Int.Input(
                    "max_tokens",
                    default=256,
                    min=1,
                    max=4096,
                    tooltip="Maximum new tokens to generate. Generation also stops at the model's "
                    "end-of-turn token.",
                ),
                advanced(
                    io.Boolean.Input(
                        "do_sample",
                        default=True,
                        tooltip="Sample (on) vs greedy/deterministic (off). Off ignores temperature/top-k/top-p.",
                    )
                ),
                advanced(io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05)),
                advanced(io.Int.Input("top_k", default=50, min=0, max=200)),
                advanced(io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.01)),
                advanced(io.Float.Input("min_p", default=0.0, min=0.0, max=1.0, step=0.01)),
                advanced(
                    io.Float.Input("repetition_penalty", default=1.05, min=1.0, max=2.0, step=0.01)
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Sampling seed (used when do_sample is on).",
                ),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        image=None,
        system="You are a helpful assistant.",
        max_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        min_p=0.0,
        repetition_penalty=1.05,
        seed=0,
    ) -> io.NodeOutput:
        text = run_generate(
            clip,
            prompt=prompt,
            image=image,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            do_sample=do_sample,
        )
        return io.NodeOutput(text)
