"""Qwen3-VL as an LLM, not just a text encoder.

ComfyUI 0.28+ ships a native autoregressive path for Qwen3-VL (`BaseGenerate.generate` in
`comfy/text_encoders/llama.py`, wrapped by `Qwen3VLClipModel.generate`), and the 4B/8B configs load
the LM head. So the same CLIP the Fusion nodes use as an *encoder* can also *generate text* — no
llama.cpp, no transformers, no second model in VRAM.

Wire the same CLIP you feed `Text Encode (Fusion)`: captioning a reference and then fusing it are
the same model doing two jobs, which is the whole reason this sits next to that group.
"""
