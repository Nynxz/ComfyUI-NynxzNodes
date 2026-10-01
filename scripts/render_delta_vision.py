#!/usr/bin/env python3
"""Render test: does δ-Vision make a better fused conditioning than the one Fusion Encode builds?

The probe (probe_delta_vision.py) measures hidden states; this renders them. Every variant below
shares the SAME visual block — the vanilla blend Fusion Encode already makes — and the same seed,
so the only thing that changes between tiles is the prompt half of the conditioning (the tokens
after the image), except `δ full`, which swaps the visual block too.

    A only / B only    single-source encodes, for reference
    prompt A           what Fusion Encode does today: the prompt half is source 0's reading
    prompt B           the same, from source 1 — how much does the prompt half matter at all?
    lerp 0.5           prompt half halfway between the two vanilla readings (no δ)
    δ-t lerp           vanilla readings interpolated per tap by where the δ-fused reading sits on
                       the δA→δB line — δ supplies one number per tap, never a hidden state
    δ hybrid           vanA + (δfused − δA)
    δ raw              the δ-fused pass's prompt half as-is
    δ full             the whole δ-fused conditioning, visual block included

Run with the ComfyUI venv:

    <ComfyUI>/.venv/bin/python scripts/render_delta_vision.py --comfy <ComfyUI> \
        --te qwen3vl_4b_bf16.safetensors --model krea2_turbo_bf16.safetensors \
        --vae wan_2.1_vae.safetensors --adapter qwen_embedding_adapter_step2000.safetensors \
        --images A B [--seeds 1 2] --out <dir>
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PACK = Path(__file__).resolve().parents[1]
KREA2_TAPS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
HIDDEN = 2560


def _load_pack(comfy: str):
    """Import the pack the way check_nodes.py does, so the fusion engine is the real one."""
    os.chdir(comfy)
    sys.path.insert(0, comfy)
    import utils.install_util  # noqa: F401  (claims `utils` before comfy/utils.py can)

    spec = importlib.util.spec_from_file_location(
        "nynxz_pack", PACK / "__init__.py", submodule_search_locations=[str(PACK)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nynxz_pack"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("nynxz_pack.nodes.fusion._fusion"), importlib.import_module(
        "nynxz_pack.nodes.fusion._delta_vision"
    )


def _image(path):
    pil = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).unsqueeze(0)


def _taps(cond):
    """`[1, seq, 12*H]` → `[seq, 12, H]` float32."""
    return cond[0].float().reshape(cond.shape[1], len(KREA2_TAPS), HIDDEN)


def _untaps(taps, like):
    return taps.reshape(1, taps.shape[0], -1).to(like.dtype)


def _along(point, a, b):
    """Per-tap projection coefficient of `point` on the a→b segment (all tokens at once)."""
    d = b - a
    return ((point - a) * d).sum(dim=(0, 2)) / (d * d).sum(dim=(0, 2)).clamp_min(1e-9)


def _grid(tiles, labels, size=384, columns=5):
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * size, rows * (size + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (i % columns) * size, (i // columns) * (size + 24)
        sheet.paste(tile.resize((size, size), Image.LANCZOS), (x, y + 24))
        draw.text((x + 4, y + 5), label, fill="black")
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfy", required=True)
    parser.add_argument("--te", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--adapter", default=None, help="δ-Vision adapter (not needed for --steer)")
    parser.add_argument("--images", nargs=2, required=True, metavar=("A", "B"))
    parser.add_argument("--prompt", default="", help="Fusion Encode prompt (your workflows use '')")
    parser.add_argument("--visual-size", type=int, default=512)
    parser.add_argument("--size", type=int, default=1024, help="output side, px")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--captions",
        action="store_true",
        help="instead of prompt-half variants: fused visual block + a caption as the prompt "
        "(none / δ-fused caption / caption of A / caption of B)",
    )
    parser.add_argument(
        "--steer",
        action="store_true",
        help="source-caption steering: fused visual block + plain captions of each source, mixed "
        "by concatenation or by a strength-weighted prediction mix (no δ involved)",
    )
    parser.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 0.75], help="--steer: A's share")
    parser.add_argument("--question", default="Describe this image in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    paths = [str(Path(p).resolve()) for p in args.images]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    comfy_root = str(Path(args.comfy).resolve())
    sys.argv = [sys.argv[0]]  # comfy.cli_args parses argv at import
    fusion, dv = _load_pack(comfy_root)
    import comfy.sd
    import comfy.utils
    import nodes

    clip = comfy.sd.load_clip([args.te], clip_type=comfy.sd.CLIPType.KREA2)
    adapter = dv.DeltaVisionAdapter.load(args.adapter) if args.adapter else None
    if adapter is None and not args.steer:
        parser.error("--adapter is required unless --steer")
    settings = fusion.FusionSettings(visual_size=args.visual_size)
    sources = [_image(p) for p in paths]

    # Exactly Fusion Encode's token path (no VAE → no reference latents; the semantic path only).
    profile = fusion.detect_profile(clip)
    prepared = fusion._prepare_sources(sources, settings, None, profile)
    tokens = fusion._tokenize_sources(clip, args.prompt, prepared.rgb, profile, keep_vision=True)
    conds = fusion._encode_sources(clip, tokens, prepared.rgb, ("render", args.prompt))
    gh, gw = prepared.grid_height, prepared.grid_width
    fused = fusion._fuse_conditionings(conds, tokens, gh, gw, settings)
    base, meta = fused[0]
    start, end = meta["nynxz_visual_span"]

    # The same blend, done at E (projector output) instead of at every tap, for the δ passes.
    captured = []
    for source_tokens in tokens:
        with dv.capture_visual(clip) as records:
            captured.append(dv.encode_capturing(clip, source_tokens, records)["E"])
    weights = fusion._compute_weights(2, gh, gw, settings, None, captured[0].device)
    fused_e = fusion._blend(torch.stack(captured, dim=1)[None], weights.to(captured[0].dtype), settings.preserve_norm)[0]

    def delta(visual):
        if adapter is None:
            return None
        with dv.delta_vision_pass(clip, adapter, visual=visual):
            return clip.encode_from_tokens_scheduled(tokens[0])[0][0]

    if adapter is not None:
        d_a, d_b, d_f = delta(captured[0]), delta(captured[1]), delta(fused_e)
        van_a, van_b = _taps(conds[0][0][0])[end:], _taps(conds[1][0][0])[end:]
        da, db, df = (_taps(c)[end:] for c in (d_a, d_b, d_f))
        t = _along(df, da, db)
        print(f"grid {gh}x{gw}, visual span {start}:{end}, prompt half {base.shape[1] - end} tokens")
        print("δ-fused t per tap: " + " ".join(f"{v:.2f}" for v in t.tolist()))

        def with_prompt(taps):
            cond = base.clone()
            cond[:, end:] = _untaps(taps, base)
            return cond

        variants = {
            "A only": (conds[0][0][0], conds[0][0][1]),
            "B only": (conds[1][0][0], conds[1][0][1]),
            "prompt A (today)": (base, meta),
            "prompt B": (with_prompt(van_b), meta),
            "lerp 0.5": (with_prompt(0.5 * (van_a + van_b)), meta),
            "δ-t lerp": (with_prompt(van_a + t[None, :, None] * (van_b - van_a)), meta),
            "δ hybrid": (with_prompt(van_a + (df - da)), meta),
            "δ raw": (with_prompt(df), meta),
            "δ full": (d_f.to(base.dtype), meta),
        }

    captions = None
    if args.captions:
        # Generation reads a blend correctly under δ — the one thing δ was validated on. Tokenize
        # the question with the same prepared pixels so E's grid matches the captured blend.
        def caption(image, visual=None):
            toks = clip.tokenize(args.question, images=[image])
            if visual is None:
                ids = clip.generate(toks, do_sample=False, max_length=args.max_tokens)
            else:
                with dv.delta_vision_pass(clip, adapter, visual=visual):
                    ids = clip.generate(toks, do_sample=False, max_length=args.max_tokens)
            return clip.decode(ids).strip().replace("\n", " ")

        captions = {
            "no prompt": "",
            "caption δ fused": caption(prepared.rgb[0], fused_e),
            "caption A": caption(prepared.rgb[0]),
            "caption B": caption(prepared.rgb[1]),
        }
        variants = {}
        for label, text in captions.items():
            print(f"  {label:<16} {text}")
            toks = fusion._tokenize_sources(clip, text, prepared.rgb, profile, keep_vision=True)
            encoded = fusion._encode_sources(clip, toks, prepared.rgb, ("render", text))
            cond, cond_meta = fusion._fuse_conditionings(encoded, toks, gh, gw, settings)[0]
            variants[label] = (cond, cond_meta)
        t = torch.zeros(len(KREA2_TAPS))

    if args.steer:
        # Plain captions (the real model, no δ), one per source, then two ways to mix them over
        # the SAME fused visual block. A caption is short enough to read the whole thing; the
        # sampler-side mix is the knob: Comfy averages the conds' predictions weighted by strength.
        def caption(image):
            toks = clip.tokenize(args.question, images=[image])
            ids = clip.generate(toks, do_sample=False, max_length=args.max_tokens)
            return clip.decode(ids).strip().replace("\n", " ")

        def encode(text):
            toks = fusion._tokenize_sources(clip, text, prepared.rgb, profile, keep_vision=True)
            encoded = fusion._encode_sources(clip, toks, prepared.rgb, ("render", text))
            return fusion._fuse_conditionings(encoded, toks, gh, gw, settings)[0]

        captions = {"A": caption(prepared.rgb[0]), "B": caption(prepared.rgb[1])}
        for key, text in captions.items():
            print(f"  caption {key}: {text}")
        none, cap_a, cap_b = encode(""), encode(captions["A"]), encode(captions["B"])
        joined = encode(captions["A"] + " " + captions["B"])
        variants = {"no prompt": none, "caption A": cap_a, "caption B": cap_b, "A+B joined": joined}
        for w in args.weights:
            variants[f"mix A {w:.2f}"] = [
                [cap_a[0], {**cap_a[1], "strength": w}],
                [cap_b[0], {**cap_b[1], "strength": 1.0 - w}],
            ]
        t = torch.zeros(len(KREA2_TAPS))

    model = comfy.sd.load_diffusion_model(args.model)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
    latent = fusion.empty_latent(args.size, args.size, vae)

    # Sample everything, then drop the DiT before decoding: bf16 Krea 2 is ~26 GB, and the Wan VAE
    # decode will not fit beside it on a 32 GB card.
    import comfy.model_management

    latents = {}
    with torch.inference_mode():
        for seed in args.seeds:
            for label, entry in variants.items():
                positive = entry if isinstance(entry[0], list) else [[entry[0], entry[1].copy()]]
                (samples,) = nodes.common_ksampler(
                    model, seed, args.steps, 1.0, args.sampler, args.scheduler, positive, positive, latent
                )
                latents[seed, label] = samples
                print(f"  seed {seed}  {label}")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        tiles = {}
        for (seed, label), samples in latents.items():
            image = nodes.VAEDecode().decode(vae, samples)[0]
            pil = Image.fromarray((image[0].clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8))
            slug = label.split(" (")[0].replace(" ", "_").replace(".", "").replace("δ", "d")
            pil.save(out / f"s{seed}_{slug}.png")
            tiles.setdefault(seed, []).append(pil)

    labels = list(variants)
    sources_row = [Image.open(p).convert("RGB") for p in paths]
    for seed, row in tiles.items():
        _grid(sources_row + row, ["source A", "source B"] + labels, columns=6 if len(labels) > 4 else len(labels) + 2).save(
            out / f"grid_s{seed}.png"
        )
    (out / "run.json").write_text(json.dumps({"images": paths, "prompt": args.prompt, "captions": captions, "t": t.tolist(), "seeds": args.seeds}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
