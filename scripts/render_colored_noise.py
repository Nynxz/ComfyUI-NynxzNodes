#!/usr/bin/env python3
"""Render test: does Colored Noise make a turbo model's seeds more diverse, and at what cost?

For each prompt, every setting renders the SAME seeds (the white noise underneath is identical, see
nodes/noise/_colored.py), so a row differs from the white row only by the spectral tilt.

    <ComfyUI>/.venv/bin/python scripts/render_colored_noise.py --comfy <ComfyUI> \
        --te qwen3vl_4b_bf16.safetensors --model krea2_turbo_bf16.safetensors \
        --vae wan_2.1_vae.safetensors --out <dir> [--prompts "…" "…"] [--seeds 8]

Diversity, per setting, is the mean pairwise distance between that row's images:
    layout   RMS distance of 16x16 grayscale thumbnails — where light, dark and mass sit
    color    distance between mean RGB — palette spread across seeds
Higher = more varied. Quality has no number here; look at the grid.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PACK = Path(__file__).resolve().parents[1]
SETTINGS = [("white", 0.0, 0.0), ("α1 s.5", 1.0, 0.5), ("α1 s1", 1.0, 1.0), ("α2 s.5", 2.0, 0.5), ("α2 s1", 2.0, 1.0)]


def _load_pack(comfy: str):
    os.chdir(comfy)
    sys.path.insert(0, comfy)
    import utils.install_util  # noqa: F401  (claims `utils` before comfy/utils.py can)

    spec = importlib.util.spec_from_file_location(
        "nynxz_pack", PACK / "__init__.py", submodule_search_locations=[str(PACK)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nynxz_pack"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("nynxz_pack.nodes.noise.colored_noise")


def _diversity(images):
    gray = [np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), np.float32) / 255 for im in images]
    rgb = [np.asarray(im, np.float32).reshape(-1, 3).mean(0) / 255 for im in images]
    pairs = list(itertools.combinations(range(len(images)), 2))
    layout = np.mean([np.sqrt(((gray[i] - gray[j]) ** 2).mean()) for i, j in pairs])
    color = np.mean([np.linalg.norm(rgb[i] - rgb[j]) for i, j in pairs])
    return float(layout), float(color)


def _sheet(rows, labels, tile=224):
    columns = max(len(r) for r in rows)
    sheet = Image.new("RGB", (110 + columns * tile, len(rows) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (row, label) in enumerate(zip(rows, labels)):
        draw.text((4, r * tile + tile // 2 - 12), label, fill="black")
        for c, im in enumerate(row):
            sheet.paste(im.resize((tile, tile), Image.LANCZOS), (110 + c * tile, r * tile))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfy", required=True)
    parser.add_argument("--te", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["a portrait photo of a woman", "a cozy cabin in a snowy forest at dusk"],
    )
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]]
    node = _load_pack(str(Path(args.comfy).resolve()))
    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import comfy.utils
    import nodes

    clip = comfy.sd.load_clip([args.te], clip_type=comfy.sd.CLIPType.KREA2)
    conds = {p: clip.encode_from_tokens_scheduled(clip.tokenize(p)) for p in args.prompts}
    model = comfy.sd.load_diffusion_model(args.model)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
    latent = nodes.EmptyLatentImage().generate(args.size, args.size)[0]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
    latent = {**latent, "samples": latent_image}

    # Sample everything, then unload the DiT before decoding (bf16 Krea 2 + Wan VAE don't share 32 GB).
    samples = {}
    with torch.inference_mode():
        for prompt, cond in conds.items():
            for label, color, strength in SETTINGS:
                for seed in range(args.seeds):
                    noise = node.ColoredNoiseSource(seed, color, strength, True).generate_noise(latent)
                    samples[prompt, label, seed] = comfy.sample.sample(
                        model, noise, args.steps, 1.0, args.sampler, args.scheduler, cond, cond,
                        latent_image, seed=seed,
                    )
            print(f"sampled: {prompt}")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        images = {}
        for key, s in samples.items():
            pixels = nodes.VAEDecode().decode(vae, {"samples": s})[0]
            images[key] = Image.fromarray((pixels[0].clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8))

    report = {}
    for p_index, prompt in enumerate(args.prompts):
        rows, labels = [], []
        report[prompt] = {}
        print(f"\n{prompt}")
        print(f"  {'setting':<10} {'layout':>7} {'color':>7}")
        for label, _, _ in SETTINGS:
            row = [images[prompt, label, seed] for seed in range(args.seeds)]
            layout, color = _diversity(row)
            report[prompt][label] = {"layout": layout, "color": color}
            print(f"  {label:<10} {layout:7.4f} {color:7.4f}")
            rows.append(row)
            labels.append(f"{label}\nlay {layout:.3f}\ncol {color:.3f}")
            for seed, im in enumerate(row):
                im.save(out / f"p{p_index}_{label.replace(' ', '_').replace('.', '')}_s{seed}.png")
        _sheet(rows, labels).save(out / f"grid_p{p_index}.png")
    (out / "report.json").write_text(json.dumps({"prompts": args.prompts, "settings": SETTINGS, "diversity": report}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
