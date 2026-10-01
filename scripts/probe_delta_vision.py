#!/usr/bin/env python3
"""Probe: is δ-Vision usable inside the fusion encode, and for which half of the conditioning?

Needs a real ComfyUI and the Krea 2 text encoder, so run it with the ComfyUI venv:

    <ComfyUI>/.venv/bin/python scripts/probe_delta_vision.py --comfy <ComfyUI> \
        --images a.png b.png [--te qwen3vl_4b_bf16.safetensors] [--adapter qwen_embedding_adapter_step2000.pt]

Everything is measured at Krea 2's 12 conditioning taps, on the two halves of the conditioning:

  visual  cos(M_l, V_l) — adapter memory vs the real evolved visual state. The adapter was only
          distilled to make TEXT outputs match, so nothing forces this high. High → the δ visual
          block can go to the DiT as-is; low → keep the vanilla visual blend (hybrid mode).
  text    cos(δ prompt tokens, vanilla prompt tokens) for the SAME image — does the δ pass read the
          image the way the real model does? `identity` is the control: M_l = E, no adapter. The
          adapter earns its place only where it beats identity.
  fusion  A and B blended at E (checkerboard, the fusion default; and a flat mean), one δ pass.
          `t` is where the fused prompt tokens sit on the A→B line (projection coefficient): 0 = A's
          reading, 1 = B's. Today's token fusion takes the prompt half verbatim from source 0, i.e.
          t = 0 by construction, whatever the blend.
  diff    the differential hybrid: text taps = vanA + (δX − δA). The δ pass's own offset from the
          real model cancels because both terms went through δ, so only the change the adapter
          reads between images is kept. Validated by X = B: does vanA + (δB − δA) land on vanB?
          Error is in units of |vanA − vanB| (target < 0.3; 1 = no better than not moving at all,
          which is what t = 0 gives). Identity (M_l = E) runs the same test as the control.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

FUSION = Path(__file__).resolve().parents[1] / "nodes" / "fusion"
KREA2_TAPS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
HIDDEN = 2560


def _load(name):
    """_delta_vision and _encoders import nothing from the pack, so load them by path."""
    spec = importlib.util.spec_from_file_location(f"probe_{name}", FUSION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


def _image(path, size):
    pil = Image.open(path).convert("RGB")
    side = min(pil.size)
    left, top = (pil.width - side) // 2, (pil.height - side) // 2
    pil = pil.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0).unsqueeze(0)


def _taps(cond):
    """Krea 2 conditioning `[1, seq, 12*2560]` → `[seq, 12, 2560]` float32."""
    return cond[0].float().reshape(cond.shape[1], len(KREA2_TAPS), HIDDEN)


def _cos(a, b):
    """Mean per-token cosine, per tap: `[n, taps, H]` pairs → `[taps]`."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(0)


def _norm_ratio(a, b):
    return (a.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).mean(0)


def _along(point, a, b):
    """Projection coefficient of `point` on the a→b segment, per tap, over all tokens at once."""
    direction = (b - a).flatten(0, 0)
    t = ((point - a) * direction).sum(dim=(0, 2)) / (direction * direction).sum(dim=(0, 2)).clamp_min(1e-9)
    return t


def _row(label, values, fmt="{:6.3f}"):
    return f"  {label:<26}" + " ".join(fmt.format(float(v)) for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfy", required=True, help="ComfyUI root (to import comfy from)")
    parser.add_argument("--images", nargs=2, required=True, metavar=("A", "B"))
    parser.add_argument("--te", default=None, help="text encoder; default <comfy>/models/text_encoders/qwen3vl_4b_bf16.safetensors")
    parser.add_argument("--adapter", default=None, help="default <comfy>/models/delta_vision/qwen_embedding_adapter_step2000.safetensors")
    parser.add_argument("--size", type=int, default=384, help="square visual grid side, multiple of 32 (fusion default 384)")
    parser.add_argument("--prompt", default="combine these into one image")
    parser.add_argument("--json", default=None, help="also write the numbers here")
    parser.add_argument("--describe", default="Describe this image in one sentence.", help="generation readout question ('' to skip)")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    comfy_root = Path(args.comfy).resolve()
    te = Path(args.te or comfy_root / "models/text_encoders/qwen3vl_4b_bf16.safetensors")
    adapter_path = Path(args.adapter or comfy_root / "models/delta_vision/qwen_embedding_adapter_step2000.safetensors")
    if args.size % 32:
        parser.error("--size must be a multiple of 32 so the grid is whole tokens")

    sys.argv = [sys.argv[0]]  # comfy.cli_args parses argv at import
    sys.path.insert(0, str(comfy_root))
    import comfy.sd

    dv = _load("_delta_vision")
    template = _load("_encoders").PROMPT_TEMPLATE

    print(f"TE       {te}\nadapter  {adapter_path}")
    clip = comfy.sd.load_clip([str(te)], clip_type=comfy.sd.CLIPType.KREA2)
    adapter = dv.DeltaVisionAdapter.load(adapter_path)
    print(f"         rank {adapter.rank}, {adapter.num_layers} layers, step {adapter.meta['step']}, base {adapter.meta['base']}")

    grid = args.size // 32
    images = {name: _image(path, args.size) for name, path in zip("AB", args.images)}
    tokens = {
        name: clip.tokenize(template.format(prompt=args.prompt), images=[img]) for name, img in images.items()
    }

    def encode(name, **delta):
        start = time.perf_counter()
        if delta:
            with dv.delta_vision_pass(clip, adapter, **delta):
                cond = clip.encode_from_tokens(tokens[name])
        else:
            cond = clip.encode_from_tokens(tokens[name])
        torch.cuda.synchronize()
        return _taps(cond), time.perf_counter() - start

    encode("A")  # warm-up: loads the model, so the timings below compare passes, not loading
    vanilla, delta, ident, captured = {}, {}, {}, {}
    for name in "AB":
        vanilla[name], t_vanilla = encode(name)
        delta[name], t_delta = encode(name, visual=None)
        ident[name], _ = encode(name, identity=True)
        with dv.capture_visual(clip) as records:
            captured[name] = dv.encode_capturing(clip, tokens[name], records)

    # Where the halves sit in the (template-trimmed) conditioning.
    seq = vanilla["A"].shape[0]
    offset = captured["A"]["length"] - seq
    vis = captured["A"]["positions"] - offset
    if not torch.equal(vis, captured["B"]["positions"] - offset):
        raise SystemExit("A and B tokenized to different layouts — give them the same --size")
    vis_end = int(vis[-1]) + 1
    text = torch.arange(vis_end, seq)
    print(f"\ngrid {grid}x{grid} = {len(vis)} visual tokens, {len(text)} prompt/tail tokens, taps {KREA2_TAPS}")

    header = f"  {'tap':<26}" + " ".join(f"{t:>6d}" for t in KREA2_TAPS)
    report: dict = {"taps": KREA2_TAPS, "grid": grid, "prompt": args.prompt, "images": args.images}

    print("\n── visual block: δ memory M_l vs real V_l ──")
    print(header)
    for name in "AB":
        c = _cos(delta[name][vis], vanilla[name][vis])
        r = _norm_ratio(delta[name][vis], vanilla[name][vis])
        ci = _cos(ident[name][vis], vanilla[name][vis])
        print(_row(f"{name} cos(M, V)", c))
        print(_row(f"{name} cos(E, V) [identity]", ci))
        print(_row(f"{name} |M|/|V|", r))
        report[f"visual_{name}"] = {"cos": c.tolist(), "cos_identity": ci.tolist(), "norm_ratio": r.tolist()}

    print("\n── prompt tokens: δ pass vs vanilla, same image ──")
    print(header)
    for name in "AB":
        c = _cos(delta[name][text], vanilla[name][text])
        ci = _cos(ident[name][text], vanilla[name][text])
        print(_row(f"{name} cos δ", c))
        print(_row(f"{name} cos identity", ci))
        report[f"text_{name}"] = {"cos": c.tolist(), "cos_identity": ci.tolist()}
    cross = _cos(vanilla["A"][text], vanilla["B"][text])
    print(_row("vanilla A vs B (scale)", cross))
    report["text_vanilla_AB"] = cross.tolist()

    # Cosine barely moves between unrelated images (a big shared component dominates), so measure
    # the error in units of what swapping the image does: 1.0 = as far off as reading the other image.
    print("\n── prompt tokens: error ÷ |vanilla A − vanilla B|  (lower is better, 1 = wrong image) ──")
    print(header)
    signal = (vanilla["A"][text] - vanilla["B"][text]).norm(dim=-1).mean(0)
    for name in "AB":
        err = (delta[name][text] - vanilla[name][text]).norm(dim=-1).mean(0) / signal
        err_i = (ident[name][text] - vanilla[name][text]).norm(dim=-1).mean(0) / signal
        print(_row(f"{name} δ", err))
        print(_row(f"{name} identity", err_i))
        report[f"text_{name}"].update(rel_err=err.tolist(), rel_err_identity=err_i.tolist())

    # Differential hybrid, validated on the one case with a ground truth: swap the image entirely.
    # recon = vanA + (δB − δA) should be vanB. (The A-anchored and B-anchored errors are the same
    # vector with opposite sign, so one number per tap covers both directions.)
    print("\n── differential: vanA + (δB − δA) vs vanB, error ÷ |vanA − vanB|  (target < 0.3) ──")
    print(header)
    van_step = vanilla["B"][text] - vanilla["A"][text]
    report["diff"] = {}
    for label, run in (("δ", delta), ("identity", ident)):
        step = run["B"][text] - run["A"][text]
        err = (step - van_step).norm(dim=-1).mean(0) / signal
        direction = _cos(step, van_step)
        scale = _norm_ratio(step, van_step)
        print(_row(f"{label} rel err", err))
        print(_row(f"{label} cos(step, van step)", direction))
        print(_row(f"{label} |step|/|van step|", scale))
        # Best single gain per tap: if the direction is right and only the length is off, a scalar
        # fixes it. err at the optimal gain is the floor that a per-tap scale could reach.
        gain = (step * van_step).sum(dim=(0, 2)) / (step * step).sum(dim=(0, 2)).clamp_min(1e-9)
        err_gain = (gain[None, :, None] * step - van_step).norm(dim=-1).mean(0) / signal
        print(_row(f"{label} best gain", gain))
        print(_row(f"{label} rel err @ best gain", err_gain))
        report["diff"][label] = {
            "rel_err": err.tolist(),
            "cos_step": direction.tolist(),
            "norm_ratio_step": scale.tolist(),
            "best_gain": gain.tolist(),
            "rel_err_best_gain": err_gain.tolist(),
        }

    # Fusion at E. Checkerboard with block 2 is the node's default pattern.
    ea, eb = captured["A"]["E"], captured["B"]["E"]
    rows = torch.arange(grid).repeat_interleave(grid)
    cols = torch.arange(grid).repeat(grid)
    pick_b = (((rows // 2) + (cols // 2)) % 2 == 1).to(ea.device).unsqueeze(1)
    blends = {"checker": torch.where(pick_b, eb, ea), "mean": 0.5 * (ea + eb)}

    print("\n── fusion: one δ pass over E-blended A+B; t = position of prompt tokens on A→B ──")
    print(header)
    for label, fused_e in blends.items():
        fused, t_fused = encode("A", visual=fused_e)
        t_delta_line = _along(fused[text], delta["A"][text], delta["B"][text])
        t_vanilla_line = _along(fused[text], vanilla["A"][text], vanilla["B"][text])
        print(_row(f"{label}: t on δA→δB", t_delta_line))
        print(_row(f"{label}: t on vanA→vanB", t_vanilla_line))
        hybrid = vanilla["A"][text] + (fused[text] - delta["A"][text])
        t_hybrid = _along(hybrid, vanilla["A"][text], vanilla["B"][text])
        print(_row(f"{label}: hybrid t on vanA→vanB", t_hybrid))
        print(_row(f"{label}: cos vs vanilla A", _cos(fused[text], vanilla["A"][text])))
        print(_row(f"{label}: cos vs vanilla B", _cos(fused[text], vanilla["B"][text])))
        report[f"fusion_{label}"] = {
            "t_delta": t_delta_line.tolist(),
            "t_vanilla": t_vanilla_line.tolist(),
            "t_hybrid": t_hybrid.tolist(),
            "cos_vanilla_A": _cos(fused[text], vanilla["A"][text]).tolist(),
            "cos_vanilla_B": _cos(fused[text], vanilla["B"][text]).tolist(),
        }
    print("  (current token fusion: prompt half = vanilla A, i.e. t = 0 on vanA→vanB at every tap)")

    if args.describe:
        # The readable test: ask the LM what it sees. Greedy, so runs are comparable.
        question = clip.tokenize  # bound below per image
        print(f"\n── describe: \"{args.describe}\" (greedy, {args.max_tokens} tokens) ──")
        report["describe"] = {}

        def describe(name, **delta_kwargs):
            toks = question(args.describe, images=[images[name]])
            if delta_kwargs:
                with dv.delta_vision_pass(clip, adapter, **delta_kwargs):
                    ids = clip.generate(toks, do_sample=False, max_length=args.max_tokens)
            else:
                ids = clip.generate(toks, do_sample=False, max_length=args.max_tokens)
            return clip.decode(ids).strip().replace("\n", " ")

        runs = [
            ("vanilla A", "A", {}),
            ("vanilla B", "B", {}),
            ("δ A", "A", {"visual": None}),
            ("δ B", "B", {"visual": None}),
            ("identity A (no adapter)", "A", {"identity": True}),
            ("δ checker(A,B)", "A", {"visual": blends["checker"]}),
            ("δ mean(A,B)", "A", {"visual": blends["mean"]}),
        ]
        for label, name, kw in runs:
            text_out = describe(name, **kw)
            report["describe"][label] = text_out
            print(f"  {label:<24} {text_out}")

    print(f"\ntiming (last B): vanilla {t_vanilla*1e3:.0f} ms, δ pass {t_delta*1e3:.0f} ms, fused δ {t_fused*1e3:.0f} ms")
    print("  hooks overwrite the visual stream but still compute it, so no speed-up is expected here")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
