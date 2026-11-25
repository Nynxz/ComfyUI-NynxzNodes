#!/usr/bin/env python3
"""`apply_regions` end to end on every model family, against ComfyUI's own Krea 2, Z-Image (NextDiT)
and MOSS-Image DiTs at a tiny size with random weights, on CPU, with real LoRA files.

Run with ComfyUI's Python: python scripts/check_region_families.py /path/to/ComfyUI

Each family is driven the way the sampler drives it: through a ModelPatcher, its wrappers merged
into the options, `patch_model` around the forwards. The same behaviour is asserted on all three.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

torch = extension = model_patcher = ops = folder_paths = safetensors = None
apply = settings_mod = families = krea = zimage = moss = None
LORAS: dict[str, Path] = {}
SAMPLE = None


def _setup(comfy_path: str) -> None:
    global torch, extension, model_patcher, ops, folder_paths, safetensors, SAMPLE
    global apply, settings_mod, families, krea, zimage, moss
    sys.path.insert(0, comfy_path)
    argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.cli_args

    sys.argv = argv[:1]
    import comfy.model_patcher as model_patcher_mod
    import comfy.ops as ops_mod
    import comfy.patcher_extension as extension_mod
    import folder_paths as folder_paths_mod
    import safetensors.torch as safetensors_mod
    import torch as torch_mod
    from _harness import load

    torch, extension, model_patcher, ops = torch_mod, extension_mod, model_patcher_mod, ops_mod
    folder_paths, safetensors = folder_paths_mod, safetensors_mod
    folder_paths.get_full_path_or_raise = lambda folder, name: str(LORAS[name])
    apply = load("regions._apply")
    settings_mod = load("regions._settings")
    families = load("regions._families")
    krea = load("regions._krea")
    zimage = load("regions._zimage")
    moss = load("regions._moss")
    SAMPLE = torch.tensor([1.0, 0.7, 0.4, 0.1, 0.0])


def _randomise(model):
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2, generator=generator)
    return model.eval()


class Tiny:
    """One family's model at a tiny size, and how the sampler calls it."""

    name = ""
    latent = (2, 4, 8, 8)
    text_dim = 32

    def model(self):
        raise NotImplementedError

    def routable(self, path: str) -> bool:
        raise NotImplementedError

    def forward(self, model, x, t, context, options):
        raise NotImplementedError


class Krea(Tiny):
    name = "Krea 2"

    def model(self):
        from comfy.ldm.krea2.model import SingleStreamDiT

        return SingleStreamDiT(
            features=64,
            tdim=32,
            txtdim=16,
            heads=2,
            kvheads=2,
            multiplier=2,
            layers=4,
            patch=2,
            channels=4,
            txtlayers=2,
            txtheads=2,
            txtkvheads=2,
            dtype=torch.float32,
            device="cpu",
            operations=ops.disable_weight_init,
        )

    def routable(self, path):
        return krea.routable(path) or krea.TEXT_MARKER in path

    def forward(self, model, x, t, context, options):
        return model(x, t, context, None, None, options)


class ZImage(Tiny):
    name = "Z-Image"
    text_dim = 24

    def model(self):
        from comfy.ldm.lumina.model import NextDiT

        return NextDiT(
            patch_size=2,
            in_channels=4,
            dim=64,
            n_layers=3,
            n_refiner_layers=1,
            n_heads=2,
            multiple_of=16,
            cap_feat_dim=24,
            axes_dims=(8, 12, 12),
            axes_lens=(1, 64, 64),
            pad_tokens_multiple=4,
            dtype=torch.float32,
            device="cpu",
            operations=ops.disable_weight_init,
        )

    def routable(self, path):
        return zimage._wanted(path)

    def forward(self, model, x, t, context, options):
        return model(x, t, context, context.shape[1], None, transformer_options=options)


class Moss(Tiny):
    name = "MOSS-Image"
    latent = (2, 8, 4, 4)

    def model(self):
        from comfy.ldm.moss_image.model import MOSSImage

        return MOSSImage(
            dim=64,
            ffn_dim=96,
            num_heads=2,
            num_layers=2,
            text_dim=32,
            latent_channels=8,
            dtype=torch.float32,
            device="cpu",
            operations=ops.disable_weight_init,
        )

    def routable(self, path):
        return moss._routable(path)

    def forward(self, model, x, t, context, options):
        return model(x, t, context, options)


def mask(where: str):
    out = torch.zeros((8, 8))
    if where == "all":
        out[:] = 1.0
    elif where == "left":
        out[:, :4] = 1.0
    return out


def region(where, loras=(), conditioning=None):
    out = {
        "mask": mask(where),
        "fit": "stretch",
        "strength": 1.0,
        "denoise": 1.0,
        "loras": [{"lora_name": name, "strength": 1.0} for name in loras],
    }
    if conditioning is not None:
        out["conditioning"] = conditioning
    return out


#: The schedule off and no release, so a region's LoRA reaches nothing it doesn't claim.
STILL = {"track": 0.0, "preheat": 1.0, "release": 0.0, "smooth": 0}


class Family:
    """The checks every family must pass. Subclassed once per family below."""

    tiny: Tiny

    @classmethod
    def setUpClass(cls):
        cls.model = _randomise(cls.tiny.model())
        cls.lora = cls.write_lora(cls.tiny.name)

    @classmethod
    def write_lora(cls, name: str) -> str:
        generator = torch.Generator().manual_seed(1)
        state = {}
        for path, module in cls.model.named_modules():
            full = f"diffusion_model.{path}"
            if isinstance(module, torch.nn.Linear) and cls.tiny.routable(full):
                out, inp = module.weight.shape
                state[f"{full}.lora_down.weight"] = torch.randn((4, inp), generator=generator) * 0.3
                state[f"{full}.lora_up.weight"] = torch.randn((out, 4), generator=generator) * 0.3
                state[f"{full}.alpha"] = torch.tensor(4.0)
        file = Path(tempfile.mkdtemp()) / f"{name}.safetensors"
        safetensors.save_file(state, str(file))
        LORAS[name] = file
        return name

    def patcher(self):
        holder = torch.nn.Module()
        holder.diffusion_model = self.model
        holder.model_options = {}
        holder.model_config = types.SimpleNamespace(unet_config={})
        return model_patcher.ModelPatcher(holder, torch.device("cpu"), torch.device("cpu"))

    def context(self, seed: int, length: int = 5, batch: int = 2):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn((batch, length, self.tiny.text_dim), generator=generator) * 2.0

    def sample(self, patcher, context=None):
        """Three steps of a cond + uncond batch, as the sampler would run them."""
        context = self.context(10) if context is None else context
        options = dict(patcher.model_options.get("transformer_options", {}))
        extension.merge_nested_dicts(
            options.setdefault("wrappers", {}), patcher.wrappers, copy_dict1=False
        )
        x = torch.randn(self.tiny.latent, generator=torch.Generator().manual_seed(7))
        outs = []
        patcher.patch_model()
        try:
            for sigma in SAMPLE[:3]:
                step = {
                    **options,
                    "sigmas": sigma.reshape(1),
                    "sample_sigmas": SAMPLE,
                    "cond_or_uncond": [0, 1],
                }
                t = torch.full((x.shape[0],), float(sigma))
                with torch.no_grad():
                    outs.append(self.tiny.forward(self.model, x, t, context, step))
        finally:
            patcher.unpatch_model()
        return torch.stack(outs)

    def regioned(self, regions, **settings):
        settings = settings_mod.RegionSettings(**{**STILL, **settings})
        return apply.apply_regions(self.patcher(), regions, settings)

    def assert_same(self, out, expected, other):
        """`out` matches `expected` to within a thousandth of how far `other` is from it: routing
        leaves what a region doesn't claim a weight of 1e-4, never exactly zero."""
        gap = float((expected - other).abs().max())
        self.assertGreater(gap, 0.05)
        self.assertLess(float((out - expected).abs().max()), gap / 1000)

    def test_the_family_is_recognised(self):
        self.assertEqual(families.family_of(self.patcher(), "Regions Apply").name, self.tiny.name)

    def test_a_lora_over_the_canvas_changes_the_image(self):
        plain = self.sample(self.patcher())
        out = self.sample(self.regioned([region("all", [self.lora])]))
        self.assertGreater(float((out - plain).abs().max()), 0.05)

    def test_a_lora_on_a_region_claiming_nothing_changes_nothing(self):
        plain = self.sample(self.patcher())
        everywhere = self.sample(self.regioned([region("all", [self.lora])]))
        nowhere = self.sample(self.regioned([region("none", [self.lora])]))
        self.assert_same(nowhere, plain, everywhere)

    def test_a_prompt_on_a_region_claiming_nothing_changes_nothing(self):
        own = self.context(11, length=4, batch=1)
        plain = self.sample(self.patcher())
        everywhere = self.sample(self.regioned([region("all", conditioning=own)]))
        nowhere = self.sample(self.regioned([region("none", conditioning=own)]))
        self.assert_same(nowhere, plain, everywhere)

    def test_a_region_prompt_changes_its_half_most(self):
        own = self.context(12, length=4, batch=1)
        plain = self.sample(self.patcher())
        out = self.sample(self.regioned([region("left", conditioning=own)]))
        changed = (out - plain).abs()[:, 0].mean(dim=(0, 1))
        half = changed.shape[-1] // 2
        self.assertGreater(float(changed[:, :half].mean()), float(changed[:, half:].mean()))

    def test_isolation_runs_and_changes_the_image(self):
        regions = [region("left", [self.lora]), region("none", [self.lora])]
        regions[1]["mask"] = 1.0 - regions[0]["mask"]
        quiet = self.sample(self.regioned(regions))
        isolated = self.sample(self.regioned(regions, isolation=2.0))
        self.assertGreater(float((isolated - quiet).abs().max()), 1e-3)

    def test_unpatching_leaves_the_model_as_it_was(self):
        before = self.sample(self.patcher())
        self.sample(self.regioned([region("all", [self.lora])], isolation=1.0))
        torch.testing.assert_close(self.sample(self.patcher()), before)

    def test_regions_twice_are_refused(self):
        settings = settings_mod.RegionSettings(**STILL)
        once = apply.apply_regions(self.patcher(), [region("all", [self.lora])], settings)
        with self.assertRaisesRegex(ValueError, "already has regions"):
            apply.apply_regions(once, [region("all", [self.lora])], settings)


class KreaFamily(Family, unittest.TestCase):
    tiny = Krea()

    def test_the_layout_lock_runs(self):
        out = self.sample(
            self.regioned([region("left", [self.lora])], layout_lock=0.5, preheat=1.0)
        )
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_tracking_runs(self):
        out = self.sample(self.regioned([region("left", [self.lora])], track=0.6, refine=2))
        self.assertTrue(bool(torch.isfinite(out).all()))


class ZImageFamily(Family, unittest.TestCase):
    tiny = ZImage()


class ZImageWide(ZImage):
    """A canvas whose token count differs from the text block's, unlike the square one."""

    latent = (2, 4, 8, 12)


class ZImageWideFamily(Family, unittest.TestCase):
    tiny = ZImageWide()


class MossFamily(Family, unittest.TestCase):
    tiny = Moss()


class Unknown(unittest.TestCase):
    def test_another_model_is_refused_naming_every_family(self):
        holder = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=object()))
        with self.assertRaisesRegex(ValueError, "Z-Image, MOSS-Image, Krea 2"):
            families.family_of(holder, "Regions Apply")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path or not (Path(path) / "comfy").is_dir():
        sys.exit(
            "Pass a ComfyUI checkout: python scripts/check_region_families.py /path/to/ComfyUI"
        )
    _setup(path)
    unittest.main(argv=sys.argv[:1], verbosity=2)
