#!/usr/bin/env python3
"""Checks for layout prompts — each mode writes exactly what its model was trained on.

Run: python scripts/check_layout_prompt.py
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest

from _harness import NODES, load

prompt = load("layout._prompt")


def box(**over):
    base = {"id": "a", "name": "Lamp", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "desc": "a lamp"}
    return {**base, **over}


def layout(mode, boxes, **over):
    return {
        "mode": mode,
        "width": 1024,
        "height": 768,
        "prompt": "A room <obj_1>",
        **over,
        "boxes": boxes,
    }


class Ideogram(unittest.TestCase):
    def test_caption_keys_follow_the_trained_order(self):
        text = prompt.compile_prompt(
            layout(
                "ideogram",
                [
                    box(),
                    box(
                        id="b",
                        kind="text",
                        text="OPEN",
                        desc="neon sign",
                        palette=["#f00", "00ff00"],
                    ),
                ],
                background="a dim bar",
                style={
                    "on": True,
                    "kind": "photo",
                    "medium": "photograph",
                    "aesthetics": "moody",
                    "lighting": "neon",
                    "photo": "35mm",
                },
            )
        )
        caption = json.loads(text)
        self.assertEqual(
            list(caption),
            ["high_level_description", "style_description", "compositional_deconstruction"],
        )
        self.assertEqual(
            list(caption["style_description"]), ["aesthetics", "lighting", "photo", "medium"]
        )
        text_el = caption["compositional_deconstruction"]["elements"][1]
        self.assertEqual(list(text_el), ["type", "bbox", "text", "desc", "color_palette"])
        self.assertEqual(text_el["color_palette"], ["#FF0000", "#00FF00"])
        self.assertEqual(
            caption["compositional_deconstruction"]["elements"][0]["bbox"], [200, 100, 600, 400]
        )
        self.assertNotIn(", ", text)
        self.assertNotIn('": ', text)

    def test_a_character_is_an_object_to_ideogram(self):
        caption = json.loads(prompt.compile_prompt(layout("ideogram", [box(kind="character")])))
        self.assertEqual(caption["compositional_deconstruction"]["elements"][0]["type"], "obj")

    def test_style_off_leaves_it_out(self):
        caption = json.loads(prompt.compile_prompt(layout("ideogram", [box()])))
        self.assertNotIn("style_description", caption)


class Klein(unittest.TestCase):
    def test_caption_then_table_like_core(self):
        text = prompt.compile_prompt(
            layout("klein", [box(), box(id="b", kind="text", text="OPEN", desc="neon sign")])
        )
        rows = [
            {"id": "obj_1", "bbox": [200, 100, 600, 400], "desc": "a lamp"},
            {"id": "obj_2", "bbox": [200, 100, 600, 400], "desc": 'neon sign "OPEN"'},
        ]
        self.assertEqual(text, "A room <obj_1> " + json.dumps(rows, ensure_ascii=False))

    def test_hidden_boxes_are_left_out_and_ids_stay_dense(self):
        text = prompt.compile_prompt(layout("klein", [box(hidden=True), box(id="b")]))
        self.assertIn('"obj_1"', text)
        self.assertNotIn('"obj_2"', text)

    def test_uncited_boxes_are_warned_about(self):
        warned = prompt.warnings(layout("klein", [box(), box(id="b")]))
        self.assertIn("Not cited in the prompt: <obj_2>.", warned)

    def test_no_boxes_is_just_the_prompt(self):
        self.assertEqual(prompt.compile_prompt(layout("klein", [])), "A room <obj_1>")


class Moss(unittest.TestCase):
    def test_boxes_are_centred_pixels_yxyx(self):
        text = prompt.compile_prompt(
            layout(
                "moss", [box(kind="character", tags=["1girl"])], meta=["poster"], canvas="single"
            )
        )
        value = json.loads(text)
        self.assertEqual(list(value), ["summary", "items", "bbox_coord_space", "bbox_format"])
        self.assertEqual(list(value["summary"]), ["meta", "canvas", "caption"])
        item = value["items"][0]
        self.assertEqual(item["label"], "character")
        self.assertEqual(list(item), ["label", "caption", "tags", "bbox"])
        # y 0.2*768=153.6 -> 153-384; x 0.1*1024=102.4 -> 102-512; y2 0.6*768=460.8 -> 461-384;
        # x2 0.4*1024=409.6 -> 410-512
        self.assertEqual(item["bbox"], [-231, -410, 77, -102])
        self.assertEqual(value["bbox_coord_space"], "abs_centralized")
        self.assertNotIn(", ", text)

    def test_text_items_carry_their_text(self):
        value = json.loads(
            prompt.compile_prompt(layout("moss", [box(kind="text", text="EXPLORE")]))
        )
        self.assertEqual(value["items"][0]["text"], "EXPLORE")

    def test_canvas_snaps_to_sixteen(self):
        parsed = prompt.parse({"mode": "moss", "width": 1000, "height": 7})
        self.assertEqual((parsed["width"], parsed["height"]), (992, 16))

    def test_canvas_respects_custom_snap_multiple(self):
        p8 = prompt.parse({"mode": "moss", "width": 1000, "height": 7, "settings": {"snap": 8}})
        self.assertEqual((p8["width"], p8["height"]), (1000, 8))
        p32 = prompt.parse(
            {"mode": "moss", "width": 1000, "height": 1000, "settings": {"snap": 32}}
        )
        self.assertEqual((p32["width"], p32["height"]), (992, 992))
        p64 = prompt.parse(
            {"mode": "moss", "width": 1000, "height": 1000, "settings": {"snap": 64}}
        )
        self.assertEqual((p64["width"], p64["height"]), (1024, 1024))


class Elements(unittest.TestCase):
    def test_elements_match_core_create_bounding_boxes(self):
        parsed = prompt.parse(layout("moss", [box(kind="text", text="HI", palette=["#abc"])]))
        self.assertEqual(
            prompt.elements(parsed),
            [
                {
                    "type": "text",
                    "bbox": [200, 100, 600, 400],
                    "text": "HI",
                    "desc": "a lamp",
                    "color_palette": ["#AABBCC"],
                }
            ],
        )

    def test_bounding_boxes_are_core_pixel_frames(self):
        parsed = prompt.parse(layout("moss", [box(), box(id="b", hidden=True)]))
        self.assertEqual(
            prompt.bounding_boxes(parsed), [[{"x": 102, "y": 154, "width": 307, "height": 307}]]
        )
        self.assertEqual(prompt.bounding_boxes(prompt.parse(layout("moss", []))), [])

    def test_anything_stored_parses(self):
        for raw in (None, [], "x", {"boxes": "no"}, {"boxes": [1, {"x": "nan"}]}):
            parsed = prompt.parse(raw)
            self.assertIn(parsed["mode"], prompt.MODES)
            prompt.compile_prompt(raw)


class RegionPrompt(unittest.TestCase):
    def parsed(self, *boxes, scene="A rainy street at night"):
        return prompt.parse(layout("moss", list(boxes), prompt=scene))

    def test_a_box_reads_the_scene_then_itself(self):
        scene = self.parsed(box(desc="a red umbrella"))
        self.assertEqual(
            prompt.region_prompt(scene, scene["boxes"][0]),
            "A rainy street at night. a red umbrella.",
        )

    def test_a_text_box_quotes_its_text(self):
        scene = self.parsed(box(kind="text", desc="a neon sign", text="OPEN"))
        self.assertEqual(
            prompt.region_prompt(scene, scene["boxes"][0]),
            'A rainy street at night. a neon sign. The text "OPEN".',
        )

    def test_a_box_that_says_nothing_has_no_prompt_of_its_own(self):
        scene = self.parsed(box(desc=""))
        self.assertEqual(prompt.region_prompt(scene, scene["boxes"][0]), "")

    def test_punctuation_already_there_is_kept(self):
        scene = self.parsed(box(desc="a cat!"), scene="Dusk.")
        self.assertEqual(prompt.region_prompt(scene, scene["boxes"][0]), "Dusk. a cat!")


AMBER = {
    "id": "amber",
    "name": "Amber",
    "trigger": "amb3r",
    "refs": [
        {"ref": "amber.png", "type": "input", "strength": 1.0, "fit": "cover"},
        {"ref": "muted.png", "on": False},
        {"ref": "zero.png", "strength": 0},
    ],
    "loras": [{"name": "amber.safetensors", "strength": 0.8}, {"name": "off", "on": False}],
}
KAI = {"id": "kai", "name": "Kai", "trigger": "", "refs": [], "loras": []}


class Mixins(unittest.TestCase):
    def scene(self, *boxes):
        return prompt.parse(
            {**layout("moss", list(boxes), prompt="Anime style"), "mixins": [AMBER, KAI, "junk"]}
        )

    def test_mixins_parse_as_fresco_reads_a_layer(self):
        mixins = self.scene()["mixins"]
        self.assertEqual([m["id"] for m in mixins], ["amber", "kai"])
        amber = mixins[0]
        self.assertEqual([r["ref"] for r in amber["refs"]], ["amber.png"])
        self.assertEqual(amber["refs"][0]["fit"], "cover")
        self.assertEqual(amber["loras"], [{"lora_name": "amber.safetensors", "strength": 0.8}])

    def test_a_box_names_mixins_in_its_own_order_and_unknown_ids_drop(self):
        scene = self.scene(box(mixins=["kai", "ghost", "amber", "kai"]))
        names = [m["name"] for m in prompt.mixins_of(scene, scene["boxes"][0])]
        self.assertEqual(names, ["Kai", "Amber"])

    def test_trigger_words_lead_the_box_prompt(self):
        scene = self.scene(box(mixins=["amber", "kai"], desc="she waves"))
        self.assertEqual(
            prompt.region_prompt(scene, scene["boxes"][0]), "Anime style. amb3r. she waves."
        )

    def test_a_trigger_alone_is_enough_for_a_prompt(self):
        scene = self.scene(box(mixins=["amber"], desc=""))
        self.assertEqual(prompt.region_prompt(scene, scene["boxes"][0]), "Anime style. amb3r.")

    def test_a_mixin_s_description_leads_every_box_that_uses_it(self):
        described = {**AMBER, "desc": "Amber, a girl with a copper bob"}
        raw = {
            **layout("moss", [box(mixins=["amber"], desc="she waves"), box(id="b", desc="a tree")]),
            "prompt": "Anime style",
            "mixins": [described],
        }
        items = json.loads(prompt.compile_prompt(raw))["items"]
        self.assertEqual(items[0]["caption"], "Amber, a girl with a copper bob. she waves.")
        self.assertEqual(items[1]["caption"], "a tree")
        klein = prompt.compile_prompt({**raw, "mode": "klein"})
        self.assertIn('"desc": "Amber, a girl with a copper bob. she waves."', klein)
        scene = prompt.parse(raw)
        self.assertEqual(
            prompt.region_prompt(scene, scene["boxes"][0]),
            "Anime style. amb3r. Amber, a girl with a copper bob. she waves.",
        )
        self.assertEqual(prompt.region_prompt(scene, scene["boxes"][1]), "Anime style. a tree.")

    def test_a_box_with_only_a_described_mixin_is_not_empty(self):
        raw = {
            **layout("moss", [box(mixins=["amber"], desc="")]),
            "mixins": [{**AMBER, "desc": "Amber"}],
        }
        self.assertNotIn("Lamp has no description.", prompt.warnings(raw))

    def test_old_layouts_have_no_mixins(self):
        scene = prompt.parse(layout("moss", [box()]))
        self.assertEqual((scene["mixins"], scene["boxes"][0]["mixins"]), ([], []))

    def test_cast_era_layouts_migrate_to_mixins(self):
        raw = {
            **layout("moss", [box(cast=["amber"], desc="she waves")], prompt="Anime style"),
            "cast": [AMBER],
        }
        scene = prompt.parse(raw)
        self.assertEqual([m["id"] for m in scene["mixins"]], ["amber"])
        self.assertEqual(scene["boxes"][0]["mixins"], ["amber"])
        self.assertEqual(
            prompt.region_prompt(scene, scene["boxes"][0]), "Anime style. amb3r. she waves."
        )

    def test_fresco_scenes_parse_into_regions_mode(self):
        fresco = {
            "width": 1344,
            "height": 768,
            "prompt": "Rainy night",
            "background": {"on": True, "prompt": "wet asphalt", "denoise": 0.1},
            "settings": {"overlap": "share", "visual_size": 512},
            "layers": [
                {
                    "id": "l1",
                    "name": "Amber",
                    "color": "#4f8cff",
                    "on": True,
                    "prompt": "yellow coat",
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.6,
                    "refs": [
                        {"ref": "amber.png", "type": "input", "strength": 1.0, "fit": "cover"}
                    ],
                    "loras": [{"name": "amber.safetensors", "strength": 0.8}],
                    "denoise": 0.75,
                    "influence": 1.4,
                    "grow": 4,
                    "feather": 16,
                    "locked": True,
                },
                {
                    "id": "l2",
                    "name": "Off",
                    "on": False,
                    "prompt": "hidden",
                    "x": 0,
                    "y": 0,
                    "w": 0.2,
                    "h": 0.2,
                },
            ],
        }
        scene = prompt.parse(json.dumps(fresco))
        self.assertEqual(scene["mode"], "regions")
        self.assertEqual(prompt.compile_prompt(scene), "Rainy night")
        self.assertEqual(scene["settings"], {"overlap": "share", "visual_size": 512})
        self.assertTrue(scene["backdrop"]["on"])
        self.assertEqual(scene["backdrop"]["prompt"], "wet asphalt")
        self.assertEqual(scene["boxes"][0]["desc"], "yellow coat")
        self.assertEqual(scene["boxes"][0]["color"], "#4F8CFF")
        self.assertTrue(scene["boxes"][0]["locked"])
        self.assertFalse(scene["boxes"][0]["hidden"])
        self.assertTrue(scene["boxes"][1]["hidden"])
        self.assertEqual(len(scene["boxes"][0]["refs"]), 1)
        self.assertEqual(len(scene["boxes"][0]["loras"]), 1)

    def test_regions_mode_allows_boxes_with_only_loras_or_refs(self):
        raw = layout(
            "regions",
            [box(desc="", loras=[{"name": "style.safetensors", "strength": 0.7}])],
            prompt="A street",
        )
        self.assertEqual(prompt.compile_prompt(raw), "A street")
        self.assertEqual(prompt.warnings(raw), [])


try:
    import torch

    layout_regions = load("layout._regions")
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "needs torch")
class LayoutRegions(unittest.TestCase):
    def regions(self, *boxes):
        return layout_regions.regions(prompt.parse(layout("moss", list(boxes), prompt="A room")))

    def test_each_shown_box_is_a_region_with_its_mask_and_prompt(self):
        (lamp,) = self.regions(box(x=0.0, y=0.0, w=0.5, h=1.0, desc="a lamp"))
        mask = lamp["mask"]
        height, width = mask.shape
        self.assertAlmostEqual(width / height, 1024 / 768, delta=0.02)
        self.assertEqual(float(mask[:, : width // 2 - 1].min()), 1.0)
        self.assertEqual(float(mask[:, width // 2 + 1 :].max()), 0.0)
        self.assertEqual(lamp["prompt"], "A room. a lamp.")
        self.assertEqual((lamp["fit"], lamp["loras"], lamp["label"]), ("stretch", [], "Lamp"))

    def test_the_upper_box_owns_the_overlap(self):
        under, over = self.regions(
            box(id="a", x=0.0, y=0.0, w=1.0, h=1.0, desc="a wall"),
            box(id="b", x=0.25, y=0.25, w=0.5, h=0.5, desc="a door"),
        )
        height, width = under["mask"].shape
        centre = (height // 2, width // 2)
        self.assertEqual(float(over["mask"][centre]), 1.0)
        self.assertEqual(float(under["mask"][centre]), 0.0)
        self.assertEqual(float(under["mask"][0, 0]), 1.0)

    def test_hidden_and_silent_boxes(self):
        regions = self.regions(box(id="a", hidden=True), box(id="b", desc=""))
        self.assertEqual(len(regions), 1)
        self.assertNotIn("prompt", regions[0])


@unittest.skipIf(torch is None, "needs torch")
class MixinRegions(unittest.TestCase):
    def setUp(self):
        self.encoded = []
        fusion_pkg = types.ModuleType("nynxz_pack.fusion")
        fusion_pkg.__path__ = [str(NODES / "fusion")]
        engine = types.ModuleType("nynxz_pack.fusion._fusion")
        paths = types.ModuleType("nynxz_pack._paths")

        class Settings:
            def __init__(self, visual_size, prompt_mode, prompt_blend, **extra):
                self.visual_size = visual_size
                self.prompt = (prompt_mode, prompt_blend)
                self.extra = extra

        def encode_fusion(clip, text, sources, settings, strengths, fits):
            if clip == "text-only":
                raise ValueError("Visual fusion requires a Qwen3-VL text encoder")
            self.encoded.append(
                (
                    text,
                    len(sources),
                    strengths,
                    fits,
                    settings.visual_size,
                    settings.prompt,
                    settings.extra,
                )
            )
            return [[torch.ones((1, 4, 8)), {}]]

        engine.FusionSettings, engine.encode_fusion = Settings, encode_fusion
        paths.media_file = lambda name: None if name.startswith("gone.png") else name
        self.saved = {
            name: sys.modules.get(name)
            for name in (
                "nynxz_pack.fusion",
                "nynxz_pack.fusion._fusion",
                "nynxz_pack.fusion._sources",
                "nynxz_pack._paths",
            )
        }
        sys.modules.update(
            {
                "nynxz_pack.fusion": fusion_pkg,
                "nynxz_pack.fusion._fusion": engine,
                "nynxz_pack._paths": paths,
            }
        )
        sys.modules.pop("nynxz_pack.fusion._sources", None)
        sources = importlib.import_module("nynxz_pack.fusion._sources")
        sources.load_image = lambda path: torch.zeros((1, 8, 8, 3))

    def tearDown(self):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def regions(self, boxes, mixins, clip=None, **prompt_mode):
        scene = prompt.parse({**layout("moss", boxes, prompt="Anime style"), "mixins": mixins})
        return layout_regions.regions(scene, clip, 512, **prompt_mode)

    def test_mixin_loras_ride_on_every_box_naming_them(self):
        one, two, plain = self.regions(
            [box(id="a", mixins=["amber"]), box(id="b", mixins=["amber"]), box(id="c")],
            [{**AMBER, "refs": []}],
        )
        lora = [{"lora_name": "amber.safetensors", "strength": 0.8}]
        self.assertEqual((one["loras"], two["loras"], plain["loras"]), (lora, lora, []))
        self.assertNotIn("conditioning", one)

    def test_references_are_read_into_the_box_prompt(self):
        (region,) = self.regions([box(mixins=["amber"], desc="she waves")], [AMBER], clip="vl")
        self.assertEqual(region["conditioning"].shape, (1, 4, 8))
        self.assertEqual(region["prompt"], "Anime style. amb3r. she waves.")
        self.assertEqual(
            self.encoded,
            [
                (
                    "Anime style. amb3r. she waves.",
                    1,
                    [1.0],
                    ["cover"],
                    512,
                    ("blend clean", 0.5),
                    {},
                )
            ],
        )

    def test_the_box_prompt_can_be_read_how_the_node_says(self):
        self.regions(
            [box(mixins=["amber"])],
            [AMBER],
            clip="vl",
            prompt_mode="append clean",
            prompt_blend=0.2,
        )
        self.assertEqual(self.encoded[0][5], ("append clean", 0.2))

    def test_mixin_fusion_settings_flow_to_encoder_and_box_blend_can_override(self):
        styled = {
            **AMBER,
            "blend": "salient",
            "softness": 0.75,
            "styleMode": "gist",
            "styleStrength": 0.4,
        }
        self.regions([box(mixins=["amber"])], [styled], clip="vl")
        self.assertEqual(
            self.encoded[0][6],
            {
                "blend_strength": 0.75,
                "content_mode": "saliency",
                "content_strength": 0.7,
                "style_mode": "gist",
                "style_strength": 0.4,
            },
        )
        self.encoded.clear()
        self.regions(
            [box(mixins=["amber"], blend="tiled", softness=0.2)],
            [styled],
            clip="vl",
        )
        self.assertEqual(
            self.encoded[0][6],
            {
                "blend_strength": 0.2,
                "fusion_method": "spatial-grid",
                "style_mode": "gist",
                "style_strength": 0.4,
            },
        )

    def test_references_without_a_clip_are_skipped_with_a_warning(self):
        with self.assertLogs(level="WARNING") as logs:
            (region,) = self.regions([box(mixins=["amber"], desc="she waves")], [AMBER])
        self.assertIn("wire the CLIP", logs.output[0])
        self.assertNotIn("conditioning", region)
        self.assertEqual(region["prompt"], "Anime style. amb3r. she waves.")
        self.assertEqual(self.encoded, [])

    def test_an_encoder_that_cannot_see_images_names_the_box(self):
        with self.assertRaisesRegex(ValueError, 'Box "Lamp".*can\'t read them'):
            self.regions([box(mixins=["amber"])], [AMBER], clip="text-only")

    def test_missing_reference_files_are_skipped(self):
        gone = {**AMBER, "refs": [{"ref": "gone.png"}]}
        (region,) = self.regions([box(mixins=["amber"])], [gone], clip="vl")
        self.assertNotIn("conditioning", region)
        self.assertEqual(self.encoded, [])

    def test_per_box_loras_and_refs_combine_with_mixins(self):
        b = box(
            mixins=["amber"],
            desc="waves",
            refs=[{"ref": "extra.png", "type": "input", "strength": 0.5, "fit": "contain"}],
            loras=[{"name": "coat.safetensors", "strength": 0.6}],
        )
        (region,) = self.regions([b], [AMBER], clip="vl")
        self.assertEqual(
            region["loras"],
            [
                {"lora_name": "amber.safetensors", "strength": 0.8},
                {"lora_name": "coat.safetensors", "strength": 0.6},
            ],
        )
        self.assertEqual(self.encoded[0][1], 2)
        self.assertEqual(self.encoded[0][2], [1.0, 0.5])

    def test_shared_mixin_refs_cache_image_load_and_identical_conditionings(self):
        loads = []
        sources = sys.modules["nynxz_pack.fusion._sources"]
        orig = sources.load_image

        def counting_load(path):
            loads.append(path)
            return orig(path)

        sources.load_image = counting_load
        boxes = [
            box(id="a", x=0.0, y=0.0, w=0.5, h=1.0, mixins=["amber"], desc="stands"),
            box(id="b", x=0.5, y=0.0, w=0.5, h=1.0, mixins=["amber"], desc="stands"),
        ]
        r1, r2 = self.regions(boxes, [AMBER], clip="vl")
        self.assertEqual(loads, ["amber.png [input]"])
        self.assertEqual(len(self.encoded), 1)
        self.assertIs(r1["conditioning"], r2["conditioning"])

    def test_scene_base_refs_encode_positive_and_respect_scope(self):
        base = {
            **layout(
                "regions",
                [
                    box(id="a", x=0.0, y=0.0, w=0.5, h=1.0, mixins=["amber"], desc="waves"),
                    box(id="b", x=0.5, y=0.0, w=0.5, h=1.0, desc="a wooden table"),
                ],
                prompt="Watercolor room",
            ),
            "mixins": [AMBER],
            "refs": [{"ref": "style.png", "type": "input", "strength": 0.6, "fit": "contain"}],
            "blend": "salient",
            "softness": 0.8,
            "styleMode": "gist",
            "styleStrength": 0.35,
        }

        # Scope "all": both boxes fuse the scene base ref (Amber gets [amber, style], table gets [style]).
        parsed_all = prompt.parse({**base, "refsScope": "all"})
        regions_all = layout_regions.regions(parsed_all, "vl", 512)
        self.assertEqual(len(self.encoded), 2)
        self.assertEqual(self.encoded[0][1], 2)
        self.assertEqual(self.encoded[0][2], [1.0, 0.6])
        self.assertEqual(self.encoded[1][1], 1)
        self.assertEqual(self.encoded[1][2], [0.6])
        self.assertEqual(self.encoded[1][6]["style_mode"], "gist")
        self.assertEqual(self.encoded[1][6]["style_strength"], 0.35)
        self.assertIn("conditioning", regions_all[0])
        self.assertIn("conditioning", regions_all[1])

        # Scope "plain": Amber stays pure to her own ref; plain text box inherits scene ref.
        self.encoded.clear()
        parsed_plain = prompt.parse({**base, "refsScope": "plain"})
        layout_regions.regions(parsed_plain, "vl", 512)
        self.assertEqual(self.encoded[0][1], 1)
        self.assertEqual(self.encoded[0][2], [1.0])
        self.assertEqual(self.encoded[1][1], 1)
        self.assertEqual(self.encoded[1][2], [0.6])

        # Scope "base": only main positive gets scene base refs; plain box has no conditioning override.
        self.encoded.clear()
        parsed_base = prompt.parse({**base, "refsScope": "base"})
        regions_base = layout_regions.regions(parsed_base, "vl", 512)
        self.assertEqual(len(self.encoded), 1)
        self.assertEqual(self.encoded[0][1], 1)
        self.assertNotIn("conditioning", regions_base[1])

        # Main positive encoding uses Fusion Encode with scene refs & settings.
        self.encoded.clear()
        encoded_pos, first_img = layout_regions.encode_scene_positive(
            parsed_base, "Watercolor room", "vl", 512
        )
        self.assertIsNotNone(first_img)
        self.assertEqual(encoded_pos[0][0].shape, (1, 4, 8))
        self.assertEqual(self.encoded[0][0], "Watercolor room")
        self.assertEqual(self.encoded[0][1], 1)
        self.assertEqual(self.encoded[0][6]["content_mode"], "saliency")
        self.assertEqual(self.encoded[0][6]["style_mode"], "gist")


try:
    from PIL import Image as PILImage

    sheet = load("layout._sheet")
except ImportError:
    PILImage = None


@unittest.skipIf(PILImage is None, "needs Pillow")
class MixinSheet(unittest.TestCase):
    RED = (200, 30, 30)

    def render(self, mixins, boxes):
        scene = prompt.parse({**layout("moss", boxes, width=1024, height=1024), "mixins": mixins})
        return sheet.render(scene, lambda ref: PILImage.new("RGB", (64, 64), self.RED))

    def test_a_mixin_s_image_fills_the_boxes_that_use_it_on_the_map(self):
        out = self.render(
            [AMBER, {**KAI, "color": "#10B981"}],
            [
                box(id="a", x=0.0, y=0.0, w=0.5, h=0.5, mixins=["amber"]),
                box(id="b", x=0.5, y=0.5, w=0.5, h=0.5, mixins=["kai"]),
            ],
        )
        self.assertEqual(out.width, sheet.WIDTH)
        # A square layout's map is as tall as it's allowed to be, centred across the sheet.
        side = min(sheet.WIDTH - 2 * sheet.PAD, sheet.MAP_MAX_HEIGHT)
        left = sheet.PAD + (sheet.WIDTH - 2 * sheet.PAD - side) // 2

        def at(fx, fy):
            return out.getpixel((left + round(fx * side), sheet.PAD + round(fy * side)))

        self.assertEqual(at(0.25, 0.4), self.RED)
        kai = at(0.6, 0.6)
        self.assertNotEqual(kai, self.RED)
        self.assertGreater(kai[1], kai[0])
        self.assertEqual(at(0.75, 0.25), (255, 255, 255))

    def test_a_layout_without_mixins_says_so(self):
        out = self.render([], [box()])
        self.assertLess(out.height, 200)


if __name__ == "__main__":
    unittest.main()
