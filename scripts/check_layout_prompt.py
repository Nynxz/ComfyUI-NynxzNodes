#!/usr/bin/env python3
"""Checks for layout prompts — each mode writes exactly what its model was trained on.

Run: python scripts/check_layout_prompt.py
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "nodes"
spec = importlib.util.spec_from_file_location("layout_prompt", NODES / "layout" / "_prompt.py")
prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt)


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

    def test_old_layouts_have_no_mixins(self):
        scene = prompt.parse(layout("moss", [box()]))
        self.assertEqual((scene["mixins"], scene["boxes"][0]["mixins"]), ([], []))


try:
    import sys
    import types

    import torch

    for name, folder in (
        ("nynxz_pack", ""),
        ("nynxz_pack.layout", "layout"),
        ("nynxz_pack.fresco", "fresco"),
        ("nynxz_pack.regions", "regions"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(NODES / folder)]
        sys.modules[name] = package
    sys.modules["nynxz_pack.layout._prompt"] = prompt
    regions_spec = importlib.util.spec_from_file_location(
        "nynxz_pack.layout._regions", NODES / "layout" / "_regions.py"
    )
    layout_regions = importlib.util.module_from_spec(regions_spec)
    sys.modules["nynxz_pack.layout._regions"] = layout_regions
    regions_spec.loader.exec_module(layout_regions)
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
        fusion_pkg.__path__ = []
        engine = types.ModuleType("nynxz_pack.fusion._fusion")
        loader = types.ModuleType("nynxz_pack.fusion.fusion_input")

        class Settings:
            def __init__(self, visual_size, prompt_mode, prompt_blend):
                self.visual_size = visual_size
                self.prompt = (prompt_mode, prompt_blend)

        def encode_fusion(clip, text, sources, settings, strengths, fits):
            if clip == "text-only":
                raise ValueError("Visual fusion requires a Qwen3-VL text encoder")
            self.encoded.append(
                (text, len(sources), strengths, fits, settings.visual_size, settings.prompt)
            )
            return [[torch.ones((1, 4, 8)), {}]]

        engine.FusionSettings, engine.encode_fusion = Settings, encode_fusion
        loader._resolve_path = lambda ref, kind: None if ref == "gone.png" else f"{kind}/{ref}"
        loader._load_image = lambda path: torch.zeros((1, 8, 8, 3))
        self.saved = {
            name: sys.modules.get(name)
            for name in (
                "nynxz_pack.fusion",
                "nynxz_pack.fusion._fusion",
                "nynxz_pack.fusion.fusion_input",
            )
        }
        sys.modules.update(
            {
                "nynxz_pack.fusion": fusion_pkg,
                "nynxz_pack.fusion._fusion": engine,
                "nynxz_pack.fusion.fusion_input": loader,
            }
        )

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
            [("Anime style. amb3r. she waves.", 1, [1.0], ["cover"], 512, ("blend clean", 0.5))],
        )

    def test_the_box_prompt_can_be_read_how_the_node_says(self):
        self.regions(
            [box(mixins=["amber"])],
            [AMBER],
            clip="vl",
            prompt_mode="append clean",
            prompt_blend=0.2,
        )
        self.assertEqual(self.encoded[0][-1], ("append clean", 0.2))

    def test_references_without_a_clip_say_to_wire_one(self):
        with self.assertRaisesRegex(ValueError, "Wire the CLIP"):
            self.regions([box(mixins=["amber"])], [AMBER])

    def test_an_encoder_that_cannot_see_images_names_the_box(self):
        with self.assertRaisesRegex(ValueError, 'Box "Lamp".*can\'t read them'):
            self.regions([box(mixins=["amber"])], [AMBER], clip="text-only")

    def test_missing_reference_files_are_skipped(self):
        gone = {**AMBER, "refs": [{"ref": "gone.png"}]}
        (region,) = self.regions([box(mixins=["amber"])], [gone], clip="vl")
        self.assertNotIn("conditioning", region)
        self.assertEqual(self.encoded, [])


if __name__ == "__main__":
    unittest.main()
