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


if __name__ == "__main__":
    unittest.main()
