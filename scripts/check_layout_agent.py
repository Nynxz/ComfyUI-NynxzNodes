#!/usr/bin/env python3
"""Checks for the layout agent — the ask it builds and how its answers are read back.

Run: python scripts/check_layout_agent.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "nodes" / "layout"
package = types.ModuleType("layout_pkg")
package.__path__ = [str(ROOT)]
sys.modules["layout_pkg"] = package
for name in ("_prompt", "_agent"):
    spec = importlib.util.spec_from_file_location(f"layout_pkg.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"layout_pkg.{name}"] = module
    spec.loader.exec_module(module)
agent = sys.modules["layout_pkg._agent"]

REPLY = """Sure! Here is the layout:
```json
{"caption": "A bar <obj_1> with a sign <obj_2>", "background": "brick wall",
 "boxes": [
  {"label": "object", "name": "Brass lamp", "desc": "a brass lamp", "bbox_2d": [100, 200, 450, 800]},
  {"label": "text", "name": "Sign", "desc": "red neon", "text": "OPEN", "bbox_2d": [550, 150, 900, 400]},
  {"label": "object", "name": "Broken", "desc": "x", "bbox_2d": [1, 2, 3]}
 ]}
```"""


class Reading(unittest.TestCase):
    def test_boxes_come_back_in_the_editors_shape(self):
        read = agent.read_reply(REPLY)
        self.assertEqual(read["prompt"], "A bar <obj_1> with a sign <obj_2>")
        self.assertEqual(read["background"], "brick wall")
        self.assertEqual(len(read["boxes"]), 2)
        lamp, sign = read["boxes"]
        self.assertAlmostEqual(lamp["x"], 0.1)
        self.assertAlmostEqual(lamp["y"], 0.2)
        self.assertAlmostEqual(lamp["w"], 0.35)
        self.assertAlmostEqual(lamp["h"], 0.6)
        self.assertEqual((sign["kind"], sign["text"]), ("text", "OPEN"))

    def test_the_last_answer_wins_and_thinking_is_skipped(self):
        text = '{"note": "thinking"} then {"boxes": [{"bbox_2d": [0, 0, 500, 500]}]}'
        self.assertEqual(len(agent.read_reply(text)["boxes"]), 1)

    def test_swapped_corners_and_fractions_are_fixed(self):
        read = agent.read_reply('{"boxes": [{"bbox_2d": [0.6, 0.5, 0.2, 0.1]}]}')
        box = read["boxes"][0]
        self.assertAlmostEqual(box["x"], 0.2)
        self.assertAlmostEqual(box["y"], 0.1)

    def test_no_boxes_is_none(self):
        for text in ("", "no json here", '{"boxes": []}', '{"caption": "x"}', "{broken"):
            self.assertIsNone(agent.read_reply(text))


class Asking(unittest.TestCase):
    def test_klein_asks_for_citations(self):
        text = agent.build_request("a poster", "klein", 1024, 768, False)
        self.assertIn("<obj_N>", text)
        self.assertIn("1024x768", text)
        self.assertIn("Request: a poster", text)

    def test_an_empty_request_with_an_image_lays_the_image_out(self):
        self.assertIn("as it is", agent.build_request("", "moss", 512, 512, True))

    def test_a_revision_carries_the_current_boxes(self):
        current = agent.current_layout(
            {"mode": "moss", "boxes": [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "desc": "lamp"}]}
        )
        self.assertEqual(current["boxes"][0]["bbox_2d"], [100, 200, 400, 600])
        self.assertIn('"lamp"', agent.build_request("move it", "moss", 512, 512, False, current))


if __name__ == "__main__":
    unittest.main()
