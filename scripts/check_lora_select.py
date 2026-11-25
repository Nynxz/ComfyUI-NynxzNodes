#!/usr/bin/env python3
"""Checks for the LoRA Loader's `select` input: choosing a group's LoRAs from the graph.

Run: python scripts/check_lora_select.py
"""

from __future__ import annotations

import unittest

from _harness import load

s = load("lora._select")


def rows():
    return [
        {"on": True, "name": "detail.safetensors", "strength": 1},
        {
            "on": True,
            "name": "styles/film_grain.safetensors",
            "strength": 0.9,
            "group": "Styles",
            "slot": 0,
        },
        {
            "on": False,
            "name": "styles/anime.safetensors",
            "strength": 0.7,
            "group": "Styles",
            "slot": 1,
        },
        {
            "on": False,
            "name": "styles/oil.safetensors",
            "strength": 1,
            "group": "Styles",
            "slot": 2,
        },
        {"on": True, "name": "turbo.safetensors", "strength": 1, "group": "Speed-ups", "slot": 0},
    ]


def result(text):
    out, problems = s.apply(rows(), text)
    return [(r["on"], r["strength"]) for r in out], problems


class LoraSelect(unittest.TestCase):
    def test_nothing_asked_changes_nothing(self):
        for text in ("", None, "  \n# just a note"):
            out, problems = s.apply(rows(), text)
            self.assertEqual(out, rows())
            self.assertEqual(problems, [])

    def test_a_place_picks_that_one_only(self):
        got, problems = result("Styles: 1")
        self.assertEqual(got, [(True, 1), (False, 0.9), (True, 0.7), (False, 1), (True, 1)])
        self.assertEqual(problems, [])

    def test_places_count_from_zero(self):
        got, problems = result("Styles: 0")
        self.assertEqual(got[1:4], [(True, 0.9), (False, 0.7), (False, 1)])
        self.assertEqual(problems, [])
        self.assertIn("has no place 3", " ".join(result("Styles: 3")[1]))

    def test_strengths_and_several(self):
        got, _ = result("styles: 0, 2@0.5")
        self.assertEqual(got[1:4], [(True, 0.9), (False, 0.7), (True, 0.5)])

    def test_by_name(self):
        got, _ = result("Styles: Film_Grain@0.4")
        self.assertEqual(got[1:4], [(True, 0.4), (False, 0.7), (False, 1)])

    def test_off_and_all_and_several_groups(self):
        got, problems = result("Styles: all; Speed-ups: off")
        self.assertEqual(got, [(True, 1), (True, 0.9), (True, 0.7), (True, 1), (False, 1)])
        self.assertEqual(problems, [])
        self.assertEqual(result("Speed-ups = none")[0][4], (False, 1))

    def test_ungrouped_and_unnamed_groups_are_left_alone(self):
        got, _ = result("Styles: off")
        self.assertEqual(got[0], (True, 1))
        self.assertEqual(got[4], (True, 1))

    def test_a_given_strength_replaces_a_separate_clip_strength(self):
        stack = [{"on": False, "name": "a", "strength": 1, "clip": 0.2, "group": "G", "slot": 0}]
        out, _ = s.apply(stack, "G: 0@0.6")
        self.assertEqual(out, [{"on": True, "name": "a", "strength": 0.6, "group": "G", "slot": 0}])
        out, _ = s.apply(stack, "G: 0")
        self.assertEqual(out[0]["clip"], 0.2)

    def test_what_went_wrong_is_said(self):
        self.assertIn("no group named 'nope'", " ".join(result("Nope: 1")[1]))
        self.assertIn("has no place 9", " ".join(result("Styles: 9")[1]))
        self.assertIn("has no LoRA named 'zzz'", " ".join(result("Styles: zzz")[1]))
        self.assertTrue(result("Styles 2")[1])
        self.assertTrue(result("Styles: 2@strong")[1])
        # A line that made no sense doesn't stop the ones that did.
        self.assertEqual(result("garbage\nSpeed-ups: off")[0][4], (False, 1))


class LoraSelectLine(unittest.TestCase):
    def test_the_text_a_select_node_writes(self):
        self.assertEqual(s.line("Styles", "2"), "Styles: 2")
        self.assertEqual(s.line(" Styles ", " 1, 3@0.5 "), "Styles: 1, 3@0.5")
        self.assertEqual(s.line("Styles", ""), "Styles: all")
        self.assertEqual(s.line("", "2"), "")

    def test_it_reads_back(self):
        got, problems = result(s.line("Styles", "1@0.5") + "\n" + s.line("Speed-ups", "off"))
        self.assertEqual(got[1:], [(False, 0.9), (True, 0.5), (False, 1), (False, 1)])
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
