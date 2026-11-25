#!/usr/bin/env python3
"""Checks for saved layouts — what reaches the disk, and what never does.

Run: python scripts/check_layout_library.py
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "nodes"
spec = importlib.util.spec_from_file_location("layout_library", NODES / "layout" / "library.py")
library = importlib.util.module_from_spec(spec)
spec.loader.exec_module(library)


class Library(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_saved_layout_lists_and_loads_back(self):
        layout = {"width": 1024, "layers": [{"id": "a"}]}
        preview = [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5, "color": "#ff0000"}]
        self.assertEqual(
            library.save_layout(self.root, "regions", "  Two  people ", layout, preview),
            "Two people",
        )
        listed = library.list_layouts(self.root, "regions")
        self.assertEqual([item["name"] for item in listed], ["Two people"])
        self.assertEqual(listed[0]["preview"], preview)
        self.assertEqual(library.load_layout(self.root, "regions", "Two people")["layout"], layout)

    def test_layouts_of_another_mode_stay_apart(self):
        library.save_layout(self.root, "regions", "a", {}, None)
        self.assertEqual(library.list_layouts(self.root, "ideogram"), [])

    def test_names_and_modes_that_would_leave_the_folder_are_refused(self):
        for name in ["../escape", "a/b", "..", ".hidden", "", "a\\b", "c:d"]:
            self.assertIsNone(library.save_layout(self.root, "regions", name, {}, None), name)
        for mode in ["../x", "Regions", "", "a/b"]:
            self.assertIsNone(library.save_layout(self.root, mode, "ok", {}, None), mode)
        self.assertEqual(os.listdir(self.root), [])

    def test_deleting_removes_only_that_layout(self):
        library.save_layout(self.root, "regions", "keep", {}, None)
        library.save_layout(self.root, "regions", "drop", {}, None)
        self.assertTrue(library.delete_layout(self.root, "regions", "drop"))
        self.assertFalse(library.delete_layout(self.root, "regions", "drop"))
        self.assertEqual(
            [item["name"] for item in library.list_layouts(self.root, "regions")], ["keep"]
        )

    def test_an_oversized_layout_is_refused(self):
        huge = "x" * (library.MAX_BYTES + 1)
        self.assertIsNone(library.save_layout(self.root, "regions", "big", huge, None))


class Browsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_lists_every_mode_with_its_mode(self):
        library.save_layout(self.root, "regions", "One", {}, None)
        library.save_layout(self.root, "moss", "Two", {}, None)
        listed = {(item["mode"], item["name"]) for item in library.list_all(self.root)}
        self.assertEqual(listed, {("regions", "One"), ("moss", "Two")})

    def test_rename_moves_the_file_and_keeps_the_layout(self):
        library.save_layout(self.root, "moss", "Old", {"k": 1}, None)
        self.assertEqual(library.rename_layout(self.root, "moss", "Old", " New  name "), "New name")
        self.assertIsNone(library.load_layout(self.root, "moss", "Old"))
        self.assertEqual(library.load_layout(self.root, "moss", "New name")["layout"], {"k": 1})

    def test_rename_never_overwrites_another(self):
        library.save_layout(self.root, "moss", "A", {"a": 1}, None)
        library.save_layout(self.root, "moss", "B", {"b": 1}, None)
        self.assertIsNone(library.rename_layout(self.root, "moss", "A", "B"))
        self.assertIsNone(library.rename_layout(self.root, "moss", "A", "../x"))
        self.assertEqual(library.load_layout(self.root, "moss", "B")["layout"], {"b": 1})

    def test_saved_mixins_are_id_keyed_and_never_clobber_same_names(self):
        amber = {"id": "c1", "name": "Amber", "trigger": "amb3r", "refs": [], "loras": []}
        first = library.save_mixin(self.root, "  Amber ", amber, category="Characters")
        self.assertIsNotNone(first)
        self.assertEqual(first["name"], "Amber")
        self.assertEqual(first["category"], "Characters")
        # Saving a second preset with the same display name gets a distinct ID instead of overwriting.
        second = library.save_mixin(self.root, "Amber", {**amber, "trigger": "v2"}, category="Alt")
        self.assertIsNotNone(second)
        self.assertNotEqual(first["id"], second["id"])
        self.assertIsNone(library.save_mixin(self.root, "Bad", amber, mixin_id="../escape"))
        self.assertEqual(library.list_all(self.root), [])
        listed = library.list_mixins(self.root)
        self.assertEqual(len(listed), 2)
        # Updating by ID renames in place without creating a third preset.
        updated = library.save_mixin(
            self.root, "Amber V2", amber, mixin_id=first["id"], category="Characters"
        )
        self.assertEqual(updated["id"], first["id"])
        self.assertEqual(updated["name"], "Amber V2")
        self.assertEqual(len(library.list_mixins(self.root)), 2)
        self.assertTrue(library.delete_mixin(self.root, first["id"]))
        self.assertTrue(library.delete_mixin(self.root, second["id"]))
        self.assertEqual(library.list_mixins(self.root), [])


if __name__ == "__main__":
    unittest.main()
