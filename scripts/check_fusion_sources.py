#!/usr/bin/env python3
"""Reference images load as fusion sources: RGBA kept only when the file has transparency, and a
reference that is missing or unreadable is skipped.

Run: python scripts/check_fusion_sources.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

from _harness import ROOT, load, stub_comfy
from PIL import Image

stub_comfy()
paths = types.ModuleType(f"{ROOT}._paths")
paths.media_file = lambda name: (lambda p: p if Path(p).is_file() else None)(
    name.rsplit(" [", 1)[0]
)
sys.modules[paths.__name__] = paths
sources = load("fusion._sources")


class Sources(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        Image.new("RGBA", (5, 3), (255, 0, 0, 128)).save(self.dir / "cut.png")
        Image.new("RGB", (4, 2)).save(self.dir / "photo.jpg")
        (self.dir / "junk.png").write_bytes(b"not an image")

    def ref(self, name: str) -> dict:
        return {"ref": str(self.dir / name), "type": "input", "strength": 0.5, "fit": "cover"}

    def test_transparency_keeps_its_alpha(self):
        self.assertEqual(tuple(sources.load_image(self.dir / "cut.png").shape), (1, 3, 5, 4))
        self.assertEqual(tuple(sources.load_image(self.dir / "photo.jpg").shape), (1, 2, 4, 3))

    def test_missing_and_unreadable_references_are_skipped(self):
        with self.assertLogs(level="WARNING"):
            images, strengths, fits = sources.load_refs(
                [self.ref("cut.png"), self.ref("junk.png"), self.ref("gone.png")], "Test"
            )
        self.assertEqual((len(images), strengths, fits), (1, [0.5], ["cover"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
