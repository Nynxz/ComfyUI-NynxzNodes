#!/usr/bin/env python3
"""Checks for Get from Index — one node over every way ComfyUI batches something.

Run: python scripts/check_batch_index.py

The risk in a wildcard node is that it silently does the WRONG thing for a type nobody tested:
returning a bare item where a batch was expected, dropping a latent's mask, or quietly reusing a
tensor the caller still owns. So every kind is checked for the same three properties — right
item, batch dimension kept, input left alone.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

NODES = Path(__file__).resolve().parents[1] / "nodes"
spec = importlib.util.spec_from_file_location("batch_index", NODES / "batch" / "_index.py")
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


def images(count):
    """A batch where item k is filled with k, so the one that comes back is identifiable."""
    return torch.stack([torch.full((8, 8, 3), float(k)) for k in range(count)])


def latent(count, mask=None):
    out = {"samples": torch.stack([torch.full((4, 8, 8), float(k)) for k in range(count)])}
    if mask is not None:
        out["noise_mask"] = mask
    return out


def audio(count):
    return {
        "waveform": torch.stack([torch.full((2, 100), float(k)) for k in range(count)]),
        "sample_rate": 44100,
    }


class Counting(unittest.TestCase):
    def test_a_tensor_counts_its_first_dimension(self):
        self.assertEqual(index.count_of(images(5)), 5)

    def test_a_latent_counts_its_samples(self):
        self.assertEqual(index.count_of(latent(3)), 3)

    def test_an_audio_counts_its_waveform(self):
        self.assertEqual(index.count_of(audio(4)), 4)

    def test_a_list_counts_its_entries(self):
        self.assertEqual(index.count_of([1, 2, 3]), 3)

    def test_a_conditioning_counts_its_entries(self):
        conditioning = [[torch.zeros(1, 4, 8), {}], [torch.zeros(1, 4, 8), {}]]
        self.assertEqual(index.count_of(conditioning), 2)

    def test_something_that_is_not_a_batch_says_what_it_handles(self):
        with self.assertRaises(ValueError) as caught:
            index.count_of(42)
        self.assertIn("LATENT", str(caught.exception))

    def test_a_dict_without_a_batch_key_is_not_a_batch(self):
        with self.assertRaises(ValueError):
            index.count_of({"sample_rate": 44100})


class Resolving(unittest.TestCase):
    """Index conventions, which must match core's so the node is not a surprise."""

    def test_a_plain_index_is_itself(self):
        self.assertEqual(index.resolve(2, 5), 2)

    def test_negative_counts_from_the_end(self):
        self.assertEqual(index.resolve(-1, 5), 4)
        self.assertEqual(index.resolve(-5, 5), 0)

    def test_clamp_takes_the_nearest_real_item(self):
        self.assertEqual(index.resolve(99, 5), 4)
        self.assertEqual(index.resolve(-99, 5), 0)

    def test_wrap_counts_round(self):
        self.assertEqual(index.resolve(5, 5, "wrap"), 0)
        self.assertEqual(index.resolve(7, 5, "wrap"), 2)
        self.assertEqual(index.resolve(-1, 5, "wrap"), 4)

    def test_error_says_the_valid_range(self):
        with self.assertRaises(ValueError) as caught:
            index.resolve(9, 5, "error")
        message = str(caught.exception)
        self.assertIn("0..4", message)
        self.assertIn("-1..-5", message)

    def test_an_empty_batch_is_refused_whatever_the_mode(self):
        for mode in index.OUT_OF_RANGE:
            with self.assertRaisesRegex(ValueError, "empty"):
                index.resolve(0, 0, mode)


class Slicing(unittest.TestCase):
    """The same three properties for every kind: right item, batch kept, input untouched."""

    def test_an_image_batch_yields_the_asked_for_image(self):
        out, at, count = index.take(images(5), 3)
        self.assertEqual(out.shape, (1, 8, 8, 3))  # batch dimension KEPT
        self.assertEqual(out[0, 0, 0, 0].item(), 3.0)
        self.assertEqual((at, count), (3, 5))

    def test_the_last_image_is_reachable_with_minus_one(self):
        out, at, _ = index.take(images(5), -1)
        self.assertEqual(out[0, 0, 0, 0].item(), 4.0)
        self.assertEqual(at, 4)

    def test_a_latent_yields_the_asked_for_sample(self):
        out, _, _ = index.take(latent(4), 2)
        self.assertEqual(out["samples"].shape[0], 1)
        self.assertEqual(out["samples"][0, 0, 0, 0].item(), 2.0)

    def test_a_latent_keeps_its_other_keys(self):
        source = latent(3)
        source["extra"] = "kept"
        self.assertEqual(index.take(source, 1)[0]["extra"], "kept")

    def test_a_per_item_noise_mask_is_sliced_with_the_samples(self):
        mask = torch.stack([torch.full((1, 8, 8), float(k)) for k in range(3)])
        out, _, _ = index.take(latent(3, mask), 2)
        self.assertEqual(out["noise_mask"].shape[0], 1)
        self.assertEqual(out["noise_mask"][0, 0, 0, 0].item(), 2.0)

    def test_a_single_noise_mask_covers_the_whole_batch_and_is_kept(self):
        """core's rule: one mask applies to every item, so it survives the slice whole."""
        out, _, _ = index.take(latent(3, torch.ones(1, 1, 8, 8)), 2)
        self.assertEqual(out["noise_mask"].shape[0], 1)
        self.assertEqual(out["noise_mask"].sum().item(), 64)

    def test_a_latent_reports_which_item_it_came_from(self):
        self.assertEqual(index.take(latent(4), 2)[0]["batch_index"], [2])

    def test_an_audio_yields_the_asked_for_waveform_and_keeps_its_rate(self):
        out, _, _ = index.take(audio(3), 1)
        self.assertEqual(out["waveform"].shape[0], 1)
        self.assertEqual(out["waveform"][0, 0, 0].item(), 1.0)
        self.assertEqual(out["sample_rate"], 44100)

    def test_a_list_yields_a_one_entry_list(self):
        out, _, _ = index.take(["a", "b", "c"], 1)
        self.assertEqual(out, ["b"])

    def test_a_conditioning_entry_comes_back_wrapped_as_a_conditioning(self):
        """A CONDITIONING is a list, so one entry must still be a list to stay wireable."""
        first, second = [torch.zeros(1, 4, 8), {"tag": 0}], [torch.ones(1, 4, 8), {"tag": 1}]
        out, _, _ = index.take([first, second], 1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1]["tag"], 1)

    def test_a_tuple_stays_a_tuple(self):
        self.assertIsInstance(index.take(("a", "b"), 0)[0], tuple)


class Length(unittest.TestCase):
    def test_length_takes_a_sub_batch(self):
        out, _, _ = index.take(images(5), 1, length=3)
        self.assertEqual(out.shape[0], 3)
        self.assertEqual([out[k, 0, 0, 0].item() for k in range(3)], [1.0, 2.0, 3.0])

    def test_length_is_trimmed_to_what_is_there(self):
        """Not an error — the same forgiving behaviour as core's nodes."""
        out, _, _ = index.take(images(5), 3, length=99)
        self.assertEqual(out.shape[0], 2)

    def test_length_below_one_still_yields_an_item(self):
        self.assertEqual(index.take(images(5), 0, length=0)[0].shape[0], 1)

    def test_length_slices_a_latents_mask_too(self):
        mask = torch.stack([torch.full((1, 8, 8), float(k)) for k in range(4)])
        out, _, _ = index.take(latent(4, mask), 1, length=2)
        self.assertEqual(out["noise_mask"].shape[0], 2)


class DoesNotMutateTheInput(unittest.TestCase):
    """A node that hands back a view of its input corrupts whatever else is wired to it."""

    def test_the_source_tensor_is_not_shared(self):
        source = images(3)
        out, _, _ = index.take(source, 1)
        out[0, 0, 0, 0] = 99.0
        self.assertEqual(source[1, 0, 0, 0].item(), 1.0)

    def test_the_source_latent_dict_is_not_modified(self):
        source = latent(3)
        index.take(source, 1)
        self.assertEqual(source["samples"].shape[0], 3)
        self.assertNotIn("batch_index", source)

    def test_the_source_list_is_not_modified(self):
        source = ["a", "b", "c"]
        index.take(source, 1)
        self.assertEqual(source, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1], verbosity=2)
