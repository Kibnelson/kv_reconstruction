"""Checks for the genuine E8 projection and complete exact record format."""
import itertools
import unittest

import numpy as np

from kv_research.e8_codec import (decode_matrix, encode_matrix, is_e8_point,
    nearest_e8, _pack_coordinates, _unpack_coordinates, _coordinates_size)
from kv_smoke.codec import float32_to_bf16_words


class E8Tests(unittest.TestCase):
    def test_nearest_membership_and_idempotence(self):
        rng = np.random.default_rng(410)
        x = rng.normal(size=(1000, 8)) * 7
        q = nearest_e8(x)
        self.assertTrue(np.all(is_e8_point(q)))
        self.assertTrue(np.array_equal(nearest_e8(q), q))
        self.assertTrue(np.all(np.sum((x - q) ** 2, axis=1) <= 1.0 + 1e-12))

    def test_nearest_vs_exhaustive_local_cosets(self):
        # Enumerate the three nearest integer choices in each coordinate for
        # both cosets.  Any nearer candidate must occur in this neighborhood.
        rng = np.random.default_rng(42)
        offsets = np.asarray(list(itertools.product((-1, 0, 1), repeat=8)))
        for x in np.vstack((rng.uniform(-2, 2, size=(12, 8)), np.zeros((1, 8)),
                            np.full((1, 8), .25), np.arange(8)[None] * .5)):
            best = np.inf
            for shift in (0.0, .5):
                integers = np.rint(x - shift).astype(np.int32) + offsets
                integers = integers[(integers.sum(axis=1) & 1) == 0]
                candidates = integers.astype(np.float64) + shift
                best = min(best, float(np.min(np.sum((candidates - x) ** 2, axis=1))))
            actual = float(np.sum((nearest_e8(x) - x) ** 2))
            self.assertAlmostEqual(actual, best, places=12)

    def test_parity_coordinate_packing(self):
        rng = np.random.default_rng(5)
        for n in (1, 7, 8, 9, 31):
            for scale in (0.0, .1, 1.0, 10.0, 1000.0):
                points = nearest_e8(rng.normal(size=(n, 8)) * scale)
                minimum, width, blob = _pack_coordinates(points)
                self.assertEqual(len(blob), _coordinates_size(n, width))
                self.assertTrue(np.array_equal(_unpack_coordinates(blob, n, minimum, width), points))

    def test_all_bf16_patterns_and_extremes_exact(self):
        rng = np.random.default_rng(101)
        cases = [np.arange(65536, dtype=np.uint16).reshape(256, 256),
                 np.zeros((7, 31), dtype=np.uint16),
                 np.full((8, 64), 0x3F80, dtype=np.uint16),
                 float32_to_bf16_words(rng.normal(size=(19, 37)).astype(np.float32)),
                 np.empty((0, 8), dtype=np.uint16),
                 np.asarray([[0, 0x8000, 1, 0x8001, 0x7F7F, 0xFF7F, 0x7FC1, 0xFFC1]], dtype=np.uint16)]
        for words in cases:
            for method in ("e8_float", "e8_symbol"):
                for group_size in (8, 64, 256):
                    encoded = encode_matrix(words, method=method, group_size=group_size)
                    self.assertTrue(np.array_equal(decode_matrix(encoded), words),
                                    (method, group_size, words.shape))
                    self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
                    self.assertEqual(encoded.stats["serialized_bytes"], len(encoded.data))
                    # Raw fallback bounds each record payload, not total
                    # bytes against unframed raw tensors.
                    n = (words.size + group_size - 1) // group_size
                    self.assertLessEqual(len(encoded.data), words.nbytes + 20 + n * 5)

    def test_config_variants_self_describing(self):
        rng = np.random.default_rng(111)
        words = float32_to_bf16_words(rng.normal(size=(15, 192)).astype(np.float32))
        for step in (1, 5, 7, 10, 14):
            e = encode_matrix(words, "e8_symbol", symbol_step_log2=step)
            self.assertTrue(np.array_equal(decode_matrix(e.data), words))
        for scale in (.125, .25, .5, 1.0, 2.0):
            e = encode_matrix(words, "e8_float", float_scale_multiplier=scale)
            self.assertTrue(np.array_equal(decode_matrix(e.data), words))

    def test_format_rejects_corruption_and_truncation(self):
        e = encode_matrix(np.full((8, 64), 0x3F80, dtype=np.uint16), "e8_symbol")
        for payload in (e.data[:10], e.data[:-1], e.data + b"x", b"FAIL" + e.data[4:]):
            with self.assertRaises(ValueError):
                decode_matrix(payload)
        with self.assertRaises(ValueError):
            nearest_e8(np.ones(7))
        with self.assertRaises(ValueError):
            encode_matrix(np.ones((1, 8), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
