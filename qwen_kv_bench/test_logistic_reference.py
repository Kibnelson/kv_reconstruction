"""Bit-exact CPU logistic arithmetic streams, beyond ideal probability budgets."""
from fractions import Fraction
import copy
import json
import math
import unittest
import zlib

import numpy as np

from qwen_kv_bench import logistic_reference as c


class LogisticReferenceTests(unittest.TestCase):
    def test_ordered_alphabet_finite_bounds_and_cdf_normalization(self):
        words = np.arange(65536, dtype=np.uint16)
        ranks = c.ordered(words)
        np.testing.assert_array_equal(c.from_ordered(ranks), words)
        finite = (words & 0x7f80) != 0x7f80
        self.assertEqual((int(ranks[finite].min()), int(ranks[finite].max()), int(finite.sum())), (128, 65407, 65280))
        self.assertEqual(c.ordered(np.array([0x8000, 0], dtype=np.uint16)).tolist(), [32767, 32768])
        self.assertEqual(c.BOUNDARIES[32768], 0)
        self.assertTrue(np.isneginf(c.BOUNDARIES[128]))
        self.assertTrue(np.isposinf(c.BOUNDARIES[65408]))
        for mean, scale in ((0.0, 1.0), (.37, .002), (-1.5, .3), (1e308, 1e-40)):
            table = np.array([c._cdf(rank, mean, scale) for rank in range(65537)], dtype=np.uint64)
            self.assertEqual(int(table[0]), 0)
            self.assertEqual(int(table[-1]), c.TOTAL)
            frequency = np.diff(table)
            self.assertTrue(np.all(frequency >= 1))
            self.assertEqual(int(frequency.sum()), c.TOTAL)
            np.testing.assert_array_equal(frequency[:128], np.ones(128, dtype=np.uint64))
            np.testing.assert_array_equal(frequency[65408:], np.ones(128, dtype=np.uint64))

    def test_actual_arithmetic_bytes_against_fraction_interval_oracle(self):
        # Independent infinite-precision arithmetic oracle for a small constant
        # dyadic distribution.  These short products fit exactly in 64 bits.
        bounds = [(0, 1), (1, 4), (4, 8)]
        symbols = [2, 1, 0, 2, 1, 1, 2, 0, 1, 2]
        encoder = c._ArithmeticEncoder(total=8)
        low, high = Fraction(0), Fraction(1)
        ideal_bits = 0.0
        for symbol in symbols:
            a, b = bounds[symbol]
            width = high - low
            high, low = low + width * Fraction(b, 8), low + width * Fraction(a, 8)
            encoder.update(a, b)
            ideal_bits -= math.log2((b - a) / 8)
        payload, arithmetic_bits, nbits = encoder.finish()
        full = int.from_bytes(payload, "big")
        point = Fraction(full, 1 << (8 * len(payload)))
        self.assertLessEqual(low, point)
        self.assertLess(point, high)
        self.assertLessEqual(arithmetic_bits, ideal_bits + 2.000001)
        self.assertEqual(nbits, arithmetic_bits + 64)
        decoded, a, b = [], Fraction(0), Fraction(1)
        for _ in symbols:
            position = (point - a) / (b - a)
            for symbol, (left, right) in enumerate(bounds):
                if Fraction(left, 8) <= position < Fraction(right, 8):
                    width = b - a
                    b, a = a + width * Fraction(right, 8), a + width * Fraction(left, 8)
                    decoded.append(symbol)
                    break
        self.assertEqual(decoded, symbols)

    def test_long_underflow_run_and_explicit_lookahead_guard(self):
        encoder = c._ArithmeticEncoder(total=4)
        for _ in range(257):
            encoder.update(1, 3)  # Entirely in the E3 middle-half interval.
        payload, arithmetic_bits, nbits = encoder.finish()
        self.assertEqual(arithmetic_bits, 259)
        self.assertEqual(nbits, arithmetic_bits + 64)
        point = Fraction(int.from_bytes(payload, "big"), 1 << (8 * len(payload)))
        width = Fraction(1, 1 << 257)
        self.assertLessEqual(Fraction(1, 2) - width / 2, point)
        self.assertLess(point, Fraction(1, 2) + width / 2)
        decoder = c._ArithmeticDecoder(payload, nbits, total=4)
        for _ in range(257):
            self.assertIn(decoder.cumulative(), (1, 2))
            decoder.update(1, 3)

    def test_all_65536_patterns_including_nan_payloads_infinities_and_zeros(self):
        words = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        mean, logs = np.zeros(words.shape, dtype=np.float64), np.zeros(words.shape, dtype=np.float64)
        payload, descriptor = c.encode(words, mean, logs)
        np.testing.assert_array_equal(c.decode(payload, descriptor, mean, logs), words)
        self.assertEqual(descriptor["nonfinite_symbols"], 256)
        self.assertGreater(descriptor["integer_floor_symbols"], 0)
        self.assertEqual(descriptor["cdf_strategy"], "shared_exact_table")
        self.assertEqual(descriptor["payload_bytes"], len(payload))
        self.assertEqual(descriptor["payload_crc32"], zlib.crc32(payload))
        self.assertGreater(len(payload) * 8, descriptor["arithmetic_bits"])
        self.assertLess(abs(descriptor["arithmetic_bits"] - descriptor["integer_cdf_bits"]), 3)
        json.dumps(descriptor, allow_nan=False)

    def test_per_value_distributions_empty_pages_shapes_and_scale_clamps(self):
        rng = np.random.default_rng(45)
        for shape in ((0,), (0, 7), (1,), (7,), (3, 5), (9, 7)):
            words = rng.integers(0, 65536, size=shape, dtype=np.uint16)
            mean = rng.normal(0, 3, size=shape).astype(np.float32)
            logs = rng.uniform(-120, 120, size=shape).astype(np.float32)
            payload, desc = c.encode(words, mean, logs)
            np.testing.assert_array_equal(c.decode(payload, desc, mean, logs), words)
            self.assertEqual(desc["clamped_scale_values"], int(np.count_nonzero((logs < -100) | (logs > 100))))
            self.assertEqual(desc["padding_bits"], (-desc["nbits"]) % 8)
            self.assertGreaterEqual(desc["nbits"], 66)
            self.assertNotIn("mu", desc)
            self.assertNotIn("log_scale", desc)
            self.assertNotIn("words", desc)

    def test_independent_symbol_search_matches_full_cdf_bisection(self):
        rng = np.random.default_rng(88)
        for mean, log_scale in ((0., 0.), (.37, -6.), (-1.5, -.3), (0., -100.), (1e308, 1.)):
            scale = math.exp(log_scale)
            for value in rng.integers(0, c.TOTAL, size=35, dtype=np.uint64):
                value = int(value)
                symbol, a, b, _ = c._find_symbol(value, mean, scale)
                self.assertLessEqual(a, value)
                self.assertLess(value, b)
                self.assertEqual(a, c._cdf(symbol, mean, scale))
                self.assertEqual(b, c._cdf(symbol + 1, mean, scale))

    def test_corruption_prediction_mismatch_and_invalid_shapes_rejected(self):
        words = np.array([[0, 0x8000, 0x0001, 0x8001, 0x7f80, 0xff80, 0x7fc1]], dtype=np.uint16)
        mean, logs = np.zeros(words.shape), np.full(words.shape, -90.)
        payload, desc = c.encode(words, mean, logs)
        changed = bytearray(payload)
        changed[0] ^= 128
        with self.assertRaisesRegex(ValueError, "CRC32"):
            c.decode(changed, desc, mean, logs)
        with self.assertRaises(ValueError):
            c.decode(payload[:-1], desc, mean, logs)
        with self.assertRaisesRegex(ValueError, "predictor SHA256"):
            c.decode(payload, desc, mean + .01, logs)
        wrong_crc = dict(desc, words_crc32=desc["words_crc32"] ^ 1)
        with self.assertRaisesRegex(ValueError, "word CRC32"):
            c.decode(payload, wrong_crc, mean, logs)
        guard = bytearray(payload)
        bit = desc["arithmetic_bits"]
        guard[bit >> 3] |= 1 << (7 - (bit & 7))
        guard_desc = dict(desc, payload_crc32=zlib.crc32(guard))
        with self.assertRaisesRegex(ValueError, "lookahead guard"):
            c.decode(guard, guard_desc, mean, logs)
        empty = np.empty(0, dtype=np.uint16)
        empty_prediction = np.empty(0, dtype=np.float64)
        pad, pad_desc = c.encode(empty, empty_prediction, empty_prediction)
        pad = bytearray(pad)
        pad[-1] |= 1
        pad_desc["payload_crc32"] = zlib.crc32(pad)
        with self.assertRaisesRegex(ValueError, "byte padding"):
            c.decode(pad, pad_desc, empty_prediction, empty_prediction)
        invalid = copy.deepcopy(desc)
        invalid["shape"] = [c.MAX_SYMBOLS + 1]
        with self.assertRaisesRegex(ValueError, "safety limit"):
            c.decode(payload, invalid, mean, logs)
        with self.assertRaises(ValueError):
            c.encode(words.astype(np.int32), mean, logs)
        with self.assertRaisesRegex(ValueError, "shapes"):
            c.encode(words, mean.reshape(-1), logs)
        for bad in (np.full(words.shape, np.nan), np.full(words.shape, np.inf)):
            with self.assertRaisesRegex(ValueError, "finite"):
                c.encode(words, bad, logs)


if __name__ == "__main__":
    unittest.main()
