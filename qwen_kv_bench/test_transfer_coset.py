"""Exact-symbol and factorization checks for the transfer coset reference."""
import importlib.util
import unittest

import numpy as np

from qwen_kv_bench import transfer_coset as codec


HAS_CONSTRICTION = importlib.util.find_spec("constriction") is not None


class ProbabilityTests(unittest.TestCase):
    def test_all_bit_patterns_and_probability_partition(self):
        words = np.arange(65536, dtype=np.uint16)
        np.testing.assert_array_equal(codec.word_of(codec.ordered(words)), words)
        mu = np.array([-100., 0., 100.])
        inv = np.array([.5, 1., 10.])
        gn = codec.group_num(mu, inv)
        np.testing.assert_array_equal(gn.sum(1), np.full(3, codec.TOTAL))
        for group in range(512):
            mn = codec.mant_num(mu, inv, np.full(3, group, np.int64))
            self.assertTrue(np.all(mn > 0))
            np.testing.assert_array_equal(mn.sum(1), gn[:, group])

    def test_matched_joint_distribution_identity(self):
        rng = np.random.default_rng(234)
        words = rng.integers(0, 65536, 257, dtype=np.uint16)
        mu = rng.normal(0, 4, len(words))
        inv = np.exp(rng.normal(0, .5, len(words)))
        for bits in (0, 1, 3, 7):
            stats = codec.coset_diagnostics(mu, inv, words, bits)
            self.assertEqual(stats["direct_ideal_bits"], stats["coset_model_ideal_bits"])
            self.assertEqual(sum(stats["residue_histogram"]), len(words))

    def test_bad_prediction_or_settings_rejected(self):
        for mu, inv in (([np.nan], [1.]), ([0.], [0.]), ([0.], [np.inf])):
            with self.assertRaises(ValueError):
                codec.group_num(mu, inv)
        for mode, bits in (("missing", 1), ("direct", 1), ("coset_model", -1), ("coset_uniform", 8)):
            with self.assertRaises(ValueError):
                codec.code_block(None, np.empty(0), np.empty(0), mode=mode, residue_bits=bits)

    def test_serialized_cdf_table_is_used_and_checked(self):
        mu, inv = np.array([0.25]), np.array([1.])
        table = codec.F_TAB.copy()
        np.testing.assert_array_equal(codec.group_num(mu, inv, table), codec.group_num(mu, inv))
        table[0] = 1
        with self.assertRaises(ValueError):
            codec.group_num(mu, inv, table)
        with self.assertRaises(ValueError):
            codec.group_num(mu, inv, codec.F_TAB[:-1])


@unittest.skipUnless(HAS_CONSTRICTION, "optional constriction package is unavailable")
class StreamTests(unittest.TestCase):
    def _roundtrip(self, words, mode, bits, batch=256):
        import constriction
        encoder = constriction.stream.queue.RangeEncoder()
        ideal_enc = 0.
        for start in range(0, len(words), batch):
            block = words[start:start + batch]
            mu = np.linspace(-2., 2., len(block))
            inv = np.full(len(block), .75)
            actual, ideal = codec.code_block(encoder, mu, inv, block, mode, bits)
            np.testing.assert_array_equal(actual, block)
            ideal_enc += ideal
        stream = encoder.get_compressed()
        decoder = constriction.stream.queue.RangeDecoder(stream)
        decoded, ideal_dec = [], 0.
        for start in range(0, len(words), batch):
            count = min(batch, len(words) - start)
            mu = np.linspace(-2., 2., count)
            inv = np.full(count, .75)
            block, ideal = codec.code_block(decoder, mu, inv, mode=mode, residue_bits=bits)
            decoded.append(block)
            ideal_dec += ideal
        np.testing.assert_array_equal(np.concatenate(decoded), words)
        self.assertAlmostEqual(ideal_enc, ideal_dec, places=9)
        return len(stream) * 4, ideal_enc

    def test_all_original_bf16_patterns_roundtrip(self):
        words = np.arange(65536, dtype=np.uint16)
        for mode, bits in (("direct", 0), ("coset_model", 4), ("coset_uniform", 4)):
            with self.subTest(mode=mode):
                self._roundtrip(words, mode, bits)

    def test_residue_extremes_and_model_ideal_equality(self):
        rng = np.random.default_rng(12)
        words = rng.integers(0, 65536, 769, dtype=np.uint16)
        _, direct_ideal = self._roundtrip(words, "direct", 0)
        for bits in (0, 1, 7):
            _, split_ideal = self._roundtrip(words, "coset_model", bits)
            self.assertAlmostEqual(direct_ideal, split_ideal, places=9)
            self._roundtrip(words, "coset_uniform", bits)


if __name__ == "__main__":
    unittest.main()
