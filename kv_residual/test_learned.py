import unittest
import numpy as np

from kv_research.learned_codec import LearnedModel
from kv_residual.learned import encode_matrix, decode_matrix


class LearnedResidualTests(unittest.TestCase):
    @staticmethod
    def model():
        book = np.repeat((np.arange(16, dtype=np.uint16) * 2048)[:, None], 8, axis=1)
        return LearnedModel({"x": book}, {"n_centroids": 16}, {"trained": False})

    def test_all_bf16_patterns_matched_corrections(self):
        words = np.random.default_rng(931).permutation(np.arange(65536, dtype=np.uint16)).reshape(256, 256)
        model = self.model()
        for assignment in ("radius", "l1"):
            for correction in ("coset", "raw", "rice", "zigzag_deflate", "centered_deflate",
                               "centered_packed", "zigzag_packed", "best_entropy", "best"):
                with self.subTest(assignment=assignment, correction=correction):
                    encoded = encode_matrix(words, model, "x", correction, assignment, force=True)
                    self.assertTrue(np.array_equal(words, decode_matrix(encoded, model)))
                    self.assertNotEqual(encoded.stats["accepted_coding_mode"], "raw")
                    self.assertEqual(encoded.stats["stored_predictor_vectors"], words.size // 8)
                    self.assertEqual(encoded.stats["mechanism_counts"]["stored_learned_vectors"], words.size // 8)
                    self.assertEqual(encoded.stats["record_modes"], {encoded.stats["accepted_coding_mode"]: 1})
                    self.assertEqual(encoded.stats["mechanism_counts"]["stored_entropy_records"]
                                     + encoded.stats["mechanism_counts"]["stored_coset_records"], 1)
                    self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))

    def test_tails_empty_noncontiguous_and_literal_fallback(self):
        rng = np.random.default_rng(480)
        matrices = [np.empty((0, 19), np.uint16), rng.integers(0, 65536, (1, 1), np.uint16),
                    rng.integers(0, 65536, (13, 31), np.uint16)[:, ::2]]
        model = self.model()
        for words in matrices:
            for correction in ("coset", "best_entropy", "best"):
                for force in (False, True):
                    encoded = encode_matrix(words, model, "x", correction, force=force)
                    self.assertTrue(np.array_equal(words, decode_matrix(encoded, model)))
                    self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
        encoded = encode_matrix(matrices[1], model, "x", "raw", force=False)
        self.assertEqual(encoded.stats["accepted_coding_mode"], "raw")

    def test_best_selects_actual_serialized_bytes(self):
        rng = np.random.default_rng(39)
        model = self.model()
        # Structured and irregular residuals exercise the two correction modes.
        for words in (np.full((64, 64), 6145, np.uint16),
                      rng.integers(0, 65536, (64, 64), np.uint16)):
            coset = encode_matrix(words, model, "x", "coset")
            entropy = encode_matrix(words, model, "x", "best_entropy")
            best = encode_matrix(words, model, "x", "best")
            self.assertEqual(len(best.data), min(len(coset.data), len(entropy.data)))

    def test_checksum_model_identity_and_contracts(self):
        words = np.zeros((16, 64), np.uint16); model = self.model()
        encoded = encode_matrix(words, model, "x", "best")
        for position in (4, 18, 55, len(encoded.data) - 5):
            damaged = bytearray(encoded.data); damaged[position] ^= 1
            with self.assertRaisesRegex(ValueError, "checksum"):
                decode_matrix(damaged, model)
        other = self.model(); other.config["different"] = True
        with self.assertRaisesRegex(ValueError, "SHA256"):
            decode_matrix(encoded, other)
        for bad in (encoded.data[:10], encoded.data[:-1], encoded.data + b"trailing"):
            with self.assertRaises(ValueError):
                decode_matrix(bad, model)
        with self.assertRaises(ValueError):
            encode_matrix(words, model, "unknown")
        with self.assertRaises(ValueError):
            encode_matrix(words, model, "x", assignment="mse")


if __name__ == "__main__":
    unittest.main()
