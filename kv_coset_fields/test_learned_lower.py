import tempfile
import unittest
from pathlib import Path
import numpy as np

from kv_coset_fields.learned_lower import (LowerModel, train_model, save_model,
    load_model, model_bytes, encode_lower, decode_lower)


class LearnedLowerTests(unittest.TestCase):
    @staticmethod
    def model(conditional=False):
        book = np.repeat((np.arange(16, dtype=np.uint8) * 8)[:, None], 8, axis=1)
        means = {"layer_00_K": np.arange(256, dtype=np.uint8) & 127} if conditional else {}
        return LowerModel({"layer_00_K": book}, means,
                          {"n_centroids": 16, "conditional": conditional}, {"trained": False})

    def test_all_bf16_patterns_with_known_exponents(self):
        words = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        lower = (((words >> 8) & 128) | (words & 127)).astype(np.uint8)
        exponents = ((words >> 7) & 255).astype(np.uint8)
        for conditional in (False, True):
            model = self.model(conditional)
            encoded = encode_lower(lower, exponents, model, "layer_00_K")
            recovered = decode_lower(encoded, exponents, model, "layer_00_K")
            restored_words = ((recovered.astype(np.uint16) & 128) << 8) | (exponents.astype(np.uint16) << 7) | (recovered & 127)
            self.assertTrue(np.array_equal(restored_words, words))
            self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
        # The nonconditional structured sequence genuinely exercises cosets.
        encoded = encode_lower(lower, exponents, self.model(), "layer_00_K")
        self.assertEqual(encoded.stats["mode"], "predicted")

    def test_signs_tails_raw_fallback_and_empty(self):
        rng = np.random.default_rng(47)
        for shape in ((0, 4), (1, 1), (7, 23), (17, 64)):
            lower = rng.integers(0, 256, shape, dtype=np.uint8)
            exponent = rng.integers(0, 256, shape, dtype=np.uint8)
            encoded = encode_lower(lower, exponent, self.model(), "layer_00_K")
            self.assertTrue(np.array_equal(lower, decode_lower(encoded, exponent, self.model(), "layer_00_K")))
        self.assertEqual(encoded.stats["mode"], "raw_fallback")

    def test_modular_wrap_near_zero(self):
        model = LowerModel({"x": np.zeros((16, 8), np.uint8)}, {},
                           {"n_centroids": 16, "conditional": False}, {})
        # Fixed base64 plus prototype64 predicts0 on the fraction circle.
        model.codebooks["x"][:] = 64
        lower = np.tile(np.array([127, 0, 1, 126, 2, 255, 128, 129], np.uint8), (64, 1))
        exponent = np.full(lower.shape, 255, np.uint8)
        encoded = encode_lower(lower, exponent, model, "x")
        self.assertEqual(encoded.stats["mode"], "predicted")
        self.assertTrue(np.array_equal(lower, decode_lower(encoded, exponent, model, "x")))

    def test_training_models_serialization_and_identity(self):
        rng = np.random.default_rng(91)
        training = {"x": rng.integers(0, 65536, (64, 64), dtype=np.uint16)}
        for conditional in (False, True):
            model = train_model(training, {"conditional": conditional, "iterations": 2,
                                          "max_samples_per_key": 128})
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "lower.bin"
                stats = save_model(model, path); restored = load_model(path)
                self.assertEqual(model_bytes(model), model_bytes(restored))
                self.assertEqual(stats["serialized_bytes"], path.stat().st_size)
                self.assertEqual(stats["metadata_and_header_bytes"] + stats["parameter_array_bytes"], path.stat().st_size)
                lower = np.zeros((16, 16), np.uint8); exponent = lower.copy()
                encoded = encode_lower(lower, exponent, model, "x")
                other = self.model()
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    decode_lower(encoded, exponent, other, "x")
                self.assertTrue(np.array_equal(lower, decode_lower(encoded, exponent, restored, "x")))

    def test_corruption_and_contract_errors(self):
        lower = np.zeros((64, 16), np.uint8); exponent = lower.copy(); model = self.model()
        encoded = encode_lower(lower, exponent, model, "layer_00_K")
        for bad in (encoded.data[:10], encoded.data[:-1], encoded.data + b"trailing"):
            with self.assertRaises(ValueError):
                decode_lower(bad, exponent, model, "layer_00_K")
        with self.assertRaises(ValueError):
            decode_lower(encoded, exponent, model, "wrong")
        with self.assertRaises(ValueError):
            decode_lower(encoded, exponent[:1], model, "layer_00_K")
        with self.assertRaises(ValueError):
            load_model(model_bytes(model)[:-1])

    def test_forced_cosets_random_all_bf16_and_tails(self):
        rng = np.random.default_rng(681)
        words = rng.permutation(np.arange(65536, dtype=np.uint16)).reshape(256, 256)
        examples = [words, words[:3, :29]]
        for source in examples:
            lower = (((source >> 8) & 128) | (source & 127)).astype(np.uint8)
            exponent = ((source >> 7) & 255).astype(np.uint8)
            model = self.model(conditional=True)
            encoded = encode_lower(lower, exponent, model, "layer_00_K", force_cosets=True)
            recovered = decode_lower(encoded, exponent, model, "layer_00_K")
            self.assertTrue(np.array_equal(lower, recovered))
            restored = ((recovered.astype(np.uint16) & 128) << 8) | (exponent.astype(np.uint16) << 7) | (recovered & 127)
            self.assertTrue(np.array_equal(source, restored))
            self.assertEqual(encoded.stats["mode"], "predicted")
            self.assertTrue(encoded.stats["forced_coset_diagnostic"])
            self.assertEqual(encoded.stats["candidate_raw_vectors"], 0)
            self.assertEqual(encoded.stats["candidate_predicted_vectors"], (lower.size + 7) // 8)
            self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
            self.assertGreater(encoded.stats["component_bytes"]["modular_corrections"], 0)
            self.assertGreater(len(encoded.data), lower.nbytes)


if __name__ == "__main__":
    unittest.main()
