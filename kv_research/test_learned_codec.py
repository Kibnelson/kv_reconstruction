"""Exactness, frozen model identity, and byte accounting for learned control."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kv_research.learned_codec import (train_model, model_bytes, save_model, load_model,
                                      encode_matrix, decode_matrix, _pack_corrections,
                                      _unpack_corrections)


class LearnedCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(104)
        # Training and evaluation have different random inputs. The vectors
        # have structured magnitudes so compressed groups really occur.
        cls.training = (rng.integers(15500, 15800, (160, 64), dtype=np.uint16)
                        | (rng.integers(0, 2, (160, 64), dtype=np.uint16) << 15))
        cls.model = train_model({"layer_00_K": cls.training},
                                {"iterations": 3, "max_samples_per_key": 256})

    def test_exact_tails_nonfinite_and_all_bf16_words(self):
        rng = np.random.default_rng(330)
        examples = [np.arange(65536, dtype=np.uint16).reshape(256, 256),
                    rng.integers(15500, 15800, (7, 23), dtype=np.uint16),
                    np.array([[0, 0x8000, 1, 0x8001, 0x7F80, 0xFF80,
                               0x7FC0, 0xFFFF, 0x7F7F]], dtype=np.uint16),
                    np.empty((0, 64), dtype=np.uint16)]
        for example in examples:
            encoded = encode_matrix(example, self.model, "layer_00_K")
            restored = decode_matrix(encoded, self.model)
            self.assertTrue(np.array_equal(example, restored))
            self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
        compressed = encode_matrix(examples[1], self.model, "layer_00_K")
        self.assertGreater(compressed.stats["mode_counts"].get("1", 0), 0)

    def test_all_variable_widths(self):
        rng = np.random.default_rng(91)
        widths = np.arange(16, dtype=np.uint16)
        values = rng.integers(0, 32768, (16, 8), dtype=np.uint16)
        packed = _pack_corrections(values, widths)
        restored = _unpack_corrections(packed, widths)
        expected = values & ((1 << widths.astype(np.uint32)) - 1)[:, None]
        self.assertTrue(np.array_equal(restored, expected))
        self.assertEqual(len(packed), int(widths.sum()))

    def test_model_roundtrip_and_accounting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frozen.lkvw"
            stats = save_model(self.model, path)
            restored = load_model(path)
            self.assertEqual(stats["serialized_bytes"], path.stat().st_size)
            self.assertEqual(stats["prototype_bytes"] + stats["metadata_and_header_bytes"], path.stat().st_size)
            self.assertEqual(self.model.model_id, restored.model_id)
            self.assertEqual(model_bytes(self.model), model_bytes(restored))
            encoded = encode_matrix(self.training[:2], self.model, "layer_00_K")
            self.assertTrue(np.array_equal(self.training[:2], decode_matrix(encoded, restored)))

    def test_unknown_model_and_key_rejected(self):
        different = train_model({"layer_00_K": self.training},
                                {"seed": 77, "iterations": 2, "max_samples_per_key": 256})
        record = encode_matrix(self.training[:2], self.model, "layer_00_K")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            decode_matrix(record, different)
        with self.assertRaisesRegex(ValueError, "unknown"):
            encode_matrix(self.training[:2], self.model, "unavailable")

    def test_training_and_format_errors_rejected(self):
        with self.assertRaises(ValueError):
            train_model({"bad": np.zeros((8, 8), np.float32)})
        record = encode_matrix(self.training[:3], self.model, "layer_00_K")
        for corrupted in (record.data[:10], record.data[:-1], record.data + b"garbage"):
            with self.assertRaises(ValueError):
                decode_matrix(corrupted, self.model)
        with self.assertRaises(ValueError):
            load_model(model_bytes(self.model)[:-1])

    def test_supported_codebook_sizes(self):
        for count in (16, 64, 256):
            model = train_model({"layer_00_K": self.training},
                                {"n_centroids": count, "iterations": 1,
                                 "max_samples_per_key": 256})
            source = self.training[8:11]
            encoded = encode_matrix(source, model, "layer_00_K")
            self.assertTrue(np.array_equal(source, decode_matrix(encoded, model)))
            self.assertEqual(count * 8 * 2, model.codebooks["layer_00_K"].nbytes)


if __name__ == "__main__":
    unittest.main()
