"""Correctness and accounting checks for learned+residual E8/scalar records."""
import struct
import unittest
import zlib

import numpy as np

from kv_research.e8_codec import nearest_e8, is_e8_point, _pack_coordinates, _unpack_coordinates
from kv_research.learned_codec import LearnedModel
from kv_residual.learned_e8 import encode_matrix, decode_matrix, _HEADER
from kv_smoke.codec import float32_to_bf16_words


def fixture_model():
    rng = np.random.default_rng(701)
    book = rng.integers(15300, 16400, size=(16, 8), dtype=np.uint16)
    book[0] = 0
    book[1] = np.arange(8, dtype=np.uint16) + 16000
    book.flags.writeable = False
    return LearnedModel({"test": book}, {"n_centroids": 16, "vector_dim": 8}, {})


class LearnedE8Tests(unittest.TestCase):
    def test_real_residual_e8_membership_and_coordinate_roundtrip(self):
        rng = np.random.default_rng(91)
        residual = rng.integers(-1000, 1001, size=(100, 8))
        points = nearest_e8(residual / 128.0)
        self.assertTrue(np.all(is_e8_point(points)))
        self.assertGreater(int(np.any(points != 0, axis=1).sum()), 0)
        self.assertGreater(int(np.any(points != np.floor(points), axis=1).sum()), 0)
        minimum, width, blob = _pack_coordinates(points)
        self.assertTrue(np.array_equal(_unpack_coordinates(blob, len(points), minimum, width), points))

    def test_all_bf16_patterns_nonfinite_tails_and_signs_exact(self):
        model = fixture_model()
        rng = np.random.default_rng(808)
        cases = [np.arange(65536, dtype=np.uint16).reshape(256, 256),
                 np.zeros((5, 31), dtype=np.uint16),
                 np.full((1, 64), 0x8000, dtype=np.uint16),
                 np.asarray([[0, 0x8000, 1, 0x8001, 0x7F7F, 0xFF7F, 0x7F80, 0xFFC1]], dtype=np.uint16),
                 float32_to_bf16_words(rng.normal(size=(19, 37)).astype(np.float32)),
                 np.empty((0, 8), dtype=np.uint16)]
        for words in cases:
            for family in ("e8", "scalar"):
                for correction in ("coset", "best"):
                    encoded = encode_matrix(words, model, "test", step_log2=7,
                        group_size=3072, family=family, correction=correction, force=True)
                    decoded = decode_matrix(encoded, model)
                    self.assertTrue(np.array_equal(words, decoded), (words.shape, family, correction))
                    self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))

    def test_scale_group_and_fallback_options(self):
        rng = np.random.default_rng(92)
        words = float32_to_bf16_words(rng.normal(size=(7, 192)).astype(np.float32))
        model = fixture_model()
        for step in (1, 4, 7, 10, 14):
            for family in ("e8", "scalar"):
                for force in (False, True):
                    encoded = encode_matrix(words, model, "test", step_log2=step,
                        group_size=64, family=family, correction="best", force=force)
                    self.assertTrue(np.array_equal(decode_matrix(encoded.data, model), words))
                    if force:
                        self.assertEqual(encoded.stats["mode_counts"], {"1": 21})
                        counts = encoded.stats["mechanism_counts"]
                        self.assertEqual(counts["learned_vectors"], words.size // 8)
                        self.assertEqual(counts["e8_vectors"], words.size // 8 if family == "e8" else 0)
                        if family == "e8" and step == 7:
                            self.assertGreater(counts["nonzero_e8_vectors"], 0)
                            self.assertGreater(counts["half_integer_e8_vectors"], 0)
                    self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))

    def test_zero_residual_is_an_actual_e8_record(self):
        model = fixture_model()
        words = model.codebooks["test"][1].reshape(1, 8).copy()
        for correction in ("coset", "best"):
            encoded = encode_matrix(words, model, "test", step_log2=7,
                group_size=64, family="e8", correction=correction, force=True)
            self.assertEqual(encoded.stats["mode_counts"], {"1": 1})
            self.assertEqual(encoded.stats["branch_diagnostics"]["nonzero_residual_vectors"], 0)
            self.assertTrue(np.array_equal(decode_matrix(encoded, model), words))

    def test_step14_coset_half_modulus_boundary_uses_full_magnitude(self):
        book = np.full((16, 8), 20000, dtype=np.uint16)
        model = LearnedModel({"test": book}, {"n_centroids": 16, "vector_dim": 8}, {})
        words = np.full((1, 8), 20000, dtype=np.uint16)
        words[0, 0] = 3616
        encoded = encode_matrix(words, model, "test", step_log2=14,
            group_size=3072, family="e8", correction="coset", force=True)
        self.assertEqual(encoded.stats["mode_counts"], {"1": 1})
        self.assertEqual(encoded.stats["candidate_maximum_error_histogram"], {"16384": 1})
        self.assertTrue(np.array_equal(decode_matrix(encoded, model), words))

    def test_model_identity_format_and_nested_checksum_rejected(self):
        model = fixture_model()
        words = np.full((4, 64), 0x3F80, dtype=np.uint16)
        encoded = encode_matrix(words, model, "test", step_log2=7,
            group_size=3072, family="e8", correction="best", force=True)
        for data in (encoded.data[:10], encoded.data[:-1], encoded.data + b"x", b"FAIL" + encoded.data[4:]):
            with self.assertRaises(ValueError):
                decode_matrix(data, model)
        wrong = fixture_model()
        wrong.config["identity_test"] = 1
        with self.assertRaises(ValueError):
            decode_matrix(encoded, wrong)
        # Corrupt explicit coordinate width, not an otherwise valid data bit.
        corrupt = bytearray(encoded.data)
        group_start = _HEADER.size + len("test")
        corrupt[group_start + 5 + 2] = 17
        corrupt[-4:] = struct.pack("<I", zlib.crc32(corrupt[:-4]))
        with self.assertRaises(ValueError):
            decode_matrix(corrupt, model)
        corrupt = bytearray(encoded.data)
        corrupt[-5] ^= 128
        corrupt[-4:] = struct.pack("<I", zlib.crc32(corrupt[:-4]))
        with self.assertRaises(ValueError):
            decode_matrix(corrupt, model)
        # Complete-record CRC also protects learned IDs, E8 coordinates and
        # signs, independently of the nested residual checksum.
        for index in (0, _HEADER.size + len("test") + 8, len(encoded.data) - 1):
            corrupt = bytearray(encoded.data)
            corrupt[index] ^= 1
            with self.assertRaises(ValueError):
                decode_matrix(corrupt, model)


if __name__ == "__main__":
    unittest.main()
