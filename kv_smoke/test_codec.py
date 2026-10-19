"""Meaningful mechanism checks; no learned/serving performance claims."""
import unittest

import numpy as np

from codec import (bf16_words_to_float32, decode_matrix, encode_matrix,
                   float32_to_bf16_words, inverse_ordered_u16, ordered_u16,
                   _coset_decode, _coset_payload, _pack_bits, _unpack_bits)


class CodecTests(unittest.TestCase):
    def test_bijection_exhaustive_including_all_nonfinites(self):
        words = np.arange(65536, dtype=np.uint16)
        self.assertTrue(np.array_equal(inverse_ordered_u16(ordered_u16(words)), words))

    def test_bf16_rounding_ties(self):
        f32_bits = np.asarray([0x3F808000, 0x3F818000, 0xBF808000, 0x00000000,
                               0x80000000, 0x7F800000, 0xFF800000, 0x7F800001], dtype=np.uint32)
        words = float32_to_bf16_words(f32_bits.view(np.float32))
        self.assertEqual(words[:7].tolist(), [0x3F80, 0x3F82, 0xBF80, 0, 0x8000, 0x7F80, 0xFF80])
        self.assertNotEqual(int(words[7] & 0x7F), 0)

    def test_pack_bits_every_supported_width_and_partial_byte(self):
        rng = np.random.default_rng(19)
        for width in range(17):
            values = rng.integers(0, max(1, 1 << width), size=37, dtype=np.uint16)
            payload = _pack_bits(values, width)
            self.assertTrue(np.array_equal(_unpack_bits(payload, values.size, width), values))

    def test_coset_error_bound_and_exact_prediction(self):
        rng = np.random.default_rng(7)
        for requested in range(2, 16):
            radius = (1 << (requested - 1)) - 1
            c = rng.integers(radius, 65536 - radius, size=71, dtype=np.int32)
            error = rng.integers(-radius, radius + 1, size=71, dtype=np.int32)
            error[0] = radius
            original = inverse_ordered_u16(c + error)
            prediction = inverse_ordered_u16(c)
            width, payload, measured = _coset_payload(original, prediction)
            self.assertEqual(width, requested)
            self.assertEqual(measured, radius)
            self.assertTrue(np.array_equal(_coset_decode(prediction, width, payload), original))
        prediction = np.asarray([0, 0x8000, 0x3F80], dtype=np.uint16)
        width, payload, _ = _coset_payload(prediction, prediction)
        self.assertEqual(width, 0)
        self.assertEqual(payload, b"")
        self.assertTrue(np.array_equal(_coset_decode(prediction, width, payload), prediction))

    def test_all_methods_exact_on_varied_inputs(self):
        rng = np.random.default_rng(123)
        cases = [
            float32_to_bf16_words(rng.normal(size=(19, 37)).astype(np.float32)),
            np.zeros((4, 31), dtype=np.uint16),
            np.full((9, 13), 0x3F80, dtype=np.uint16),
            np.arange(65536, dtype=np.uint16).reshape(256, 256),
            np.asarray([[0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x7F7F,
                         0xFF7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1]], dtype=np.uint16),
            np.empty((0, 5), dtype=np.uint16),
        ]
        for words in cases:
            for method in ("raw", "exponent", "zero_coset", "lattice2_coset", "lattice4_coset", "lattice2_sign_coset"):
                for group_size in (8, 64, 257):
                    encoded = encode_matrix(words, method, group_size)
                    self.assertEqual(len(encoded.data), encoded.stats["serialized_bytes"])
                    self.assertTrue(np.array_equal(decode_matrix(encoded), words), (method, group_size))
                    self.assertEqual(encoded.stats["payload_bytes"] + encoded.stats["container_overhead_bytes"],
                                     len(encoded.data))
                    self.assertEqual(sum(encoded.stats["component_bytes"].values()), len(encoded.data))
                    self.assertEqual(sum(encoded.stats["accepted_payload_bytes_by_mode"].values()),
                                     encoded.stats["payload_bytes"])

    def test_record_never_larger_than_same_raw_container(self):
        words = float32_to_bf16_words(np.random.default_rng(0).normal(size=(31, 64)).astype(np.float32))
        raw = encode_matrix(words, "raw", 64)
        for method in ("exponent", "zero_coset", "lattice2_coset", "lattice4_coset", "lattice2_sign_coset"):
            encoded = encode_matrix(words, method, 64)
            self.assertLessEqual(len(encoded.data), len(raw.data))

    def test_corrupt_or_truncated_container_rejected(self):
        data = encode_matrix(np.zeros((2, 8), dtype=np.uint16), "lattice4_coset", 8).data
        for broken in (data[:10], data[:-1], data + b"x", b"FAIL" + data[4:]):
            with self.assertRaises(ValueError):
                decode_matrix(broken)


if __name__ == "__main__":
    unittest.main()
