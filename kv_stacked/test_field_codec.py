import unittest

import numpy as np

from kv_stacked.field_codec import encode_matrix, decode_matrix, STRATEGIES


class FieldCodecTest(unittest.TestCase):
    def test_full_bf16_domain(self):
        words = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        for strategy in STRATEGIES:
            record = encode_matrix(words, strategy=strategy)
            np.testing.assert_array_equal(decode_matrix(record.data), words)
            self.assertEqual(sum(record.stats["component_bytes"].values()), len(record.data))

    def test_shapes_and_exact_patterns(self):
        rng = np.random.default_rng(47)
        for shape in ((0, 192), (2, 0), (1, 1), (3, 17), (16, 192), (19, 5)):
            for pattern in ("random", "zero", "narrow"):
                a = rng.integers(0, 65536, size=shape, dtype=np.uint16)
                if pattern == "zero":
                    a[:] = 0x8000
                if pattern == "narrow":
                    a = (a & 0x807F) | np.uint16(124 << 7)
                for strategy in STRATEGIES:
                    item = encode_matrix(a, strategy=strategy)
                    np.testing.assert_array_equal(decode_matrix(item), a)
                    self.assertLessEqual(len(item.data), a.nbytes + 24)

    def test_forced_entropy_modes(self):
        a = np.full((16, 192), 0x3F81, dtype=np.uint16)
        for strategy in ("deflate", "packed_deflate"):
            item = encode_matrix(a, strategy=strategy)
            self.assertIn("deflate", item.stats["mode"])
            np.testing.assert_array_equal(decode_matrix(item), a)

    def test_nonfinite_payload_bits(self):
        # All NaN payloads and both infinities can be entropy-coded directly.
        a = (np.arange(256, dtype=np.uint16) & 0x7F) | np.uint16(0x7F80)
        a[128:] |= np.uint16(0x8000)
        a = np.tile(a, 12).reshape(16, 192)
        for strategy in ("packed", "deflate", "packed_deflate", "best"):
            item = encode_matrix(a, strategy=strategy)
            self.assertNotEqual(item.stats["mode"], "raw_words")
            np.testing.assert_array_equal(decode_matrix(item), a)

    def test_corruption(self):
        a = np.full((16, 192), 0x3F81, dtype=np.uint16)
        data = encode_matrix(a).data
        for corrupted in (data[:-1], data + b"x", b"FAIL" + data[4:], data[:24] + bytes([data[24] ^ 1]) + data[25:]):
            with self.assertRaises(ValueError):
                decode_matrix(corrupted)


if __name__ == "__main__":
    unittest.main()
