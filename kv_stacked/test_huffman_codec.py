import unittest

import numpy as np

from kv_stacked.huffman_codec import encode_matrix, decode_matrix, _codes


class HuffmanCodecTest(unittest.TestCase):
    def test_full_bf16_domain(self):
        a = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        record = encode_matrix(a)
        np.testing.assert_array_equal(decode_matrix(record), a)
        self.assertEqual(record.stats["mode"], "raw_words")

    def test_constant_nonfinite_and_empty(self):
        for word in (0, 0x8000, 0x7F80, 0xFF80, 0x7F81, 0xFFFF, 0x3F80):
            a = np.full((16, 192), word, dtype=np.uint16)
            record = encode_matrix(a)
            self.assertEqual(record.stats["max_code_length"], 0)
            np.testing.assert_array_equal(decode_matrix(record), a)
        for shape in ((0, 192), (3, 0)):
            a = np.empty(shape, dtype=np.uint16)
            np.testing.assert_array_equal(decode_matrix(encode_matrix(a)), a)

    def test_random_narrow_tail_and_accounting(self):
        rng = np.random.default_rng(57)
        for shape in ((1, 1), (3, 17), (16, 192), (19, 31)):
            a = rng.integers(0, 65536, size=shape, dtype=np.uint16)
            for narrow in (False, True):
                if narrow:
                    exponents = rng.choice([121, 124, 125, 127], size=shape, p=[.05, .15, .7, .1])
                    a = (a & 0x807F) | (exponents.astype(np.uint16) << 7)
                record = encode_matrix(a)
                np.testing.assert_array_equal(decode_matrix(record), a)
                self.assertEqual(sum(record.stats["component_bytes"].values()), len(record.data))
                self.assertLessEqual(len(record.data), a.nbytes + 24)

    def test_corruption_and_invalid_tables(self):
        a = np.full((16, 192), 0x3F81, dtype=np.uint16)
        data = encode_matrix(a).data
        for bad in (data[:-1], data + b"x", b"FAIL" + data[4:], data[:24] + bytes([data[24] ^ 1]) + data[25:]):
            with self.assertRaises(ValueError):
                decode_matrix(bad)
        for lengths in ({0: 1, 1: 1, 2: 1}, {0: 0, 1: 2}):
            with self.assertRaises(ValueError):
                _codes(lengths)


if __name__ == "__main__":
    unittest.main()
