import unittest

import numpy as np

from kv_residual.entropy import encode, decode, STRATEGIES, _zigzag, _unzigzag, _rice_encode, _rice_decode


class ResidualEntropyTest(unittest.TestCase):
    def test_boundaries_and_shapes(self):
        boundary = np.array([-2147483648, 2147483647, -65535, 65535, -32768, 32767, -128, 127, -1, 0, 1], dtype=np.int32)
        patterns = [boundary, np.empty((0, 8), dtype=np.int32), np.zeros((3, 17), dtype=np.int32),
                    np.full((9,), -9123, dtype=np.int32), np.array(3, dtype=np.int32)]
        for a in patterns:
            for strategy in STRATEGIES:
                item = encode(a, strategy)
                np.testing.assert_array_equal(decode(item), a)
                self.assertEqual(decode(item).shape, a.shape)
                self.assertEqual(sum(item.stats["component_bytes"].values()), len(item.data))
                self.assertLessEqual(len(item.data), a.nbytes + 24 + 4 * a.ndim)

    def test_full_signed16_domain(self):
        a = np.arange(-32768, 32768, dtype=np.int32).reshape(256, 256)
        for strategy in STRATEGIES:
            np.testing.assert_array_equal(decode(encode(a, strategy)), a)

    def test_varied_random_distributions(self):
        rng = np.random.default_rng(29)
        patterns = [rng.integers(-2147483648, 2147483648, 977, dtype=np.int32),
                    rng.integers(-65535, 65536, 117, dtype=np.int32),
                    np.rint(rng.laplace(0, 3, 3072)).astype(np.int32),
                    np.rint(rng.normal(0, 400, 3072)).astype(np.int32)]
        for a in patterns:
            for strategy in STRATEGIES:
                np.testing.assert_array_equal(decode(encode(a, strategy)), a)

    def test_rice_and_zigzag_direct(self):
        a = np.array([-2147483648, -100, -1, 0, 1, 255, 2147483647], dtype=np.int32)
        z = _zigzag(a)
        np.testing.assert_array_equal(_unzigzag(z), a)
        k, total, payload = _rice_encode(z)
        np.testing.assert_array_equal(_rice_decode(payload, len(z), k, total), z)

    def test_corruption(self):
        a = np.arange(-5, 6, dtype=np.int32)
        for strategy in STRATEGIES:
            data = encode(a, strategy).data
            for bad in (data[:-1], data + b"x", b"FAIL" + data[4:], data[:-1] + bytes([data[-1] ^ 1])):
                with self.assertRaises(ValueError):
                    decode(bad)


if __name__ == "__main__":
    unittest.main()
