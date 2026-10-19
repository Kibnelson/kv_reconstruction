import unittest
import struct
import zlib
import numpy as np
from kv_coset_fields import temporal

class TemporalTests(unittest.TestCase):
    def test_all_symbols_predictors_and_tails(self):
        rng = np.random.default_rng(35)
        examples = [np.arange(256, dtype=np.uint8).reshape(16, 16),
                    rng.integers(0, 256, (11, 23), dtype=np.uint8),
                    np.empty((0, 23), dtype=np.uint8),
                    np.full((32, 23), 87, dtype=np.uint8),
                    np.zeros((32, 16), dtype=np.uint8),
                    rng.integers(0, 256, (5, 46), dtype=np.uint8)[:, ::2]]
        for x in examples:
            exponents = rng.integers(0, 256, x.shape, dtype=np.uint8)
            for predictor in temporal.PREDICTORS:
                record = temporal.encode_lower(x, exponents, predictor)
                np.testing.assert_array_equal(temporal.decode_lower(record, exponents), x)
                self.assertEqual(sum(record.stats['component_bytes'].values()), len(record.data))

    def test_coset_exhaustive(self):
        x = np.arange(256, dtype=np.uint16)[:, None]
        p = np.arange(256, dtype=np.uint16)[None, :]
        np.testing.assert_array_equal(temporal._recover(p, x, 8), np.broadcast_to(x, (256, 256)).astype(np.uint8))
        for b in range(9):
            result = temporal._recover(p, x & ((1 << b) - 1), b)
            self.assertTrue(np.all((result.astype(np.uint16) & ((1 << b) - 1)) == (x & ((1 << b) - 1))))

    def test_full_bf16_enclosing_fields(self):
        words = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        lower = (((words >> 8) & 128) | (words & 127)).astype(np.uint8)
        exp = ((words >> 7) & 255).astype(np.uint8)
        for predictor in temporal.PREDICTORS:
            record = temporal.encode_lower(lower, exp, predictor)
            lo = temporal.decode_lower(record, exp).astype(np.uint16)
            restored = ((lo & 128) << 8) | (lo & 127) | (exp.astype(np.uint16) << 7)
            np.testing.assert_array_equal(restored, words)

    def test_corruption_and_wrong_side_information(self):
        x = np.zeros((16, 16), dtype=np.uint8)
        exp = np.full_like(x, 124)
        record = temporal.encode_lower(x, exp)
        for blob in (record.data[:-1], record.data + b'x', record.data[:10] + bytes([record.data[10] ^ 1]) + record.data[11:]):
            with self.assertRaises(ValueError):
                temporal.decode_lower(blob, exp)
        wrong = exp.copy()
        wrong[0, 0] ^= 1
        with self.assertRaises(ValueError):
            temporal.decode_lower(record, wrong)
        # Recompute whole checksum to exercise a malformed group mode check.
        damaged = bytearray(record.data)
        damaged[temporal._HEADER.size] = 99
        damaged[-4:] = struct.pack('<I', zlib.crc32(damaged[:-4]))
        with self.assertRaises(ValueError):
            temporal.decode_lower(damaged, exp)

if __name__ == '__main__':
    unittest.main()
