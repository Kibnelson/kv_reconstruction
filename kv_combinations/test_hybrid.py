import unittest
import numpy as np
from kv_combinations import hybrid
from kv_research import learned_codec as learned
from kv_smoke import codec

class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(45)
        x = rng.integers(15400, 15800, (80, 64), dtype=np.uint16)
        cls.model = learned.train_model({'global': x}, {'iterations': 2, 'max_samples_per_key': 128})

    def encode(self, x):
        return hybrid.encode_matrix(x, learned.encode_matrix(x, self.model), codec.encode_matrix(x, 'exponent'))

    def test_exact_boundaries_and_accounting(self):
        examples = [np.arange(65536, dtype=np.uint16).reshape(256, 256),
                    np.full((1, 65), 15555, dtype=np.uint16),
                    np.empty((0, 64), dtype=np.uint16),
                    np.array([[0, 0x8000, 0x7f80, 0xffff, 1]], dtype=np.uint16)]
        for x in examples:
            enc = self.encode(x)
            np.testing.assert_array_equal(hybrid.decode_matrix(enc, self.model), x)
            self.assertEqual(sum(enc.stats['component_bytes'].values()), len(enc.data))
            self.assertGreaterEqual(enc.stats['bytes_saved_vs_learned_source'], 0)

    def test_bad_inputs(self):
        x = np.full((2, 64), 15555, dtype=np.uint16)
        enc = self.encode(x)
        for data in (enc.data[:-1], enc.data + b'x', b'bad'):
            with self.assertRaises(ValueError):
                hybrid.decode_matrix(data, self.model)
        with self.assertRaises(ValueError):
            hybrid.encode_matrix(x, learned.encode_matrix(x, self.model), codec.encode_matrix(x, 'exponent', group_size=128))

    def test_each_selector_mode(self):
        rng = np.random.default_rng(51)
        predictable = np.tile(self.model.codebooks['global'][0], 8)
        exponent_friendly = (np.uint16(124 << 7) | rng.integers(0, 128, 64, dtype=np.uint16))
        raw_friendly = (np.arange(64, dtype=np.uint16) * np.uint16(1000))
        raw_friendly[1] = 0x7fc0
        x = np.stack([predictable, exponent_friendly, raw_friendly])
        enc = self.encode(x)
        self.assertEqual(set(enc.stats['mode_counts']), {'0', '1', '2'})
        np.testing.assert_array_equal(hybrid.decode_matrix(enc, self.model), x)

if __name__ == '__main__':
    unittest.main()
