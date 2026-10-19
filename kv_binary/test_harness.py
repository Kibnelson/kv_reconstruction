"""Integration checks for exact cached-inner wrappers and state charging."""
import unittest
import numpy as np
from kv_binary.run_eval import factories, encode_candidate, verify_cache
from kv_stacked import wrapper


class HarnessTests(unittest.TestCase):
    def test_wrapper_uses_identical_inner_records_without_reencoding(self):
        words = np.arange(17 * 7, dtype=np.uint16).reshape(17, 7)
        raw = [(words, words ^ np.uint16(0x8000))]
        calls = [0]
        def encoder(w, layer, kind):
            calls[0] += 1
            return wrapper.raw_matrix(w)
        config = {'candidates': {'base': {'kind': 'test'},
            'wrapped': {'kind': 'record_deflate', 'base': 'base', 'level': 6}}}
        cache = {}
        encoded, total, stage, dependency, extra = encode_candidate('wrapped', raw, config, {'base': encoder}, cache)
        self.assertEqual(calls[0], 4)
        self.assertTrue(extra['wrapper_inner_byte_identity'])
        self.assertEqual(extra['wrapped_record_count'], 4)
        self.assertEqual(total, stage + dependency)
        verify_cache(raw, encoded, lambda b, l, k: wrapper.decode_raw_matrix(wrapper.unwrap(b)))
        encode_candidate('base', raw, config, {'base': encoder}, cache)
        self.assertEqual(calls[0], 4)

    def test_control_factory_and_shared_state_accounting(self):
        config = {'candidates': {'huffman': {'kind': 'huffman'},
            'learned': {'kind': 'learned', 'options': {'correction': 'best_entropy', 'assignment': 'radius'}},
            'learned_wrap': {'kind': 'record_deflate', 'base': 'learned', 'level': 6}}}
        encoders, decoders, states, details = factories(config)
        self.assertEqual(states['huffman'], 0)
        self.assertEqual(states['learned'], 70095)
        self.assertEqual(states['learned_wrap'], states['learned'])
        self.assertEqual(details['learned_wrap']['wrapped_base'], 'learned')
        words = (np.arange(17 * 16, dtype=np.uint16).reshape(17, 16) % 128) + 15000
        raw = [(words, words ^ np.uint16(0x8000))]
        encoded, *_ = encode_candidate('learned_wrap', raw, config, encoders, {})
        verify_cache(raw, encoded, decoders['learned_wrap'])

    def test_missing_wrapper_dependency(self):
        with self.assertRaises(ValueError):
            factories({'candidates': {'bad': {'kind': 'record_deflate', 'base': 'absent'}}})


if __name__ == '__main__':
    unittest.main()
