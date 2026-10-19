import unittest
import numpy as np
from kv_residual import direct

class DirectTests(unittest.TestCase):
    def test_exact_all_patterns_and_accounting(self):
        words=np.arange(65536,dtype=np.uint16).reshape(256,256)
        for strategy in ('best','rice','zigzag_deflate'):
            encoded=direct.encode_matrix(words,strategy)
            np.testing.assert_array_equal(words,direct.decode_matrix(encoded.data))
            self.assertEqual(sum(encoded.stats['component_bytes'].values()),len(encoded.data))
    def test_empty_tail_noncontiguous_and_damage(self):
        for words in (np.empty((0,7),np.uint16),np.arange(91,dtype=np.uint16).reshape(7,13)[:,::2]):
            encoded=direct.encode_matrix(words)
            np.testing.assert_array_equal(words,direct.decode_matrix(encoded.data))
            bad=bytearray(encoded.data);bad[-1]^=1
            with self.assertRaises(ValueError):direct.decode_matrix(bad)

if __name__=='__main__':unittest.main()
