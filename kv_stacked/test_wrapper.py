import unittest
import numpy as np
from kv_stacked.wrapper import wrap,unwrap,raw_matrix,decode_raw_matrix

class WrapperTest(unittest.TestCase):
    def test_roundtrip_modes_and_patterns(self):
        rng=np.random.default_rng(41)
        for payload in (b'',b'abc'*2000,rng.bytes(7000)):
            encoded=wrap(payload)
            self.assertEqual(unwrap(encoded),payload)
            self.assertEqual(sum(encoded.stats['component_bytes'].values()),len(encoded.data))
    def test_all_words_shuffles_and_tails(self):
        for shape in ((256,256),(0,64),(1,3)):
            x=np.arange(shape[0]*shape[1],dtype=np.uint16).reshape(shape)
            for shuffle in (False,True):
                raw=raw_matrix(x,shuffle)
                np.testing.assert_array_equal(decode_raw_matrix(unwrap(wrap(raw))),x)
    def test_corruption_and_bounds(self):
        e=wrap(b'prefix'*100).data
        for bad in (e[:5],e[:-1],e+b'x',e[:9]+b'\xff'*4+e[13:]):
            with self.assertRaises(ValueError):unwrap(bad)
        with self.assertRaises(ValueError):unwrap(e,max_bytes=2)

if __name__=='__main__':unittest.main()
