import unittest
import numpy as np
from kv_research import container
from kv_smoke.codec import encode_matrix,decode_matrix

class ContainerTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91)
        self.data=[tuple(rng.integers(0,65536,(33,64),dtype=np.uint16) for _ in range(2)) for _ in range(2)]
        self.enc=lambda w,l,k:encode_matrix(w,'exponent')
        self.dec=lambda b,l,k:decode_matrix(b)
    def test_roundtrip_and_accounting(self):
        obj=container.encode_cache(self.data,self.enc)
        out=container.decode_cache(obj.data,self.dec)
        self.assertEqual(sum(obj.stats['component_bytes'].values()),len(obj.data))
        for a,b in zip(self.data,out):
            for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    def test_later_tile_independent_of_first(self):
        obj=container.encode_cache(self.data,self.enc)
        blob=bytearray(obj.data); s=container.shape(blob)
        blob[s['payload_start']]=0
        k,v=container.decode_tile(blob,1,2,self.dec)
        np.testing.assert_array_equal(k,self.data[1][0][32:])
        np.testing.assert_array_equal(v,self.data[1][1][32:])
    def test_truncation_rejected(self):
        obj=container.encode_cache(self.data,self.enc)
        with self.assertRaises(ValueError): container.decode_cache(obj.data[:-1],self.dec)

if __name__=='__main__':unittest.main()
