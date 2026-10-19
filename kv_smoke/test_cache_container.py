import unittest
import struct
import numpy as np
from codec import float32_to_bf16_words
from cache_container import encode_cache, decode_cache, decode_cache_tile, cache_shape

METHODS=["raw","exponent","zero_coset","lattice2_coset","lattice4_coset","lattice2_sign_coset"]

class CacheTests(unittest.TestCase):
    def fixtures(self):
        rng=np.random.default_rng(20261018)
        return [(float32_to_bf16_words(rng.standard_normal((17,24),dtype=np.float32)),
                 float32_to_bf16_words(rng.standard_normal((17,24),dtype=np.float32)))
                for _ in range(2)]

    def test_full_and_partial_tiles(self):
        original=self.fixtures()
        for method in METHODS:
            encoded=encode_cache(original,method,group_size=7,tile_tokens=5)
            recovered=decode_cache(encoded.data)
            for a,b in zip(original,recovered):
                for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
            shape=cache_shape(encoded.data)
            for layer in range(2):
                for tile in range(shape["ntiles"]):
                    k,v,start=decode_cache_tile(encoded.data,layer,tile)
                    np.testing.assert_array_equal(k,original[layer][0][start:start+5])
                    np.testing.assert_array_equal(v,original[layer][1][start:start+5])

    def test_later_tile_does_not_decode_earlier_payload(self):
        original=self.fixtures()
        encoded=encode_cache(original,"exponent",tile_tokens=5)
        damaged=bytearray(encoded.data)
        first_payload=struct.unpack_from("<Q",damaged,32)[0]
        damaged[first_payload:first_payload+4]=b"BAD!"
        k,v,start=decode_cache_tile(bytes(damaged),1,3)
        np.testing.assert_array_equal(k,original[1][0][start:start+5])
        np.testing.assert_array_equal(v,original[1][1][start:start+5])

    def test_truncation_rejected(self):
        encoded=encode_cache(self.fixtures(),"raw",tile_tokens=5)
        with self.assertRaises(ValueError): decode_cache(encoded.data[:-1])

if __name__=="__main__":
    unittest.main()

