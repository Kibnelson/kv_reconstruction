import struct
import unittest
from dataclasses import dataclass
import numpy as np
from kv_coset_fields import combined
from kv_stacked import huffman_codec

@dataclass
class Fake:
    data:bytes
    stats:dict

class CombinedTests(unittest.TestCase):
    def test_fallback_identical_and_all_words(self):
        words=np.arange(65536,dtype=np.uint16).reshape(256,256)
        def larger(lower,exp):return Fake(lower.tobytes()+b'overhead',{})
        encoded=combined.encode_matrix(words,larger)
        self.assertEqual(encoded.data,huffman_codec.encode_matrix(words).data)
        recovered=combined.decode_matrix(encoded.data,lambda b,e:None)
        np.testing.assert_array_equal(words,recovered)

    def test_selected_layout_exact_and_corruption(self):
        exp=np.tile(np.arange(64,dtype=np.uint16),(16,1))+100
        words=((exp<<7)|7).astype(np.uint16)
        def enc(lower,exponents):
            assert np.all(lower==7)
            return Fake(b'\x07',{'component_bytes':{'constant':1}})
        def dec(blob,exponents):
            if len(blob)!=1:raise ValueError('invalid constant')
            return np.full(exponents.shape,blob[0],np.uint8)
        encoded=combined.encode_matrix(words,enc)
        self.assertEqual(encoded.data[:4],combined.MAGIC)
        self.assertEqual(sum(encoded.stats['component_bytes'].values()),len(encoded.data))
        np.testing.assert_array_equal(words,combined.decode_matrix(encoded.data,dec,2))
        damaged=bytearray(encoded.data);damaged[-1]^=1
        with self.assertRaises(ValueError):combined.decode_matrix(damaged,dec,2)
        with self.assertRaises(ValueError):combined.decode_matrix(encoded.data,dec,3)
        with self.assertRaises(ValueError):combined.decode_matrix(encoded.data[:20],dec)

    def test_fields_preserve_every_word(self):
        words=np.arange(65536,dtype=np.uint16).reshape(256,256)
        lower,exp=combined.fields(words)
        np.testing.assert_array_equal(words,((lower.astype(np.uint16)&128)<<8)|(lower&127)|(exp.astype(np.uint16)<<7))

    def test_forced_expansion_is_explicit(self):
        words=np.full((4,8),0x3f80,np.uint16)
        def enc(lower,exponents):return Fake(lower.tobytes()+b'extra',{})
        def dec(blob,exponents):return np.frombuffer(blob[:-5],dtype=np.uint8).copy().reshape(exponents.shape)
        baseline=huffman_codec.encode_matrix(words)
        encoded=combined.encode_matrix(words,enc,audit_decoder=dec,allow_expansion=True)
        self.assertGreater(len(encoded.data),len(baseline.data))
        self.assertEqual(encoded.stats['forced_expansion']['records'],1)
        np.testing.assert_array_equal(words,combined.decode_matrix(encoded.data,dec))

if __name__=='__main__':unittest.main()
