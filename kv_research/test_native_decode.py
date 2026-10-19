"""Exact native/reference agreement and malformed-record checks."""
import struct
import unittest
import numpy as np
try:
    from . import learned_codec as codec
    from .native_decode import NativeDecoder, decode_matrix
except ImportError:
    import learned_codec as codec
    from native_decode import NativeDecoder, decode_matrix

class NativeDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(104)
        cls.training=rng.integers(15500,15800,(160,64),dtype=np.uint16)
        cls.model=codec.train_model({'layer_00_K':cls.training},{'iterations':2,'max_samples_per_key':256})
        cls.decoder=NativeDecoder(cls.model)

    def test_real_encoder_all_words_tails_empty_and_compressed(self):
        rng=np.random.default_rng(330)
        cases=[np.arange(65536,dtype=np.uint16).reshape(256,256),
            rng.integers(15500,15800,(7,23),dtype=np.uint16),
            np.array([[0,0x8000,1,0x8001,0x7f80,0xff80,0x7fc0,0xffff,0x7f7f]],dtype=np.uint16),
            np.empty((0,64),dtype=np.uint16)]
        for x in cases:
            for group in (8,64,128):
                enc=codec.encode_matrix(x,self.model,'layer_00_K',group_size=group)
                np.testing.assert_array_equal(self.decoder.decode_matrix(enc),x)
                np.testing.assert_array_equal(decode_matrix(enc,self.model),codec.decode_matrix(enc,self.model))

    def test_every_correction_width(self):
        # Explicit compressed records cover widths 0..15, including padded tails,
        # independent of whether the encoder would choose a raw fallback.
        for qbits in (4,6,8):
            ncentroids=1<<qbits
            book=np.full((ncentroids,8),16384,dtype=np.uint16)
            model=codec.LearnedModel({'global':book},{'n_centroids':ncentroids},{})
            dec=NativeDecoder(model)
            widths=np.arange(16,dtype=np.uint16)
            rng=np.random.default_rng(71)
            values=rng.integers(0,32768,(16,8),dtype=np.uint16)
            ids=np.arange(16,dtype=np.uint16)
            signs=rng.integers(0,2,128,dtype=np.uint16)
            payload=codec._pack(ids,qbits)+codec._pack(widths,4)+codec._pack(signs,1)+codec._pack_corrections(values,widths)
            for count in (128,127):
                header=codec._RECORD_HEADER.pack(b'LKVR',1,qbits,0,128,1,count,bytes.fromhex(model.model_id),6)+b'global'
                encoded=header+codec._GROUP_HEADER.pack(1,len(payload))+payload
                np.testing.assert_array_equal(dec.decode_matrix(encoded),codec.decode_matrix(encoded,model))

    def test_malformed_records(self):
        encoded=codec.encode_matrix(self.training[:2],self.model,'layer_00_K').data
        for bad in (encoded[:10],encoded[:-1],encoded+b'x',encoded[:20]+bytes(32)+encoded[52:]):
            with self.assertRaises(ValueError): self.decoder.decode_matrix(bad)
        # Invalid mode and malformed sizes are rejected by the native parser.
        body_start=codec._RECORD_HEADER.size+len('layer_00_K')
        for position,replacement in ((body_start,b'\x02'),(body_start+1,b'\xff'*4)):
            bad=encoded[:position]+replacement+encoded[position+len(replacement):]
            with self.assertRaises(ValueError): self.decoder.decode_matrix(bad)

if __name__=='__main__': unittest.main()
