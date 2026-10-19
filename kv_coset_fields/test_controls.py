import unittest
import struct
import numpy as np
from kv_coset_fields import controls

class ControlTests(unittest.TestCase):
    def test_controls(self):
        for strategy in ('deflate','huffman','split_deflate'):
            for lower in (np.arange(256,dtype=np.uint8).reshape(16,16),np.full((5,17),157,np.uint8),
                          np.empty((0,17),np.uint8),np.arange(60,dtype=np.uint8).reshape(5,12)[:,::2]):
                exp=np.full(lower.shape,125,np.uint8)
                encoded=controls.encode_lower(lower,exp,strategy)
                np.testing.assert_array_equal(lower,controls.decode_lower(encoded.data,exp,strategy))
                self.assertEqual(sum(encoded.stats['component_bytes'].values()),len(encoded.data))
                damaged=bytearray(encoded.data);damaged[-1]^=1
                with self.assertRaises(ValueError):controls.decode_lower(damaged,exp,strategy)

    def test_split_boundaries_and_padding(self):
        lower=np.array([[128,5,127]],dtype=np.uint8)
        exp=np.full(lower.shape,125,np.uint8)
        record=controls.encode_lower(lower,exp,'split_deflate')
        for data in (record.data[:3],struct.pack('<I',0)+record.data[4:],record.data+b'bad'):
            with self.assertRaises(ValueError):controls.decode_lower(data,exp,'split_deflate')
        # Properly checksummed wrappers with invalid packed padding still fail.
        from kv_stacked import wrapper
        a=wrapper.wrap(bytes([0x81]),6).data
        b=wrapper.wrap(bytes([0,0,0]),6).data
        bad=struct.pack('<I',len(a))+a+b
        with self.assertRaises(ValueError):controls.decode_lower(bad,exp,'split_deflate')

    def test_empty_huffman_protocol(self):
        exp=np.empty((0,7),dtype=np.uint8)
        for blob in (controls.HEADER.pack(b'KVLH',1,0,0)+b'\x00\x00',
                     controls.HEADER.pack(b'KVLH',0,0,0)+b'x',
                     controls.HEADER.pack(b'KVLH',0,0,1)):
            with self.assertRaises(ValueError):controls.decode_lower(blob,exp,'huffman')

if __name__=='__main__':unittest.main()
