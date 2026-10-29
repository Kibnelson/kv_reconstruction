"""Integration byte/word parity; software interpreter checks are not timings."""
import gc
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from kv_compare import mixed_transfer as mt
from kv_compare.causal_packet import Packet, sha
from kv_compare.framed_baselines import Baseline
from kv_compare.mixed_fast import FastMixedTransfer
from kv_compare.mixed_decode_fast import DecodeFastMixedTransfer
from tests.test_mapped_splitzip import Encoded, tables, oracle
from tests.test_mixed_transfer import FakeBaseline, state


class OracleEncoded(Encoded):
    @property
    def compressed_bytes(self):
        return (self.n + 1)//2 + self.n + 7*self.n_esc


class OracleCodec:
    """Independent bit-field software oracle; never a native performance claim."""
    def __init__(self, device='cpu'):
        self.__dict__.update(tables(device).__dict__)

    def encode(self, tensor):
        words=tensor.contiguous().view(torch.int16).reshape(1,1,-1)
        value=oracle(self,words,torch.tensor([0],dtype=torch.int64,device=words.device))
        return OracleEncoded(**value.__dict__)

    def decode(self, e):
        codes=torch.empty(e.n,dtype=torch.int64,device=e.pk.device)
        codes[0::2]=e.pk.long() >> 4;codes[1::2]=e.pk.long() & 15
        exp=self.dec_lut[codes].long()
        raw=((e.sm.long() & 128)<<8)|(exp<<7)|(e.sm.long() & 127)
        pos=e.chunk_id.long()*1024+e.local_pos.long()
        raw[pos]=((e.sm[pos].long() & 128)<<8)|(e.esc_val.long()<<7)|(e.sm[pos].long() & 127)
        return raw.to(torch.int16).view(torch.bfloat16)


class OracleBaseline(FakeBaseline):
    def __init__(self, method, device, **kwargs):
        super().__init__(method,device,**kwargs)
        self.codec=OracleCodec(device)
        self.module=SimpleNamespace(ChunkLocalGPUEncoded=OracleEncoded)


class ContractTests(unittest.TestCase):
    def test_execution_flags_reject_implicit_numeric_options(self):
        for flags in (dict(mapped_io=1),dict(selective_decode=1)):
            with self.assertRaises(ValueError):
                DecodeFastMixedTransfer('missing','missing','missing','x',[True],'cpu',**flags)

    def test_zero_mask_keeps_original_whole_request_frame(self):
        with tempfile.TemporaryDirectory() as raw:
            _,lut=state(raw)
            with mock.patch.object(mt,'verify_splitzip_source',return_value={}),mock.patch.object(mt,'Baseline',OracleBaseline):
                codec=DecodeFastMixedTransfer('missing','unused',lut,sha(lut),[False,False],'cpu',
                    mapped_io=True,selective_decode=True)
                reference=OracleBaseline('splitzip_official_framed','cpu',state=lut)
                words=torch.ones(2,64,256,dtype=torch.bfloat16).view(torch.int16)
                packet=codec.encode(words)
                self.assertEqual(packet.export(),reference.encode(words).export())
                self.assertTrue(torch.equal(codec.decode(packet),words))
                self.assertFalse(codec.execution_plan['mapped_io_active'])


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_all_four_modes_preserve_complete_wire_and_original_words(self):
        with tempfile.TemporaryDirectory() as raw:
            path,lut=state(raw,layers=3,tokens=64)
            words=torch.randn(3,64,256,generator=torch.Generator().manual_seed(712)).bfloat16().view(torch.int16)
            words[0,0,:11]=torch.tensor([0,-32768,1,127,128,0x3f80,-16512,0x7f80,-128,0x7fc1,-1],dtype=torch.int16)
            original=words.clone();mask=[False,True,False]
            with mock.patch.object(mt,'verify_splitzip_source',return_value={}),mock.patch.object(mt,'Baseline',OracleBaseline):
                reference=FastMixedTransfer(path,'unused',lut,sha(lut),mask,'cpu',arm='dense_uniform',bundle=True)
                wire=reference.encode(words).export()
                held=None
                for mapped,context in ((False,False),(True,False),(False,True),(True,True)):
                    codec=DecodeFastMixedTransfer(path,'unused',lut,sha(lut),mask,'cpu',arm='dense_uniform',
                        selective=False,bundle=True,mapped_io=mapped,selective_decode=context)
                    packet=codec.encode(words);self.assertEqual(packet.export(),wire)
                    value=codec.decode(Packet.restore(wire,'cpu'))
                    self.assertTrue(torch.equal(value,original));self.assertTrue(torch.equal(words,original))
                    if held is not None:self.assertTrue(torch.equal(held,original))
                    held=value
                    accounting=codec.accounting(packet)
                    self.assertEqual(accounting['packet_bytes'],len(wire))
                    self.assertEqual(accounting['fallback_decoded_buffer_bytes'],0 if mapped else 2*2*64*256)
                    if context:self.assertEqual(codec.execution_plan['decoder_context_layers'],[])
                    self.assertEqual(packet.export(),wire)
                later=codec.decode(Packet.restore(wire,'cpu'))
                self.assertNotEqual(held.data_ptr(),later.data_ptr());self.assertTrue(torch.equal(held,original))

    def test_damaged_frames_rejected_before_mapped_or_native_decode(self):
        with tempfile.TemporaryDirectory() as raw:
            path,lut=state(raw,layers=3,tokens=64);mask=[False,True,False]
            with mock.patch.object(mt,'verify_splitzip_source',return_value={}),mock.patch.object(mt,'Baseline',OracleBaseline):
                codec=DecodeFastMixedTransfer(path,'unused',lut,sha(lut),mask,'cpu',arm='dense_uniform',
                    selective=False,bundle=True,mapped_io=True,selective_decode=True)
                words=torch.ones(3,64,256,dtype=torch.bfloat16).view(torch.int16)
                wire=codec.encode(words).export()
                packet=Packet.restore(wire,'cpu');packet.header['extra']['fallback_layers']=[1]
                with mock.patch('kv_compare.mixed_decode_fast.decode_layers_into') as mapped:
                    with self.assertRaises(ValueError):codec.decode(packet)
                    mapped.assert_not_called()
                with self.assertRaises(ValueError):codec.decode_native(object())


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET')!='1','actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_full_wire_exact_words_and_borrowed_packet_lifetimes(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'runs/_llama_official/sources/splitzip'
        mt.verify_splitzip_source(source)
        self.assertIn('RTX A4500',torch.cuda.get_device_name(0))
        with tempfile.TemporaryDirectory() as raw:
            path,_=state(raw,layers=3,tokens=256,rank=True)
            baseline=Baseline('splitzip_official_framed','cuda:0',source=source)
            book=tables('cuda:0')
            for name in ('enc_lut','enc_lut_marked','common_lut','dec_lut'):
                setattr(baseline.codec,name,getattr(book,name))
            import numpy as np
            lut=Path(raw)/'actual_luts.npz'
            np.savez(lut,**{name:getattr(baseline.codec,name).cpu().numpy()
                for name in ('enc_lut','enc_lut_marked','common_lut','dec_lut')})
            patterns=torch.arange(65536,dtype=torch.int32,device='cuda:0').to(torch.int16).reshape(256,256)
            words=torch.ones(3,256,256,dtype=torch.bfloat16,device='cuda:0').view(torch.int16)
            words[0].copy_(patterns);words[2].copy_(patterns.flip(0));original=words.clone()
            mask=[False,True,False]
            reference=FastMixedTransfer(path,source,lut,sha(lut),mask,'cuda:0',selective=True,bundle=True)
            wire=reference.encode(words).export();held=[]
            for mapped,context in ((False,False),(True,False),(False,True),(True,True)):
                codec=DecodeFastMixedTransfer(path,source,lut,sha(lut),mask,'cuda:0',
                    mapped_io=mapped,selective_decode=context)
                packet=codec.encode(words);self.assertEqual(packet.export(),wire)
                stream=torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    incoming=Packet.restore(wire,'cuda:0');value=codec.decode(incoming);del incoming
                gc.collect();torch.cuda.synchronize()
                self.assertTrue(torch.equal(value,original));held.append(value)
                later=codec.decode(Packet.restore(wire,'cuda:0'));torch.cuda.synchronize()
                self.assertTrue(torch.equal(later,original))
                self.assertTrue(all(torch.equal(old,original) for old in held))
                self.assertTrue(torch.equal(words,original));self.assertEqual(packet.export(),wire)


if __name__=='__main__':unittest.main()
