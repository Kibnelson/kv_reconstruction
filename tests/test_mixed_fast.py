"""Lossless mixed execution tests; CPU fixtures are not native performance."""
import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare import mixed_transfer as mt
from kv_compare import mixed_fast as fast
from kv_compare.causal_packet import Packet, sha
from kv_compare.framed_baselines import Baseline, unframe
from tests.test_mixed_transfer import FakeBaseline, FakeSZ, state


class CountSZ(FakeSZ):
    def __init__(self, device):
        super().__init__(device); self.encodes = 0; self.decodes = 0
    def encode(self, tensor):
        self.encodes += 1
        return super().encode(tensor)
    def decode(self, encoded):
        self.decodes += 1
        return super().decode(encoded)


class CountBaseline(FakeBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codec = CountSZ(self.device)


def fields(packet):
    _, parts, _ = unframe(packet, packet.header['method'], packet.header['state_sha256'])
    return parts


class ContractTests(unittest.TestCase):
    def test_execution_flags_are_explicit_and_dense_selective_is_rejected(self):
        for options in (dict(selective=1), dict(bundle=1), dict(selective=True, arm='dense_uniform')):
            with self.assertRaises(ValueError):
                fast.FastMixedTransfer('missing', 'missing', 'missing', 'x', [True], 'cpu', **options)

    def test_zero_mask_still_uses_whole_request_baseline_without_predictor(self):
        with tempfile.TemporaryDirectory() as raw:
            _, lut = state(raw)
            with mock.patch.object(mt, 'verify_splitzip_source', return_value={}), mock.patch.object(mt, 'Baseline', CountBaseline), mock.patch.object(mt, 'load_state') as load:
                codec = fast.FastMixedTransfer('missing', 'unused', lut, sha(lut), [False, False], 'cpu',
                    selective=True, bundle=True)
                load.assert_not_called()
                words = torch.arange(2*64*256, dtype=torch.int32).to(torch.int16).reshape(2,64,256)
                reference = CountBaseline('splitzip_official_framed', 'cpu', state=lut)
                packet = codec.encode(words)
                self.assertEqual(packet.export(), reference.encode(words).export())
                self.assertTrue(torch.equal(codec.decode(packet), words))
                self.assertFalse(codec.bundled)
                self.assertEqual(codec.other.codec.encodes, 1)
                self.assertEqual(codec.other.codec.decodes, 1)
                self.assertEqual(codec.execution_plan['feature_passes'], 0)
                self.assertEqual(codec.accounting(packet)['bundle_index_bytes'], 0)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_bundle_has_one_native_call_and_same_rans_with_owned_exact_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            path, lut = state(raw, layers=3)
            words = torch.randn(3,64,256, generator=torch.Generator().manual_seed(904)).bfloat16().view(torch.int16)
            words[0,0,:11] = torch.tensor([0,-32768,1,127,128,0x3f80,-16512,0x7f80,-128,0x7fc1,-1], dtype=torch.int16)
            mask = [False, True, False]
            with mock.patch.object(mt, 'verify_splitzip_source', return_value={}), mock.patch.object(mt, 'Baseline', CountBaseline):
                reference = mt.MixedTransfer(path,'unused',lut,sha(lut),mask,'cpu',arm='dense_uniform')
                expected = reference.encode(words)
                codec = fast.FastMixedTransfer(path,'unused',lut,sha(lut),mask,'cpu',arm='dense_uniform',bundle=True)
                packet = codec.encode(words); wire = packet.export()
                self.assertEqual(reference.other.codec.encodes, 2)
                self.assertEqual(codec.other.codec.encodes, 1)
                for name in ('rans_words', 'rans_offsets', 'rans_relative'):
                    self.assertTrue(torch.equal(fields(expected)[name], fields(packet)[name]))
                restored = Packet.restore(wire, 'cpu')
                held = codec.decode(restored)
                self.assertEqual(codec.other.codec.decodes, 1)
                changed = words.clone(); changed[:,0,:] ^= 0x21
                later = codec.decode(codec.encode(changed))
                self.assertTrue(torch.equal(held, words))
                self.assertTrue(torch.equal(later, changed))
                self.assertNotEqual(held.data_ptr(), later.data_ptr())
                self.assertEqual(packet.export(), wire)
                accounting = codec.accounting(packet)
                self.assertEqual(accounting['packet_bytes'], len(wire))
                self.assertEqual(accounting['execution_plan']['native_splitzip_decode_calls'], 1)
                self.assertEqual(accounting['fallback_gather_bytes'], 2*2*64*256)
                self.assertEqual(accounting['expanded_arena_bytes'], 0)

    def test_bundle_checks_directory_order_and_escapes_before_native_use(self):
        with tempfile.TemporaryDirectory() as raw:
            path, lut = state(raw)
            words = torch.ones(2,64,256,dtype=torch.bfloat16).view(torch.int16)
            with mock.patch.object(mt, 'verify_splitzip_source', return_value={}), mock.patch.object(mt, 'Baseline', CountBaseline):
                codec = fast.FastMixedTransfer(path,'unused',lut,sha(lut),[True,False],'cpu',arm='dense_uniform',bundle=True)
                wire = codec.encode(words).export()
                for name in ('rans_offsets', 'bundle_chunk_id'):
                    packet = Packet.restore(wire,'cpu'); fields(packet)[name][0] = -1
                    with mock.patch.object(codec.native,'decode_mixed') as native, mock.patch.object(codec.other.codec,'decode') as sz:
                        with self.assertRaises(ValueError): codec.decode(packet)
                        native.assert_not_called(); sz.assert_not_called()
                packet = Packet.restore(wire,'cpu'); packet.header['extra']['fallback_layers'] = [0]
                with self.assertRaises(ValueError): codec.decode(packet)
                packet = codec.encode(words); packet.header['extra']['fallback_layers'][0] = 0
                with self.assertRaises(ValueError): codec.decode(packet)
                self.assertEqual(codec.fallback_layers, (1,))
                packet = Packet.restore(wire,'cpu'); packet.header['extra']['splitzip_bundle']['n'] += 1
                with self.assertRaises(ValueError): codec.decode(packet)
                with self.assertRaises(ValueError): codec.decode_native(object())
                broken = bytearray(wire); broken[-33] ^= 1
                with self.assertRaises(ValueError): Packet.restore(broken,'cpu')

    def test_selective_and_combined_preserve_their_reference_wire(self):
        with tempfile.TemporaryDirectory() as raw:
            path, lut = state(raw, tokens=256, rank=True)
            words = torch.ones(2,256,256,dtype=torch.bfloat16).view(torch.int16)
            with mock.patch.object(mt,'verify_splitzip_source',return_value={}), mock.patch.object(mt,'Baseline',CountBaseline):
                codecs = [fast.FastMixedTransfer(path,'unused',lut,sha(lut),[False,True],'cpu',
                    selective=s, bundle=b) for s,b in ((False,False),(True,False),(False,True),(True,True))]
                identity = torch.eye(256,dtype=torch.int8)
                for codec in codecs:
                    codec.native.requant = 'kernel'
                    codec.rank['Bq'].copy_(identity.expand(2,-1,-1))
                    codec.rank['Uq'].copy_(identity.expand(2,-1,-1))
                    codec.rank['m1'].fill_(0.25)
                packets = [c.encode(words) for c in codecs]
                self.assertEqual(packets[0].export(), packets[1].export())
                self.assertEqual(packets[2].export(), packets[3].export())
                for p in packets[1:]:
                    for name in ('rans_words','rans_offsets','rans_relative'):
                        self.assertTrue(torch.equal(fields(packets[0])[name],fields(p)[name]))
                self.assertTrue(torch.equal(codecs[-1].decode(Packet.restore(packets[-1].export(),'cpu')), words))
                self.assertEqual(codecs[-1].execution_plan['surprise_passes'], 0)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def _source_and_lut(self, raw):
        source = Path(__file__).resolve().parents[1]/'runs/_llama_official/sources/splitzip'
        mt.verify_splitzip_source(source)
        codec = Baseline('splitzip_official_framed','cuda:0',source=source)
        lut = Path(raw)/'actual_luts.npz'
        sample = torch.randn(65536,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(891)).bfloat16().view(torch.int16)
        codec.calibrate(sample,lut)
        return source,lut

    def test_actual_gpu_bundle_and_selective_wire_parity(self):
        with tempfile.TemporaryDirectory() as raw:
            path,_ = state(raw,layers=3,tokens=256,rank=True)
            source,lut = self._source_and_lut(raw)
            words = torch.randn(3,256,256,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(893)).bfloat16().view(torch.int16)
            original = words.clone()
            for mask in ([False,True,False], [False,True,True], [True,True,True]):
                codecs = [fast.FastMixedTransfer(path,source,lut,sha(lut),mask,'cuda:0',selective=s,bundle=b)
                    for s,b in ((False,False),(True,False),(False,True),(True,True))]
                packets = [c.encode(words) for c in codecs]
                wires = [p.export() for p in packets]
                self.assertEqual(wires[0],wires[1]); self.assertEqual(wires[2],wires[3])
                for p in packets[1:]:
                    for name in ('rans_words','rans_offsets','rans_relative'):
                        self.assertTrue(torch.equal(fields(packets[0])[name],fields(p)[name]))
                held = codecs[-1].decode(Packet.restore(wires[-1],'cuda:0'))
                self.assertTrue(torch.equal(held,original))
                self.assertTrue(torch.equal(codecs[-1].decode(packets[-1]),original))
                self.assertTrue(torch.equal(held,original))
                self.assertTrue(torch.equal(words,original))
                self.assertEqual(packets[-1].export(),wires[-1])
                self.assertEqual(codecs[-1].accounting(packets[-1])['packet_bytes'],len(wires[-1]))

    def test_actual_gpu_extrema_cross_stream_ownership_and_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            path,_ = state(raw,tokens=256)
            source,lut = self._source_and_lut(raw)
            words = torch.ones(2,256,256,dtype=torch.bfloat16,device='cuda:0').view(torch.int16)
            words[0] = torch.arange(65536,dtype=torch.int32,device='cuda:0').to(torch.int16).reshape(256,256)
            original = words.clone()
            codec = fast.FastMixedTransfer(path,source,lut,sha(lut),[False,True],'cuda:0',arm='dense_uniform',bundle=True)
            wire = codec.encode(words).export(); packet = Packet.restore(wire,'cuda:0')
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                held = codec.decode(packet)
                del packet
            gc.collect(); torch.cuda.synchronize()
            self.assertTrue(torch.equal(held,original))
            other_stream = torch.cuda.Stream()
            packet = Packet.restore(wire,'cuda:0')
            with torch.cuda.stream(other_stream):
                second = codec.decode(packet)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(second,original)); self.assertTrue(torch.equal(held,original))
            self.assertNotEqual(held.data_ptr(),second.data_ptr())
            packet = Packet.restore(wire,'cuda:0'); fields(packet)['bundle_chunk_id'][0] = -1
            with mock.patch.object(codec.native,'decode_mixed') as native, mock.patch.object(codec.other.codec,'decode') as sz:
                with self.assertRaises(ValueError): codec.decode(packet)
                native.assert_not_called(); sz.assert_not_called()


if __name__ == '__main__': unittest.main()
