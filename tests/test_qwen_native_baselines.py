"""Portable framing, causal AQUA reconstruction and native CUDA preflight."""
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
from qwen_kv_bench import matched_transfer as matched, matched_aqua as aqua, matched_exans as exans
from scripts import bench_qwen_matched_gpu as runner
from tests.test_qwen_matched_gpu import captures
try:import torch
except ImportError:torch=None

ROOT=Path(__file__).resolve().parents[1]
AQUA_SOURCE=Path(os.environ.get('AQUA_VALIDATION_SOURCE',ROOT/'runs/_matched_baselines/aqua'))
CUDA=torch is not None and torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET')!='1'


def check_aqua(test,device):
    train,scale,target=captures(128);train['positions']=np.arange(96)
    target=dict(target,K_bits=np.repeat(target['K_bits'],3,axis=1),V_bits=np.repeat(target['V_bits'],3,axis=1))
    g=dict(layers=5,heads=2,head_dim=64,theta=10000.)
    for bits in (2,4):
        with tempfile.TemporaryDirectory() as tmp:
            meta,arrays,sha=aqua.fit_state(AQUA_SOURCE,[train],g,bits,device,'TRAIN only software fixture')
            test.assertIn('reconstructed',meta['training_context'])
            path=Path(tmp)/'a.npz';aqua.save_state(path,meta,arrays,sha)
            adapter=aqua.Aqua(device,AQUA_SOURCE,path)
            words=np.concatenate((target['K_bits'],target['V_bits']),axis=-1)
            source=torch.from_numpy(words.copy()).to(device);record=adapter.encode(source,19)
            decoded=adapter.decode(record);quality=runner.reconstruction_quality(words,decoded.cpu().numpy())
            test.assertGreater(quality['relative_mse'],0);test.assertFalse(adapter.lossless)
            wire,counts=matched.export_record(adapter,record,{'model':'synthetic AQUA'})
            test.assertEqual(counts['complete_bytes'],len(wire))
            test.assertEqual(counts['payload_bytes'],sum(x.numel()*x.element_size() for x in record.parts.values()))
            # Packed bits + TWO BF16 affine parameters per group, raw first
            # layer and BF16 cos/sin. The portable envelope is additional.
            # Quanto packs along the grouped tensor's first dimension, and
            # this deliberately odd row count adds padding to each plane.
            groups=3*128//64
            per_later_kind=((groups+(8//bits)-1)//(8//bits))*64+groups*4
            test.assertEqual(counts['payload_bytes'],4*2*per_later_kind+2*3*128*2+2*3*64*2)
            expected=decoded.cpu().numpy().copy();del source,record,decoded,adapter
            fresh=aqua.Aqua(device,AQUA_SOURCE,path)
            restored=fresh.decode(matched.import_record(fresh,wire,{'model':'synthetic AQUA'}))
            np.testing.assert_array_equal(restored.cpu().numpy(),expected)
            # Changing the saved coefficients must invalidate frozen state.
            arrays['k1_bias'][0]^=np.uint16(1);aqua.save_state(path,meta,arrays,sha)
            with test.assertRaisesRegex(ValueError,'state hash mismatch'):aqua.Aqua(device,AQUA_SOURCE,path)


class ReferenceTests(unittest.TestCase):
    def test_physical_exans_geometry_and_native_header(self):
        self.assertEqual(exans.FRAME_OFFSET,500304)
        self.assertEqual([exans.block_count(n) for n in [2,exans.RAW_BYTES,exans.RAW_BYTES+2]],[1,1,2])
        header=bytearray(64);struct.pack_into('<IHHQ',header,0,exans.HEADER_MAGIC,2,64,0)
        struct.pack_into('<II',header,16,exans.RAW_BYTES,exans.FRAME_OFFSET+16)
        struct.pack_into('<II',header,40,exans.FRAME_OFFSET,16)
        self.assertEqual(exans.valid_native_length(header),exans.FRAME_OFFSET+16)
        struct.pack_into('<I',header,20,exans.RECORD_BYTES+1)
        with self.assertRaisesRegex(ValueError,'invalid native'):exans.valid_native_length(header)

    @unittest.skipUnless(torch is not None,'Torch unavailable')
    def test_exans_raw_fallback_packet_restores_without_native_call(self):
        adapter=exans.Exans.__new__(exans.Exans);adapter.device='cpu';adapter.max_blocks=1
        adapter.state_sha256='software raw fallback';words=np.arange(48,dtype=np.uint16).reshape(2,3,8)
        data=torch.from_numpy(words.copy()).view(torch.uint8).reshape(-1)
        record=matched.Record(words.shape,5,{'0':data},dict(native_block_bytes=exans.RAW_BYTES,
            native_slot_bytes=exans.RECORD_BYTES,blocks=[dict(kind='raw',original_bytes=data.numel(),stored_bytes=data.numel())]))
        wire,counts=matched.export_record(adapter,record,{'model':'software fallback'})
        self.assertEqual(counts['payload_bytes'],words.nbytes)
        np.testing.assert_array_equal(adapter.decode(matched.import_record(adapter,wire,{'model':'software fallback'})).numpy(),words)

    def test_lossy_rates_are_excluded_from_lossless_rankings(self):
        def row(method,lossless):
            return dict(track='software',method=method,tokens=32,values=80,bytes=dict(complete_bytes=40,payload_bytes=20),
                encode=dict(wall_median_ms=1.),decode=dict(wall_median_ms=1.),raw_chunks=0,exact=True,lossless=lossless,
                quality=dict(squared_error_sum=0.,original_squared_sum=10.,word_match_fraction=1.))
        result=dict(status='software',gpu_measured=False,availability={},rows=[row('block32_A',True),row('aquakv_quanto2',False)])
        with tempfile.TemporaryDirectory() as tmp:
            runner.save_report(Path(tmp),result)
            self.assertEqual(result['comparisons_vs_block32_A'],[])
            text=(Path(tmp)/'summary.md').read_text();self.assertIn('KV relative MSE',text)
            self.assertNotIn('aquakv_quanto2',text.split('AQUA-KV with released')[0])

    def test_known_reconstruction_metrics(self):
        original=np.full((2,3,8),0x3f80,dtype=np.uint16);restored=original.copy();restored[1,2,7]=0x4000
        q=runner.reconstruction_quality(original,restored)
        self.assertEqual(q['word_match_fraction'],47/48);self.assertEqual(q['relative_mse'],1/48)
        self.assertEqual(q['per_layer_kind'][-1]['squared_error_sum'],1.)

    @unittest.skipUnless(torch is not None and AQUA_SOURCE.is_dir(),'optional pinned AQUA source absent')
    def test_released_aqua_path_portable_packing_and_fresh_decoder(self):check_aqua(self,'cpu')

    @unittest.skipUnless(torch is not None and AQUA_SOURCE.is_dir(),'optional pinned AQUA source absent')
    def test_aqua_later_fit_uses_reconstructed_inputs(self):
        train,_,_=captures(64);train=dict(train,K_bits=train['K_bits'][:3],V_bits=train['V_bits'][:3],positions=np.zeros(96))
        api=aqua.load_upstream(AQUA_SOURCE);inputs=[]
        def zero_solver(x,y,**kwargs):
            inputs.append(x.clone())
            return torch.zeros((y.shape[1],x.shape[1]),dtype=x.dtype),torch.zeros(y.shape[1],dtype=x.dtype)
        api.fit_linear_regression=zero_solver
        with mock.patch.object(aqua,'load_upstream',return_value=api):
            aqua.fit_state(AQUA_SOURCE,[train],dict(layers=3,heads=1,head_dim=64,theta=10000.),2,'cpu','causal-fit-test')
        original_k=torch.from_numpy(train['K_bits'][1].copy()).view(torch.bfloat16)
        original_v=torch.from_numpy(train['V_bits'][1].copy()).view(torch.bfloat16)
        q=api.QuantoQuantizer(2,q_group_size=64,axis=0)
        reconstructed_k=q.quantize_dequantize(original_k);reconstructed_v=q.quantize_dequantize(original_v)
        self.assertFalse(torch.equal(reconstructed_k,original_k))
        self.assertTrue(torch.equal(inputs[2],reconstructed_k))
        self.assertTrue(torch.equal(inputs[3][:,64:],reconstructed_v))


@unittest.skipUnless(CUDA,'actual CUDA hardware unavailable')
class CudaTests(unittest.TestCase):
    # Optional dependencies are REQUIRED for this explicitly invoked comparison.
    # Missing sources, build/runtime errors and absent libraries fail preflight.
    def test_aqua_both_widths_reconstructed_training_and_packet_only_decode(self):
        with torch.cuda.stream(torch.cuda.Stream()):check_aqua(self,'cuda:0')

    def test_splitzip_all_bf16_patterns_and_fresh_decoder(self):
        words=np.arange(65536,dtype=np.uint16).reshape(2,16,2048)
        source=ROOT/'runs/_matched_baselines/splitzip/codec_gpu.py'
        with tempfile.TemporaryDirectory() as tmp,torch.cuda.stream(torch.cuda.Stream()):
            path=Path(tmp)/'state.npz';original=torch.from_numpy(words.copy()).to('cuda:0')
            adapter=matched.SplitZip('cuda:0',source,original,path);record=adapter.encode(original,17)
            wire,_=matched.export_record(adapter,record,{'model':'all patterns'})
            del adapter,record,original
            fresh=matched.SplitZip('cuda:0',source,state_path=path)
            np.testing.assert_array_equal(fresh.decode(matched.import_record(fresh,wire,{'model':'all patterns'})).cpu().numpy(),words)

    def test_exans_native_record_all_patterns_and_overflow_fallback(self):
        source=ROOT/'runs/_matched_baselines/exans';n=exans.RAW_BYTES//2
        # All 65536 patterns embedded in mostly compressible data. Require
        # the REAL native ANS record, so a raw fallback cannot mask a bug.
        words=np.full(n,0x3f80,dtype=np.uint16);words[:65536]=np.arange(65536,dtype=np.uint16)
        cases=[(words.reshape(2,244,1024),'exans'),
               (np.arange(n,dtype=np.uint32).astype(np.uint16).reshape(2,244,1024),'raw')]
        with torch.cuda.stream(torch.cuda.Stream()):
            for words,expected_mode in cases:
                adapter=exans.Exans('cuda:0',source,words.nbytes)
                try:
                    original=torch.from_numpy(words.copy()).to('cuda:0');record=adapter.encode(original,7)
                    self.assertEqual(record.attributes['blocks'][0]['kind'],expected_mode)
                    wire,counts=matched.export_record(adapter,record,{'model':'native ExANS edge cases'})
                    self.assertEqual(counts['payload_bytes'],record.parts['0'].numel())
                finally:adapter.close()
                del adapter,record,original
                fresh=exans.Exans('cuda:0',source,words.nbytes)
                try:
                    output=fresh.decode(matched.import_record(fresh,wire,{'model':'native ExANS edge cases'}))
                    np.testing.assert_array_equal(output.cpu().numpy(),words)
                finally:fresh.close()


if __name__=='__main__':unittest.main()
