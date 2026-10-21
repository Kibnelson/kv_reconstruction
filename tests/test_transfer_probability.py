"""Exactness, causality, accounting and device parity of the four-arm study."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from qwen_kv_bench import transfer_reference as ref
from qwen_kv_bench import transfer_probability as c


def fixture(seed=1,tokens=24):
    rng=np.random.default_rng(seed);h=rng.normal(size=(2,tokens,6)).astype(np.float32)
    values=np.concatenate((h[...,:4],h[...,2:6]),axis=2)
    values=values+(0.03+np.abs(h[...,:1])*.15)*rng.normal(size=values.shape)
    f=values.astype(np.float32);u=f.view(np.uint32)
    words=((u+0x7fff+((u>>16)&1))>>16).astype(np.uint16)
    return dict(K_bits=words[...,:4],V_bits=words[...,4:],H=h,Hk=np.array(1),hd=np.array(4),
                base=np.array(10000.),positions=np.arange(tokens),ids=np.arange(tokens)+seed*100,sel=np.arange(2))


@unittest.skipUnless(importlib.util.find_spec('constriction'),'constriction required')
class ProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=ref.fit([fixture(1),fixture(2)],fixture(3),block_size=2,max_positions=65536)
        cls.base.state_sha256=ref._state_digest(cls.base)
        cls.b,cls.traces=c.fit_context(cls.base,[fixture(4)])
        cls.states={'A':c.empty_state(cls.base),'B':cls.b};cls.encoders={}
        for arm,kind in [('C','kv'),('D','hidden')]:
            cls.states[arm],cls.encoders[arm]=c.fit_hint(cls.base,cls.b,[fixture(4)],cls.traces,kind,bits=2,sketch_width=2)

    def test_original_A_stream_and_framing(self):
        a=fixture(8);source=ref.words(a)
        legacy,_=ref.encode(self.base,source)
        _,size=ref.PREFIX.unpack_from(legacy)
        packet,stats=c.encode(self.base,self.states['A'],a)
        self.assertEqual(c.parse_packet(self.base,self.states['A'],packet)[3],legacy[ref.PREFIX.size+size:])
        self.assertEqual(stats['framing_bytes'],c.HEADER.size)

    def test_all_arms_exact_and_hints_paid(self):
        a=fixture(9,tokens=7);before=ref._state_digest(self.base)
        for arm,state in self.states.items():
            ids=c.hint_ids(self.encoders[arm],c.hint_source(self.base,a,self.encoders[arm]['kind'])) if arm in self.encoders else None
            packet,stats=c.encode(self.base,state,a,ids)
            restored,decoded=c.decode(self.base,state,packet)
            np.testing.assert_array_equal(restored,ref.words(a))
            self.assertEqual(stats['parameter_sha256'],decoded['parameter_sha256'])
            self.assertEqual(len(packet),stats['framing_bytes']+stats['hint_bytes']+stats['main_stream_bytes'])
            self.assertEqual(stats['hint_bytes'],(2*7*2+7)//8 if ids is not None else 0)
        self.assertEqual(before,ref._state_digest(self.base))

    def test_future_channels_do_not_enter_current_context(self):
        a=fixture(10);changed=copy.deepcopy(a)
        changed['V_bits'][0,:,3]^=np.uint16(1)
        left=c.trace(self.base,a,self.b);right=c.trace(self.base,changed,self.b)
        # Last current-layer V group is unknown until all earlier groups decode.
        for x,y in zip(left[:4],right[:4]):
            np.testing.assert_array_equal(x['context'],y['context'])
            np.testing.assert_array_equal(x['mu'],y['mu'])

    def test_hint_padding_checked_and_reversible(self):
        for bits in (1,2,4):
            ids=(np.arange(15)%(1 << bits)).astype(np.uint8).reshape(3,5)
            packet=c.pack_ids(ids,bits)
            np.testing.assert_array_equal(c.unpack_ids(packet,bits,ids.shape),ids)
        with self.assertRaises(ValueError):c.unpack_ids(b'\xff',1,(1,1))
        with self.assertRaises(ValueError):c.pack_ids(np.array([4]),2)

    def test_corrupt_wrong_state_or_packet_rejected(self):
        a=fixture(11);state=self.states['B'];packet,_=c.encode(self.base,state,a)
        for p in (packet[:-1],packet[:30]+bytes([packet[30]^1])+packet[31:],packet[:-1]+bytes([packet[-1]^128])):
            with self.assertRaises((ValueError,AssertionError,RuntimeError,EOFError)):
                c.decode(self.base,state,p)
        wrong=copy.deepcopy(state);wrong.context_strength=0
        with self.assertRaises(ValueError):c.decode(self.base,wrong,packet)

    def test_state_and_encoder_reload_need_no_training_data(self):
        with tempfile.TemporaryDirectory() as td:
            state_path=Path(td)/'state.npz';encoder_path=Path(td)/'encoder.npz'
            c.save_state(self.states['D'],state_path);c.save_encoder(self.encoders['D'],encoder_path)
            state=c.load_state(state_path);encoder=c.load_encoder(encoder_path)
            a=fixture(12);ids=c.hint_ids(encoder,c.hint_source(self.base,a,'hidden'))
            packet,_=c.encode(self.base,state,a,ids)
            a.pop('H')
            restored,_=c.decode(self.base,state,packet)
            np.testing.assert_array_equal(restored,ref.words(a))

    def test_all_BF16_patterns_raw_fallback(self):
        words=np.arange(65536,dtype=np.uint16).reshape(2,4096,8)
        packet,stats=c.encode_words(self.base,self.states['A'],words)
        restored,_=c.decode(self.base,self.states['A'],packet)
        self.assertTrue(stats['raw_fallback'])
        np.testing.assert_array_equal(restored,words)

    def test_fixed_error_zero_has_zero_bucket(self):
        target=np.zeros((2,2),np.uint16);mu=np.zeros((2,2))
        bucket,signed,large=c._buckets(target,mu,np.zeros((2,10),np.int64))
        self.assertFalse(bucket.any());self.assertFalse(signed.any());self.assertFalse(large.any())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
    def test_device_integer_hints_and_actual_decode_equal_cpu(self):
        self._device_roundtrip('cpu')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
    def test_cuda_integer_hints_and_actual_decode_equal_cpu(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('actual CUDA device unavailable')
        self._device_roundtrip('cuda:0')

    def _device_roundtrip(self,device):
        from qwen_kv_bench.transfer_probability_gpu import TorchBackend
        a=fixture(13,tokens=3)
        for arm,state in self.states.items():
            backend=TorchBackend(self.base,state,device);ids=None
            if arm in self.encoders:
                encoder=self.encoders[arm];source=c.hint_source(self.base,a,encoder['kind'])
                ids=c.hint_ids(encoder,source)
                np.testing.assert_array_equal(ids,c.hint_ids(encoder,source,backend))
            packet,cpu_encoded=c.encode(self.base,state,a,ids)
            device_packet,device_encoded=c.encode(self.base,state,a,ids,backend)
            self.assertEqual(packet,device_packet)
            restored,decoded=c.decode(self.base,state,packet,backend)
            np.testing.assert_array_equal(restored,ref.words(a))
            self.assertEqual(decoded['parameter_sha256'],cpu_encoded['parameter_sha256'])

    def test_complete_runner_fresh_process_decode(self):
        from scripts import bench_transfer_probability as runner
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);acts=root/'acts';acts.mkdir()
            for name,seed in [('train0',1),('train1',2),('train2',3),('tables',4),('test_text',5)]:
                np.savez(acts/(name+'.npz'),**fixture(seed))
            result=runner.main(['--act-dir',str(acts),'--parameter-device','cpu','--device','cpu',
                '--block-size','2','--page-tokens','2','--page-starts','0','--test-cases','1',
                '--sketch-width','2','--repeats','1','--warmup','0','--output',str(root/'run')])
            self.assertEqual(result,0)
            r=json.loads((root/'run/results.json').read_text())
            self.assertEqual(set(r['aggregate']),set(c.ARMS))
            self.assertTrue(all(x['all_exact'] for x in r['aggregate'].values()))
            self.assertTrue(all(len(x['packet_only_decoder_inputs'])==3 for x in r['test_rows']))
            self.assertTrue(r['baseline_mean_state_unchanged'])
            self.assertFalse(r['device']['gpu'])
            from scripts.package_transfer_probability import package
            manifest=package(root/'run',root/'evidence')
            self.assertGreater(manifest['validation']['checks'],0)
            self.assertFalse((root/'evidence/calibration/base.npz').exists())
            self.assertTrue((root/'evidence/insights.json').is_file())
            packet=root/'run'/r['test_rows'][0]['packet']
            packet.write_bytes(packet.read_bytes()+b'X')
            with self.assertRaises(ValueError):package(root/'run',root/'bad_evidence')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
    def test_cuda_request_never_falls_back_to_cpu(self):
        from scripts import bench_transfer_probability as runner
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run'
            with mock.patch('torch.cuda.is_available',return_value=False):
                with self.assertRaises(RuntimeError):
                    runner.main(['--parameter-device','cuda:0','--output',str(root)])
            report=json.loads((root/'results.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertNotIn('test_rows',report)


if __name__=='__main__':unittest.main()
