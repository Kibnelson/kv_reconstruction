import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from qwen_kv_bench import matched_transfer as matched
from qwen_kv_bench import transfer_reference as ref, transfer_probability as prob
from qwen_kv_bench import transfer_t1 as t1, transfer_t1_reference as oracle
from qwen_kv_bench.transfer_gpu_predictor import numpy_prediction_sweep
from scripts import bench_qwen_matched_gpu as runner, package_qwen_matched_gpu as packager
from tests.test_pool_attention import bits
try:
    import torch
except ImportError:
    torch=None


def captures(channels=8):
    rng=np.random.default_rng(991)
    def one(n):
        return dict(K_bits=bits(rng.normal(scale=.25,size=(5,n,channels))),
            V_bits=bits(rng.normal(scale=.25,size=(5,n,channels))),
            Hk=np.array(1),hd=np.array(channels),base=np.array(10000.))
    return one(96),one(32),one(1)


def check_causal(test,device,interpreter=False):
    train,scale,capture=captures();capture['positions']=np.array([7])
    for mode in ['previous','aqua','block','strong_layers']:
        base=ref.fit([train],scale,mode=mode,block_size=8,max_positions=128)
        state,_=prob.fit_context(base,[train])
        e=t1.Engine(base,state,device,input_chunk=512,entropy_backend='cached',_interpreter=interpreter)
        words=ref.words(capture);source=torch.from_numpy(words.copy()).to(device)
        records=e._walk(source,7);rows=prob.trace(base,capture,state)
        for (_,_,mu,inverse,offset),r in zip(records,rows):
            expected=prob.offsets(state,r['context'],r['layer'],r['group'])
            np.testing.assert_array_equal(mu.cpu().numpy(),r['mu'])
            np.testing.assert_array_equal(offset.cpu().numpy(),expected)
            np.testing.assert_array_equal(inverse.cpu().numpy(),r['inverse']*state.scale_table[expected+prob.MAX_OFFSET,None])
        # The unmodified scalar symbol oracle checks all binary probabilities
        # and resulting bytes independently of the device decode path.
        means=np.concatenate([r[2].cpu().numpy()[0] for r in records])
        inv=np.concatenate([r[3].cpu().numpy()[0] for r in records])
        packet=e.encode(source,7,raw_fallback=False)
        test.assertEqual(packet.payload.cpu().numpy().tobytes(),oracle.encode_lane(words[:,0].ravel(),means,inv))
        del source,records,rows
        fresh=t1.Engine(base,state,device,input_chunk=512,entropy_backend='cached',_interpreter=interpreter)
        restored,error,_=fresh.decode(packet)
        np.testing.assert_array_equal(restored.cpu().numpy(),words);np.testing.assert_array_equal(error.cpu().numpy(),0)
        np.testing.assert_array_equal(e.p.prediction_sweep(torch.from_numpy(words.copy()).to(device),7).cpu().numpy(),
                                      numpy_prediction_sweep(base,words,7))


def check_permutation_packet(test,device,interpreter=False):
    train,scale,capture=captures();base=ref.fit([train],scale,mode='strong_layers',block_size=8,max_positions=128)
    state,_=prob.fit_context(base,[train]);name='strong_layers8_B'
    adapter=matched.Predictive(name,base,state,device,_interpreter=interpreter)
    # Source uses HF split-half order, not the predictor's adjacent order.
    original=matched.key_order(ref.words(capture),8,inverse=True)
    source=torch.from_numpy(original.copy()).to(device)
    record=adapter.encode(source,11);adapter.check_encode(record)
    wire,counts=matched.export_record(adapter,record,{'model':'synthetic5layer'})
    test.assertEqual(counts['complete_bytes'],len(wire))
    test.assertEqual(counts['complete_bytes'],counts['framing_bytes']+counts['payload_bytes'])
    del source,record,adapter
    fresh=matched.Predictive(name,base,state,device,_interpreter=interpreter)
    imported=matched.import_record(fresh,wire,{'model':'synthetic5layer'})
    decoded=fresh.decode(imported);fresh.check_decode()
    np.testing.assert_array_equal(decoded.cpu().numpy(),original)
    with test.assertRaises(ValueError):matched.import_record(fresh,wire,{'model':'different'})
    with test.assertRaises(ValueError):matched.import_record(fresh,wire[:-1]+bytes([wire[-1]^1]),{'model':'synthetic5layer'})


class ReferenceTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None,'Torch unavailable')
    def test_numpy2_with_legacy_nvcomp_is_refused_before_device_setup(self):
        import types
        nvidia=types.ModuleType('nvidia');nvcomp=types.ModuleType('nvidia.nvcomp');nvidia.nvcomp=nvcomp
        with mock.patch.dict('sys.modules',{'nvidia':nvidia,'nvidia.nvcomp':nvcomp}), \
                mock.patch('importlib.metadata.version',return_value='4.2.0.14'), \
                mock.patch.object(np,'__version__','2.3.5'), \
                mock.patch.object(torch.cuda,'current_stream',side_effect=AssertionError('device must not be touched')):
            with self.assertRaisesRegex(RuntimeError,'misreads NumPy 2'):
                matched.Nvcomp('Bitcomp',torch.device('cpu'))

    def test_fixed_width_arm_uses_original_frozen_means_without_refitting(self):
        train,scale,_=captures();base=ref.fit([train],scale,mode='strong_layers',block_size=8,max_positions=128)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'calibration/wikitext2';directory.mkdir(parents=True)
            p=directory/'strong_layers8_base.npz';ref.save_model(base,p)
            runner.write(directory/'fit_manifest.json',dict(arms={'original_B':dict(mode='strong_layers',group=8,
                base_file_sha256=runner.sha_file(p))}))
            with mock.patch.dict(runner.FIXED_WIDTH_ABLATIONS,{'strong_layers8_A':('strong_layers',8,'A')}), \
                    mock.patch.object(ref,'fit',side_effect=AssertionError('no refitting')), \
                    mock.patch.object(matched,'Predictive') as predictor:
                runner.adapter_for('strong_layers8_A','wikitext2',Path(tmp),'cpu')
                args=predictor.call_args.args
                self.assertEqual(args[1].state_sha256,base.state_sha256);self.assertEqual(args[2].arm,'A')
                self.assertEqual(args[2].base_sha256,base.state_sha256)
                p.write_bytes(p.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'differs from the original fit manifest'):
                    runner.adapter_for('strong_layers8_A','wikitext2',Path(tmp),'cpu')

    def test_reused_capture_report_cannot_overwrite_the_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);report=path/'results.json';report.write_text('original evidence')
            r=subprocess.run([sys.executable,str(Path(runner.__file__)), '--output',tmp,
                '--captures-from',tmp,'--phase','bench'],capture_output=True,text=True)
            self.assertEqual(r.returncode,2)
            self.assertIn('original report is preserved',r.stderr)
            self.assertEqual(report.read_text(),'original evidence')
    def test_distinct_layers_deduplicated_and_causal(self):
        self.assertEqual([ref.context_layers('strong_layers',l) for l in range(6)],
                         [(),(0,),(0,1),(0,1,2),(0,2,3),(0,3,4)])
        train,scale,capture=captures()
        base=ref.fit([train],scale,mode='strong_layers',block_size=8,max_positions=128)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'base.npz';ref.save_model(base,p);loaded=ref.load_model(p)
            self.assertEqual(loaded.state_sha256,base.state_sha256)
            for arm in ['A','B']:
                state=prob.empty_state(loaded,arm)
                packet,stats=prob.encode(loaded,state,capture)
                self.assertFalse(stats['raw_fallback'])
                decoded,_=prob.decode(loaded,state,packet)
                np.testing.assert_array_equal(decoded,ref.words(capture))
            packet,_=ref.encode(loaded,ref.words(capture));decoded,_=ref.decode(loaded,packet)
            np.testing.assert_array_equal(decoded,ref.words(capture))

    def test_pair_permutation_preserves_every_original_word(self):
        words=np.arange(2*3*256,dtype=np.uint16).reshape(2,3,256)
        adjacent=matched.key_order(words,128)
        np.testing.assert_array_equal(adjacent[0,0,:8],[0,64,1,65,2,66,3,67])
        np.testing.assert_array_equal(matched.key_order(adjacent,128,inverse=True),words)
        np.testing.assert_array_equal(adjacent[...,128:],words[...,128:])

    def test_sampling_is_reproducible_and_does_not_restart_positions(self):
        train,_,_=captures();selected,meta=runner.select_rows([train],7,71)
        again,meta2=runner.select_rows([train],7,71)
        self.assertEqual(meta,meta2);self.assertEqual(meta['selected_rows'],7)
        np.testing.assert_array_equal(selected[0]['K_bits'],train['K_bits'][:,selected[0]['positions']])
        np.testing.assert_array_equal(selected[0]['positions'],again[0]['positions'])
        class Tokenizer:
            eos_token_id=2
            def encode(self,text,add_special_tokens=False):return list(map(int,text.split()))
        rows=enumerate([{'text':'3 4 5 6 7 8'},{'text':'9 10 11 12 13'}])
        a=next(runner.blocks(rows,Tokenizer(),1,3));b=next(runner.blocks(rows,Tokenizer(),1,3))
        self.assertEqual(a['ids'],[3,4,5]);self.assertEqual(b['ids'],[9,10,11])
        self.assertNotEqual(a['ranges'][0]['row'],b['ranges'][0]['row'])

    def test_coverage_does_not_count_overlapping_requests_twice(self):
        entry=dict(name='heldout',tokens=100,layers=4,shape=[100,8])
        c=runner.coverage([entry],[16,32])
        self.assertEqual(c['contexts'][0]['merged_token_intervals'],[[0,32],[68,100]])
        self.assertEqual(c['unique_test_word_positions'],4*64*16)

    def test_report_pairs_rates_and_times_with_the_same_B_control(self):
        def row(method,size,enc,dec):
            return dict(track='software',method=method,tokens=32,values=80,
                bytes=dict(complete_bytes=size,payload_bytes=size),encode=dict(wall_median_ms=enc),
                decode=dict(wall_median_ms=dec),raw_chunks=0,exact=True)
        result=dict(status='software_check',gpu_measured=False,availability={'software':{
            'nvcomp_bitcomp':dict(status='available',persistent_device_bytes=None)}},
            rows=[row('block32_B',80,1.,1.),row('strong_layers32_B',64,2.,4.)])
        with tempfile.TemporaryDirectory() as tmp:
            runner.save_report(Path(tmp),result)
            comparison=result['comparisons_vs_block32_B'][0]
            self.assertAlmostEqual(comparison['bpv_saved_vs_block32_B'],1.6)
            self.assertEqual(comparison['encode_speedup_vs_block32_B'],.5)
            self.assertEqual(comparison['decode_speedup_vs_block32_B'],.25)
            self.assertIn('unmeasured',(Path(tmp)/'summary.md').read_text())
        result['rows'][0]['method']='block32_A';runner.summarize(result)
        self.assertEqual(result['comparisons_vs_block32_B'],[])
        self.assertAlmostEqual(result['comparisons_vs_block32_A'][0]['bpv_saved_vs_block32_A'],1.6)

    @unittest.skipUnless(torch is not None,'Torch unavailable')
    def test_raw_framing_and_publishable_allowlist(self):
        a=matched.Raw(torch.device('cpu'));words=np.arange(64,dtype=np.uint16).reshape(2,2,16)
        packet=a.encode(torch.from_numpy(words),3);wire,counts=matched.export_record(a,packet,{'model':'test'})
        np.testing.assert_array_equal(a.decode(matched.import_record(a,wire,{'model':'test'})).numpy(),words)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir()
            report=dict(format='qwen_matched_gpu_report_v1',status='failed',gpu_measured=False)
            (source/'results.json').write_text(json.dumps(report));(source/'summary.md').write_text('software check')
            (source/'raw-KV.t').write_bytes(b'never publish');(source/'token_plan.json').write_text('private')
            target=packager.package(source,Path(tmp)/'report')
            self.assertEqual({p.name for p in target.iterdir()},{'results.json','summary.md','package_manifest.json'})


INTERPRETER=torch is not None and os.environ.get('TRITON_INTERPRET')=='1' and importlib.util.find_spec('triton') is not None
CUDA=torch is not None and torch.cuda.is_available() and importlib.util.find_spec('triton') is not None


@unittest.skipUnless(INTERPRETER,'explicit interpreter software check only')
class InterpreterTests(unittest.TestCase):
    def test_all_predictor_ablation_means_bytes_and_causal_decode(self):check_causal(self,'cpu',True)
    def test_original_layout_portable_packet_and_fresh_decoder(self):check_permutation_packet(self,'cpu',True)

    def test_future_columns_are_rejected(self):
        train,scale,_=captures();base=ref.fit([train],scale,mode='strong_layers',block_size=8,max_positions=128)
        base.predictors[4][0]['columns'][0]=len(ref.context_layers('strong_layers',4))*16
        base.state_sha256=ref._state_digest(base);state=prob.empty_state(base,'A')
        with self.assertRaisesRegex(ValueError,'unavailable decoder context'):
            t1.Engine(base,state,'cpu',_interpreter=True)

    @unittest.skipUnless(os.environ.get('SPLITZIP_VALIDATION_SOURCE'),'optional pinned native source absent')
    def test_native_splitzip_escape_accounting_and_packet_only_decode(self):
        source=Path(os.environ['SPLITZIP_VALIDATION_SOURCE'])
        rng=np.random.default_rng(77);words=rng.integers(0,65536,(2,1,1028),dtype=np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp)/'state.npz'
            adapter=matched.SplitZip('cpu',source,torch.from_numpy(words.copy()),state)
            record=adapter.encode(torch.from_numpy(words.copy()),9)
            self.assertGreater(record.attributes['n_esc'],0)
            payload=sum(p.numel()*p.element_size() for p in record.parts.values())
            self.assertEqual(payload,words.size*3//2+7*record.attributes['n_esc'])
            wire,_=matched.export_record(adapter,record,{'model':'native software check'})
            del record,adapter
            fresh=matched.SplitZip('cpu',source,state_path=state)
            decoded=fresh.decode(matched.import_record(fresh,wire,{'model':'native software check'}))
            np.testing.assert_array_equal(decoded.numpy(),words)


@unittest.skipUnless(CUDA,'actual CUDA hardware unavailable')
class CudaTests(unittest.TestCase):
    def test_all_predictor_ablation_means_bytes_and_causal_decode(self):check_causal(self,'cuda:0')
    def test_original_layout_portable_packet_and_fresh_decoder(self):check_permutation_packet(self,'cuda:0')
    def test_production_group64_causal_decode(self):
        train,scale,capture=captures(64)
        base=ref.fit([train],scale,mode='strong_layers',block_size=64,max_positions=128)
        state,_=prob.fit_context(base,[train]);engine=t1.Engine(base,state,'cuda:0',entropy_backend='cached',input_chunk=512)
        words=ref.words(capture);packet=engine.encode(torch.from_numpy(words.copy()).to('cuda:0'),7,raw_fallback=False)
        decoded,error,_=engine.decode(packet)
        np.testing.assert_array_equal(decoded.cpu().numpy(),words);np.testing.assert_array_equal(error.cpu().numpy(),0)


@unittest.skipUnless(CUDA,'actual CUDA hardware unavailable')
class CudaNativeTests(unittest.TestCase):
    def check_codec(self,algorithm):
        try:import nvidia.nvcomp
        except ImportError as error:self.skipTest('optional native library unavailable: '+str(error))
        device=torch.device('cuda:0');stream=torch.cuda.Stream(device=device)
        # All BF16 patterns, including NaN payloads, infinities and signed zero.
        words=np.arange(65536,dtype=np.uint16).reshape(2,16,2048)
        with torch.cuda.stream(stream):
            adapter=matched.Nvcomp(algorithm,device);source=torch.from_numpy(words.copy()).to(device)
            record=adapter.encode(source,0)
            wire,_=matched.export_record(adapter,record,{'model':'native smoke; every BF16 pattern'})
            del record,source,adapter
            fresh=matched.Nvcomp(algorithm,device)
            output=fresh.decode(matched.import_record(fresh,wire,{'model':'native smoke; every BF16 pattern'}))
            torch.cuda.synchronize(device);np.testing.assert_array_equal(output.cpu().numpy(),words)

    def test_bitcomp_typed_word_roundtrip_before_capture_or_fit(self):self.check_codec('Bitcomp')
    def test_cascaded_typed_word_roundtrip_before_capture_or_fit(self):self.check_codec('Cascaded')


if __name__=='__main__':unittest.main()
