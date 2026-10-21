import json
import contextlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from scripts import bench_gpu_path as run
from scripts.package_gpu_path import package
try:
    import torch
except ImportError:
    torch=None


class RunnerTests(unittest.TestCase):
    def test_failed_byte_parity_and_AB_pairing_are_preserved_per_input_chunk(self):
        def timing(wall):return dict(wall_median_ms=wall,cuda_interval_median_ms=wall,incremental_peak_allocated_bytes=1,
            samples=[dict(peak_allocated_bytes=10,peak_reserved_bytes=20)])
        def row(arm,chunk,size,digest):
            r=dict(dataset='software',name='test',arm=arm,predictor_input_chunk=chunk,complete_T_bpv=8.,external_saving_fraction=.5,
                raw_fallback=False,exact=True,serial_P_T_filesystem_T_P_seconds=1.,serial_cycle_memory={},complete_external_T_bytes=size,complete_T_sha256=digest)
            for key in ['kv_to_T_gpu_body','T_to_KV_gpu','P_to_T_gpu_body','T_to_P_gpu_composed','raw_control_P_to_words','raw_control_words_to_P']:r[key]=timing(10. if chunk==64 else 5.)
            return r
        # This synthetic failed report includes mismatched A bytes. Reporting
        # must retain that failure and pair B with its corresponding variant.
        r=dict(status='failed',coverage={'gpu_measured':False},attention=[],transfer=[row('A',64,1000,'A'),row('A',512,1200,'changed A'),row('B',64,900,'B'),row('B',512,900,'B')])
        h=run.highlights(r)
        self.assertFalse(h['chunk_vs_64'][0]['complete_T_bytes_identical'])
        self.assertTrue(h['chunk_vs_64'][1]['complete_T_bytes_identical'])
        self.assertAlmostEqual(h['B_vs_A'][1]['B_packet_saving_fraction'],.25)
        self.assertEqual(h['B_vs_A'][1]['predictor_input_chunk'],512)
        self.assertEqual(h['status'],'failed')

    def test_input_chunk_comparisons_require_successful_reference_order(self):
        self.assertEqual(run.input_chunks('64,512,0'),[64,512,0])
        for bad in ['512,64','64,-1','64,64','64,','']:
            with self.assertRaises(ValueError):run.input_chunks(bad)

    def test_profiler_is_requested_only_in_the_fresh_decoder_process(self):
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'storage').mkdir()
            args=argparse.Namespace(device='cuda:0',warmup=1,repeats=3)
            def child(cmd,**kwargs):
                output=Path(cmd[cmd.index('--output')+1]);output.mkdir(parents=True)
                (output/'decoder.json').write_text(json.dumps(dict(status='completed',gpu_measured=True,decoded_words_sha256='known hash',bottleneck_diagnostics={'test_only':True})))
                return type('Result',(),{'returncode':0,'stdout':'synthetic child control'})()
            with patch.object(run.subprocess,'run',side_effect=child) as process:
                r=run.isolated_decode(root/'packet.t1',{'model':'synthetic'},dict(kind='fixture',path=root),root,'software',args,'known hash',input_chunk=512,diagnose=True)
            cmd=process.call_args.args[0]
            self.assertIn('--diagnostics',cmd)
            self.assertEqual(cmd[cmd.index('--predictor-input-chunk')+1],'512')
            self.assertFalse(r['original_KV_supplied'])
    def test_missing_cuda_is_recorded_as_failure_without_cpu_success(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run.p,'_cuda_torch',side_effect=RuntimeError('CUDA unavailable')):
            out=Path(tmp)/'report'
            self.assertEqual(run.main(['--output',str(out)]),1)
            r=json.loads((out/'results.json').read_text())
            self.assertEqual(r['status'],'failed');self.assertFalse(r['coverage']['gpu_measured'])
            self.assertEqual(r['attention'],[]);self.assertEqual(r['transfer'],[])

    def test_packaging_never_includes_encoded_KV_or_capture_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'run';source.mkdir();(source/'storage').mkdir()
            r=dict(format='gpu_kv_path_batch_v1',status='completed',coverage=dict(gpu_measured=True))
            (source/'results.json').write_text(json.dumps(r));(source/'summary.md').write_text('Synthetic packaging test\n')
            (source/'storage'/'private.t1').write_bytes(b'private encoded KV')
            (source/'capture.npz').write_bytes(b'private capture')
            destination=package(source,root/'packaged')
            names={p.name for p in destination.iterdir()}
            self.assertEqual(names,{'results.json','summary.md','package_manifest.json'})
            with zipfile.ZipFile(destination.with_suffix('.zip')) as z:
                self.assertFalse(any('private' in x or 'capture' in x or 'storage' in x for x in z.namelist()))

    def test_failed_report_cannot_be_silently_packaged_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'run';source.mkdir()
            r=dict(format='gpu_kv_path_batch_v1',status='failed',coverage=dict(gpu_measured=False))
            (source/'results.json').write_text(json.dumps(r));(source/'summary.md').write_text('failed\n')
            with self.assertRaises(ValueError):package(source,root/'rejected')
            dest=package(source,root/'debug',allow_failed=True)
            m=json.loads((dest/'package_manifest.json').read_text());self.assertEqual(m['status'],'failed');self.assertFalse(m['gpu_measured'])


INTERPRETER=os.environ.get('TRITON_INTERPRET')=='1' and torch is not None and importlib.util.find_spec('triton') is not None


@unittest.skipUnless(INTERPRETER,'CPU orchestration test with actual interpreted kernels; no GPU timing')
class OrchestrationTests(unittest.TestCase):
    def test_attention_benchmark_uses_layer_sequence_with_both_cache_forms(self):
        import argparse
        import numpy as np
        from tests.test_pool_attention import bits
        from qwen_kv_bench import pool_p1 as p
        g=p.Layout(2,8,1,8,4);w=bits(np.random.default_rng(45).normal(size=g.shape))
        s=p.calibration(np.full((2,16),127,np.uint8),np.full((2,16),3,np.uint8))
        case=run.common.Case('synthetic',w,g,s,dict(token_start=0))
        args=argparse.Namespace(arena_ratio=1.,gqa_multiple=2)
        def unmeasured(fn,*a,**kw):return fn(),dict(test_only_cpu_interpreter=True)
        with patch.object(p,'_cuda_torch',return_value=(torch,torch.device('cpu'))),patch.object(torch.cuda,'device',return_value=contextlib.nullcontext()),patch.object(torch.cuda,'synchronize'),patch.object(run,'measure',side_effect=unmeasured):
            for tail in [0,3]:
                result=run.attention_case(case,args,torch,torch.device('cpu'),tail)
                self.assertTrue(result['original_dense_GPU_input_released_for_packed_timing'])
                self.assertTrue(result['paired_P1_released_for_dense_timing'])
                self.assertLess(result['maximum_output_abs_difference'],3e-6)

    def test_transfer_benchmark_orchestration_roundtrips_and_releases_source(self):
        import argparse
        import numpy as np
        from tests.test_transfer_t1 import model_and_words
        from qwen_kv_bench import pool_p1 as p,transfer_t1 as t1
        base,state,w,capture=model_and_words();e=t1.Engine(base,state,'cpu',_interpreter=True)
        g=p.Layout(2,1,1,8,4);s=p.calibration(np.full((2,16),127,np.uint8),np.full((2,16),3,np.uint8))
        case=run.common.Case('synthetic',w,g,s,dict(token_start=0,model_identity={'model':'synthetic'}))
        args=argparse.Namespace(arena_ratio=.1,gqa_multiple=2,cpu_t0_control=False)
        def unmeasured(fn,*a,**kw):return fn(),dict(test_only_cpu_interpreter=True)
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'_cuda_torch',return_value=(torch,torch.device('cpu'))),patch.object(torch.cuda,'device',return_value=contextlib.nullcontext()),patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'memory_allocated',return_value=0),patch.object(torch.cuda,'memory_reserved',return_value=0),patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(torch.cuda,'max_memory_reserved',return_value=0),patch.object(run,'measure',side_effect=unmeasured):
            result=run.transfer_case(case,e,args,torch,torch.device('cpu'),Path(tmp),'software',{'model':'synthetic'})
            self.assertTrue(result['exact']);self.assertTrue(result['restored_attention_matches_dense'])
            self.assertTrue(result['serial_cycle_memory']['source_pool_released_after_file_and_directory_fsync'])
            self.assertTrue(result['parameter_oracle']['mean_scale_offset_exact'])
            self.assertTrue(result['P']['requested_capacity_rejected'])
            accounting=result['T_packet_accounting']
            self.assertEqual(accounting['identity_mode'],'sha256')
            self.assertEqual(accounting['complete_bytes'],result['complete_external_T_bytes'])
            self.assertEqual(accounting['complete_bytes'],accounting['payload_bytes']+accounting['directory_bytes']+accounting['envelope_bytes'])
            self.assertEqual(result['identity_compaction_saved_bytes'],result['legacy_full_identity_accounting']['complete_bytes']-accounting['complete_bytes'])
            # The parent delegates selected diagnostics, never enabling its
            # own profiler even when a requested child profile is unavailable.
            with patch.object(run,'isolated_decode',return_value={'bottleneck_diagnostics':{'test_only_child':True}}),patch('scripts.gpu_path_diagnostics.run',side_effect=AssertionError('parent must not profile')):
                second=run.transfer_case(case,e,args,torch,torch.device('cpu'),Path(tmp),'software_tuned',{'model':'synthetic'},dict(kind='fixture'),
                    diagnose=True,expected_T_sha256=result['complete_T_sha256'])
            self.assertTrue(second['packet_bytes_match_chunk64_reference'])
            self.assertTrue(second['bottleneck_diagnostics']['test_only_child'])


if __name__=='__main__':unittest.main()
