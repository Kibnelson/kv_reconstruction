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


if __name__=='__main__':unittest.main()
