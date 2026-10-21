from contextlib import nullcontext
from types import SimpleNamespace
import importlib.util
import os
import unittest
from unittest.mock import patch
from scripts import gpu_path_diagnostics as diagnostics


class DiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1' and importlib.util.find_spec('torch') and importlib.util.find_spec('triton'),
        'explicit CPU orchestration validation only')
    def test_prepared_encoder_controls_and_packet_only_decode_orchestration(self):
        import torch
        from qwen_kv_bench import transfer_t1 as t1
        from tests.test_transfer_t1 import model_and_words
        base,state,words,_=model_and_words();engine=t1.Engine(base,state,'cpu',_interpreter=True)
        # The proxy exercises control flow with interpreted kernels and no
        # timing. It cannot be used by the public CUDA benchmark entry point.
        proxy=SimpleNamespace(interpreter=False,_walk=engine._walk,_encode_prepared=engine._encode_prepared,
            encode=engine.encode,decode=engine.decode)
        def unmeasured(fn,*args):return fn(),dict(test_only_cpu_interpreter=True)
        def software_profile(fn,*args):return fn(),dict(status='test_only_cpu_interpreter',CUDA_activity_observed=False)
        with patch.object(diagnostics,'profile_cuda',side_effect=software_profile):
            r=diagnostics.run(proxy,torch.from_numpy(words),0,{'model':'synthetic'},None,torch,SimpleNamespace(type='cuda'),unmeasured)
        self.assertTrue(r['prepared_entropy_bytes_match_complete_encode'])
        self.assertFalse(r['decode_profile']['CUDA_activity_observed'])
        self.assertTrue(r['parameter_replay_gpu']['test_only_cpu_interpreter'])

    def test_profile_sums_device_events_without_double_counting_CPU_parents(self):
        cpu,cuda='CPU','CUDA'
        events=[SimpleNamespace(device_type=cuda,name='kernel',device_time_total=1500.),
            SimpleNamespace(device_type=cuda,name='kernel',device_time_total=2500.),
            SimpleNamespace(device_type=cuda,name='T1/walk',device_time_total=9000.),
            SimpleNamespace(device_type=cuda,name='another user range',device_time_total=8000.,is_user_annotation=True),
            SimpleNamespace(device_type=cpu,name='aten::parent',device_time_total=4000.)]
        averages=[SimpleNamespace(device_type=cpu,key='aten::mul',count=7,self_cpu_time_total=200.,cpu_time_total=300.,device_time_total=4000.),
            SimpleNamespace(device_type=cpu,key='T1/parameters',count=2,self_cpu_time_total=10.,cpu_time_total=400.,device_time_total=4000.)]
        profiler=SimpleNamespace(events=lambda:events,key_averages=lambda:averages)
        torch=SimpleNamespace(autograd=SimpleNamespace(DeviceType=SimpleNamespace(CPU=cpu,CUDA=cuda)))
        r=diagnostics.summarize_profile(profiler,torch,.02)
        self.assertEqual(r['CUDA_device_event_count'],2)
        self.assertEqual(r['CUDA_device_event_total_ms'],4.)
        self.assertEqual(r['excluded_CUDA_annotation_event_count'],2)
        self.assertEqual(r['Torch_operator_call_count'],7)
        self.assertEqual(r['profiled_synchronized_wall_ms'],20.)
        self.assertEqual(r['regions'][0]['name'],'T1/parameters')

    def test_instrumentation_restores_methods_after_an_exception(self):
        class Engine:
            def _walk(self):return 'words'
            def _mean(self):pass
            def _context(self):pass
            def _parameters(self):pass
            def _advance(self):pass
            def _encode_prepared(self):pass
        engine=Engine();torch=SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name:nullcontext()))
        with self.assertRaises(RuntimeError):
            with diagnostics.stage_regions(engine,torch):
                self.assertEqual(engine._walk(),'words')
                raise RuntimeError('diagnostic failed')
        self.assertEqual(engine.__dict__,{})
        self.assertEqual(engine._walk(),'words')

    def test_missing_CUDA_profiler_is_explicit_and_does_not_execute_a_fake_profile(self):
        torch=SimpleNamespace(profiler=SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU='CPU',CUDA='CUDA'),supported_activities=lambda:{'CPU'}))
        def forbidden():raise AssertionError('unsupported profile should not execute')
        value,report=diagnostics.profile_cuda(forbidden,None,torch,None)
        self.assertIsNone(value);self.assertFalse(report['CUDA_activity_observed'])
        self.assertEqual(report['status'],'CUDA_profiler_unavailable')


if __name__=='__main__':unittest.main()
