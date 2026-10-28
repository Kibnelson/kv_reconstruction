"""Complete packet checks for entropy parallelism, including inactive lanes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from kv_compare.lite_stream_layout import geometry,fetch_budgets

HAVE_RUNTIME=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('triton') is not None
if HAVE_RUNTIME:
    import torch
    from kv_compare.causal_packet import Packet,float_to_words,word_sha,sha
    from kv_compare.lite_fast import FastEngine
    from kv_compare.lite_fused import FusedEngine
    from tests.test_llama_causal import tiny_state


class AccountingTests(unittest.TestCase):
    def test_stream_states_directory_and_inactive_lanes_are_charged(self):
        g=[geometry(65536,v,b) for v,b in ((128,8),(64,8),(32,8),(64,32),(32,32))]
        self.assertEqual([x['programs'] for x in g],[64,128,256,32,64])
        self.assertEqual([x['state_and_directory_bpv'] for x in g],[.3125,.625,1.25,.53125,1.0625])
        tail=geometry(19,8,8)
        self.assertEqual(tail['streams'],3); self.assertEqual(tail['state_bytes'],32)
        self.assertEqual(tail['directory_bytes'],8)
        with self.assertRaises(ValueError): geometry(0,32,8)

    def test_larger_packets_tighten_the_decode_target(self):
        baseline=dict(method='splitzip_official_framed',packet_bytes=1000000,decode_ms=1.,mode='lossless',kappa=0)
        a=dict(method='lite_a',packet_bytes=500000,decode_ms=10.,mode='lossless',kappa=0)
        b=dict(a,method='lite_b',packet_bytes=750000)
        rows=fetch_budgets([baseline,a,b],links=[1.])
        self.assertEqual([x['decode_budget_ms'] for x in rows],[5.,3.])
        self.assertTrue(all(not x['measured_network'] for x in rows))
        self.assertEqual(fetch_budgets([a]),[])

    def test_package_refuses_cpu_success_and_keeps_packets_local(self):
        from scripts import package_lite_streams as package
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'; run.mkdir()
            (run/'results.json').write_text(json.dumps(dict(status='completed',gpu_measured=False)))
            with self.assertRaisesRegex(ValueError,'CPU evidence'):
                package.main([str(run),'--destination',str(Path(tmp)/'bad')])
            (run/'results.json').write_text(json.dumps(dict(status='failed',gpu_measured=False,jobs={})))
            (run/'summary.md').write_text('software fixture only\n')
            (run/'private.kvc').write_bytes(b'local only')
            self.assertEqual(package.main([str(run),'--destination',str(Path(tmp)/'report')]),0)
            self.assertEqual({p.name for p in (Path(tmp)/'report').iterdir()},
                {'results.json','summary.md','package_manifest.json'})


def packet_parity(test,device):
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=False
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); tiny_state(root)
        # The CPU interpreter runs the same dynamic loops with fewer steps;
        # actual CUDA preflight exercises the exact production configurations.
        configurations=(((128,8),(64,8),(32,8),(64,32),(32,32)) if device.type=='cuda'
                        else ((8,8),(4,8),(2,8),(4,32),(2,32)))
        options=dict(predictor_format='bf16',values_per_stream=configurations[0][0],block=8)
        control=FusedEngine(root,device,**options)
        words=float_to_words(torch.linspace(-.35,.35,2*7*8).reshape(2,7,8)).to(device,torch.int16)
        # Tiny requests deliberately have inactive lanes and partial streams.
        for kappa in (0.,.5):
            p=control.encode(words,start=7,kappa=kappa); wire=p.export()
            intended=control.last_reconstruction.clone()
            for v,b in configurations:
                with test.subTest(v=v,b=b,kappa=kappa):
                    opts=dict(options,values_per_stream=v,block=b)
                    encoder=FusedEngine(root,device,**opts)
                    packet=encoder.encode(words,start=7,kappa=kappa)
                    test.assertEqual(packet.header['values_per_stream'],v)
                    test.assertEqual(packet.header['block'],b)
                    test.assertTrue(torch.equal(encoder.last_reconstruction,intended))
                    decoded=encoder.decode(Packet.restore(packet.export(),device))
                    test.assertTrue(torch.equal(decoded,intended))
                    # A reference implementation can decode the new wire using
                    # only the options described in that packet's header.
                    reference=FastEngine(root,device,**opts)
                    test.assertTrue(torch.equal(reference.decode(Packet.restore(packet.export(),device)),intended))
                    if not kappa: test.assertTrue(torch.equal(decoded,words))
                    if (v,b)==configurations[0]: test.assertEqual(packet.export(),wire)
            patterns=torch.arange(65536,dtype=torch.int32).to(device,torch.int16).reshape(2,4096,8)
            raw=encoder.encode(patterns)
            test.assertTrue(raw.header['raw'])
            test.assertTrue(torch.equal(encoder.decode(raw),patterns))


@unittest.skipUnless(HAVE_RUNTIME and os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_complete_stream_layouts_reconstruct_identical_words(self): packet_parity(self,torch.device('cpu'))

    def test_complete_worker_and_packet_only_receiver(self):
        from scripts import bench_lite_streams as bench
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state=root/'state'; state.mkdir(); tiny_state(state)
            words=float_to_words(torch.arange(48).float().reshape(2,3,8)/100).to(torch.int16)
            source=root/'input.npy'; np.save(source,words.numpy().astype('<u2'),allow_pickle=False)
            control=FusedEngine(state,'cpu',values_per_stream=8,block=8)
            p=control.encode(words,start=7); original=root/'original.kvc'; original.write_bytes(p.export())
            entry=dict(document='tiny',input=str(source),input_file_sha256=sha(source),reference_packet=str(original),
                reference_packet_sha256=sha(original),original_word_sha256=word_sha(words),
                decoded_word_sha256=word_sha(words),start=7)
            q=dict(method='lite_v2_b8_fixture',codec=dict(state=str(state),engine_options=dict(
                predictor_format='bf16',values_per_stream=2,block=8)),entries=[entry],
                output=str(root/'output'),warmup=1,repeats=2,require_original_packet=False)
            bench.worker(q,torch,torch.device('cpu'))
            result=json.loads((root/'output/worker_results.json').read_text())
            self.assertEqual(result['status'],'software_checked'); self.assertFalse(result['gpu_measured'])
            row=result['rows'][0]
            self.assertTrue(row['original_exact']); self.assertEqual(row['status'],'verified')
            self.assertNotEqual(row['packet_sha256'],entry['reference_packet_sha256'])
            proof=json.loads((root/'output/receiver.json').read_text())
            self.assertFalse(proof['gpu_measured']); self.assertTrue(proof['no_source_kv_or_predictions'])

    def test_gpu_entry_never_substitutes_cpu_results(self):
        from scripts import bench_lite_streams as bench
        from unittest import mock
        if torch.cuda.is_available(): self.skipTest('CPU-only host required')
        with tempfile.TemporaryDirectory() as tmp,mock.patch('sys.stderr'):
            self.assertEqual(bench.main(['--output',tmp]),1)
            result=json.loads((Path(tmp)/'results.json').read_text())
            self.assertEqual(result['status'],'failed'); self.assertFalse(result['gpu_measured'])

    def test_raw_fallback_charges_no_entropy_states(self):
        from scripts import bench_lite_streams as bench
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state=root/'state'; state.mkdir(); tiny_state(state)
            words=torch.arange(65536,dtype=torch.int32).to(torch.int16).reshape(2,4096,8)
            source=root/'input.npy'; np.save(source,words.numpy().astype('<u2'),allow_pickle=False)
            reference=FusedEngine(state,'cpu',values_per_stream=128,block=8).encode(words)
            path=root/'original.kvc'; path.write_bytes(reference.export())
            e=dict(document='patterns',input=str(source),input_file_sha256=sha(source),
                reference_packet=str(path),reference_packet_sha256=sha(path),
                original_word_sha256=word_sha(words),decoded_word_sha256=word_sha(words),start=0)
            q=dict(method='lite_raw_fixture',codec=dict(state=str(state),engine_options=dict(
                predictor_format='bf16',values_per_stream=32,block=8)),entries=[e],
                output=str(root/'output'),warmup=1,repeats=2,require_original_packet=False)
            bench.worker(q,torch,torch.device('cpu'))
            result=json.loads((root/'output/worker_results.json').read_text()); r=result['rows'][0]
            self.assertEqual(r['mandatory_stream_state_bytes'],0); self.assertEqual(r['directory_bytes'],0)
            self.assertEqual(r['coder_geometry'],[]); self.assertEqual(r['payload_bytes'],131072)
            self.assertTrue(r['original_exact'])


@unittest.skipUnless(HAVE_RUNTIME and torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'),'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_stream_layout_packets(self): packet_parity(self,torch.device('cuda',torch.cuda.current_device()))


if __name__=='__main__': unittest.main()
