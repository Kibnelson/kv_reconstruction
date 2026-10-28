"""Bitwise checks for execution-only changes, including the copied loop guard."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np
import torch

from kv_compare.causal_packet import Engine, Packet, float_to_words, json_bytes, sha, write_json
from kv_compare.lite_fast import FastEngine
from kv_compare.lite_fused import FusedEngine, _run_columns, scatter_columns
from tests.test_llama_causal import tiny_state


class StructureTests(unittest.TestCase):
    def test_execution_loop_only_replaces_four_unique_column_assignments(self):
        original = ast.parse(textwrap.dedent(inspect.getsource(Engine._run)))
        class Stores(ast.NodeTransformer):
            count = 0
            def visit_Assign(self, node):
                targets = {
                    'cur[:, pos]': ('cur', 'pos'),
                    'recovered[l][:, ch]': ('recovered[l]', 'ch'),
                    'self.last_reconstruction[l][:, ch]': ('self.last_reconstruction[l]', 'ch'),
                    "f[:, layer['perm']]": ('f', "layer['perm']"),
                }
                target = ast.unparse(node.targets[0])
                if target not in targets:
                    return node
                self.count += 1
                dst, col = targets[target]
                call = ast.parse(f'self.scatter({dst}, {col}, value)').body[0]
                call.value.args[-1] = node.value
                return call
        edit = Stores(); changed = edit.visit(original)
        changed.body[0].name = '_run_columns'
        actual = ast.parse(textwrap.dedent(inspect.getsource(_run_columns)))
        self.assertEqual(edit.count, 4)
        self.assertEqual(ast.dump(changed, include_attributes=False), ast.dump(actual, include_attributes=False))

    def test_profile_does_not_add_operators_ranges_and_memory_to_kernel_times(self):
        from types import SimpleNamespace as NS
        from kv_compare.lite_execution_profile import summarize
        def event(name,lo,hi,user=False,cpu=False):
            return NS(name=name,time_range=NS(start=lo,end=hi),is_user_annotation=user,
                device_type=torch.autograd.DeviceType.CPU if cpu else torch.autograd.DeviceType.CUDA,
                self_device_time_total=hi-lo)
        events=[event('_decode_kernel',0,100),event('_scatter_columns',80,140),
            event('lite/rotation',0,200,True),event('aten::index',0,100,cpu=True),
            event('Memcpy DtoH (Device -> Pinned)',150,190)]
        p=NS(events=lambda:events,key_averages=lambda:[])
        out=summarize(p,1.)
        self.assertEqual(out['cuda_kernel_calls'],2)
        self.assertAlmostEqual(out['cuda_kernel_duration_sum_ms'],.16)
        self.assertAlmostEqual(out['cuda_kernel_busy_union_ms'],.14)
        self.assertEqual(len(out['device_transfers_and_memsets']),1)

    def test_package_refuses_cpu_success(self):
        from scripts.package_lite_execution import verify
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'CPU evidence'):
                verify(Path(tmp),dict(status='completed',gpu_measured=False))


def compare_rotations(test, device):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); tiny_state(root)
        for layout in ('half', 'adjacent'):
            for scale in (1., 1.25):
                # Build valid pair groups for each layout, retaining reordered
                # key and value channels. These are checked by Engine itself.
                tiny_state(root)
                if layout == 'adjacent':
                    for entry in json.loads((root / 'state.json').read_text())['files']:
                        path = root / entry['path']
                        with np.load(path) as archive:
                            a = dict(archive)
                        a['perm'] = np.arange(8)
                        np.savez(path, **a)
                meta = json.loads((root / 'state.json').read_text())
                meta.update(rope_layout=layout, attention_scaling=scale)
                for entry in meta['files']:
                    entry['sha256'] = sha(root / entry['path'])
                meta.pop('state_sha256'); meta['state_sha256'] = hashlib.sha256(json_bytes(meta)).hexdigest()
                write_json(root / 'state.json', meta)
                ref, fused = FastEngine(root, device), FusedEngine(root, device)
                for start, n in ((0, 3), (7, 5), (8190, 2)):
                    positions = torch.arange(start, start+n, device=device)
                    for g in ref.layers[0]['groups']:
                        for dtype in (torch.float32, torch.float64):
                            # Strided source, zeros and exact powers test masks,
                            # partner lookup, signs and output bit patterns.
                            x = torch.linspace(-.75, .75, n*g['columns']*2,
                                dtype=dtype, device=device).reshape(n, -1)[:, ::2]
                            x[0, 0] = -0.
                            with ref.rope.request(start, n, device), fused.rope.request(start, n, device):
                                for inverse in (False, True):
                                    a, b = ref.rotate(x,g,positions,inverse), fused.rotate(x,g,positions,inverse)
                                    test.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)),
                                        (layout,scale,start,dtype,inverse,a,b))


def compare_stores(test, device):
    for dtype in (torch.float32, torch.int16):
        src = torch.arange(7*10, dtype=torch.int32, device=device).to(dtype).reshape(7,10)[:, ::2]
        columns = torch.tensor([8,0,4,2,6], dtype=torch.int64, device=device)
        a = torch.full((7,20), -9, dtype=dtype, device=device)[:, ::2]
        b = torch.full((7,20), -9, dtype=dtype, device=device)[:, ::2]
        a[:, columns] = src
        scatter_columns(b, columns, src)
        test.assertTrue(torch.equal(a, b))


def compare_packets(test, device):
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); tiny_state(root)
        for block, fmt in ((8, 'bf16'), (32, 'fp32')):
            opts = dict(predictor_format=fmt, block=block, values_per_stream=2)
            ref = FastEngine(root, device, **opts)
            for start, n in ((7,5), (2,3)):
                gen = torch.Generator().manual_seed(start)
                words = float_to_words(torch.randn(2,n,8,generator=gen)*.2).to(device,torch.int16)
                for kappa in (0., .5, 1.):
                    p = ref.encode(words, start=start, kappa=kappa)
                    expected = ref.last_reconstruction.clone(); wire = p.export()
                    for stores, rope in ((True,False),(False,True),(True,True)):
                        engine = FusedEngine(root,device,fuse_scatter=stores,fuse_rope=rope,**opts)
                        with test.subTest(block=block,fmt=fmt,start=start,kappa=kappa,stores=stores,rope=rope):
                            q = engine.encode(words,start=start,kappa=kappa)
                            test.assertEqual(q.export(),wire)
                            test.assertTrue(torch.equal(engine.last_reconstruction,expected))
                            test.assertTrue(torch.equal(engine.decode(Packet.restore(wire,device)),expected))
                            test.assertTrue(torch.equal(ref.decode(Packet.restore(q.export(),device)),expected))
                            test.assertIsNone(engine.rope._request_key)
            engine = FusedEngine(root,device,**opts)
            words = float_to_words(torch.full((2,3,8),.1)).to(device,torch.int16)
            test.assertTrue(torch.equal(engine.calibrate(words),ref.calibrate(words)))
            patterns = torch.arange(65536,dtype=torch.int32).to(device,torch.int16).reshape(2,4096,8)
            p = engine.encode(patterns)
            test.assertTrue(p.header['raw'])
            test.assertTrue(torch.equal(engine.decode(p),patterns))


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_rotations_and_non_unit_scale_fallback(self): compare_rotations(self,torch.device('cpu'))
    def test_unique_stores(self): compare_stores(self,torch.device('cpu'))
    def test_complete_packet_compatibility(self): compare_packets(self,torch.device('cpu'))

    def test_complete_worker_and_independent_receiver(self):
        from scripts import bench_lite_execution as bench
        from kv_compare.causal_packet import word_sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); state=root/'state'; state.mkdir(); tiny_state(state)
            words=float_to_words(torch.arange(48).float().reshape(2,3,8)/100).to(torch.int16)
            input_path=root/'input.npy'; np.save(input_path,words.numpy().astype('<u2'),allow_pickle=False)
            options=dict(predictor_format='bf16',values_per_stream=2,block=8)
            reference=FastEngine(state,'cpu',**options); entries=[]
            for kappa in (0.,.5):
                packet=reference.encode(words,start=7,kappa=kappa)
                path=root/f'k{kappa:g}.kvc'; path.write_bytes(packet.export())
                entries.append(dict(document='tiny',input=str(input_path),input_file_sha256=sha(input_path),
                    packet=str(path),packet_sha256=sha(path),original_word_sha256=word_sha(words),
                    decoded_word_sha256=word_sha(reference.last_reconstruction)))
            q=dict(method='lite_fixture',codec=dict(arm='fused',state=str(state),engine_options=options),
                entries=entries,output=str(root/'output'),warmup=1,repeats=2)
            bench.worker(q,torch,torch.device('cpu'))
            result=json.loads((root/'output/worker_results.json').read_text())
            self.assertEqual(result['status'],'software_checked'); self.assertFalse(result['gpu_measured'])
            self.assertEqual(len(result['rows']),2)
            self.assertTrue(all(r['packet_matches_original'] and r['status']=='verified' for r in result['rows']))
            proof=json.loads((root/'output/receiver.json').read_text())
            self.assertFalse(proof['gpu_measured']); self.assertTrue(proof['no_source_kv_or_predictions'])

    def test_gpu_entry_refuses_cpu_substitution(self):
        from scripts import bench_lite_execution as bench
        from unittest import mock
        if torch.cuda.is_available(): self.skipTest('CPU-only host required')
        with tempfile.TemporaryDirectory() as tmp, mock.patch('sys.stderr'):
            self.assertEqual(bench.main(['--output',tmp]),1)
            result=json.loads((Path(tmp)/'results.json').read_text())
            self.assertFalse(result['gpu_measured']); self.assertEqual(result['status'],'failed')


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'),'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_rotations_and_unique_stores(self):
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
        compare_rotations(self,torch.device('cuda',torch.cuda.current_device()))
        compare_stores(self,torch.device('cuda',torch.cuda.current_device()))
    def test_actual_gpu_complete_packet_compatibility(self):
        compare_packets(self,torch.device('cuda',torch.cuda.current_device()))


if __name__=='__main__': unittest.main()
