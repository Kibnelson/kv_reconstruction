"""Exact post-group fusion and high-link accounting; CUDA remains a separate gate."""
import ast
import inspect
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest

import torch
from kv_compare.causal_packet import Packet,float_to_words,json_bytes,write_json
from kv_compare.core import words_to_float
from kv_compare.lite_fused import FusedEngine,_run_columns
from kv_compare.lite_finish import FinishEngine,_run_finish
from tests.test_llama_causal import tiny_state


class StructureTests(unittest.TestCase):
    def test_loop_only_changes_the_post_group_conversion_and_writes(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(_run_columns)))
        changed=0
        def edit(node):
            nonlocal changed
            for field,value in ast.iter_fields(node):
                if isinstance(value,list):
                    result=[]; i=0
                    while i<len(value):
                        child=value[i]
                        if isinstance(child,ast.Assign) and ast.unparse(child)=='xrec = words_to_float(rec)':
                            self.assertEqual(ast.unparse(value[i+1]),'self.scatter(cur, pos, self.rotate(xrec, g, positions, inverse=True))')
                            self.assertIsInstance(value[i+2],ast.AugAssign)
                            self.assertEqual(ast.unparse(value[i+3].test),'source is None')
                            replacement=ast.parse('xrec = self.finish_group(rec, cur, g, positions, recovered[l] if source is None else self.last_reconstruction[l] if kappa else None)').body[0]
                            result.extend([replacement,value[i+2]]); changed+=1; i+=4
                        else:
                            if isinstance(child,ast.AST): edit(child)
                            result.append(child); i+=1
                    setattr(node,field,result)
                elif isinstance(value,ast.AST): edit(value)
        edit(tree); tree.body[0].name='_run_finish'
        expected=ast.parse(textwrap.dedent(inspect.getsource(_run_finish)))
        self.assertEqual(changed,1)
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(expected,include_attributes=False))

    def test_400_gigabit_targets_use_actual_packet_bytes_and_both_baselines(self):
        from scripts.bench_lite_finish import budgets,fetch_model,LINKS
        rows=[dict(method='raw_bf16',packet_bytes=58720573,decode_ms=.34927157685160637),
              dict(method='splitzip_official_framed',packet_bytes=44197315,decode_ms=.5547609180212021),
              dict(method='lite_finish',packet_bytes=33360893,decode_ms=96.2359425611794)]
        b=budgets(rows,links=[400.]); self.assertEqual(len(b),2)
        sz=next(x for x in b if x['baseline']=='splitzip_official_framed')
        self.assertAlmostEqual(sz['decode_budget_ms'],.7714893580212021)
        self.assertFalse(sz['measured_network']); self.assertIn(400.,LINKS)
        self.assertFalse(fetch_model(rows)['measured_network'])

    def test_cpu_cannot_publish_gpu_success(self):
        from scripts.package_lite_finish import verify
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'CPU evidence'):
                verify(Path(tmp),dict(status='completed',gpu_measured=False))


def parity(test,device):
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=False
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); tiny_state(root)
        for fmt in ('bf16','fp32'):
            opts=dict(values_per_stream=2,block=8,predictor_format=fmt)
            ref=FusedEngine(root,device,**opts); finish=FinishEngine(root,device,**opts)
            for start,n in ((0,3),(7,5)):
                gen=torch.Generator().manual_seed(start+n)
                words=float_to_words(torch.randn(2,n,8,generator=gen)*.3).to(device,torch.int16)
                for kappa in (0.,.5,1.):
                    with test.subTest(fmt=fmt,start=start,kappa=kappa):
                        p=ref.encode(words,start=start,kappa=kappa); expected=ref.last_reconstruction.clone()
                        q=finish.encode(words,start=start,kappa=kappa)
                        test.assertEqual(p.export(),q.export())
                        test.assertTrue(torch.equal(finish.last_reconstruction,expected))
                        test.assertTrue(torch.equal(finish.decode(Packet.restore(p.export(),device)),expected))
                        test.assertTrue(torch.equal(ref.decode(Packet.restore(q.export(),device)),expected))
            small=float_to_words(torch.full((2,3,8),.1)).to(device,torch.int16)
            test.assertTrue(torch.equal(ref.calibrate(small),finish.calibrate(small)))
            patterns=torch.arange(65536,dtype=torch.int32).to(device,torch.int16).reshape(2,4096,8)
            raw=finish.encode(patterns); test.assertTrue(raw.header['raw'])
            test.assertTrue(torch.equal(finish.decode(raw),patterns))


def finish_stores(test,device):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); tiny_state(root); ref=FusedEngine(root,device); fused=FinishEngine(root,device)
        n=5; positions=torch.arange(n,device=device)
        for g in ref.layers[0]['groups']:
            floats=torch.linspace(-.5,.5,n*g['columns']*2,device=device).reshape(n,-1)[:,::2]
            rec_store=torch.empty(n,g['columns']*2,device=device,dtype=torch.int16)
            rec=rec_store[:,::2]; rec.copy_(float_to_words(floats).to(torch.int16)); rec[0,0]=-32768
            rec[1,0]=1  # Smallest positive BF16 subnormal must retain its bits.
            a=torch.full((n,16),-9.,device=device)[:,::2]
            b=torch.full((n,16),-9.,device=device)[:,::2]
            out=torch.full((n,16),17,dtype=torch.int16,device=device)[:,::2]
            expected_out=out.clone(); expected_out[:,g['ch']]=rec
            with ref.rope.request(0,n,device),fused.rope.request(0,n,device):
                expected=words_to_float(rec); a[:,g['pos']]=ref.rotate(expected,g,positions,inverse=True)
                value=fused.finish_group(rec,b,g,positions,out)
                test.assertTrue(torch.equal(value.view(torch.int32),expected.float().view(torch.int32)))
                test.assertTrue(torch.equal(a.view(torch.int32),b.view(torch.int32)))
                test.assertTrue(torch.equal(out,expected_out))


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_complete_packet_parity(self): parity(self,torch.device('cpu'))
    def test_strided_finish_stores_and_signed_zero(self): finish_stores(self,torch.device('cpu'))
    def test_scale_fallback_and_disabled_fusion(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tiny_state(root)
            meta=json.loads((root/'state.json').read_text()); meta.pop('state_sha256')
            meta['attention_scaling']=1.25; meta['state_sha256']=hashlib.sha256(json_bytes(meta)).hexdigest()
            write_json(root/'state.json',meta)
            opts=dict(values_per_stream=2,block=8)
            ref=FusedEngine(root,'cpu',**opts); candidate=FinishEngine(root,'cpu',**opts)
            x=float_to_words(torch.full((2,3,8),.1)).to(torch.int16)
            self.assertEqual(ref.encode(x).export(),candidate.encode(x).export())
            disabled=FinishEngine(root,'cpu',fusedfinish=False,**opts)
            self.assertEqual(ref.encode(x).export(),disabled.encode(x).export())


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'),'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_bytes_and_words(self): parity(self,torch.device('cuda',torch.cuda.current_device()))
    def test_actual_gpu_finish_stores(self): finish_stores(self,torch.device('cuda',torch.cuda.current_device()))
    def test_actual_gpu_graph_reads_new_packets_with_fused_finish(self):
        from kv_compare.lite_graph import GraphDecoder
        from tests.test_lite_graph import packets
        device=torch.device('cuda',torch.cuda.current_device())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tiny_state(root); ref,entries=packets(root,device)
            candidate=FinishEngine(root,device,values_per_stream=2,block=8)
            graph=GraphDecoder(candidate,warmup=2); held=None
            for p,expected in (*entries,entries[0]):
                out=graph.decode(p); self.assertTrue(torch.equal(out,expected))
                if held is None: held=out
                else: self.assertTrue(torch.equal(held,entries[0][1]))
            self.assertEqual(graph.capture_count,1); graph.close()


if __name__=='__main__': unittest.main()
