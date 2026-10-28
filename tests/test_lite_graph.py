"""Staging protocol checks and separate actual-CUDA graph correctness gates."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from kv_compare.causal_packet import Packet,float_to_words,word_sha
from kv_compare.lite_fused import FusedEngine
from kv_compare.lite_graph import GraphDecoder,header_key,stage_packet
from tests.test_llama_causal import tiny_state


def packets(root,device,start=0):
    engine=FusedEngine(root,device,values_per_stream=2,block=8)
    a=float_to_words(torch.zeros(2,5,8)).to(device,torch.int16)
    b=float_to_words(torch.linspace(-10000.,10000.,80).reshape(2,5,8)).to(device,torch.int16)
    pa=engine.encode(a,start=start); pb=engine.encode(b,start=start)
    if pa.payload.numel()==pb.payload.numel(): raise AssertionError('fixture must change actual payload length')
    return engine,((pa,a),(pb,b))


class ProtocolTests(unittest.TestCase):
    def test_cpu_cannot_be_labeled_as_cuda_graph(self):
        with self.assertRaisesRegex(RuntimeError,'actual CUDA'):
            GraphDecoder(SimpleNamespace(device=torch.device('cpu')))

    def test_header_key_binds_semantics_but_not_payload_length(self):
        h=dict(shape=[2,5,8],start=0,kappa=0.,state_sha256='state',block=8,values_per_stream=2)
        a=Packet(h,torch.zeros(1,dtype=torch.int64),torch.zeros(8,dtype=torch.int16))
        b=Packet(dict(h),torch.zeros(1,dtype=torch.int64),torch.zeros(12,dtype=torch.int16))
        self.assertEqual(header_key(a),header_key(b))
        b.header['start']=7
        self.assertNotEqual(header_key(a),header_key(b))

    def test_staging_copies_new_offsets_and_payload_and_rejects_overflow(self):
        payload=torch.full((20,),-1,dtype=torch.int16); starts=torch.zeros(2,dtype=torch.int64)
        p=Packet({},torch.tensor([0,5]),torch.arange(12,dtype=torch.int16))
        self.assertEqual(stage_packet(p,payload,starts),40)
        self.assertTrue(torch.equal(starts,p.starts)); self.assertTrue(torch.equal(payload[:12],p.payload))
        q=Packet({},torch.tensor([0,4]),torch.full((9,),3,dtype=torch.int16))
        stage_packet(q,payload,starts)
        self.assertTrue(torch.equal(starts,q.starts)); self.assertTrue(torch.equal(payload[:9],q.payload))
        with self.assertRaisesRegex(ValueError,'staging'):
            stage_packet(p,payload[:5],starts)

    def test_gpu_runner_refuses_cpu_substitution(self):
        from scripts import bench_lite_graph as bench
        if torch.cuda.is_available(): self.skipTest('CPU-only host required')
        with tempfile.TemporaryDirectory() as tmp,mock.patch('sys.stderr'):
            self.assertEqual(bench.main(['--output',tmp]),1)
            r=json.loads((Path(tmp)/'results.json').read_text())
            self.assertFalse(r['gpu_measured']); self.assertEqual(r['status'],'failed')

    def test_decode_only_scoreboard_uses_actual_bytes(self):
        from scripts.bench_lite_graph import aggregate_decode,fetch_model
        rows=[dict(method=m,document=d,packet_bytes=n,values=100,complete_bpv=8*n/100,
            saved_percent=100*(1-n/200),decode=dict(wall_median_ms=t),original_exact=True)
            for m,n,t in (('splitzip_official_framed',150,1.),('lite_graph',100,2.)) for d in ('a','b')]
        a=aggregate_decode(rows); model=fetch_model(a)
        self.assertFalse(model['measured_network'])
        self.assertTrue(a)

    def test_package_rejects_cpu_success_and_excludes_buffers(self):
        from scripts import package_lite_graph as package
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


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET')=='1','explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_padded_staging_decodes_changing_packet_lengths_without_source_kv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tiny_state(root); engine,entries=packets(root,torch.device('cpu'))
            p=entries[0][0]; _,capacity,count=engine.layout(p.header['shape'][1])
            static=Packet(dict(p.header),torch.empty(count,dtype=torch.int64),torch.empty(capacity,dtype=torch.int16))
            outputs=[]
            for packet,expected in (*entries,entries[0]):
                engine.validate(packet)
                stage_packet(packet,static.payload,static.starts)
                # This is staging/decoder software evidence, not graph execution.
                out=engine.decode(static,validate=False).clone(); outputs.append(out)
                self.assertTrue(torch.equal(out,expected))
            self.assertTrue(torch.equal(outputs[0],entries[0][1]))


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'),'actual CUDA required')
class CudaTests(unittest.TestCase):
    def setUp(self):
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
        self.device=torch.device('cuda',torch.cuda.current_device())

    def test_actual_graph_alternates_payload_lengths_and_owns_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tiny_state(root); engine,entries=packets(root,self.device)
            graph=GraphDecoder(engine,warmup=2)
            graph.prepare(entries[0][0]); outputs=[]
            for p,expected in (*entries,entries[0],entries[1]):
                out=graph.decode(Packet.restore(p.export(),self.device)); outputs.append(out)
                self.assertTrue(torch.equal(out,expected))
                self.assertEqual(graph.capture_count,1)
            self.assertTrue(torch.equal(outputs[0],entries[0][1]))
            self.assertTrue(torch.equal(outputs[1],entries[1][1]))
            self.assertNotEqual(outputs[0].data_ptr(),outputs[2].data_ptr())
            bad=Packet(dict(entries[0][0].header),entries[0][0].starts.clone(),entries[0][0].payload)
            bad.starts[0]=1
            with mock.patch.object(graph,'_prepare_validated',side_effect=AssertionError('replay reached')):
                with self.assertRaises(ValueError): graph.decode(bad)
            phase=graph.measure_phases(entries[1][0],repeats=1)
            self.assertTrue(phase['gpu_measured']); self.assertEqual(graph.capture_count,1)
            graph.close()

    def test_position_changes_recapture_and_raw_fallback_remains_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tiny_state(root); engine,entries=packets(root,self.device)
            p0,expected=entries[0]; p7=engine.encode(expected,start=7)
            graph=GraphDecoder(engine,warmup=1)
            for p in (p0,p7,p0): self.assertTrue(torch.equal(graph.decode(p),expected))
            self.assertEqual(graph.capture_count,3)
            patterns=torch.arange(65536,dtype=torch.int32).to(self.device,torch.int16).reshape(2,4096,8)
            raw=engine.encode(patterns)
            self.assertTrue(torch.equal(graph.decode(raw),patterns)); self.assertEqual(graph.capture_count,3)
            other=torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(other):
                with self.assertRaisesRegex(RuntimeError,'one CUDA stream'): graph.decode(p0)
            graph.close()

    def test_capture_failure_is_not_reported_as_eager_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tiny_state(root); engine,entries=packets(root,self.device)
            graph=GraphDecoder(engine,warmup=1)
            with mock.patch.object(torch.cuda,'graph',side_effect=RuntimeError('fixture capture failure')):
                with self.assertRaisesRegex(RuntimeError,'fixture capture failure'): graph.prepare(entries[0][0])
            self.assertEqual(graph.capture_count,0); self.assertIsNone(graph._entry)


if __name__=='__main__': unittest.main()
