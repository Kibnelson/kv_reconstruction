"""Frozen inputs, balanced timing and honest host-transport models."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.bench_team_transport import (ARMS,LINKS,aggregate,arm_order,checked_wire,
    finish_samples,identity,link_models,main,profile_worker,report,valid_receiver_request)
from scripts.package_team_transport import check_samples,verify


class RunnerTests(unittest.TestCase):
    def test_all_link_models_charge_host_import_and_egress_without_encoding(self):
        self.assertEqual(LINKS,(1.,100.,200.,300.,400.))
        a=dict(tokens=512,method='team_dense_uniform',arm='optimized',packet_bytes=1_000_000,
            export_ms=3.,import_ms=4.,receive_ms=6.,roundtrip_ms=10.,gpu_resident_decode_ms=2.)
        r=link_models([a])
        x=next(v for v in r['rows'] if v['effective_gbps']==1.)
        self.assertEqual((x['wire_ms'],x['host_blob_fetch_ms'],x['gpu_compressed_egress_ms']), (8.,14.,18.))
        self.assertEqual(x['component_egress_diagnostic_ms'],17.)
        self.assertEqual(x['gpu_direct_context_ms'],10.)
        self.assertFalse(x['measured_network'])
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):link_models([a],[value])

    def test_predeclared_balanced_rotation_and_reversal(self):
        orders=[arm_order(i) for i in range(6)]
        self.assertEqual(orders[:4],[ARMS,ARMS[::-1],ARMS[::-1],ARMS])
        self.assertEqual(sum(o[0]=='reference' for o in orders),3)
        self.assertTrue(all(set(o)==set(ARMS) for o in orders))
        for bad in (-1,1.5,True):
            with self.assertRaises(ValueError):arm_order(bad)

    def test_samples_require_measured_positive_cuda_intervals_and_correct_median(self):
        record=finish_samples(dict(samples=[{'wall_ms':3.,'cuda_interval_ms':1.},{'wall_ms':5.,'cuda_interval_ms':2.}],warmup_samples=[]))
        self.assertEqual(record['wall_median_ms'],4.)
        check_samples(record,0,2)
        with self.assertRaises(ValueError):check_samples(dict(record,wall_median_ms=1.),0,2)
        with self.assertRaises(ValueError):check_samples(record,0,3)

    def test_exact_immutable_wire_is_mandatory(self):
        import hashlib
        blob=b'original complete packet'
        e=dict(packet_bytes=len(blob),packet_sha256=hashlib.sha256(blob).hexdigest())
        self.assertTrue(checked_wire(blob,e))
        for bad in (bytearray(blob),blob+b'x',b'x'+blob[1:]):
            with self.assertRaises(AssertionError):checked_wire(bad,e)

    def test_receiver_rejects_source_arrays_and_nonpredeclared_arms(self):
        q=dict(codec={'kind':'team'},entries=[dict(packet='parent.kvp')],output='proof.json',arms=list(ARMS))
        self.assertTrue(valid_receiver_request(q))
        with self.assertRaises(ValueError):valid_receiver_request(dict(q,inputs=['original.npy']))
        with self.assertRaises(ValueError):valid_receiver_request(dict(q,entries=[{'path':'original.npy'}]))
        with self.assertRaises(ValueError):valid_receiver_request(dict(q,arms=['optimized']))

    def test_lengths_and_arms_stay_separate_in_aggregation(self):
        rows=[]
        for n in (512,8192):
            for arm in ARMS:
                for d in ('test_text','test_code'):
                    r=dict(tokens=n,method='team_dense_uniform',arm=arm,document=f'{d}_t{n}',
                        packet_bytes=n,complete_bpv=9.,original_exact=True,gpu_resident_decode={'wall_median_ms':2.})
                    r.update({m:{'wall_median_ms':n/100} for m in ('export','import','receive','roundtrip')})
                    rows.append(r)
        a=aggregate(rows)
        self.assertEqual(len(a),4)
        self.assertEqual({(r['tokens'],r['arm']) for r in a},{(n,a) for n in (512,8192) for a in ARMS})
        self.assertEqual(next(r['receive_ms'] for r in a if r['tokens']==8192),81.92)

    def test_cpu_gate_runs_before_parent_packet_or_state_reads(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'
            with patch('scripts.bench_team_transport.read_parent',side_effect=AssertionError('parent read before CUDA gate')) as parent:
                with contextlib.redirect_stderr(io.StringIO()):
                    status=main(['--device','cpu','--output',str(out),'--parent-run',str(Path(d)/'missing')])
            self.assertEqual(status,1);parent.assert_not_called()
            data=json.loads((out/'results.json').read_text())
            self.assertEqual(data['status'],'failed');self.assertFalse(data['gpu_measured'])
            self.assertIn('actual CUDA required',data['error'])
            self.assertTrue(verify(out,data))

    def test_packager_rejects_completed_cpu_run(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'CPU evidence'):
                verify(Path(d),dict(format='team_transport_v1',status='completed',gpu_measured=False))

    def test_recipe_identity_binds_old_packet_sha_and_new_source(self):
        r=dict(parent_results_sha256='a',selections={'packet_sha256':'b'},code={'transport.py':'c'})
        self.assertEqual(identity(r),identity(dict(reversed(list(r.items())))))
        self.assertNotEqual(identity(r),identity(dict(r,selections={'packet_sha256':'changed'})))
        self.assertNotEqual(identity(r),identity(dict(r,code={'transport.py':'changed'})))

    def test_phase_report_distinguishes_missing_operation_from_measured_zero(self):
        diagnostics={arm:dict(export={'phases':{'D2H':{'wall_ms':0.}}},
            import_record={'phases':{'H2D':{'wall_ms':2.}}}) for arm in ARMS}
        result=dict(status='completed',gpu_measured=True,profiles={'example':dict(method='team_dense_uniform',diagnostics=diagnostics)})
        with tempfile.TemporaryDirectory() as d:
            report(result,Path(d));text=(Path(d)/'summary.md').read_text()
        self.assertIn('Dtype/layout copy',text)
        self.assertIn('Checksum input copy',text)
        self.assertIn('| 0.000 | \u2014 |',text)
        self.assertIn('| \u2014 | 2.000 | \u2014 |',text)

    def test_verification_hash_and_output_copy_are_outside_both_profilers(self):
        active={'torch':False,'cpu':False};calls={'wire':0,'words':0,'public':0}
        class FakeProfile:
            def __enter__(self):active['torch']=True;return self
            def __exit__(self,*args):active['torch']=False
            def export_chrome_trace(self,path):Path(path).write_text('{}')
            def key_averages(self):return []
        class FakeCPU:
            def enable(self):active['cpu']=True
            def disable(self):active['cpu']=False
        class FakeCodec:
            def graph_prepare(self,packet):return self
            def graph_decode(self,packet):return object()
        blob=b'tiny exact fixture'
        def export(packet,**kwargs):
            self.assertTrue(active['torch'] and active['cpu']);calls['public']+=1
            return blob,{'profiled':True,'phases':{}}
        def ingest(data,device,**kwargs):
            if kwargs.get('profile'):
                self.assertTrue(active['torch'] and active['cpu']);calls['public']+=1
            return object(),{'profiled':bool(kwargs.get('profile')),'phases':{}}
        def check_wire(data,entry):
            self.assertFalse(active['torch'] or active['cpu'])
            self.assertEqual(data,blob);calls['wire']+=1
        def check_words(value,entry):
            self.assertFalse(active['torch'] or active['cpu']);calls['words']+=1
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device:None))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);packet=root/'parent.kvp';packet.write_bytes(blob)
            q=dict(codec={'kind':'team'},entries=[dict(packet=str(packet),document='test_text_t8192',packet_sha256='asserted')],
                method='team_dense_uniform',tokens=8192,output=str(root/'profile'))
            with patch('torch.profiler.profile',return_value=FakeProfile()), \
                 patch('torch.profiler.record_function',side_effect=lambda name:contextlib.nullcontext()), \
                 patch('scripts.bench_team_transport.cProfile.Profile',return_value=FakeCPU()), \
                 patch('scripts.bench_team_transport.pstats.Stats',return_value=SimpleNamespace(stats={})), \
                 patch('scripts.bench_team_transport.implementations',return_value=(export,ingest)), \
                 patch('scripts.bench_team_transport.own_codec',return_value=FakeCodec()), \
                 patch('scripts.bench_team_transport.checked_wire',side_effect=check_wire), \
                 patch('scripts.bench_team_transport.check_original',side_effect=check_words), \
                 patch('scripts.bench_team_transport.runtime_snapshot',return_value={}):
                profile_worker(q,fake_torch,'cuda:0')
            receipt=json.loads((root/'profile/profile.json').read_text())
        self.assertEqual(calls,dict(wire=3,words=2,public=4))
        self.assertTrue(receipt['assertions_outside_profilers'])
        self.assertFalse(active['torch'] or active['cpu'])


if __name__=='__main__':unittest.main()
