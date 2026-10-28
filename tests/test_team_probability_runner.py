from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import bench_team_probability as bench


def entry(document='test_text_t512',variant='control',method='team_dense_uniform'):
    return dict(id=method+'_'+variant+'_'+document,document=document,shape=[28,512,2048],
        values=28*512*2048,packet='/packets/'+variant+'_'+document+'.kvp',packet_sha256='a'*64,
        packet_bytes=100000,original_word_sha256='b'*64,complete_bpv=8*100000/(28*512*2048),state_sha256='c'*64)


def selections():
    result={}
    for n in bench.LENGTHS:
        for method in bench.METHODS:
            result[f'{method}_t{n}']=dict(method=method,tokens=n,
                codec=dict(state='/fixed/state',manifest_sha256='d'*64,state_sha256='c'*64,arm=method.removeprefix('team_')),
                entries=[entry(f'test_{d}_t{n}',method=method) for d in ('text','code')],
                arms=[bench.decode_arm(n)] if method in bench.OWN else ['native'])
    return result


def encodes(selected):
    result={}
    for name,q in selected.items():
        if q['method'] not in bench.OWN:continue
        records=[]
        for variant in bench.VARIANTS:
            for e in q['entries']:
                value=dict(e,id=q['method']+'_'+variant+'_'+e['document'])
                records.append(dict(variant=variant,entry=value,codec=dict(q['codec'],state='/'+variant+'/state')))
        result[name]=dict(method=q['method'],tokens=q['tokens'],records=records)
    return result


class RunnerTests(unittest.TestCase):
    def test_state_comparison_is_bit_exact_including_signed_zero(self):
        import torch
        a=dict(x=torch.tensor([0.,-0.],dtype=torch.float32),rank=256)
        self.assertTrue(bench._same_tensors(a,copy.deepcopy(a),torch))
        self.assertFalse(bench._same_tensors(a,dict(a,x=torch.tensor([0.,0.])),torch))
        self.assertFalse(bench._same_tensors(a,dict(a,rank=256.),torch))

    def test_modes_remain_predeclared_and_frozen_reference(self):
        self.assertEqual(bench.decode_arm(512),'graph_reference')
        self.assertEqual(bench.decode_arm(8192),'direct_reference')
        for n in (0,256,1024,4096):
            with self.assertRaises(ValueError):bench.decode_arm(n)

    def test_parent_requires_completed_gpu_one_warp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'results.json'
            for value in (dict(format='team_register_v1',status='failed',gpu_measured=True,recipe=dict(decoder_warps=1)),
                          dict(format='team_register_v1',status='completed',gpu_measured=False,recipe=dict(decoder_warps=1)),
                          dict(format='team_register_v1',status='completed',gpu_measured=True,recipe=dict(decoder_warps=2)),
                          dict(format='team_lookup_v1',status='completed',gpu_measured=True,recipe=dict(decoder_warps=1))):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):bench.read_parent(directory)

    def test_plan_keeps_both_variants_and_before_after_native_controls(self):
        selected=selections();plan=bench.selected_plan(selected,encodes(selected))
        self.assertEqual(len(plan),20)
        self.assertTrue(all(q['round']=='before' for _,q in plan[:6]))
        self.assertTrue(all(q['round']=='after' for _,q in plan[-6:]))
        self.assertEqual([n.removeprefix('before_') for n,_ in plan[:6]],
                         [n.removeprefix('after_') for n,_ in plan[-6:]][::-1])
        own=plan[6:-6]
        self.assertEqual([q['round'] for _,q in own],['control','calibrated','calibrated','control']*2)
        self.assertEqual({q['codec']['state'] for _,q in own},{'/control/state','/calibrated/state'})
        self.assertTrue(all(q['arms']==[bench.decode_arm(q['tokens'])] for _,q in own))
        broken=encodes(selected);broken[next(iter(broken))]['records'].pop()
        with self.assertRaises(ValueError):bench.selected_plan(selected,broken)

    def test_encoder_request_binds_selected_state_before_test(self):
        selected=selections();name='team_dense_uniform_t512';base=selected[name]
        calibrated=dict(selected_state='/selected/state',selected_manifest_sha256='e'*64,
            selected_checks={m:dict(codec=dict(state='/selected/state',arm=m.removeprefix('team_'))) for m in bench.OWN})
        inputs=[dict(tokens=512,document='test_text_t512'),dict(tokens=8192,document='test_text_t8192')]
        q=bench.encode_request(name,base,calibrated,{},inputs,'r'*64,Path('/run'))
        self.assertEqual(q['inputs'],inputs[:1]);self.assertEqual(q['codecs']['control'],base['codec'])
        self.assertEqual(q['codecs']['calibrated']['state'],'/selected/state')
        self.assertEqual(q['selected_manifest_sha256'],'e'*64)
        self.assertEqual(q['output'],'/run/encode_'+name)

    def test_capture_recipe_opens_only_metadata(self):
        from kv_compare.causal_study import MODEL,REVISION
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'captures';root.mkdir()
            meta=dict(model=MODEL,revision=REVISION,layers=28)
            path=root/'meta.json';path.write_text(json.dumps(meta))
            source=dict(meta_sha256=bench.digest(path),model=MODEL,revision=REVISION,
                documents={n:dict(tokens=8192) for n in [*(f'train{i}' for i in range(6)),'tables','test_text','test_code']})
            with patch('numpy.load',side_effect=AssertionError('KV arrays cannot be opened before selection')):
                q=bench.capture_recipe(directory,dict(metadata=dict(capture_source=source)))
            self.assertEqual(q['calibration_document'],'tables');self.assertEqual(q['calibration_tokens'],8192)
            self.assertIn('committed before TEST',q['test_selection'])
            path.write_text(json.dumps(dict(meta,layers=27)))
            with self.assertRaises(ValueError):bench.capture_recipe(directory,dict(metadata=dict(capture_source=source)))

    def test_segment_hashes_bind_exact_payload_and_relative_starts(self):
        import torch
        class Packet:
            payload=torch.tensor([1,2,3,4,5,6],dtype=torch.int16)
            offsets=torch.tensor([0,2,6],dtype=torch.int64)
            relative_starts=torch.tensor([256,259],dtype=torch.int32)
            def validate(self):return self
        p=Packet();receipt=bench.segment_receipt(p)
        self.assertEqual(receipt['segment_words'],[2,4]);self.assertEqual(receipt['relative_starts'],[256,259])
        self.assertEqual(receipt['segment_sha256'][0],hashlib.sha256(p.payload[:2].numpy().tobytes()).hexdigest())
        p.payload[1]=9
        self.assertNotEqual(bench.segment_receipt(p)['segment_sha256'],receipt['segment_sha256'])

    def test_selected_segment_proof_preserves_factor_specific_bytes(self):
        observations=[dict(factor=.5,segment_words=[10,20,30,40],segment_sha256=['a','b','c','d'],relative_starts=[1,2,3,4]),
                      dict(factor=1.,segment_words=[11,21,31,41],segment_sha256=['A','B','C','D'],relative_starts=[5,6,7,8])]
        self.assertEqual(bench.selected_segments(observations,[.5,1.],2),
            dict(segment_words=[10,21,30,41],segment_sha256=['a','B','c','D'],relative_starts=[1,6,3,8]))
        with self.assertRaises(ValueError):bench.selected_segments(observations,[1.],2)

    def test_calibration_request_excludes_test_inputs(self):
        recipe=dict(selections=selections(),capture=dict(root='/capture',calibration_document='tables'))
        q=bench.calibration_request(recipe,'r'*64,Path('/run'))
        self.assertEqual(set(q),{'original_codec','capture','factors','recipe_identity','output'})
        self.assertEqual(q['factors'],[.5,.75,1.,1.25,1.5])
        self.assertNotIn('inputs',q)

    def test_link_models_keep_full_decode_and_both_native_targets(self):
        values=[dict(tokens=8192,method='team_rank256_uniform',arm='direct_reference',round='calibrated',packet_bytes=100_000_000,decode_ms=10.),
                dict(tokens=8192,method='splitzip_official_framed',arm='native',round='before',packet_bytes=120_000_000,decode_ms=2.),
                dict(tokens=8192,method='exans_official_fixed_framed',arm='native',round='after',packet_bytes=130_000_000,decode_ms=5.)]
        model=bench.link_models(values);own=[x for x in model['rows'] if x['round']=='calibrated']
        self.assertEqual([x['effective_gbps'] for x in own],list(bench.LINKS))
        self.assertEqual(own[1]['wire_ms'],8.);self.assertEqual(own[1]['precompressed_serial_ms'],18.)
        self.assertEqual(set(own[1]['baseline_targets']),{'splitzip_official_framed','exans_official_fixed_framed'})
        self.assertFalse(model['measured_network'])

    def test_actual_cuda_gate_precedes_parent_and_capture_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out'
            with patch.object(bench,'cuda_check',side_effect=RuntimeError('actual CUDA required')), \
                 patch.object(bench,'read_parent') as parent,patch.object(bench,'capture_recipe') as capture:
                self.assertEqual(bench.main(['--output',str(out)]),1)
                parent.assert_not_called();capture.assert_not_called()
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertFalse(result['gpu_measured'])
            self.assertTrue(bench.verify(out,result));self.assertIn('actual CUDA required',(out/'summary.md').read_text())

    def test_failed_report_packages_and_completed_cpu_is_rejected(self):
        from scripts import package_team_probability as package
        result=dict(format='team_probability_v1',status='failed',gpu_measured=False,jobs={},rows=[],
            aggregates=[],link_models=bench.link_models([]),error='CUDA gate failed')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            (run/'results.json').write_text(json.dumps(result));bench.report(result,run)
            (run/'fake.kvp').write_bytes(b'bulk data excluded')
            (run/'tensors.npz').write_bytes(b'bulk tensors excluded')
            self.assertEqual(package.main([str(run),'--destination',str(root/'report')]),0)
            manifest=json.loads((root/'report/package_manifest.json').read_text())
            self.assertNotIn('fake.kvp',manifest['files']);self.assertNotIn('tensors.npz',manifest['files'])
            self.assertIn('results.json',manifest['files'])
            with self.assertRaises(ValueError):bench.verify(run,dict(result,status='completed'))

    def test_single_reference_arm_job_proofs_and_negative_samples(self):
        from scripts import package_team_register as helpers
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);name='control_team_dense_uniform_t512';folder=run/name;folder.mkdir()
            selected=selections();base=dict(selected['team_dense_uniform_t512'],round='control')
            recipe=dict(warmup=2,repeats=8,device='fixture A4500')
            q=dict(base,decoder_warps=1,output=str(folder),warmup=2,repeats=8)
            policy=dict(deterministic_algorithms_enabled=False)
            record=dict(samples=[dict(wall_ms=1.,cuda_interval_ms=.9)]*8,
                warmup_samples=[dict(wall_ms=1.,cuda_interval_ms=.9)]*2,wall_median_ms=1.)
            rows=[];checks=[]
            for e in q['entries']:
                ident='control_graph_reference_'+e['id']
                rows.append(dict(e,id=ident,parent_id=e['id'],method=q['method'],arm='graph_reference',round='control',
                    tokens=512,status='verified',original_exact=True,decoded_word_sha256=e['original_word_sha256'],
                    decode=copy.deepcopy(record),kernel_decode=dict(record,warmup_samples=record['warmup_samples'][:1]),accounting={}))
                checks.append(dict(id=ident,verified=True,word_sha256=e['original_word_sha256'],packet_sha256=e['packet_sha256'],shape=e['shape']))
            immutable=[dict(id=e['id'],packet_sha256=e['packet_sha256'],bytes=e['packet_bytes'],unchanged=True) for e in q['entries']]
            proof=dict(gpu_measured=True,no_source_kv_or_predictions=True,alternating_packets=True,independent_returned_outputs=True,
                graph_capture_counts={'graph_reference':1},checks=checks,runtime_policy=policy,runtime_policy_after=policy,
                packet_immutability=immutable,inverse_proofs={'graph_reference':{}},kernel_provenance={'graph_reference':{}})
            data=dict(status='completed',gpu_measured=True,method=q['method'],round='control',device='fixture A4500',
                codec_provenance=q['codec'],rows=rows,graph_capture_counts={'graph_reference':1},
                runtime_policy=policy,runtime_policy_after=policy,packet_immutability=immutable,
                inverse_proofs=proof['inverse_proofs'],kernel_provenance=proof['kernel_provenance'],
                sample_order=bench.sample_order(q['entries'],q['arms'],2,8))
            receiver={k:q[k] for k in ('method','codec','entries','arms','round','decoder_warps')}
            receiver['output']=str(folder/'receiver.json')
            def refresh():
                for filename,value in (('request.json',q),('receiver.json',proof),('receiver_request.json',receiver)):
                    (folder/filename).write_text(json.dumps(value))
                data['receiver_sha256']=bench.digest(folder/'receiver.json')
                (folder/'worker_results.json').write_text(json.dumps(data))
                return dict(status='completed',rows=data['rows'],artifacts=bench._artifacts(folder,
                    ('request.json','worker_results.json','receiver.json','receiver_request.json'),run))
            with patch.object(helpers,'check_inverse_proof'),patch.object(helpers,'check_kernel_provenance'),patch.object(helpers,'check_accounting'):
                self.assertTrue(bench.verify_job(run,name,refresh(),base,recipe))
                proof['graph_capture_counts']={'graph_reference':1,'graph_register':1}
                with self.assertRaises(ValueError):bench.verify_job(run,name,refresh(),base,recipe)
                proof['graph_capture_counts']={'graph_reference':1}
                rows[0]['decode']['samples'][0]['wall_ms']=float('nan')
                with self.assertRaises(ValueError):bench.verify_job(run,name,refresh(),base,recipe)

    def test_report_exposes_bytes_time_calibration_and_scope(self):
        result=dict(status='failed',gpu_measured=False,aggregates=[],error='fixture')
        with tempfile.TemporaryDirectory() as directory:
            bench.report(result,directory);text=(Path(directory)/'summary.md').read_text()
        for s in ('TABLES','TEST','1 Gb/s','100 Gb/s','200 Gb/s','300 Gb/s','400 Gb/s','Packet MiB','NIC/SSD','inverse surprise scales'):
            self.assertIn(s,text)


if __name__=='__main__':unittest.main()
