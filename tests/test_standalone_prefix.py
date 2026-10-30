"""Frozen exact prefix-compaction protocol, independent of a tensor runtime."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

from scripts import bench_standalone_prefix as bench


def selections():
    return bench.fixed_selections()


def policy():
    return bench.verify_previous()[0]


def jobs():
    return bench.job_plan(selections(), policy())


def receiver_request(q):
    return {k:q[k] for k in ('method','tokens','codec','entries','execution')} | dict(output='/receiver.json')


def parity_rows():
    rows=[]; fixed=policy()
    for _,q in jobs():
        rows.extend(dict(copy.deepcopy(row),method=q['method'])
            for row in fixed['reference_packets'][f'team_{bench.method_arm(q["method"])}_t{q["tokens"]}'])
    return rows


def aggregate(method, encode=20, decode=10, phase='paired', tokens=8192, size=1000):
    return dict(method=method,tokens=tokens,round=phase,packet_bytes=size,complete_bpv=9.0,
        batched=dict(encode_ms=encode,decode_ms=decode),isolated=dict(encode_ms=encode,decode_ms=decode))


def accounting(q):
    mode,arm=bench.method_parts(q['method']);L,T,W=q['entries'][0]['shape']
    result=dict(encoder_product='cublas',decoder_product='cublas',additional_shared_device_bytes=45568,
        full_inverse_shared_bytes=0,coder_table_bytes=103256,rolling_encoder=True,
        encoder_accumulator_buffers=2,feature_reuse=True,decoder_chunks=None,
        effective_decoder_chunks=2,encoder_feature_buffer_count=1,encoder_feature_bytes=T*W,
        decoder_accumulator_buffers=2,decoder_accumulator_bytes=T*W*4,
        decoder_rank_scratch_bytes=T*256 if arm=='rank256_uniform' else 0,
        unchanged_encoder_header_chunks=2,
        shared_predictor_bytes=q['reference_entries'][0]['accounting']['shared_predictor_bytes'],
        inverse_mode='packed',original_coder_table_bytes=57688,
        sparse_inverse_shared_bytes=45568,inverse_proof_verified=True,inverse_checked_slots=655360,
        sparse_inverse_coarse_bytes=40960,sparse_inverse_exception_bytes=4608,
        inverse_algorithm='packed_cutoff_sparse',
        decoder_kernel='_core_ldec_kernel' if mode=='ref' else '_prefix_ldec_kernel')
    if mode!='ref':
        result.update(decoder_prefix_algorithm='warp_ballot_popcount',prefix_geometry=128,
            prefix_proof_verified=True,prefix_runtime_verified=True,prefix_production_layout_count=1,
            prefix_production_layouts=[dict(verified=True,target='cuda sm86 warp32',prefix_call_sites=2,
                ballots=8,pack=1,pure=False,layouts=['checked identity layout'],
                ttgir_sha256='a'*64,ptx_sha256='b'*64,compiled_hash='compiled')])
    return result


class ReceiptAndPlanTests(unittest.TestCase):
    def test_completed_inverse_receipt_and_packed_reference_binding(self):
        fixed,receipt=bench.verify_previous()
        self.assertEqual(receipt['package']['files'],169)
        self.assertEqual(set(fixed['reference_packets']),
            {f'team_{arm}_t{n}' for arm in bench.ARMS for n in bench.LENGTHS})
        self.assertTrue(all(row['method'].startswith('inverse_packed_')
            for rows in fixed['reference_packets'].values() for row in rows))
        value=dict(fixed);identity=value.pop('policy_sha256')
        self.assertEqual(bench.digest_json(value),identity)
        self.assertEqual(fixed['additional_shared_device_bytes_by_mode'],dict(ref=45568,warp=45568))

    def test_eight_jobs_keep_twenty_two_workers_and_frozen_words(self):
        original=selections();before=copy.deepcopy(original);fixed=policy()
        planned=bench.job_plan(original,fixed);controls=bench.control_plan(original)
        self.assertEqual(len(planned),8);self.assertEqual(len(controls),12)
        self.assertTrue(all(q['round'] in ('before','after') for _,q in controls))
        self.assertEqual(len(planned)+len(controls)+len(bench.compact_plan(original,fixed)),22)
        self.assertEqual(sum(q['method'] in bench.REFERENCES for _,q in planned),4)
        self.assertEqual(sum(q['method'] in bench.CANDIDATES for _,q in planned),4)
        for _,q in planned:
            mode,arm=bench.method_parts(q['method']);reference=original[f'team_{arm}_t{q["tokens"]}']
            self.assertEqual(q['entries'],reference['entries'])
            spec=dict(q['codec']);self.assertEqual(spec.pop('kind'),'standalone_prefix')
            self.assertEqual(spec.pop('prefix_mode'),bench.MODE_FLAGS[mode])
            self.assertEqual(dict(spec,kind='team'),reference['codec'])
        self.assertEqual(original,before)
        for value in (True,512.0,128,'8192'):
            with self.assertRaises(ValueError):bench.execution_for(value)
        for method in ('inverse_packed_dense_uniform','prefix_ref_dense_exact','prefix_warp_dense_uniform_extra'):
            with self.assertRaises(ValueError):bench.method_parts(method)


class ReceiverTests(unittest.TestCase):
    def test_strict_state_packet_and_prefix_allowlists(self):
        for _,q in jobs():bench.validate_receiver_request(receiver_request(q))
        base=receiver_request(jobs()[0][1]);bad=[]
        for key,value in (('mask',[True]*28),('predictions','/oracle'),('decoder_chunks',4),('full_lut',True)):
            q=copy.deepcopy(base);q['codec'][key]=value;bad.append(q)
        for key,value in (('kind','standalone_inverse'),('state_sha256','a'*64),('arm','rank256_uniform'),('prefix_mode',False)):
            q=copy.deepcopy(base);q['codec'][key]=value;bad.append(q)
        q=copy.deepcopy(base);q['raw_entries']=[];bad.append(q)
        for key,value in (('source_kv','/oracle'),('packet_sha256','a'*64),('packet_bytes',1),('shape',[28,1,2048])):
            q=copy.deepcopy(base);q['entries'][0][key]=value;bad.append(q)
        q=copy.deepcopy(base);q['entries'][1]['document']=q['entries'][0]['document'];bad.append(q)
        for q in bad:
            with self.assertRaises(ValueError):bench.validate_receiver_request(q)

    def test_candidate_cannot_claim_reference_or_change_receiver_geometry(self):
        base=receiver_request(next(q for _,q in jobs() if q['method'] in bench.CANDIDATES))
        for mode in (None,'packed','masked',True,1):
            q=copy.deepcopy(base);q['codec']['prefix_mode']=mode
            with self.assertRaises(ValueError):bench.validate_receiver_request(q)
        for _,job in jobs():
            q=receiver_request(job)
            for i,entry in enumerate(q['entries']):entry['packet']=f'/new/actual_{i}.kvc'
            bench.validate_receiver_request(q)
            codec=SimpleNamespace(graph_capture_count=int(q['tokens']==512))
            self.assertEqual(bench.graph_count(q,codec),int(q['tokens']==512))
            codec.graph_capture_count+=1
            with self.assertRaises(AssertionError):bench.graph_count(q,codec)


    def test_profile_reference_factory_keeps_packed_tables_and_candidate_stays_standalone(self):
        q=next(q for _,q in jobs() if q['method'] in bench.CANDIDATES)
        state=q['codec']['state_sha256']
        inverse=ModuleType('kv_compare.standalone_inverse_codec')
        candidate=ModuleType('kv_compare.standalone_prefix_codec')
        inverse.StandaloneInverseCodec=Mock(return_value=SimpleNamespace(state_sha256=state))
        candidate.StandalonePrefixCodec=Mock(return_value=SimpleNamespace(state_sha256=state))
        with patch.dict(sys.modules,{inverse.__name__:inverse,candidate.__name__:candidate}), \
                patch.object(bench,'sha',return_value=q['codec']['manifest_sha256']):
            bench.codec_for(q,'cuda:0',reference=True)
            inverse.StandaloneInverseCodec.assert_called_once_with(q['codec']['state'],'cuda:0',
                bench.method_arm(q['method']),inverse_mode='packed',decoder_mode='graph')
            candidate.StandalonePrefixCodec.assert_not_called()
            bench.codec_for(q,'cuda:0')
            candidate.StandalonePrefixCodec.assert_called_once_with(q['codec']['state'],'cuda:0',
                bench.method_arm(q['method']),decoder_mode='graph')


class ParityAndAccountingTests(unittest.TestCase):
    def test_eight_candidate_pairs_and_sixteen_own_frozen_rows(self):
        rows=parity_rows();proofs=bench.validate_packet_equivalence(rows)
        self.assertEqual(len(rows),16);self.assertEqual(len(proofs),8)
        self.assertTrue(all(row['original_wire_equal'] for row in proofs))
        index=next(i for i,row in enumerate(rows) if row['method'] in bench.CANDIDATES)
        bad=[]
        for key,value in (('packet_bytes',1),('packet_sha256','a'*64),('original_word_sha256','a'*64),('shape',[28,1,2048])):
            x=copy.deepcopy(rows);x[index][key]=value;bad.append(x)
        bad.extend((copy.deepcopy(rows[:-1]),copy.deepcopy(rows)+[copy.deepcopy(rows[0])]))
        for value in bad:
            with self.assertRaises(ValueError):bench.validate_packet_equivalence(value)

    def test_both_modes_charge_same_tables_and_candidate_requires_runtime_gate(self):
        for _,q in jobs():
            info=accounting(q);bench.validate_accounting(info,q)
            self.assertEqual(bench.expected_extra(q['method']),45568)
            for key,value in (('additional_shared_device_bytes',0),('effective_decoder_chunks',4),
                ('encoder_feature_bytes',0),('shared_predictor_bytes',0),('coder_table_bytes',57688)):
                with self.assertRaises(ValueError):bench.validate_accounting(dict(info,**{key:value}),q)
            if q['method'] in bench.CANDIDATES:
                for key,value in (('prefix_runtime_verified',False),('prefix_proof_verified',False),
                    ('prefix_geometry',32),('decoder_kernel','_core_ldec_kernel'),
                    ('prefix_production_layouts',[]),('prefix_production_layout_count',0)):
                    with self.assertRaises(ValueError):bench.validate_accounting(dict(info,**{key:value}),q)


    def test_production_receipts_reject_missing_failed_or_unbound_compiler_variants(self):
        q=next(q for _,q in jobs() if q['method'] in bench.CANDIDATES)
        base=accounting(q)
        for key,value in (('verified',False),('prefix_call_sites',1),('ballots',4),
            ('ttgir_sha256','bad'),('ptx_sha256',''),('target','cuda sm90'),('layouts',[]),('pack',4),('pure',True)):
            info=copy.deepcopy(base);info['prefix_production_layouts'][0][key]=value
            with self.assertRaises(ValueError):bench.validate_accounting(info,q)
        codec=SimpleNamespace(prefix_mode='warp',_prefix_production_layouts={})
        with self.assertRaises(ValueError):bench.production_layout_snapshot(codec)
        codec._prefix_production_layouts[('variant',)]=copy.deepcopy(base['prefix_production_layouts'][0])
        before=bench.production_layout_snapshot(codec)
        self.assertEqual(before['count'],1)
        self.assertEqual(before,bench.production_layout_snapshot(codec))
        codec._prefix_production_layouts[('other',)]=copy.deepcopy(base['prefix_production_layouts'][0])
        self.assertNotEqual(before,bench.production_layout_snapshot(codec))
        self.assertIsNone(bench.production_layout_snapshot(SimpleNamespace()))


class MeasurementTests(unittest.TestCase):
    def test_profile_counts_prove_prefix_and_unchanged_feature_controller(self):
        for arm in bench.ARMS:
            for mode in ('packed','warp'):
                cpu={('/src/layerwise.py',1,'_product'):(243,243,.1,.2,{}),
                    ('/src/standalone_schedule_codec.py',2,'encode_native'):(3,3,.1,.2,{}),
                    ('/src/standalone_schedule_codec.py',3,'product'):(84,84,.1,.2,{})}
                kernels=[dict(name='_prefix_ldec_kernel' if mode=='warp' else '_core_ldec_kernel',calls=168),
                    dict(name='_lenc_kernel',calls=84)]
                if arm=='rank256_uniform':kernels.append(dict(name='_igemm_rq_kernel',calls=243))
                kernels.extend(dict(name=name,calls=81) for name in ('_lfeat_kernel','_lsurp_kernel','_lprior_kernel'))
                options=dict(layers=28,chunks=2,prefix_mode=mode,arm=arm,calls=3)
                proof=bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,**options)
                self.assertTrue(proof['verified'])
                changed=copy.deepcopy(kernels);changed[0]['name']='_ldec_kernel'
                with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=changed,cpu_stats=cpu,**options)
                for module,name in (('gpu_coder.py','slice_tables'),('standalone_inverse_core.py','build_inverse_tables'),
                    ('standalone_prefix_core.py','runtime_prefix_proof'),
                    ('standalone_prefix_core.py','validate_compiled_prefix'),('jit.py','warmup'),('layerwise.py','model')):
                    changed=copy.deepcopy(cpu);changed[('/src/'+module,1,name)]=(1,1,.1,.2,{})
                    with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=changed,**options)

    def test_target_budgets_use_measured_bytes_and_complete_baseline_objective(self):
        base=aggregate('splitzip_official_framed',1,3,'before',size=100000000)
        after=aggregate('splitzip_official_framed',2,2,'after',size=150000000)
        candidate=aggregate(bench.CANDIDATES[0],4,4,size=50000000)
        result=bench.target_budgets([base,after,candidate]);self.assertEqual(len(result),2)
        row=next(x for x in result if x['effective_gbps']==100)
        self.assertAlmostEqual(row['candidate_wire_ms'],4)
        self.assertAlmostEqual(row['decode_budget_ms'],7)
        self.assertTrue(row['target_met']);self.assertEqual(row['splitzip_round'],'before')
        self.assertFalse(row['measured_network'])
        self.assertEqual(bench.target_budgets([candidate]),[])
        modeled=bench.link_models([base,after,candidate]);self.assertFalse(modeled['measured_network'])
        self.assertEqual({r['effective_gbps'] for r in modeled['rows']},set(bench.LINKS))

    def test_paired_insights_and_report_have_all_fixed_arms_and_targets(self):
        items=[aggregate(q['method'],tokens=q['tokens']) for _,q in jobs()]
        items.append(aggregate('splitzip_official_framed',phase='before'))
        insights=bench.paired_insights(items);self.assertEqual(len(insights),8)
        self.assertTrue(all(row['reference_method'].startswith('prefix_ref_') for row in insights))
        with tempfile.TemporaryDirectory() as raw:
            bench.report(dict(status='failed',gpu_measured=False,aggregates=items,
                paired_insights=insights,link_models=bench.link_models(items)),Path(raw))
            value=(Path(raw)/'summary.md').read_text()
            for text in ('Packed reference','Warp prefix','Timing: **isolated**','45,568','Decode budget ms'):
                self.assertIn(text,value)
            self.assertNotIn('Masked search',value)

    def test_memory_summary_uses_measured_per_document_median_peaks(self):
        rows=[]
        for document,offset in (('text',0),('code',100)):
            row=dict(tokens=8192,method=bench.REFERENCES[0],round='paired',document=document)
            for operation in ('encode','decode'):
                row[operation]=dict(isolated=dict(samples=[dict(incremental_peak_torch_bytes=value+offset)
                    for value in (100,300,200,500,400)]))
            rows.append(row)
        self.assertEqual(bench.memory_aggregates(rows)[0]['encode_incremental_peak_mib'],350/(1024**2))
        with self.assertRaises(ValueError):bench.memory_aggregates(rows[:1])

    def test_actual_cuda_gate_and_stdlib_parent_import(self):
        self.assertEqual(bench.CUDA_TESTS,('tests.test_standalone_prefix_core.CudaTests','tests.test_standalone_prefix_codec.CudaTests'))
        self.assertGreaterEqual(bench.CUDA_TEST_COUNT,2)
        source=(bench.ROOT/'scripts/bench_standalone_prefix.py').read_text()
        self.assertIn('str(CUDA_TEST_COUNT)',source)
        subprocess.run([sys.executable,'-c','import sys;import scripts.bench_standalone_prefix;assert "torch" not in sys.modules'],cwd=bench.ROOT,check=True)
        process=subprocess.run([sys.executable,'scripts/bench_standalone_prefix.py','--help'],cwd=bench.ROOT,
            text=True,capture_output=True,check=True)
        self.assertIn('Standalone',process.stdout)


if __name__=='__main__':unittest.main()
