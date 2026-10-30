"""Frozen standalone inverse protocol, independent of a tensor runtime."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from scripts import bench_standalone_inverse as bench


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


def aggregate(method, encode=20, decode=10, phase='paired', tokens=8192):
    return dict(method=method,tokens=tokens,round=phase,packet_bytes=1000,complete_bpv=9.0,
        batched=dict(encode_ms=encode,decode_ms=decode),isolated=dict(encode_ms=encode,decode_ms=decode))


def accounting(q):
    mode,arm=bench.method_parts(q['method']);extra=bench.expected_extra(q['method'])
    L,T,W=q['entries'][0]['shape']
    result=dict(encoder_product='cublas',decoder_product='cublas',additional_shared_device_bytes=extra,
        full_inverse_shared_bytes=0,coder_table_bytes=57688+extra,rolling_encoder=True,
        encoder_accumulator_buffers=2,feature_reuse=True,decoder_chunks=None,
        effective_decoder_chunks=2,encoder_feature_buffer_count=1,encoder_feature_bytes=T*W,
        decoder_accumulator_buffers=2,decoder_accumulator_bytes=T*W*4,
        decoder_rank_scratch_bytes=T*256 if arm=='rank256_uniform' else 0,
        unchanged_encoder_header_chunks=2,
        shared_predictor_bytes=q['reference_entries'][0]['accounting']['shared_predictor_bytes'])
    if mode!='ref':
        result.update(inverse_mode=mode,original_coder_table_bytes=57688,
            sparse_inverse_shared_bytes=extra,inverse_proof_verified=True,inverse_checked_slots=655360,
            sparse_inverse_coarse_bytes=40960 if mode=='packed' else 0,
            sparse_inverse_exception_bytes=4608 if mode=='packed' else 0,
            inverse_algorithm='packed_cutoff_sparse' if mode=='packed' else 'masked_coarse_binary_search',
            decoder_kernel='_core_ldec_kernel')
    return result


class ReceiptAndPlanTests(unittest.TestCase):
    def test_completed_schedule_receipt_and_single_feature_reference_binding(self):
        fixed,receipt=bench.verify_previous()
        self.assertEqual(receipt['package']['files'],189)
        self.assertEqual(set(fixed['reference_packets']),
            {f'team_{arm}_t{n}' for arm in bench.ARMS for n in bench.LENGTHS})
        for rows in fixed['reference_packets'].values():
            self.assertTrue(all(row['method'].startswith('schedule_feature_') for row in rows))
        value=dict(fixed);identity=value.pop('policy_sha256')
        self.assertEqual(bench.digest_json(value),identity)
        self.assertEqual(fixed['additional_shared_device_bytes_by_mode'],dict(ref=0,packed=45568,masked=0))
        self.assertNotIn('mask',fixed)

    def test_twelve_jobs_keep_twenty_six_workers_and_original_frozen_words(self):
        original=selections();before=copy.deepcopy(original);fixed=policy()
        planned=bench.job_plan(original,fixed);controls=bench.control_plan(original)
        self.assertEqual(len(planned),12);self.assertEqual(len(controls),12)
        self.assertTrue(all(q['round'] in ('before','after') for _,q in controls))
        self.assertTrue(all(not q['method'].startswith('team_') for _,q in controls))
        self.assertEqual(len(planned)+len(controls)+len(bench.compact_plan(original,fixed)),26)
        self.assertEqual(sum(q['method'] in bench.REFERENCES for _,q in planned),4)
        self.assertEqual(sum(q['method'] in bench.CANDIDATES for _,q in planned),8)
        self.assertEqual(sum(q['tokens']==512 for _,q in planned),6)
        for _,q in planned:
            mode,arm=bench.method_parts(q['method'])
            reference=original[f'team_{arm}_t{q["tokens"]}']
            self.assertEqual(q['entries'],reference['entries'])
            spec=dict(q['codec']);self.assertEqual(spec.pop('kind'),'standalone_inverse')
            self.assertEqual(spec.pop('inverse_mode'),bench.MODE_FLAGS[mode])
            self.assertEqual(dict(spec,kind='team'),reference['codec'])
        self.assertEqual(original,before)

    def test_only_fixed_method_names_and_integer_lengths(self):
        for method in ('schedule_feature_dense_uniform','inverse_ref_dense_exact',
            'inverse_full_lut_dense_uniform','inverse_packed_dense_uniform_extra'):
            with self.assertRaises(ValueError):bench.method_parts(method)
        for value in (True,512.0,128,'8192'):
            with self.assertRaises(ValueError):bench.execution_for(value)


class ReceiverTests(unittest.TestCase):
    def test_strict_state_packet_and_inverse_allowlists(self):
        for _,q in jobs():bench.validate_receiver_request(receiver_request(q))
        base=receiver_request(jobs()[0][1]);bad=[]
        for key,value in (('mask',[True]*28),('splitzip_source','/vendor'),('predictions','/oracle'),
            ('feature_reuse',True),('decoder_chunks',4),('full_lut',True)):
            q=copy.deepcopy(base);q['codec'][key]=value;bad.append(q)
        for key,value in (('kind','standalone_schedule'),('state_sha256','a'*64),
            ('manifest_sha256','a'*64),('arm','rank256_uniform'),('inverse_mode',False)):
            q=copy.deepcopy(base);q['codec'][key]=value;bad.append(q)
        q=copy.deepcopy(base);q['raw_entries']=[];bad.append(q)
        for key,value in (('source_kv','/oracle'),('packet_sha256','a'*64),('packet_bytes',1),
            ('shape',[28,1,2048]),('state_sha256','a'*64)):
            q=copy.deepcopy(base);q['entries'][0][key]=value;bad.append(q)
        q=copy.deepcopy(base);q['entries'][1]['document']=q['entries'][0]['document'];bad.append(q)
        q=copy.deepcopy(base);q['execution']='direct_search';bad.append(q)
        for q in bad:
            with self.assertRaises(ValueError):bench.validate_receiver_request(q)

    def test_candidate_cannot_change_mode_or_claim_reference(self):
        base=receiver_request(next(q for _,q in jobs() if '_packed_' in q['method']))
        for mode in (None,'masked',True,1,'full'):
            q=copy.deepcopy(base);q['codec']['inverse_mode']=mode
            with self.assertRaises(ValueError):bench.validate_receiver_request(q)

    def test_relocated_actual_packets_and_original_graph_counts(self):
        for _,job in jobs():
            q=receiver_request(job)
            for i,entry in enumerate(q['entries']):entry['packet']=f'/new/actual_{i}.kvc'
            bench.validate_receiver_request(q)
            codec=SimpleNamespace(graph_capture_count=int(q['tokens']==512))
            self.assertEqual(bench.graph_count(q,codec),int(q['tokens']==512))
            codec.graph_capture_count+=1
            with self.assertRaises(AssertionError):bench.graph_count(q,codec)


class ParityAndAccountingTests(unittest.TestCase):
    def test_sixteen_candidate_pairs_and_twenty_four_own_frozen_rows(self):
        rows=parity_rows();proofs=bench.validate_packet_equivalence(rows)
        self.assertEqual(len(rows),24);self.assertEqual(len(proofs),16)
        self.assertTrue(all(row['original_wire_equal'] for row in proofs))
        self.assertEqual(bench.validate_packet_equivalence(rows[:1],complete=False),[])
        index=next(i for i,row in enumerate(rows) if row['method'] in bench.CANDIDATES)
        bad=[]
        for key,value in (('packet_bytes',1),('packet_sha256','a'*64),('original_word_sha256','a'*64),
            ('shape',[28,1,2048]),('values',1)):
            x=copy.deepcopy(rows);x[index][key]=value;bad.append(x)
        bad.extend((copy.deepcopy(rows[:-1]),copy.deepcopy(rows)+[copy.deepcopy(rows[0])]))
        for value in bad:
            with self.assertRaises(ValueError):bench.validate_packet_equivalence(value)

    def test_packed_tables_charged_and_masked_reference_have_no_extra_state(self):
        for _,q in jobs():
            info=accounting(q);bench.validate_accounting(info,q)
            self.assertEqual(info['additional_shared_device_bytes'],45568 if '_packed_' in q['method'] else 0)
            for key,value in (('effective_decoder_chunks',4),('encoder_feature_bytes',0),
                ('shared_predictor_bytes',0),('coder_table_bytes',1310720),('unchanged_encoder_header_chunks',8)):
                with self.assertRaises(ValueError):bench.validate_accounting(dict(info,**{key:value}),q)
            if q['method'] in bench.CANDIDATES:
                for key,value in (('inverse_proof_verified',False),('inverse_checked_slots',65536),
                    ('sparse_inverse_shared_bytes',999),('decoder_kernel','_full_lut_ldec_kernel')):
                    with self.assertRaises(ValueError):bench.validate_accounting(dict(info,**{key:value}),q)


class MeasurementTests(unittest.TestCase):
    def test_profile_counts_prove_inverse_kernel_and_unchanged_feature_controller(self):
        for arm in bench.ARMS:
            for mode in (None,'packed','masked'):
                products=243
                cpu={('/src/layerwise.py',1,'_product'):(products,products,.1,.2,{}),
                    ('/src/standalone_schedule_codec.py',2,'encode_native'):(3,3,.1,.2,{}),
                    ('/src/standalone_schedule_codec.py',3,'product'):(84,84,.1,.2,{})}
                kernels=[dict(name='_core_ldec_kernel' if mode else '_ldec_kernel',calls=168),
                    dict(name='_lenc_kernel',calls=84)]
                if arm=='rank256_uniform':kernels.append(dict(name='_igemm_rq_kernel',calls=products))
                kernels.extend(dict(name=name,calls=81) for name in ('_lfeat_kernel','_lsurp_kernel','_lprior_kernel'))
                options=dict(layers=28,chunks=2,inverse_mode=mode,arm=arm,calls=3)
                proof=bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,**options)
                self.assertTrue(proof['verified']);self.assertEqual(proof['total_product_calls']['actual'],243)
                changed=copy.deepcopy(cpu);changed[('/src/layerwise.py',1,'_product')]=(242,242,.1,.2,{})
                with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=changed,**options)
                changed=copy.deepcopy(kernels);changed[0]['calls']-=1
                with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=changed,cpu_stats=cpu,**options)
                for module,name in (('gpu_coder.py','slice_tables'),('standalone_inverse_core.py','build_inverse_tables')):
                    changed=copy.deepcopy(cpu);changed[('/src/'+module,1,name)]=(1,1,.1,.2,{})
                    with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=changed,**options)

    def test_profile_rejects_swapped_inverse_or_reintroduced_forward_work(self):
        cpu={('/src/layerwise.py',1,'_product'):(243,243,.1,.2,{}),
            ('/src/standalone_schedule_codec.py',2,'encode_native'):(3,3,.1,.2,{}),
            ('/src/standalone_schedule_codec.py',3,'product'):(84,84,.1,.2,{})}
        kernels=[dict(name='_core_ldec_kernel',calls=168),dict(name='_lenc_kernel',calls=84)]
        kernels.extend(dict(name=name,calls=81) for name in ('_lfeat_kernel','_lsurp_kernel','_lprior_kernel'))
        options=dict(layers=28,chunks=2,inverse_mode='packed',arm='dense_uniform',calls=3)
        bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,**options)
        for key,value in (('chunks',4),('inverse_mode','full'),('calls',True)):
            with self.assertRaises(ValueError):
                bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,**dict(options,**{key:value}))
        changed=copy.deepcopy(kernels);changed[0]['name']='_ldec_kernel'
        with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=changed,cpu_stats=cpu,**options)
        for module,name in (('layerwise.py','model'),('standalone_rolling_codec.py','encode_native')):
            changed=copy.deepcopy(cpu);changed[('/src/'+module,1,name)]=(1,1,.1,.2,{})
            with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=changed,**options)
        for name in ('_full_lut_ldec_kernel','_model_kernel'):
            with self.assertRaises(ValueError):
                bench.diagnostic_backend_proof(kernels=kernels+[dict(name=name,calls=1)],cpu_stats=cpu,**options)

    def test_links_preserve_units_and_complete_baseline_objectives(self):
        items=[aggregate('splitzip_official_framed',1,8,'before'),
            aggregate('splitzip_official_framed',9,2,'after'),aggregate(bench.CANDIDATES[0],4,4)]
        result=bench.link_models(items);self.assertFalse(result['measured_network'])
        self.assertEqual({r['effective_gbps'] for r in result['rows']},set(bench.LINKS))
        row=next(r for r in result['rows'] if r['method']==bench.CANDIDATES[0]
            and r['effective_gbps']==1 and r['timing_basis']=='isolated')
        self.assertAlmostEqual(row['wire_ms'],.008);self.assertAlmostEqual(row['fresh_serial_ms'],8.008)
        self.assertEqual(row['splitzip_targets']['precompressed']['round'],'after')
        self.assertEqual(row['splitzip_targets']['fresh']['round'],'before')

    def test_current_single_feature_pairing_and_report_cover_both_algorithms(self):
        items=[aggregate(q['method'],tokens=q['tokens']) for _,q in jobs()]
        insights=bench.paired_insights(items);self.assertEqual(len(insights),16)
        self.assertTrue(all(row['reference_method'].startswith('inverse_ref_') for row in insights))
        with tempfile.TemporaryDirectory() as raw:
            bench.report(dict(status='failed',gpu_measured=False,aggregates=items,
                paired_insights=insights,link_models=bench.link_models(items)),Path(raw))
            value=(Path(raw)/'summary.md').read_text()
            for text in ('Packed inverse','Masked search','Timing: **isolated**','45,568','current single-feature rolling reference'):
                self.assertIn(text,value)
            self.assertNotIn('Chunks 8',value)

    def test_memory_summary_uses_measured_per_document_median_peaks(self):
        rows=[]
        for document,offset in (('text',0),('code',100)):
            row=dict(tokens=8192,method=bench.REFERENCES[0],round='paired',document=document)
            for operation in ('encode','decode'):
                row[operation]=dict(isolated=dict(samples=[dict(incremental_peak_torch_bytes=value+offset)
                    for value in (100,300,200,500,400)]))
            rows.append(row)
        item=bench.memory_aggregates(rows)[0]
        self.assertEqual(item['encode_incremental_peak_mib'],350/(1024**2))
        bad=copy.deepcopy(rows);bad[0]['encode']['isolated']['samples'].pop()
        with self.assertRaises(ValueError):bench.memory_aggregates(bad)
        with self.assertRaises(ValueError):bench.memory_aggregates(rows[:1])

    def test_actual_cuda_gates_and_stdlib_parent_import(self):
        self.assertEqual(bench.CUDA_TESTS,('tests.test_standalone_inverse_core.CudaTests','tests.test_standalone_inverse_codec.CudaTests'))
        self.assertEqual(bench.CUDA_TEST_COUNT,3)
        source=(bench.ROOT/'scripts/bench_standalone_inverse.py').read_text()
        self.assertIn('str(CUDA_TEST_COUNT)',source)
        subprocess.run([sys.executable,'-c',
            'import sys;import scripts.bench_standalone_inverse;assert "torch" not in sys.modules'],cwd=bench.ROOT,check=True)
        process=subprocess.run([sys.executable,'scripts/bench_standalone_inverse.py','--help'],cwd=bench.ROOT,
            text=True,capture_output=True,check=True)
        self.assertIn('Standalone',process.stdout)


if __name__=='__main__':unittest.main()
