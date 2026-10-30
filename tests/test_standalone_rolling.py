"""Frozen standalone rolling experiment protocol, without a tensor runtime."""
import copy
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts import bench_standalone_rolling as bench


def selections():
    return bench.fixed_selections()


def policy():
    return bench.verify_previous()[0]


def requests():
    return [{key:q[key] for key in ('method','tokens','codec','entries','execution')}
        | dict(output='/receiver.json') for _,q in bench.job_plan(selections(),policy())]


def parity_rows():
    fixed=policy(); rows=[]
    for n in bench.LENGTHS:
        for arm in bench.ARMS:
            refs=fixed['reference_packets'][f'team_{arm}_t{n}']
            rows.extend(copy.deepcopy(refs))
            for method in bench.OWN:
                if bench.method_arm(method)==arm:
                    rows.extend(dict(copy.deepcopy(row),method=method) for row in refs)
    return rows


def aggregate(method,encode,decode,phase='paired',tokens=8192):
    return dict(method=method,tokens=tokens,round=phase,packet_bytes=1000,
        complete_bpv=9.0,batched=dict(encode_ms=encode,decode_ms=decode),
        isolated=dict(encode_ms=encode,decode_ms=decode))


class ReceiptAndPlanTests(unittest.TestCase):
    def test_completed_compute_receipt_and_original_standalone_references(self):
        fixed,receipt=bench.verify_previous()
        self.assertEqual(receipt['package']['files'],183)
        self.assertEqual(set(fixed['reference_packets']),
            {f'team_{arm}_t{n}' for arm in bench.ARMS for n in bench.LENGTHS})
        value=dict(fixed); identity=value.pop('policy_sha256')
        self.assertEqual(bench.digest_json(value),identity)
        self.assertEqual(fixed['additional_shared_device_bytes'],0)
        self.assertNotIn('masks',fixed)

    def test_six_fixed_jobs_twenty_four_full_workers_with_current_cached_control(self):
        source=selections(); before=copy.deepcopy(source); fixed=policy()
        jobs=bench.job_plan(source,fixed)
        self.assertEqual(len(jobs),6)
        self.assertEqual(len(jobs)+len(bench.frozen.job_plan(source))+len(bench.compact_plan(source,fixed)),24)
        self.assertEqual(sum(q['method']==bench.CACHE for _,q in jobs),2)
        for _,q in jobs:
            arm=bench.method_arm(q['method']); ref=source[f'team_{arm}_t{q["tokens"]}']
            self.assertEqual(q['entries'],ref['entries'])
            self.assertEqual(q['execution'],'compact_search' if q['tokens']==512 else 'direct_search')
            spec=dict(q['codec']); self.assertEqual(spec.pop('kind'),bench.method_kind(q['method']))
            self.assertEqual(dict(spec,kind='team'),ref['codec'])
            self.assertEqual(set(spec),{'state','state_sha256','manifest_sha256','arm'})
        self.assertEqual(source,before)

    def test_only_fixed_arms_and_integer_lengths(self):
        for method in ('team_dense_uniform','rolling_dense_exact','rolling_dense_uniform_extra',
            'compute_cached_model_dense_uniform','cached_dense_control_extra'):
            with self.assertRaises(ValueError): bench.method_arm(method)
        for tokens in (True,512.0,128,'512'):
            with self.assertRaises(ValueError): bench.execution_for(tokens)


class ReceiverTests(unittest.TestCase):
    def test_receiver_strict_state_packet_and_mode_allowlists(self):
        for q in requests(): bench.validate_receiver_request(q)
        base=requests()[0]; bad=[]
        for key,value in (('mask',[True]*28),('splitzip_source','/vendor'),('cache_model',True),
            ('decoder_tiles',[128,128,64,4]),('predictions','/oracle'),('full_lut',True)):
            x=copy.deepcopy(base); x['codec'][key]=value; bad.append(x)
        for key,value in (('state_sha256','a'*64),('arm','rank256_uniform'),('kind','team')):
            x=copy.deepcopy(base); x['codec'][key]=value; bad.append(x)
        x=copy.deepcopy(base);x['raw_entries']=[];bad.append(x)
        for key,value in (('source_kv','/oracle'),('packet_sha256','a'*64),('packet_bytes',1),
            ('shape',[28,32,2048]),('state_sha256','a'*64)):
            x=copy.deepcopy(base); x['entries'][0][key]=value;bad.append(x)
        x=copy.deepcopy(base);x['entries'][1]['document']=x['entries'][0]['document'];bad.append(x)
        x=copy.deepcopy(base);x['execution']='direct_search';bad.append(x)
        x=copy.deepcopy(base);x['method']=bench.CACHE;bad.append(x)
        for q in bad:
            with self.assertRaises(ValueError):bench.validate_receiver_request(q)

    def test_relocated_actual_packets_preserve_original_identity(self):
        q=requests()[0]
        for i,entry in enumerate(q['entries']): entry['packet']=f'/new/actual_{i}.kvp'
        bench.validate_receiver_request(q)

    def test_graph_counts_follow_original_execution(self):
        for q in requests():
            codec=SimpleNamespace(graph_capture_count=int(q['tokens']==512))
            self.assertEqual(bench.graph_count(q,codec),int(q['tokens']==512))
            codec.graph_capture_count+=1
            with self.assertRaises(AssertionError):bench.graph_count(q,codec)


class ParityAndAccountingTests(unittest.TestCase):
    def test_twelve_candidate_document_groups_preserve_original_packets(self):
        rows=parity_rows();proofs=bench.validate_packet_equivalence(rows)
        self.assertEqual(len(proofs),12)
        self.assertTrue(all(p['original_wire_equal'] for p in proofs))
        self.assertEqual(bench.validate_packet_equivalence(rows[:1],complete=False),[])
        candidate=next(i for i,row in enumerate(rows) if row['method'] in bench.OWN)
        bad=[]
        for key,value in (('packet_bytes',1),('packet_sha256','a'*64),('original_word_sha256','a'*64),
            ('shape',[28,1,2048]),('values',1)):
            x=copy.deepcopy(rows);x[candidate][key]=value;bad.append(x)
        bad.extend((copy.deepcopy(rows[:-1]),copy.deepcopy(rows)+[copy.deepcopy(rows[0])]))
        for items in bad:
            with self.assertRaises(ValueError):bench.validate_packet_equivalence(items)

    def test_transient_two_buffer_accounting_and_no_extra_shared_state(self):
        for _,q in bench.job_plan(selections(),policy()):
            info=dict(encoder_product='cublas',decoder_product='cublas',
                additional_shared_device_bytes=0,full_inverse_shared_bytes=0,coder_table_bytes=57688,
                shared_predictor_bytes=q['reference_entries'][0]['accounting']['shared_predictor_bytes'])
            if q['method'] in bench.ROLLING:
                info.update(rolling_encoder=True,encoder_accumulator_buffers=2)
            else:info.update(cache_model=True,model_cache_shares_native_tables=True)
            bench.validate_accounting(info,q)
            for key,value in (('decoder_product','triton'),('coder_table_bytes',1310720),
                ('additional_shared_device_bytes',1),('shared_predictor_bytes',0)):
                with self.assertRaises(ValueError):bench.validate_accounting(dict(info,**{key:value}),q)


class MeasurementTests(unittest.TestCase):
    def test_full_profile_schedule_counts_reduce_rank_products_and_dense_model(self):
        for arm in bench.ARMS:
            for rolling in (False,True):
                expected=243 if rolling or arm=='dense_uniform' else 321
                cpu={('/src/layerwise.py',1,'_product'):(expected,expected,.1,.2,{})}
                kernels=[]
                if arm=='rank256_uniform':kernels.append(dict(name='_igemm_rq_kernel',calls=expected))
                if rolling:
                    cpu[('/src/standalone_rolling_codec.py',1,'encode_native')]=(3,3,.1,.2,{})
                    cpu[('/src/standalone_rolling_codec.py',2,'product')]=(84,84,.1,.2,{})
                else:
                    cpu[('/src/layerwise.py',1,'model')]=(3,3,.1,.2,{})
                    if arm=='dense_uniform':
                        cpu[('/src/gpu_coder.py',1,'slice_tables')]=(3,3,.1,.2,{})
                        kernels.append(dict(name='_model_kernel',calls=3))
                if rolling or arm=='rank256_uniform':
                    kernels.extend(dict(name=name,calls=81) for name in ('_lfeat_kernel','_lsurp_kernel','_lprior_kernel'))
                settings=dict(layers=28,chunks=2,rolling=rolling,cache_model=False,arm=arm,calls=3)
                proof=bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,**settings)
                self.assertTrue(proof['verified'])
                self.assertEqual(proof['total_product_calls']['expected'],expected)
                if rolling:
                    self.assertEqual(proof['encoder_products_per_call'],27)
                    self.assertEqual(proof['warm_slice_tables']['calls'],0)
                changed=copy.deepcopy(cpu);changed[('/src/layerwise.py',1,'_product')]=(expected-1,expected-1,.1,.2,{})
                with self.assertRaises(ValueError):bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=changed,**settings)

    def test_cache_control_profile_has_no_layerwise_model_or_table_build(self):
        cpu={('/src/layerwise.py',1,'_product'):(243,243,.1,.2,{})}
        kernels=[dict(name='_model_kernel',calls=3)]
        proof=bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,
            layers=28,chunks=2,rolling=False,cache_model=True,arm='dense_uniform',calls=3)
        self.assertTrue(proof['verified'])
        self.assertEqual(proof['native_model_calls']['actual'],0)

    def test_link_units_and_independent_complete_baseline_rounds(self):
        items=[aggregate('splitzip_official_framed',1,8,'before'),
            aggregate('splitzip_official_framed',9,2,'after'),aggregate(bench.OWN[0],4,4)]
        result=bench.link_models(items)
        self.assertFalse(result['measured_network'])
        self.assertEqual({r['effective_gbps'] for r in result['rows']},set(bench.LINKS))
        row=next(r for r in result['rows'] if r['method']==bench.OWN[0]
            and r['effective_gbps']==1 and r['timing_basis']=='isolated')
        self.assertAlmostEqual(row['wire_ms'],.008)
        self.assertAlmostEqual(row['precompressed_serial_ms'],4.008)
        self.assertAlmostEqual(row['fresh_serial_ms'],8.008)
        self.assertEqual(row['splitzip_targets']['precompressed']['round'],'after')
        self.assertEqual(row['splitzip_targets']['fresh']['round'],'before')

    def test_paired_cache_reference_and_report_distinguish_timings(self):
        items=[]
        for n in bench.LENGTHS:
            for arm in bench.ARMS:items.append(aggregate('team_'+arm,40,20,tokens=n))
            items.extend(aggregate(method,20,10,tokens=n) for method in bench.OWN)
        result=bench.paired_insights(items)
        self.assertEqual(len(result),16)
        cache=[r for r in result if r['reference_method']==bench.CACHE]
        self.assertEqual(len(cache),4)
        self.assertTrue(all(r['method']=='rolling_dense_uniform' and r['encode_speedup']==1 for r in cache))
        with tempfile.TemporaryDirectory() as raw:
            bench.report(dict(status='failed',gpu_measured=False,aggregates=items,paired_insights=result),Path(raw))
            text=(Path(raw)/'summary.md').read_text()
            self.assertIn('cached_dense_control',text);self.assertIn('Timing: **isolated**',text)
            self.assertIn('two INT32 product matrices',text)

    def test_actual_cuda_gates_and_no_cpu_fallback(self):
        self.assertEqual(bench.CUDA_TESTS,('tests.test_standalone_rolling_codec.CudaTests',))
        self.assertEqual(bench.CUDA_TEST_COUNT,2)
        source=(bench.ROOT/'scripts/bench_standalone_rolling.py').read_text()
        self.assertIn('str(CUDA_TEST_COUNT)',source)
        self.assertEqual(bench.CODER_TABLE_BYTES,57688)

    def test_parent_import_and_help_are_stdlib_only(self):
        subprocess.run([sys.executable,'-c',
            'import sys; import scripts.bench_standalone_rolling; assert "torch" not in sys.modules'],cwd=bench.ROOT,check=True)
        value=subprocess.run([sys.executable,'scripts/bench_standalone_rolling.py','--help'],
            cwd=bench.ROOT,text=True,capture_output=True,check=True)
        self.assertIn('Standalone',value.stdout)


if __name__=='__main__':unittest.main()
