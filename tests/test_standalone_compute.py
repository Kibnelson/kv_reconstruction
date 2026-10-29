"""Fixed standalone compute experiment protocol; no tensor runtime."""
import copy
import subprocess
import sys
from types import SimpleNamespace
import unittest

from scripts import bench_standalone_compute as bench


def selections():
    return bench.fixed_selections()


def policy():
    return bench.verify_previous()[0]


def requests():
    return [{key:q[key] for key in ('method', 'tokens', 'codec', 'entries', 'execution')}
        | dict(output='/receiver.json') for _,q in bench.job_plan(selections(), policy())]


def parity_rows():
    fixed = policy(); rows = []
    for n in bench.LENGTHS:
        for arm in bench.ARMS:
            refs = fixed['reference_packets'][f'team_{arm}_t{n}']
            rows.extend(copy.deepcopy(refs))
            for method in bench.OWN:
                if bench.method_arm(method) == arm:
                    rows.extend(dict(copy.deepcopy(row), method=method) for row in refs)
    return rows


def aggregate(method, encode, decode, phase='paired', tokens=8192):
    return dict(method=method, tokens=tokens, round=phase, packet_bytes=1000,
        batched=dict(encode_ms=encode, decode_ms=decode), isolated=dict(encode_ms=encode, decode_ms=decode))


class ReceiptAndPlanTests(unittest.TestCase):
    def test_latest_actual_cuda_receipt_and_original_standalone_references(self):
        fixed, receipt = bench.verify_previous()
        self.assertEqual(receipt['package']['files'], 141)
        self.assertEqual(set(fixed['reference_packets']),
            {f'team_{arm}_t{n}' for arm in bench.ARMS for n in bench.LENGTHS})
        value = dict(fixed); identity = value.pop('policy_sha256')
        self.assertEqual(bench.digest_json(value), identity)
        self.assertEqual(fixed['additional_shared_device_bytes'], 0)
        self.assertNotIn('masks', fixed)

    def test_ten_fixed_jobs_and_twenty_eight_full_workers(self):
        source = selections(); before = copy.deepcopy(source); fixed = policy()
        jobs = bench.job_plan(source, fixed)
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len(jobs) + len(bench.frozen.job_plan(source)) + len(bench.compact_plan(source, fixed)), 28)
        for _,q in jobs:
            mode, arm = bench.method_parts(q['method'])
            ref = source[f'team_{arm}_t{q["tokens"]}']
            self.assertEqual(q['entries'], ref['entries'])
            self.assertEqual(q['execution'], 'compact_search' if q['tokens'] == 512 else 'direct_search')
            spec = dict(q['codec']); tiles, cache = bench.MODE_FLAGS[mode]
            self.assertEqual(spec.pop('decoder_tiles'), list(tiles) if tiles is not None else None)
            self.assertIs(spec.pop('cache_model'), cache)
            spec['kind'] = 'team'; self.assertEqual(spec, ref['codec'])
            self.assertNotIn('splitzip_source', q['codec']); self.assertNotIn('mask', q['codec'])
            self.assertNotIn('full_lut', q['codec'])
            if mode == 'cached_model': self.assertEqual(arm, 'dense_uniform')
        self.assertEqual(source, before)

    def test_declared_modes_cannot_be_combined_or_coerced(self):
        for method in ('team_dense_uniform', 'compute_cached_model_rank256_uniform',
            'compute_gemm128_dense_exact', 'compute_gemm128_dense_uniform_extra',
            'compute_gemm64_cached_model_dense_uniform', 'standalone_lut_dense_uniform'):
            with self.assertRaises(ValueError): bench.method_parts(method)
        for tokens in (True, 512.0, 128, '512'):
            with self.assertRaises(ValueError): bench.execution_for(tokens)


class ReceiverTests(unittest.TestCase):
    def test_fresh_receiver_has_strict_state_mode_and_packet_allowlists(self):
        for q in requests(): bench.validate_receiver_request(q)
        base = requests()[-1]; bad = []
        for key,value in (('mask', [True]*28), ('splitzip_source', '/vendor'), ('full_lut', True),
            ('predictions', '/oracle')):
            x = copy.deepcopy(base); x['codec'][key] = value; bad.append(x)
        x = copy.deepcopy(base); x['raw_entries'] = []; bad.append(x)
        x = copy.deepcopy(base); x['codec']['cache_model'] = False; bad.append(x)
        x = copy.deepcopy(base); x['codec']['cache_model'] = 1; bad.append(x)
        x = copy.deepcopy(base); x['codec']['decoder_tiles'] = [128,128,64,4]; bad.append(x)
        x = copy.deepcopy(base); x['codec']['state_sha256'] = 'a'*64; bad.append(x)
        x = copy.deepcopy(base); x['codec']['arm'] = 'rank256_uniform'; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['source_kv'] = '/oracle'; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['packet_sha256'] = 'a'*64; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['packet_bytes'] += 1; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['shape'][1] += 1; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['state_sha256'] = 'a'*64; bad.append(x)
        x = copy.deepcopy(base); x['entries'][1]['document'] = x['entries'][0]['document']; bad.append(x)
        x = copy.deepcopy(base); x['execution'] = 'compact_search'; bad.append(x)
        gemm = requests()[0]
        for value in ([128.0,128,64,4], (128,128,64,4), [64,128,64,4], None):
            x = copy.deepcopy(gemm); x['codec']['decoder_tiles'] = value; bad.append(x)
        x = copy.deepcopy(gemm); x['codec']['cache_model'] = True; bad.append(x)
        for q in bad:
            with self.assertRaises(ValueError): bench.validate_receiver_request(q)

    def test_changed_paths_preserve_original_wire_identity(self):
        q = requests()[0]
        for i,entry in enumerate(q['entries']): entry['packet'] = f'/new/actual_{i}.kvp'
        bench.validate_receiver_request(q)

    def test_graph_counts_follow_original_execution(self):
        for q in requests():
            codec = SimpleNamespace(graph_capture_count=int(q['tokens'] == 512))
            self.assertEqual(bench.graph_count(q,codec), int(q['tokens'] == 512))
            codec.graph_capture_count += 1
            with self.assertRaises(AssertionError): bench.graph_count(q,codec)


class ParityAndAccountingTests(unittest.TestCase):
    def test_all_twenty_candidate_document_groups_preserve_exact_packets(self):
        rows = parity_rows(); proofs = bench.validate_packet_equivalence(rows)
        self.assertEqual(len(proofs), 20)
        self.assertTrue(all(p['original_wire_equal'] for p in proofs))
        self.assertEqual(bench.validate_packet_equivalence(rows[:1],complete=False), [])
        bad = []
        candidate = next(i for i,row in enumerate(rows) if row['method'] in bench.OWN)
        for key,value in (('packet_bytes',1),('packet_sha256','a'*64),('original_word_sha256','a'*64),
            ('shape',[28,1,2048]),('values',1)):
            x = copy.deepcopy(rows); x[candidate][key] = value; bad.append(x)
        bad.append(copy.deepcopy(rows[:-1])); bad.append(copy.deepcopy(rows)+[copy.deepcopy(rows[0])])
        for values in bad:
            with self.assertRaises(ValueError): bench.validate_packet_equivalence(values)

    def test_no_extra_device_state_and_original_coder_tables(self):
        for _,q in bench.job_plan(selections(),policy()):
            mode,_ = bench.method_parts(q['method']); tiles,cache = bench.MODE_FLAGS[mode]
            accounting = dict(decoder_gemm_tiles=list(tiles) if tiles is not None else None,
                cache_model=cache,decoder_product='triton' if tiles is not None else 'cublas',
                encoder_product='cublas',additional_shared_device_bytes=0,full_inverse_shared_bytes=0,
                coder_table_bytes=57688,model_cache_shares_native_tables=cache,
                shared_predictor_bytes=q['reference_entries'][0]['accounting']['shared_predictor_bytes'])
            bench.validate_accounting(accounting,q)
            for key,value in (('encoder_product','triton'),('coder_table_bytes',1310720),
                ('full_inverse_shared_bytes',1310720),('additional_shared_device_bytes',1),
                ('shared_predictor_bytes',0),('cache_model',not cache)):
                with self.assertRaises(ValueError): bench.validate_accounting(dict(accounting,**{key:value}),q)


class MeasurementTests(unittest.TestCase):
    def test_diagnostic_backend_counts_and_complete_cpu_stats(self):
        cpu = {('/src/team_onestep/gpu_coder.py', 76, 'slice_tables'):(3,3,.1,.2,{}),
            ('/src/irrelevant.py', 1, 'slice_tables'):(10,10,.1,.2,{})}
        kernels = [dict(name='_igemm_kernel', calls=162),
            dict(name='_igemm_rq_kernel', calls=81), dict(name='_ldec_kernel', calls=168)]
        settings = dict(layers=28,chunks=2,decoder_tiles=(128,128,64,4),
            cache_model=False,arm='dense_uniform',calls=3)
        proof = bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,**settings)
        self.assertEqual(proof['decoder_gemm']['actual_calls'],162)
        self.assertEqual(proof['decoder_gemm']['expected_calls'],162)
        self.assertEqual(proof['warm_slice_tables']['calls'],3)
        self.assertTrue(proof['warm_slice_tables']['counted_from_complete_pstats'])
        self.assertTrue(proof['warm_slice_tables']['matches_reference_expectation'])
        kernels[0]['calls'] = 161
        with self.assertRaises(ValueError): bench.diagnostic_backend_proof(kernels=kernels,cpu_stats=cpu,**settings)
        with self.assertRaises(ValueError): bench.diagnostic_backend_proof(kernels=kernels,cpu_stats={},
            **dict(settings,decoder_tiles=None))
        proof = bench.diagnostic_backend_proof(kernels=[],cpu_stats={},
            **dict(settings,decoder_tiles=None,cache_model=True))
        self.assertTrue(proof['warm_slice_tables']['cached_zero_verified'])
        with self.assertRaises(ValueError): bench.diagnostic_backend_proof(kernels=[],cpu_stats=cpu,
            **dict(settings,decoder_tiles=None,cache_model=True))
        proof = bench.diagnostic_backend_proof(kernels=[],cpu_stats={},
            **dict(settings,decoder_tiles=None,arm='rank256_uniform'))
        self.assertEqual(proof['decoder_gemm']['expected_calls'],0)
        self.assertEqual(proof['warm_slice_tables']['reference_expected_calls'],0)

    def test_link_units_and_independent_complete_baseline_rounds(self):
        items = [aggregate('splitzip_official_framed',1,8,'before'),
            aggregate('splitzip_official_framed',9,2,'after'), aggregate(bench.OWN[0],4,4)]
        result = bench.link_models(items)
        self.assertFalse(result['measured_network'])
        self.assertEqual({r['effective_gbps'] for r in result['rows']},set(bench.LINKS))
        row = next(r for r in result['rows'] if r['method'] == bench.OWN[0]
            and r['effective_gbps'] == 1 and r['timing_basis'] == 'isolated')
        self.assertAlmostEqual(row['wire_ms'],.008)
        self.assertAlmostEqual(row['precompressed_serial_ms'],4.008)
        self.assertAlmostEqual(row['fresh_serial_ms'],8.008)
        self.assertEqual(row['splitzip_targets']['precompressed']['round'],'after')
        self.assertEqual(row['splitzip_targets']['fresh']['round'],'before')

    def test_each_mode_compares_against_original_standalone_reference(self):
        items = []
        for n in bench.LENGTHS:
            for arm in bench.ARMS: items.append(aggregate('team_'+arm,40,20,tokens=n))
            items.extend(aggregate(method,20,10,tokens=n) for method in bench.OWN)
        result = bench.paired_insights(items)
        self.assertEqual(len(result),20)
        self.assertEqual({r['mode'] for r in result},set(bench.MODES))
        self.assertTrue(all(r['complete_bytes_delta'] == 0 and r['decode_speedup'] == 2 for r in result))

    def test_actual_cuda_gates_are_the_two_new_compute_implementations(self):
        self.assertEqual(bench.CUDA_TESTS,('tests.test_standalone_products.CudaTests',
            'tests.test_standalone_compute_codec.CudaTests'))
        source = (bench.ROOT/'scripts/bench_standalone_compute.py').read_text()
        self.assertIn('r.testsRun==2',source); self.assertIn('expected_tests=2',source)
        self.assertEqual(bench.CODER_TABLE_BYTES,57688)

    def test_parent_import_and_help_are_stdlib_only(self):
        subprocess.run([sys.executable,'-c',
            'import sys; import scripts.bench_standalone_compute; assert "torch" not in sys.modules'],
            cwd=bench.ROOT,check=True)
        value = subprocess.run([sys.executable,'scripts/bench_standalone_compute.py','--help'],
            cwd=bench.ROOT,text=True,capture_output=True,check=True)
        self.assertIn('Standalone',value.stdout)


if __name__ == '__main__':
    unittest.main()
