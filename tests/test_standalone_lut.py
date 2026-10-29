"""Standalone full-inverse benchmark protocol checks; no tensor runtime."""
import copy
import json
import subprocess
import sys
from types import SimpleNamespace
import unittest

from scripts import bench_standalone_lut as bench


def selections():
    return bench.fixed_selections()


def policy():
    return bench.verify_previous()[0]


def requests():
    result = []
    for _, q in bench.job_plan(selections(), policy()):
        result.append({key:q[key] for key in ('method', 'tokens', 'codec', 'entries', 'execution')}
            | dict(output='/receiver.json'))
    return result


def parity_rows():
    p = policy(); rows = []
    for n in bench.LENGTHS:
        for arm in bench.ARMS:
            for ref in p['reference_packets'][f'team_{arm}_t{n}']:
                rows.append(copy.deepcopy(ref))
                rows.append(dict(copy.deepcopy(ref), method='standalone_lut_' + arm))
    return rows


def aggregate(method, e, d, phase='paired', n=8192):
    return dict(method=method, tokens=n, round=phase, packet_bytes=1000,
        batched=dict(encode_ms=e, decode_ms=d), isolated=dict(encode_ms=e, decode_ms=d))


class ReceiptAndPlanTests(unittest.TestCase):
    def test_completed_actual_cuda_comparison_and_original_packet_receipts(self):
        fixed, receipt = bench.verify_previous()
        self.assertEqual(receipt['package']['files'], 214)
        self.assertEqual(set(fixed['reference_packets']),
            {f'team_{arm}_t{n}' for arm in bench.ARMS for n in bench.LENGTHS})
        value = dict(fixed); digest = value.pop('policy_sha256')
        self.assertEqual(bench.digest_json(value), digest)
        self.assertEqual(fixed['full_lut_bytes'], 1310720)
        self.assertNotIn('masks', fixed)

    def test_focused_plan_preserves_standalone_arms_and_original_bytes(self):
        source = selections(); before = copy.deepcopy(source); fixed = policy()
        jobs = bench.job_plan(source, fixed)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len(jobs) + len(bench.frozen.job_plan(source)) + len(bench.compact_plan(source, fixed)), 22)
        for _, q in jobs:
            arm = bench.method_arm(q['method'])
            ref = source[f'team_{arm}_t{q["tokens"]}']
            self.assertEqual(q['entries'], ref['entries'])
            spec = dict(q['codec']); self.assertIs(spec.pop('full_lut'), True)
            spec['kind'] = 'team'; self.assertEqual(spec, ref['codec'])
            self.assertEqual(q['execution'], 'compact_search' if q['tokens'] == 512 else 'direct_search')
            self.assertEqual(len(q['reference_entries']), 2)
            self.assertNotIn('splitzip_source', q['codec']); self.assertNotIn('mask', q['codec'])
        self.assertEqual(source, before)

    def test_method_and_length_are_explicit(self):
        for method in ('team_dense_uniform', 'standalone_lut_dense_exact', 'standalone_lut_rank128_uniform',
            'standalone_lut_dense_uniform_extra', 'mixed_rank256_k28'):
            with self.assertRaises(ValueError): bench.method_arm(method)
        for tokens in (True, 512.0, 128, '512'):
            with self.assertRaises(ValueError): bench.execution_for(tokens)


class ReceiverTests(unittest.TestCase):
    def test_receiver_accepts_only_exact_state_and_original_packet_assertions(self):
        for q in requests():
            bench.validate_receiver_request(q)
        base = requests()[-1]; bad = []
        x = copy.deepcopy(base); x['raw_entries'] = []; bad.append(x)
        x = copy.deepcopy(base); x['codec']['mask'] = [True] * 28; bad.append(x)
        x = copy.deepcopy(base); x['codec']['splitzip_source'] = '/vendor'; bad.append(x)
        x = copy.deepcopy(base); x['codec']['predictions'] = '/oracle'; bad.append(x)
        x = copy.deepcopy(base); x['codec']['full_lut'] = False; bad.append(x)
        x = copy.deepcopy(base); x['codec']['full_lut'] = 1; bad.append(x)
        x = copy.deepcopy(base); x['codec']['state_sha256'] = 'f' * 64; bad.append(x)
        x = copy.deepcopy(base); x['codec']['arm'] = 'dense_uniform'; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['source_kv'] = '/oracle'; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['packet_sha256'] = 'a' * 64; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['packet_bytes'] += 1; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['shape'][1] += 1; bad.append(x)
        x = copy.deepcopy(base); x['entries'][0]['state_sha256'] = 'a' * 64; bad.append(x)
        x = copy.deepcopy(base); x['entries'][1]['document'] = x['entries'][0]['document']; bad.append(x)
        x = copy.deepcopy(base); x['execution'] = 'compact_search'; bad.append(x)
        for q in bad:
            with self.assertRaises(ValueError): bench.validate_receiver_request(q)

    def test_actual_encoded_packet_paths_may_change_but_wire_identity_may_not(self):
        q = requests()[0]
        for i, entry in enumerate(q['entries']):
            entry['packet'] = f'/new_run/actual_{i}.kvp'
        bench.validate_receiver_request(q)

    def test_graph_execution_remains_fixed_and_counts_are_checked(self):
        for q in requests():
            codec = SimpleNamespace(graph_capture_count=int(q['tokens'] == 512))
            self.assertEqual(bench.graph_count(q, codec), int(q['tokens'] == 512))
            codec.graph_capture_count += 1
            with self.assertRaises(AssertionError): bench.graph_count(q, codec)


class PacketParityTests(unittest.TestCase):
    def test_both_standalone_arms_preserve_complete_original_packets(self):
        rows = parity_rows(); proofs = bench.validate_packet_equivalence(rows)
        self.assertEqual(len(proofs), 8)
        self.assertTrue(all(r['original_wire_equal'] for r in proofs))
        self.assertEqual(bench.validate_packet_equivalence(rows[:1], complete=False), [])
        self.assertEqual(len(bench.validate_packet_equivalence(rows[:2], complete=False)), 1)
        bad = []
        for key, value in (('packet_bytes', 1), ('packet_sha256', 'a' * 64), ('original_word_sha256', 'a' * 64),
            ('shape', [28, 1, 2048]), ('values', 1)):
            x = copy.deepcopy(rows); x[1][key] = value; bad.append(x)
        bad.append(copy.deepcopy(rows[:-1]))
        bad.append(copy.deepcopy(rows) + [copy.deepcopy(rows[0])])
        for values in bad:
            with self.assertRaises(ValueError): bench.validate_packet_equivalence(values)


class MeasurementTests(unittest.TestCase):
    def test_same_complete_control_rounds_and_link_units(self):
        aggregates = [aggregate('splitzip_official_framed', 1, 8, 'before'),
            aggregate('splitzip_official_framed', 9, 2, 'after'),
            aggregate('standalone_lut_rank256_uniform', 4, 4)]
        result = bench.link_models(aggregates)
        self.assertFalse(result['measured_network'])
        self.assertEqual({r['effective_gbps'] for r in result['rows']}, set(bench.LINKS))
        row = next(r for r in result['rows'] if r['method'] == bench.OWN[1]
            and r['effective_gbps'] == 1 and r['timing_basis'] == 'isolated')
        self.assertAlmostEqual(row['wire_ms'], .008)
        self.assertAlmostEqual(row['precompressed_serial_ms'], 4.008)
        self.assertAlmostEqual(row['fresh_serial_ms'], 8.008)
        self.assertEqual(row['splitzip_targets']['precompressed']['round'], 'after')
        self.assertEqual(row['splitzip_targets']['fresh']['round'], 'before')

    def test_paired_comparison_keeps_dense_and_rank_separate(self):
        rows = []
        for n in bench.LENGTHS:
            for method in bench.OWN:
                rows += [aggregate(bench.reference_method(method), 40, 20, n=n),
                    aggregate(method, 20, 10, n=n)]
        result = bench.paired_insights(rows)
        self.assertEqual(len(result), 8)
        self.assertEqual({r['arm'] for r in result}, set(bench.ARMS))
        self.assertTrue(all(r['complete_bytes_delta'] == 0 and r['decode_speedup'] == 2 for r in result))

    def test_two_actual_cuda_gates_and_full_shared_table_are_declared(self):
        self.assertEqual(bench.CUDA_TESTS, ('tests.test_standalone_full_lut.CudaTests',
            'tests.test_standalone_lut_codec.CudaTests'))
        source = (bench.ROOT / 'scripts/bench_standalone_lut.py').read_text()
        self.assertIn('r.testsRun==2', source)
        self.assertIn('expected_tests=2', source)
        self.assertEqual(bench.FULL_LUT_BYTES, 10 * 65536 * 2)

    def test_parent_import_and_help_do_not_import_a_tensor_runtime(self):
        subprocess.run([sys.executable, '-c',
            'import sys; import scripts.bench_standalone_lut; assert "torch" not in sys.modules'],
            cwd=bench.ROOT, check=True)
        result = subprocess.run([sys.executable, 'scripts/bench_standalone_lut.py', '--help'],
            cwd=bench.ROOT, text=True, capture_output=True, check=True)
        self.assertIn('Standalone', result.stdout)


if __name__ == '__main__':
    unittest.main()
