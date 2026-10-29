"""Software protocol checks; no GPU runtime or performance is needed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from scripts import bench_lossless_mixed_fast as bench


def inputs():
    recipe = json.loads((bench.ROOT / bench.frozen.FROZEN / 'results.json').read_text())
    return recipe['recipe']['selections']


def policy():
    return bench.verify_previous()[0]


def paired_rows():
    result = []
    for n in bench.LENGTHS:
        for k in bench.KGRID:
            for document in ('test_text', 'test_code'):
                for mode in bench.MODES:
                    bundled = mode in ('bundle', 'both')
                    result.append(dict(method=f'fast_{mode}_k{k}', tokens=n,
                        document=f'{document}_t{n}', original_word_sha256='1' * 64,
                        packet_bytes=950 if bundled else 1000,
                        packet_sha256=('b' if bundled else 'a') * 64,
                        rans_fields={name: dict(bytes=64, sha256='c' * 64)
                            for name in ('rans_words', 'rans_offsets', 'rans_relative')}))
    return result


def aggregate(method, phase, encode, decode, packet_bytes=1000, tokens=512):
    return dict(method=method, round=phase, tokens=tokens, packet_bytes=packet_bytes,
        batched=dict(encode_ms=encode, decode_ms=decode),
        isolated=dict(encode_ms=encode, decode_ms=decode))


class FrozenPlanTests(unittest.TestCase):
    def test_previous_exact_package_and_policy_are_bound(self):
        value, receipt = bench.verify_previous()
        self.assertEqual(receipt['package']['files'], 177)
        self.assertEqual([i for i, chosen in enumerate(value['masks']['2']) if chosen], [1, 27])
        self.assertEqual(sum(value['masks']['14']), 14)
        clone = dict(value); digest = clone.pop('policy_sha256')
        self.assertEqual(bench.digest_json(clone), digest)
        self.assertEqual(set(value['reference_packets']), {
            f'mixed_rank256_k{k}_t{n}' for k in bench.KGRID for n in bench.LENGTHS})

    def test_all_execution_arms_reuse_masks_words_and_state(self):
        source, frozen = inputs(), policy()
        original = copy.deepcopy(source)
        jobs = bench.job_plan(source, frozen)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len(bench.compact_plan(source, frozen)), 2)
        self.assertEqual(len(bench.frozen.job_plan(source)) + len(jobs) + 2, 34)
        for _, q in jobs:
            mode, k = bench.method_parts(q['method'])
            self.assertEqual(q['codec']['mask'], frozen['masks'][str(k)])
            self.assertEqual((q['codec']['selective'], q['codec']['bundle']), bench.FLAGS[mode])
            self.assertEqual(q['entries'], source[f"team_rank256_uniform_t{q['tokens']}"]['entries'])
            self.assertEqual(q['codec']['predictor_state_sha256'], source['team_rank256_uniform_t512']['codec']['state_sha256'])
            self.assertEqual(len(q['reference_entries']), 2)
            self.assertEqual(q['policy_sha256'], frozen['policy_sha256'])
        self.assertEqual(source, original)

    def test_method_names_exclude_lossy_or_undeclared_variants(self):
        for name in bench.METHODS:
            mode, k = bench.method_parts(name)
            self.assertIn(mode, bench.MODES); self.assertIn(k, (2, 14))
        for name in ('fast_lossy_k2', 'fast_both_k28', 'fast_both_k2_extra', 'fast_both_k02'):
            with self.assertRaises(ValueError): bench.method_parts(name)


class ReceiverAndEquivalenceTests(unittest.TestCase):
    def requests(self):
        for _, q in bench.job_plan(inputs(), policy()):
            entries = [{**{k:e[k] for k in ('id', 'document', 'shape', 'values', 'original_word_sha256')},
                'packet':'/actual.kvc', 'packet_sha256':'a' * 64, 'packet_bytes':1000} for e in q['entries']]
            yield {k:q[k] for k in ('method', 'tokens', 'codec', 'execution')} | dict(entries=entries, output='/receiver.json')

    def test_receiver_sees_only_checked_wire_shared_state_and_assertions(self):
        requests = list(self.requests())
        self.assertEqual(len(requests), 16)
        for q in requests: bench.validate_receiver_request(q)
        base = requests[-1]
        bad = []
        x=copy.deepcopy(base); x['raw_entries']=[]; bad.append(x)
        x=copy.deepcopy(base); x['entries'][0]['raw_words']='/oracle.npy'; bad.append(x)
        x=copy.deepcopy(base); x['codec']['raw_words']='/oracle.npy'; bad.append(x)
        x=copy.deepcopy(base); x['codec']['selective']=False; bad.append(x)
        x=copy.deepcopy(base); x['codec']['bundle']=1; bad.append(x)
        x=copy.deepcopy(base); x['codec']['mask'][0]=not x['codec']['mask'][0]; bad.append(x)
        x=copy.deepcopy(base); x['entries'][0]['shape'][1]+=1; bad.append(x)
        x=copy.deepcopy(base); x['entries'][1]['document']=x['entries'][0]['document']; bad.append(x)
        for q in bad:
            with self.assertRaises(ValueError): bench.validate_receiver_request(q)

    def test_all_four_arms_must_preserve_required_packet_and_rans_bytes(self):
        rows = paired_rows()
        proofs = bench.validate_packet_equivalence(rows)
        self.assertEqual(len(proofs), 8)
        self.assertTrue(all(x['all_rans_fields_equal'] and x['selective_reference_wire_equal']
                            and x['both_bundle_wire_equal'] for x in proofs))
        variants = []
        x=copy.deepcopy(rows); x[1]['packet_sha256']='d' * 64; variants.append(x)
        x=copy.deepcopy(rows); x[3]['packet_bytes']+=4; variants.append(x)
        x=copy.deepcopy(rows); x[2]['rans_fields']['rans_words']['sha256']='d' * 64; variants.append(x)
        x=copy.deepcopy(rows); x[1]['original_word_sha256']='d' * 64; variants.append(x)
        x=copy.deepcopy(rows); x.append(dict(x[0])); variants.append(x)
        x=copy.deepcopy(rows); x.pop(); variants.append(x)
        for rows in variants:
            with self.assertRaises(ValueError): bench.validate_packet_equivalence(rows)

    def test_incomplete_progress_never_claims_complete_equivalence(self):
        rows = paired_rows()
        self.assertEqual(bench.validate_packet_equivalence(rows[:3], complete=False), [])
        self.assertEqual(len(bench.validate_packet_equivalence(rows[:4], complete=False)), 1)
        with self.assertRaises(ValueError): bench.validate_packet_equivalence(rows[:4])


class MeasurementTests(unittest.TestCase):
    def test_link_units_and_objectives_keep_real_baseline_rounds(self):
        rows = [aggregate('splitzip_official_framed', 'before', 1, 8),
                aggregate('splitzip_official_framed', 'after', 9, 2),
                aggregate('fast_both_k2', 'paired', 4, 4)]
        modeled = bench.link_models(rows)
        self.assertFalse(modeled['measured_network'])
        self.assertEqual({x['effective_gbps'] for x in modeled['rows']}, {1, 5, 10, 25, 50, 75, 100, 200, 300, 400})
        row = next(x for x in modeled['rows'] if x['method']=='fast_both_k2'
                   and x['timing_basis']=='isolated' and x['effective_gbps']==1)
        self.assertAlmostEqual(row['wire_ms'], .008)
        self.assertAlmostEqual(row['precompressed_serial_ms'], 4.008)
        self.assertAlmostEqual(row['fresh_serial_ms'], 8.008)
        pre, fresh = row['splitzip_targets']['precompressed'], row['splitzip_targets']['fresh']
        self.assertEqual(pre['round'], 'after')
        self.assertEqual(fresh['round'], 'before')
        self.assertAlmostEqual(pre['decode_budget_ms'], 2)
        self.assertAlmostEqual(fresh['decode_budget_ms'], 5)
        self.assertLess(pre['packet_budget_bytes'], 0)  # Bytes alone cannot satisfy this target.

    def test_paired_insights_do_not_search_test_for_a_winner(self):
        rows = []
        for n in bench.LENGTHS:
            for k in bench.KGRID:
                for mode in bench.MODES:
                    e = 20 if mode in ('selective','both') else 40
                    d = 10 if mode in ('bundle','both') else 20
                    size = 950 if mode in ('bundle','both') else 1000
                    rows.append(aggregate(f'fast_{mode}_k{k}', 'paired', e, d, size, n))
        changes = bench.paired_insights(rows)
        self.assertEqual(len(changes), 24)
        both = [r for r in changes if r['mode']=='both']
        self.assertTrue(all(r['encode_speedup']==2 and r['decode_speedup']==2
                            and r['complete_bytes_delta']==-50 for r in both))
        selective = [r for r in changes if r['mode']=='selective']
        self.assertTrue(all(r['complete_bytes_delta']==0 and r['decode_speedup']==1 for r in selective))

    def test_parent_import_and_cli_help_do_not_create_tensor_runtime(self):
        subprocess.run([sys.executable, '-c',
            'import sys; import scripts.bench_lossless_mixed_fast; assert "torch" not in sys.modules'],
            cwd=bench.ROOT, check=True)
        result = subprocess.run([sys.executable, 'scripts/bench_lossless_mixed_fast.py', '--help'],
            cwd=bench.ROOT, text=True, capture_output=True, check=True)
        self.assertIn('Lossless', result.stdout)


if __name__ == '__main__': unittest.main()
