"""Software checks for calibration and fresh-receiver contracts, without CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from scripts import bench_combined_transfer as bench


def selections():
    result = {}
    for n in bench.LENGTHS:
        for method in ('raw_bf16', 'team_rank256_uniform', 'splitzip_official_framed',
                       'exans_official_fixed_framed'):
            codec = {
                'team_rank256_uniform': dict(kind='team', state='/state',
                    manifest_sha256='1' * 64, state_sha256='2' * 64),
                'splitzip_official_framed': dict(kind='splitzip_official_framed',
                    source='/splitzip', state='/luts.npz', state_sha256='3' * 64),
                'exans_official_fixed_framed': dict(kind='exans', source='/exans',
                    library='/libexans.so', library_sha256='4' * 64, blocks=59 if n == 512 else 941),
                'raw_bf16': dict(kind='raw_bf16'),
            }[method]
            entries = [dict(id=f'{method}_{doc}_{n}', document=f'test_{doc}_t{n}',
                shape=[28, n, 2048], values=28 * n * 2048,
                packet=f'/{method}/{doc}/{n}.kvc', packet_sha256='5' * 64,
                packet_bytes=100, original_word_sha256='6' * 64) for doc in ('text', 'code')]
            result[f'{method}_t{n}'] = dict(method=method, tokens=n, codec=codec, entries=entries)
    return result


def calibration_rows():
    return [dict(document='tables', tokens=n, k=k, packet_bytes=1_000_000,
        encode_ms=1.0, decode_ms=1.0) for n in bench.LENGTHS for k in bench.KGRID]


def policy():
    value = bench.freeze_policy(calibration_rows())
    value['masks'] = {str(k): [i < k for i in range(28)] for k in bench.KGRID}
    value['policy_sha256'] = bench.digest_json(value)
    return value


class CalibrationTests(unittest.TestCase):
    def test_rank_uses_native_bytes_and_words_with_stable_layer_ties(self):
        sz = [100] * 28
        words = [50] * 28
        words[9], words[3] = 10, 20
        words[27] = 60
        order, benefits = bench.rank_layer_order(sz, words)
        self.assertEqual(order[:2], [9, 3])
        self.assertEqual(order[2:-1], [i for i in range(27) if i not in (3, 9)])
        self.assertEqual(order[-1], 27)
        self.assertEqual(benefits[9], 640)
        self.assertEqual(benefits[27], -160)
        for bad in ([1] * 27, [True] * 28, [-1] * 28):
            with self.assertRaises(ValueError):
                bench.rank_layer_order(bad, words)

    def test_policy_units_and_objectives_use_full_costs(self):
        rows = calibration_rows()
        for row in rows:
            row.update(packet_bytes=1_000_000, encode_ms=100.0, decode_ms=100.0)
            if row['k'] == 0:
                row.update(encode_ms=1.0, decode_ms=1.0)
            if row['k'] == 28:
                row.update(packet_bytes=500_000, encode_ms=10.0, decode_ms=2.0)
        chosen = bench.freeze_policy(rows)
        first = [p for p in chosen['policies'] if p['tokens'] == 8192 and p['effective_gbps'] == 1]
        self.assertEqual({p['objective']: p['k'] for p in first}, dict(precompressed=28, fresh=0))
        pre = next(p for p in first if p['objective'] == 'precompressed')
        self.assertEqual(pre['calibration_serial_ms'], 6.0)  # 4 ms wire + 2 ms decode.
        fast = [p for p in chosen['policies'] if p['effective_gbps'] == 400]
        self.assertTrue(all(p['k'] == 0 for p in fast))

    def test_policy_rejects_test_selection_missing_duplicate_and_invalid_samples(self):
        mutations = []
        x = calibration_rows(); x[0]['document'] = 'test_text_t512'; mutations.append(x)
        x = calibration_rows(); x.pop(); mutations.append(x)
        x = calibration_rows(); x.append(dict(x[-1])); mutations.append(x)
        x = calibration_rows(); x.append(dict(x[-1], tokens=1024)); mutations.append(x)
        x = calibration_rows(); x[0]['packet_bytes'] = 0; mutations.append(x)
        x = calibration_rows(); x[0]['decode_ms'] = float('nan'); mutations.append(x)
        x = calibration_rows(); x[0]['encode_ms'] = -1; mutations.append(x)
        for rows in mutations:
            with self.assertRaises(ValueError): bench.freeze_policy(rows)

    def test_policy_ties_choose_zero_and_digest_is_order_independent(self):
        value = bench.freeze_policy(calibration_rows())
        self.assertEqual(len(value['policies']), 20)
        self.assertTrue(all(row['k'] == 0 for row in value['policies']))
        self.assertEqual(bench.digest_json({'a': 1, 'b': 2}), bench.digest_json({'b': 2, 'a': 1}))

    def test_test_application_uses_frozen_choice_even_if_another_test_arm_is_faster(self):
        frozen = policy()
        rows = []
        for n in bench.LENGTHS:
            for k in bench.KGRID:
                for doc in ('text', 'code'):
                    ms = 100 if k == 0 else 1
                    timing = dict(batched=dict(wall_mean_ms=ms), isolated=dict(wall_median_ms=ms + 1))
                    rows.append(dict(tokens=n, method=f'mixed_rank256_k{k}', document=f'test_{doc}_t{n}',
                        packet_bytes=1_000_000, encode=timing, decode=copy.deepcopy(timing)))
        applied = bench.select_policy_rows(rows, frozen)
        self.assertEqual(len(applied), 80)
        self.assertTrue(all(row['k'] == 0 for row in applied))
        row = next(x for x in applied if x['tokens'] == 8192 and x['effective_gbps'] == 1
                   and x['timing_basis'] == 'batched' and x['objective'] == 'fresh')
        self.assertEqual(row['serial_ms'], 208)
        self.assertFalse(row['measured_network'])
        with self.assertRaises(ValueError): bench.select_policy_rows(rows[:-9], frozen)


class ReceiverAndPlanTests(unittest.TestCase):
    def requests(self):
        for _, q in bench.new_job_plan(selections(), policy()):
            yield {k: q[k] for k in ('method', 'tokens', 'codec', 'execution')} | dict(entries=q['entries'], output='/receiver.json')

    def test_fresh_receiver_accepts_only_packets_state_and_original_word_assertions(self):
        requests = list(self.requests())
        self.assertEqual(len(requests), 12)
        for q in requests: bench.validate_receiver_request(q)
        base = requests[1]
        mutations = []
        x = copy.deepcopy(base); x['raw_entries'] = []; mutations.append(x)
        x = copy.deepcopy(base); x['entries'][0]['raw_words'] = '/oracle.npy'; mutations.append(x)
        x = copy.deepcopy(base); x['entries'][0]['shape'][1] += 1; mutations.append(x)
        x = copy.deepcopy(base); x['entries'][1]['document'] = x['entries'][0]['document']; mutations.append(x)
        x = copy.deepcopy(base); x['codec']['raw_words'] = '/oracle.npy'; mutations.append(x)
        for q in mutations:
            with self.assertRaises(ValueError): bench.validate_receiver_request(q)

    def test_job_plan_preserves_target_words_and_physical_exans_capacity(self):
        source, frozen = selections(), policy()
        original = copy.deepcopy(source)
        jobs = bench.new_job_plan(source, frozen)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(source, original)
        for _, q in jobs:
            n = q['tokens']
            self.assertEqual(q['entries'], source[f'team_rank256_uniform_t{n}']['entries'])
            self.assertEqual(q['raw_entries'], source[f'raw_bf16_t{n}']['entries'])
            self.assertEqual(q['policy_sha256'], frozen['policy_sha256'])
            if q['method'] == bench.COMPACT:
                self.assertEqual(q['codec']['blocks'], 59 if n == 512 else 941)
            else:
                self.assertEqual(sum(q['codec']['mask']), int(q['method'].rsplit('k', 1)[1]))

    def test_parent_import_does_not_import_torch_or_create_gpu_state(self):
        code = 'import sys; import scripts.bench_combined_transfer; assert "torch" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], cwd=bench.ROOT, check=True)

    def test_worker_commands_preserve_each_existing_entrypoint(self):
        old = bench.worker_argv('/python', '/request.json', 'splitzip_official_framed')
        new = bench.worker_argv('/python', '/request.json', bench.COMPACT)
        self.assertEqual(old[1], str(bench.ROOT / 'scripts/frozen_transfer_worker.py'))
        self.assertEqual(old[2:], ['--request', '/request.json', '--device', 'cuda:0'])
        self.assertEqual(new[1], str(Path(bench.__file__).resolve()))
        self.assertEqual(new[2:], ['--worker', '/request.json', '--device', 'cuda:0'])

    def test_extension_manifest_matches_unchanged_team_body_and_original_core(self):
        root = bench.ROOT / 'kv_compare'
        manifest = json.loads((root / 'team_extensions/source_manifest.json').read_text())
        self.assertEqual(manifest['commit'], '8a40dd98282ed364a8fe1077d171a1922277c1ec')
        self.assertEqual(manifest['files']['mixed.py'], '3193e510df697087ea300e233217b04303f1370b543ce10bf8f87a2f577442e7')
        for name, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((root / 'team_extensions' / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((root / 'team_onestep/source_manifest.json').read_bytes()).hexdigest(), manifest['core_manifest_sha256'])


if __name__ == '__main__': unittest.main()
