"""Scientific/accounting guards for the GPU-resident decode experiment."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import bench_team_gpu as bench


def entry(document='test_text_t512'):
    shape = [28, 512, 2048]; values = 28*512*2048
    return dict(id='parent_'+document, document=document, shape=shape, values=values,
                packet='/frozen/'+document+'.kvp', packet_sha256='a'*64,
                packet_bytes=values, original_word_sha256='b'*64,
                complete_bpv=8., state_sha256='c'*64)


def row(method='team_dense_uniform', arm='compact', phase='paired', document='test_text_t512', wall=5.):
    e = entry(document)
    return dict(e, method=method, arm=arm, round=phase, tokens=512, original_exact=True,
                decode=dict(samples=[dict(wall_ms=wall, cuda_interval_ms=wall-.5)], wall_median_ms=wall))


class RunnerTests(unittest.TestCase):
    def test_balanced_arm_order_and_documents(self):
        entries = [entry(), entry('test_code_t512')]
        schedule = bench.sample_order(entries, bench.ARMS, 2, 6)
        self.assertEqual([s['arms'][0] for s in schedule[2:]], ['reference', 'compact', 'compact', 'reference', 'reference', 'compact'])
        self.assertEqual(sum(s['arms'][0] == 'compact' for s in schedule[2:]), 3)
        for i, item in enumerate(schedule):
            self.assertEqual(item['documents'], [e['document'] for e in (entries if i % 2 == 0 else entries[::-1])])
        native = bench.sample_order(entries, ['native'], 2, 6)
        self.assertTrue(all(s['arms'] == ['native'] for s in native))

    def test_baselines_bracket_paired_methods(self):
        selections = {}
        for n in bench.LENGTHS:
            for method in bench.METHODS:
                selections[f'{method}_t{n}'] = dict(method=method, tokens=n)
        plan = bench.selected_plan(selections)
        self.assertEqual(len(plan), 16)
        self.assertTrue(all(q['round'] == 'before' for _, q in plan[:6]))
        self.assertTrue(all(q['round'] == 'paired' for _, q in plan[6:10]))
        self.assertTrue(all(q['round'] == 'after' for _, q in plan[10:]))
        self.assertEqual([name.removeprefix('before_') for name, _ in plan[:6]],
                         [name.removeprefix('after_') for name, _ in plan[10:]][::-1])

    def test_aggregate_keeps_rounds_and_full_api_timing(self):
        rows = [row(), row(document='test_code_t512', wall=7.),
                row('splitzip_official_framed', 'native', 'before', wall=1.),
                row('splitzip_official_framed', 'native', 'after', wall=2.)]
        result = bench.aggregate(rows)
        self.assertEqual(len(result), 3)
        own = next(r for r in result if r['arm'] == 'compact')
        self.assertEqual(own['decode_ms'], 6.)
        self.assertEqual(own['decode_cuda_ms'], 5.5)
        self.assertIsNone(own['kernel_decode_ms'])
        self.assertEqual(own['saved_percent'], 50.)
        self.assertEqual({a['round'] for a in result}, {'paired', 'before', 'after'})

    def test_link_model_uses_bytes_plus_full_decode_and_conservative_control(self):
        values = [dict(tokens=512, method='team_dense_uniform', arm='compact', round='paired', packet_bytes=100_000_000, decode_ms=10.),
                  dict(tokens=512, method='splitzip_official_framed', arm='native', round='before', packet_bytes=120_000_000, decode_ms=2.),
                  dict(tokens=512, method='splitzip_official_framed', arm='native', round='after', packet_bytes=120_000_000, decode_ms=3.)]
        record = bench.link_models(values, (100., 400.))
        own = [r for r in record['rows'] if r['arm'] == 'compact']
        self.assertEqual(own[0]['wire_ms'], 8.)
        self.assertEqual(own[0]['precompressed_serial_ms'], 18.)
        self.assertEqual(own[0]['ideal_overlap_ms'], 10.)
        self.assertAlmostEqual(own[0]['decode_budget_ms'], 3.6)
        self.assertEqual(own[0]['splitzip_round'], 'before')
        self.assertAlmostEqual(own[1]['decode_budget_ms'], 2.4)
        self.assertFalse(record['measured_network'])
        for invalid in ((0.,), (-1.,), (float('nan'),), (float('inf'),)):
            with self.assertRaises(ValueError): bench.link_models(values, invalid)

    def test_only_original_complete_word_coverage_is_accepted(self):
        rows = []; checks = {}
        for document in ('test_text_t512', 'test_code_t512'):
            e = entry(document)
            rows.append(dict(e, method='team_dense_uniform', status='verified', original_exact=True,
                             decoded_word_sha256=e['original_word_sha256']))
            checks[e['id']] = dict(id=e['id'], verified=True, word_sha256=e['original_word_sha256'],
                                  packet_sha256=e['packet_sha256'], shape=e['shape'])
        valid = bench.validate_entries('team_dense_uniform', 512, rows, checks)
        self.assertEqual(valid, [entry(), entry('test_code_t512')])
        changed = copy.deepcopy(rows); changed[1]['values'] -= 1
        with self.assertRaises(ValueError): bench.validate_entries('team_dense_uniform', 512, changed, checks)
        changed = copy.deepcopy(rows); changed[0]['decoded_word_sha256'] = 'd'*64
        with self.assertRaises(ValueError): bench.validate_entries('team_dense_uniform', 512, changed, checks)
        with self.assertRaises(ValueError): bench.validate_entries('team_dense_uniform', 512, rows[:1], checks)

    def test_receiver_excludes_source_or_prediction_inputs(self):
        q = dict(method='team_dense_uniform', codec={}, entries=[entry(), entry('test_code_t512')],
                 arms=list(bench.ARMS), output='/tmp/receiver.json', round='paired')
        bench.valid_receiver_request(q)
        for key in ('inputs', 'captures', 'source_run', 'predictions'):
            changed = dict(q, **{key:[]})
            with self.assertRaises(ValueError): bench.valid_receiver_request(changed)
            changed = copy.deepcopy(q); changed['entries'][0][key] = 'forbidden'
            with self.assertRaises(ValueError): bench.valid_receiver_request(changed)

    def test_parent_receipts_reject_symlink_components_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'parent'; root.mkdir()
            folder = root/'actual'; folder.mkdir(); (folder/'data.json').write_text('{}')
            self.assertEqual(bench.parent_path(root, 'actual/data.json'), folder/'data.json')
            (root/'alias').symlink_to(folder, target_is_directory=True)
            with self.assertRaises(ValueError): bench.parent_path(root, 'alias/data.json')
            outside = Path(directory)/'outside'; outside.write_text('{}')
            with self.assertRaises(ValueError): bench.parent_path(root, '../outside')

    def test_device_gate_precedes_parent_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'output'
            with patch.object(bench, 'cuda_check', side_effect=RuntimeError('actual CUDA required')), \
                 patch.object(bench, 'read_parent') as read:
                self.assertEqual(bench.main(['--output', str(out), '--parent-run', '/must/not/read']), 1)
                read.assert_not_called()
            result = json.loads((out/'results.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['gpu_measured'])
            self.assertIn('actual CUDA required', (out/'summary.md').read_text())

    def test_cuda_kernel_summary_excludes_cpu_and_copy_events(self):
        events = [dict(cat='kernel', ph='X', name='rANS', dur=2000),
                  dict(cat='kernel', ph='X', name='rANS', dur=3000),
                  dict(cat='kernel', ph='X', name='gemm', dur=1000),
                  dict(cat='cpu_op', ph='X', name='rANS', dur=100000),
                  dict(cat='gpu_memcpy', ph='X', name='Memcpy', dur=100000)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.json'; path.write_text(json.dumps(dict(traceEvents=events)))
            self.assertEqual(bench.profile_kernels(path), [dict(name='rANS', calls=2, total_ms=5.),
                                                          dict(name='gemm', calls=1, total_ms=1.)])

    def test_packager_rejects_false_gpu_and_invalid_timings(self):
        from scripts.package_team_gpu import check_samples, verify
        record = dict(samples=[dict(wall_ms=1., cuda_interval_ms=.5)],
                      warmup_samples=[], wall_median_ms=1.)
        check_samples(record, 0, 1)
        for field, value in (('wall_ms', float('nan')), ('wall_ms', -1.),
                             ('cuda_interval_ms', float('inf')), ('cuda_interval_ms', -1.)):
            changed = copy.deepcopy(record); changed['samples'][0][field] = value
            with self.assertRaises(ValueError): check_samples(changed, 0, 1)
        with self.assertRaises(ValueError): check_samples(record, 1, 1)
        with self.assertRaises(ValueError): check_samples(dict(record, wall_median_ms=2.), 0, 1)
        failed = dict(format='team_gpu_v1', status='failed', gpu_measured=False,
                      jobs={}, profiles={}, rows=[], aggregates=[], link_models=bench.link_models([]))
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(verify(Path(directory), failed))
            with self.assertRaises(ValueError): verify(Path(directory), dict(failed, status='completed'))

    def test_profile_assertions_excluded_at_runtime(self):
        """Execute the real profile orchestration with guarded mock contexts."""
        active = {'torch':False, 'cpu':False}
        class Prof:
            def __enter__(self): active['torch'] = True; return self
            def __exit__(self, *args): active['torch'] = False
            def key_averages(self): return []
            def export_chrome_trace(self, path): Path(path).write_text('{"traceEvents":[]}')
        class CPU:
            def enable(self): active['cpu'] = True
            def disable(self): active['cpu'] = False
        class Codec:
            def close(self): pass
        class Cuda:
            @staticmethod
            def synchronize(*args): pass
        class Torch:
            cuda = Cuda()
        import contextlib
        class Stats:
            def __init__(self, obj): self.stats = {}
        def check(*args):
            self.assertFalse(active['cpu']); self.assertFalse(active['torch'])
        record = dict(samples=[dict(wall_ms=1., cuda_interval_ms=1.)], warmup_samples=[], wall_median_ms=1.)
        with tempfile.TemporaryDirectory() as directory:
            q = dict(method='splitzip_official_framed', codec={}, arms=['native'], entries=[entry('test_text_t8192')],
                     tokens=8192, output=directory, round='profile')
            with patch.object(bench, 'runtime_snapshot', return_value={}), \
                 patch.object(bench, 'build_codecs', return_value={'native':Codec()}), \
                 patch.object(bench, 'load_packets', return_value=[object()]), \
                 patch.object(bench, 'prepare_codecs', return_value={}), \
                 patch.object(bench, 'decode_call', return_value=object()), \
                 patch.object(bench, 'check_original', side_effect=check), \
                 patch.object(bench, 'timed', side_effect=lambda fn,*args:(fn(), record)), \
                 patch.object(bench.cProfile, 'Profile', return_value=CPU()), \
                 patch.object(bench.pstats, 'Stats', Stats), \
                 patch('torch.profiler.profile', return_value=Prof()), \
                 patch('torch.profiler.record_function', side_effect=lambda *args:contextlib.nullcontext()):
                bench.profile_worker(q, Torch(), 'cuda:0')
            result = json.loads((Path(directory)/'profile.json').read_text())
            self.assertTrue(result['assertions_outside_profilers'])
            self.assertEqual(result['profile_calls'], 3)


if __name__ == '__main__': unittest.main()
