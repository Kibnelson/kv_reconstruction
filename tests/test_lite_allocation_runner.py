"""Accounting and runtime-policy checks without a GPU performance substitute."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.utils.deterministic

from scripts.bench_lite_allocation import budgets, fetch_model, main, runtime_snapshot
from scripts.package_lite_allocation import verify


class RunnerTests(unittest.TestCase):
    def test_latest_link_budgets_use_complete_packets_and_retimed_controls(self):
        rows = [dict(method='raw_bf16', packet_bytes=58720573, decode_ms=.3496813587844372),
                dict(method='splitzip_official_framed', packet_bytes=44197315, decode_ms=.5124039016664028),
                dict(method='lite_allocation_fill_off_graph', packet_bytes=33360893, decode_ms=88.22381356731057)]
        limits = budgets(rows, links=[1., 400.])
        sz = {r['effective_gbps']: r for r in limits if r['baseline']=='splitzip_official_framed'}
        self.assertAlmostEqual(sz[1.]['decode_budget_ms'], 87.20377990166641)
        self.assertAlmostEqual(sz[400.]['decode_budget_ms'], .7291323416664028)
        estimate = fetch_model(rows, links=[1.])
        self.assertFalse(estimate['measured_network'])
        self.assertAlmostEqual(estimate['links'][0]['estimates'][-1]['fetch_ms'], 355.1109575673106)
        for rate in (0., -1., float('nan'), float('inf')):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                fetch_model(rows, links=[rate])

    def test_native_fill_flag_is_inactive_without_deterministic_algorithms(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            torch.utils.deterministic.fill_uninitialized_memory = True
            torch.use_deterministic_algorithms(False)
            self.assertFalse(runtime_snapshot(torch)['effective_uninitialized_fill'])
            torch.use_deterministic_algorithms(True)
            self.assertTrue(runtime_snapshot(torch)['effective_uninitialized_fill'])
            torch.utils.deterministic.fill_uninitialized_memory = False
            self.assertFalse(runtime_snapshot(torch)['effective_uninitialized_fill'])
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = fill
            torch.use_deterministic_algorithms(enabled, warn_only=warn)

    def test_cpu_execution_is_recorded_as_failure_before_loading_parent(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--output', tmp, '--device', 'cpu']), 1)
            result = json.loads((Path(tmp)/'results.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['gpu_measured'])
            self.assertEqual(result['rows'], [])

    def test_packager_rejects_cpu_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'CPU evidence'):
                verify(Path(tmp), dict(status='completed', gpu_measured=False))


if __name__ == '__main__':
    unittest.main()
