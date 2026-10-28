"""Balanced comparison order and honest CPU/lookup accounting gates."""
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.bench_lite_entropy import ARMS, arm_order, budgets, entropy_identity, main


class RunnerTests(unittest.TestCase):
    def test_eight_round_order_balances_each_position_and_direction(self):
        rounds = [arm_order(i) for i in range(8)]
        self.assertTrue(all(set(order)==set(ARMS) and len(order)==len(ARMS) for order in rounds))
        for position in range(len(ARMS)):
            self.assertEqual(Counter(order[position] for order in rounds), Counter({arm:2 for arm in ARMS}))
        self.assertEqual(rounds[4], rounds[0][::-1])
        for index in (-1, 1.5, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                arm_order(index)

    def test_receivers_ignore_cold_build_time_but_bind_lookup_configuration_and_bytes(self):
        a = dict(lookup='full', refine=2, full_lut_bytes=1310720,
                 full_lut_sha256='original-table', full_lut_construction_wall_ms=3.)
        b = dict(a, full_lut_construction_wall_ms=100.)
        self.assertEqual(entropy_identity(a), entropy_identity(b))
        for changed in (dict(b, refine=0), dict(b, full_lut_bytes=0), dict(b, full_lut_sha256='other-table')):
            self.assertNotEqual(entropy_identity(a), entropy_identity(changed))

    def test_latest_high_link_budget_uses_retimed_complete_packets(self):
        rows = [dict(method='splitzip_official_framed', packet_bytes=44197315, decode_ms=.5316790193319321),
                dict(method='lite_entropy_reference_graph', packet_bytes=33360893, decode_ms=83.24340730905533)]
        result = budgets(rows, links=[1., 400.])
        self.assertAlmostEqual(result[0]['decode_budget_ms'], 87.22305501933193)
        self.assertAlmostEqual(result[1]['decode_budget_ms'], .7484074593319321)
        self.assertTrue(all(not r['measured_network'] for r in result))

    def test_cpu_cannot_produce_a_completed_gpu_comparison(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--output', tmp, '--device', 'cpu']), 1)
            result = json.loads((Path(tmp)/'results.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['gpu_measured'])
            self.assertEqual(result['rows'], [])


if __name__ == '__main__':
    unittest.main()
