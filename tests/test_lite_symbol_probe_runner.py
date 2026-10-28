"""Data-selection boundaries and honest projections for the symbol probe."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.bench_lite_symbol_probe import (
    LINKS, layer_insights, main, projected_packet, required_budgets, select_candidate)
from kv_compare.lite_symbol_probe import cost, fit_tables, fold_counts


class RunnerTests(unittest.TestCase):
    def test_selection_uses_validation_and_predeclared_tie_order(self):
        rows = [dict(radius=255, contexts=32, validation_total_bits=10., test_bits=1.),
                dict(radius=127, contexts=32, validation_total_bits=10., test_bits=1000.),
                dict(radius=127, contexts=1, validation_total_bits=10., test_bits=2000.)]
        self.assertEqual((select_candidate(rows)['radius'], select_candidate(rows)['contexts']), (127, 1))
        rows[0]['validation_total_bits'] = 9.
        self.assertEqual(select_candidate(rows)['radius'], 255)

    def test_projection_charges_padded_states_directory_escapes_and_new_header(self):
        counts = np.zeros((1, 256), dtype=np.int64); counts[0, 127] = 64; counts[0, -1] = 1
        table = fit_tables(counts); rate = cost(counts, table)
        geometry = dict(values=65, state_bytes=32*4, directory_bytes=8)
        original = dict(format='causal_kv_v1', table_sha256='old', shape=[1, 65, 1])
        p = projected_packet(rate, geometry, original, 127, 1, table['table_sha256'])
        self.assertAlmostEqual(p['estimated_packet_bits'], rate['symbol_bits']+16+8*(128+8+p['hypothetical_framing_bytes']))
        self.assertEqual(p['hypothetical_header']['symbol_radius'], 127)
        self.assertEqual(p['hypothetical_header']['symbol_contexts'], 1)
        self.assertNotEqual(p['hypothetical_header']['format'], original['format'])
        self.assertFalse(p['actual_new_packet'])

    def test_all_five_requested_targets_are_budgets_without_candidate_latency(self):
        self.assertEqual(LINKS, (1., 100., 200., 300., 400.))
        estimates = [dict(method='selected', estimated_packet_bytes=33360893)]
        baselines = [dict(method='splitzip_official_framed', packet_bytes=44197315, decode_ms=.5169091746211052)]
        rows = required_budgets(estimates, baselines)
        target = next(r for r in rows if r['effective_gbps']==300.)
        self.assertAlmostEqual(target['decode_budget_ms'], .8058804279544385)
        self.assertTrue(all(not r['candidate_decode_measured'] and r['candidate_packet_estimated'] and not r['baseline_retimed'] for r in rows))
        self.assertTrue(all('candidate_decode_ms' not in r and 'fetch_ms' not in r for r in rows))

    def test_layer_insights_accept_unconditional_selection_without_test_tuning(self):
        counts = np.zeros((1, 32, 4096), dtype=np.int64)
        counts[0, 0, 2047] = 10; counts[0, 1, 2048] = 5
        unconditional = fit_tables(fold_counts(counts.sum(0), 127, contexts=1))
        rows = layer_insights(counts, unconditional, unconditional, 127)
        self.assertEqual(rows[0]['values'], 15)
        self.assertAlmostEqual(rows[0]['context_gain_bpv'], 0.)

    def test_cpu_does_not_publish_a_gpu_rate_probe(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--output', tmp, '--device', 'cpu']), 1)
            result = json.loads((Path(tmp)/'results.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['gpu_measured'])


if __name__ == '__main__':
    unittest.main()
