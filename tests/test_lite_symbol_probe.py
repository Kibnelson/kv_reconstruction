"""Exact reference symbols/rate accounting and known-KV replay parity."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from kv_compare.core import float_to_words, words_to_float
from kv_compare.lite_symbol_probe import (DEFAULT_RADII, ESCAPE_DIAGNOSTIC_LABELS,
    cost, escape_diagnostics, fit_tables, fold_counts, reconstruct, replay,
    scale_classes, symbolize, symbol_histograms, update_histograms)


class SymbolTests(unittest.TestCase):
    def test_every_bf16_pattern_round_trips_with_near_and_far_predictions(self):
        words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        own = words_to_float(words).float()
        finite = ((words.to(torch.int32) & 0x7F80) != 0x7F80)
        for radius in DEFAULT_RADII:
            for mu in (own, torch.zeros_like(own), torch.full_like(own, -0.)):
                with self.subTest(radius=radius, prediction='own' if mu is own else 'zero'):
                    symbols, raw = symbolize(words, mu, radius)
                    restored = reconstruct(symbols, mu, raw, radius)
                    self.assertTrue(torch.equal(restored, words))
                    self.assertTrue(bool((symbols[~finite] == 2 * radius + 1).all()))
                    if mu is own:
                        self.assertTrue(bool((symbols[finite] == radius).all()))
        # Adjacent integer ranks preserve the distinction between signed zeros.
        words = torch.tensor([0, -32768], dtype=torch.int16)
        for mu in (torch.zeros(2), torch.full((2,), -0.)):
            symbols, raw = symbolize(words, mu, 1)
            self.assertTrue(torch.equal(reconstruct(symbols, mu, raw, 1), words))
            self.assertEqual(symbols.tolist(), [1, 0] if not torch.signbit(mu[0]) else [2, 1])

    def test_prediction_rounds_ties_to_even_and_nonfinite_predictions_escape(self):
        # Exact halfway float32 bit patterns above even/odd BF16 words.
        mu = torch.tensor([0x3F808000, 0x3F818000, -1082097664, -1082032128], dtype=torch.int32).view(torch.float32)
        expected = torch.tensor([0x3F80, 0x3F82, 0xBF80, 0xBF82], dtype=torch.int32).to(torch.int16)
        symbols, raw = symbolize(expected, mu, 0)
        self.assertEqual(symbols.tolist(), [0, 0, 0, 0])
        self.assertTrue(torch.equal(reconstruct(symbols, mu, raw, 0), expected))
        bad = torch.tensor([float('inf'), -float('inf'), float('nan'), 3.4e38])
        original = torch.tensor([1, 0, -32768, 0x1234], dtype=torch.int16)
        symbols, raw = symbolize(original, bad, 127)
        self.assertEqual(symbols.tolist(), [255] * 4)
        self.assertTrue(torch.equal(reconstruct(symbols, bad, raw, 127), original))

    def test_scale_classes_have_fixed_integer_exponent_definition(self):
        mu = torch.tensor([1., -4., 0., 2. ** -133, 1., 0.])
        s = torch.tensor([1., 1., 1., 1., 2. ** -30, 2. ** -126])
        self.assertEqual(scale_classes(mu, s).tolist(), [7, 5, 31, 31, 0, 7])
        self.assertTrue(torch.equal(scale_classes(mu, s), scale_classes(mu, s, validate=False)))
        for invalid in (0., -1., float('inf'), float('nan')):
            with self.assertRaisesRegex(ValueError, 'positive'):
                scale_classes(torch.zeros(1), torch.tensor([invalid]))

    def test_one_histogram_folds_to_direct_symbols_and_preserves_layer_totals(self):
        words = torch.arange(301, dtype=torch.int32).mul(211).to(torch.int16)
        mu = torch.linspace(-.5, .5, words.numel())
        s = torch.linspace(.01, .3, words.numel())
        counts = symbol_histograms(words, mu, s)
        classes = scale_classes(mu, s)
        for radius, found in counts.items():
            symbols, _ = symbolize(words, mu, radius)
            alphabet = 2 * radius + 2
            direct = torch.bincount((classes * alphabet + symbols).flatten(), minlength=32 * alphabet).reshape(32, alphabet)
            self.assertTrue(torch.equal(found, direct)); self.assertEqual(int(found.sum()), words.numel())
            layers = torch.stack((counts[2047], counts[2047] * 3))
            folded = fold_counts(layers, radius)
            self.assertTrue(torch.equal(folded.sum(-1), layers.sum(-1)))
            unconditional = fold_counts(layers.numpy(), radius, contexts=1)
            np.testing.assert_array_equal(unconditional, folded.sum(-2, keepdim=True).numpy())
        target = {r: torch.zeros_like(c) for r, c in counts.items()}
        update_histograms(target, words, mu, s)
        for r in counts: self.assertTrue(torch.equal(target[r], counts[r]))

    def test_escape_mechanism_counts_are_predeclared_and_overlap(self):
        words = float_to_words(torch.tensor([-1., 1., .25, -0., 0.])).to(torch.int16)
        mu = torch.tensor([1., 0., .25, 0., -0.])
        diag = escape_diagnostics(words, mu, radii=(0, 127))
        col = {name: i for i, name in enumerate(ESCAPE_DIAGNOSTIC_LABELS)}
        self.assertEqual(diag[:, col['values']].tolist(), [5, 5])
        self.assertEqual(diag[:, col['finite_sign_crossing']].tolist(), [3, 3])
        self.assertEqual(diag[:, col['finite_predicted_zero_subnormal']].tolist(), [3, 3])
        for i, radius in enumerate((0, 127)):
            symbols, _ = symbolize(words, mu, radius)
            self.assertEqual(int(diag[i, col['escapes']]), int((symbols == 2 * radius + 1).sum()))
        self.assertGreater(int(diag[0, col['escape_finite_sign_crossing']]), 0)
        self.assertGreater(int(diag[0, col['escape_finite_predicted_zero_subnormal']]), 0)

    def test_positive_frequency_normalization_zero_rows_ties_and_large_counts(self):
        fitted = fit_tables(np.array([[1, 1, 0]], dtype=np.int64))
        np.testing.assert_array_equal(fitted['freq'], [[32768, 32767, 1]])
        for counts in (np.zeros((32, 256), dtype=np.int64),
                       np.array([[2 ** 62, 2 ** 62, 0]], dtype=np.int64)):
            a = fit_tables(counts); b = fit_tables(counts)
            self.assertTrue(np.all(a['freq'] > 0))
            np.testing.assert_array_equal(a['freq'].sum(-1), np.full(counts.shape[0], 65536))
            self.assertTrue(np.all(np.diff(a['cdf'], axis=-1) > 0))
            self.assertTrue(np.all(a['cdf'][:, 0] == 0)); self.assertTrue(np.all(a['cdf'][:, -1] == 65536))
            self.assertEqual(a['table_sha256'], b['table_sha256'])
            physical = a['freq'].nbytes + a['cdf'].nbytes
            self.assertEqual(a['physical_probe_table_bytes'], physical)
            self.assertEqual(a['projected_inverse_lut_bytes'], counts.shape[0] * 65536 * 2)
            self.assertEqual(a['projected_decoder_table_bytes'], physical + counts.shape[0] * 65536 * 2)

    def test_rate_cost_charges_each_escape_raw_word_and_does_not_refit(self):
        train = np.array([[1, 1, 1, 1]], dtype=np.int64)
        table = fit_tables(train); frozen = table['freq'].copy()
        validation = np.array([[0, 3, 0, 2]], dtype=np.int64)
        score = cost(validation, table)
        self.assertEqual(score['values'], 5); self.assertEqual(score['escapes'], 2)
        self.assertEqual(score['symbol_bits'], 10.); self.assertEqual(score['escape_bits'], 32)
        self.assertEqual(score['total_bits'], 42.); self.assertEqual(score['bpv'], 8.4)
        cost(np.array([[100, 0, 0, 0]], dtype=np.int64), table)
        np.testing.assert_array_equal(table['freq'], frozen)


def replay_parity(test, device):
    from kv_compare.lite_finish import FinishEngine
    from tests.test_llama_causal import tiny_state
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); tiny_state(root)
        source = float_to_words(torch.linspace(-.3, .3, 80).reshape(2, 5, 8)).to(device, torch.int16)
        for fmt in ('bf16', 'fp32'):
            engine = FinishEngine(root, device, predictor_format=fmt, values_per_stream=2, block=8)
            observed = []
            original = engine.lossless.encode
            def spy(words, mu, s, **kwargs):
                observed.append((words.clone(), mu.clone(), s.clone()))
                return original(words, mu, s, **kwargs)
            engine.lossless.encode = spy
            engine.encode(source, start=7)
            engine.lossless.encode = original
            emitted = []
            def callback(layer, group, words, mu, s):
                emitted.append((layer, group, words.clone(), mu.clone(), s.clone()))
                counts = symbol_histograms(words, mu, s, radii=(2047,), validate=False)[2047]
                test.assertEqual(int(counts.sum()), words.numel())
            # No entropy entry point may be reached by known-KV replay.
            from unittest import mock
            with (mock.patch.object(engine.lossless, 'encode', side_effect=AssertionError('entropy encode reached')),
                  mock.patch.object(engine.lossless, 'decode', side_effect=AssertionError('entropy decode reached'))):
                replay(engine, source, 7, callback)
            order = [(l, g) for l, layer in enumerate(engine.layers) for g in range(len(layer['groups']))]
            test.assertEqual([(l, g) for l, g, *_ in emitted], order)
            test.assertEqual(len(observed), len(emitted))
            for expected, (_, _, *actual) in zip(observed, emitted):
                for a, b in zip(expected, actual):
                    test.assertEqual(a.dtype, b.dtype)
                    test.assertTrue(torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a,
                                                b.view(torch.int32) if b.dtype == torch.float32 else b))
            test.assertEqual(sum(w.numel() for _, _, w, _, _ in emitted), source.numel())
        bad = source.clone(); bad[0, 0, 0] = 0x7FC1
        with test.assertRaisesRegex(ValueError, 'raw fallback'):
            replay(engine, bad, 0, lambda *args: None)
        with test.assertRaises(ValueError): replay(engine, source.float(), 0, lambda *args: None)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def setUp(self):
        self.enabled = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.enabled, warn_only=self.warn)

    def test_known_kv_predictions_and_scales_match_original_encoder_bitwise(self):
        replay_parity(self, torch.device('cpu'))


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'), 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def setUp(self):
        self.enabled = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device('cuda', torch.cuda.current_device())

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.use_deterministic_algorithms(self.enabled, warn_only=self.warn)

    def test_actual_cuda_replay_matches_encoder_and_unweighted_histograms_work(self):
        replay_parity(self, self.device)

    def test_actual_cuda_all_word_patterns_preserve_escape_payloads(self):
        words = torch.arange(65536, dtype=torch.int32).to(self.device, torch.int16)
        mu = words_to_float(words).float(); s = torch.ones_like(mu)
        for radius in DEFAULT_RADII:
            symbols, raw = symbolize(words, mu, radius)
            self.assertTrue(torch.equal(reconstruct(symbols, mu, raw, radius), words))
        counts = symbol_histograms(words, mu, s, radii=(2047,), validate=False)[2047]
        self.assertEqual(int(counts.sum()), words.numel())


if __name__ == '__main__':
    unittest.main()
