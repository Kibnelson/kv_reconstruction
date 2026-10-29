"""Selective lossless preparation parity; interpreter checks are not timings."""
import functools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare import mixed_transfer as mt
from kv_compare.causal_packet import Packet, sha
from kv_compare.mixed_selective import selective_model, selective_plan
from tests.test_mixed_transfer import FakeBaseline, state


class PlanTests(unittest.TestCase):
    def test_sparse_mask_reads_only_two_predecessor_features(self):
        mask = [i in (1, 27) for i in range(28)]
        p = selective_plan(mask)
        self.assertEqual(p['feature_layers'], [0, 26])
        self.assertEqual(p['surprise_layers'], [])
        self.assertEqual(p['prior_reset_layers'], [1, 27])
        self.assertEqual((p['feature_passes'], p['model_product_passes']), (2, 0))
        self.assertEqual((p['full_model_feature_passes'], p['full_model_product_passes']), (27, 26))

    def test_runs_anchor_and_all_true_plans_cover_every_required_read(self):
        p = selective_plan([True, True, False, True, True, True])
        self.assertEqual(p['feature_layers'], [0, 2, 3, 4])
        self.assertEqual(p['surprise_layers'], [0, 3, 4])
        self.assertEqual(p['model_product_layers'], [3, 4])
        self.assertEqual(p['prior_reset_layers'], [0, 3])
        p = selective_plan([True] * 28)
        self.assertEqual(p['feature_layers'], list(range(27)))
        self.assertEqual(p['surprise_layers'], list(range(27)))
        self.assertEqual(p['model_product_layers'], list(range(1, 27)))
        self.assertEqual(selective_plan([False] * 28)['feature_passes'], 0)
        self.assertEqual(selective_plan([True])['surprise_passes'], 0)
        for mask in ([], [1, 0], [None], 'bits'):
            with self.assertRaises(ValueError): selective_plan(mask)


def make_cpu_codec(root, mask):
    path, lut = state(root, layers=len(mask), tokens=256, rank=True)
    codec = mt.MixedTransfer(path, 'unused', lut, sha(lut), mask, 'cpu')
    # The interpreter uses the original kernel-requant path; actual CUDA smoke
    # below retains the frozen epilogue path and its production geometry.
    codec.native.requant = 'kernel'
    # Nonzero products make feature/context mistakes observable in the packet,
    # rather than relying only on the zero-predictor semantic fixture.
    identity = torch.eye(256, dtype=torch.int8)
    codec.rank['Bq'].copy_(identity.expand(len(mask), -1, -1))
    codec.rank['Uq'].copy_(identity.expand(len(mask), -1, -1))
    codec.rank['m1'].fill_(0.25)
    codec.rank['m0'].fill_(0.125)
    return codec


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_needed_context_and_effective_priors_match_full_model(self):
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(mt, 'verify_splitzip_source', return_value={}), mock.patch.object(mt, 'Baseline', FakeBaseline):
                codec = make_cpu_codec(raw, [False, True, True])
                words = torch.randn(3, 256, 256, generator=torch.Generator().manual_seed(812)).bfloat16().view(torch.int16)
                full_x, full_prior = codec.native.model(words)
                for mask in ([False, True, False], [False, True, True], [True, True, True], [True, False, True]):
                    codec.native.ours = list(mask)
                    plan = selective_plan(mask)
                    selected_x, selected_prior = selective_model(codec.native, words)
                    for layer in plan['feature_layers']:
                        self.assertTrue(torch.equal(selected_x[layer], full_x[layer]))
                    for layer in plan['prior_layers']:
                        self.assertTrue(torch.equal(selected_prior[layer], full_prior[layer]))
                    for layer in plan['prior_reset_layers']:
                        self.assertTrue(torch.equal(selected_prior[layer], torch.ones_like(selected_prior[layer])))
                with self.assertRaises(ValueError): selective_model(codec.native, words[:, :, ::2])
                rank = codec.native.rank; codec.native.rank = None
                with self.assertRaises(ValueError): selective_model(codec.native, words)
                codec.native.rank = rank

    def test_complete_packet_bytes_and_actual_decode_match_unchanged_encoder(self):
        for mask in ([False, True, False], [False, True, True]):
            with self.subTest(mask=mask), tempfile.TemporaryDirectory() as raw:
                with mock.patch.object(mt, 'verify_splitzip_source', return_value={}), mock.patch.object(mt, 'Baseline', FakeBaseline):
                    codec = make_cpu_codec(raw, mask)
                    words = torch.ones(3, 256, 256, dtype=torch.bfloat16).view(torch.int16)
                    words[1, 0, :5] = torch.tensor([0, -32768, 1, 127, 128], dtype=torch.int16)
                    expected = codec.encode(words).export()
                    codec.native.model = functools.partial(selective_model, codec.native)
                    packet = codec.encode(words)
                    self.assertEqual(packet.export(), expected)
                    self.assertTrue(torch.equal(codec.decode(Packet.restore(expected, 'cpu')), words))
                    self.assertEqual(packet.export(), expected)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_frozen_rank_wire_parity(self):
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/combined_transfer_a4500_20261029_080157_2956660'
        q = json.loads((report / 'mixed_rank256_k14_t512/request.json').read_text())['codec']
        selection = json.loads((report / 'policy.json').read_text())
        self.assertEqual(sha(Path(q['state']) / 'manifest.json'), q['manifest_sha256'])
        words = torch.ones(28, 512, 2048, dtype=torch.bfloat16, device='cuda:0').view(torch.int16)
        patterns = [0, -32768, 1, 127, 128, 0x3f80, -16512, 0x7f80, -128, 0x7fc1, -1]
        words[1, 0, :len(patterns)] = torch.tensor(patterns, dtype=torch.int16, device='cuda:0')
        original = words.clone()
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for mask in (selection['masks']['2'], selection['masks']['14'], [True] * 28):
            codec = mt.MixedTransfer(q['state'], q['splitzip_source'], q['splitzip_state'],
                q['splitzip_state_sha256'], mask, 'cuda:0')
            self.assertEqual(codec.predictor_state_sha256, q['predictor_state_sha256'])
            original_model = codec.native.model
            expected = codec.encode(words).export()
            codec.native.model = functools.partial(selective_model, codec.native)
            actual = codec.encode(words)
            self.assertEqual(actual.export(), expected)
            held = codec.decode(Packet.restore(expected, 'cuda:0'))
            self.assertTrue(torch.equal(held, original))
            self.assertTrue(torch.equal(words, original))
            codec.native.model = original_model
            self.assertTrue(torch.equal(codec.decode(actual), original))
            self.assertTrue(torch.equal(held, original))
            self.assertEqual(actual.export(), expected)


if __name__ == '__main__': unittest.main()
