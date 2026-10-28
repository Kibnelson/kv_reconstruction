"""Fixed-inv spread calibration and actual independent-head packet composition."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch

from kv_compare.team_probability import (FACTORS, assert_spread_only,
    choose_head_factors, head_word_costs, scale_state)
from kv_compare.team_packet import TeamCodec, save_state
from kv_compare.team_lookup import TeamLookupCodec
from tests.test_team_packet import fixed_state, packet_fixture


def rank_state(consts):
    generator = torch.Generator().manual_seed(921)
    return dict(rank=256,
        Bq=torch.randint(-2, 3, (2, 256, 256), generator=generator, dtype=torch.int8),
        Uq=torch.randint(-2, 3, (2, 256, 256), generator=generator, dtype=torch.int8),
        m1=torch.full((2, 256), 0.01), m0=torch.full((2, 256), 0.5),
        sa=torch.full((2, 256), 0.01), b=consts['b'].clone(),
        sp=consts['sp'].clone(), inv=consts['inv'].clone())


class ProbabilityTests(unittest.TestCase):
    def test_scale_clones_state_and_preserves_actual_predictors_and_inverse(self):
        consts = fixed_state()
        rank = rank_state(consts)
        old_const, old_rank = copy.deepcopy(consts), copy.deepcopy(rank)
        candidate, reduced = scale_state(consts, rank, [0.5, 1.5], [1.25, 0.75])
        self.assertTrue(assert_spread_only(consts, rank, candidate, reduced))
        for actual, expected in ((consts, old_const), (rank, old_rank)):
            for name, value in expected.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(actual[name].view(torch.uint8), value.view(torch.uint8)), name)
        self.assertTrue(torch.equal(candidate['sp'][:, :128], consts['sp'][:, :128] * 0.5))
        self.assertTrue(torch.equal(candidate['sp'][:, 128:], consts['sp'][:, 128:] * 1.5))
        self.assertTrue(torch.equal(reduced['sp'][:, :128], rank['sp'][:, :128] * 1.25))
        self.assertTrue(torch.equal(reduced['sp'][:, 128:], rank['sp'][:, 128:] * 0.75))
        for state, source in ((candidate, consts), (reduced, rank)):
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    self.assertNotEqual(value.data_ptr(), source[name].data_ptr(), name)
        unmodified, no_rank = scale_state(consts, None, [1.0, 1.0], None)
        self.assertIsNone(no_rank)
        self.assertTrue(all(torch.equal(v.view(torch.uint8), unmodified[k].view(torch.uint8)) for k, v in consts.items()))

    def test_rejects_invalid_factors_geometry_or_overflow(self):
        consts = fixed_state()
        for factors in ([0, 1], [-1, 1], [float('nan'), 1], [float('inf'), 1], [True, 1], [1], [1, 1, 1], None):
            with self.subTest(factors=factors), self.assertRaises(ValueError):
                scale_state(consts, None, factors, None)
        for hd in (64, True, 0, 127):
            with self.assertRaises(ValueError):
                scale_state(consts, None, [1, 1], None, head_dim=hd)
        with self.assertRaises(ValueError):
            scale_state(consts, None, [1, 1], [1, 1])
        with self.assertRaises(ValueError):
            scale_state(consts, rank_state(consts), [1, 1], None)
        consts['sp'][0, 0] = torch.finfo(torch.float32).max
        with self.assertRaises(ValueError):
            scale_state(consts, None, [2, 1], None)

    def test_frozen_proof_detects_every_other_change_including_signed_zero(self):
        consts = fixed_state()
        for name in consts:
            if name == 'sp':
                continue
            candidate, _ = scale_state(consts, None, [0.5, 1.5], None)
            value = candidate[name].reshape(-1)
            value[0] = value[0] + 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                assert_spread_only(consts, None, candidate, None)
        candidate, _ = scale_state(consts, None, [1, 1], None)
        candidate['b'][0, 0] = -0.0
        with self.assertRaises(ValueError):
            assert_spread_only(consts, None, candidate, None)
        candidate, _ = scale_state(consts, None, [1, 1], None)
        candidate.pop('sin')
        with self.assertRaises(ValueError):
            assert_spread_only(consts, None, candidate, None)

    def test_head_costs_use_actual_segments_and_charge_states_and_padding(self):
        packet, _arena, _starts = packet_fixture(gap=3)
        actual = head_word_costs(packet)
        lengths = packet.offsets[1:] - packet.offsets[:-1]
        self.assertEqual(actual, [int(lengths[::2].sum()), int(lengths[1::2].sum())])
        self.assertEqual(sum(actual) * 2, packet.payload_bytes)
        self.assertGreater(packet.packet_bytes, sum(actual) * 2)
        packet.offsets[1] += 1
        with self.assertRaises(ValueError):
            head_word_costs(packet)

    def test_actual_cost_selection_is_independent_and_ties_are_deterministic(self):
        observations = [dict(factor=f, head_words=[100, 200, 300]) for f in FACTORS]
        observations[0]['head_words'][0] = 80
        observations[1]['head_words'][1] = 190
        observations[3]['head_words'][1] = 190
        selected = choose_head_factors(list(reversed(observations)))
        self.assertEqual(selected['factors'], [0.5, 0.75, 1.0])
        self.assertEqual(selected['head_words'], [80, 190, 300])
        self.assertEqual(selected['total_words'], 570)
        self.assertEqual(selected['baseline_words'], 600)
        self.assertEqual(selected['factor_grid'], list(FACTORS))
        self.assertEqual(choose_head_factors([dict(factor=f, head_words=[1]) for f in FACTORS])['factors'], [1.0])

    def test_selection_rejects_missing_duplicate_or_noninteger_costs(self):
        valid = [dict(factor=f, head_words=[1, 2]) for f in FACTORS]
        cases = [valid[:-1], valid + [valid[0]], valid + [dict(factor=2, head_words=[1, 2])],
                 [dict(factor=f, head_words=[0, 1]) for f in FACTORS],
                 [dict(factor=f, head_words=[True, 1]) for f in FACTORS],
                 [dict(factor=f, head_words=[1.0, 2]) for f in FACTORS]]
        bad_shape = copy.deepcopy(valid); bad_shape[0]['head_words'] = [1]
        cases.append(bad_shape)
        for observations in cases:
            with self.subTest(observations=observations), self.assertRaises(ValueError):
                choose_head_factors(observations)
        with self.assertRaises(ValueError):
            choose_head_factors(valid, factors=(0.5, 0.75))


def check_real_composition(test, device, with_rank):
    tokens = 256 if with_rank else 64
    consts = fixed_state(layers=2, tokens=tokens)
    generator = torch.Generator().manual_seed(401)
    consts['Aq'][1] = torch.randint(-2, 3, (256, 256), generator=generator, dtype=torch.int8)
    consts['sa'].fill_(0.01)
    consts['b'] = torch.randn(2, 256, generator=generator) * 0.03
    angle = torch.arange(tokens).float()[:, None] * 0.01 + torch.arange(64).float()[None, :] * 0.003
    consts['cos'], consts['sin'] = angle.cos(), angle.sin()
    rank = rank_state(consts) if with_rank else None
    words = torch.randn(2, tokens, 256, generator=generator).bfloat16().view(torch.int16).to(device)
    initial = words.clone()
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / 'base'; save_state(base, consts, rank)
        arms = ('dense_uniform', 'rank256_uniform') if with_rank else ('dense_uniform',)
        for arm in arms:
            frozen = TeamCodec(base, device, arm)
            _features, prior = frozen.model(words)
            frozen_packet = frozen.encode(words)
            samples = {}
            for name, factors in (('lo', [0.5, 0.5]), ('hi', [1.5, 1.5]), ('mixed', [0.5, 1.5])):
                candidate, reduced = scale_state(consts, rank, factors, factors if rank is not None else None)
                path = Path(td) / (arm + '_' + name)
                save_state(path, candidate, reduced, metadata={'purpose': 'software calibration proof', 'factors': factors})
                encoder = TeamCodec(path, device, arm)
                features, changed_prior = encoder.model(words)
                test.assertTrue(torch.equal(changed_prior.view(torch.int32), prior.view(torch.int32)))
                packet = encoder.encode_prepared(words, features, changed_prior)
                wire = packet.export()
                decoder = TeamLookupCodec(path, device, arm, decoder_mode='direct', lookup=False)
                output = decoder.decode(packet)
                test.assertTrue(torch.equal(output, words))
                test.assertEqual(packet.export(), wire)
                test.assertTrue(torch.equal(words, initial))
                with test.assertRaises(ValueError):
                    decoder.decode(frozen_packet)
                samples[name] = packet
            mixed = samples['mixed']
            for segment in range(mixed.header['segments']):
                selected = samples['lo' if segment % 2 == 0 else 'hi']
                a, b = (int(v) for v in mixed.offsets[segment:segment + 2])
                c, d = (int(v) for v in selected.offsets[segment:segment + 2])
                test.assertEqual(b - a, d - c)
                test.assertTrue(torch.equal(mixed.payload[a:b], selected.payload[c:d]))
                test.assertEqual(int(mixed.relative_starts[segment]), int(selected.relative_starts[segment]))


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU Triton interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_fixed_inverse_preserves_priors_and_exact_head_stream_composition(self):
        check_real_composition(self, 'cpu', with_rank=False)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_cuda_dense_rank_fixed_priors_exact_words_and_head_composition(self):
        check_real_composition(self, 'cuda:0', with_rank=True)


if __name__ == '__main__':
    unittest.main()
