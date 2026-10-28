"""Static partner reuse preserves words, predictions, features and coding state."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare.team_register import (TeamRegisterCodec, _half_swap, _register_lane_mu,
    kernel_provenance_for, register_source_identity, triton, tl)
from kv_compare.team_lookup import TeamLookupCodec
from kv_compare.team_packet import TeamCodec, TeamPacket, save_state
from kv_compare.team_onestep import layerwise as lw
from tests.test_team_packet import fixed_state


def nontrivial_state(tokens=256):
    consts = fixed_state(layers=2, tokens=tokens)
    generator = torch.Generator().manual_seed(337)
    consts['Aq'][1] = torch.randint(-2, 3, (256, 256), generator=generator, dtype=torch.int8)
    consts['sa'].fill_(0.01)
    consts['b'] = torch.randn(2, 256, generator=generator) * 0.03
    angles = torch.arange(tokens).float()[:, None] * 0.01 + torch.arange(64).float()[None, :] * 0.003
    consts['cos'], consts['sin'] = angles.cos(), angles.sin()
    return consts


def decoder_snapshot(codec):
    return {name: ([v.clone() for v in codec.buffers[name]] if isinstance(codec.buffers[name], list)
                   else codec.buffers[name].clone())
            for name in ('xbuf', 'ssum', 'state', 'pos', 'smult', 'acc')}


def assert_snapshot_equal(test, expected, actual):
    for name in expected:
        values = zip(expected[name], actual[name]) if isinstance(expected[name], list) else [(expected[name], actual[name])]
        for a, b in values:
            test.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name)


class RegisterTests(unittest.TestCase):
    def test_provenance_options_and_wire_keep_fixed_state(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            reference = TeamCodec(td, 'cpu', 'dense_uniform')
            for mode in ('graph', 'direct'):
                for register in (False, True):
                    codec = TeamRegisterCodec(td, 'cpu', 'dense_uniform', decoder_mode=mode,
                                              register_partner=register)
                    self.assertEqual(codec.header((2, 64, 256)), reference.header((2, 64, 256)))
                    self.assertEqual(codec.kernel_provenance, kernel_provenance_for(
                        decoder_mode=mode, register_partner=register, decoder_warps=1))
                    self.assertFalse(codec.lookup)
                    self.assertTrue(codec.native.search)
                    self.assertEqual(codec.native.opts, {'enable_fp_fusion': False})
                    self.assertTrue(codec.inverse_proof['native_tables_verified'])
        identity = register_source_identity()
        root = Path(__file__).resolve().parents[1]
        for name, wanted in identity['sources'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), wanted)
        for kw in ({'decoder_mode': 'other', 'register_partner': True},
                   {'decoder_mode': 'graph', 'register_partner': 1},
                   {'decoder_mode': 'graph', 'register_partner': True, 'decoder_warps': True},
                   {'decoder_mode': 'graph', 'register_partner': True, 'decoder_warps': 3}):
            with self.assertRaises(ValueError):
                kernel_provenance_for(**kw)

    def test_launch_substitution_is_instance_local_and_only_for_decoder(self):
        calls = []
        def recorded(native, kernel, grid, *args, warps=1, **kwargs):
            calls.append((native, kernel, grid, args, warps, kwargs))
            return kernel
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state())
            with mock.patch.object(lw.LayerwiseOneStep, '_launch', new=recorded):
                candidate = TeamRegisterCodec(td, 'cpu', 'dense_uniform', register_partner=True, decoder_warps=2)
                reference = TeamRegisterCodec(td, 'cpu', 'dense_uniform', register_partner=False)
                unchanged = TeamCodec(td, 'cpu', 'dense_uniform')
                result = candidate.native._launch(lw._ldec_kernel, (3,), 'input', FIRST=True)
                self.assertEqual(result.__name__, '_register_ldec_kernel')
                self.assertEqual(calls[-1][4], 2)
                self.assertIs(reference.native._launch(lw._ldec_kernel, (1,)), lw._ldec_kernel)
                self.assertEqual(calls[-1][4], 1)
                self.assertIs(unchanged.native._launch(lw._ldec_kernel, (1,)), lw._ldec_kernel)
                product = lw._igemm_rq_kernel
                self.assertIs(candidate.native._launch(product, (2,), 'a', warps=4, BM=128), product)
                self.assertEqual(calls[-1][4:], (4, {'BM': 128}))


if triton is not None:
    @triton.jit
    def _swap_probe(source, output, H2: tl.constexpr):
        i = tl.program_id(0) * (2 * H2) + tl.arange(0, 2 * H2)
        value = tl.load(source + i)
        tl.store(output + i, _half_swap(value, H2))

    @triton.jit
    def _mu_probe(acc, sa, bias, cos, sin, reference, candidate, W: tl.constexpr, H2: tl.constexpr):
        pid = tl.program_id(0)
        row, g = pid // 2, pid % 2
        cc = tl.arange(0, 2 * H2)
        own = g * (2 * H2) + cc
        low = cc < H2
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        sao, bo = tl.load(sa + row * W + own), tl.load(bias + row * W + own)
        sap, bp = tl.load(sa + row * W + part), tl.load(bias + row * W + part)
        cs, sn = tl.load(cos + row * H2 + hh), tl.load(sin + row * H2 + hh)
        old = lw._lane_mu(acc + row * W, own, part, sao, bo, sap, bp, cs, sn, low, g == 0)
        new = _register_lane_mu(acc + row * W, own, sao, bo, cs, sn, low, g == 0, H2)
        tl.store(reference + row * W + own, old)
        tl.store(candidate + row * W + own, new)


def check_mu_bits(test, device, warps=1):
    generator = torch.Generator().manual_seed(1997)
    acc = torch.randint(-(1 << 27), 1 << 27, (8, 256), generator=generator, dtype=torch.int32)
    edges = torch.tensor([-(1 << 25) - 1, -(1 << 25), -1, 0, 1, (1 << 24) - 1,
                          (1 << 24) + 1, (1 << 25) + 1], dtype=torch.int32)
    acc[:, :8] = edges
    scales = torch.randn(8, 256, generator=generator) * 0.03
    bias = torch.randn(8, 256, generator=generator) * 0.1
    angles = torch.arange(8 * 64).float().reshape(8, 64) * 0.013
    acc, scales, bias = (v.to(device) for v in (acc, scales, bias))
    cos, sin = angles.cos().to(device), angles.sin().to(device)
    expected = torch.empty((8, 256), dtype=torch.float32, device=device)
    actual = torch.empty_like(expected)
    _mu_probe[(16,)](acc, scales, bias, cos, sin, expected, actual, W=256, H2=64,
                     num_warps=warps, enable_fp_fusion=False)
    test.assertTrue(torch.equal(expected.view(torch.int32), actual.view(torch.int32)))


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU Triton interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_prediction_float32_bits_with_nontrivial_rope_and_large_accumulators(self):
        check_mu_bits(self, 'cpu')

    def test_half_swap_all_bf16_patterns(self):
        words = torch.arange(65536, dtype=torch.int32)
        result = torch.empty_like(words)
        _swap_probe[(512,)](words, result, H2=64)
        expected = words.reshape(512, 2, 64).flip(1).reshape(-1)
        self.assertTrue(torch.equal(result, expected))

    def test_real_native_uniform_and_exact_words_features_and_states(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, nontrivial_state(tokens=64))
            generator = torch.Generator().manual_seed(347)
            words = torch.randn(2, 64, 256, generator=generator).bfloat16().view(torch.int16)
            for arm in ('dense_uniform', 'dense_exact'):
                with self.subTest(arm=arm):
                    source = TeamCodec(td, 'cpu', arm)
                    packet = source.encode(words)
                    wire = packet.export()
                    reference = TeamRegisterCodec(td, 'cpu', arm, decoder_mode='direct', register_partner=False)
                    candidate = TeamRegisterCodec(td, 'cpu', arm, decoder_mode='direct', register_partner=True)
                    expected = reference.decode(packet)
                    snapshot = decoder_snapshot(reference)
                    actual = candidate.decode(packet)
                    self.assertTrue(torch.equal(expected, words))
                    self.assertTrue(torch.equal(actual, expected))
                    assert_snapshot_equal(self, snapshot, decoder_snapshot(candidate))
                    self.assertEqual(packet.export(), wire)
                    self.assertFalse(hasattr(candidate, 'packed'))
                    self.assertFalse(hasattr(candidate, 'output'))
                    if arm == 'dense_uniform':
                        graph_path = TeamRegisterCodec(td, 'cpu', arm, decoder_mode='graph')
                        self.assertTrue(torch.equal(graph_path.kernel_decode(packet), words))
                        assert_snapshot_equal(self, snapshot, decoder_snapshot(graph_path))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_cuda_modes_words_features_and_owned_outputs(self):
        for warps in (1, 2):
            check_mu_bits(self, 'cuda:0', warps=warps)
        with tempfile.TemporaryDirectory() as td:
            consts = nontrivial_state(tokens=256)
            generator = torch.Generator().manual_seed(772)
            rank = dict(rank=256,
                Bq=torch.randint(-2, 3, (2, 256, 256), generator=generator, dtype=torch.int8),
                Uq=torch.randint(-2, 3, (2, 256, 256), generator=generator, dtype=torch.int8),
                m1=torch.full((2, 256), 0.01), m0=torch.full((2, 256), 0.5),
                sa=torch.full((2, 256), 0.01), b=consts['b'].clone(),
                sp=consts['sp'].clone(), inv=consts['inv'].clone())
            save_state(td, consts, rank)
            words = [torch.randn(2, 256, 256, generator=generator).bfloat16().view(torch.int16).to('cuda:0'),
                     torch.full((2, 256, 256), 5., dtype=torch.bfloat16, device='cuda:0').view(torch.int16)]
            for arm in ('dense_uniform', 'dense_exact', 'rank256_uniform'):
                source = TeamCodec(td, 'cuda:0', arm)
                packets = [source.encode(w) for w in words]
                initial = [p.payload.clone() for p in packets]
                for mode, warps in ((m, w) for m in ('graph', 'direct') for w in (1, 2)):
                    with self.subTest(arm=arm, mode=mode, warps=warps):
                        reference = TeamRegisterCodec(td, 'cuda:0', arm, decoder_mode=mode, register_partner=False, decoder_warps=warps)
                        candidate = TeamRegisterCodec(td, 'cuda:0', arm, decoder_mode=mode, register_partner=True, decoder_warps=warps)
                        largest = max(packets, key=lambda p:p.payload.numel())
                        for codec in (reference, candidate):
                            codec.prepare(largest)
                            if mode == 'graph':
                                codec.reserve_payload_words(largest.payload.numel())
                                codec.graph_prepare(largest)
                        expected = reference.decode(packets[0]); held = candidate.decode(packets[0])
                        self.assertTrue(torch.equal(held, expected))
                        assert_snapshot_equal(self, decoder_snapshot(reference), decoder_snapshot(candidate))
                        for i in (1, 0):
                            expected = reference.decode(packets[i]); actual = candidate.decode(packets[i])
                            self.assertTrue(torch.equal(actual, expected))
                            self.assertTrue(torch.equal(actual, words[i]))
                            self.assertTrue(torch.equal(held, words[0]))
                            self.assertTrue(torch.equal(packets[i].payload, initial[i]))
                            assert_snapshot_equal(self, decoder_snapshot(reference), decoder_snapshot(candidate))
                        self.assertEqual(candidate.graph_capture_count, int(mode == 'graph'))

    def test_actual_cuda_all_bf16_patterns_and_stream_changes(self):
        with tempfile.TemporaryDirectory() as td:
            save_state(td, fixed_state(layers=1, tokens=256))
            source = TeamCodec(td, 'cuda:0', 'dense_uniform')
            words = torch.arange(65536, dtype=torch.int32, device='cuda:0').to(torch.int16).reshape(1, 256, 256)
            wire = source.encode(words).export()
            for mode, warps in ((m, w) for m in ('graph', 'direct') for w in (1, 2)):
                codec = TeamRegisterCodec(td, 'cuda:0', 'dense_uniform', decoder_mode=mode, decoder_warps=warps)
                packet = TeamPacket.import_bytes(wire, 'cuda:0')
                codec.prepare(packet)
                first, second = torch.cuda.Stream(), torch.cuda.Stream()
                with torch.cuda.stream(first):
                    held = codec.decode(packet)
                with torch.cuda.stream(second):
                    actual = codec.decode(packet)
                with torch.cuda.stream(first):
                    again = codec.decode(packet)
                torch.cuda.current_stream().wait_stream(first)
                self.assertTrue(torch.equal(held, words))
                self.assertTrue(torch.equal(actual, words))
                self.assertTrue(torch.equal(again, words))
                self.assertEqual(codec.graph_capture_count, int(mode == 'graph'))


if __name__ == '__main__':
    unittest.main()
