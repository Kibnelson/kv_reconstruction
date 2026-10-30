"""Frozen-wire rolling encoder, product reuse and original decoder checks."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare.standalone_rolling_codec import RollingEncoder, StandaloneRollingCodec
from kv_compare.team_lookup import TeamLookupCodec
from kv_compare.team_packet import TeamPacket, save_state
from kv_compare.team_onestep import gpu_coder
from tests.test_team_packet import fixed_state, packet_fixture


def fixture(layers=4, tokens=64):
    generator = torch.Generator().manual_seed(814)
    consts = fixed_state(layers=layers, tokens=tokens)
    consts['Aq'][1:].copy_(torch.randint(-3, 4, (layers - 1, 256, 256),
        generator=generator, dtype=torch.int8))
    consts['sa'].fill_(0.003)
    consts['b'].copy_(torch.randn(layers, 256, generator=generator) * 0.1)
    angle = torch.arange(tokens, dtype=torch.float32)[:, None] * torch.arange(1, 65)[None, :] * 0.0003
    consts['cos'], consts['sin'] = angle.cos(), angle.sin()
    rank = dict(rank=256,
        Bq=torch.randint(-3, 4, (layers, 256, 256), generator=generator, dtype=torch.int8),
        Uq=torch.randint(-3, 4, (layers, 256, 256), generator=generator, dtype=torch.int8),
        m1=torch.full((layers, 256), 0.003), m0=torch.full((layers, 256), 0.5),
        sa=consts['sa'].clone(), b=consts['b'].clone(), sp=consts['sp'].clone(), inv=consts['inv'].clone())
    original = torch.randn(layers, tokens, 256, generator=generator).bfloat16().view(torch.int16)
    original[-1, 0, :11] = torch.tensor([0, -32768, 1, 127, 128, 0x3f80, -16512,
        0x7f80, -128, 0x7fc1, -1], dtype=torch.int16)
    return consts, rank, original


class ContractTests(unittest.TestCase):
    def test_original_decoder_identity_memory_accounting_and_invalid_inputs(self):
        with self.assertRaises(ValueError):
            StandaloneRollingCodec('missing', 'cpu', 'dense_exact')
        with tempfile.TemporaryDirectory() as raw:
            save_state(raw, fixed_state(tokens=256))
            reference = TeamLookupCodec(raw, 'cpu', 'dense_uniform', decoder_mode='direct')
            codec = StandaloneRollingCodec(raw, 'cpu', 'dense_uniform')
            header = codec.header((2, 256, 256))
            self.assertEqual(header, reference.header((2, 256, 256)))
            self.assertIs(codec.native.decode.__func__, reference.native.decode.__func__)
            self.assertIs(codec.native.model.__func__, reference.native.model.__func__)
            self.assertIs(codec.native.encode.__func__, reference.native.encode.__func__)
            self.assertEqual(vars(codec.rolling_encoder), {'native': codec.native})
            packet, _, _ = packet_fixture(header)
            info = codec.accounting(packet)
            self.assertEqual(info['additional_shared_device_bytes'], 0)
            self.assertEqual(info['encoder_accumulator_buffers'], 2)
            self.assertEqual(info['encoder_accumulator_bytes'], 2 * 256 * 256 * 4)
            self.assertEqual(info['encoder_prediction_products_per_call'], 1)
            self.assertEqual(info['coder_table_bytes'], reference.coder_table_bytes)
            self.assertIsNone(info['fallback_codec'])
            words = torch.zeros((2, 256, 256), dtype=torch.int16)
            for invalid in (words.float(), words.transpose(1, 2), words[:, :63]):
                with self.assertRaises(ValueError):
                    codec.encode(invalid)
            for start in (1, True):
                with self.assertRaises(ValueError):
                    codec.encode(words, start=start)
            with self.assertRaises(RuntimeError):
                codec.encode_prepared(words, None, None)
            for options in (dict(out=torch.empty(1, dtype=torch.int16)),
                            dict(layer_words=torch.empty(2, dtype=torch.int32))):
                with self.assertRaises(ValueError):
                    codec.rolling_encoder.encode_native(words, **options)
            S, cap = codec.native.layout(256)
            storage = torch.empty(S * cap, dtype=torch.int16)
            overlapping = storage[:words.numel()].view_as(words)
            with self.assertRaises(ValueError):
                codec.rolling_encoder.encode_native(overlapping, out=storage)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def check_wire(self, arm, layers, tokens):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, original = fixture(layers, tokens)
            save_state(raw, consts, rank)
            reference = TeamLookupCodec(raw, 'cpu', arm, decoder_mode='direct')
            expected = reference.encode(original).export()
            codec = StandaloneRollingCodec(raw, 'cpu', arm)
            with mock.patch.object(codec.native, '_product', wraps=codec.native._product) as products:
                with mock.patch.object(codec.native, 'model', side_effect=AssertionError('forward model called')):
                    with mock.patch.object(gpu_coder, 'slice_tables', side_effect=AssertionError('warm tables rebuilt')):
                        actual = codec.encode(original)
            self.assertEqual(products.call_count, max(0, layers - 1))
            self.assertEqual(actual.export(), expected)
            restored = codec.decode(actual)
            self.assertTrue(torch.equal(restored, original))
            self.assertEqual(actual.export(), expected)
            if arm != 'dense_uniform':
                return
            # Optional native output and per-layer stream counts retain their
            # original interface, including layer-zero alignment/states.
            old_counts = torch.empty(layers, dtype=torch.int64)
            native_arena, old_starts = reference.native.encode(original, layer_words=old_counts)
            new_counts = torch.empty_like(old_counts)
            arena = torch.empty_like(native_arena)
            returned, starts = codec.rolling_encoder.encode_native(original, out=arena, layer_words=new_counts)
            self.assertIs(returned, arena)
            self.assertTrue(torch.equal(starts, old_starts))
            self.assertTrue(torch.equal(new_counts, old_counts))
            # Unread alignment padding and unused arena regions are unspecified.
            # Framing canonicalizes padding; compare states and read data here.
            S, cap = codec.native.layout(tokens)
            for segment in range(S):
                base = int(((starts[segment].to(torch.int64) - 256) // 8) * 8)
                end = (segment + 1) * cap
                start = int(starts[segment])
                self.assertTrue(torch.equal(returned[base:base + 256], native_arena[base:base + 256]))
                self.assertTrue(torch.equal(returned[start:end], native_arena[start:end]))

    def test_dense_four_layers_complete_wire_and_layer_counts(self):
        self.check_wire('dense_uniform', 4, 64)

    def test_rank_three_layers_complete_wire_and_product_reuse(self):
        self.check_wire('rank256_uniform', 3, 256)

    def test_single_layer_anchor_with_no_prediction_product(self):
        self.check_wire('dense_uniform', 1, 64)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_nonzero_predictors_all_bf16_patterns_and_unchanged_wire(self):
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for arm in ('dense_uniform', 'rank256_uniform'):
            with tempfile.TemporaryDirectory() as raw:
                consts, rank, _ = fixture(3, 256)
                save_state(raw, consts, rank)
                original = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(1, 256, 256)
                original = torch.cat((original, original.roll(77, 2), original.roll(41, 1)), 0).to('cuda:0')
                reference = TeamLookupCodec(raw, 'cuda:0', arm, decoder_mode='direct')
                codec = StandaloneRollingCodec(raw, 'cuda:0', arm)
                wire = reference.encode(original).export()
                with mock.patch.object(codec.native, '_product', wraps=codec.native._product) as products:
                    with mock.patch.object(codec.native, 'model', side_effect=AssertionError('forward model called')):
                        with mock.patch.object(gpu_coder, 'slice_tables', side_effect=AssertionError('warm preparation rebuilt')):
                            packet = codec.encode(original)
                self.assertEqual(products.call_count, 2)
                self.assertEqual(packet.export(), wire)
                self.assertTrue(torch.equal(codec.decode(packet), original))
                self.assertEqual(packet.export(), wire)

    def test_actual_gpu_frozen_dense_rank_wire_graph_direct_and_output_lifetimes(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/standalone_compute_a4500_20261029_151844_3927903'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for method in ('team_dense_uniform', 'team_rank256_uniform'):
            for tokens in (512, 8192):
                folder = report / f'{method}_t{tokens}'
                spec = json.loads((folder / 'request.json').read_text())['codec']
                rows = json.loads((folder / 'worker_results.json').read_text())['rows']
                wires = [Path(row['packet']).read_bytes() for row in rows]
                for wire, row in zip(wires, rows):
                    self.assertEqual(len(wire), row['packet_bytes'])
                    self.assertEqual(hashlib.sha256(wire).hexdigest(), row['packet_sha256'])
                mode = 'graph' if tokens == 512 else 'direct'
                codec = StandaloneRollingCodec(spec['state'], 'cuda:0', spec['arm'], decoder_mode=mode)
                packets = [TeamPacket.import_bytes(wire, 'cuda:0') for wire in wires]
                largest = max(packets, key=lambda p: p.payload.numel())
                codec.prepare(largest)
                if mode == 'graph':
                    codec.reserve_payload_words(largest.payload.numel())
                    codec.graph_prepare(largest)
                for packet, wire, row in zip(packets, wires, rows):
                    original = codec.decode(packet)
                    check_original(original, row)
                    with mock.patch.object(codec.native, '_product', wraps=codec.native._product) as products:
                        with mock.patch.object(codec.native, 'model', side_effect=AssertionError('forward model called')):
                            actual = codec.encode(original)
                    self.assertEqual(products.call_count, 27)
                    self.assertEqual(actual.export(), wire)
                    del original, actual
                immutable = packets[0]
                producer = torch.cuda.current_stream()
                callers = [torch.cuda.Stream(), torch.cuda.Stream()]
                for stream in callers:
                    stream.wait_stream(producer)
                held = []
                for turn, index in enumerate((0, 1, 0)):
                    with torch.cuda.stream(callers[turn % 2]):
                        value = codec.decode(packets[index])
                    held.append((value, rows[index]))
                del packets, largest
                gc.collect(); torch.cuda.synchronize()
                for value, row in held:
                    check_original(value, row)
                self.assertEqual(len({value.data_ptr() for value, _ in held}), 3)
                self.assertEqual(immutable.export(), wires[0])
                self.assertEqual(codec.graph_capture_count, int(mode == 'graph'))
                self.assertEqual(codec.accounting(immutable)['additional_shared_device_bytes'], 0)
                del codec, held, immutable, value
                gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
