"""Frozen-wire feature-storage and decoder-partition checks."""
import ast
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from kv_compare.standalone_rolling_codec import RollingEncoder, StandaloneRollingCodec
from kv_compare.standalone_schedule_codec import FeatureReuseRollingEncoder, StandaloneScheduleCodec
from kv_compare.team_packet import TeamPacket, save_state
from tests.test_standalone_rolling_codec import fixture
from tests.test_team_packet import packet_fixture


class ContractTests(unittest.TestCase):
    def test_borrowed_payload_is_recorded_on_caller_even_without_child_streams(self):
        codec = StandaloneScheduleCodec.__new__(StandaloneScheduleCodec)
        codec.device = SimpleNamespace(type='cuda')
        packet = SimpleNamespace(payload=mock.Mock())
        caller, result = object(), object()
        with mock.patch.object(StandaloneRollingCodec, 'decode', return_value=result) as decode:
            with mock.patch.object(torch.cuda, 'current_stream', return_value=caller) as current:
                self.assertIs(codec.decode(packet), result)
        decode.assert_called_once_with(packet)
        current.assert_called_once_with(codec.device)
        packet.payload.record_stream.assert_called_once_with(caller)

    def test_independent_options_preserve_sender_identity_and_charge_workspace(self):
        for options in (dict(feature_reuse=1), dict(decoder_chunks=True), dict(decoder_chunks=2),
                        dict(decoder_chunks=4.0), dict(feature_reuse=True, decoder_chunks=1)):
            with self.assertRaises(ValueError):
                StandaloneScheduleCodec('missing', 'cpu', 'dense_uniform', **options)
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, _ = fixture(3, 256)
            save_state(raw, consts, rank)
            reference = StandaloneRollingCodec(raw, 'cpu', 'dense_uniform')
            for options in (dict(feature_reuse=True), dict(decoder_chunks=1),
                            dict(decoder_chunks=4), dict(decoder_chunks=8)):
                codec = StandaloneScheduleCodec(raw, 'cpu', 'dense_uniform', **options)
                head = codec.header((3, 256, 256))
                self.assertEqual(head, reference.header((3, 256, 256)))
                self.assertEqual(codec.native.chunks, 2)
                self.assertEqual(head['chunks'], 2)
                packet, _, _ = packet_fixture(head)
                codec.prepare(packet)
                info = codec.accounting(packet)
                effective = options.get('decoder_chunks', 2)
                self.assertEqual(info['effective_decoder_chunks'], effective)
                self.assertEqual(len(codec.buffers['acc']), effective)
                self.assertEqual([tuple(x.shape) for x in codec.buffers['acc']], [(256 // effective, 256)] * effective)
                self.assertEqual(sum(x.numel() * x.element_size() for x in codec.buffers['acc']), 256 * 256 * 4)
                self.assertEqual(info['decoder_accumulator_bytes'], 256 * 256 * 4)
                self.assertEqual(info['additional_shared_device_bytes'], 0)
                self.assertEqual(info['coder_table_bytes'], reference.coder_table_bytes)
                self.assertEqual(info['unchanged_encoder_header_chunks'], 2)
                self.assertIsNone(info['fallback_codec'])
                if options.get('feature_reuse'):
                    self.assertEqual(info['encoder_feature_buffer_count'], 1)
                    self.assertEqual(info['encoder_feature_bytes'], 256 * 256)
                    self.assertIsNone(codec.decoder_native)
                else:
                    self.assertIs(codec.native.decode.__self__, codec.decoder_native)
                    self.assertIs(codec.native.buffers.__self__, codec.decoder_native)
                    self.assertIs(codec.decoder_native.k, codec.native.k)
                    self.assertIs(codec.decoder_native.rank, codec.native.rank)
                    self.assertIs(codec.decoder_native.tables, codec.native.tables)
                    self.assertIs(codec.rolling_encoder.native, codec.native)
                    self.assertEqual(info['encoder_feature_buffer_count'], 2)

    def test_bad_token_block_and_rank_tile_partitions_fail_before_decode(self):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, _ = fixture(3, 256)
            save_state(raw, consts, rank)
            for arm, tokens, chunks in (('dense_uniform', 128, 8), ('rank256_uniform', 256, 4)):
                codec = StandaloneScheduleCodec(raw, 'cpu', arm, decoder_chunks=chunks)
                packet, _, _ = packet_fixture(codec.header((3, tokens, 256)))
                with mock.patch.object(codec.native, 'decode') as decode:
                    with self.assertRaises(ValueError):
                        codec.decode(packet)
                    decode.assert_not_called()
                self.assertIsNone(codec.shape)

    def test_feature_reuse_changes_only_allocation_and_feature_launch_location(self):
        original = textwrap.dedent(inspect.getsource(RollingEncoder.encode_native))
        old = '''    # No feature of the last layer is consumed by any predictor.
    features = torch.empty((max(0, L - 1), T, W), dtype=torch.int8, device=dev)
    for layer in range(L - 1):
        native._launch(lw._lfeat_kernel, (lw.triton.cdiv(T * W, 1024),),
            words[layer], features[layer], k['isx'][layer], k['cos'], k['sin'],
            0, T * W, W, native.C, HD=native.hd, BLOCK=1024, warps=4)
'''
        self.assertEqual(original.count(old), 1)
        expected = original.replace(old,
            '    features = torch.empty((T, W), dtype=torch.int8, device=dev) if L > 1 else None\n')
        old_product = '            native._product(features[layer - 1], layer, destination, scratch)'
        new_product = '''            previous_layer = layer - 1
            native._launch(lw._lfeat_kernel, (lw.triton.cdiv(T * W, 1024),),
                words[previous_layer], features, k['isx'][previous_layer], k['cos'], k['sin'],
                0, T * W, W, native.C, HD=native.hd, BLOCK=1024, warps=4)
            native._product(features, layer, destination, scratch)'''
        self.assertEqual(expected.count(old_product), 1)
        expected = expected.replace(old_product, new_product)
        actual = textwrap.dedent(inspect.getsource(FeatureReuseRollingEncoder.encode_native))
        self.assertEqual(ast.dump(ast.parse(actual), include_attributes=False),
                         ast.dump(ast.parse(expected), include_attributes=False))


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_feature_reuse_nonzero_dense_complete_wire_and_single_storage(self):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, original = fixture(3, 64)
            save_state(raw, consts, rank)
            reference = StandaloneRollingCodec(raw, 'cpu', 'dense_uniform')
            expected = reference.encode(original).export()
            codec = StandaloneScheduleCodec(raw, 'cpu', 'dense_uniform', feature_reuse=True)
            product = codec.native._product
            features = []
            def record(x, *args, **kwargs):
                features.append(x.data_ptr())
                return product(x, *args, **kwargs)
            with mock.patch.object(codec.native, '_product', side_effect=record):
                packet = codec.encode(original)
            self.assertEqual(len(features), 2)
            self.assertEqual(len(set(features)), 1)
            self.assertEqual(packet.export(), expected)
            self.assertTrue(torch.equal(codec.decode(packet), original))
            self.assertEqual(packet.export(), expected)
            self.assertEqual(vars(codec.rolling_encoder), {'native': codec.native})

    def test_dense_chunk_one_and_four_reconstruct_identical_original_packet(self):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, original = fixture(2, 128)
            save_state(raw, consts, rank)
            reference = StandaloneRollingCodec(raw, 'cpu', 'dense_uniform')
            wire = reference.encode(original).export()
            for chunks in (1, 4):
                codec = StandaloneScheduleCodec(raw, 'cpu', 'dense_uniform', decoder_chunks=chunks)
                packet = TeamPacket.import_bytes(wire, 'cpu')
                with mock.patch.object(codec.decoder_native, '_product', wraps=codec.decoder_native._product) as product:
                    result = codec.decode(packet)
                self.assertEqual(product.call_count, chunks)
                self.assertTrue(torch.equal(result, original))
                self.assertEqual(packet.export(), wire)
                self.assertEqual(codec.native.chunks, 2)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_feature_reuse_all_bf16_patterns_and_original_wire(self):
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for arm in ('dense_uniform', 'rank256_uniform'):
            with tempfile.TemporaryDirectory() as raw:
                consts, rank, _ = fixture(3, 256)
                save_state(raw, consts, rank)
                words = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(1, 256, 256)
                words = torch.cat((words, words.roll(77, 2), words.roll(41, 1)), 0).to('cuda:0')
                reference = StandaloneRollingCodec(raw, 'cuda:0', arm)
                expected = reference.encode(words).export()
                codec = StandaloneScheduleCodec(raw, 'cuda:0', arm, feature_reuse=True)
                product = codec.native._product
                pointers = []
                def record(x, *args, **kwargs):
                    pointers.append(x.data_ptr())
                    return product(x, *args, **kwargs)
                with mock.patch.object(codec.native, '_product', side_effect=record):
                    packet = codec.encode(words)
                self.assertEqual(len(pointers), 2)
                self.assertEqual(len(set(pointers)), 1)
                self.assertEqual(packet.export(), expected)
                self.assertTrue(torch.equal(codec.decode(packet), words))
                self.assertEqual(packet.export(), expected)

    def test_actual_gpu_frozen_wire_partitions_and_cross_stream_input_lifetimes(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/standalone_rolling_a4500_20261029_191839_664765'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for arm in ('dense_uniform', 'rank256_uniform'):
            for tokens in (512, 8192):
                folder = report / f'rolling_{arm}_t{tokens}'
                spec = json.loads((folder / 'request.json').read_text())['codec']
                rows = json.loads((folder / 'worker_results.json').read_text())['rows']
                wires = [Path(row['packet']).read_bytes() for row in rows]
                for wire, row in zip(wires, rows):
                    self.assertEqual(len(wire), row['packet_bytes'])
                    self.assertEqual(hashlib.sha256(wire).hexdigest(), row['packet_sha256'])
                options = [dict(feature_reuse=True)]
                if tokens == 8192:
                    options += [dict(decoder_chunks=chunks) for chunks in (1, 4, 8)]
                for flags in options:
                    mode = 'graph' if tokens == 512 else 'direct'
                    codec = StandaloneScheduleCodec(spec['state'], 'cuda:0', arm, decoder_mode=mode, **flags)
                    packets = [TeamPacket.import_bytes(wire, 'cuda:0') for wire in wires]
                    largest = max(packets, key=lambda p: p.payload.numel())
                    codec.prepare(largest)
                    if mode == 'graph':
                        codec.reserve_payload_words(largest.payload.numel())
                        codec.graph_prepare(largest)
                    for packet, wire, row in zip(packets, wires, rows):
                        original = codec.decode(packet)
                        check_original(original, row)
                        actual = codec.encode(original)
                        self.assertEqual(actual.export(), wire)
                        del original, actual
                    # Each packet comes from the producer stream. In the
                    # one-chunk arm its payload is consumed on the caller,
                    # without any native child stream to record that use.
                    immutable = packets[0]
                    payload_sizes = [p.payload.numel() for p in packets]
                    producer = torch.cuda.current_stream()
                    callers = [torch.cuda.Stream(), torch.cuda.Stream()]
                    for stream in callers:
                        stream.wait_stream(producer)
                    held = []
                    for turn, index in enumerate((0, 1, 0)):
                        with torch.cuda.stream(callers[turn % 2]):
                            value = codec.decode(packets[index])
                        held.append((value, rows[index]))
                    del packets, largest, packet
                    gc.collect()
                    # Churn the original allocation stream before synchronizing
                    # consumers; a dropped input must not be reused too early.
                    churn = []
                    with torch.cuda.stream(producer):
                        for _ in range(2):
                            for size in payload_sizes:
                                churn.append(torch.empty(size, dtype=torch.int16, device='cuda:0').fill_(0))
                    torch.cuda.synchronize()
                    for value, row in held:
                        check_original(value, row)
                    self.assertEqual(len({value.data_ptr() for value, _ in held}), 3)
                    self.assertEqual(immutable.export(), wires[0])
                    self.assertEqual(codec.graph_capture_count, int(mode == 'graph'))
                    info = codec.accounting(immutable)
                    self.assertEqual(info['additional_shared_device_bytes'], 0)
                    self.assertEqual(info['unchanged_encoder_header_chunks'], 2)
                    self.assertEqual(info['decoder_accumulator_bytes'], tokens * 2048 * 4)
                    self.assertEqual(len(codec.buffers['acc']), flags.get('decoder_chunks', 2))
                    del codec, held, immutable, value, churn
                    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
