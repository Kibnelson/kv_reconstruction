"""Exact full-packet inverse choices, graph/direct replay and output ownership."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare.standalone_inverse_codec import StandaloneInverseCodec
from kv_compare.standalone_schedule_codec import StandaloneScheduleCodec
from kv_compare.team_packet import TeamPacket, save_state
from kv_compare.team_onestep import gpu_coder
from tests.test_standalone_rolling_codec import fixture
from tests.test_team_packet import packet_fixture


class ContractTests(unittest.TestCase):
    def test_inverse_mode_tables_and_original_sender_identity(self):
        for mode in ('full', 'ref', None, True):
            with self.assertRaises(ValueError):
                StandaloneInverseCodec('missing', 'cpu', 'dense_uniform', inverse_mode=mode)
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, _ = fixture(2, 64)
            save_state(raw, consts, rank)
            reference = StandaloneScheduleCodec(raw, 'cpu', 'dense_uniform', feature_reuse=True)
            for mode in ('masked', 'packed'):
                codec = StandaloneInverseCodec(raw, 'cpu', 'dense_uniform', inverse_mode=mode)
                header = codec.header((2, 64, 256))
                self.assertEqual(header, reference.header((2, 64, 256)))
                self.assertEqual(codec.native.chunks, 2)
                self.assertEqual(codec.native.product, 'cublas')
                self.assertTrue(codec.feature_reuse)
                self.assertIsNone(codec.decoder_chunks)
                self.assertIs(codec.native._launch.__self__, codec)
                packet, _, _ = packet_fixture(header)
                info = codec.accounting(packet)
                extra = 45568 if mode == 'packed' else 0
                self.assertEqual(info['additional_shared_device_bytes'], extra)
                self.assertEqual(info['sparse_inverse_shared_bytes'], extra)
                self.assertEqual(info['coder_table_bytes'], 57688 + extra)
                self.assertEqual(info['original_coder_table_bytes'], 57688)
                self.assertEqual(info['inverse_checked_slots'], 655360)
                self.assertEqual(info['full_inverse_shared_bytes'], 0)
                self.assertIsNone(info['fallback_codec'])
                if mode == 'masked':
                    self.assertIsNone(codec.core_coarse)
                    self.assertIsNone(codec.core_sparse)
                else:
                    self.assertEqual(tuple(codec.core_coarse.shape), (10, 1024))
                    self.assertEqual(codec.core_sparse.numel(), 2304)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_both_inverse_modes_preserve_complete_wire_and_original_words(self):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, original = fixture(2, 64)
            save_state(raw, consts, rank)
            reference = StandaloneScheduleCodec(raw, 'cpu', 'dense_uniform', feature_reuse=True)
            wire = reference.encode(original).export()
            for mode in ('masked', 'packed'):
                codec = StandaloneInverseCodec(raw, 'cpu', 'dense_uniform', inverse_mode=mode)
                with mock.patch.object(gpu_coder, 'slice_tables', side_effect=AssertionError('warm tables rebuilt')):
                    packet = codec.encode(original)
                    self.assertEqual(packet.export(), wire)
                    held = codec.decode(packet)
                    again = codec.decode(packet)
                self.assertTrue(torch.equal(held, original))
                self.assertTrue(torch.equal(again, original))
                self.assertNotEqual(held.data_ptr(), again.data_ptr())
                self.assertEqual(packet.export(), wire)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_frozen_dense_rank_inverse_modes_graph_direct_and_lifetimes(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/standalone_schedule_a4500_20261029_195745_1506118'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for method in ('schedule_feature_dense_uniform', 'schedule_feature_rank256_uniform'):
            for tokens in (512, 8192):
                folder = report / f'{method}_t{tokens}'
                spec = json.loads((folder / 'request.json').read_text())['codec']
                rows = json.loads((folder / 'worker_results.json').read_text())['rows']
                wires = [Path(row['packet']).read_bytes() for row in rows]
                for wire, row in zip(wires, rows):
                    self.assertEqual(len(wire), row['packet_bytes'])
                    self.assertEqual(hashlib.sha256(wire).hexdigest(), row['packet_sha256'])
                for mode in ('masked', 'packed'):
                    execution = 'graph' if tokens == 512 else 'direct'
                    codec = StandaloneInverseCodec(spec['state'], 'cuda:0', spec['arm'],
                        decoder_mode=execution, inverse_mode=mode)
                    packets = [TeamPacket.import_bytes(wire, 'cuda:0') for wire in wires]
                    largest = max(packets, key=lambda p: p.payload.numel())
                    codec.prepare(largest)
                    if execution == 'graph':
                        codec.reserve_payload_words(largest.payload.numel())
                        codec.graph_prepare(largest)
                    for packet, wire, row in zip(packets, wires, rows):
                        original = codec.decode(packet)
                        check_original(original, row)
                        self.assertEqual(codec.encode(original).export(), wire)
                        del original
                    immutable = packets[0]
                    payload_words = packets[1].payload.numel()
                    del packet, largest
                    producer = torch.cuda.current_stream()
                    callers = [torch.cuda.Stream(), torch.cuda.Stream()]
                    for stream in callers:
                        stream.wait_stream(producer)
                    held = []
                    for turn, index in enumerate((0, 1, 0)):
                        with torch.cuda.stream(callers[turn % 2]):
                            value = codec.decode(packets[index])
                        held.append((value, rows[index]))
                    del packets
                    gc.collect()
                    with torch.cuda.stream(producer):
                        churn = torch.empty(payload_words, dtype=torch.int16, device='cuda:0')
                        churn.fill_(0x1234)
                    torch.cuda.synchronize()
                    for value, row in held:
                        check_original(value, row)
                    self.assertEqual(len({v.data_ptr() for v, _ in held}), 3)
                    self.assertEqual(immutable.export(), wires[0])
                    self.assertEqual(codec.graph_capture_count, int(execution == 'graph'))
                    self.assertEqual(codec.accounting(immutable)['additional_shared_device_bytes'],
                        45568 if mode == 'packed' else 0)
                    del codec, held, immutable, value, churn
                    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
