"""Exact-wire and stream-lifetime checks for critical-first mixed decoding."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare import mixed_transfer as mt
from kv_compare.causal_packet import Packet, sha
from kv_compare.mixed_decode_fast import DecodeFastMixedTransfer
from kv_compare.mixed_decode_overlap import OverlapMixedTransfer
from kv_compare.team_packet import save_state
from tests.test_mixed_decode_fast import OracleBaseline
from tests.test_mixed_transfer import state
from tests.test_team_packet import fixed_state


class ContractTests(unittest.TestCase):
    def test_explicit_flags_and_required_execution_path(self):
        for flags in (dict(overlap_decode=1), dict(mapped_io=False), dict(selective_decode=False)):
            with self.assertRaises(ValueError):
                OverlapMixedTransfer('missing', 'missing', 'missing', 'x', [True], 'cpu', **flags)

    def test_no_rans_uses_original_frame_without_partition(self):
        with tempfile.TemporaryDirectory() as raw:
            _, lut = state(raw)
            with mock.patch.object(mt, 'verify_splitzip_source', return_value={}), mock.patch.object(mt, 'Baseline', OracleBaseline):
                codec = OverlapMixedTransfer('missing', 'unused', lut, sha(lut), [False, False], 'cpu')
                words = torch.ones(2, 64, 256, dtype=torch.bfloat16).view(torch.int16)
                packet = codec.encode(words)
                self.assertTrue(torch.equal(codec.decode(packet), words))
                self.assertFalse(codec.execution_plan['overlap_decode_active'])
                self.assertEqual(codec.accounting(packet)['partition_index_bytes'], 0)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_nonzero_predictor_adjacent_layers_exact_wire_and_owned_outputs(self):
        with tempfile.TemporaryDirectory() as raw:
            _, lut = state(raw, layers=4)
            consts = fixed_state(layers=4, tokens=64)
            consts['Aq'][1:].copy_(torch.eye(256, dtype=torch.int8).expand(3, -1, -1))
            consts['sa'].fill_(0.25)
            path = Path(raw) / 'nonzero'
            save_state(path, consts)
            mask = [False, True, True, False]
            original = torch.randn(4, 64, 256, generator=torch.Generator().manual_seed(631)).bfloat16().view(torch.int16)
            with mock.patch.object(mt, 'verify_splitzip_source', return_value={}), mock.patch.object(mt, 'Baseline', OracleBaseline):
                args = (path, 'unused', lut, sha(lut), mask, 'cpu')
                old = DecodeFastMixedTransfer(*args, arm='dense_uniform', selective=False,
                    mapped_io=True, selective_decode=True)
                codec = OverlapMixedTransfer(*args, arm='dense_uniform', selective=False)
                wire = old.encode(original).export()
                self.assertEqual(codec.encode(original).export(), wire)
                packet = Packet.restore(wire, 'cpu')
                first = codec.decode(packet)
                codec.buffers['xbuf'].fill_(83)
                codec.buffers['ssum'].fill_(123456)
                second = codec.decode(packet)
                self.assertTrue(torch.equal(first, original))
                self.assertTrue(torch.equal(second, original))
                self.assertNotEqual(first.data_ptr(), second.data_ptr())
                self.assertEqual(packet.export(), wire)
                self.assertEqual(codec.execution_plan['critical_fallback_layers'], [0])
                self.assertEqual(codec.execution_plan['bulk_fallback_layers'], [3])
                info = codec.accounting(packet)
                self.assertEqual(info['packet_bytes'], len(wire))
                self.assertEqual(info['fallback_decoded_buffer_bytes'], 0)
                self.assertEqual(info['critical_fallback_output_bytes'], 32768)
                self.assertEqual(info['bulk_fallback_output_bytes'], 32768)
                damaged = Packet.restore(wire, 'cpu')
                damaged.header['extra']['fallback_layers'] = [1, 3]
                with mock.patch('kv_compare.mixed_decode_overlap.decode_subset_into') as subset:
                    with self.assertRaises(ValueError):
                        codec.decode(damaged)
                    subset.assert_not_called()
                with self.assertRaises(ValueError):
                    codec.decode_native(old.unpack(packet))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_frozen_packets_cross_streams_delays_and_owned_outputs(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/lossless_decode_a4500_20261029_125616_867717'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for count in (2, 14):
            codec = None
            for tokens in (512, 8192):
                folder = report / f'decode_both_k{count}_t{tokens}'
                spec = json.loads((folder / 'request.json').read_text())['codec']
                rows = json.loads((folder / 'worker_results.json').read_text())['rows']
                if codec is None:
                    codec = OverlapMixedTransfer(spec['state'], spec['splitzip_source'], spec['splitzip_state'],
                        spec['splitzip_state_sha256'], spec['mask'], 'cuda:0', selective=True, bundle=True)
                    self.assertEqual(codec.predictor_state_sha256, spec['predictor_state_sha256'])
                wires = [Path(row['packet']).read_bytes() for row in rows]
                for wire, row in zip(wires, rows):
                    self.assertEqual(len(wire), row['packet_bytes'])
                    self.assertEqual(hashlib.sha256(wire).hexdigest(), row['packet_sha256'])
                # Parse while inputs are live, then issue back-to-back native
                # calls on different caller streams without intervening waits.
                packets = [Packet.restore(wire, 'cuda:0') for wire in wires]
                immutable_packet = packets[0]
                parsed = [codec.unpack(packet) for packet in packets]
                producer = torch.cuda.current_stream()
                callers = [torch.cuda.Stream(), torch.cuda.Stream()]
                for caller in callers:
                    caller.wait_stream(producer)
                # Delay bulk independently. Correctness must rely on its final
                # join rather than an accidental early completion.
                with torch.cuda.stream(codec._bulk_stream):
                    torch.cuda._sleep(2_000_000)
                held = []
                for turn, index in enumerate((0, 1, 0)):
                    with torch.cuda.stream(callers[turn % 2]):
                        torch.cuda._sleep(100_000)
                        held.append((codec.decode_native(parsed[index]), rows[index]))
                del parsed, packets
                gc.collect()
                torch.cuda.synchronize()
                for value, row in held:
                    check_original(value, row)
                self.assertEqual(len({v.data_ptr() for v, _ in held}), 3)
                self.assertEqual(immutable_packet.export(), wires[0])
                self.assertEqual(codec.accounting(immutable_packet)['fallback_decoded_buffer_bytes'], 0)
                del held, immutable_packet
            del codec
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
