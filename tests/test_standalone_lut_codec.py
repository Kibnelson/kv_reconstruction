"""Standalone full-LUT dispatch, wire parity and real-CUDA API checks."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare.standalone_lut_codec import StandaloneLUTCodec
from kv_compare.team_lookup import TeamLookupCodec
from kv_compare.team_packet import TeamPacket, save_state
from tests.test_team_packet import fixed_state, packet_fixture


class ContractTests(unittest.TestCase):
    def test_shared_table_and_original_header_stay_separate(self):
        with self.assertRaises(ValueError):
            StandaloneLUTCodec('missing', 'cpu', 'dense_uniform', full_lut=1)
        with tempfile.TemporaryDirectory() as raw:
            save_state(raw, fixed_state())
            for mode in ('graph', 'direct'):
                reference = TeamLookupCodec(raw, 'cpu', 'dense_uniform', decoder_mode=mode)
                codec = StandaloneLUTCodec(raw, 'cpu', 'dense_uniform', decoder_mode=mode)
                self.assertEqual(codec.header((2, 64, 256)), reference.header((2, 64, 256)))
                self.assertEqual(codec.state_sha256, reference.state_sha256)
                packet, _, _ = packet_fixture(codec.header((2, 64, 256)))
                info, old = codec.accounting(packet), reference.accounting(packet)
                self.assertEqual(info['packet_bytes'], old['packet_bytes'])
                self.assertEqual(info['shared_predictor_bytes'], old['shared_predictor_bytes'])
                self.assertEqual(info['coder_table_bytes']-old['coder_table_bytes'], 1310720)
                self.assertEqual(info['full_inverse_shared_bytes'], 1310720)
                self.assertTrue(info['standalone_all_layers'])
                self.assertIsNone(info['fallback_codec'])
                for name in ('other', 'fallback_layers', 'mask', 'splitzip_source'):
                    self.assertFalse(hasattr(codec, name))

    def test_bad_directory_is_rejected_before_decoder_launch(self):
        with tempfile.TemporaryDirectory() as raw:
            save_state(raw, fixed_state())
            codec = StandaloneLUTCodec(raw, 'cpu', 'dense_uniform')
            packet, _, _ = packet_fixture(codec.header((2, 64, 256)))
            packet.offsets[0] = -1
            with mock.patch.object(codec.native, 'decode') as native:
                with self.assertRaises(ValueError):
                    codec.decode(packet)
                native.assert_not_called()
            packet, _, _ = packet_fixture(codec.header((2, 64, 256)))
            codec.full_inverse[0, 0] = 1
            with mock.patch.object(codec.native, 'decode') as native:
                with self.assertRaises(ValueError):
                    codec.decode(packet)
                native.assert_not_called()


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_complete_packet_and_nonzero_predictor_owned_reconstruction(self):
        with tempfile.TemporaryDirectory() as raw:
            generator = torch.Generator().manual_seed(917)
            consts = fixed_state(layers=2, tokens=64)
            consts['Aq'][1] = torch.randint(-2, 3, (256, 256), generator=generator, dtype=torch.int8)
            consts['sa'].fill_(0.01)
            save_state(raw, consts)
            original = torch.randn(2, 64, 256, generator=generator).bfloat16().view(torch.int16)
            original[0, 0, :6] = torch.tensor([0, -32768, 0x7f80, -128, 0x7fc1, -1], dtype=torch.int16)
            reference = TeamLookupCodec(raw, 'cpu', 'dense_uniform', decoder_mode='direct')
            codec = StandaloneLUTCodec(raw, 'cpu', 'dense_uniform')
            wire = reference.encode(original).export()
            self.assertEqual(codec.encode(original).export(), wire)
            packet = TeamPacket.import_bytes(wire, 'cpu')
            held = codec.decode(packet)
            codec.buffers['xbuf'].fill_(111)
            codec.buffers['ssum'].fill_(777)
            again = codec.decode(packet)
            self.assertTrue(torch.equal(held, original))
            self.assertTrue(torch.equal(again, original))
            self.assertNotEqual(held.data_ptr(), again.data_ptr())
            self.assertEqual(packet.export(), wire)
            self.assertFalse(hasattr(codec, 'packed'))
            self.assertFalse(hasattr(codec, 'output'))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_dense_rank_graph_direct_frozen_packets_and_lifetimes(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/lossless_decode_a4500_20261029_125616_867717'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for method in ('team_dense_uniform', 'team_rank256_uniform'):
            for tokens in (512, 8192):
                folder = report / f'{method}_t{tokens}'
                spec = json.loads((folder / 'request.json').read_text())['codec']
                rows = json.loads((folder / 'worker_results.json').read_text())['rows']
                mode = 'graph' if tokens == 512 else 'direct'
                codec = StandaloneLUTCodec(spec['state'], 'cuda:0', spec['arm'], decoder_mode=mode)
                self.assertEqual(codec.state_sha256, spec['state_sha256'])
                wires = [Path(r['packet']).read_bytes() for r in rows]
                for wire, row in zip(wires, rows):
                    self.assertEqual(hashlib.sha256(wire).hexdigest(), row['packet_sha256'])
                    self.assertEqual(len(wire), row['packet_bytes'])
                packets = [TeamPacket.import_bytes(wire, 'cuda:0') for wire in wires]
                immutable = packets[0]
                largest = max(packets, key=lambda p: p.payload.numel())
                codec.prepare(largest)
                if mode == 'graph':
                    codec.reserve_payload_words(largest.payload.numel())
                    codec.graph_prepare(largest)
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
                gc.collect()
                torch.cuda.synchronize()
                for value, row in held:
                    check_original(value, row)
                self.assertEqual(len({v.data_ptr() for v, _ in held}), 3)
                self.assertEqual(codec.graph_capture_count, int(mode == 'graph'))
                self.assertEqual(immutable.export(), wires[0])
                self.assertEqual(codec.accounting(immutable)['full_inverse_shared_bytes'], 1310720)
                del codec, held, immutable, value
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
