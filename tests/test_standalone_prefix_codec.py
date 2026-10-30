"""Warp-prefix controller contracts and full frozen-packet CUDA restoration."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from kv_compare import standalone_prefix_core as core
from kv_compare.standalone_inverse_codec import StandaloneInverseCodec
from kv_compare.standalone_prefix_codec import StandalonePrefixCodec
from kv_compare.team_onestep import layerwise as lw
from kv_compare.team_packet import TeamPacket, save_state
from tests.test_standalone_rolling_codec import fixture
from tests.test_team_packet import packet_fixture


class ContractTests(unittest.TestCase):
    def test_unchanged_sender_tables_shared_bytes_and_explicit_cpu_scope(self):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, _ = fixture(2, 64)
            save_state(raw, consts, rank)
            reference = StandaloneInverseCodec(raw, 'cpu', 'dense_uniform', inverse_mode='packed')
            candidate = StandalonePrefixCodec(raw, 'cpu', 'dense_uniform')
            header = candidate.header((2, 64, 256))
            self.assertEqual(header, reference.header((2, 64, 256)))
            self.assertIs(candidate.native._launch.__self__, candidate)
            self.assertIs(candidate._original_launch.__self__, candidate.native)
            self.assertIs(candidate.native.decode.__func__, reference.native.decode.__func__)
            self.assertIs(candidate.native.model.__func__, reference.native.model.__func__)
            self.assertIs(candidate.native.encode.__func__, reference.native.encode.__func__)
            self.assertTrue(candidate.feature_reuse)
            self.assertIsNone(candidate.decoder_chunks)
            self.assertEqual(candidate.native.chunks, 2)
            self.assertEqual(candidate.native.product, 'cublas')
            self.assertTrue(torch.equal(candidate.core_coarse, reference.core_coarse))
            self.assertTrue(torch.equal(candidate.core_sparse, reference.core_sparse))
            packet, _, _ = packet_fixture(header)
            info = candidate.accounting(packet)
            self.assertEqual(info['decoder_kernel'], '_prefix_ldec_kernel')
            self.assertEqual(info['decoder_prefix_algorithm'], 'warp_ballot_popcount')
            self.assertEqual(info['prefix_mode'], 'warp')
            self.assertEqual(info['prefix_geometry'], 128)
            self.assertEqual(info['prefix_warps'], 1)
            self.assertTrue(info['prefix_proof_verified'])
            self.assertFalse(info['prefix_runtime_verified'])
            self.assertEqual(info['prefix_additional_shared_device_bytes'], 0)
            self.assertEqual(info['additional_shared_device_bytes'], 45568)
            self.assertEqual(info['coder_table_bytes'], 103256)
            self.assertEqual(info['original_coder_table_bytes'], 57688)
            self.assertIsNone(info['fallback_codec'])
            with mock.patch.object(StandaloneInverseCodec, 'decode') as decode:
                with self.assertRaisesRegex(RuntimeError, 'actual CUDA'):
                    candidate.decode(packet)
                decode.assert_not_called()

    def test_decoder_only_instance_dispatch_preserves_original_arguments(self):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, _ = fixture(2, 64)
            save_state(raw, consts, rank)
            candidate = StandalonePrefixCodec(raw, 'cpu', 'dense_uniform')
            args = tuple(object() for _ in range(26))
            expected = object()
            candidate._original_launch = mock.Mock(return_value=expected)
            with mock.patch.object(candidate, '_verify_production_variant') as verify:
                self.assertIs(candidate._launch_prefix(lw._ldec_kernel, (3,), *args,
                    H2=64, TB=32, FIRST=True, LAST=False), expected)
                verify.assert_called_once()
            call = candidate._original_launch.call_args
            self.assertIs(call.args[0], core._prefix_ldec_kernel)
            self.assertEqual(call.args[1], (3,))
            self.assertEqual(call.args[2:22], args[:20])
            self.assertIs(call.args[22], candidate.core_coarse)
            self.assertIs(call.args[23], candidate.core_sparse)
            self.assertEqual(call.args[24:], args[20:])
            self.assertEqual(call.kwargs,
                dict(H2=64, TB=32, FIRST=True, LAST=False, PACKED=True))
            candidate._original_launch.reset_mock()
            self.assertIs(candidate._launch_prefix(lw._lfeat_kernel, (9,), *args,
                BLOCK=1024, warps=4), expected)
            candidate._original_launch.assert_called_once_with(lw._lfeat_kernel, (9,),
                *args, BLOCK=1024, warps=4)
            candidate._original_launch.reset_mock()
            for options in (dict(H2=32, TB=32), dict(H2=64, TB=64),
                            dict(H2=64, TB=32, warps=4), dict(H2=64, TB=32, warps=True),
                            dict(H2=64, TB=32, PACKED=True)):
                with self.assertRaises(ValueError):
                    candidate._launch_prefix(lw._ldec_kernel, (3,), *args, **options)
            with self.assertRaises(ValueError):
                candidate._launch_prefix(lw._ldec_kernel, (3,), *args[:-1], H2=64, TB=32)
            candidate._original_launch.assert_not_called()

    def test_failed_prefix_proof_stops_before_candidate_dispatch(self):
        with tempfile.TemporaryDirectory() as raw:
            consts, rank, _ = fixture(2, 64)
            save_state(raw, consts, rank)
            with mock.patch.object(core, 'prefix_proof', return_value={'verified': False}):
                with self.assertRaisesRegex(ValueError, 'software proof'):
                    StandalonePrefixCodec(raw, 'cpu', 'dense_uniform')

    def test_production_specializations_are_verified_before_use_and_never_compiled_in_capture(self):
        codec = StandalonePrefixCodec.__new__(StandalonePrefixCodec)
        codec.device = SimpleNamespace(type='cuda')
        codec.native = SimpleNamespace(opts={'enable_fp_fusion': False})
        codec.prefix_runtime_proof = {'verified': True}
        codec._prefix_production_layouts = {}
        storage = torch.empty(64, dtype=torch.int32)
        args = (storage,) * 22 + (0, 512, 256, 2, 1, 0.25)
        options = dict(FIRST=True, LAST=False, H2=64, TB=32, PACKED=True)
        compiled = object()
        receipt = dict(verified=True, hash='checked_mock_compiled_variant')
        with mock.patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False) as capturing:
            with mock.patch.object(core._prefix_ldec_kernel, 'warmup', return_value=compiled) as warmup:
                with mock.patch.object(core, 'validate_compiled_prefix', return_value=receipt) as validate:
                    codec._verify_production_variant((1,), args, options)
                    warmup.assert_called_once_with(*args, grid=(1,), num_warps=1,
                        enable_fp_fusion=False, **options)
                    validate.assert_called_once_with(compiled, expected_calls=2)
                    self.assertEqual(len(codec._prefix_production_layouts), 1)
                    capturing.return_value = True
                    codec._verify_production_variant((1,), args, options)
                    self.assertEqual(warmup.call_count, 1)
                    self.assertEqual(validate.call_count, 1)
                    with self.assertRaisesRegex(RuntimeError, 'CUDA capture'):
                        codec._verify_production_variant((1,), args[:-5] + (8192,) + args[-4:], options)
                    self.assertEqual(warmup.call_count, 1)
                    capturing.return_value = False
                    codec._verify_production_variant((1,), args[:22] + (256,) + args[23:], options)
                    self.assertEqual(warmup.call_count, 2)
                    codec._verify_production_variant((1,), (storage[1:],) + args[1:], options)
                    self.assertEqual(warmup.call_count, 3)
                    codec._verify_production_variant((1,), (storage.to(torch.int16),) + args[1:], options)
                    self.assertEqual(warmup.call_count, 4)
                    validate.return_value = {'verified': False}
                    with self.assertRaisesRegex(ValueError, 'compiled production'):
                        codec._verify_production_variant((1,), args, dict(options, FIRST=False))
                    self.assertEqual(len(codec._prefix_production_layouts), 4)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1',
                     'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_frozen_dense_rank_graph_direct_packets_and_lifetimes(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/standalone_inverse_a4500_20261030_060024_1787751'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for arm in ('dense_uniform', 'rank256_uniform'):
            for tokens in (512, 8192):
                folder = report / f'inverse_packed_{arm}_t{tokens}'
                spec = json.loads((folder / 'request.json').read_text())['codec']
                rows = json.loads((folder / 'worker_results.json').read_text())['rows']
                wires = [Path(row['packet']).read_bytes() for row in rows]
                for wire, row in zip(wires, rows):
                    self.assertEqual(len(wire), row['packet_bytes'])
                    self.assertEqual(hashlib.sha256(wire).hexdigest(), row['packet_sha256'])
                execution = 'graph' if tokens == 512 else 'direct'
                codec = StandalonePrefixCodec(spec['state'], 'cuda:0', arm,
                    decoder_mode=execution)
                self.assertTrue(codec.prefix_runtime_proof['verified'])
                packets = [TeamPacket.import_bytes(wire, 'cuda:0') for wire in wires]
                largest = max(packets, key=lambda packet: packet.payload.numel())
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
                self.assertEqual(len({value.data_ptr() for value, _ in held}), 3)
                self.assertEqual(immutable.export(), wires[0])
                self.assertEqual(codec.graph_capture_count, int(execution == 'graph'))
                info = codec.accounting(immutable)
                self.assertTrue(info['prefix_proof_verified'])
                self.assertTrue(info['prefix_runtime_verified'])
                self.assertGreater(info['prefix_production_layout_count'], 0)
                self.assertTrue(all(row['verified'] for row in info['prefix_production_layouts']))
                self.assertEqual(info['additional_shared_device_bytes'], 45568)
                self.assertEqual(info['prefix_additional_shared_device_bytes'], 0)
                del codec, held, immutable, value, churn
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
