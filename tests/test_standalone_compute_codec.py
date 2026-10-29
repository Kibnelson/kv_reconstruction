"""Exact-packet standalone decoder products and Dense encoder preparation."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from kv_compare.standalone_compute_codec import StandaloneComputeCodec
from kv_compare.team_lookup import TeamLookupCodec
from kv_compare.team_packet import TeamPacket, save_state
from kv_compare.team_onestep import gpu_coder
from tests.test_team_packet import fixed_state, packet_fixture


class ContractTests(unittest.TestCase):
    def test_options_and_owned_decoder_do_not_change_encoder_identity(self):
        for options in (dict(cache_model=1), dict(cache_model=True, decoder_tiles=(64,128,64,4)),
                        dict(decoder_tiles=(32,128,64,4))):
            with self.assertRaises(ValueError):
                StandaloneComputeCodec('missing', 'cpu', 'dense_uniform', **options)
        with self.assertRaises(ValueError):
            StandaloneComputeCodec('missing', 'cpu', 'rank256_uniform', cache_model=True)
        with tempfile.TemporaryDirectory() as raw:
            save_state(raw, fixed_state(tokens=256))
            reference = TeamLookupCodec(raw, 'cpu', 'dense_uniform', decoder_mode='direct')
            for tiles in ((64,128,64,4), (128,128,64,4)):
                codec = StandaloneComputeCodec(raw, 'cpu', 'dense_uniform', decoder_tiles=tiles)
                self.assertEqual(codec.header((2,256,256)), reference.header((2,256,256)))
                self.assertEqual(codec.native.product, 'cublas')
                self.assertEqual(codec.decoder_native.product, 'triton')
                self.assertIs(codec.decoder_native.k, codec.native.k)
                self.assertIs(codec.decoder_native.tables, codec.native.tables)
                self.assertIs(codec.native.decode.__self__, codec.decoder_native)
                packet, _, _ = packet_fixture(codec.header((2,256,256)))
                info = codec.accounting(packet)
                self.assertEqual(info['coder_table_bytes'], reference.coder_table_bytes)
                self.assertEqual(info['additional_shared_device_bytes'], 0)
                self.assertEqual(info['full_inverse_shared_bytes'], 0)
                self.assertEqual(info['encoder_product'], 'cublas')
                self.assertEqual(info['decoder_product'], 'triton')
                self.assertIsNone(info['fallback_codec'])
                for name in ('other', 'mask', 'splitzip_source', 'full_inverse'):
                    self.assertFalse(hasattr(codec, name))

    def test_unsupported_decoder_geometry_rejected_without_cublas_fallback(self):
        with tempfile.TemporaryDirectory() as raw:
            save_state(raw, fixed_state())
            codec = StandaloneComputeCodec(raw, 'cpu', 'dense_uniform', decoder_tiles=(128,128,64,4))
            packet, _, _ = packet_fixture(codec.header((2,64,256)))
            with mock.patch.object(codec.native, 'decode') as decode:
                with self.assertRaises(ValueError):
                    codec.decode(packet)
                decode.assert_not_called()


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_cached_dense_model_reuses_preparation_and_preserves_complete_wire(self):
        with tempfile.TemporaryDirectory() as raw:
            generator = torch.Generator().manual_seed(642)
            consts = fixed_state(tokens=64)
            consts['Aq'][1].copy_(torch.randint(-2,3,(256,256),generator=generator,dtype=torch.int8))
            consts['sa'].fill_(0.01)
            save_state(raw, consts)
            original = torch.randn(2,64,256,generator=generator).bfloat16().view(torch.int16)
            reference = TeamLookupCodec(raw, 'cpu', 'dense_uniform', decoder_mode='direct')
            wire = reference.encode(original).export()
            codec = StandaloneComputeCodec(raw, 'cpu', 'dense_uniform', cache_model=True)
            self.assertIs(codec.cached_model.tables, codec.native.tables)
            with mock.patch.object(gpu_coder, 'slice_tables', side_effect=AssertionError('warm encoder rebuilt CDF tables')):
                packet = codec.encode(original)
                self.assertEqual(packet.export(), wire)
                held = codec.decode(packet)
                again = codec.decode(packet)
            self.assertTrue(torch.equal(held, original))
            self.assertTrue(torch.equal(again, original))
            self.assertNotEqual(held.data_ptr(), again.data_ptr())
            self.assertEqual(packet.export(), wire)

    def test_decoder_only_product_exact_words_and_unchanged_sender_wire(self):
        with tempfile.TemporaryDirectory() as raw:
            generator = torch.Generator().manual_seed(391)
            consts = fixed_state(tokens=128)
            consts['Aq'][1].copy_(torch.randint(-2,3,(256,256),generator=generator,dtype=torch.int8))
            consts['sa'].fill_(0.01)
            save_state(raw, consts)
            original = torch.randn(2,128,256,generator=generator).bfloat16().view(torch.int16)
            reference = TeamLookupCodec(raw, 'cpu', 'dense_uniform', decoder_mode='direct')
            wire = reference.encode(original).export()
            codec = StandaloneComputeCodec(raw, 'cpu', 'dense_uniform', decoder_tiles=(64,128,64,4))
            packet = codec.encode(original)
            self.assertEqual(packet.export(), wire)
            held = codec.decode(packet)
            codec.buffers['xbuf'].fill_(99)
            again = codec.decode(packet)
            self.assertTrue(torch.equal(held, original))
            self.assertTrue(torch.equal(again, original))
            self.assertNotEqual(held.data_ptr(), again.data_ptr())
            self.assertEqual(packet.export(), wire)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_all_compute_modes_frozen_wire_graph_direct_and_lifetimes(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/standalone_lut_a4500_20261029_143932_3085067'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for method in ('team_dense_uniform', 'team_rank256_uniform'):
            for tokens in (512,8192):
                folder = report / f'{method}_t{tokens}'
                spec = json.loads((folder/'request.json').read_text())['codec']
                rows = json.loads((folder/'worker_results.json').read_text())['rows']
                wires = [Path(row['packet']).read_bytes() for row in rows]
                for wire,row in zip(wires,rows):
                    self.assertEqual(len(wire),row['packet_bytes'])
                    self.assertEqual(hashlib.sha256(wire).hexdigest(),row['packet_sha256'])
                options = [dict(decoder_tiles=(128,128,64,4)),dict(decoder_tiles=(64,128,64,4))]
                if method == 'team_dense_uniform':
                    options.append(dict(cache_model=True))
                for flags in options:
                    mode = 'graph' if tokens == 512 else 'direct'
                    codec = StandaloneComputeCodec(spec['state'],'cuda:0',spec['arm'],decoder_mode=mode,**flags)
                    packets = [TeamPacket.import_bytes(wire,'cuda:0') for wire in wires]
                    immutable = packets[0]
                    largest = max(packets,key=lambda p:p.payload.numel())
                    codec.prepare(largest)
                    if mode == 'graph':
                        codec.reserve_payload_words(largest.payload.numel())
                        codec.graph_prepare(largest)
                    producer = torch.cuda.current_stream()
                    callers = [torch.cuda.Stream(),torch.cuda.Stream()]
                    for stream in callers:
                        stream.wait_stream(producer)
                    held = []
                    for turn,index in enumerate((0,1,0)):
                        with torch.cuda.stream(callers[turn%2]):
                            value = codec.decode(packets[index])
                        held.append((value,rows[index]))
                    del packets,largest
                    gc.collect();torch.cuda.synchronize()
                    for value,row in held:
                        check_original(value,row)
                    self.assertEqual(len({v.data_ptr() for v,_ in held}),3)
                    self.assertEqual(immutable.export(),wires[0])
                    self.assertEqual(codec.graph_capture_count,int(mode=='graph'))
                    info = codec.accounting(immutable)
                    self.assertEqual(info['full_inverse_shared_bytes'],0)
                    self.assertEqual(info['additional_shared_device_bytes'],0)
                    self.assertEqual(info['coder_table_bytes'],57688)
                    if flags.get('cache_model'):
                        with mock.patch.object(gpu_coder,'slice_tables',side_effect=AssertionError('warm preparation repeated')):
                            self.assertEqual(codec.encode(held[0][0]).export(),wires[0])
                    del codec,held,immutable,value
                    gc.collect();torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
