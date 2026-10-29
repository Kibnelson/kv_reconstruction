"""Lossless selective decoder checks; interpreter work is not GPU timing."""
import gc
import hashlib
import json
import os
from pathlib import Path
import unittest

import torch

from kv_compare.causal_packet import Packet
from kv_compare.mixed_decode_context import decode_selective, skipplan
from kv_compare.mixed_fast import FastMixedTransfer
from kv_compare.team_extensions.mixed import MixedOneStep
from tests.test_mixed_transfer import FakeSZ
from tests.test_team_packet import fixed_state


def digest(words):
    return hashlib.sha256(words.detach().cpu().contiguous().numpy().astype('<i2').tobytes()).hexdigest()


def native(mask, tokens=64):
    consts = fixed_state(layers=len(mask), tokens=tokens)
    # A nonzero predictor makes context errors observable in decoded words.
    for layer in range(1, len(mask)):
        consts['Aq'][layer].copy_(torch.eye(256, dtype=torch.int8))
    consts['sa'].fill_(0.25)
    return MixedOneStep(consts, 128, 128, mask, FakeSZ('cpu'),
        tokens_per_program=32, chunks=2, mode='uniform', search=True)


class PlanTests(unittest.TestCase):
    def test_dead_context_and_state_carry_are_distinct_across_gaps(self):
        p = skipplan([False, True, False, True])
        self.assertEqual(p['live_context_layers'], [])
        self.assertEqual(p['dead_context_layers'], [1, 3])
        self.assertEqual(p['state_carry_layers'], [1])
        self.assertEqual(p['prior_reset_layers'], [1, 3])
        p = skipplan([True, True, False, True, True, True])
        self.assertEqual(p['live_context_layers'], [0, 3, 4])
        self.assertEqual(p['dead_context_layers'], [1, 5])
        self.assertEqual(p['state_carry_layers'], [0, 1, 3, 4])
        self.assertEqual(p['feature_output_passes'], 3)
        self.assertEqual(p['surprise_output_passes'], 3)

    def test_endpoint_plans_and_explicit_mask_contract(self):
        self.assertEqual(skipplan([True])['dead_context_layers'], [0])
        self.assertEqual(skipplan([True] * 28)['live_context_layers'], list(range(27)))
        self.assertEqual(skipplan([False] * 28)['rans_layers'], [])
        for mask in ([], [1, 0], [None], 'bits'):
            with self.assertRaises(ValueError): skipplan(mask)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_original_streams_decode_exactly_for_gaps_runs_anchor_and_all_layers(self):
        gen = torch.Generator().manual_seed(2039)
        words = torch.randn(4, 64, 256, generator=gen).bfloat16().view(torch.int16)
        original = words.clone(); expected_digest = digest(original)
        for mask in ([False, True, False, True], [True, True, False, True],
                     [True, False, False, False], [True, True, True, True]):
            with self.subTest(mask=mask):
                codec = native(mask)
                encoded = codec.encode(words)
                stream, starts = encoded['stream'].clone(), encoded['starts'].clone()
                reference = codec.decode_mixed(encoded, 64)
                self.assertTrue(torch.equal(reference, original))
                bufs = codec.buffers(64)
                bufs['xbuf'].fill_(77); bufs['ssum'].fill_(111111)
                held = decode_selective(codec, encoded, 64, bufs=bufs)
                self.assertEqual(digest(held), expected_digest)
                self.assertTrue(torch.equal(held, original))
                later = decode_selective(codec, encoded, 64, bufs=bufs)
                self.assertTrue(torch.equal(later, original))
                self.assertTrue(torch.equal(held, original))
                self.assertNotEqual(held.data_ptr(), later.data_ptr())
                self.assertTrue(torch.equal(encoded['stream'], stream))
                self.assertTrue(torch.equal(encoded['starts'], starts))
                self.assertTrue(torch.equal(words, original))

    def test_prefilled_fallback_all_bf16_patterns_and_native_boundaries(self):
        words = torch.ones(2, 256, 256, dtype=torch.bfloat16).view(torch.int16)
        words[0] = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256)
        original = words.clone(); codec = native([False, True], tokens=256)
        encoded = codec.encode(words)
        self.assertTrue(torch.equal(codec.decode_mixed(encoded, 256), original))
        prefilled = torch.empty_like(words); prefilled[0].copy_(words[0])
        compact = dict(encoded, sz={})
        actual = decode_selective(codec, compact, 256, out=prefilled)
        self.assertIs(actual, prefilled)
        self.assertEqual(digest(actual), digest(original))
        self.assertTrue(torch.equal(actual, original))
        with self.assertRaises(ValueError): decode_selective(codec, compact, 256)
        with self.assertRaises(ValueError): decode_selective(codec, encoded, 0)
        with self.assertRaises(ValueError): decode_selective(codec, encoded, 256, out=words.float())
        with self.assertRaises(ValueError): decode_selective(codec, dict(encoded, sz={0: None, 1: None}), 256)
        zero = native([False, False], tokens=256)
        self.assertTrue(torch.equal(decode_selective(zero, zero.encode(words), 256), original))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_frozen_packets_owned_outputs_and_stream_lifetimes(self):
        from scripts.bench_team_packet import check_original
        root = Path(__file__).resolve().parents[1]
        report = root / 'benchmark_results/lossless_mixed_fast_a4500_20261029_100916_1488711'
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for count in (2, 14):
            folder = report / f'fast_both_k{count}_t512'
            spec = json.loads((folder / 'request.json').read_text())['codec']
            rows = json.loads((folder / 'worker_results.json').read_text())['rows']
            codec = FastMixedTransfer(spec['state'], spec['splitzip_source'], spec['splitzip_state'],
                spec['splitzip_state_sha256'], spec['mask'], 'cuda:0', selective=True, bundle=True)
            self.assertEqual(codec.predictor_state_sha256, spec['predictor_state_sha256'])
            buffers = codec.native.buffers(512)
            held = []
            for index in (0, 1, 0):
                row = rows[index]
                wire = Path(row['packet']).read_bytes()
                self.assertEqual(len(wire), row['packet_bytes'])
                self.assertEqual(hashlib.sha256(wire).hexdigest(), row['packet_sha256'])
                packet = Packet.restore(wire, 'cuda:0')
                parsed = codec.unpack(packet)
                reference = codec.decode(packet)
                check_original(reference, row)
                torch.cuda.synchronize()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    output = torch.empty(parsed.shape, dtype=torch.int16, device='cuda:0')
                    values = codec.other.codec.decode(parsed.native['bundle']).view(torch.int16)
                    output.index_copy_(0, codec._fallback_index,
                        values.reshape(len(codec.fallback_layers), 512, codec.W))
                    output = decode_selective(codec.native, parsed.native, 512, out=output, bufs=buffers)
                    packet.payload.record_stream(stream)
                    del packet, parsed, values
                gc.collect(); torch.cuda.synchronize()
                check_original(output, row)
                self.assertTrue(torch.equal(output, reference))
                for old_value, old_row in held: check_original(old_value, old_row)
                held.append((output, row))
                self.assertEqual(Packet.restore(wire, 'cuda:0').export(), wire)
            self.assertEqual(len({value.data_ptr() for value, _ in held}), 3)


if __name__ == '__main__': unittest.main()
