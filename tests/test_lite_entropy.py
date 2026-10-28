"""Exact lookup intervals, unchanged kernel math, and separate GPU parity gates."""
import ast
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np
import torch

from kv_compare.causal_packet import Packet, float_to_words
from kv_compare.gpu_coder import SliceCoder, _decode_kernel
from kv_compare.lite_entropy import (EntropyEngine, EntropySliceCoder, FULL_LUT_WIDTH,
                                    _decode_full_kernel, full_slice_lookup)


ARMS = tuple((lookup, refine) for lookup in ('coarse', 'full') for refine in (0, 1, 2))


class StructureTests(unittest.TestCase):
    def test_complete_lookup_matches_every_original_cumulative_interval(self):
        original = SliceCoder('cpu', values_per_stream=2, block=8)
        full = full_slice_lookup(original.cum, original.off, original.K)
        self.assertEqual(full.shape, (10, 65536)); self.assertEqual(full.dtype, np.uint16)
        self.assertEqual(full.nbytes, 1310720)
        slots = np.arange(FULL_LUT_WIDTH)
        for c in range(10):
            off, count = int(original.off[c]), int(original.K[c])
            row = original.cum.numpy()[off:off + count + 2]
            k = full[c].astype(np.int64)
            self.assertTrue(np.array_equal(k, np.searchsorted(row, slots, side='right') - 1))
            self.assertTrue(np.all(row[k] <= slots))
            self.assertTrue(np.all(slots < row[k + 1]))
            self.assertEqual(k[-1], count)  # The final slot is the escape.

    def test_full_kernel_only_replaces_stage_one_lookup(self):
        original = ast.parse(textwrap.dedent(inspect.getsource(_decode_kernel.fn)))
        replacement = ast.parse('k = tl.load(lut_ptr + c * LUT_W + slot).to(tl.int32)').body[0]
        changed = 0
        def edit(node):
            nonlocal changed
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    result = []; i = 0
                    while i < len(value):
                        child = value[i]
                        if isinstance(child, ast.Assign) and ast.unparse(child) == 'b = slot >> LUT_SHIFT':
                            self.assertEqual(ast.unparse(value[i + 1]), 'klo = tl.load(lut_ptr + c * LUT_W + b)')
                            self.assertEqual(ast.unparse(value[i + 2]), 'khi = tl.load(lut_ptr + c * LUT_W + b + 1) + 1')
                            self.assertIsInstance(value[i + 3], ast.While)
                            self.assertEqual(ast.unparse(value[i + 4]), 'k = klo')
                            result.append(replacement); changed += 1; i += 5
                        else:
                            if isinstance(child, ast.AST): edit(child)
                            result.append(child); i += 1
                    setattr(node, field, result)
                elif isinstance(value, ast.AST): edit(value)
        edit(original); original.body[0].name = '_decode_full_kernel'
        candidate = ast.parse(textwrap.dedent(inspect.getsource(_decode_full_kernel.fn)))
        self.assertEqual(changed, 1)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(candidate, include_attributes=False))

    def test_original_tables_and_encoder_shared_and_all_table_bytes_charged(self):
        original = SliceCoder('cpu', values_per_stream=2, block=8)
        original_bytes = sum(t.numel() * t.element_size() for t in
                             (original.cum, original.off, original.K, original.lut))
        for lookup, refine in ARMS:
            coder = EntropySliceCoder(original, lookup, refine)
            for name in ('cum', 'off', 'K', 'lut'):
                self.assertIs(getattr(coder, name), getattr(original, name))
            self.assertIs(coder.encode.__func__, original.encode.__func__)
            self.assertEqual(coder.metadata['retained_original_table_bytes'], original_bytes)
            extra = 1310720 if lookup == 'full' else 0
            self.assertEqual(coder.metadata['extra_shared_bytes'], extra)
            self.assertEqual(coder.metadata['total_entropy_table_bytes'], original_bytes + extra)
            if lookup == 'full':
                digest = hashlib.sha256(coder.full_lut.numpy().astype('<u2').tobytes()).hexdigest()
                self.assertEqual(coder.full_lut_sha256, digest)
                self.assertGreater(coder.metadata['full_lut_construction_wall_ms'], 0)
            else:
                self.assertIsNone(coder.full_lut_sha256)
        invalid = (('other', 2), ('full', -1), ('full', 3), ('full', True), ('full', 1.))
        for lookup, refine in invalid:
            with self.subTest(lookup=lookup, refine=refine), self.assertRaises(ValueError):
                EntropySliceCoder(original, lookup, refine)

    def test_default_decoder_delegates_to_inherited_anchor(self):
        original = SliceCoder('cpu', values_per_stream=2, block=8)
        coder = EntropySliceCoder(original)
        marker = object()
        with mock.patch.object(SliceCoder, 'decode', return_value=marker) as method:
            self.assertIs(coder.decode(None, None, None, None, 1), marker)
            method.assert_called_once_with(None, None, None, None, 1)


def coder_cases(device):
    words, mus, spreads = [], [], []
    def append(word, mu, spread=1.):
        words.append(int(word)); mus.append(float(mu)); spreads.append(float(spread))
    def bf16(value):
        return int(float_to_words(torch.tensor(value, dtype=torch.float32)))
    def ordered(word):
        return (~word & 65535) if word & 32768 else word ^ 32768
    def word(index):
        return index ^ 32768 if index & 32768 else ~index & 65535
    for c in range(10):
        for sign in (-1, 1):
            mu = sign * 2. ** (c + 3)
            for delta in (-24., 0., 24.):
                append(bf16(mu + delta), mu)
            # Both sides of an internal slice boundary at the narrowest and
            # widest classes; ordering differs for negative BF16 values.
            if c in (0, 9):
                delta = -24. + 2. ** (c - 3)
                center = ordered(bf16(mu + delta))
                for index in (center - 1, center, center + 1):
                    append(word(index), mu)
    for w in (0, 32768, 1, 32769, 127, 32895, 128, 32896,
              0x7F80, 0xFF80, 0x7FC1, 0xFFFF):
        append(w, 0.)
    for value in (-100., 100.): append(bf16(value), 0.)  # finite raw escapes
    append(bf16(100.), 100., .001)  # one-word slice, no second symbol
    # Exercise the final partial iteration and inactive program lanes.
    while len(words) % 2 == 0 or ((len(words) + 1) // 2) % 8 == 0:
        append(0, 0.)
    return (torch.tensor(words, dtype=torch.int32).to(device, torch.int16),
            torch.tensor(mus, dtype=torch.float32, device=device),
            torch.tensor(spreads, dtype=torch.float32, device=device))


def coder_parity(test, device, values_per_stream=2, block=8):
    words, mu, spread = coder_cases(device)
    if values_per_stream == 64 and block == 32:
        # Thirty-three programs, a partial tail and inactive lanes at the
        # production stream geometry; this remains a correctness test.
        count = 65537
        repeats = (count + words.numel() - 1) // words.numel()
        words, mu, spread = (tensor.repeat(repeats)[:count] for tensor in (words, mu, spread))
    bits = mu.abs() + spread * 2.
    classes = (((bits.view(torch.int32) >> 23) & 255) -
               ((spread.view(torch.int32) >> 23) & 255) - 3).clamp(0, 9)
    test.assertEqual(set(classes.cpu().tolist()), set(range(10)))
    reference = SliceCoder(device, values_per_stream=values_per_stream, block=block, warps=1)
    encoded, starts, n = reference.encode(words, mu, spread)
    test.assertTrue(torch.equal(reference.decode(encoded, starts, mu, spread, n), words))
    for lookup, refine in ARMS:
        with test.subTest(lookup=lookup, refine=refine, V=values_per_stream, B=block):
            coder = EntropySliceCoder(reference, lookup, refine)
            decoded = coder.decode(encoded, starts, mu, spread, n)
            test.assertTrue(torch.equal(decoded, words))
            # A single subnormal gives all but one stream lane no values.
            one = torch.tensor([1], dtype=torch.int16, device=device)
            one_mu = torch.zeros(1, device=device); one_s = torch.ones(1, device=device)
            buf, offsets, count = reference.encode(one, one_mu, one_s)
            test.assertTrue(torch.equal(coder.decode(buf, offsets, one_mu, one_s, count), one))


def engine_parity(test, device):
    from kv_compare.lite_finish import FinishEngine
    from tests.test_llama_causal import tiny_state
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); tiny_state(root)
        options = dict(values_per_stream=2, block=8, predictor_format='bf16')
        source = float_to_words(torch.linspace(-.35, .35, 80).reshape(2, 5, 8)).to(device, torch.int16)
        reference = FinishEngine(root, device, **options)
        packet = reference.encode(source, start=7); wire = packet.export()
        for lookup, refine in ARMS:
            with test.subTest(lookup=lookup, refine=refine):
                candidate = EntropyEngine(root, device, entropy_lookup=lookup, entropy_refine=refine, **options)
                test.assertEqual(candidate.shared_bytes, reference.shared_bytes)
                test.assertEqual(candidate.slice_table_sha, reference.slice_table_sha)
                test.assertTrue(torch.equal(candidate.decode(Packet.restore(wire, device)), source))
                test.assertEqual(candidate.encode(source, start=7).export(), wire)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def setUp(self):
        self.enabled = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.enabled, warn_only=self.warn)

    def test_all_widths_signed_zero_subnormals_boundaries_escapes_and_tails(self):
        coder_parity(self, torch.device('cpu'))

    def test_complete_old_packet_decoding_and_unchanged_encoding(self):
        engine_parity(self, torch.device('cpu'))


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'), 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def setUp(self):
        self.enabled = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device('cuda', torch.cuda.current_device())

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.use_deterministic_algorithms(self.enabled, warn_only=self.warn)

    def test_actual_cuda_production_v64_b32_differential_cases(self):
        coder_parity(self, self.device, values_per_stream=64, block=32)

    def test_actual_cuda_complete_packet_parity(self):
        engine_parity(self, self.device)

    def test_actual_cuda_fresh_graph_receivers_alternate_packet_sizes(self):
        from kv_compare.lite_allocation import allocation_policy
        from kv_compare.lite_graph import GraphDecoder
        from tests.test_lite_graph import packets
        from tests.test_llama_causal import tiny_state
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            _, entries = packets(root, self.device)
            wires = [packet.export() for packet, _ in entries]
            self.assertNotEqual(len(wires[0]), len(wires[1]))
            for lookup, refine in ARMS:
                with allocation_policy(False):
                    candidate = EntropyEngine(root, self.device, entropy_lookup=lookup,
                        entropy_refine=refine, values_per_stream=2, block=8)
                    graph = GraphDecoder(candidate, warmup=2); outputs = []
                    try:
                        for i in (0, 1, 0):
                            value = graph.decode(Packet.restore(wires[i], self.device)); outputs.append(value)
                            self.assertTrue(torch.equal(value, entries[i][1]))
                            self.assertEqual(graph.capture_count, 1)
                        self.assertTrue(torch.equal(outputs[0], entries[0][1]))
                        self.assertNotEqual(outputs[0].data_ptr(), outputs[2].data_ptr())
                    finally:
                        graph.close()


if __name__ == '__main__':
    unittest.main()
