"""Exhaustive inverse proof and exact standalone decoder semantic checks."""
import hashlib
import inspect
import os
import unittest

import numpy as np
import torch

from kv_compare import standalone_full_lut as full
from kv_compare.team_onestep import gpu_coder as gc
from kv_compare.team_onestep import layerwise as lw
from tests.test_team_packet import fixed_state


def install_inverse(native, inverse):
    """Test-local launch adapter; no source or global dispatch is changed."""
    original = native._launch
    position = list(inspect.signature(lw._ldec_kernel.fn).parameters).index('tail_ptr') + 1

    def launch(kernel, grid, *args, **kwargs):
        if kernel is lw._ldec_kernel:
            return original(full._full_lut_ldec_kernel, grid,
                *args[:position], inverse, *args[position:], **kwargs)
        return original(kernel, grid, *args, **kwargs)

    native._launch = launch


def fixture(device, mode='uniform'):
    consts = fixed_state(layers=3, tokens=64)
    consts['Aq'][1:].copy_(torch.eye(256, dtype=torch.int8).expand(2, -1, -1))
    consts['sa'].fill_(0.25)
    consts = {name:value.to(device) for name,value in consts.items()}
    native = lw.LayerwiseOneStep(consts, 128, 128, tokens_per_program=32, chunks=2,
        mode=mode, search=True)
    words = torch.randn(3, 64, 256, generator=torch.Generator().manual_seed(357)).bfloat16().view(torch.int16).to(device)
    words[2, 0, :11] = torch.tensor([0, -32768, 1, 127, 128, 0x3f80, -16512,
        0x7f80, -128, 0x7fc1, -1], dtype=torch.int16, device=device)
    return native, words


class ProofTests(unittest.TestCase):
    def test_exhaustive_cdf_proof_and_owned_storage(self):
        table, proof = full.build_full_inverse('cpu')
        self.assertEqual(table.dtype, torch.uint16)
        self.assertTrue(table.is_contiguous())
        self.assertEqual(tuple(table.shape), (gc.CLASSES, gc.M))
        self.assertEqual(proof['checked_slots'], 655360)
        self.assertEqual(proof['table_bytes'], 1310720)
        self.assertTrue(proof['verified'])
        digest = hashlib.sha256(table.numpy().astype('<u2').tobytes()).hexdigest()
        self.assertEqual(digest, proof['table_sha256'])
        self.assertEqual(digest, proof['coarse_search_sha256'])
        self.assertEqual(digest, proof['central_tail_reference_sha256'])
        other, other_proof = full.build_full_inverse('cpu')
        self.assertNotEqual(other.data_ptr(), table.data_ptr())
        table[0, 0] = 1234
        self.assertNotEqual(int(table[0, 0]), int(other[0, 0]))
        proof['sources'].clear()
        self.assertEqual(len(full.full_inverse_proof()['sources']), 2)
        self.assertEqual(other_proof, full.full_inverse_proof())

    def test_each_cdf_boundary_and_tail_matches_searchsorted(self):
        table, proof = full.build_full_inverse('cpu')
        tables = gc.slice_tables('cpu')
        for c in range(gc.CLASSES):
            off, count = int(tables['off'][c]), int(tables['K'][c]) + 2
            cum = tables['cum'][off:off + count].numpy().astype(np.int64)
            slots = np.unique(np.clip(np.concatenate((cum - 1, cum, cum + 1)), 0, gc.M - 1))
            expected = np.searchsorted(cum, slots, side='right') - 1
            self.assertTrue(np.array_equal(table[c].numpy()[slots], expected))
        self.assertIn('fixed_table_sha256', proof)


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_full_lut_original_stream_uniform_and_exact_words(self):
        inverse, _ = full.build_full_inverse('cpu')
        for mode in ('uniform', 'exact'):
            native, words = fixture('cpu', mode)
            encoded, starts = native.encode(words)
            frozen_stream, frozen_starts = encoded.clone(), starts.clone()
            reference = native.decode(encoded, starts, 64)
            self.assertTrue(torch.equal(reference, words))
            install_inverse(native, inverse)
            buffers = native.buffers(64)
            buffers['xbuf'].fill_(91); buffers['ssum'].fill_(123456)
            first = native.decode(encoded, starts, 64, bufs=buffers)
            later = native.decode(encoded, starts, 64, bufs=buffers)
            self.assertTrue(torch.equal(first, words))
            self.assertTrue(torch.equal(later, words))
            self.assertTrue(torch.equal(reference, first))
            self.assertNotEqual(first.data_ptr(), later.data_ptr())
            self.assertTrue(torch.equal(encoded, frozen_stream))
            self.assertTrue(torch.equal(starts, frozen_starts))



@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_every_inverse_slot_and_original_native_streams(self):
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        inverse, proof = full.build_full_inverse('cuda:0')
        output = torch.empty_like(inverse)
        full._probe_inverse_kernel[(gc.triton.cdiv(inverse.numel(), 1024),)](
            inverse, output, N=inverse.numel(), BLOCK=1024, num_warps=4)
        torch.cuda.synchronize()
        digest = hashlib.sha256(output.cpu().numpy().astype('<u2').tobytes()).hexdigest()
        self.assertEqual(digest, proof['coarse_search_sha256'])
        for mode in ('uniform', 'exact'):
            native, words = fixture('cuda:0', mode)
            encoded, starts = native.encode(words)
            frozen_stream, frozen_starts = encoded.clone(), starts.clone()
            reference = native.decode(encoded, starts, 64)
            install_inverse(native, inverse)
            first = native.decode(encoded, starts, 64)
            later = native.decode(encoded, starts, 64)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(reference, words))
            self.assertTrue(torch.equal(first, words))
            self.assertTrue(torch.equal(later, words))
            self.assertNotEqual(first.data_ptr(), later.data_ptr())
            self.assertTrue(torch.equal(encoded, frozen_stream))
            self.assertTrue(torch.equal(starts, frozen_starts))

        # Every original BF16 bit pattern, including both zero signs,
        # subnormals, infinities and all NaN payloads, traverses the same
        # unchanged encoder and both stage-1 inversion implementations.
        constants = {name:value.to('cuda:0') for name,value in
                     fixed_state(layers=1, tokens=256).items()}
        native = lw.LayerwiseOneStep(constants, 128, 128, tokens_per_program=32,
                                    chunks=2, mode='uniform', search=True)
        words = torch.arange(65536, dtype=torch.int32, device='cuda:0').to(torch.int16).reshape(1, 256, 256)
        encoded, starts = native.encode(words)
        frozen_stream, frozen_starts = encoded.clone(), starts.clone()
        reference = native.decode(encoded, starts, 256)
        install_inverse(native, inverse)
        actual = native.decode(encoded, starts, 256)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(reference, words))
        self.assertTrue(torch.equal(actual, words))
        self.assertTrue(torch.equal(encoded, frozen_stream))
        self.assertTrue(torch.equal(starts, frozen_starts))


if __name__ == '__main__':
    unittest.main()
