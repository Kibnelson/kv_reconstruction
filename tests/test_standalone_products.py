"""Exact integer-product checks; CPU interpreter work is not GPU timing."""
import os
import unittest
from unittest import mock

import torch

from kv_compare.standalone_products import ALLOWED_TILES, exact_product, validate_tiles


def matrices(M, N, K, device, *, extremes=False):
    """Return contiguous views with nonzero offsets in all three operands."""
    generator = torch.Generator().manual_seed(9701 + K)
    if extremes:
        x_cpu = torch.full((M + 1, K), -128, dtype=torch.int8)
        a_cpu = torch.full((N + 1, K), -128, dtype=torch.int8)
        x_cpu[1::2, 1::2] = 127
        a_cpu[1::2, ::2] = 127
    else:
        x_cpu = torch.randint(-128, 128, (M + 1, K), generator=generator, dtype=torch.int8)
        a_cpu = torch.randint(-128, 128, (N + 1, K), generator=generator, dtype=torch.int8)
    x = x_cpu.to(device)[1:]
    A = a_cpu.to(device)[1:]
    allocation = torch.full((M + 1, N), 123456789, dtype=torch.int32, device=device)
    return x, A, allocation[1:], allocation


class ContractTests(unittest.TestCase):
    def test_explicit_supported_tiles(self):
        for tiles in ALLOWED_TILES:
            self.assertEqual(validate_tiles(tiles), tiles)
            self.assertEqual(validate_tiles(list(tiles)), tiles)
        for tiles in (None, (128, 128, 64), (128, 128, 64, True), (128, 64, 64, 4), '128,128,64,4'):
            with self.assertRaises(ValueError):
                validate_tiles(tiles)

    def test_rejects_dtype_layout_geometry_and_bounds_before_launch(self):
        x = torch.zeros(128, 64, dtype=torch.int8)
        A = torch.zeros(128, 64, dtype=torch.int8)
        out = torch.zeros(128, 128, dtype=torch.int32)
        cases = ((x.float(), A, out), (x, A.long(), out), (x, A, out.short()),
                 (x[0], A, out), (x.t(), A, out), (x, A.t(), out),
                 (x, A, out.t()), (x[:64], A, out[:64]),
                 (x, A[:, :32].contiguous(), out), (x, A, out[:, :64].contiguous()))
        with mock.patch('kv_compare.standalone_products.lw._igemm_kernel') as kernel:
            for args in cases:
                with self.subTest(shapes=[list(v.shape) for v in args]):
                    with self.assertRaises(ValueError):
                        exact_product(*args, ALLOWED_TILES[0])
            wide_x = torch.empty(64, 131072, dtype=torch.int8)
            wide_A = torch.empty(128, 131072, dtype=torch.int8)
            wide_out = torch.empty(64, 128, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, 'INT32 accumulation bound'):
                exact_product(wide_x, wide_A, wide_out, ALLOWED_TILES[1])
            kernel.__getitem__.assert_not_called()

    def test_no_implicit_cpu_fallback_or_output_alias(self):
        x = torch.zeros(128, 512, dtype=torch.int8)
        A = torch.zeros(128, 512, dtype=torch.int8)
        out = torch.zeros(128, 128, dtype=torch.int32)
        with mock.patch.dict(os.environ, {'TRITON_INTERPRET': '0'}):
            with self.assertRaises(RuntimeError):
                exact_product(x, A, out, ALLOWED_TILES[0])
        with mock.patch.dict(os.environ, {'TRITON_INTERPRET': '1'}):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                exact_product(x, A, x.view(torch.int32), ALLOWED_TILES[0])
            with self.assertRaises(ValueError):
                exact_product(x, torch.empty_like(A, device='meta'), out, ALLOWED_TILES[0])


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU Triton interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_both_tiles_random_extremes_and_offset_views_equal_int64_reference(self):
        for tiles in ALLOWED_TILES:
            M, N, K = tiles[0], tiles[1], 64
            for extremes in (False, True):
                with self.subTest(tiles=tiles, extremes=extremes):
                    x, A, out, allocation = matrices(M, N, K, 'cpu', extremes=extremes)
                    before_x, before_A = x.clone(), A.clone()
                    expected = x.long() @ A.long().t()
                    self.assertIs(exact_product(x, A, out, tiles), out)
                    self.assertTrue(torch.equal(out.long(), expected))
                    self.assertTrue(torch.equal(x, before_x))
                    self.assertTrue(torch.equal(A, before_A))
                    self.assertTrue(bool((allocation[0] == 123456789).all()))
                    self.assertGreater(x.storage_offset(), 0)
                    self.assertGreater(A.storage_offset(), 0)
                    self.assertGreater(out.storage_offset(), 0)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_fixed_tiles_rank_dense_signed_extremes_and_reference_parity(self):
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        for tiles in ALLOWED_TILES:
            for K in (256, 2048):
                for extremes in (False, True):
                    with self.subTest(tiles=tiles, K=K, extremes=extremes):
                        x, A, out, allocation = matrices(128, 256, K, 'cuda:0', extremes=extremes)
                        before_x, before_A = x.clone(), A.clone()
                        expected = x.cpu().long() @ A.cpu().long().t()
                        reference = torch._int_mm(x, A.t())
                        self.assertIs(exact_product(x, A, out, tiles), out)
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(out, reference))
                        self.assertTrue(torch.equal(out.cpu().long(), expected))
                        self.assertTrue(torch.equal(x, before_x))
                        self.assertTrue(torch.equal(A, before_A))
                        self.assertTrue(bool((allocation[0] == 123456789).all()))
                        self.assertTrue(all(value.is_contiguous() and value.storage_offset() > 0
                            for value in (x, A, out)))


if __name__ == '__main__':
    unittest.main()
