"""Exact packed/masked inverse proofs and unchanged word-decoder arithmetic."""
import ast
import hashlib
import inspect
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from kv_compare import standalone_inverse_core as core
from kv_compare.team_onestep import gpu_coder as gc
from kv_compare.team_onestep import layerwise as lw
from tests.test_team_packet import fixed_state


def install_core(native, coarse, sparse, *, packed):
    """Instance-local adapter used by semantic tests, without source mutation."""
    original = native._launch
    position = list(inspect.signature(lw._ldec_kernel.fn).parameters).index('tail_ptr') + 1

    def launch(kernel, grid, *args, **kwargs):
        if kernel is lw._ldec_kernel:
            return original(core._core_ldec_kernel, grid,
                *args[:position], coarse, sparse, *args[position:], PACKED=packed, **kwargs)
        return original(kernel, grid, *args, **kwargs)

    native._launch = launch


def truth():
    tables = gc.slice_tables('cpu')
    result = []
    for c in range(gc.CLASSES):
        off, n = int(tables['off'][c]), int(tables['K'][c]) + 2
        cumulative = tables['cum'][off:off + n].numpy()
        result.append(np.searchsorted(cumulative, np.arange(65536), side='right') - 1)
    return np.asarray(result, dtype=np.uint16)


def probe_expected(n, mixed):
    index = np.arange(n)
    if mixed:
        c, slot = index % 10, ((index // 10) * 4051 + 65535) & 65535
    else:
        c, slot = index // 65536, index & 65535
    return truth()[c, slot]


def function(path, name):
    module = ast.parse(Path(path).read_text())
    return next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name)


class Normalize(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in ('gc', 'lw'):
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == '_dec_value_core':
            node.func.id = '_dec_value'
            node.args = [arg for arg in node.args if not (
                isinstance(arg, ast.Name) and arg.id in ('coarse_ptr', 'sparse_ptr', 'PACKED'))]
        return node


class ProofTests(unittest.TestCase):
    def test_all_slots_both_algorithms_and_owned_tables(self):
        coarse, sparse, proof = core.build_inverse_tables('cpu')
        self.assertEqual((coarse.dtype, tuple(coarse.shape)), (torch.int32, (10, 1024)))
        self.assertEqual((sparse.dtype, sparse.numel()), (torch.uint16, 2304))
        self.assertTrue(coarse.is_contiguous() and sparse.is_contiguous())
        self.assertTrue(proof['verified'])
        self.assertEqual(proof['checked_slots'], 655360)
        self.assertEqual((proof['coarse_bytes'], proof['sparse_bytes'], proof['table_bytes']),
                         (40960, 4608, 45568))
        self.assertEqual(proof['exception_counts_by_class'], [16, 6, 4, 2, 2, 2, 2, 1, 1, 0])
        self.assertEqual(hashlib.sha256(coarse.numpy().astype('<i4').tobytes()).hexdigest(),
                         proof['coarse_sha256'])
        self.assertEqual(hashlib.sha256(sparse.numpy().astype('<u2').tobytes()).hexdigest(),
                         proof['sparse_sha256'])
        expected = truth()
        digest = hashlib.sha256(expected.astype('<u2').tobytes()).hexdigest()
        for key in ('reference_inverse_sha256', 'coarse_search_sha256',
                    'masked_search_sha256', 'packed_inverse_sha256'):
            self.assertEqual(proof[key], digest)
        self.assertTrue(all(item['masked']['simulated_loads'] <= item['original']['simulated_loads']
                            for item in proof['searches']))
        c2, s2, p2 = core.build_inverse_tables('cpu')
        self.assertNotEqual(coarse.data_ptr(), c2.data_ptr())
        self.assertNotEqual(sparse.data_ptr(), s2.data_ptr())
        coarse[0, 0] = 0
        sparse[0] = 1000
        self.assertNotEqual(int(sparse[0]), int(s2[0]))
        proof['sources'].clear()
        self.assertEqual(p2, core.inverse_proof())

    def test_exact_buckets_boundaries_and_safe_sparse_addresses(self):
        coarse, sparse, _ = core.build_inverse_tables('cpu')
        entries = coarse.numpy().view(np.uint32)
        values = sparse.numpy()
        slots = np.arange(65536)
        expected = truth()
        for c in range(10):
            actual = core._packed_inverse(entries, values, c, slots)
            self.assertTrue(np.array_equal(actual, expected[c]))
            entry = entries[c, slots >> 6]
            rare = (entry >> 31) != 0
            address = np.where(rare, entry & 0x7fffffff, 0) + (slots & 63)
            self.assertTrue(np.all(address >= 0) and np.all(address < len(values)))
        direct = entries[(entries >> 31) == 0]
        self.assertTrue(np.all(((direct >> 10) & 127) <= 64))
        self.assertTrue(np.any(((direct >> 10) & 127) == 64))

    def test_only_stage_one_changes_and_native_layer_kernel_is_preserved(self):
        parent = Path(core.__file__).parent
        old = function(parent / 'team_onestep/gpu_coder.py', '_dec_value')
        new = function(core.__file__, '_dec_value_core')
        branch = next(i for i, node in enumerate(new.body)
                      if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'PACKED')
        new.body[branch:branch + 1] = new.body[branch].orelse
        masked = []
        for node in ast.walk(new):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'tl'
                    and node.func.attr == 'load' and node.keywords
                    and 'mid' in ast.unparse(node.args[0])):
                masked.append(node)
        self.assertEqual(len(masked), 1)
        self.assertEqual({keyword.arg for keyword in masked[0].keywords}, {'mask', 'other'})
        self.assertEqual(ast.unparse(next(k.value for k in masked[0].keywords if k.arg == 'mask')),
                         'khi - klo > 1')
        self.assertEqual(ast.unparse(next(k.value for k in masked[0].keywords if k.arg == 'other')), '0')
        masked[0].keywords = []
        new.name = old.name
        new.args.args = [arg for arg in new.args.args
                         if arg.arg not in ('coarse_ptr', 'sparse_ptr', 'PACKED')]
        self.assertEqual(ast.dump(Normalize().visit(new)), ast.dump(Normalize().visit(old)))
        old = function(parent / 'team_onestep/layerwise.py', '_ldec_kernel')
        new = function(core.__file__, '_core_ldec_kernel')
        core_calls = [node for node in ast.walk(new) if isinstance(node, ast.Call)
                      and ((isinstance(node.func, ast.Name) and node.func.id == '_dec_value_core')
                           or (isinstance(node.func, ast.Attribute) and node.func.attr == '_dec_value_core'))]
        self.assertEqual(len(core_calls), 1)
        self.assertIsInstance(core_calls[0].func, ast.Name)
        new.name = old.name
        new.args.args = [arg for arg in new.args.args
                         if arg.arg not in ('coarse_ptr', 'sparse_ptr', 'PACKED')]
        self.assertEqual(ast.dump(Normalize().visit(new)), ast.dump(Normalize().visit(old)))


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_mixed_classes_tail_and_resolved_lanes_match_cdf(self):
        coarse, sparse, _ = core.build_inverse_tables('cpu')
        tables = gc.slice_tables('cpu')
        n = 1024
        expected = torch.from_numpy(probe_expected(n, True))
        for packed in (False, True):
            output = torch.empty(n, dtype=torch.uint16)
            core._probe_inverse_kernel[(gc.triton.cdiv(n, 128),)](
                tables['cumf'], tables['lut'], coarse, sparse, output,
                N=n, BLOCK=128, PACKED=packed, MIXED=True, num_warps=1)
            self.assertTrue(torch.equal(output, expected))


def native_fixture(device, mode):
    constants = fixed_state(layers=2, tokens=64)
    constants['Aq'][1].copy_(torch.eye(256, dtype=torch.int8))
    constants['sa'].fill_(0.25)
    constants = {name:value.to(device) for name,value in constants.items()}
    native = lw.LayerwiseOneStep(constants, 128, 128, tokens_per_program=32,
                                chunks=2, mode=mode, search=True)
    words = torch.randn(2, 64, 256, generator=torch.Generator().manual_seed(648)).bfloat16().view(torch.int16)
    words[1, 0, :11] = torch.tensor([0, -32768, 1, 127, 128, 0x3f80, -16512,
                                   0x7f80, -128, 0x7fc1, -1], dtype=torch.int16)
    return native, words.to(device)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_all_655360_inverse_slots_for_both_cores(self):
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        coarse, sparse, proof = core.build_inverse_tables('cuda:0')
        tables = gc.slice_tables('cuda:0')
        n = proof['checked_slots']
        for packed in (False, True):
            for mixed in (False, True):
                output = torch.empty(n, dtype=torch.uint16, device='cuda:0')
                core._probe_inverse_kernel[(gc.triton.cdiv(n, 1024),)](
                    tables['cumf'], tables['lut'], coarse, sparse, output,
                    N=n, BLOCK=1024, PACKED=packed, MIXED=mixed, num_warps=4)
                torch.cuda.synchronize()
                self.assertTrue(np.array_equal(output.cpu().numpy(), probe_expected(n, mixed)))

    def test_actual_gpu_original_streams_both_word_modes_and_all_bf16_patterns(self):
        self.assertIn('RTX A4500', torch.cuda.get_device_name(0))
        coarse, sparse, _ = core.build_inverse_tables('cuda:0')
        for packed in (False, True):
            for mode in ('uniform', 'exact'):
                native, words = native_fixture('cuda:0', mode)
                encoded, starts = native.encode(words)
                old_encoded, old_starts = encoded.clone(), starts.clone()
                reference = native.decode(encoded, starts, 64)
                install_core(native, coarse, sparse, packed=packed)
                first = native.decode(encoded, starts, 64)
                later = native.decode(encoded, starts, 64)
                torch.cuda.synchronize()
                for result in (reference, first, later):
                    self.assertTrue(torch.equal(result, words))
                self.assertNotEqual(first.data_ptr(), later.data_ptr())
                self.assertTrue(torch.equal(encoded, old_encoded))
                self.assertTrue(torch.equal(starts, old_starts))
            constants = {name:value.to('cuda:0') for name,value in
                         fixed_state(layers=1, tokens=256).items()}
            native = lw.LayerwiseOneStep(constants, 128, 128, tokens_per_program=32,
                                        chunks=2, mode='uniform', search=True)
            words = torch.arange(65536, dtype=torch.int32, device='cuda:0').to(torch.int16).reshape(1, 256, 256)
            encoded, starts = native.encode(words)
            old_encoded, old_starts = encoded.clone(), starts.clone()
            install_core(native, coarse, sparse, packed=packed)
            actual = native.decode(encoded, starts, 256)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual, words))
            self.assertTrue(torch.equal(encoded, old_encoded))
            self.assertTrue(torch.equal(starts, old_starts))


if __name__ == '__main__':
    unittest.main()
