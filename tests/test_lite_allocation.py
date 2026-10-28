"""Allocation policy restoration, poison checks, and a separate CUDA graph gate."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import torch.utils.deterministic

from kv_compare.causal_packet import Packet, float_to_words
from kv_compare.lite_allocation import allocation_policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.enabled = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.fill = torch.utils.deterministic.fill_uninitialized_memory
        torch.use_deterministic_algorithms(True, warn_only=True)

    def tearDown(self):
        torch.utils.deterministic.fill_uninitialized_memory = self.fill
        torch.use_deterministic_algorithms(self.enabled, warn_only=self.warn)

    def test_nested_policy_restores_enclosing_and_original_flags(self):
        torch.utils.deterministic.fill_uninitialized_memory = True
        with allocation_policy(False) as outer:
            self.assertFalse(outer['fill_uninitialized_memory'])
            self.assertTrue(outer['previous_fill_uninitialized_memory'])
            self.assertTrue(outer['deterministic_algorithms_enabled'])
            self.assertTrue(outer['deterministic_warn_only'])
            with allocation_policy(True) as inner:
                self.assertTrue(inner['fill_uninitialized_memory'])
                self.assertFalse(inner['previous_fill_uninitialized_memory'])
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
        self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_exception_restores_flag_without_changing_deterministic_mode(self):
        torch.utils.deterministic.fill_uninitialized_memory = False
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with allocation_policy(True):
                raise RuntimeError('fixture failure')
        self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_invalid_configuration_does_not_change_flags(self):
        for value in (0, 1, None, 'false'):
            with self.subTest(value=value), self.assertRaises(TypeError):
                with allocation_policy(value):
                    self.fail('invalid policy entered')
        torch.use_deterministic_algorithms(False)
        with self.assertRaisesRegex(RuntimeError, 'already be enabled'):
            with allocation_policy(False):
                self.fail('nondeterministic policy entered')
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, self.fill)

    def test_real_zeros_and_ones_stay_initialized(self):
        for fill in (True, False):
            with allocation_policy(fill):
                self.assertTrue(torch.equal(torch.zeros(7), torch.tensor([0.] * 7)))
                self.assertTrue(torch.equal(torch.ones(7), torch.tensor([1.] * 7)))


@contextmanager
def poison_empty_allocations(floating, integral):
    """Test-only poison of explicit Python allocations, never a runtime option.

    C++ internal allocations are not intercepted. Tests therefore supplement
    this perturbation with normal fill-on/off runs and an actual-CUDA gate.
    """
    calls = {}
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in ('empty', 'empty_like', 'empty_strided'):
            original = getattr(torch, name)
            calls[name] = 0
            def poisoned(*args, _original=original, _name=name, **kwargs):
                tensor = _original(*args, **kwargs)
                calls[_name] += 1
                tensor.fill_(floating if tensor.is_floating_point() else integral)
                return tensor
            stack.enter_context(mock.patch.object(torch, name, poisoned))
        yield calls


def finite_words(device, n=5):
    # A non-contiguous original source also exercises logical-view writes.
    storage = torch.empty_strided((2, n, 8), (n * 16, 16, 2), dtype=torch.int16, device=device)
    values = torch.linspace(-.4, .4, 2 * n * 8).reshape(2, n, 8)
    storage.copy_(float_to_words(values).to(device, torch.int16))
    return storage


def allocation_parity(test, device):
    from kv_compare.lite_finish import FinishEngine
    from tests.test_llama_causal import tiny_state
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); tiny_state(root)
        options = dict(predictor_format='bf16', values_per_stream=2, block=8)
        with allocation_policy(True):
            source = finite_words(device)
            reference = FinishEngine(root, device, **options)
            expected = reference.encode(source, start=7)
            wire = expected.export()
            test.assertFalse(expected.header['raw'])
            test.assertTrue(torch.equal(reference.decode(expected), source))
        for floating, integral in ((float('nan'), -21846), (-123.25, 21845)):
            with allocation_policy(False), poison_empty_allocations(floating, integral) as calls:
                candidate = FinishEngine(root, device, **options)
                source = finite_words(device)
                packet = candidate.encode(source, start=7)
                test.assertEqual(packet.export(), wire)
                restored = candidate.decode(Packet.restore(wire, device))
                test.assertTrue(torch.equal(restored, source))
                # All three explicit allocation entry points were perturbed.
                test.assertTrue(all(calls.values()), calls)
        # Every original BF16 pattern, including payload NaNs/signed zero,
        # follows the inherited, fully accounted raw fallback under both flags.
        patterns = torch.arange(65536, dtype=torch.int32).to(device, torch.int16).reshape(2, 4096, 8)
        wires = []
        for fill in (True, False):
            with allocation_policy(fill):
                candidate = FinishEngine(root, device, **options)
                packet = candidate.encode(patterns, start=0)
                test.assertTrue(packet.header['raw'])
                test.assertTrue(torch.equal(candidate.decode(packet), patterns))
                wires.append(packet.export())
        test.assertEqual(wires[0], wires[1])


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def setUp(self):
        self.enabled = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.enabled, warn_only=self.warn)

    def test_byte_and_word_parity_with_poisoned_allocations_and_raw_fallback(self):
        allocation_parity(self, torch.device('cpu'))


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get('TRITON_INTERPRET'), 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def setUp(self):
        self.enabled = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device('cuda', torch.cuda.current_device())

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.use_deterministic_algorithms(self.enabled, warn_only=self.warn)

    def test_actual_cuda_word_byte_and_poison_parity(self):
        allocation_parity(self, self.device)

    def test_actual_cuda_graph_both_policies_change_packets_and_own_outputs(self):
        from kv_compare.lite_finish import FinishEngine
        from kv_compare.lite_graph import GraphDecoder
        from tests.test_lite_graph import packets
        from tests.test_llama_causal import tiny_state
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            _, entries = packets(root, self.device)
            self.assertNotEqual(entries[0][0].payload.numel(), entries[1][0].payload.numel())
            # Serialized packets are created before either allocation arm;
            # each receiver has only fixed state and these unchanged packets.
            wires = [packet.export() for packet, _ in entries]
            for fill in (True, False):
                with allocation_policy(fill):
                    engine = FinishEngine(root, self.device, values_per_stream=2, block=8)
                    graph = GraphDecoder(engine, warmup=2)
                    outputs = []
                    try:
                        for i in (0, 1, 0):
                            received = Packet.restore(wires[i], self.device)
                            value = graph.decode(received); outputs.append(value)
                            self.assertTrue(torch.equal(value, entries[i][1]))
                            self.assertEqual(graph.capture_count, 1)
                        self.assertTrue(torch.equal(outputs[0], entries[0][1]))
                        self.assertTrue(torch.equal(outputs[1], entries[1][1]))
                        self.assertNotEqual(outputs[0].data_ptr(), outputs[2].data_ptr())
                    finally:
                        graph.close()


if __name__ == '__main__':
    unittest.main()
