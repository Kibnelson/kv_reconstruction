"""Software semantic and mandatory actual-CUDA partition checks."""
import os
from dataclasses import replace
import unittest
from unittest import mock

import torch

from kv_compare import mapped_decode_partition as part
from kv_compare import mapped_splitzip as mapped
from tests.test_mapped_splitzip import MODULE, oracle, tables


def make_plan(mask, device='cpu'):
    index = torch.tensor([i for i, ours in enumerate(mask) if not ours], dtype=torch.int64, device=device)
    return part.prepare_partition(mapped.prepare_mapping(index, len(mask)), mask)


class ContractTests(unittest.TestCase):
    def test_fixed_predecessor_partition_and_owned_version_guards(self):
        plan = make_plan([False, True, False, False, True, False])
        self.assertEqual(plan.critical_layers, (0, 3)); self.assertEqual(plan.bulk_layers, (2, 5))
        self.assertEqual(plan.critical_ordinals.tolist(), [0, 2])
        self.assertEqual(plan.bulk_ordinals.tolist(), [1, 3])
        self.assertEqual(plan.critical_flags.tolist(), [1, 0, 1, 0])
        self.assertEqual(plan.shared_bytes, 36)
        words = torch.zeros(6, 4, 256, dtype=torch.int16)
        plan.critical_ordinals[0] = 1
        with self.assertRaises(ValueError):part._validate_plan(plan, words)
        plan = make_plan([False, True, False, False, True, False])
        plan.mapping.tensor[0] = 2
        with self.assertRaises(ValueError):part._validate_plan(plan, words)
        plan = make_plan([False, True, False, False, True, False])
        for name in ('bulk_ordinals', 'critical_flags'):
            other = make_plan([False, True, False, False, True, False])
            getattr(other, name)[0] = 0
            with self.assertRaises(ValueError):part._validate_plan(other, words)
        with self.assertRaises(ValueError):part._validate_plan(replace(plan, versions=()), words)

    def test_rejects_wrong_mask_map_and_implicit_cpu_fallback(self):
        mapping = mapped.prepare_mapping(torch.tensor([0, 2], dtype=torch.int64), 4)
        for mask in ([False, True, True, False], [False, 1, False, True], [False, True]):
            with self.assertRaises(ValueError):part.prepare_partition(mapping, mask)
        with self.assertRaises(ValueError):part.prepare_partition(mapping.tensor, [False, True, False, True])
        plan = part.prepare_partition(mapping, [False, True, False, True])
        words = torch.zeros(4, 4, 256, dtype=torch.int16); encoded = oracle(tables(), words, mapping.tensor)
        with mock.patch.dict(os.environ, {'TRITON_INTERPRET': '0'}):
            with self.assertRaises(RuntimeError):part.decode_subset_into(tables(), encoded, plan, words, 'critical')
        with self.assertRaises(ValueError):part.decode_subset_into(tables(), encoded, plan, words, 'other')


@unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'explicit CPU interpreter required')
class InterpreterTests(unittest.TestCase):
    def test_all_bf16_words_disjoint_physical_layers_and_reverse_order(self):
        mask = [False, True, False, False, True, False]
        plan = make_plan(mask); codec = tables()
        # Every BF16 pattern appears once across the four fallback layers.
        patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(4, 64, 256)
        words = torch.full((6, 64, 256), -123, dtype=torch.int16)
        for ordinal, physical in enumerate(plan.mapping.layers):words[physical] = patterns[ordinal]
        encoded = oracle(codec, words, plan.mapping.tensor)
        original = words.clone()
        for order in (('critical', 'bulk'), ('bulk', 'critical')):
            out = torch.full_like(words, 2468)
            first = order[0]
            self.assertIs(part.decode_subset_into(codec, encoded, plan, out, first), out)
            selected = plan.critical_layers if first == 'critical' else plan.bulk_layers
            for layer in range(6):
                self.assertTrue(torch.equal(out[layer], words[layer]) if layer in selected else bool((out[layer] == 2468).all()))
            part.decode_subset_into(codec, encoded, plan, out, order[1], validated=True)
            for layer in plan.mapping.layers:self.assertTrue(torch.equal(out[layer], words[layer]))
            self.assertTrue(bool((out[[1, 4]] == 2468).all()))
        self.assertTrue(torch.equal(words, original))
        # Validated frames do not promise escape-record sorting. Reordering
        # all escape fields together must leave reconstruction unchanged.
        shuffled = replace(encoded, **{name: getattr(encoded, name).flip(0).contiguous()
            for name in ('chunk_id', 'local_pos', 'esc_val')})
        out = torch.full_like(words, 2468)
        for name in ('critical', 'bulk'):
            part.decode_subset_into(codec, shuffled, plan, out, name, validated=True)
        for layer in plan.mapping.layers:
            self.assertTrue(torch.equal(out[layer], words[layer]))

    def test_empty_partition_all_escapes_and_no_escapes(self):
        for mask in ([True, False, False], [False, True, False, True]):
            plan = make_plan(mask); codec = tables()
            for bits in (0x3f80, 0x7fc1):
                words = torch.full((len(mask), 4, 256), bits, dtype=torch.int16)
                encoded = oracle(codec, words, plan.mapping.tensor)
                out = torch.full_like(words, -55)
                part.decode_subset_into(codec, encoded, plan, out, 'critical')
                part.decode_subset_into(codec, encoded, plan, out, 'bulk')
                for layer in range(len(mask)):
                    self.assertTrue(torch.equal(out[layer], words[layer]) if not mask[layer] else bool((out[layer] == -55).all()))

    def test_rejects_malformed_escape_and_field_before_kernel(self):
        plan = make_plan([False, True, False]); codec = tables()
        words = torch.full((3, 4, 256), 0x7fc1, dtype=torch.int16)
        original = oracle(codec, words, plan.mapping.tensor)
        for field, value in (('chunk_id', -1), ('local_pos', 1024)):
            broken = replace(original, **{field:getattr(original,field).clone()})
            getattr(broken,field)[0] = value
            with self.assertRaises(ValueError):part.decode_subset_into(codec, broken, plan, words, 'critical')
        broken = replace(original, sm=original.sm[:-1])
        with self.assertRaises(ValueError):part.decode_subset_into(codec, broken, plan, words, 'bulk', validated=True)
        with self.assertRaises(ValueError):part.decode_subset_into(codec, original, plan, words, 'critical', validated=1)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TRITON_INTERPRET') != '1', 'actual CUDA required')
class CudaTests(unittest.TestCase):
    def test_actual_gpu_partitions_all_patterns_escape_modes_and_independent_streams(self):
        device = 'cuda:0'; mask = [False, True, False, False, True, False]
        plan = make_plan(mask, device); codec = tables(device)
        patterns = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).reshape(4, 64, 256)
        words = torch.zeros(6, 64, 256, dtype=torch.int16, device=device)
        for ordinal, layer in enumerate(plan.mapping.layers):words[layer] = patterns[ordinal]
        for current in (words, torch.full_like(words, 0x3f80), torch.full_like(words, 0x7fc1)):
            encoded = mapped.encode_layers(codec, MODULE, current, plan.mapping)
            packet_copy = {name:getattr(encoded,name).clone() for name in ('pk','sm','chunk_id','local_pos','esc_val')}
            out = torch.full_like(current, 2468)
            critical, bulk = torch.cuda.Stream(), torch.cuda.Stream()
            critical.wait_stream(torch.cuda.current_stream());bulk.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(critical):part.decode_subset_into(codec, encoded, plan, out, 'critical', validated=True)
            with torch.cuda.stream(bulk):part.decode_subset_into(codec, encoded, plan, out, 'bulk', validated=True)
            torch.cuda.synchronize()
            for layer in plan.mapping.layers:self.assertTrue(torch.equal(out[layer], current[layer]))
            self.assertTrue(bool((out[[1,4]] == 2468).all()))
            for name,value in packet_copy.items():self.assertTrue(torch.equal(value,getattr(encoded,name)))
            held = out.clone(); other = torch.empty_like(out)
            part.decode_subset_into(codec, encoded, plan, other, 'bulk', validated=True)
            part.decode_subset_into(codec, encoded, plan, other, 'critical', validated=True)
            torch.cuda.synchronize(); self.assertTrue(torch.equal(out,held))


if __name__ == '__main__':unittest.main()
