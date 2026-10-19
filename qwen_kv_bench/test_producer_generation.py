"""Actual tiny Qwen continuation from producer packets and frozen state only.

This offline random-weight fixture tests reconstruction, hook integration and
generation correctness. It does not measure pretrained-model savings or speed.
"""
import tempfile
import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_kv_bench import gpu_ops, producer_codec, producer_models
from qwen_kv_bench.model_runtime import capture, generation_check
from qwen_kv_bench.producer_hooks import ProducerCapture


def _packet_callback(encoded_records, state, layer_count, total_tokens):
    """Close over packets/state/addresses, never a producer record or hidden KV."""
    packets = tuple(encoded_records)

    def restore():
        grouped = [[] for _ in range(layer_count)]
        # Deliberately decode out of sequence: explicit record positions, not
        # list insertion order, must determine the reconstructed prefix.
        for layer, start, tokens, packet in reversed(packets):
            k, v = producer_codec.decode(gpu_ops.packet_to_device(packet, "cpu"), state)
            grouped[layer].append((start, tokens, k.cpu().numpy().copy(), v.cpu().numpy().copy()))
        result = []
        for records in grouped:
            records.sort(key=lambda record: record[0])
            cursor = 0
            for start, tokens, k, v in records:
                if start != cursor or len(k) != tokens or k.shape != v.shape:
                    raise AssertionError("producer packets have incomplete or overlapping prefix coverage")
                cursor += tokens
            if cursor != total_tokens:
                raise AssertionError("producer packet prefix is incomplete")
            result.append((np.concatenate([record[2] for record in records]),
                           np.concatenate([record[3] for record in records])))
        return result

    return restore


class ProducerGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261020)
        config = Qwen3Config(vocab_size=79, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=64, attention_dropout=0.)
        config._attn_implementation = "sdpa"
        cls.model = Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
        cls.tile_tokens = 7
        training_ids = [[1 + ((i * step + offset) % 77) for i in range(length)]
                        for step, offset, length in ((3, 2, 23), (5, 11, 24), (7, 29, 25))]
        training_records = []
        for ids in training_ids:
            captured, records, audit, _ = cls.capture_producers(ids)
            if not audit["all_stage_targets_exact"] or audit["coverage"] != "full":
                raise AssertionError("TRAIN producer hooks must match actual complete KV targets")
            training_records.append(records)
            del captured
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.models = producer_models.fit(training_records, directory.name, rank=4, seed=71,
            training_metadata=[dict(split="train", name=f"tiny_qwen_producer_train_{i}")
                               for i in range(3)],
            model_identity=dict(model="local-random-tiny-qwen3-bf16", config=config.to_dict()))
        cls.frozen_model_bytes = {kind: producer_models.model_bytes(model)
                                  for kind, model in cls.models.items()}
        # DEV is never an argument to fitting, and is captured after both model
        # artifacts are frozen. Tile7/chunk8 deliberately split record boundaries.
        cls.dev, records, cls.audit, context = cls.capture_producers(
            [1 + ((i * 11 + 47) % 77) for i in range(19)])
        cls.rope_changes_keys = any(np.any(record["k"] != record["k_pre_rope"])
                                   for record in records)
        cls.norm_changes_keys = any(np.any(record["k_proj"] != record["k_pre_rope"])
                                   for record in records)
        cls.callbacks = {}
        for spec in producer_codec.recipes():
            state = producer_codec.make_state(spec, cls.models, context, device="cpu")
            packets = []
            for record in records:
                packet = producer_codec.encode(record, spec, state)
                host = gpu_ops.packet_to_cpu(packet)
                # Reconstruct from the authoritative stored header and fields,
                # discarding the encoder's host descriptor and source tensors.
                wire = gpu_ops.packet_from_cpu_parts(host.header.clone(),
                    {name: field.clone() for name, field in host.fields.items()})
                packets.append((record["layer"], record["start"], record["tokens"], wire))
            cls.callbacks[spec["name"]] = _packet_callback(packets, state, 2, 19)
        # Future callbacks retain only encoded packets and decoder state. Clear
        # all original hidden/projection/target record mappings before invoking
        # any decoder or generation correctness check.
        for record in records:
            record.clear()
        records.clear()
        for record_set in training_records:
            for record in record_set:
                record.clear()
            record_set.clear()
        training_records.clear()

    @classmethod
    def capture_producers(cls, ids):
        with ProducerCapture(cls.model, total_tokens=len(ids), tile_tokens=cls.tile_tokens,
                             max_tiles=None) as hooks:
            captured = capture(cls.model, ids, "cpu", chunk_tokens=8)
        records, audit, context = hooks.finalize(captured["matrices"])
        return captured, records, audit, context

    def test_all_nine_producer_recipes_full_prefix_continuation(self):
        self.assertEqual(len(self.callbacks), 9)
        self.assertIn("producer_hidden_abs_exponent", self.callbacks)
        self.assertEqual(self.dev["geometry"]["tokens"], 19)
        self.assertEqual(self.dev["geometry"]["head_dim"], 16)
        self.assertEqual(self.audit["coverage"], "full")
        self.assertEqual(self.audit["selected_tile_pairs"], 6)
        self.assertEqual(self.audit["chunks_observed"], 3)
        self.assertTrue(self.audit["all_stage_targets_exact"])
        self.assertTrue(self.audit["hooks_removed"])
        self.assertTrue(self.rope_changes_keys)
        self.assertTrue(self.norm_changes_keys)
        for model in self.models.values():
            self.assertTrue(model.metadata["training_only"])
            self.assertEqual(len(model.metadata["training_record_sets"]), 3)
            self.assertEqual(len(set(model.metadata["training_hashes"])), 3)
            self.assertTrue(all(row["split"] == "train"
                                for row in model.metadata["training_record_sets"]))
        for name, callback in self.callbacks.items():
            with self.subTest(recipe=name):
                restored = callback()
                self.assertEqual(len(restored), 2)
                for original_pair, restored_pair in zip(self.dev["matrices"], restored):
                    for original, decoded in zip(original_pair, restored_pair):
                        np.testing.assert_array_equal(decoded, original)
                checked = generation_check(self.model, self.dev["matrices"], callback,
                    "cpu", first_token=self.dev["first_token"], steps=2)
                self.assertEqual(checked["steps"], 2)
                self.assertTrue(checked["all_logits_bitwise_equal"])
                self.assertTrue(checked["all_tokens_equal"])
                self.assertTrue(checked["all_appended_KV_bitwise_equal"])
                self.assertEqual(checked["max_abs_logit_difference"], 0.)
                for step in checked["per_step"]:
                    self.assertTrue(step["logits_finite"])
                    self.assertTrue(step["logits_bitwise_equal"])
                    self.assertTrue(step["tokens_equal"])
                    self.assertTrue(step["appended_full_cache_bitwise_equal"])
        for kind, model in self.models.items():
            self.assertEqual(producer_models.model_bytes(model), self.frozen_model_bytes[kind])


if __name__ == "__main__":
    unittest.main()
