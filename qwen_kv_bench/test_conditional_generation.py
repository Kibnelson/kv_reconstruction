"""Actual-forward integration for every conditional KV idea and its controls.

Three tiny BF16 Qwen TRAIN captures fit frozen artifacts. A fourth, full DEV
prefix is encoded inside attention and independently afterward. Reconstruction
retains only authoritative packets, decoder state and addresses; all original
producer record mappings are cleared. This verifies functionality, not real
model savings, the forty-percent target, or GPU performance.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_kv_bench import conditional_codec as codec
from qwen_kv_bench import conditional_models, gpu_ops
from qwen_kv_bench.model_runtime import capture, generation_check
from qwen_kv_bench.producer_hooks import ProducerCapture
from qwen_kv_bench.producer_study import _Inline


def _restore_callback(encoded_records, state, layers, total_tokens):
    """No target arrays or producer records are visible in this closure."""
    packets = tuple(encoded_records)

    def restore():
        grouped = [[] for _ in range(layers)]
        for layer, start, tokens, packet in reversed(packets):
            k, v = codec.decode(gpu_ops.packet_to_device(packet, "cpu"), state)
            grouped[layer].append((start, tokens, k.numpy().copy(), v.numpy().copy()))
        result = []
        for records in grouped:
            records.sort(key=lambda row: row[0])
            cursor = 0
            for start, tokens, k, v in records:
                if start != cursor or len(k) != tokens or k.shape != v.shape:
                    raise AssertionError("Conditional packet prefix has missing or overlapping rows")
                cursor += tokens
            if cursor != total_tokens:
                raise AssertionError("Conditional packet prefix is incomplete")
            result.append((np.concatenate([row[2] for row in records]),
                           np.concatenate([row[3] for row in records])))
        return result

    return restore


class ConditionalGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261022)
        config = Qwen3Config(vocab_size=79, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=64, attention_dropout=0.)
        config._attn_implementation = "sdpa"
        cls.model = Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
        cls.tile_tokens = 7
        train_ids = [[1 + ((i * step + offset) % 77) for i in range(length)]
                     for step, offset, length in ((3, 2, 23), (5, 11, 24), (7, 29, 25))]
        training = []
        for ids in train_ids:
            captured, records, audit, _ = cls.capture_producers(ids, split="train")
            if not audit["all_stage_targets_exact"] or audit["coverage"] != "full":
                raise AssertionError("Actual TRAIN producer targets must be complete and exact")
            training.append(records)
            del captured
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.models = conditional_models.fit(training, directory.name, rank=4, seed=71,
            training_metadata=[dict(split="train", name=f"tiny_conditional_train_{i}") for i in range(3)],
            model_identity=dict(model="local-random-tiny-qwen3-conditional", config=config.to_dict()))
        cls.frozen = {name: (model.model_id, Path(model.path).read_bytes()) for name, model in cls.models.items()}
        cls.specs = tuple(codec.recipes())
        inline = _Inline(cls.specs, cls.models, "cpu", codec=codec)
        cls.dev, records, cls.audit, context = cls.capture_producers(
            [1 + ((i * 11 + 47) % 77) for i in range(19)], split="dev", on_tile=inline)
        cls.inline = inline.report()
        cls.rope_changes_keys = any(np.any(record["k"] != record["k_pre_rope"]) for record in records)
        cls.callbacks, cls.record_sizes, cls.states, cls.encoded = {}, {}, {}, {}
        for spec in cls.specs:
            state = codec.make_state(spec, cls.models, context, device="cpu")
            encoded, sizes = [], {}
            for record in records:
                packet = codec.encode(record, spec, state)
                stats = packet.stats
                key = (record["layer"], record["start"], record["tokens"])
                sizes[key] = (stats["serialized_bytes"], stats["component_bytes"])
                host = gpu_ops.packet_to_cpu(packet)
                # Discard encoder host descriptors/analysis sidecars. Only
                # independently parsed authoritative headers/fields survive.
                wire = gpu_ops.packet_from_cpu_parts(host.header.clone(),
                    {name: field.clone() for name, field in host.fields.items()})
                encoded.append((*key, wire))
            cls.states[spec["name"]] = state
            cls.encoded[spec["name"]] = tuple(encoded)
            cls.record_sizes[spec["name"]] = sizes
            cls.callbacks[spec["name"]] = _restore_callback(encoded, state, 2, 19)
        for record in records:
            record.clear()
        records.clear()
        for record_set in training:
            for record in record_set:
                record.clear()
            record_set.clear()
        training.clear()
        del inline, context

    @classmethod
    def capture_producers(cls, ids, *, split, on_tile=None):
        with ProducerCapture(cls.model, len(ids), tile_tokens=cls.tile_tokens,
                             max_tiles=None, on_tile=on_tile) as hooks:
            captured = capture(cls.model, ids, "cpu", chunk_tokens=8)
        records, audit, context = hooks.finalize(captured["matrices"])
        for record in records:
            record["split"] = split
        return captured, records, audit, context

    def test_all_ideas_and_controls_restore_complete_DEV_and_continue(self):
        names = {spec["name"] for spec in self.specs}
        self.assertGreaterEqual(len(names), 12)
        self.assertTrue({"conditional_huffman_exponent", "conditional_huffman_exponent_sign",
            "conditional_tier_sign", "conditional_mean_predictor", "conditional_raw_fullword",
            "conditional_anchor_r8", "conditional_anchor_r32", "conditional_latent_r8",
            "conditional_latent_r32"}.issubset(names))
        self.assertEqual(names, set(codec.generation_representatives()))
        self.assertEqual(self.dev["geometry"]["head_dim"], 16)
        self.assertNotEqual(self.model.config.hidden_size // self.model.config.num_attention_heads, 16)
        self.assertTrue(self.rope_changes_keys)
        for name, callback in self.callbacks.items():
            with self.subTest(recipe=name):
                restored = callback()
                for actual, expected in zip(restored, self.dev["matrices"]):
                    for got, original in zip(actual, expected):
                        np.testing.assert_array_equal(got, original)
                check = generation_check(self.model, self.dev["matrices"], callback,
                    "cpu", first_token=self.dev["first_token"], steps=2)
                self.assertEqual(check["steps"], 2)
                self.assertTrue(check["all_logits_bitwise_equal"])
                self.assertTrue(check["all_tokens_equal"])
                self.assertTrue(check["all_appended_KV_bitwise_equal"])
                self.assertEqual(check["max_abs_logit_difference"], 0.)
                self.assertTrue(all(row["logits_finite"] for row in check["per_step"]))

    def test_true_forward_inline_bytes_state_and_model_freeze_match(self):
        self.assertTrue(self.audit["all_stage_targets_exact"])
        self.assertTrue(self.audit["hooks_removed"])
        self.assertEqual(self.audit["coverage"], "full")
        self.assertEqual(self.audit["selected_tile_pairs"], 6)
        self.assertEqual(self.audit["on_tile_callback_count"], 6)
        self.assertEqual(self.audit["chunks_observed"], 3)
        self.assertIn("attention_forward_post_hook", self.audit["on_tile_stage"])
        observed = {row["name"]: row for row in self.inline["methods"]}
        self.assertEqual(set(observed), set(self.callbacks))
        for name, row in observed.items():
            with self.subTest(recipe=name):
                self.assertEqual(row["status"], "ok", row.get("error"))
                self.assertEqual(row["records_checked"], 6)
                self.assertEqual(row["state_id"], self.states[name].state_id)
                sizes = {(t["layer"], t["start"], t["tokens"]): (t["serialized_bytes"], t["component_bytes"])
                         for t in row["records"]}
                self.assertEqual(sizes, self.record_sizes[name])
                self.assertEqual(row["serialized_record_bytes"], sum(value[0] for value in sizes.values()))
                state = self.states[name]
                self.assertEqual(state.state_bytes, max(state.serialized_bytes, state.physical_nbytes))
        for name, model in self.models.items():
            self.assertEqual((model.model_id, Path(model.path).read_bytes()), self.frozen[name])
            self.assertTrue(model.metadata["training_only"])
            sets = model.metadata.get("training_record_sets", model.metadata.get("training_cases"))
            self.assertEqual(len(sets), 3)
            self.assertTrue(all(row["split"] == "train" for row in sets))

    def test_missing_packet_cannot_use_original_inputs_to_fill_prefix(self):
        name = "conditional_huffman_exponent_sign"
        callback = _restore_callback(self.encoded[name][:-1], self.states[name], 2, 19)
        with self.assertRaisesRegex(AssertionError, "incomplete|missing"):
            callback()


if __name__ == "__main__":
    unittest.main()
