"""Actual tiny-Qwen producer observations, without model downloads."""
import json
import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_kv_bench.model_runtime import capture
from qwen_kv_bench.producer_hooks import ProducerCapture


def _model(device="cpu"):
    torch.manual_seed(701)
    config = Qwen3Config(vocab_size=48, hidden_size=48, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, attention_dropout=0.)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval().to(device=device, dtype=torch.bfloat16)


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().copy()


def _hook_inventory(model):
    return {name: (len(module._forward_pre_hooks), len(module._forward_hooks))
            for name, module in model.named_modules()}


class ProducerHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_stages_chunk_stitch_and_logits_kv_are_unmodified(self):
        model = _model()
        ids = list(range(1, 20))
        stages = {i: {name: [] for name in ("hidden", "k_proj", "k_pre_rope", "v")}
                  for i in range(2)}
        cosines, sines, logits = [], [], []
        handles = []
        for index, layer in enumerate(model.model.layers):
            for module, name in ((layer.input_layernorm, "hidden"),
                    (layer.self_attn.k_proj, "k_proj"), (layer.self_attn.k_norm, "k_pre_rope"),
                    (layer.self_attn.v_proj, "v")):
                handles.append(module.register_forward_hook(lambda m, a, o, i=index, n=name:
                    stages[i][n].append(_bits(o[0].reshape(o.shape[1], -1)))))
        def observe_rotary(module, args, output):
            cosines.append(_bits(output[0][0]))
            sines.append(_bits(output[1][0]))
        handles.append(model.model.rotary_emb.register_forward_hook(observe_rotary))
        # append returns None; the outer function must also return None so it
        # cannot replace logits or feature outputs.
        def observe_logits(module, args, output):
            logits.append(_bits(output.logits))
        handles.append(model.register_forward_hook(observe_logits))
        try:
            baseline = capture(model, ids, "cpu", chunk_tokens=8)
        finally:
            for handle in handles:
                handle.remove()
        full_stages = {i: {name: np.concatenate(values) for name, values in row.items()}
                       for i, row in stages.items()}
        full_cos, full_sin = np.concatenate(cosines), np.concatenate(sines)
        observed_logits = []
        logit_handle = model.register_forward_hook(lambda m, a, o: observed_logits.append(_bits(o.logits)))
        before = _hook_inventory(model)
        try:
            with ProducerCapture(model, len(ids), tile_tokens=7, max_tiles=None) as hooks:
                observed = capture(model, ids, "cpu", chunk_tokens=8)
            self.assertEqual(_hook_inventory(model), before)
            records, audit, context = hooks.finalize(observed["matrices"])
        finally:
            logit_handle.remove()
        self.assertEqual(baseline["first_token"], observed["first_token"])
        for left, right in zip(logits, observed_logits):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(baseline["matrices"], observed["matrices"]):
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(len(records), 6)
        self.assertEqual(audit["logical_KV_records"], 12)
        self.assertEqual(audit["coverage"], "full")
        self.assertTrue(audit["all_stage_targets_exact"])
        self.assertTrue(audit["v_hook_equals_cache"])
        self.assertTrue(audit["k_rope_replay_equals_cache"])
        self.assertTrue(audit["exported_cache_equals_hook_cache"])
        self.assertTrue(audit["hooks_removed"])
        self.assertEqual(audit["hook_counts"], {"root_pre": 3, "rotary": 3, "root_post": 3})
        for row in audit["inner_stage_counts"].values():
            self.assertEqual(set(row.values()), {3})
        for record in records:
            layer, start, count = record["layer"], record["start"], record["tokens"]
            stop = start + count
            for name in ("hidden", "k_proj", "k_pre_rope", "v"):
                np.testing.assert_array_equal(record[name], full_stages[layer][name][start:stop])
            np.testing.assert_array_equal(record["cos"], full_cos[start:stop])
            np.testing.assert_array_equal(record["sin"], full_sin[start:stop])
            np.testing.assert_array_equal(record["k"], baseline["matrices"][layer][0][start:stop])
            self.assertEqual(record["head_dim"], 16)
            self.assertEqual(record["hidden"].shape, (count, 48))
            self.assertEqual(record["k_proj"].shape, (count, 32))
            self.assertEqual(record["k_pre_rope"].shape, (count, 32))
            self.assertEqual(record["cos"].shape, (count, 16))
            self.assertEqual(record["k_norm_weight"].shape, (16,))
            self.assertEqual(record["k_norm_eps"], model.config.rms_norm_eps)
            self.assertEqual(record["k"].dtype, np.uint16)
            self.assertFalse(record["hidden"].flags.writeable)
        self.assertNotEqual(model.config.hidden_size // model.config.num_attention_heads, audit["head_dim"])
        self.assertEqual(context["rope_inv_freq"].dtype, np.float32)
        self.assertEqual(context["rope_inv_freq"].shape, (8,))
        self.assertTrue(context["rope_stable"])
        self.assertEqual(set(context["norm_weights"]), {0, 1})
        np.testing.assert_array_equal(context["rope_inv_freq"], model.model.rotary_emb.inv_freq.float().numpy())
        json.dumps(audit)

    def test_selection_is_fixed_bounded_and_final_tail_is_preserved(self):
        model = _model()
        ids = [i % 47 + 1 for i in range(33)]
        before = _hook_inventory(model)
        with ProducerCapture(model, 33, tile_tokens=7, max_tiles=2) as hooks:
            captured = capture(model, ids, "cpu", chunk_tokens=8)
            records, audit, context = hooks.finalize(captured["matrices"])
            self.assertEqual(_hook_inventory(model), before)
        self.assertEqual([(r["layer"], r["start"], r["tokens"]) for r in records],
            [(0, 0, 7), (1, 28, 5)])
        self.assertEqual(audit["selected_tile_pairs"], 2)
        self.assertEqual(audit["total_tile_pairs"], 10)
        self.assertEqual(audit["coverage"], "sampled")
        self.assertTrue(audit["all_stage_targets_exact"])
        self.assertLess(sum(r["tokens"] for r in records), 33 * 2)
        self.assertFalse(audit["raw_hidden_features_persisted"])
        self.assertEqual(set(context["norm_weights"]), {0, 1})
        self.assertTrue(all(not record["_buffers"] for record in hooks._records))

    def test_callback_builds_exact_recipes_inside_attention_before_model_return(self):
        from qwen_kv_bench import gpu_ops as g
        model = _model()
        events, encoded = [], []
        root_hook = model.register_forward_hook(lambda m, a, o: events.append(("root_return",)))
        next_layer = model.model.layers[1].input_layernorm.register_forward_hook(
            lambda m, a, o: events.append(("layer1_norm",)))
        before = _hook_inventory(model)
        def on_tile(record, context):
            events.append(("recipe", record["layer"], record["start"]))
            self.assertIsNotNone(hooks._frame)
            self.assertEqual(set(context["norm_weights"]), {0, 1})
            self.assertEqual(context["rope_inv_freq"].shape, (8,))
            self.assertTrue(context["rope_stable"])
            for kind in ("k", "v"):
                words = torch.from_numpy(record[kind].copy())
                packet = g.encode(words, method="prefill", predictor="midpoint",
                    correction="residual", group_size=8)
                self.assertTrue(torch.equal(g.decode(packet).to(torch.int32), words.to(torch.int32)))
                encoded.append((record["layer"], record["start"], kind, packet.physical_nbytes))
            self.assertTrue(all(name in record for name in
                ("hidden", "k_proj", "k_pre_rope", "cos", "sin", "k_norm_weight")))
        try:
            with ProducerCapture(model, 9, tile_tokens=7, max_tiles=None, on_tile=on_tile) as hooks:
                captured = capture(model, list(range(1, 10)), "cpu", chunk_tokens=5)
            self.assertEqual(_hook_inventory(model), before)
            records, audit, context = hooks.finalize(captured["matrices"])
        finally:
            root_hook.remove()
            next_layer.remove()
        self.assertEqual(len(encoded), 8)
        self.assertEqual(audit["on_tile_callback_count"], 4)
        self.assertEqual(audit["selected_tiles_ready_inside_attention"], 4)
        self.assertGreater(audit["on_tile_callback_wall_seconds"], 0)
        self.assertIn("attention_forward_post_hook", audit["on_tile_stage"])
        first_return = events.index(("root_return",))
        self.assertFalse(any(event[0] == "recipe" for event in events[:first_return]))
        # Both layer-0 tiles complete during chunk two, before layer one's
        # normalization resumes, and every recipe precedes model return.
        second_norm = [i for i, event in enumerate(events) if event == ("layer1_norm",)][1]
        self.assertLess(events.index(("recipe", 0, 0)), second_norm)
        self.assertLess(events.index(("recipe", 0, 7)), second_norm)
        self.assertLess(max(i for i, event in enumerate(events) if event[0] == "recipe"), len(events) - 1)
        self.assertEqual(events[-1], ("root_return",))
        self.assertTrue(audit["all_stage_targets_exact"])
        self.assertEqual(len(records), 4)

    def test_callback_exception_removes_hooks_and_preserves_failure(self):
        model = _model()
        before = _hook_inventory(model)
        def fail(record, context):
            raise RuntimeError("intentional inline recipe failure")
        hooks = ProducerCapture(model, 3, tile_tokens=7, max_tiles=None, on_tile=fail)
        with self.assertRaisesRegex(RuntimeError, "intentional inline recipe failure"):
            with hooks:
                capture(model, [1, 2, 3], "cpu", chunk_tokens=3)
        self.assertEqual(_hook_inventory(model), before)
        self.assertFalse(hooks._handles)

    def test_failure_removes_only_our_hooks_and_preserves_original_exception(self):
        model = _model()
        def fail(module, args):
            raise RuntimeError("intentional projection failure")
        existing = model.model.layers[0].self_attn.k_proj.register_forward_pre_hook(fail)
        before = _hook_inventory(model)
        hooks = ProducerCapture(model, 19, tile_tokens=7, max_tiles=None)
        try:
            with self.assertRaisesRegex(RuntimeError, "intentional projection failure"):
                with hooks:
                    capture(model, list(range(1, 20)), "cpu", chunk_tokens=8)
            self.assertEqual(_hook_inventory(model), before)
            self.assertFalse(hooks._handles)
            with self.assertRaisesRegex(ValueError, "failed, incomplete"):
                hooks.finalize([])
        finally:
            existing.remove()

    def test_reject_incomplete_batch_or_noncontiguous_positions_with_cleanup(self):
        model = _model()
        before = _hook_inventory(model)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            with ProducerCapture(model, 19, tile_tokens=7, max_tiles=None):
                capture(model, [1, 2, 3], "cpu", chunk_tokens=8)
        self.assertEqual(_hook_inventory(model), before)
        for kwargs, message in ((dict(input_ids=torch.ones((2, 3), dtype=torch.long)), "batch size one"),
                (dict(input_ids=torch.ones((1, 3), dtype=torch.long),
                    position_ids=torch.tensor([[0, 2, 3]])), "contiguous")):
            with self.assertRaisesRegex(ValueError, message):
                with ProducerCapture(model, 3, tile_tokens=7, max_tiles=None):
                    with torch.inference_mode():
                        model(**kwargs, use_cache=True)
            self.assertEqual(_hook_inventory(model), before)

    def test_replay_mismatch_is_reported_and_never_silently_passes(self):
        model = _model()
        hooks = ProducerCapture(model, 9, tile_tokens=7, max_tiles=None)
        actual = hooks._rotate[0]
        def wrong_replay(*args):
            query, key = actual(*args)
            bits = key.contiguous().view(torch.int16).clone()
            bits.reshape(-1)[0] ^= 1
            return query, bits.view(torch.bfloat16)
        hooks._rotate[0] = wrong_replay
        with hooks:
            captured = capture(model, list(range(1, 10)), "cpu", chunk_tokens=5)
        records, audit, context = hooks.finalize(captured["matrices"])
        self.assertFalse(audit["all_stage_targets_exact"])
        self.assertFalse(audit["k_rope_replay_equals_cache"])
        self.assertTrue(audit["v_hook_equals_cache"])
        self.assertTrue(audit["exported_cache_equals_hook_cache"])
        self.assertGreater(sum(r["k_rope_replay_cache_mismatched_words"] for r in audit["selected_records"]), 0)
        self.assertEqual(len(records), 4)
        self.assertTrue(context["rope_stable"])

    def test_changing_rotary_context_is_detected_even_when_actual_k_replays(self):
        model = _model()
        chunks = []
        def change_context(module, args, kwargs):
            if chunks:
                module.model.rotary_emb.inv_freq.mul_(1.01)
            chunks.append(1)
        existing = model.register_forward_pre_hook(change_context, with_kwargs=True)
        try:
            with ProducerCapture(model, 9, tile_tokens=7, max_tiles=None) as hooks:
                captured = capture(model, list(range(1, 10)), "cpu", chunk_tokens=5)
            records, audit, context = hooks.finalize(captured["matrices"])
        finally:
            existing.remove()
        self.assertTrue(audit["all_stage_targets_exact"])
        self.assertFalse(audit["rope_stable"])
        self.assertFalse(context["rope_stable"])
        self.assertEqual(len(records), 4)

    def test_unsupported_models_parameters_dtypes_and_reentry(self):
        with self.assertRaisesRegex(ValueError, "Qwen3ForCausalLM"):
            ProducerCapture(torch.nn.Linear(3, 3), 3)
        model = _model()
        for kwargs in (dict(total_tokens=0), dict(total_tokens=3, tile_tokens=0),
                dict(total_tokens=3, max_tiles=1), dict(total_tokens=3, max_tiles=True)):
            with self.assertRaises(ValueError):
                ProducerCapture(model, **kwargs)
        float_model = model.to(dtype=torch.float32)
        before = _hook_inventory(float_model)
        with self.assertRaisesRegex(ValueError, "actual BF16"):
            with ProducerCapture(float_model, 3):
                pass
        self.assertEqual(_hook_inventory(float_model), before)
        model = _model()
        hooks = ProducerCapture(model, 3)
        with hooks:
            captured = capture(model, [1, 2, 3], "cpu", chunk_tokens=3)
        hooks.finalize(captured["matrices"])
        with self.assertRaisesRegex(ValueError, "previously finalized"):
            hooks.finalize(captured["matrices"])
        with self.assertRaisesRegex(RuntimeError, "twice"):
            hooks.__enter__()

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA hardware; producer GPU replay requires an actual GPU")
    def test_cuda_original_device_replay_and_unmodified_model(self):
        model = _model("cuda")
        ids = list(range(1, 20))
        baseline = capture(model, ids, "cuda", chunk_tokens=8)
        with ProducerCapture(model, len(ids), tile_tokens=7, max_tiles=None) as hooks:
            observed = capture(model, ids, "cuda", chunk_tokens=8)
        records, audit, context = hooks.finalize(observed["matrices"])
        self.assertTrue(audit["all_stage_targets_exact"])
        self.assertEqual(baseline["first_token"], observed["first_token"])
        for left, right in zip(baseline["matrices"], observed["matrices"]):
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(len(records), 6)
        self.assertTrue(context["rope_stable"])


if __name__ == "__main__":
    unittest.main()
