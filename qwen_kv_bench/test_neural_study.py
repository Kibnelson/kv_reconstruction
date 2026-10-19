"""Study gates and independently restored frozen control packets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from qwen_kv_bench import neural_study as study
from qwen_kv_bench.fixtures import DEV, TEST, TRAIN
from qwen_kv_bench.test_fixtures import FakeTokenizer


class NeuralStudyTests(unittest.TestCase):
    @staticmethod
    def _args(*arguments):
        args = study._parser().parse_args(list(arguments))
        study._defaults(args)
        return args

    def test_dev_defaults_preserve_tiny_mode_and_explicit_budget(self):
        self.assertEqual(self._args().dev_tokens, 128)
        tiny = self._args("--tiny")
        self.assertEqual(tiny.dev_tokens, 8)
        self.assertEqual(tiny.device, "cpu")
        # An explicitly impossible budget must be rejected by preflight,
        # rather than silently changing the requested experiment geometry.
        self.assertEqual(self._args("--dev-tokens", "16").dev_tokens, 16)

    def test_input_preflight_covers_requested_jobs_and_preserves_full_chat(self):
        tokenizer = FakeTokenizer()
        args = self._args("--train-tokens", "512", "--dev-tokens", "512",
                          "--lengths", "512", "1024", "--cases", "2")
        original = study.build_input_ids
        observed = []

        def build(tokenizer_arg, fixture, target_tokens):
            self.assertIs(tokenizer_arg, tokenizer)
            ids = original(tokenizer_arg, fixture, target_tokens)
            observed.append((fixture, target_tokens, ids))
            return ids

        with mock.patch.object(study, "build_input_ids", side_effect=build):
            result = study._preflight_inputs(tokenizer, args)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["kind"], "chat_template")
        expected = [(fixture["split"], fixture["name"], "--train-tokens", 512) for fixture in TRAIN]
        expected += [(DEV[0]["split"], DEV[0]["name"], "--dev-tokens", 512)]
        expected += [(fixture["split"], fixture["name"], "--lengths", length)
                     for length in (512, 1024) for fixture in TEST[:2]]
        self.assertEqual([(row["split"], row["name"], row["argument"], row["target_tokens"])
                          for row in result["prompts"]], expected)
        self.assertEqual(len(observed), len(expected))
        for row, (fixture, target, ids) in zip(result["prompts"], observed):
            self.assertEqual(len(ids), target)
            self.assertEqual(row["input_ids_sha256"], study._hash(ids))
            text = tokenizer.decode(ids)
            self.assertIn(fixture["prefix"] + "\n\n", text)
            self.assertIn("\n\n" + fixture["suffix"], text)
            if fixture.get("needle"):
                self.assertIn(fixture["needle"], text)
            self.assertEqual(text.count("<|im_start|>"), 3)
            self.assertTrue(text.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertNotIn(TEST[2]["name"], [row["name"] for row in result["prompts"]])

    def test_tiny_input_preflight_does_not_require_a_tokenizer(self):
        args = self._args("--tiny")
        with mock.patch.object(study, "build_input_ids", side_effect=AssertionError("tiny must use synthetic IDs")) as builder:
            result = study._preflight_inputs(None, args)
        builder.assert_not_called()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["kind"], "synthetic")
        self.assertEqual(result["prompts"], [])

    def test_invalid_dev_or_test_budget_fails_before_capture_and_fitting(self):
        for argument, dev_tokens, test_tokens, fixture_name in (
            ("--dev-tokens", 16, 512, DEV[0]["name"]),
            ("--lengths", 512, 16, TEST[0]["name"]),
        ):
            with self.subTest(argument=argument), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                stages = []

                def load_model(*args, **kwargs):
                    validator = kwargs.get("tokenizer_validator")
                    self.assertTrue(callable(validator), "preflight must run inside loader before weights")
                    stages.append("tokenizer_ready")
                    validator(FakeTokenizer())
                    stages.append("weights_would_load")
                    raise AssertionError("invalid input budgets must fail before weights")

                with mock.patch.object(study.runtime, "load_model", side_effect=load_model), \
                     mock.patch.object(study.runtime, "doctor", return_value={}), \
                     mock.patch.object(study.runtime, "capture") as capture, \
                     mock.patch("qwen_kv_bench.neural_probability.fit") as fit, \
                     mock.patch.object(study, "_note"):
                    status = study.main(["--device", "cpu", "--train-tokens", "512",
                        "--dev-tokens", str(dev_tokens), "--lengths", str(test_tokens),
                        "--cases", "1", "--steps", "1", "--width", "16", "--output", str(output)])
                self.assertEqual(status, 1)
                capture.assert_not_called()
                fit.assert_not_called()
                self.assertEqual(stages, ["tokenizer_ready"])
                report = json.loads((output / "results.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertIn("Input preflight failed", report["error"])
                self.assertIn(argument + "=16", report["error"])
                self.assertIn(fixture_name, report["error"])
                self.assertIn("fixed template/user tokens", report["error"])
                self.assertFalse((output / "models").exists())

    def test_continuation_gate_requires_all_finite_bitwise_evidence(self):
        good = dict(all_logits_bitwise_equal=True, all_tokens_equal=True,
                    all_appended_KV_bitwise_equal=True, per_step=[dict(logits_finite=True)])
        self.assertTrue(study._generation_passed(good))
        for key in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal"):
            bad = copy.deepcopy(good)
            bad[key] = False
            self.assertFalse(study._generation_passed(bad))
            del bad[key]
            self.assertFalse(study._generation_passed(bad))
        for step in ({}, {"logits_finite": False}):
            self.assertFalse(study._generation_passed(dict(good, per_step=[step])))
        self.assertFalse(study._generation_passed({}))
        self.assertFalse(study._generation_passed(dict(good, per_step=[])))

    def test_frozen_control_real_container_and_train_only(self):
        # Include signed zero, subnormal, NaN payloads and infinities: anchor
        # coding is bit exact even though neural conditioning needs finite KV.
        words = np.array([[0, 0x8000], [1, 0x7fc1], [0x7f80, 0xff80]], np.uint16)
        captures = [[(words, words[:, ::-1].copy()), (words ^ np.uint16(0x123), words.copy())]]
        state = study.fit_anchor(captures, [dict(split="train", name="fixture")], "test", "cpu")
        result = study._baseline(captures[0], state, [0, 2], 2, 2)
        self.assertTrue(result["container_roundtrip_verified"])
        self.assertEqual(result["raw_sample_bytes"], 2 * 2 * words.nbytes)
        self.assertEqual(result["records"], 4)
        self.assertEqual(result["packet_bytes"], result["inner_packet_bytes"] + result["indexed_envelope_bytes"])
        with self.assertRaises(ValueError):
            study.fit_anchor(captures, [dict(split="test")], "test", "cpu")


if __name__ == "__main__":
    unittest.main()
