"""Regression checks for experiment validity, not model quality."""
import contextlib
import hashlib
import io
import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from qwen_kv_bench import runner


class RunnerTests(unittest.TestCase):
    def test_missing_torch_stops_before_self_test_discovery(self):
        original = importlib.util.find_spec
        stderr = io.StringIO()
        def absent_torch(name, *args, **kwargs):
            return None if name == "torch" else original(name, *args, **kwargs)
        with mock.patch.object(importlib.util, "find_spec", side_effect=absent_torch), \
             mock.patch("unittest.TestLoader.discover") as discover, \
             contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as result:
                runner.main(["--self-test"])
        self.assertEqual(result.exception.code, 2)
        discover.assert_not_called()
        self.assertIn("torch", stderr.getvalue())
        self.assertIn("python -m pip install torch", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_missing_torch_keeps_doctor_available(self):
        from qwen_kv_bench import model_runtime
        report = dict(selected_device=None, selection_error="PyTorch is not installed")
        stdout = io.StringIO()
        with mock.patch.object(runner, "_require_runtime_packages", side_effect=AssertionError("doctor must remain available")), \
             mock.patch.object(model_runtime, "doctor", return_value=report), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(runner.main(["--doctor"]), 2)
        self.assertIn("PyTorch is not installed", stdout.getvalue())

    def test_identical_nonfinite_logits_cannot_pass_generation(self):
        report = {"generation_checks": [], "warnings": []}
        args = SimpleNamespace(generation_steps=1, dev_tokens=16, tile_tokens=16)
        measured = dict(all_tokens_equal=True, all_logits_bitwise_equal=True,
                        all_appended_KV_bitwise_equal=True, per_step=[{"logits_finite": False}])
        rows = [dict(name="control", track="cpu_reference", status="ok",
                     measurement={"savings_including_state_percent": 0})]
        with mock.patch.object(runner, "_capacity", return_value={"can_attempt": True}), \
             mock.patch.object(runner, "_reference_callback", return_value=lambda: []), \
             mock.patch.object(runner, "_release"), \
             mock.patch("qwen_kv_bench.model_runtime.generation_check", return_value=measured):
            runner._generation(None, {"matrices": [], "first_token": 1}, rows,
                               {"control": object()}, [], None, "cpu", args, report, lambda: None)
        self.assertEqual(report["generation_checks"][0]["status"], "failed_correctness")

    def test_native_binary_uses_identical_reference_rotation(self):
        from kv_binary.binary_codec import make_state as reference_state
        model = SimpleNamespace(codebooks={"layer_00_K": np.full((64, 8), 15000, np.uint16)}, model_id="1" * 64)
        artifacts = SimpleNamespace(models={"learned64": model})
        spec = dict(method="binary", options=dict(dimension=32, transform="dense", amplitude_precision="bf16"))
        state = runner._native_state(artifacts, spec, "cpu")
        expected = reference_state(model, 32, "dense", 104729).rotation
        key = (32, "dense", 104729)
        np.testing.assert_array_equal(state.rotations[key].numpy(), expected)
        self.assertEqual(state.rotation_ids[key], hashlib.sha256(expected.tobytes()).hexdigest())

    def test_unavailable_device_fails_before_download(self):
        from qwen_kv_bench import model_runtime
        with mock.patch.object(model_runtime, "select_device", side_effect=RuntimeError("Unavailable test device")), \
             mock.patch.object(model_runtime, "load_model") as load, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                runner.main(["--device", "cuda", "--preset", "smoke"])
        self.assertEqual(result.exception.code, 2)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
