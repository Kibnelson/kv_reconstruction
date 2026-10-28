"""Software parity for the isolated optimization; actual CUDA has its own test."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
    raise unittest.SkipTest("PyTorch and Triton required")
import torch
from kv_compare.causal_packet import Engine, Packet, float_to_words, word_sha, write_json
from kv_compare.core import Rope
from kv_compare.lite_fast import FastEngine, RequestRope
from scripts import bench_lite_fast as bench
from scripts import package_lite_fast as package
from tests.test_llama_causal import tiny_state


class RopeTests(unittest.TestCase):
    def test_unindexed_cuda_key_uses_the_current_explicit_index(self):
        fast = RequestRope([1., .01], 1, 4)
        with mock.patch.object(torch.cuda, "current_device", return_value=2) as current:
            with fast.request(7, 5, "cuda"):
                self.assertEqual(fast._request_key, ("cuda:2", 7, 11, 5))
        current.assert_called_once_with()

    def test_same_float64_tables_and_rotations_for_changed_ranges_layouts_and_scales(self):
        for layout in ("half", "adjacent"):
            for scale in (1., 1.25):
                ref = Rope([1., .01], 2, 4, layout, scale)
                fast = RequestRope([1., .01], 2, 4, layout, scale)
                for start, n in ((0, 3), (7, 5), (2, 3), (8190, 2), (7, 5)):
                    positions = torch.arange(start, start + n)
                    x = torch.arange(n * 8, dtype=torch.float64).reshape(n, 8) / 11
                    with fast.request(start, n, positions.device):
                        for inverse in (False, True):
                            self.assertTrue(torch.equal(ref.rotate_heads(x, positions, inverse),
                                                        fast.rotate_heads(x, positions, inverse)))
                        a = fast.tables(positions)
                        b = fast.tables(positions)
                        self.assertIs(a, b)
                        self.assertEqual(next(iter(fast._tables)), ("cpu", start, start + n - 1, n))
                self.assertIsNone(fast._request_key)

    def test_bound_lookup_never_converts_position_tensor_to_python_integer(self):
        fast = RequestRope([1., .01], 1, 4)
        positions = torch.arange(7, 12)
        with fast.request(7, 5, positions.device), mock.patch.object(torch.Tensor, "__int__",
                side_effect=AssertionError("device scalar readback")):
            first = fast.tables(positions)   # cache miss
            self.assertIs(fast.tables(positions), first)  # cache hit
        with mock.patch.object(torch.Tensor, "__int__", side_effect=AssertionError("reference readback")):
            with self.assertRaisesRegex(AssertionError, "reference readback"):
                Rope([1., .01], 1, 4).tables(positions)

    def test_request_guard_metadata_and_error_cleanup(self):
        fast = RequestRope([1., .01], 1, 4)
        with self.assertRaisesRegex(ValueError, "metadata"):
            with fast.request(0, 3, torch.device("cpu")):
                fast.tables(torch.arange(4))
        self.assertIsNone(fast._request_key)
        with fast.request(0, 3, torch.device("cpu")):
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                with fast.request(1, 2, torch.device("cpu")):
                    pass
        positions = torch.arange(3)
        for a, b in zip(fast.tables(positions), Rope([1., .01], 1, 4).tables(positions)):
            self.assertTrue(torch.equal(a, b))


def packet_parity(test, device):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); tiny_state(root)
        for fmt in ("fp32", "bf16"):
            options = dict(predictor_format=fmt, values_per_stream=2, block=8)
            ref, fast = Engine(root, device, **options), FastEngine(root, device, **options)
            for start, n in ((0, 3), (7, 5), (2, 3)):
                gen = torch.Generator().manual_seed(start + n)
                words = float_to_words(torch.randn(2, n, 8, generator=gen) * .2).to(device, torch.int16)
                for kappa in (0., .5, 1.):
                    with test.subTest(device=str(device), fmt=fmt, start=start, kappa=kappa):
                        p = ref.encode(words, start=start, kappa=kappa)
                        q = fast.encode(words, start=start, kappa=kappa)
                        test.assertEqual(p.export(), q.export())
                        intended = ref.last_reconstruction.clone()
                        test.assertTrue(torch.equal(fast.last_reconstruction, intended))
                        test.assertTrue(torch.equal(fast.decode(Packet.restore(p.export(), device)), intended))
                        test.assertTrue(torch.equal(ref.decode(Packet.restore(q.export(), device)), intended))
                        if not kappa:
                            test.assertTrue(torch.equal(intended, words))
                        test.assertIsNone(fast.rope._request_key)


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU interpreter required")
class InterpreterTests(unittest.TestCase):
    def test_bidirectional_packet_compatibility_lossless_and_lossy_after_range_changes(self):
        packet_parity(self, torch.device("cpu"))

    def test_calibration_raw_fallback_and_failed_run_do_not_change_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            options = dict(predictor_format="bf16", values_per_stream=2, block=8)
            ref, fast = Engine(root, "cpu", **options), FastEngine(root, "cpu", **options)
            words = float_to_words(torch.ones(2, 3, 8) * .1).to(torch.int16)
            self.assertTrue(torch.equal(ref.calibrate(words), fast.calibrate(words)))
            with mock.patch.object(Engine, "_run", side_effect=RuntimeError("fixture failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    fast.encode(words)
            self.assertIsNone(fast.rope._request_key)
            self.assertEqual(ref.encode(words).export(), fast.encode(words).export())
            patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(2, 4096, 8)
            p = ref.encode(patterns); q = fast.encode(patterns)
            self.assertTrue(q.header["raw"])
            self.assertEqual(p.export(), q.export())
            self.assertTrue(torch.equal(fast.decode(q), patterns))

    def test_paired_measurements_and_fresh_receiver_on_actual_software_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            options = dict(predictor_format="bf16", values_per_stream=2, block=8)
            engines = {"reference": Engine(root, "cpu", **options), "host_rope": FastEngine(root, "cpu", **options)}
            words = float_to_words(torch.arange(48).reshape(2, 3, 8).float() / 100).to(torch.int16)
            entries = []
            for kappa in (0., .5, 1.):
                p = engines["reference"].encode(words, start=9, kappa=kappa)
                wire = p.export(); path = root / f"k{kappa:g}.kvc"; path.write_bytes(wire)
                old = dict(id=f"tiny_k{kappa:g}", document="tiny", original_word_sha256=word_sha(words),
                           decoded_word_sha256=word_sha(engines["reference"].last_reconstruction))
                row = bench.compare_case(engines, words, wire, old, SimpleNamespace(warmup=1, repeats=2),
                                         torch, torch.device("cpu"))
                self.assertTrue(row["packet_matches_original"])
                self.assertEqual(row["sample_order"]["encode"],
                    [["reference", "host_rope"], ["host_rope", "reference"], ["reference", "host_rope"]])
                self.assertEqual(len(row["decode"]["host_rope"]["samples"]), 2)
                entries.append(dict(id=old["id"], packet=str(path),
                    packet_sha256=hashlib.sha256(wire).hexdigest(), expected_word_sha256=old["decoded_word_sha256"]))
            write_json(root / "request.json", dict(state_dir=str(root), engine_options=options,
                entries=entries, output=str(root / "proof.json")))
            proc = subprocess.run([sys.executable, str(Path(bench.__file__)), "--device", "cpu",
                "--software-check", "--verify-request", str(root / "request.json")],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            proof = json.loads((root / "proof.json").read_text())
            self.assertFalse(proof["gpu_measured"])
            self.assertTrue(proof["no_source_kv_or_prediction_arrays"])
            self.assertEqual(len(proof["checks"]), 3)


class PackagingTests(unittest.TestCase):
    def test_completed_report_checks_original_packet_and_receiver_proof(self):
        # Protocol fixture only, no GPU timing or performance claim. The actual
        # software packet/reconstruction path is exercised above separately.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / "run"; run.mkdir()
            base_dir = root / "base"; base_dir.mkdir()
            packet = base_dir / "original.kvc"; packet.write_bytes(b"protocol fixture")
            old = dict(id="fixture", policy="lite", document="tiny", kappa=0., packet="original.kvc",
                packet_sha256=package.sha(packet), packet_bytes=packet.stat().st_size, values=8,
                decoded_word_sha256="words", original_word_sha256="words", state_sha256="state", original_exact=True)
            write_json(run / "base_results.json", dict(rows=[old]))
            recipe = dict(base_run=str(base_dir), base_results_sha256=package.sha(run / "base_results.json"),
                          warmup=1, repeats=2, profile=False)
            proof = dict(gpu_measured=True, no_source_kv_or_prediction_arrays=True, checks=[
                dict(id="fixture", verified=True, packet_sha256=old["packet_sha256"], word_sha256="words", shape=[1, 2, 4])])
            write_json(run / "independent_replay.json", proof)
            row = dict(old, shape=[1, 2, 4], status="matched", packet_matches_original=True,
                       complete_bpv=8 * old["packet_bytes"] / 8)
            row.update({d: {a: dict(samples=[{}, {}], warmup_samples=[{}]) for a in bench.ARMS}
                        for d in ("encode", "decode")})
            result = dict(status="completed", gpu_measured=True, recipe=recipe,
                identity=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                receiver_sha256=package.sha(run / "independent_replay.json"), rows=[row], selected_ids=["fixture"], profiles={})
            package.validate(run, result)
            packet.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "packet changed"):
                package.validate(run, result)
            packet.write_bytes(b"protocol fixture")
            (run / "independent_replay.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "receiver proof changed"):
                package.validate(run, result)

    def test_cpu_cannot_publish_success_and_partial_reports_exclude_tensors_and_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / "run"; run.mkdir()
            result = dict(status="completed", gpu_measured=False)
            write_json(run / "results.json", result)
            with self.assertRaisesRegex(ValueError, "CPU evidence"):
                package.main([str(run), "--destination", str(root / "bad")])
            result["status"] = "failed"; write_json(run / "results.json", result)
            (run / "summary.md").write_text("CPU software fixture only\n")
            (run / "model.npy").write_bytes(b"local only")
            (run / "traces").mkdir(); (run / "traces/huge.json.gz").write_bytes(b"local only")
            self.assertEqual(package.main([str(run), "--destination", str(root / "report")]), 0)
            self.assertEqual({p.name for p in (root / "report").iterdir()},
                             {"results.json", "summary.md", "package_manifest.json"})
            (run / "console.log").symlink_to(run / "model.npy")
            with self.assertRaisesRegex(ValueError, "outside-run"):
                package.main([str(run), "--destination", str(root / "symlink")])

    def test_real_gpu_request_fails_without_creating_a_cpu_benchmark(self):
        if torch.cuda.is_available():
            self.skipTest("requires CPU-only host")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, str(Path(bench.__file__)), "--output", tmp],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 1)
            result = json.loads((Path(tmp) / "results.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["gpu_measured"])
            self.assertEqual(result["rows"], [])


@unittest.skipUnless(torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET"), "actual CUDA required")
class CudaTests(unittest.TestCase):
    def test_actual_gpu_packet_and_word_parity(self):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        packet_parity(self, torch.device("cuda", torch.cuda.current_device()))


if __name__ == "__main__":
    unittest.main()
