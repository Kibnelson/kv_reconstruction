"""Lane-layout compatibility, fresh receiver protocol and comparison accounting."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from kv_compare.lite_scoreboard import aggregate, network_model, validate_comparison
from scripts import package_lite_baselines as package

HAVE_RUNTIME = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("triton") is not None
if HAVE_RUNTIME:
    import torch
    from kv_compare.causal_packet import Packet, float_to_words, word_sha, write_json
    from kv_compare.lite_fast import FastEngine
    from kv_compare.gpu_coder import SliceCoder
    from scripts import bench_lite_baselines as bench
    from tests.test_llama_causal import tiny_state


class ScoreboardTests(unittest.TestCase):
    def rows(self):
        return [dict(id=method + doc, method=method, document=doc, kappa=0., mode="lossless",
            shape=[1, 2, 4], values=8, original_word_sha256=doc, decoded_word_sha256=doc,
            original_exact=True, status="verified", packet_bytes=size, complete_bpv=size,
            saved_percent=100 * (1 - size / 16), encode=dict(wall_median_ms=enc), decode=dict(wall_median_ms=dec))
            for method, size, enc, dec in (("raw_bf16", 16, 1, 1), ("splitzip_official_framed", 12, 2, 2), ("lite_b8", 8, 4, 5))
            for doc in ("text", "code")]

    def test_coverage_ratios_and_decimal_bits_vs_bytes(self):
        rows = self.rows(); validate_comparison(rows)
        summaries = aggregate(rows)
        lite = next(r for r in summaries if r["method"] == "lite_b8")
        self.assertEqual(lite["vs_baseline"]["splitzip_official_framed"]["decode_time_ratio"], 2.5)
        self.assertAlmostEqual(lite["vs_baseline"]["splitzip_official_framed"]["packet_bytes_ratio"], 2 / 3)
        case = network_model(summaries, [1.])["links"][0]
        e = next(r for r in case["estimates"] if r["method"] == "lite_b8")
        self.assertAlmostEqual(e["transfer_ms"], 8 * 8 / 1e9 * 1000)
        self.assertAlmostEqual(e["fresh_ms"] - e["precompressed_ms"], 4.)
        self.assertFalse(network_model(summaries)["measured_network"])
        with self.assertRaisesRegex(ValueError, "positive"):
            network_model(summaries, [0.])

    def test_lossy_is_not_in_lossless_network_winners_even_if_fixture_matches(self):
        rows = self.rows()
        rows.append(dict(rows[0], id="fp8", method="fp8_e4m3fn_saturating", mode="lossy", packet_bytes=1))
        summaries = aggregate(rows)
        methods = {r["method"] for r in network_model(summaries)["links"][0]["estimates"]}
        self.assertNotIn("fp8_e4m3fn_saturating", methods)
        for bad in (rows[:-2], [dict(rows[0], original_word_sha256="changed"), *rows[1:]]):
            with self.assertRaises(ValueError):
                validate_comparison(bad)

    def test_packaging_refuses_cpu_success_and_excludes_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / "run"; run.mkdir()
            (run / "results.json").write_text(json.dumps(dict(status="completed", gpu_measured=False)))
            with self.assertRaisesRegex(ValueError, "CPU evidence"):
                package.main([str(run), "--destination", str(root / "bad")])
            (run / "results.json").write_text(json.dumps(dict(status="failed", gpu_measured=False, jobs={})))
            (run / "summary.md").write_text("software fixture only\n")
            (run / "payload.kvc").write_bytes(b"local only")
            self.assertEqual(package.main([str(run), "--destination", str(root / "report")]), 0)
            self.assertEqual({p.name for p in (root / "report").iterdir()}, {"results.json", "summary.md", "package_manifest.json"})


def lane_parity(test, device):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); tiny_state(root)
        control = FastEngine(root, device, predictor_format="bf16", values_per_stream=2, block=64)
        generator = torch.Generator().manual_seed(23)
        words = float_to_words(torch.randn(2, 5, 8, generator=generator) * .2).to(device, torch.int16)
        expected = {}
        for kappa in (0., .5, 1.):
            p = control.encode(words, start=7, kappa=kappa)
            expected[kappa] = (p.export(), control.last_reconstruction.clone())
        for block in (8, 16, 32, 64):
            engine = FastEngine(root, device, predictor_format="bf16", values_per_stream=2, block=block)
            for kappa in (0., .5, 1.):
                with test.subTest(block=block, kappa=kappa, device=str(device)):
                    p = engine.encode(words, start=7, kappa=kappa)
                    test.assertEqual(p.header["block"], block)
                    test.assertTrue(torch.equal(engine.last_reconstruction, expected[kappa][1]))
                    out = engine.decode(Packet.restore(p.export(), device))
                    test.assertTrue(torch.equal(out, expected[kappa][1]))
                    if not kappa:
                        test.assertTrue(torch.equal(out, words))
                    if block == 64:
                        test.assertEqual(p.export(), expected[kappa][0])
                    else:
                        test.assertNotEqual(p.export(), expected[kappa][0])
            # All IEEE patterns, including exceptional payloads, remain exact
            # via the existing charged raw fallback for every block width.
            patterns = torch.arange(65536, dtype=torch.int32).to(device, torch.int16).reshape(2, 4096, 8)
            p = engine.encode(patterns)
            test.assertTrue(p.header["raw"])
            test.assertTrue(torch.equal(engine.decode(Packet.restore(p.export(), device)), patterns))


@unittest.skipUnless(HAVE_RUNTIME and os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU interpreter required")
class InterpreterTests(unittest.TestCase):
    def test_all_lane_widths_reproduce_lossless_and_lossy_reconstructions(self):
        lane_parity(self, torch.device("cpu"))

    def test_actual_complete_worker_and_fresh_receiver_use_json_string_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); state = root / "state"; state.mkdir(); tiny_state(state)
            words = float_to_words(torch.arange(48).reshape(2, 3, 8).float() / 100).to(torch.int16)
            source = root / "input.npy"; np.save(source, words.numpy().astype("<u2"), allow_pickle=False)
            control = FastEngine(state, "cpu", predictor_format="bf16", values_per_stream=2, block=64)
            expected = {}
            for k in (0., .5, 1.):
                p = control.encode(words, start=9, kappa=k)
                expected[f"tiny:{k:g}"] = dict(start=9, packet_sha256=hashlib.sha256(p.export()).hexdigest(),
                    decoded_word_sha256=word_sha(control.last_reconstruction))
            q = dict(method="lite_b8", codec=dict(kind="lite", state=str(state),
                engine_options=dict(predictor_format="bf16", values_per_stream=2, block=8)),
                output=str(root / "output"), warmup=1, repeats=2, kappas=[0., .5, 1.], expected=expected,
                inputs=[dict(document="tiny", path=str(source), file_sha256=bench.digest(source),
                             original_word_sha256=word_sha(words))])
            bench.worker(q, torch, torch.device("cpu"))
            data = json.loads((root / "output/worker_results.json").read_text())
            self.assertEqual(data["status"], "software_checked"); self.assertFalse(data["gpu_measured"])
            self.assertEqual(len(data["rows"]), 3)
            self.assertTrue(all(r["status"] == "verified" for r in data["rows"]))
            self.assertTrue(all(not r["packet_matches_original"] for r in data["rows"]))
            validate_comparison(data["rows"])
            proof = json.loads((root / "output/receiver.json").read_text())
            self.assertTrue(proof["no_source_kv_or_predictions"])
            self.assertFalse(proof["gpu_measured"])

    def test_small_groups_launch_more_independent_programs_with_same_stream_length(self):
        configurations = []
        for block in (64, 32, 16, 8):
            coder = SliceCoder("cpu", values_per_stream=128, block=block, warps=1)
            configurations.append(coder.layout(512 * 128)[:2])
        self.assertEqual(configurations, [(512, 8), (512, 16), (512, 32), (512, 64)])

    def test_native_workers_keep_the_original_baseline_runtime_setup(self):
        from scripts import bench_llama_baselines as baseline
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "q.json"
            for kind in ("raw_bf16", "splitzip_official_framed", "exans"):
                write_json(request, dict(codec=dict(kind=kind)))
                with mock.patch.object(bench, "device_check", side_effect=AssertionError("Lite policy leaked")), \
                        mock.patch.object(baseline, "device_check", return_value=(torch, torch.device("cpu"))) as check, \
                        mock.patch.object(bench, "receive"):
                    self.assertEqual(bench.main(["--device", "cpu", "--software-check", "--receive", str(request)]), 0)
                    check.assert_called_once_with("cpu", True)

    def test_actual_gpu_entry_never_substitutes_a_cpu_benchmark(self):
        if torch.cuda.is_available():
            self.skipTest("requires CPU-only host")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("sys.stderr"):
                self.assertEqual(bench.main(["--output", tmp]), 1)
            result = json.loads((Path(tmp) / "results.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["gpu_measured"])
            self.assertEqual(result["rows"], [])


@unittest.skipUnless(HAVE_RUNTIME and torch.cuda.is_available() and not os.environ.get("TRITON_INTERPRET"), "actual CUDA required")
class CudaTests(unittest.TestCase):
    def test_actual_gpu_lane_layout_packet_replay(self):
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
        lane_parity(self, torch.device("cuda", torch.cuda.current_device()))


if __name__ == "__main__":
    unittest.main()
