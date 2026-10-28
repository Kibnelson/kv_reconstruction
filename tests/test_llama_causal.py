"""Causal packet checks. Invoke with TRITON_INTERPRET=1 for CPU software checks."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import importlib.util
import numpy as np
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
    raise unittest.SkipTest("PyTorch and Triton are required for causal packet tests")
import torch
from kv_compare.causal_entropy import ESC, LIMIT, ResidualCoder, compact, residual_tables
from kv_compare.causal_packet import Engine, Packet, json_bytes, sha, word_sha, write_json
from kv_compare.core import float_to_words, words_to_float


def tiny_state(root, adaptive=True):
    rng = np.random.default_rng(1)
    W = 8; perm = np.array([0, 2, 1, 3, 4, 5, 6, 7])
    starts = np.arange(W) // 2 * 2
    files = []
    for l in range(2):
        ne = W if l else 0
        A = rng.normal(0, .08, (W, ne + W)).astype(np.float32)
        for j in range(W):
            A[j, ne + starts[j]:] = 0
        path = root / f"layer_{l:03d}.npz"
        np.savez(path, A=A, b=np.linspace(-.2, .2, W).astype(np.float32),
            spread=np.full(W, .3, np.float32), perm=perm, starts=starts,
            kinds=np.asarray(["P"] if l else [], dtype="U3"))
        files.append(dict(path=path.name, sha256=sha(path)))
    state = dict(format="causal_fitted_v1", policy="tiny", adaptive=adaptive,
        fit_key="tiny", kv_heads=1, head_dim=4, inv_freq=[1., .01],
        rope_layout="half", attention_scaling=1., model="tiny", revision="tiny", files=files)
    state["state_sha256"] = hashlib.sha256(json_bytes(state)).hexdigest()
    write_json(root / "state.json", state)


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU interpreter run required")
class CausalTests(unittest.TestCase):
    def test_receiver_rebuilds_context_for_lossless_and_lossy_nonzero_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            torch.manual_seed(8)
            original = float_to_words(torch.randn(2, 5, 8) * .2).to(torch.int16)
            for fmt, kappa in ((fmt,k) for fmt in ("fp32","bf16") for k in (0., .5, 1.)):
                with self.subTest(kappa=kappa, predictor_format=fmt):
                    enc = Engine(root, "cpu", predictor_format=fmt, values_per_stream=2, block=8)
                    packet = enc.encode(original, start=7, kappa=kappa)
                    intended = enc.last_reconstruction.clone()
                    wire = packet.export()
                    # An independent object has only fitted files and packet bytes.
                    dec = Engine(root, "cpu", predictor_format=fmt, values_per_stream=2, block=8)
                    restored = dec.decode(Packet.restore(wire, "cpu"))
                    self.assertTrue(torch.equal(restored, intended))
                    if not kappa:
                        self.assertTrue(torch.equal(restored, original))
                    else:
                        self.assertFalse(torch.equal(restored, original))
                        x = words_to_float(original)
                        bound = .5 * kappa * .3 + (x.abs() + .5 * kappa * .3) * 2**-8 + 1e-7
                        self.assertTrue(bool(((words_to_float(restored) - x).abs() <= bound).all()))
                    self.assertEqual(set(packet.header), {"format","state_sha256","shape","start","kappa",
                        "raw","predictor_format","values_per_stream","block","table_sha256"})

    def test_separate_receiver_process_has_only_packet_and_fixed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            options = dict(predictor_format="bf16", values_per_stream=2, block=8)
            enc = Engine(root, "cpu", **options)
            x = float_to_words(torch.randn(2, 3, 8) * .1).to(torch.int16)
            p = enc.encode(x, kappa=.5)
            wire = p.export(); packet = root / "wire.kvc"; packet.write_bytes(wire)
            expected = word_sha(enc.last_reconstruction)
            request = dict(engine_options=options, output=str(root / "verified.json"), entries=[
                dict(id="tiny", state_dir=str(root), packet=str(packet),
                     packet_sha256=hashlib.sha256(wire).hexdigest(),
                     expected_word_sha256=expected, decoded=str(root / "received.npy"))])
            write_json(root / "request.json", request)
            script = Path(__file__).resolve().parents[1] / "scripts/bench_llama_causal.py"
            r = subprocess.run([sys.executable, str(script), "--device", "cpu", "--software-check",
                "--verify-request", str(root / "request.json")], capture_output=True, text=True, timeout=90)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            proof = json.loads((root / "verified.json").read_text())
            self.assertFalse(proof["gpu_measured"])
            self.assertTrue(proof["no_source_kv_or_prediction_arrays"])
            self.assertEqual(proof["checks"][0]["word_sha256"], expected)

    def test_fit_cache_reuse_and_runtime_widths_are_tamper_checked(self):
        from kv_compare.predict import Study
        from kv_compare.causal_study import fit_policy, calibrate_runtime
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); captures = root / "captures"; captures.mkdir()
            rng = np.random.default_rng(4)
            for name in ("train0","train1","tables","test"):
                w = float_to_words(torch.from_numpy(rng.normal(0,.2,(2,12,8)))).numpy().astype(np.uint16)
                np.save(captures / (name + "_K_bits.npy"), w[:, :, :4])
                np.save(captures / (name + "_V_bits.npy"), w[:, :, 4:])
                np.save(captures / (name + "_ids.npy"), np.arange(12))
            write_json(captures / "meta.json", dict(model="tiny", revision="tiny", kv_heads=1, query_heads=1,
                head_dim=4, inv_freq=[1.,.01], rope_layout="half", attention_scaling=1.))
            study = Study(captures, "cpu", ["train0","train1"], "tables", ["test"], log=lambda x:None)
            directory = root / "fit"
            fit_policy(study, "lite", directory, {"tiny":True}, lambda x:None)
            calibrate_runtime(study, directory, "bf16", lambda x:None)
            first = (directory / "state.json").read_bytes(); messages = []
            fit_policy(study, "lite", directory, {"tiny":True}, messages.append)
            self.assertEqual((directory / "state.json").read_bytes(), first)
            self.assertEqual(sum("reuse fitted layer" in x for x in messages), 2)
            engine = Engine(directory, "cpu", predictor_format="bf16", values_per_stream=2, block=8)
            words = torch.stack([study.tables.words(l, "cpu") for l in range(study.L)]).to(torch.int16)
            np.testing.assert_allclose(engine.calibrate(words).numpy(), np.load(directory / "runtime_spread.npy"), rtol=1e-6)
            with (directory / "runtime_spread.npy").open("ab") as f: f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "spread file changed"):
                Engine(directory, "cpu", predictor_format="bf16")

    def test_complete_codec_quality_and_package_on_tiny_original_bf16_llama(self):
        from argparse import Namespace
        from unittest import mock
        from transformers import LlamaConfig, LlamaForCausalLM
        from kv_compare.capture import capture_ids, model_meta
        from kv_compare.predict import Study
        from kv_compare.causal_study import fit_policy, calibrate_runtime
        from scripts import bench_llama_causal as bench
        from scripts import package_llama_causal as package
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); captures = root / "captures"; out = root / "out"; out.mkdir()
            torch.manual_seed(13)
            cfg = LlamaConfig(vocab_size=37, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                max_position_embeddings=64, rope_theta=500000.)
            model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
            meta = model_meta(model, "tiny-original-bf16", "test")
            for i,name in enumerate(("train0","train1","tables","test_text","test_code")):
                capture_ids(model, ((np.arange(12) * (i + 2) + 1) % 37).tolist(),
                            captures, name, torch.device("cpu"))
            write_json(captures / "meta.json", meta)
            study = Study(captures, "cpu", ["train0","train1"], "tables",
                          ["test_text","test_code"], log=lambda x:None)
            states = {}
            for policy in ("block32","lite"):
                states[policy] = root / policy
                fit_policy(study, policy, states[policy], {"tiny": True}, lambda x:None)
                study._f.clear()
                calibrate_runtime(study, states[policy], "bf16", lambda x:None)
            args = Namespace(predictor_format="bf16", values_per_stream=2, block=8,
                kappas=[.5,1.], policies=["block32","lite"], tokens=4,
                warmup=1, repeats=1, software_check=True, continuation=3)
            result = dict(format="causal_compare_v1", status="running", gpu_measured=False)
            bench.run_codec(args, result, study, study.tests, states, out, torch, torch.device("cpu"))
            self.assertEqual(len(result["rows"]), 12)
            self.assertTrue(result["codec_completed"])
            self.assertTrue(all(r["status"] == "verified" for r in result["rows"]))
            # A resumed packet stage validates/replays saved bytes, without
            # repeating encode/decode timing samples.
            with mock.patch.object(bench, "timed", side_effect=AssertionError("repeated measurement")):
                bench.run_codec(args, result, study, study.tests, states, out, torch, torch.device("cpu"))
            with mock.patch("huggingface_hub.snapshot_download", return_value=str(root)), \
                    mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model):
                bench.run_quality(args, result, study.tests, out, torch, torch.device("cpu"))
            self.assertTrue(result["quality_completed"])
            self.assertTrue(all(abs(r["quality"]["ppl_change_percent"]) < 1e-6
                for r in result["rows"] if r["mode"] == "lossless"))
            result["status"] = "completed"
            write_json(out / "results.json", result); bench.report(result, out)
            package.main([str(out), "--destination",str(root / "report")])
            files = json.loads((root / "report/package_manifest.json").read_text())["files"]
            self.assertNotIn("packets", files)
            self.assertFalse(any(p.suffix in (".npy",".npz",".kvc") for p in (root / "report").rglob("*")))
            packet = out / result["rows"][0]["packet"]
            with packet.open("ab") as f: f.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                package.main([str(out), "--destination",str(root / "tampered_report")])

    def test_all_bf16_patterns_preserved_by_charged_raw_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            engine = Engine(root, "cpu", predictor_format="fp32", values_per_stream=2, block=8)
            words = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(2, 4096, 8)
            packet = engine.encode(words)
            self.assertTrue(packet.header["raw"])
            self.assertGreater(len(packet.export()), words.numel() * 2)
            self.assertTrue(torch.equal(engine.decode(Packet.restore(packet.export(), "cpu")), words))

    def test_wire_corruption_state_mismatch_and_invalid_offsets_stop_before_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tiny_state(root)
            engine = Engine(root, "cpu", predictor_format="fp32", values_per_stream=2, block=8)
            x = float_to_words(torch.ones(2, 3, 8) * .1).to(torch.int16)
            p = engine.encode(x); wire = bytearray(p.export()); wire[-40] ^= 1
            with self.assertRaisesRegex(ValueError, "checksum"): Packet.restore(bytes(wire), "cpu")
            bad = Packet(dict(p.header, state_sha256="wrong"), p.starts, p.payload)
            with self.assertRaisesRegex(ValueError, "state/options"): engine.decode(bad)
            starts = p.starts.clone(); starts[0] = -1
            with self.assertRaisesRegex(ValueError, "directory"): engine.decode(Packet(p.header, starts, p.payload))

    def test_residual_tail_escape_and_compact_directory_roundtrip(self):
        c = ResidualCoder("cpu", .5, values_per_stream=2, block=8)
        sym = torch.tensor([0, LIMIT, 2 * LIMIT, ESC, 100, ESC, 19, LIMIT, 254], dtype=torch.int32)
        raw = torch.tensor([0, 0, 0, -32768, 0, 32704, 0, 0, 0], dtype=torch.int16)
        ids = torch.tensor([0, 32, 64], dtype=torch.int32)
        _, P, cap = c.layout(len(sym))
        buf = torch.empty(P * cap, dtype=torch.int16)
        starts = torch.empty(P, dtype=torch.int64)
        c.encode(sym, raw, ids, 3, out=buf, starts=starts)
        packed, offsets = compact(buf, starts, torch.zeros(P, dtype=torch.int64),
            (torch.arange(P, dtype=torch.int64) + 1) * cap)
        out, rw = c.decode(packed, offsets, ids, 3, len(sym))
        self.assertTrue(torch.equal(out, sym))
        self.assertTrue(torch.equal(rw[sym == ESC], raw[sym == ESC]))
        self.assertLess(packed.numel(), buf.numel())


class DeviceTests(unittest.TestCase):
    def test_unindexed_cuda_is_resolved_before_every_cuda_call(self):
        from scripts import bench_llama_causal as bench
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "current_device", return_value=2) as current, \
                mock.patch.object(torch.cuda, "set_device") as select, \
                mock.patch.object(torch.cuda, "synchronize") as sync, \
                mock.patch.object(torch, "ones") as ones, \
                mock.patch.object(torch, "use_deterministic_algorithms"), \
                mock.patch.object(torch, "backends", mock.Mock()):
            _, device = bench.device_check("cuda")
            self.assertEqual(device, torch.device("cuda:2"))
            current.assert_called_once_with()
            select.assert_called_once_with(device)
            sync.assert_called_once_with(device)
            self.assertEqual(ones.call_args.kwargs["device"], device)

    def test_explicit_cuda_index_is_preserved(self):
        from scripts import bench_llama_causal as bench
        from unittest import mock
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError("explicit index replaced")), \
                mock.patch.object(torch.cuda, "set_device") as select, \
                mock.patch.object(torch.cuda, "synchronize"), \
                mock.patch.object(torch, "ones"), \
                mock.patch.object(torch, "use_deterministic_algorithms"), \
                mock.patch.object(torch, "backends", mock.Mock()):
            _, device = bench.device_check("cuda:1")
            self.assertEqual(device, torch.device("cuda:1"))
            select.assert_called_once_with(device)

    def test_production_guard_still_rejects_cpu_and_interpreter(self):
        from scripts import bench_llama_causal as bench
        from unittest import mock
        for value, environment in (("cpu", {}), ("cuda", {"TRITON_INTERPRET":"1"})):
            with mock.patch.dict(os.environ, environment, clear=True), \
                    mock.patch.object(torch.cuda, "is_available", return_value=True), \
                    mock.patch.object(torch.cuda, "set_device") as select:
                with self.assertRaisesRegex(RuntimeError, "Actual CUDA required"):
                    bench.device_check(value)
                select.assert_not_called()

    def test_default_cli_device_has_an_explicit_index(self):
        from scripts import bench_llama_causal as bench
        self.assertEqual(bench.parser().parse_args([]).device, "cuda:0")


class TableTests(unittest.TestCase):
    def test_missing_cached_weight_shard_is_detected_before_fitting(self):
        from scripts import bench_llama_causal as bench
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "config.json", dict(model_type="llama",num_hidden_layers=28))
            write_json(root / "model.safetensors.index.json", dict(weight_map={"a":"shard1.safetensors","b":"shard2.safetensors"}))
            (root / "shard1.safetensors").write_bytes(b"fixture")
            with mock.patch("huggingface_hub.snapshot_download", return_value=str(root)):
                with self.assertRaisesRegex(ValueError, "shards are missing"):
                    bench.cached_model_snapshot()
                (root / "shard2.safetensors").write_bytes(b"fixture")
                self.assertEqual(bench.cached_model_snapshot(), root)
                write_json(root / "config.json",dict(model_type="llama",num_hidden_layers=28,quantization_config={"bits":4}))
                with self.assertRaisesRegex(ValueError, "unquantized"):
                    bench.cached_model_snapshot()

    def test_all_residual_tables_cover_the_same_integer_precision(self):
        for kappa in (.25, .5, 1., 2., 16.):
            cdf = residual_tables(kappa)
            self.assertTrue(np.all(cdf[:, 0] == 0))
            self.assertTrue(np.all(cdf[:, -1] == 65536))
            self.assertTrue(np.all(np.diff(cdf, axis=1) >= 1))
        for kappa in (0., -1., float("nan"), 17.):
            with self.assertRaises(ValueError): residual_tables(kappa)

    def test_no_cuda_refuses_benchmark_and_preserves_existing_report(self):
        from scripts import bench_llama_causal as bench
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root / "old"; old.mkdir()
            (old / "results.json").write_text('{"identity":"old","status":"completed"}')
            content = (old / "results.json").read_bytes()
            with mock.patch.object(bench, "device_check", side_effect=RuntimeError("Actual CUDA required")):
                code = bench.main(["--source-run",str(root / "missing"),"--output",str(old)])
                self.assertEqual(code, 1)
                self.assertEqual((old / "results.json").read_bytes(), content)
                fresh = root / "fresh"
                self.assertEqual(bench.main(["--source-run",str(root / "missing"),"--output",str(fresh)]), 1)
                result = json.loads((fresh / "results.json").read_text())
                self.assertFalse(result["gpu_measured"])
                self.assertEqual(result["status"], "failed")
                self.assertFalse((root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
