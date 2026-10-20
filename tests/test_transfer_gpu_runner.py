"""Tests for truthful device selection, paid wire bytes and fixture replay."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("bench_transfer_gpu",
    Path(__file__).resolve().parents[1] / "scripts/bench_transfer_gpu.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
HAS_TORCH = importlib.util.find_spec("torch") is not None


def act(seed, tokens=32):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(2, tokens, 8)).astype(np.float32)
    raw = values.view(np.uint32)
    words = ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
    return {"K_bits": words[..., :4], "V_bits": words[..., 4:],
            "Hk": np.array(1), "hd": np.array(4), "base": np.array(10000.)}


class TransferGpuRunnerTests(unittest.TestCase):
    def test_default_cuda_failure_records_failure_without_cpu_fallback(self):
        torch = SimpleNamespace(set_num_threads=lambda _: None,
            device=lambda _: SimpleNamespace(type="cuda"),
            cuda=SimpleNamespace(is_available=lambda: False))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with mock.patch.object(runner, "_torch", return_value=torch), mock.patch("builtins.print"):
                code = runner.main(["--replay-fixture", "not-read", "--output", str(output)])
            self.assertEqual(code, 2)
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["gpu_validated"])
            self.assertFalse(report["gpu_tested"])
            self.assertIn("no CPU fallback", report["error"])
            self.assertFalse((output / "rows.csv").exists())

    def test_fixture_checksum_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.npz").write_bytes(b"changed")
            (root / "manifest.json").write_text(json.dumps({"files": {"state.npz": {
                "bytes": 7, "sha256": "0" * 64}}}))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                runner.load_fixture(root)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch dependency")
    def test_default_available_cuda_device_gets_an_explicit_index(self):
        import torch
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "current_device", return_value=2), \
             mock.patch.object(torch.cuda, "set_device") as set_device:
            selected = runner.select_device(torch, "cuda")
        self.assertEqual(str(selected), "cuda:2")
        set_device.assert_called_once_with(torch.device("cuda:2"))

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch dependency")
    def test_complete_wire_counts_header_index_and_original_payload(self):
        import torch
        from qwen_kv_bench.transfer_pool_gpu import make_state, encode_pool, decode_pool
        words = np.arange(256, dtype=np.uint16).reshape(2, 16, 8) * np.uint16(193)
        refs = np.full((2, 8), 127, dtype=np.uint8)
        widths = np.full((2, 8), 8, dtype=np.uint8)
        state = make_state(refs, widths, "cpu")
        packet = encode_pool(torch.from_numpy(words), state)
        payload, offsets = runner.original_pool(words, refs, widths)
        self.assertEqual(packet.payload.numpy().tobytes(), payload)
        np.testing.assert_array_equal(packet.offsets.numpy(), offsets)
        blob, descriptor = runner.serialize_pool(packet)
        self.assertEqual(runner.parse_pool_wire(blob), descriptor)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            runner.parse_pool_wire(blob[:-1])
        self.assertEqual(descriptor["header_bytes"] % 8, 0)
        self.assertEqual(len(blob), descriptor["header_bytes"] + descriptor["index_bytes"] + len(payload))
        self.assertEqual(descriptor["index_bytes"], 8 * (2 + 1))
        buffer = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy())
        reconstructed = runner.pool_from_wire_tensor(buffer, descriptor)
        np.testing.assert_array_equal(decode_pool(reconstructed, state).numpy(), words)

    @unittest.skipUnless(HAS_TORCH, "optional PyTorch dependency")
    def test_cpu_timing_has_no_cuda_measurements(self):
        import torch
        calls = []
        result, timing = runner.measure_device(lambda: calls.append(1) or len(calls),
            torch, torch.device("cpu"), warmup=2, repeats=3)
        self.assertEqual(result, 5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(timing["samples"]), 3)
        self.assertIsNone(timing["cuda_event_median_ms"])
        self.assertTrue(all(value["cuda_event_ms"] is None for value in timing["samples"]))

    @unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("constriction"), "optional torch/constriction dependencies")
    def test_full_tiny_fixture_cpu_replay_stays_explicitly_cpu(self):
        from qwen_kv_bench.transfer_reference import fit, save_model, words, encode
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"; fixture.mkdir()
            model = fit([act(1), act(2)], act(3), mode="block", block_size=2, max_positions=64)
            save_model(model, fixture / "block32.npz")
            source = words(act(4, 16))
            np.save(fixture / "words.npy", source)
            refs = np.full((2, 8), 127, dtype=np.uint8)
            widths = np.full((2, 8), 4, dtype=np.uint8)
            np.savez(fixture / "pool_state.npz", refs=refs, widths=widths)
            packet, _ = encode(model, source, start=8)
            (fixture / "transfer.bin").write_bytes(packet)
            payload, offsets = runner.original_pool(source, refs, widths)
            (fixture / "pool.bin").write_bytes(payload)
            np.save(fixture / "offsets.npy", offsets)
            files = {path.name: {"bytes": path.stat().st_size, "sha256": runner.sha_file(path)} for path in fixture.iterdir()}
            manifest = {"format": "transfer_gpu_replay_fixture_v1", "gpu_validated": False,
                "state_file": "block32.npz", "pool_state_file": "pool_state.npz", "files": files,
                "pages": [{"name": "heldout", "case": "heldout", "start": 8,
                    "words_file": "words.npy", "transfer_packet_file": "transfer.bin",
                    "pool_payload_file": "pool.bin", "pool_offsets_file": "offsets.npy"}]}
            (fixture / "manifest.json").write_text(json.dumps(manifest))
            output = root / "output"
            with mock.patch("builtins.print"):
                code = runner.main(["--replay-fixture", str(fixture), "--device", "cpu", "--repeats", "1",
                    "--warmup", "0", "--threads", "1", "--coefficient-dtype", "int32", "--output", str(output)])
            self.assertEqual(code, 0, (output / "summary.md").read_text())
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["explicit_cpu_validation"])
            self.assertFalse(report["gpu_validated"])
            self.assertFalse(report["gpu_tested"])
            self.assertFalse(report["coverage"]["gpu_transfer_entropy_coder"])
            self.assertTrue(report["aggregate"]["all_pool_exact"])
            self.assertTrue(report["aggregate"]["all_predictor_mu_exact"])
            row = report["rows"][0]
            self.assertFalse(row["pool"]["h2d"]["measured"])
            self.assertTrue(row["cpu_transfer"]["validation"]["bit_exact"])
            self.assertGreater(row["pool"]["wire"]["wire_bytes"], row["pool"]["stats"]["payload_bytes"])
            self.assertTrue((output / "rows.csv").exists())


if __name__ == "__main__":
    unittest.main()
