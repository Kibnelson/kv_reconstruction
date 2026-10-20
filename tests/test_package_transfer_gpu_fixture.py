"""Synthetic artifact-integrity tests; these are not entropy-codec benchmarks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_fixture", ROOT / "scripts/package_transfer_gpu_fixture.py")
package = importlib.util.module_from_spec(spec); spec.loader.exec_module(package)
SOURCE = ROOT / "experiments/team_transfer_original/combined.py"


def make_run(root):
    run = root / "run"; run.mkdir()
    for folder in ("captures", "states", "packet_checks"):
        (run / folder).mkdir()
    rng = np.random.default_rng(713)
    identities, training, tests = [], [], []
    source_words = {}
    for split, names in (("train", ("train_one", "train_two", "train_three")), ("test", ("test_text", "test_code"))):
        for name in names:
            exponent = rng.integers(119, 132, size=(2, 32, 4), dtype=np.uint16)
            words = (exponent << 7) | rng.integers(0, 128, size=exponent.shape, dtype=np.uint16)
            ids = np.arange(32, dtype=np.int64) + (100 if name.endswith("code") else 0)
            path = run / "captures" / (name + ".npz")
            np.savez(path, K_bits=words[:, :, :2], V_bits=words[:, :, 2:], ids=ids,
                     positions=np.arange(32, dtype=np.int64), Hk=np.array(1), hd=np.array(2), base=np.array(10000.))
            identity = dict(name=name, split=split, capture_file=str(path), file=path.name,
                            kv_sha256=package.sha_array(words), ids_sha256=package.sha_array(ids))
            identities.append(identity); (training if split == "train" else tests).append(identity)
            source_words[name] = words
    fitted_input = [package.load_capture(Path(item["capture_file"])) for item in training]
    original = package.load_reference(SOURCE, fitted_input, 4, 16)
    fitted = [original["fit_pool"](layer) for layer in range(2)]
    description = dict(format="team_transfer_reference_v1", geometry=dict(layers=2, channels=2, head_dim=2, heads=1),
                       mode="block", block_size=32, metadata_rank=0, metadata_divisor=128, nblocks=[1, 1])
    arrays = {"fixture_coefficients": np.zeros((2, 4), np.int64)}
    digest = hashlib.sha256(package.canonical(description))
    for key, array in sorted(arrays.items()):
        digest.update(key.encode()); digest.update(array.dtype.str.encode())
        digest.update(package.canonical(list(array.shape))); digest.update(array.tobytes())
    state_hash = digest.hexdigest(); description["state_sha256"] = state_hash
    state_path = run / "states/block32.npz"
    np.savez_compressed(state_path, description=np.frombuffer(package.canonical(description), np.uint8), **arrays)
    rows, pool_pages = [], []
    for identity in tests:
        for start in (0, 16):
            words = source_words[identity["name"]][:, start:start + 16]
            payload = b"\x01\x02\x03\x04"
            header = dict(format="team_transfer_reference_v1", state_sha256=state_hash, shape=list(words.shape), start=start,
                          kind="range", correction="direct", residue_bits=0, metadata_bytes=0, stream_bytes=len(payload),
                          decoded_sha256=hashlib.sha256(words.tobytes()).hexdigest(), payload_sha256=hashlib.sha256(payload).hexdigest())
            header_raw = package.canonical(header)
            packet = struct.pack("<8sI", b"KVTRF001", len(header_raw)) + header_raw + payload
            path = run / "packet_checks" / (identity["name"] + f"_{start}.bin")
            path.write_bytes(packet); np.save(path.with_suffix(".npy"), words)
            path.with_suffix(".json").write_text(json.dumps(dict(inputs=[str(state_path), str(path)],
                         decoded_sha256=package.sha_array(words), stats=dict(exact_checksum_verified=True))))
            rows.append(dict(case=identity["name"], variant="block32", phase="TEST", start=start, tokens=16,
                             values=words.size, raw_bytes=words.nbytes, bit_exact=True, packet_only_subprocess=True,
                             decoder_inputs=[str(state_path), str(path)], packet_bytes=len(packet),
                             codec_stats=dict(packet_bytes=len(packet))))
            packed = sum(original["pack_block"](words[layer], *fitted[layer]).size for layer in range(2))
            pool_pages.append(dict(case=identity["name"], token_start=start, token_count=16,
                                   packed_bytes=packed, words=words.size, exact=True))
    source_identity = dict(path="scripts/review_transfer_extensions.py", sha256=package.file_sha256(ROOT / "scripts/review_transfer_extensions.py"))
    total_words = sum(row["values"] for row in rows)
    report = dict(format="transfer_extension_study_v1", status="completed", arguments=dict(block_size=32),
                  test_rows=rows, aggregate=dict(block32=dict(values=total_words, packet_bytes=sum(row["packet_bytes"] for row in rows),
                  resident_decoder_state_bytes=64, serialized_decoder_state_bytes=state_path.stat().st_size)),
                  provenance=dict(sources=[source_identity], model_identity=dict(captures=identities, model="synthetic")), scope=dict(codec_device="cpu"))
    pool = dict(format="team_original_pool_cpu_reference_v1", status="completed", source=dict(sha256=package.ORIGINAL_POOL_SHA256),
                geometry=dict(block_tokens=16, layers=2, combined_kv_channels=4), training=training, tests=tests,
                pages=pool_pages, aggregate=dict(words=total_words, packed_bytes=sum(page["packed_bytes"] for page in pool_pages)))
    (run / "results.json").write_text(json.dumps(report))
    pool_path = root / "pool.json"; pool_path.write_text(json.dumps(pool))
    return run, pool_path, report, pool


class ReplayFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.pool_path, self.report, self.pool = make_run(self.root)
        self.output = self.root / "fixture"

    def build(self):
        return package.build_fixture(self.run, self.output, self.pool_path, SOURCE, ROOT)

    def save_report(self):
        (self.run / "results.json").write_text(json.dumps(self.report))

    def test_frozen_words_state_pool_bytes_positions_and_hash_ledger(self):
        result = self.build()
        self.assertFalse(result["gpu_validated"])
        self.assertFalse(result["validation"]["transfer_entropy_decode_rerun_by_builder"])
        self.assertEqual(len(result["pages"]), 4)
        self.assertEqual((self.output / "block32.npz").read_bytes(), (self.run / "states/block32.npz").read_bytes())
        with np.load(self.output / result["pool_state_file"], allow_pickle=False) as data:
            refs, widths = data["refs"], data["widths"]
        self.assertEqual(refs.dtype, np.uint8)
        original = package.load_reference(SOURCE, [], 4, 16)
        total_pool = total_packets = 0
        for page in result["pages"]:
            words = np.load(self.output / page["words_file"], allow_pickle=False)
            offsets = np.load(self.output / page["pool_offsets_file"], allow_pickle=False)
            payload = np.frombuffer((self.output / page["pool_payload_file"]).read_bytes(), np.uint8)
            self.assertEqual(offsets.tolist()[0], 0); self.assertEqual(offsets[-1], payload.size)
            decoded = np.stack([original["unpack_block"](payload[offsets[l]:offsets[l + 1]],
                                refs[l].astype(np.int64), widths[l].astype(np.int64)) for l in range(2)])
            self.assertTrue(np.array_equal(words, decoded))
            self.assertEqual(np.load(self.output / page["positions_file"]).tolist(), list(range(page["start"], page["start"] + 16)))
            total_pool += payload.size; total_packets += (self.output / page["transfer_packet_file"]).stat().st_size
        self.assertEqual(result["expected"]["pool_payload_bytes"], total_pool)
        self.assertEqual(result["expected"]["transfer_packet_bytes"], total_packets)
        self.assertEqual(result["expected"]["pool_bits_per_value"], 8 * total_pool / result["expected"]["values"])
        for name, identity in result["files"].items():
            self.assertEqual(identity["sha256"], package.file_sha256(self.output / name))
            self.assertEqual(identity["bytes"], (self.output / name).stat().st_size)
        self.assertFalse(any("hidden" in path.name or "capture" in path.name for path in self.output.iterdir()))

    def test_changed_capture_word_is_rejected_before_output(self):
        path = self.run / "captures/test_text.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key:data[key].copy() for key in data.files}
        arrays["K_bits"][0, 0, 0] ^= 1; np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "Capture KV identity mismatch"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_changed_transfer_payload_is_rejected(self):
        path = Path(self.report["test_rows"][0]["decoder_inputs"][1])
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "payload identity mismatch"):
            self.build()

    def test_incorrect_recorded_child_output_is_rejected(self):
        path = Path(self.report["test_rows"][0]["decoder_inputs"][1]).with_suffix(".npy")
        words = np.load(path); words[0, 0, 0] ^= 1; np.save(path, words)
        with self.assertRaisesRegex(ValueError, "child decoded words mismatch"):
            self.build()

    def test_modified_frozen_state_is_rejected(self):
        path = self.run / "states/block32.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key:data[key].copy() for key in data.files}
        arrays["fixture_coefficients"][0, 0] = 1; np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "state numerical identity mismatch"):
            self.build()

    def test_pool_report_wrong_page_cost_is_rejected(self):
        self.pool["pages"][0]["packed_bytes"] += 1
        self.pool_path.write_text(json.dumps(self.pool))
        with self.assertRaisesRegex(ValueError, "Pool page bytes differ"):
            self.build()

    def test_changed_executed_source_identity_is_rejected(self):
        self.report["provenance"]["sources"][0]["sha256"] = "0" * 64; self.save_report()
        with self.assertRaisesRegex(ValueError, "Executed source hash mismatch"):
            self.build()

    def test_duplicate_page_is_rejected(self):
        self.report["test_rows"][1] = self.report["test_rows"][0]; self.save_report()
        with self.assertRaisesRegex(ValueError, "Duplicate transfer TEST page"):
            self.build()

    def test_existing_output_and_nested_output_are_rejected(self):
        self.output.mkdir(); (self.output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh or empty"):
            self.build()
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "outside the source run"):
            package.build_fixture(self.run, self.run / "nested", self.pool_path, SOURCE, ROOT)

    def test_unpinned_pool_source_never_executes(self):
        path = self.root / "modified.py"; path.write_text("raise RuntimeError('do not execute')\n")
        with self.assertRaisesRegex(ValueError, "source hash is not the pinned"):
            package.build_fixture(self.run, self.output, self.pool_path, path, ROOT)


if __name__ == "__main__":
    unittest.main()
