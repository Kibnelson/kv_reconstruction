#!/usr/bin/env python3
"""Freeze the already measured CPU pages/state for an exact-input GPU replay.

This packages evidence; it does not execute or validate any CUDA kernel. The
original pool functions are hash-pinned, fitted on the recorded TRAIN captures,
and checked against every recorded TEST page before any output is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_team_pool_reference import load_capture, load_reference, file_sha256
from review_transfer_extensions import canonical, inspect_packet, inspect_state, resolve_artifact

FORMAT = "transfer_gpu_replay_fixture_v1"
ORIGINAL_POOL_SHA256 = "b44a63d32c1d3144436a01205701a26478fed7a63c8bf27777932a3436354cf3"


def sha_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + canonical(value.shape) + value.tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_capture(path, identities):
    capture = load_capture(path)
    words = np.concatenate((capture["K_bits"], capture["V_bits"]), axis=2)
    require(bool(identities), f"No recorded identity for capture: {path.name}")
    for identity in identities:
        require(sha_array(words) == identity["kv_sha256"], f"Capture KV identity mismatch: {path.name}")
        require(sha_array(capture["ids"]) == identity["ids_sha256"], f"Capture token identity mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        positions = data["positions"].copy() if "positions" in data else np.arange(words.shape[1], dtype=np.int64)
    require(positions.shape == capture["ids"].shape == (words.shape[1],), "Capture token/position shape mismatch")
    return capture, words, positions


def build_fixture(run, output, pool_report, source, source_root=None, model_manifest=None):
    run, output, pool_report, source = map(lambda p: Path(p).resolve(), (run, output, pool_report, source))
    require(not output.is_relative_to(run), "Output must be outside the source run")
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), "Output must be fresh or empty")
    require(file_sha256(source) == ORIGINAL_POOL_SHA256, "Original pool source hash is not the pinned supplied source")
    report_path = run / "results.json"
    report, pool = json.loads(report_path.read_text()), json.loads(pool_report.read_text())
    require(report.get("format") == "transfer_extension_study_v1" and report.get("status") == "completed", "Need completed transfer study")
    require(pool.get("format") == "team_original_pool_cpu_reference_v1" and pool.get("status") == "completed", "Need completed pool reference")
    require(pool["source"]["sha256"] == ORIGINAL_POOL_SHA256, "Pool report source identity mismatch")
    require(report["arguments"]["block_size"] == 32, "Fixture supports frozen block32 baseline only")
    baseline = [row for row in report["test_rows"] if row["variant"] == "block32"]
    require(len(baseline) == len(pool["pages"]) and bool(baseline), "Transfer/pool TEST page counts differ")
    addresses = [(row["case"], row["start"], row["tokens"]) for row in baseline]
    require(len(set(addresses)) == len(addresses), "Duplicate transfer TEST page")
    source_root = Path(source_root).resolve() if source_root else ROOT
    executed_sources = []
    for identity in report["provenance"]["sources"]:
        path = (source_root / identity["path"]).resolve()
        require(path.is_relative_to(source_root) and path.is_file() and file_sha256(path) == identity["sha256"],
                f"Executed source hash mismatch: {identity['path']}")
        executed_sources.append(identity)
    model_identity = report["provenance"]["model_identity"]
    captures = model_identity["captures"]
    capture_dir = run / "captures"
    training = []
    for identity in pool["training"]:
        path = capture_dir / Path(identity["file"]).name
        other = [item for item in captures if item["split"] == "train" and Path(item["capture_file"]).name == path.name]
        require(len(other) == 1, f"Training capture linkage missing: {path.name}")
        capture, _, _ = checked_capture(path, [identity, other[0]])
        training.append(capture)
    geometry = pool["geometry"]
    require(geometry["block_tokens"] == 16, "Only original 16-token pool records supported")
    layers, _, key_channels = training[0]["K_bits"].shape
    require(all(item["K_bits"].shape == training[0]["K_bits"].shape for item in training), "TRAIN geometries differ")
    require((geometry["layers"], geometry["combined_kv_channels"]) == (layers, 2 * key_channels), "Pool geometry mismatch")
    reference = load_reference(source, training, 2 * key_channels, 16)
    fitted = [reference["fit_pool"](layer) for layer in range(layers)]
    refs, widths = (np.stack([pair[index] for pair in fitted]) for index in (0, 1))
    require(bool(np.all((refs >= 0) & (refs <= 255)) and np.all((widths >= 1) & (widths <= 8))), "Invalid pool calibration")
    # Store exactly the same integers using their smallest sufficient unsigned type.
    refs, widths = refs.astype(np.uint8), widths.astype(np.uint8)
    state_paths = {resolve_artifact(run, row["decoder_inputs"][0], "states").resolve() for row in baseline}
    require(len(state_paths) == 1, "Baseline pages must use one frozen state")
    state_path = state_paths.pop()
    state = inspect_state(state_path)
    require(state["verified"], "Frozen state numerical identity mismatch")
    cached = {}
    pages = []
    for index, row in enumerate(baseline):
        require(row.get("phase") == "TEST" and row.get("bit_exact") and row.get("packet_only_subprocess"), "Need previously exact packet-only TEST baseline")
        matches = [identity for identity in captures if identity["split"] == "test" and identity["name"] == row["case"]]
        require(len(matches) == 1, f"Missing TEST capture identity: {row['case']}")
        identity = matches[0]
        capture_path = capture_dir / Path(identity["capture_file"]).name
        pool_identity = [item for item in pool["tests"] if item["file"] == capture_path.name]
        require(len(pool_identity) == 1, "Pool TEST capture linkage missing")
        if capture_path not in cached:
            cached[capture_path] = checked_capture(capture_path, [identity, pool_identity[0]])
        capture, all_words, positions = cached[capture_path]
        start, tokens = int(row["start"]), int(row["tokens"])
        require(start >= 0 and tokens > 0 and tokens % 16 == 0 and start + tokens <= all_words.shape[1], "Invalid TEST page extent")
        words = np.ascontiguousarray(all_words[:, start:start + tokens], dtype="<u2")
        require(words.size == row["values"] and words.nbytes == row["raw_bytes"], "TEST page count mismatch")
        packet_path = resolve_artifact(run, row["decoder_inputs"][1], "packet_checks")
        header, packet_bytes, framing_bytes = inspect_packet(packet_path)
        require(header["state_sha256"] == state["state_sha256"] and header["shape"] == list(words.shape)
                and header["start"] == start and header["metadata_bytes"] == 0 and header.get("correction", "direct") == "direct"
                and header.get("residue_bits", 0) == 0, "Transfer packet is not the matching direct baseline")
        require(header["decoded_sha256"] == hashlib.sha256(words.tobytes()).hexdigest(), "Transfer packet/source words mismatch")
        require(packet_bytes == row["packet_bytes"] and packet_bytes == row["codec_stats"]["packet_bytes"], "Transfer packet byte count mismatch")
        decoded = np.load(packet_path.with_suffix(".npy"), allow_pickle=False)
        child_path = packet_path.with_suffix(".json")
        child = json.loads(child_path.read_text())
        require(np.array_equal(decoded, words) and child["decoded_sha256"] == sha_array(words)
                and child["stats"].get("exact_checksum_verified") is True, "Recorded child decoded words mismatch")
        require([Path(p).name for p in child["inputs"]] == [state_path.name, packet_path.name], "Recorded child input linkage mismatch")
        records, offsets = [], [0]
        for layer in range(layers):
            # uint8 subtract would wrap: the original arithmetic requires int64.
            ref, wid = refs[layer].astype(np.int64), widths[layer].astype(np.int64)
            for local in range(0, tokens, 16):
                block = words[layer, local:local + 16]
                record = reference["pack_block"](block, ref, wid)
                require(np.array_equal(reference["unpack_block"](record, ref, wid), block), "Original pool round trip failed")
                records.append(record.tobytes()); offsets.append(offsets[-1] + record.size)
        payload = b"".join(records)
        expected = [item for item in pool["pages"] if item["case"] == capture_path.stem and item["token_start"] == start and item["token_count"] == tokens]
        require(len(expected) == 1 and expected[0]["exact"] and expected[0]["words"] == words.size
                and expected[0]["packed_bytes"] == len(payload), "Pool page bytes differ from recorded CPU reference")
        name = f"page_{index:03d}"
        slim = {key: value for key, value in row.items() if key not in ("codec_stats", "decoder_inputs")}
        slim["codec_stats"] = {key: value for key, value in row["codec_stats"].items() if key not in ("block_diagnostics", "layer_ideal_bits")}
        pages.append(dict(name=name, case=row["case"], start=start, tokens=tokens, shape=list(words.shape),
                          values=int(words.size), words_file=name + "_words.npy", ids_file=name + "_ids.npy",
                          positions_file=name + "_positions.npy", pool_offsets_file=name + "_pool_offsets.npy",
                          pool_payload_file=name + "_pool.bin", transfer_packet_file=name + "_transfer.bin",
                          words_sha256=hashlib.sha256(words.tobytes()).hexdigest(), words_array_sha256=sha_array(words),
                          source_capture=identity, source_capture_file_sha256=file_sha256(capture_path),
                          source_full_ids=capture["ids"].tolist(), reference_row=slim,
                          pool_reference_row=expected[0], cpu_child_report_sha256=file_sha256(child_path),
                          framing_bytes=framing_bytes, pool_payload_bytes=len(payload),
                          pool_payload_sha256=hashlib.sha256(payload).hexdigest(), transfer_packet_sha256=file_sha256(packet_path),
                          _words=words, _ids=capture["ids"][start:start + tokens], _positions=positions[start:start + tokens],
                          _offsets=np.asarray(offsets, dtype="<u8"), _pool=payload, _packet=packet_path))
    total_words = sum(page["values"] for page in pages)
    pool_bytes = sum(page["pool_payload_bytes"] for page in pages)
    packet_bytes = sum(page["reference_row"]["packet_bytes"] for page in pages)
    require(total_words == pool["aggregate"]["words"] == report["aggregate"]["block32"]["values"], "Aggregate word count mismatch")
    require(pool_bytes == pool["aggregate"]["packed_bytes"] and packet_bytes == report["aggregate"]["block32"]["packet_bytes"], "Aggregate byte count mismatch")
    require(state["resident_array_bytes"] == report["aggregate"]["block32"]["resident_decoder_state_bytes"]
            and state["serialized_bytes"] == report["aggregate"]["block32"]["serialized_decoder_state_bytes"], "Frozen state accounting mismatch")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(state_path, output / "block32.npz")
    np.savez_compressed(output / "pool_state.npz", refs=refs, widths=widths)
    for page in pages:
        for key, field in (("words", "words_file"), ("ids", "ids_file"), ("positions", "positions_file"), ("offsets", "pool_offsets_file")):
            np.save(output / page[field], page.pop("_" + key), allow_pickle=False)
        (output / page["pool_payload_file"]).write_bytes(page.pop("_pool"))
        shutil.copyfile(page.pop("_packet"), output / page["transfer_packet_file"])
    if model_manifest is not None:
        shutil.copyfile(model_manifest, output / "model_download_manifest.json")
    manifest = dict(format=FORMAT, gpu_validated=False, state_file="block32.npz", pool_state_file="pool_state.npz",
                    pool_block_tokens=16, pool_record_order="layer-major, then consecutive 16-token blocks",
                    pool_state_storage="refs and widths uint8 [layer,combined_KV_channel]; cast to int64 for signed arithmetic",
                    state=state, pages=pages,
                    expected=dict(pages=len(pages), values=total_words, raw_bf16_bytes=total_words * 2,
                                  pool_payload_bytes=pool_bytes, pool_bits_per_value=8 * pool_bytes / total_words,
                                  pool_saving_fraction=1 - pool_bytes / (2 * total_words),
                                  transfer_packet_bytes=packet_bytes, transfer_bits_per_value=8 * packet_bytes / total_words,
                                  transfer_saving_fraction=1 - packet_bytes / (2 * total_words)),
                    provenance=dict(source_results_sha256=file_sha256(report_path), source_pool_results_sha256=file_sha256(pool_report),
                                    original_pool_source_sha256=ORIGINAL_POOL_SHA256, executed_sources=executed_sources,
                                    model_identity={key:value for key,value in model_identity.items() if key != "captures"},
                                    pool_train_identities=pool["training"], source_scope=report.get("scope"),
                                    builder_sha256=file_sha256(__file__)),
                    validation=dict(pool_original_cpu_roundtrips=True, recorded_transfer_child_outputs_match_source_pages=True,
                                    transfer_entropy_decode_rerun_by_builder=False, cuda_executed=False),
                    accounting=dict(pool="Original variable-size record payloads; excludes calibration, offsets, GPU indexes/alignment/allocator",
                                    transfer="Complete original CPU packets; excludes fixed frozen state",
                                    fixture="Packaging/index/NPY headers are reproducibility artifacts, not codec savings",
                                    state="Only block32 decoder state included; no hidden-state projectors or activations"))
    (output / "README.md").write_text(markdown(manifest))
    manifest["files"] = {path.name:dict(bytes=path.stat().st_size, sha256=file_sha256(path)) for path in sorted(output.iterdir()) if path.is_file()}
    manifest["fixture_payload_bytes"] = sum(info["bytes"] for info in manifest["files"].values())
    require(all(info["bytes"] < 50 * 1024 * 1024 for info in manifest["files"].values()), "Fixture file exceeds 50MiB")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def markdown(manifest):
    total = manifest["expected"]
    return ("# Exact-input CPU-to-GPU replay fixture\n\n"
            f"Frozen {total['pages']} TEST pages contain {total['values']:,} original BF16 words. "
            "All layers and the original first/final-page positions are retained. The state and CPU transfer packets are copied byte for byte.\n\n"
            f"Original pool CPU payload: {total['pool_bits_per_value']:.6f} bits/value, {100 * total['pool_saving_fraction']:.2f}% reduction. "
            f"Complete block32 CPU transfer packets: {total['transfer_bits_per_value']:.6f} bits/value, {100 * total['transfer_saving_fraction']:.2f}% reduction.\n\n"
            "These are the earlier measured CPU rates. This fixture has not been CUDA validated. A GPU replay must verify every original word, "
            "pool payload byte and integer predictor result before making GPU correctness or speed claims. The entropy coder remains CPU unless separately implemented on GPU.\n\n"
            "Pool byte rates exclude shared calibration, offsets, escape indexes and GPU allocation/alignment. Transfer rates include packet framing "
            "but exclude fixed decoder state; manifest.json reports serialized and resident array costs separately. Fixture NPY headers, token IDs and "
            "offsets are reproducibility artifacts. No model weights, hidden activations or extra hidden-metadata networks are included.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/transfer_smoke_cpu")
    parser.add_argument("--pool-report", type=Path, default=ROOT / "benchmark_results/team_pool_reference_cpu_20261020/results.json")
    parser.add_argument("--source", type=Path, default=ROOT / "experiments/team_transfer_original/combined.py")
    parser.add_argument("--source-root", type=Path, default=ROOT / "runs/transfer_smoke_cpu_executed_source")
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "benchmark_results/team_transfer_extensions_smol_cpu_20261020/model_download_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_fixture(args.run, args.output, args.pool_report, args.source, args.source_root, args.model_manifest)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(output=str(args.output), files=len(report["files"]), bytes=report["fixture_payload_bytes"], **report["expected"]), sort_keys=True))


if __name__ == "__main__":
    main()
