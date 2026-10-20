#!/usr/bin/env python3
"""Replay frozen real KV through device pool operations and predictor components.

CUDA is required by default and never silently falls back. Explicit CPU runs
validate the same tensor code but provide no GPU timing or H2D result. The
transfer entropy coder remains CPU-only; predictor replay uses known KV context.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIRE_PREFIX = struct.Struct("<8sII")
WIRE_MAGIC = b"KVPGPU01"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while piece := source.read(1 << 20):
            digest.update(piece)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _torch():
    import torch
    return torch


def select_device(torch, requested):
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("--device must select cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback was performed")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_device(operation, torch, device, warmup, repeats):
    """Measure high-level device calls; report both CUDA events and sync wall."""
    for _ in range(warmup):
        temporary = operation()
        synchronize(torch, device)
        del temporary
    samples = []
    result = None
    for _ in range(repeats):
        result = None
        synchronize(torch, device)
        baseline = None
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        if device.type == "cuda":
            start.record()
        result = operation()
        if device.type == "cuda":
            stop.record()
        synchronize(torch, device)
        wall = time.perf_counter() - begin
        sample = {"synchronized_wall_seconds": wall, "cuda_event_ms": None,
                  "peak_allocated_bytes": None, "peak_incremental_allocated_bytes": None,
                  "peak_reserved_bytes": None}
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            sample.update(cuda_event_ms=float(start.elapsed_time(stop)),
                          peak_allocated_bytes=int(peak),
                          peak_incremental_allocated_bytes=int(max(0, peak - baseline)),
                          peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
        samples.append(sample)
    report = {"warmup": warmup, "repeats": repeats, "samples": samples,
              "wall_median_seconds": float(np.median([x["synchronized_wall_seconds"] for x in samples])),
              "cuda_event_median_ms": None, "peak_incremental_allocated_bytes": None,
              "scope": "high-level tensor call with enabled validation and resident inputs/state; CUDA events measure stream interval including launch gaps, synchronized wall includes host orchestration"}
    if device.type == "cuda":
        report["cuda_event_median_ms"] = float(np.median([x["cuda_event_ms"] for x in samples]))
        report["peak_incremental_allocated_bytes"] = max(x["peak_incremental_allocated_bytes"] for x in samples)
    return result, report


def local_file(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("fixture file is missing or outside the fixture: " + str(relative))
    return candidate


def packet_metadata(packet, model):
    from qwen_kv_bench.transfer_reference import PREFIX, MAGIC, FORMAT
    if len(packet) < PREFIX.size:
        raise ValueError("truncated CPU transfer packet")
    magic, size = PREFIX.unpack_from(packet)
    if magic != MAGIC or size > 1 << 20 or len(packet) < PREFIX.size + size:
        raise ValueError("invalid CPU transfer packet framing")
    header = json.loads(packet[PREFIX.size:PREFIX.size + size])
    if header.get("format") != FORMAT or header.get("state_sha256") != model.state_sha256:
        raise ValueError("CPU packet does not match frozen transfer state")
    payload = packet[PREFIX.size + size:]
    if len(payload) != header["metadata_bytes"] + header["stream_bytes"] or hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise ValueError("CPU transfer packet payload checksum mismatch")
    metadata = None
    if model.metadata_rank:
        shape = tuple(header["shape"][:2]) + (model.metadata_rank,)
        if header["kind"] != "range" or header["metadata_bytes"] != np.prod(shape):
            raise ValueError("metadata predictor requires transmitted metadata in a range packet")
        metadata = np.frombuffer(payload[:header["metadata_bytes"]], dtype=np.int8).reshape(shape).copy()
    return header, metadata


def load_fixture(path):
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    ledger = manifest.get("files", [])
    if isinstance(ledger, dict):
        ledger = [dict(value, path=name) for name, value in ledger.items()]
    for entry in ledger:
        item = local_file(root, entry.get("path", entry.get("file")))
        if "bytes" in entry and item.stat().st_size != entry["bytes"]:
            raise ValueError("fixture file length mismatch: " + item.name)
        if "sha256" in entry and sha_file(item) != entry["sha256"]:
            raise ValueError("fixture checksum mismatch: " + item.name)
    pages = []
    for index, entry in enumerate(manifest["pages"]):
        words = np.load(local_file(root, entry["words_file"]), allow_pickle=False)
        packet_path = local_file(root, entry["transfer_packet_file"])
        pages.append({"name": entry.get("name", f"page{index}"), "case": entry.get("case"),
                      "start": int(entry["start"]), "words": words,
                      "packet": packet_path.read_bytes(), "reference_row": entry.get("reference_row", {}),
                      "pool_payload": local_file(root, entry["pool_payload_file"]).read_bytes() if entry.get("pool_payload_file") else None,
                      "pool_offsets": np.load(local_file(root, entry["pool_offsets_file"]), allow_pickle=False) if entry.get("pool_offsets_file") else None})
    with np.load(local_file(root, manifest["pool_state_file"]), allow_pickle=False) as source:
        refs, widths = source["refs"].copy(), source["widths"].copy()
    return {"state_path": local_file(root, manifest["state_file"]), "refs": refs, "widths": widths,
            "pages": pages, "provenance": {"input": "frozen replay fixture", "manifest": str(manifest_path),
                "manifest_sha256": sha_file(manifest_path), "fixture_format": manifest.get("format"),
                "files_checked": len(ledger), "fixture_gpu_validated": manifest.get("gpu_validated", False)}}


def original_pool(words, refs, widths, tile_tokens=16):
    from scripts.bench_team_pool_reference import load_reference
    reference = load_reference(ROOT / "experiments/team_transfer_original/combined.py", [], words.shape[2], tile_tokens)
    refs, widths = np.asarray(refs, dtype=np.int64), np.asarray(widths, dtype=np.int64)
    records = [reference["pack_block"](words[layer, start:start + tile_tokens], refs[layer], widths[layer])
               for layer in range(words.shape[0]) for start in range(0, words.shape[1], tile_tokens)]
    offsets = np.concatenate(([0], np.cumsum([len(item) for item in records]))).astype(np.int64)
    return b"".join(records), offsets


def load_reference_run(path, variant):
    from scripts.bench_team_pool_reference import load_reference
    root = Path(path).resolve()
    report = json.loads((root / "results.json").read_text())
    rows = [row for row in report["test_rows"] if row["variant"] == variant]
    if not rows:
        raise ValueError("reference run contains no TEST rows for " + variant)
    captures = report["provenance"]["model_identity"].get("captures", [])
    by_name = {entry["name"]: root / "captures" / Path(entry["capture_file"]).name for entry in captures}
    def load_act(path):
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key].copy() for key in ("K_bits", "V_bits", "Hk", "hd", "base")}
    train = [load_act(root / "captures" / Path(entry["capture_file"]).name)
             for entry in captures if entry.get("split") == "train"]
    if not train:
        raise ValueError("--reference-run requires full captures/states/packets; use --replay-fixture for a portable input")
    reference = load_reference(ROOT / "experiments/team_transfer_original/combined.py", train, 2 * train[0]["K_bits"].shape[2])
    fitted = [reference["fit_pool"](layer) for layer in range(train[0]["K_bits"].shape[0])]
    refs = np.stack([value[0] for value in fitted]).astype(np.uint8)
    widths = np.stack([value[1] for value in fitted]).astype(np.uint8)
    pages = []
    state_paths = set()
    for index, row in enumerate(rows):
        act = load_act(by_name[row["case"]])
        start, stop = row["start"], row["start"] + row["tokens"]
        words = np.concatenate((act["K_bits"][:, start:stop], act["V_bits"][:, start:stop]), axis=2)
        state_paths.add(root / "states" / Path(row["decoder_inputs"][0]).name)
        packet_path = root / "packet_checks" / Path(row["decoder_inputs"][1]).name
        pages.append({"name": f"{row['case']}_{start}", "case": row["case"], "start": start,
                      "words": words, "packet": packet_path.read_bytes(), "reference_row": row,
                      "pool_payload": None, "pool_offsets": None})
    if len(state_paths) != 1:
        raise ValueError("one frozen predictor state is required for a reference-run variant")
    return {"state_path": state_paths.pop(), "refs": refs, "widths": widths, "pages": pages,
            "provenance": {"input": "full reference run", "results_sha256": sha_file(root / "results.json"),
                           "variant": variant, "reference_run": str(root)}}


def serialize_pool(packet):
    """Complete paid transport wrapper; original pool payload remains unchanged."""
    descriptor = {"format": packet.format, "shape": list(packet.shape), "tile_tokens": packet.tile_tokens,
                  "offsets_count": int(packet.offsets.numel()), "offset_dtype": "little-endian int64",
                  "payload_bytes": packet.payload_nbytes}
    header = canonical(descriptor).encode()
    padding = (-(WIRE_PREFIX.size + len(header))) % 8
    prefix = WIRE_PREFIX.pack(WIRE_MAGIC, len(header), padding)
    body_start = WIRE_PREFIX.size + len(header) + padding
    offsets = packet.offsets.detach().cpu().numpy().astype("<i8", copy=False).tobytes()
    payload = packet.payload.detach().cpu().numpy().tobytes()
    blob = prefix + header + b"\0" * padding + offsets + payload
    return blob, parse_pool_wire(blob)


def parse_pool_wire(blob):
    """Read all geometry from the paid header before copying the flat wire."""
    if len(blob) < WIRE_PREFIX.size:
        raise ValueError("truncated complete pool wire")
    magic, size, padding = WIRE_PREFIX.unpack_from(blob)
    if magic != WIRE_MAGIC or size > 1 << 20 or padding > 7:
        raise ValueError("invalid complete pool wire header")
    begin = WIRE_PREFIX.size + size + padding
    if begin > len(blob) or begin % 8 or any(blob[WIRE_PREFIX.size + size:begin]):
        raise ValueError("invalid complete pool wire alignment")
    descriptor = json.loads(blob[WIRE_PREFIX.size:WIRE_PREFIX.size + size])
    shape = descriptor["shape"]
    tile = descriptor["tile_tokens"]
    if (descriptor["format"] != "team_pool_original_v1" or descriptor["offset_dtype"] != "little-endian int64"
        or len(shape) != 3 or min(shape) < 1 or tile < 1 or shape[1] % tile
        or descriptor["offsets_count"] != shape[0] * (shape[1] // tile) + 1):
        raise ValueError("invalid complete pool wire geometry")
    index_bytes = descriptor["offsets_count"] * 8
    if descriptor["payload_bytes"] < 1 or begin + index_bytes + descriptor["payload_bytes"] != len(blob):
        raise ValueError("complete pool wire length mismatch")
    return {**descriptor, "header_bytes": begin, "index_bytes": index_bytes,
            "wire_bytes": len(blob), "payload_start": begin + index_bytes}


def pool_from_wire_tensor(buffer, descriptor):
    """Use already parsed host geometry; GPU JSON parsing is not implemented."""
    from qwen_kv_bench.transfer_pool_gpu import PoolPacket
    begin = descriptor["header_bytes"]
    end = begin + descriptor["index_bytes"]
    offsets = buffer[begin:end].view(_torch().int64)
    payload = buffer[end:]
    if payload.numel() != descriptor["payload_bytes"]:
        raise ValueError("serialized pool tensor length mismatch")
    return PoolPacket(payload, offsets, tuple(descriptor["shape"]), descriptor["tile_tokens"], descriptor["format"])


def h2d_comparison(words, blob, descriptor, state, torch, device, args):
    from qwen_kv_bench.transfer_pool_gpu import decode_pool
    if device.type != "cuda":
        return {"measured": False, "reason": "explicit CPU validation; no H2D transfer was measured"}
    begin = time.perf_counter()
    host_raw = torch.from_numpy(words.copy()).view(torch.uint8).reshape(-1).pin_memory()
    host_packed = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy()).pin_memory()
    raw_destination = torch.empty(host_raw.shape, dtype=torch.uint8, device=device)
    packed_destination = torch.empty(host_packed.shape, dtype=torch.uint8, device=device)
    synchronize(torch, device)
    prepare = time.perf_counter() - begin
    _, raw_time = measure_device(lambda: raw_destination.copy_(host_raw, non_blocking=True), torch, device, args.warmup, args.repeats)
    _, packed_time = measure_device(lambda: packed_destination.copy_(host_packed, non_blocking=True), torch, device, args.warmup, args.repeats)
    def copy_unpack():
        packed_destination.copy_(host_packed, non_blocking=True)
        return decode_pool(pool_from_wire_tensor(packed_destination, descriptor), state, tiles_per_batch=args.tiles_per_batch)
    restored, chain_time = measure_device(copy_unpack, torch, device, args.warmup, args.repeats)
    if not np.array_equal(restored.cpu().numpy(), words):
        raise AssertionError("H2D complete pool wire failed exact reconstruction")
    if not np.array_equal(raw_destination.view(torch.uint16).reshape(words.shape).cpu().numpy(), words):
        raise AssertionError("raw pinned H2D copy changed BF16 patterns")
    for count, result in ((words.nbytes, raw_time), (len(blob), packed_time)):
        result["copied_bytes"] = count
        result["effective_GB_per_second"] = count / result["wall_median_seconds"] / 1e9
    return {"measured": True, "pinned_sources": True, "same_preallocated_copy_conditions": True,
            "cpu_pinned_and_destination_preparation_seconds": prepare, "raw_bf16": raw_time,
            "complete_pool_wire": packed_time, "complete_pool_wire_then_unpack": chain_time,
            "bit_exact": True, "host_descriptor_parsed_before_copy": True,
            "shared_pool_state_already_resident": True, "wire_to_raw_byte_ratio": len(blob) / words.nbytes}


def summary(result):
    if result["status"] != "completed":
        return f"# Device benchmark failed\n\n{result['error_type']}: {result['error']}\n\nNo successful GPU result is claimed.\n"
    lines = ["# Frozen KV device component replay", "",
             f"Device: `{result['device']['selected']}`. GPU measured: **{result['gpu_validated']}**. "
             "Transfer entropy coding remains CPU-only; predictor replay reads known KV.", "",
             "| Page | Pool payload bpv | Complete wire bpv | Pool exact | Predictor exact |",
             "|---|---:|---:|---|---|"]
    for row in result["rows"]:
        pool = row.get("pool")
        lines.append(f"| {row['name']} | {pool['stats']['payload_bits_per_value']:.5f} | "
                     f"{pool['wire']['wire_bits_per_value']:.5f} | {pool['bit_exact']} | "
                     f"{row.get('predictor', {}).get('exact_mu', 'not measured')} |" if pool else
                     f"| {row['name']} | — | — | not measured | {row['predictor']['exact_mu']} |")
    aggregate = result["aggregate"]
    lines += ["", f"Compared {aggregate['unique_source_values']:,} original BF16 words across "
              f"{aggregate['pages']} pages. CPU transfer packets: "
              f"**{aggregate['cpu_transfer_packet_bits_per_value']:.6f} bits/value**. "
              "This packet rate is separate from the device pool format.", "",
              "Synchronized wall medians in milliseconds (warm state):", "",
              "| Page | Pool encode | Pool decode | Raw H2D | Packed H2D + unpack | Predictor replay |",
              "|---|---:|---:|---:|---:|---:|"]
    def milliseconds(timing):
        value = timing.get("wall_median_seconds") if timing else None
        return f"{value * 1000:.4f}" if value is not None else "not measured"
    for row in result["rows"]:
        pool = row.get("pool", {})
        h2d = pool.get("h2d", {})
        lines.append(f"| {row['name']} | {milliseconds(pool.get('encode'))} | "
                     f"{milliseconds(pool.get('decode'))} | {milliseconds(h2d.get('raw_bf16'))} | "
                     f"{milliseconds(h2d.get('complete_pool_wire_then_unpack'))} | "
                     f"{milliseconds(row.get('predictor', {}).get('timing'))} |")
    cold = result["cold_state_initialization"]
    predictor_state = cold.get("predictor") or {}
    lines += ["", f"Persistent device state: pool **{cold['pool_device_tensor_bytes']:,} bytes**; "
              f"predictor **{predictor_state.get('persistent_tensor_bytes', 0):,} bytes**. "
              "The full CPU decoder state, cold setup, CUDA event intervals and per-call "
              "incremental peaks (including output and scratch) are recorded in `results.json`."]
    lines += ["", "Pool payload includes each original record's 2-byte escape header. Complete wire also "
              "includes a serialized geometry header and the INT64 block offset directory. Shared pool calibration "
              "and predictor tensors are reported separately. CUDA samples report events and synchronized wall time "
              "with warm shared state; cold initialization is separate. CPU runs leave H2D and CUDA timings unmeasured.", "",
              "These components do not demonstrate a GPU transfer entropy decoder, compressed-KV attention, "
              "or serving/concurrency speedup. CPU transfer bytes are reported separately from device component coverage.", ""]
    return "\n".join(lines)


def execute(args, output, log):
    torch = _torch()
    torch.set_num_threads(args.threads)
    device = select_device(torch, args.device)
    log("loading frozen replay input")
    bundle = load_fixture(args.replay_fixture) if args.replay_fixture else load_reference_run(args.reference_run, args.variant)
    from qwen_kv_bench.transfer_reference import load_model, decode, model_statistics
    from qwen_kv_bench.transfer_pool_gpu import make_state, encode_pool, decode_pool, packet_stats
    from qwen_kv_bench.transfer_gpu_predictor import TorchTransferPredictor, numpy_prediction_sweep
    begin = time.perf_counter()
    model = load_model(bundle["state_path"])
    cpu_state_load = time.perf_counter() - begin
    cpu_statistics = model_statistics(model)
    pages = bundle["pages"][:args.max_pages] if args.max_pages else bundle["pages"]
    if not pages:
        raise ValueError("input contains no replay pages")
    components = set(args.components.split(","))
    begin = time.perf_counter()
    pool_state = make_state(bundle["refs"], bundle["widths"], device=device) if "pool" in components else None
    synchronize(torch, device)
    pool_state_load = time.perf_counter() - begin
    begin = time.perf_counter()
    predictor = TorchTransferPredictor.from_reference(model, device=device, coefficient_dtype=args.coefficient_dtype,
        input_chunk=args.input_chunk, token_chunk=args.token_chunk) if "predictor" in components else None
    synchronize(torch, device)
    predictor_state_load = time.perf_counter() - begin
    device_info = {"selected": str(device), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                   "name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                   "cuda_available": bool(torch.cuda.is_available())}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_info.update(total_memory_bytes=int(properties.total_memory), compute_capability=[properties.major, properties.minor])
    result = {"format": "transfer_gpu_component_replay_v1", "status": "running", "arguments": vars(args),
        "device": device_info, "gpu_validated": False, "gpu_tested": False, "explicit_cpu_validation": device.type == "cpu",
        "provenance": bundle["provenance"], "rows": [],
        "cold_state_initialization": {"cpu_npz_load_seconds": cpu_state_load,
            "serialized_decoder_state_bytes": bundle["state_path"].stat().st_size,
            "cpu_full_decoder_array_bytes": cpu_statistics["frozen_array_bytes"],
            "pool_device_state_initialization_seconds": pool_state_load,
            "pool_device_tensor_bytes": pool_state.physical_nbytes if pool_state else 0,
            "predictor_device_state_initialization_seconds": predictor_state_load,
            "predictor": predictor.storage_statistics() if predictor else None,
            "scope": "one cold construction; includes validation, Python and copies; not pure PCIe timing"},
        "coverage": {"device_pool_encode": "pool" in components, "device_pool_decode": "pool" in components,
            "device_predictor_replay": "predictor" in components, "gpu_transfer_entropy_coder": False,
            "compressed_attention": False, "model_forward_measured": False,
            "predictor_context": "original known KV, not entropy-decoded symbols"}}
    for index, page in enumerate(pages):
        words = np.asarray(page["words"])
        if words.dtype != np.uint16 or words.ndim != 3 or words.shape[1] % 16:
            raise ValueError("replay pages require original uint16 [layers,tokens,channels] and complete 16-token tiles")
        log(f"page {index + 1}/{len(pages)}: {page['name']} CPU packet validation")
        header, metadata = packet_metadata(page["packet"], model)
        if tuple(header["shape"]) != words.shape or header["start"] != page["start"]:
            raise ValueError("page geometry/position differs from its CPU transfer packet")
        validation = {"performed": not args.skip_cpu_transfer_validation, "gpu": False}
        if not args.skip_cpu_transfer_validation:
            begin = time.perf_counter()
            decoded, _ = decode(model, page["packet"])
            validation.update(seconds=time.perf_counter() - begin, bit_exact=bool(np.array_equal(decoded, words)))
            if not validation["bit_exact"]:
                raise AssertionError("CPU transfer packet does not reproduce source page")
        row = {"name": page["name"], "case": page["case"], "start": page["start"],
            "shape": list(words.shape), "values": int(words.size), "raw_bf16_bytes": int(words.nbytes),
            "cpu_transfer": {"packet_bytes": len(page["packet"]), "bits_per_value": len(page["packet"]) * 8 / words.size,
                "framing_bytes": len(page["packet"]) - header["metadata_bytes"] - header["stream_bytes"],
                "metadata_bytes": header["metadata_bytes"], "entropy_stream_bytes": header["stream_bytes"],
                "resident_full_decoder_array_bytes": cpu_statistics["frozen_array_bytes"],
                "validation": validation, "codec_device": "cpu"}}
        device_words = torch.from_numpy(words.copy()).to(device)
        device_metadata = torch.from_numpy(metadata.copy()).to(device) if metadata is not None else None
        synchronize(torch, device)
        if pool_state is not None:
            log(f"page {index + 1}/{len(pages)}: pool tensor encode/decode on {device}")
            packet, encode_time = measure_device(lambda: encode_pool(device_words, pool_state, tile_tokens=16,
                tiles_per_batch=args.tiles_per_batch), torch, device, args.warmup, args.repeats)
            restored, decode_time = measure_device(lambda: decode_pool(packet, pool_state,
                tiles_per_batch=args.tiles_per_batch), torch, device, args.warmup, args.repeats)
            exact = bool(np.array_equal(restored.cpu().numpy(), words))
            payload = packet.payload.cpu().numpy().tobytes()
            offsets = packet.offsets.cpu().numpy()
            expected_payload, expected_offsets = original_pool(words, bundle["refs"], bundle["widths"])
            original_bytes = payload == expected_payload and np.array_equal(offsets, expected_offsets)
            fixture_bytes = (page["pool_payload"] is None or payload == page["pool_payload"])
            fixture_offsets = (page["pool_offsets"] is None or np.array_equal(offsets, page["pool_offsets"]))
            if not (exact and original_bytes and fixture_bytes and fixture_offsets):
                raise AssertionError("device pool did not match original bytes/offsets and exact words")
            begin = time.perf_counter()
            blob, descriptor = serialize_pool(packet)
            synchronize(torch, device)
            serialize_seconds = time.perf_counter() - begin
            descriptor["wire_bits_per_value"] = 8 * descriptor["wire_bytes"] / words.size
            row["pool"] = {"bit_exact": exact, "original_payload_and_offsets_equal": original_bytes,
                "fixture_payload_equal": fixture_bytes, "fixture_offsets_equal": fixture_offsets,
                "stats": packet_stats(packet, pool_state), "wire": descriptor,
                "host_wire_serialization_seconds_including_D2H": serialize_seconds,
                "encode": encode_time, "decode": decode_time,
                "h2d": h2d_comparison(words, blob, descriptor, pool_state, torch, device, args)}
        if predictor is not None:
            log(f"page {index + 1}/{len(pages)}: exact predictor replay on {device}")
            begin = time.perf_counter()
            expected = numpy_prediction_sweep(model, words, page["start"], metadata)
            oracle_seconds = time.perf_counter() - begin
            means, predictor_time = measure_device(lambda: predictor.prediction_sweep(device_words, page["start"], device_metadata),
                torch, device, args.warmup, args.repeats)
            actual = means.cpu().numpy()
            exact_mu = bool(np.array_equal(expected, actual))
            if not exact_mu:
                raise AssertionError("device fixed-point predictor differs from CPU means")
            row["predictor"] = {"exact_mu": exact_mu, "values_compared": int(expected.size),
                "cpu_oracle_seconds": oracle_seconds, "timing": predictor_time,
                "scope": "feature/mean replay with known KV; no probability model or entropy coder"}
        result["rows"].append(row)
        write_json(output / "checkpoint.json", result)
        log(f"page {index + 1}/{len(pages)} passed")
    result["status"] = "completed"
    result["gpu_validated"] = device.type == "cuda"
    result["gpu_tested"] = device.type == "cuda"
    values = sum(row["values"] for row in result["rows"])
    result["aggregate"] = {"pages": len(pages), "unique_source_values": values,
        "cpu_transfer_packet_bits_per_value": 8 * sum(row["cpu_transfer"]["packet_bytes"] for row in result["rows"]) / values}
    if pool_state is not None:
        result["aggregate"].update(pool_payload_bits_per_value=8 * sum(row["pool"]["stats"]["payload_bytes"] for row in result["rows"]) / values,
            pool_complete_wire_bits_per_value=8 * sum(row["pool"]["wire"]["wire_bytes"] for row in result["rows"]) / values,
            all_pool_exact=all(row["pool"]["bit_exact"] for row in result["rows"]))
    if predictor is not None:
        result["aggregate"]["all_predictor_mu_exact"] = all(row["predictor"]["exact_mu"] for row in result["rows"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay-fixture", help="frozen fixture directory or manifest.json; no model download")
    source.add_argument("--reference-run", help="full existing run with captures, states and packets")
    parser.add_argument("--variant", default="block32", help="reference-run variant; fixture state is fixed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tiles-per-batch", type=int, default=16)
    parser.add_argument("--input-chunk", type=int, default=64)
    parser.add_argument("--token-chunk", type=int, default=32)
    parser.add_argument("--coefficient-dtype", choices=("preserve", "int32"), default="preserve")
    parser.add_argument("--components", default="pool,predictor")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--skip-cpu-transfer-validation", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0 or min(args.threads, args.tiles_per_batch, args.input_chunk, args.token_chunk) < 1:
        parser.error("repeats/threads/chunks must be positive and warmup nonnegative")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("max-pages must be positive")
    if not set(args.components.split(",")) <= {"pool", "predictor"} or not args.components:
        parser.error("components must select pool,predictor or one of them")
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (output / "run.log").open("a") as stream:
            stream.write(line + "\n")
    log("starting frozen device replay")
    try:
        result = execute(args, output, log)
        code = 0
    except Exception as error:
        log(type(error).__name__ + ": " + str(error))
        result = {"format": "transfer_gpu_component_replay_v1", "status": "failed", "arguments": vars(args),
            "gpu_validated": False, "gpu_tested": False, "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "partial_checkpoint": (output / "checkpoint.json").exists()}
        code = 2
    sources = [Path(__file__), ROOT / "qwen_kv_bench/transfer_pool_gpu.py",
               ROOT / "qwen_kv_bench/transfer_gpu_predictor.py", ROOT / "qwen_kv_bench/transfer_reference.py",
               ROOT / "experiments/team_transfer_original/combined.py"]
    result["sources"] = [{"path": str(path.relative_to(ROOT)), "sha256": sha_file(path)} for path in sources]
    write_json(output / "results.json", result)
    write_json(output / "source_manifest.json", {"sources": result["sources"]})
    (output / "summary.md").write_text(summary(result))
    if result["status"] == "completed":
        with (output / "rows.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ["name", "start", "values", "raw_bf16_bytes", "cpu_transfer_bpv", "pool_payload_bpv", "pool_wire_bpv", "predictor_exact"])
            writer.writeheader()
            for row in result["rows"]:
                writer.writerow({"name": row["name"], "start": row["start"], "values": row["values"],
                    "raw_bf16_bytes": row["raw_bf16_bytes"], "cpu_transfer_bpv": row["cpu_transfer"]["bits_per_value"],
                    "pool_payload_bpv": row.get("pool", {}).get("stats", {}).get("payload_bits_per_value"),
                    "pool_wire_bpv": row.get("pool", {}).get("wire", {}).get("wire_bits_per_value"),
                    "predictor_exact": row.get("predictor", {}).get("exact_mu")})
    log(f"{result['status']}: {output / 'summary.md'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
