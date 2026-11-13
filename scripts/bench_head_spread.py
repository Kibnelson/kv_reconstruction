#!/usr/bin/env python3
"""Matched GPU test of T's head-aware spread; input captures are read-only.

TRAIN fits the unchanged mean predictor and the proposed head map. The first
half of the tables capture calibrates nominal spread; the second half selects
alpha using complete encoded packets. TEST only evaluates frozen choices.
Original T files and original run folders are never written. CPU execution is
available only with --software-check and never produces a GPU decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "d95fd10a1fae447d47c4ac6abaf77b90af1b7952"
PINNED_FILES = {
    "kv_compare/layerwise.py": "028acfea2de3cd0dc180bca576b739506c5d521a",
    "kv_compare/fused.py": "711e78ca2a6277c02e45cce34f9aa9627d31fe33",
    "kv_compare/gpu_coder.py": "f0f8976e235bf737cc661b9d2020be3cabbff618",
    "kv_compare/predict.py": "baf19b141d18308420a80ea86037d37e235315b3",
    "kv_compare/grouped.py": "885e0399f3865311ff69c9ee4eaf0cecab9398e6",
    "kv_compare/core.py": "05bd76c72f07d519d7c1f38667dff2fec3ed7c8e",
}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def check_base():
    result = []
    for name, expected in PINNED_FILES.items():
        data = (ROOT / name).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != expected:
            raise ValueError("baseline source changed: " + name)
        result.append({"path": name, "git_blob_sha": digest, "sha256": hashlib.sha256(data).hexdigest()})
    return result


def select_device(requested, software):
    import torch
    interpreted = os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")
    device = torch.device(requested)
    if device.type == "cuda":
        if interpreted or software:
            raise RuntimeError("GPU results require compiled CUDA kernels; unset TRITON_INTERPRET")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU fallback was performed")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    elif device.type != "cpu" or not software or not interpreted:
        raise ValueError("CPU use requires --software-check and TRITON_INTERPRET=1")
    return device


class Window:
    def __init__(self, doc, start, stop, name=None):
        self.name = name or doc.name
        self.K, self.V = doc.K[:, start:stop], doc.V[:, start:stop]
        self.ids = doc.ids[start:stop]
        self.Q = None if doc.Q is None else doc.Q[:, start:stop]
        self.layers, self.tokens, self.columns = self.K.shape
        self.position = start

    def words(self, layer, device, rows=None):
        import torch
        from kv_compare.core import as_words
        stop = self.tokens if rows is None else rows
        return torch.cat((as_words(np.asarray(self.K[layer, :stop]), device),
                          as_words(np.asarray(self.V[layer, :stop]), device)), dim=1)


def load_words(doc, layers, device):
    import torch
    return torch.stack([doc.words(layer, device).to(torch.int16) for layer in range(layers)])


def raw_digest(doc, layers):
    h = hashlib.sha256()
    for layer in range(layers):
        h.update(np.concatenate((doc.K[layer], doc.V[layer]), axis=1).astype("<u2", copy=False).tobytes())
    return h.hexdigest()


def measure(operation, device, warmup, repeats):
    import torch
    for _ in range(warmup):
        value = operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del value
    samples, value = [], None
    for _ in range(repeats):
        value = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        if device.type == "cuda":
            start.record()
        value = operation()
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize(device)
        samples.append({"wall_ms": 1000 * (time.perf_counter() - begin),
            "cuda_ms": start.elapsed_time(end) if device.type == "cuda" else None,
            "peak_incremental_allocated_bytes": max(0, torch.cuda.max_memory_allocated(device) - baseline)
                if device.type == "cuda" else None})
    return value, {"warmup": warmup, "repeats": repeats, "samples": samples,
        "wall_median_ms": float(np.median([x["wall_ms"] for x in samples])),
        "cuda_median_ms": float(np.median([x["cuda_ms"] for x in samples])) if device.type == "cuda" else None,
        "scope": "resident inputs, warm state, original allocator/kernel launch path; CUDA interval includes host launch gaps"}


def flattened_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return sum((flattened_tensors(v) for v in value.values()), [])
    if isinstance(value, (tuple, list)):
        return sum((flattened_tensors(v) for v in value), [])
    return []


def codec_at(constants, C, hd, TB, policy, tokens, position, legacy=False):
    from kv_compare.head_spread import HeadSpreadCodec
    from kv_compare.layerwise import LayerwiseOneStep
    k = dict(constants)
    if position < 0 or position + tokens > k["cos"].shape[0]:
        raise ValueError("window exceeds frozen RoPE tables")
    k["cos"] = k["cos"][position:position + tokens]
    k["sin"] = k["sin"][position:position + tokens]
    if legacy:
        return LayerwiseOneStep(k, C, hd, tokens_per_program=TB)
    return HeadSpreadCodec(k, C, hd, policy, tokens_per_program=TB)


def encode_packet(codec, words, predictor_hash, policy, word_hash, position):
    import torch
    from kv_compare.head_spread_wire import pack
    segments, capacity = codec.layout(words.shape[1])
    # The baseline has up to seven alignment words. Zero the arena so they
    # cannot carry uninitialized data and byte-equality checks are repeatable.
    arena = torch.zeros(segments * capacity, dtype=torch.int16, device=words.device)
    out, starts = codec.encode(words, out=arena)
    decoded = codec.decode(out, starts, words.shape[1])
    if not torch.equal(decoded, words):
        raise AssertionError("encoded packet changed BF16 words")
    packet, cost = pack(codec, out, starts, words.shape[1], predictor_hash, policy, word_hash, position)
    return packet, cost


def summarize(result):
    if result["status"] != "completed":
        return "# Head-spread experiment failed\n\n" + result["error"] + "\n"
    lines = ["# T: head-aware probability-width experiment", "",
        f"GPU measured: **{result['gpu_executed']}**. Full model: **{result['full_model']}**. "
        "No existing T branch, source or run artifact was modified.", "",
        "| Predictor | Variant | Complete bpv | Bytes saved vs original | Decode ratio | Extra policy bytes |",
        "|---|---|---:|---:|---:|---:|"]
    for mode, table in result["aggregate"].items():
        for name, row in table.items():
            lines.append(f"| {mode} | {name} | {row['bits_per_value']:.6f} | {row['bytes_saved_vs_legacy']} | "
                f"{row['decode_ratio_vs_legacy']:.3f} | {row['extra_policy_tensor_bytes']} |")
    lines += ["", "Only frozen DEV-selected head_selected/learned_selected are candidates. Shifted-head rows "
        "are negative controls. Original and matched globals must have identical packets. Shared predictor state, "
        "policy state, scratch and output are separate from packet bpv. No token-specific hint is sent.", "",
        "An actual GPU, exact packet-only reconstruction, a positive DEV and TEST byte gain, and the declared "
        "decode-time budget are required for a candidate to pass the experiment gate. Passing is evidence for "
        "review, not permission to merge. Cross-dataset confirmation remains necessary.", ""]
    for mode, decisions in result["decisions"].items():
        lines.append(f"{mode}: `{json.dumps(decisions, sort_keys=True)}`")
    return "\n".join(lines) + "\n"


def execute(args, output, log):
    import torch
    import triton
    from kv_compare import fused, grouped
    from kv_compare.predict import Study, Doc
    from kv_compare.head_spread import Policy, HeadSpreadCodec, fit_head_map
    from kv_compare.head_spread_wire import consts_digest, save_state, pack, unpack, tensors_bytes, synchronize

    device = select_device(args.device, args.software_check)
    torch.set_num_threads(args.threads)
    source_evidence = check_base()
    device_checks = None
    if device.type == "cuda":
        from kv_compare.head_spread_checks import check
        log("checking compiled CUDA kernels, stream parity and every BF16 word pattern")
        device_checks = [check(device), check(device, all_patterns=True)]
        write_json(output / "device_checks.json", device_checks)
    captures = Path(args.captures).resolve()
    names = args.train_docs.split(",")
    tests = args.test_docs.split(",")
    if (len(set(names)) != len(names) or len(set(tests)) != len(tests)
            or set(names) & set(tests) or args.tables_doc in names + tests):
        raise ValueError("TRAIN, tables and TEST document identities must be distinct")
    study = Study(captures, device, names, args.tables_doc, [], args.max_layers, log=log)
    original_tables = study.tables
    total_dev = original_tables.tokens
    half = total_dev // 2
    dev_tokens = min(args.dev_tokens or half, half)
    if dev_tokens < 32 or dev_tokens % args.tile_tokens:
        raise ValueError("DEV halves must each hold a multiple of tile_tokens and at least 32 tokens")
    study.tables = Window(original_tables, 0, dev_tokens, args.tables_doc + "_calibration")
    dev = Window(original_tables, half, half + dev_tokens, args.tables_doc + "_selection")
    study.train = [Window(d, 0, min(args.train_tokens or d.tokens, d.tokens)) for d in study.train]
    if any(d.tokens < 32 or d.tokens % args.tile_tokens for d in study.train):
        raise ValueError("TRAIN captures must hold complete token tiles")
    test_docs = [Window(Doc(captures, name), 0, args.tokens) for name in tests]
    if any(d.tokens != args.tokens for d in test_docs):
        raise ValueError("TEST capture has fewer than the requested tokens")
    inputs = []
    for name in names + [args.tables_doc] + tests:
        for suffix in ("K_bits.npy", "V_bits.npy", "ids.npy"):
            p = captures / f"{name}_{suffix}"
            inputs.append({"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                           "sha256": file_sha(p)})
    inputs.append({"path": str(captures / "meta.json"), "sha256": file_sha(captures / "meta.json")})
    log("fitting the original T predictor; nominal spread uses tables first half only")
    constants = fused.prepare(study, log=log)
    max_position = max(args.tokens, max(d.tokens for d in study.train), half + dev_tokens)
    cos, sin = study.rope.tables(torch.arange(max_position, device=device))
    constants["cos"], constants["sin"] = cos.float().contiguous(), sin.float().contiguous()
    # No future phase may fit or select from a TEST tensor. TEST docs have not
    # been loaded onto the GPU at this point.
    result = {"format": "t_head_spread_experiment_v1", "status": "running", "base_commit": BASE,
        "arguments": vars(args), "gpu_executed": device.type == "cuda", "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU interpreter",
        "torch": torch.__version__, "triton": triton.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "model_identity": study.meta, "layers": study.L,
        "full_model": study.L == original_tables.layers == int(study.meta["layers"]),
        "split": {"train_docs": names, "train_tokens_each": [d.tokens for d in study.train],
                  "nominal_spread_range": [0, dev_tokens], "selection_range": [half, half + dev_tokens],
                  "tables_doc": args.tables_doc, "test_docs": tests, "test_tokens_each": args.tokens},
        "inputs": inputs, "unchanged_baseline_sources": source_evidence,
        "device_checks": device_checks,
        "dev_rows": [], "selected_on_dev": {}, "test_rows": [], "aggregate": {}, "decisions": {},
        "test_used_for_selection": False, "automatic_merge": False}
    for mode in args.modes.split(","):
        log("predictor mode: " + mode)
        k = constants if mode == "dense" else grouped.prepare_groups(study, constants, study.W // study.hd, log=log)
        model = HeadSpreadCodec(k, study.C, study.hd, Policy(), tokens_per_program=args.tile_tokens)
        records = []
        for doc in study.train:
            log(f"TRAIN head statistics: {mode} {doc.name}")
            words = load_words(doc, study.L, device)
            features, sums = model.statistics(words)
            records.append(sums.cpu().numpy().copy())
            del words, features, sums
        learned = fit_head_map(records, study.hd, args.ridge)
        del records
        predictor_hash = consts_digest(k, study.C, study.hd, args.tile_tokens, "uniform")
        dev_words = load_words(dev, study.L, device)
        dev_hash = raw_digest(dev, study.L)
        baseline = codec_at(k, study.C, study.hd, args.tile_tokens, Policy(), dev.tokens, dev.position, True)
        legacy_packet, _ = encode_packet(baseline, dev_words, predictor_hash, Policy(), dev_hash, dev.position)
        candidates = {"global": (Policy(), len(legacy_packet))}
        matched = codec_at(k, study.C, study.hd, args.tile_tokens, Policy(), dev.tokens, dev.position)
        matched_packet, _ = encode_packet(matched, dev_words, predictor_hash, Policy(), dev_hash, dev.position)
        if matched_packet != legacy_packet:
            raise AssertionError("matched global control changed the original T packet")
        for family in ("head", "learned"):
            for alpha in args.alphas:
                p = Policy(1, alpha) if family == "head" else Policy(2, alpha, 0, learned.weights, learned.bias)
                codec = codec_at(k, study.C, study.hd, args.tile_tokens, p, dev.tokens, dev.position)
                packet, _ = encode_packet(codec, dev_words, predictor_hash, p, dev_hash, dev.position)
                name = f"{family}_{alpha:g}"
                candidates[name] = p, len(packet)
                result["dev_rows"].append({"mode": mode, "variant": name, **p.description(),
                    "packet_bytes": len(packet), "baseline_packet_bytes": len(legacy_packet),
                    "extra_policy_bytes": p.array_bytes, "bit_exact": True})
        result["dev_rows"].append({"mode": mode, "variant": "global", **Policy().description(),
            "packet_bytes": len(legacy_packet), "baseline_packet_bytes": len(legacy_packet), "bit_exact": True})
        chosen = {}
        for family in ("head", "learned"):
            eligible = [(name, p, n) for name, (p, n) in candidates.items() if name == "global" or name.startswith(family + "_")]
            name, policy, size = min(eligible, key=lambda x: (x[2], x[1].array_bytes, x[1].alpha))
            chosen[family] = policy
            result["selected_on_dev"].setdefault(mode, {})[family] = {"variant": name,
                "policy_sha256": policy.digest(), **policy.description(), "dev_packet_bytes": size,
                "dev_bytes_saved": len(legacy_packet) - size, "candidate_variants": [x[0] for x in eligible]}
        del dev_words, legacy_packet, matched_packet, packet
        frozen = json.dumps(result["selected_on_dev"][mode], sort_keys=True)
        # Explicitly persist DEV choices before the first TEST word is loaded.
        write_json(output / "selection_frozen.json", result["selected_on_dev"])
        specs = [("legacy_global", Policy(), True), ("matched_global", Policy(), False),
            ("head_selected", chosen["head"], False), ("learned_selected", chosen["learned"], False)]
        for family in ("head", "learned"):
            p = chosen[family]
            shifted = Policy(p.kind, p.alpha, 1, p.weights, p.bias) if p.kind else Policy()
            specs.append((family + "_shifted_control", shifted, False))
        mode_state = {}
        for name, p, _ in specs:
            path = output / "states" / f"{mode}_{name}.pt"
            save_state(path, k, study.C, study.hd, args.tile_tokens, "uniform", p, predictor_hash)
            mode_state[name] = path
        for doc in test_docs:
            log(f"TEST: {mode} {doc.name}, {doc.tokens} tokens, {study.L} layers")
            words = load_words(doc, study.L, device)
            source_hash = raw_digest(doc, study.L)
            expected_packet_hash = None
            for name, policy, legacy in specs:
                log(f"  {name}: encoding, decoding and independent packet check")
                codec = codec_at(k, study.C, study.hd, args.tile_tokens, policy, doc.tokens, 0, legacy)
                segments, capacity = codec.layout(doc.tokens)
                arena = torch.zeros(segments * capacity, dtype=torch.int16, device=device)
                encoded, enc_time = measure(lambda: codec.encode(words, out=arena), device, args.warmup, args.repeats)
                stream, starts = encoded
                bufs, decoded_output = codec.buffers(doc.tokens), torch.empty_like(words)
                decoded, dec_time = measure(lambda: codec.decode(stream, starts, doc.tokens, out=decoded_output, bufs=bufs),
                    device, args.warmup, args.repeats)
                if not torch.equal(decoded, words):
                    raise AssertionError("TEST word comparison failed")
                packet, cost = pack(codec, stream, starts, doc.tokens, predictor_hash, policy, source_hash)
                packet_hash = hashlib.sha256(packet).hexdigest()
                if name == "legacy_global":
                    expected_packet_hash = packet_hash
                if name == "matched_global" and packet_hash != expected_packet_hash:
                    raise AssertionError("TEST matched global packet differs from legacy")
                packet_path = output / "packets" / f"{mode}_{name}_{doc.name}.bin"
                packet_path.write_bytes(packet)
                report_path = packet_path.with_suffix(".json")
                command = [sys.executable, str(Path(__file__).resolve()), "--decode-worker",
                    "--state", str(mode_state[name]), "--packet", str(packet_path), "--worker-report", str(report_path),
                    "--device", str(device)]
                if args.software_check:
                    command += ["--software-check"]
                environment = dict(os.environ)
                if environment.get("PYTHONPATH"):
                    environment["PYTHONPATH"] = os.pathsep.join(str(Path(v).resolve()) for v in environment["PYTHONPATH"].split(os.pathsep) if v)
                child = subprocess.run(command, cwd=output / "packets", env=environment, capture_output=True, text=True)
                if child.returncode:
                    raise RuntimeError("packet-only worker failed: " + child.stderr[-4000:])
                proof = json.loads(report_path.read_text())
                if not proof["bit_exact"] or proof["words_sha256"] != source_hash or proof["gpu_executed"] != result["gpu_executed"]:
                    raise AssertionError("independent packet-only proof does not match source/device")
                row = {"mode": mode, "variant": name, "case": doc.name, "values": words.numel(),
                    "raw_bytes": words.numel() * 2, **cost, "bits_per_value": len(packet) * 8 / words.numel(),
                    "encode": enc_time, "decode": dec_time, "bit_exact": True, "packet_only_proof": proof,
                    "packet_sha256": packet_hash, "source_words_sha256": source_hash,
                    "predictor_sha256": predictor_hash, "policy_sha256": policy.digest(),
                    "extra_policy_tensor_bytes": policy.array_bytes,
                    "device_shared_predictor_tensor_bytes": tensors_bytes(k.values()),
                    "preallocated_decode_scratch_bytes": tensors_bytes(flattened_tensors(bufs)),
                    "decoded_output_bytes": decoded_output.numel() * decoded_output.element_size(),
                    "encoder_stream_arena_bytes": arena.numel() * arena.element_size(),
                    "serialized_state_bytes": mode_state[name].stat().st_size}
                result["test_rows"].append(row)
                write_json(output / "checkpoint.json", result)
                if not args.keep_packets:
                    packet_path.unlink()  # only the transient packet created above; proofs/hashes remain
                del codec, arena, stream, starts, bufs, decoded_output, decoded, encoded, packet
            del words
        if json.dumps(result["selected_on_dev"][mode], sort_keys=True) != frozen:
            raise AssertionError("TEST altered frozen DEV selection")
    for mode in args.modes.split(","):
        rows = [r for r in result["test_rows"] if r["mode"] == mode]
        base = [r for r in rows if r["variant"] == "legacy_global"]
        base_bytes = sum(r["packet_bytes"] for r in base)
        base_state_bytes = base[0]["serialized_state_bytes"]
        def time_ms(row):
            return row["decode"]["cuda_median_ms"] if result["gpu_executed"] else row["decode"]["wall_median_ms"]
        base_decode = sum(time_ms(r) for r in base)
        table = {}
        for name in sorted({r["variant"] for r in rows}):
            group = [r for r in rows if r["variant"] == name]
            n, b = sum(r["values"] for r in group), sum(r["packet_bytes"] for r in group)
            extra_state = max(0, group[0]["serialized_state_bytes"] - base_state_bytes)
            average_gain = (base_bytes - b) / len(group)
            table[name] = {"values": n, "packet_bytes": b, "bits_per_value": 8 * b / n,
                "saving_from_bf16": 1 - b / (2 * n), "bytes_saved_vs_legacy": base_bytes - b,
                "decode_ratio_vs_legacy": sum(time_ms(r) for r in group) / base_decode,
                "extra_policy_tensor_bytes": group[0]["extra_policy_tensor_bytes"],
                "extra_serialized_state_bytes": extra_state,
                "bytes_saved_after_one_extra_shared_state": base_bytes - b - extra_state,
                "bits_per_value_including_one_extra_shared_state": 8 * (b + extra_state) / n,
                "transfers_to_amortize_extra_state_at_mean_gain":
                    max(1, math.ceil(extra_state / average_gain)) if average_gain > 0 else None,
                "all_exact": True}
        result["aggregate"][mode] = table
        decisions = {}
        for family in ("head", "learned"):
            row = table[family + "_selected"]
            selection = result["selected_on_dev"][mode][family]
            decisions[family] = {"positive_dev_gain": selection["dev_bytes_saved"] > 0,
                "positive_test_packet_gain": row["bytes_saved_vs_legacy"] > 0,
                "within_decode_budget": row["decode_ratio_vs_legacy"] <= args.max_decode_ratio,
                "saves_bytes_after_one_extra_shared_state": row["bytes_saved_after_one_extra_shared_state"] > 0,
                "candidate_for_review": bool(result["gpu_executed"] and result["full_model"]
                    and selection["dev_bytes_saved"] > 0 and row["bytes_saved_vs_legacy"] > 0
                    and row["decode_ratio_vs_legacy"] <= args.max_decode_ratio),
                "automatic_merge": False}
        result["decisions"][mode] = decisions
    for record in inputs:
        p = Path(record["path"])
        if "mtime_ns" in record and (p.stat().st_size != record["bytes"] or p.stat().st_mtime_ns != record["mtime_ns"]):
            raise ValueError("input capture changed during the experiment")
    result["status"] = "completed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--software-check", action="store_true")
    parser.add_argument("--train-docs", default=",".join(f"wtrain{i}" for i in range(6)))
    parser.add_argument("--tables-doc", default="wtables")
    parser.add_argument("--test-docs", default=",".join(f"ds_wikitext2_r{i}" for i in range(4)))
    parser.add_argument("--modes", default="dense,hbh")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--dev-tokens", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--tile-tokens", type=int, default=128)
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-decode-ratio", type=float, default=1.05)
    parser.add_argument("--keep-packets", action="store_true")
    parser.add_argument("--decode-worker", action="store_true")
    parser.add_argument("--state")
    parser.add_argument("--packet")
    parser.add_argument("--worker-report")
    args = parser.parse_args(argv)
    if args.decode_worker:
        from kv_compare.head_spread_wire import decode_packet
        device = select_device(args.device, args.software_check)
        import torch
        torch.set_num_threads(args.threads)
        proof = decode_packet(Path(args.state), Path(args.packet), device)
        write_json(args.worker_report, proof)
        return 0
    if not args.output or not args.captures:
        parser.error("--captures and --output are required")
    args.alphas = [float(v) for v in args.alphas.split(",")]
    if (not args.alphas or any(not 0 < v <= 1 for v in args.alphas)
            or not set(args.modes.split(",")) <= {"dense", "hbh"} or not args.modes
            or len(set(args.modes.split(","))) != len(args.modes.split(","))
            or args.tokens < 32 or min(args.tile_tokens, args.repeats, args.threads) < 1
            or args.tokens % args.tile_tokens or args.tile_tokens & (args.tile_tokens - 1)
            or args.warmup < 0 or not math.isfinite(args.ridge) or args.ridge <= 0
            or not math.isfinite(args.max_decode_ratio) or args.max_decode_ratio <= 0):
        parser.error("invalid experiment geometry, candidates, repetition or budget")
    output = Path(args.output).resolve()
    captures = Path(args.captures).resolve()
    if output == captures or output.is_relative_to(captures) or captures.is_relative_to(output):
        parser.error("output must be separate from input captures")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("states", "packets", "share"):
        (output / name).mkdir(exist_ok=True)
    def log(message):
        text = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(text, flush=True)
        with (output / "run.log").open("a") as stream:
            stream.write(text + "\n")
    try:
        result = execute(args, output, log)
        code = 0
    except Exception as error:
        result = {"format": "t_head_spread_experiment_v1", "status": "failed", "gpu_executed": False,
            "arguments": vars(args), "error": type(error).__name__ + ": " + str(error), "traceback": traceback.format_exc()}
        log(result["error"])
        code = 2
    sources = [ROOT / "kv_compare" / name for name in ("head_spread.py", "head_spread_kernels.py", "head_spread_wire.py")]
    sources.append(Path(__file__).resolve())
    result["experiment_sources"] = [{"path": str(p.relative_to(ROOT)), "sha256": file_sha(p)} for p in sources]
    write_json(output / "results.json", result)
    (output / "summary.md").write_text(summarize(result))
    for name in ("results.json", "summary.md", "selection_frozen.json", "device_checks.json", "run.log"):
        p = output / name
        if p.is_file():
            (output / "share" / name).write_bytes(p.read_bytes())
    with zipfile.ZipFile(output / "share.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((output / "share").iterdir()):
            z.write(p, p.name)
    log(f"{result['status']}: {output / 'summary.md'}; share {output / 'share.zip'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
