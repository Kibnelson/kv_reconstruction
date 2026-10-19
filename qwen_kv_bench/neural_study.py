"""Fifth study: learned BF16 probabilities with a real CPU arithmetic stream.

Neural inference uses the selected device. Entropy coding is deliberately a
CPU correctness reference, and is timed separately. This consumes produced KV;
it does not use the hidden-state hooks of the producer studies. It measures
neither compact-resident attention nor a serving/concurrency speedup.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from . import gpu_ops as g, model_runtime as runtime, producer_codec as anchor_codec
from .fixtures import TRAIN, DEV, TEST, build_input_ids
from .reporting import write_json


def _note(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _matrices_hash(matrices):
    checksum = hashlib.sha256()
    for layer, pair in enumerate(matrices):
        for kind, words in zip(("K", "V"), pair):
            words = np.asarray(words)
            checksum.update(json.dumps([layer, kind, str(words.dtype), list(words.shape)]).encode())
            checksum.update(np.ascontiguousarray(words).tobytes())
    return checksum.hexdigest()


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default=runtime.DEFAULT_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--modes", nargs="+", choices=("per_layer", "shared"), default=["per_layer", "shared"])
    p.add_argument("--width", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--shared-steps", type=int, default=None,
                   help="Shared network updates; default steps*(layers-1) matches the per-layer aggregate update budget")
    p.add_argument("--batch-tokens", type=int, default=256)
    p.add_argument("--train-tokens", type=int)
    p.add_argument("--dev-tokens", type=int)
    p.add_argument("--lengths", nargs="+", type=int)
    p.add_argument("--cases", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--tile-tokens", type=int, default=64)
    p.add_argument("--max-pairs", type=int, default=72,
                   help="Sampled layer/tile pairs, at least the full layer count")
    p.add_argument("--generation-steps", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--output", type=Path)
    return p


def _defaults(args):
    values = dict(width=16 if args.tiny else 512, steps=2 if args.tiny else 2000,
                  train_tokens=32 if args.tiny else 2048, dev_tokens=8 if args.tiny else 64,
                  lengths=[16, 24] if args.tiny else [2048, 8192], chunk_tokens=8 if args.tiny else 512)
    for key, default in values.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.tiny and args.device == "auto":
        args.device = "cpu"
    if args.revision is None and args.model == runtime.DEFAULT_MODEL:
        args.revision = runtime.DEFAULT_REVISION


def _capture(model, tokenizer, fixture, tokens, device, args):
    if args.tiny:
        namespace = {"train": 11, "dev": 47, "test": 71}[fixture["split"]]
        topic = sum(fixture["name"].encode())
        ids = [1 + ((i * (namespace + 2) + topic + namespace) % 95) for i in range(tokens)]
    else:
        ids = build_input_ids(tokenizer, fixture, tokens)
    result = runtime.capture(model, ids, device, args.chunk_tokens)
    result["input_ids_sha256"] = _hash(ids)
    result["kv_words_sha256"] = _matrices_hash(result["matrices"])
    return result


def _rope(model):
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None or not hasattr(rotary, "inv_freq"):
        raise ValueError("Qwen rotary embedding with explicit inv_freq is required")
    rope_type = getattr(rotary, "rope_type", "default")
    if rope_type not in ("default", "linear"):
        raise ValueError("This reference requires fixed RoPE frequencies; dynamic scaling is unsupported")
    inv = rotary.inv_freq.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    scale = float(getattr(rotary, "attention_scaling", 1.0))
    return inv, scale


def fit_anchor(captures, metadata, identity, device):
    """Fit frozen exponent medians from real TRAIN KV only, without fake hidden inputs."""
    if (not captures or len(metadata) != len(captures)
            or any(row.get("split") not in ("train", "training") for row in metadata)):
        raise ValueError("Frozen anchor fitting requires explicitly labelled TRAIN captures")
    params = {}
    for layer in range(len(captures[0])):
        params[f"layer_{layer:02d}"] = {}
        for kind, index in (("K", 0), ("V", 1)):
            words = np.concatenate([capture[layer][index] for capture in captures])
            exponents = ((words.astype(np.uint32) >> 7) & 255).astype(np.uint8)
            params[f"layer_{layer:02d}"]["base_" + kind] = np.floor(np.median(exponents, axis=0)).astype(np.uint8)
    provenance = dict(format="neural_reference_frozen_exponent_v1", training_only=True,
                      training_metadata=metadata, model_identity=identity,
                      training_kv_sha256=[_matrices_hash(capture) for capture in captures],
                      rule="floor of per-channel TRAIN exponent median; produced KV only")
    arrays = {f"{layer}/{name}": array for layer, values in params.items() for name, array in values.items()}
    model_id = hashlib.sha256(anchor_codec._array_artifact(provenance, arrays)).hexdigest()
    model = SimpleNamespace(kind="frozen", params=params, config=dict(feature_transform="none"),
                            metadata=provenance, model_id=model_id, path=None)
    spec = next(x for x in anchor_codec.recipes() if x["kind"] == "frozen")
    return anchor_codec.make_state(spec, {"frozen": model}, {}, device)


def _starts(tokens, tile, layers, maximum):
    possible = list(range(0, tokens, tile))
    count = min(len(possible), maximum // layers)
    if count < 1:
        raise ValueError("--max-pairs must cover every layer")
    return [possible[i] for i in np.linspace(0, len(possible) - 1, count, dtype=np.int64)]


def _baseline(matrices, state, starts, tile_tokens, head_dim):
    """Actual frozen-exponent packet fields plus an explicit indexed envelope."""
    from . import neural_chain as chain
    raw = 0
    pages = []
    encode_seconds = decode_seconds = 0.0
    for start in starts:
        for layer, (key, value) in enumerate(matrices):
            record = dict(layer=layer, start=start, tokens=len(key[start:start + tile_tokens]), head_dim=head_dim,
                          k=key[start:start + tile_tokens], v=value[start:start + tile_tokens])
            t0 = time.perf_counter()
            prepared = anchor_codec.prepare_record(record, state.device, state.spec)
            packet = anchor_codec.encode(prepared, state.spec, state)
            payload, _ = chain._anchor_wire(packet)
            metadata = dict(version=1, kind="frozen_control", layer=layer, start=int(start),
                tokens=record["tokens"], channels=key.shape[1], anchor_state_id=state.state_id,
                words_sha256=_matrices_hash([(record["k"], record["v"])]))
            page = chain._page(metadata, payload)
            pages.append(page)
            runtime.synchronize(state.device)
            encode_seconds += time.perf_counter() - t0
            t0 = time.perf_counter()
            # Restore the actual authoritative header/fields, not a launch descriptor.
            descriptor, serialized_payload, _ = chain._read_page(bytes(page))
            if descriptor["anchor_state_id"] != state.state_id:
                raise ValueError("Frozen control state identity differs")
            restored = chain._anchor_packet(serialized_payload, state.device)
            pair = anchor_codec.decode(restored, state)
            for actual, expected in zip(pair, (record["k"], record["v"])):
                if not np.array_equal(actual.cpu().numpy(), expected):
                    raise AssertionError("Frozen baseline reconstruction differs")
            runtime.synchronize(state.device)
            decode_seconds += time.perf_counter() - t0
            raw += record["k"].nbytes + record["v"].nbytes
    container = chain.NeuralCache(tuple(pages), len(matrices), len(matrices[0][0]), matrices[0][0].shape[1], tile_tokens)
    wire = container.to_bytes()
    if chain.NeuralCache.from_bytes(wire) != container:
        raise AssertionError("Frozen control indexed container did not reopen identically")
    inner = sum(map(len, pages))
    outer = len(wire) - inner
    return dict(raw_sample_bytes=raw, packet_bytes=len(wire), inner_packet_bytes=inner,
                indexed_envelope_bytes=outer, records=len(pages),
                encode_wall_seconds=encode_seconds, decode_wall_seconds=decode_seconds,
                container_roundtrip_verified=True,
                scope="Frozen-exponent control, actual reopened CRC-framed packets and indexed container; no neural prerequisite/prediction hashes. Synchronized staging/round-trip wall time, one pass.")


def _artifact_info(model, output):
    files = []
    for path in getattr(model, "artifact_paths", [model.path]):
        path = Path(path)
        files.append(dict(file=str(path.relative_to(output)), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if sum(f["bytes"] for f in files) != model.state_bytes:
        raise AssertionError("Neural serialized state differs from complete artifact file sizes")
    return dict(model_id=model.model_id, state_bytes=model.state_bytes,
                parameter_count=model.parameter_count, files=files, metadata=model.metadata,
                share_package_scope="Full neural artifacts remain in run/models. The normal reports package omits models/; all their bytes are nevertheless charged.")


def _generation_passed(check):
    return (all(check.get(key, False) for key in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal"))
            and bool(check.get("per_step"))
            and all(step.get("logits_finite", False) for step in check["per_step"]))


def _save(output, report):
    write_json(output / "results.json", report)
    fields = ["case", "tokens", "method", "status", "sample_payload_reduction_pct", "net_single_cache_estimated_reduction_pct", "charged_state_bytes"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            for row in case["results"]:
                accounting = row.get("accounting", {})
                writer.writerow(dict(case=case["name"], tokens=case["tokens"], method=row["name"], status=row["status"],
                    sample_payload_reduction_pct=accounting.get("sample", {}).get("payload_reduction_pct"),
                    net_single_cache_estimated_reduction_pct=accounting.get("full_cache_estimate", {}).get("net_reduction_pct"),
                    charged_state_bytes=accounting.get("state", {}).get("total_charged_bytes")))
    lines = ["# Learned probability reference study", "", f"Status: **{report['status']}**", "", report["scope"], "",
             "| Case | Tokens | Method | Sample stream reduction | Net single-cache estimate | State MiB |", "|---|---:|---|---:|---:|---:|"]
    for case in report["cases"]:
        for row in case["results"]:
            accounting = row.get("accounting", {})
            def fmt(value):
                return "—" if value is None else f"{value:.2f}%"
            sample = accounting.get("sample", {}).get("payload_reduction_pct")
            net = accounting.get("full_cache_estimate", {}).get("net_reduction_pct")
            state = accounting.get("state", {}).get("total_charged_bytes")
            lines.append(f"| {case['name']} | {case['tokens']} | {row['name']} ({row['status']}) | {fmt(sample)} | {fmt(net)} | {'—' if state is None else f'{state / 2**20:.3f}'} |")
    lines += ["", "Payload rates use actual sampled packet bytes. Full-context and concurrent shared-state figures are estimates from the same sampled addresses, with full state charged. These are not measured concurrent serving results.", "", "Neural inference runs on the selected device; arithmetic encode/decode runs on the CPU. Dependency decoding is sequential across layers for each token tile. Full original KV is still used for model capture; encoded-resident attention is not implemented.", "", "All development generation checks:", ""]
    for check in report["generation_checks"]:
        lines.append(f"- {check['name']}: {check['status']}")
    if report.get("error"):
        lines += ["", "Error: " + report["error"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    from . import neural_probability as neural, neural_chain as chain
    from .neural_accounting import account
    from .runner import _source_manifest
    p = _parser()
    args = p.parse_args(argv)
    _defaults(args)
    positive = [args.width, args.steps, args.batch_tokens, args.train_tokens, args.dev_tokens,
                args.chunk_tokens, args.tile_tokens, args.max_pairs, args.threads, *args.lengths]
    if (any(value < 1 for value in positive) or not 1 <= args.generation_steps <= 64
            or (args.shared_steps is not None and args.shared_steps < 1)):
        p.error("Token counts, steps, width, threads, pair limits and generation steps must be positive")
    if len(args.modes) != len(set(args.modes)):
        p.error("--modes must not contain duplicates")
    output = args.output or Path("runs") / ("neural_probability_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    report = dict(format="qwen_neural_probability_study_v1", status="running",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        source_manifest=_source_manifest(), model_identity=None, models={}, cases=[], generation_checks=[],
        scope="Exact BF16 previous-layer probability coding with real indexed packets, GPU/CPU neural prediction, and separately timed CPU arithmetic coding. TRAIN-only fits; held-out DEV continuation and sampled TEST bytes. Complete neural/anchor state is charged. This experiment uses produced KV rather than prefill hidden metadata. No pretrained-model quality, compact-resident attention, partial-resume scheduler, high-concurrency, latency or novelty claim is established.")
    write_json(output / "source_manifest.json", report["source_manifest"])
    _save(output, report)
    began = time.perf_counter()
    try:
        device = runtime.select_device(args.device)
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        report["environment"] = runtime.doctor(device, probe=False)
        _note("Loading model for learned probability study on " + str(device))
        if args.tiny:
            from .producer_study import _tiny
            model, tokenizer, identity = _tiny(device, args)
        else:
            model, tokenizer, identity = runtime.load_model(args.model, args.revision, device, args.local_files_only)
        report["model_identity"] = identity
        head_dim = int(model.config.head_dim)
        inv, rope_scale = _rope(model)
        rope_signature = (inv.tobytes(), rope_scale)
        train_captures, training_metadata = [], []
        for fixture in TRAIN:
            _note("TRAIN capture: " + fixture["name"])
            capture = _capture(model, tokenizer, fixture, args.train_tokens, device, args)
            train_captures.append(capture["matrices"])
            training_metadata.append(dict(name=fixture["name"], split="train", tokens=args.train_tokens,
                input_ids_sha256=capture["input_ids_sha256"], kv_words_sha256=capture["kv_words_sha256"],
                geometry=capture["geometry"], timings=capture["timings"]))
            del capture
        layers = len(train_captures[0])
        channels = train_captures[0][0][0].shape[1]
        if args.max_pairs < layers:
            raise ValueError(f"--max-pairs {args.max_pairs} must cover all {layers} layers")
        report["training_captures"] = training_metadata
        train_hashes = [_matrices_hash(capture) for capture in train_captures]
        anchor = fit_anchor(train_captures, training_metadata, identity, device)
        anchor.dump(output / "calibration" / "neural_anchor_state.bin")
        report["anchor_state"] = dict(anchor_codec.state_info(anchor), file="calibration/neural_anchor_state.bin")
        _note("Fitting TRAIN-only neural probability models: " + ", ".join(args.modes))
        t0 = time.perf_counter()
        models = {}
        report["training_budget"] = dict(per_layer_updates_each=args.steps,
            per_layer_aggregate_updates=args.steps * (layers - 1),
            shared_aggregate_updates=args.shared_steps if args.shared_steps is not None else args.steps * (layers - 1),
            rule="By default the shared network receives the same total number of layer-labelled minibatch updates as all per-layer networks combined. Shared layer allocation is random; realized counts are logged. --shared-steps explicitly changes this budget.")
        for mode in args.modes:
            updates = report["training_budget"]["shared_aggregate_updates"] if mode == "shared" else args.steps
            _note(f"Neural fit {mode}: {updates} updates per network")
            models.update(neural.fit(train_captures, output / "models", model_identity=identity,
                training_metadata=training_metadata, head_dim=head_dim, rope_inv_freq=inv,
                rope_scaling=rope_scale, device=device, width=args.width, steps=updates,
                batch_tokens=args.batch_tokens, seed=args.seed, modes=(mode,)))
        report["training_wall_seconds"] = time.perf_counter() - t0
        if train_hashes != [_matrices_hash(capture) for capture in train_captures]:
            raise AssertionError("TRAIN inputs changed during fitting")
        for name, fitted in models.items():
            # Validate saved CRC/SHA-bound shards and run inference from reload,
            # so the real decode path depends on persisted state, not a fit object.
            fitted = neural.load(fitted.path, expected_model_identity=identity,
                                expected_model_id=fitted.model_id, expected_mode=name)
            models[name] = fitted
            report["models"][name] = _artifact_info(fitted, output)
        freeze = dict(model_identity=identity, source_sha256=report["source_manifest"]["sources_sha256"],
            training_kv_sha256=train_hashes, models=report["models"], anchor_state=report["anchor_state"],
            training_budget=report["training_budget"],
            protocol="All fits/hyperparameters frozen before DEV/TEST. Both requested modes are reported; TEST never selects a mode.")
        write_json(output / "frozen_manifest.json", freeze)
        report["freeze"] = freeze
        del train_captures
        gc.collect()
        _save(output, report)

        _note("DEV capture and packet-only continuation checks")
        development = _capture(model, tokenizer, DEV[0], args.dev_tokens, device, args)
        current_inv, current_scale = _rope(model)
        if (current_inv.tobytes(), current_scale) != rope_signature:
            raise ValueError("Model RoPE frequencies changed before DEV")
        for name, fitted in models.items():
            _note("DEV full chain: " + name)
            prepared = fitted.prepare(device)
            packets, measured = chain.encode_cache(development["matrices"], prepared, anchor,
                tile_tokens=args.tile_tokens, max_pairs=layers * ((args.dev_tokens + args.tile_tokens - 1) // args.tile_tokens), head_dim=head_dim)
            # Only serialized packets, frozen state, and explicit dimensions enter restore.
            wire = bytes(packets.to_bytes())
            del packets
            def restore(wire=wire, prepared=prepared):
                container = chain.NeuralCache.from_bytes(wire)
                return chain.decode_cache(container, prepared, anchor, layers=layers,
                    tokens=args.dev_tokens, channels=channels, require_full=True)
            restored = restore()
            if _matrices_hash(restored) != development["kv_words_sha256"]:
                raise AssertionError("Packet-only complete DEV cache differs")
            del restored
            check = runtime.generation_check(model, development["matrices"], restore, device,
                first_token=development["first_token"], steps=args.generation_steps)
            if not _generation_passed(check):
                raise AssertionError("DEV generation check did not pass")
            report["generation_checks"].append(dict(name=name, status="ok", result=check,
                complete_prefix_tokens=args.dev_tokens, full_cache_word_hash=development["kv_words_sha256"],
                actual_container_bytes=len(wire), chain_measurement=measured,
                decode_inputs="Serialized container, frozen neural/anchor state, geometry; no original KV in closure"))
            del restore, prepared, wire
            gc.collect()
            _save(output, report)
        del development
        for length in args.lengths:
            for fixture in TEST[:args.cases]:
                _note(f"TEST capture: {fixture['name']}, {length} tokens")
                captured = _capture(model, tokenizer, fixture, length, device, args)
                current_inv, current_scale = _rope(model)
                if (current_inv.tobytes(), current_scale) != rope_signature:
                    raise ValueError("Model RoPE frequencies changed after frozen TRAIN calibration")
                matrices = captured.pop("matrices")
                case = dict(name=fixture["name"], split="test", tokens=length, capture=captured, results=[])
                report["cases"].append(case)
                starts = _starts(length, args.tile_tokens, layers, args.max_pairs)
                baseline = _baseline(matrices, anchor, starts, args.tile_tokens, head_dim)
                baseline_account = account(baseline["raw_sample_bytes"], baseline["packet_bytes"], captured["raw_payload_bytes"],
                    0, 0, anchor.serialized_bytes, anchor.physical_nbytes)
                case["results"].append(dict(name="frozen_exponent_control", status="ok", measurement=baseline, accounting=baseline_account))
                for name, fitted in models.items():
                    _note("TEST real arithmetic stream: " + name)
                    initial_memory = runtime.reset_peak_memory(device)
                    prepared, preparation = runtime.timed_call(lambda: fitted.prepare(device), device)
                    t0 = time.perf_counter()
                    packets, measured = chain.encode_cache(matrices, prepared, anchor,
                        tile_tokens=args.tile_tokens, max_pairs=args.max_pairs, head_dim=head_dim)
                    wire = packets.to_bytes()
                    del packets
                    checked, decode_metrics = chain.decode_cache(chain.NeuralCache.from_bytes(wire), prepared, anchor,
                        layers=layers, tokens=length, channels=channels, require_full=False, return_metrics=True)
                    # The chain decoder verifies authoritative target hashes and dependency hashes.
                    wall = time.perf_counter() - t0
                    raw_sample = sum(4 * min(args.tile_tokens, length - start) * channels * layers for start in starts)
                    measured.update(actual_container_bytes=len(wire), raw_sample_bytes=raw_sample,
                        selected_tile_starts=starts, independent_packet_decode=decode_metrics,
                        encode_verify_and_independent_decode_wall_seconds=wall,
                        state_preparation=preparation, initial_device_memory=initial_memory,
                        after_device_memory=runtime.memory_snapshot(device),
                        timing_scope="One synchronized functional pass. Encode includes immediate correctness restore; independent decode is additionally measured. Neural inference device and CPU arithmetic stages are separate.")
                    accounting = account(raw_sample, len(wire), captured["raw_payload_bytes"], fitted.state_bytes,
                        prepared.physical_nbytes, anchor.serialized_bytes, anchor.physical_nbytes)
                    row = dict(name="neural_" + name, status="ok", measurement=measured, accounting=accounting,
                        parameter_count=fitted.parameter_count,
                        macs_per_token_by_layer={str(layer): fitted.macs_per_token(layer) for layer in range(1, layers)},
                        execution_contract=prepared.execution_contract,
                        matched_control_sample_bytes=baseline["packet_bytes"],
                        actual_stream_byte_delta_vs_control=len(wire) - baseline["packet_bytes"])
                    case["results"].append(row)
                    del checked, prepared, wire
                    gc.collect()
                    _save(output, report)
                del matrices
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _note(report["error"])
    report["elapsed_wall_seconds"] = time.perf_counter() - began
    _save(output, report)
    _note("Report: " + str(output / "summary.md"))
    _note("Status: " + report["status"])
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
