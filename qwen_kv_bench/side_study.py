"""Page-independent exact coding conditioned on a learned prefill side code.

Four matched shared models learn probability distributions: no retained code,
a BF16 code of same-page produced KV, a code of inverse-RoPE-aligned produced
KV, and a code of actual normalized hidden states observed during prefill. Real CPU arithmetic streams,
all side bits, framing and complete state are charged. This is a representation
experiment, not an encoded-resident attention or concurrent serving engine.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import struct
import time
import traceback
import zlib

import numpy as np
import torch

from . import model_runtime as runtime, neural_study as previous, producer_codec as anchor_codec
from .fixtures import TRAIN, DEV, TEST, build_input_ids
from .reporting import write_json

_HEADER = struct.Struct("<4sIII")
_MAGIC = b"SPGC"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _note(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default=runtime.DEFAULT_MODEL)
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--rank", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-tokens", type=int, default=64)
    p.add_argument("--train-tokens", type=int)
    p.add_argument("--dev-tokens", type=int)
    p.add_argument("--lengths", nargs="+", type=int)
    p.add_argument("--cases", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--tile-tokens", type=int)
    p.add_argument("--max-pairs", type=int, default=72)
    p.add_argument("--generation-steps", type=int, default=2)
    p.add_argument("--shared-width", type=int,
                   help="Previous-layer shared control width; default 512 (16 with --tiny)")
    p.add_argument("--shared-steps", type=int,
                   help="Previous-layer shared updates; defaults to the same --steps as each local model")
    p.add_argument("--skip-chain-control", action="store_true",
                   help="Skip the previous-layer neural control, retaining exponent and all four matched local models")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--output", type=Path)
    return p


def _defaults(args):
    for key, value in dict(rank=2 if args.tiny else 8, width=16 if args.tiny else 128,
        steps=2 if args.tiny else 200, train_tokens=24 if args.tiny else 512,
        dev_tokens=8 if args.tiny else 128, lengths=[16, 24] if args.tiny else [2048],
        chunk_tokens=8 if args.tiny else 512, tile_tokens=8 if args.tiny else 64,
        shared_width=16 if args.tiny else 512).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.shared_steps is None:
        args.shared_steps = args.steps
    if args.tiny and args.device == "auto":
        args.device = "cpu"
    if args.revision is None and args.model == runtime.DEFAULT_MODEL:
        args.revision = runtime.DEFAULT_REVISION


def _records_hash(records):
    checksum = hashlib.sha256()
    for record in sorted(records, key=lambda row: (row["layer"], row["start"])):
        checksum.update(_json({key: int(record[key]) for key in ("layer", "start", "tokens", "head_dim")}))
        for name in ("hidden", "k", "v", "k_pre_rope"):
            array = np.asarray(record[name])
            checksum.update(_json([name, str(array.dtype), list(array.shape)]))
            checksum.update(np.ascontiguousarray(array).tobytes())
    return checksum.hexdigest()


def _capture(model, tokenizer, fixture, tokens, device, args, starts=None, inline=None):
    from .producer_hooks import ProducerCapture
    if args.tiny:
        namespace = {"train": 11, "dev": 47, "test": 71}[fixture["split"]]
        topic = sum(fixture["name"].encode())
        ids = [1 + ((i * (namespace + 2) + topic + namespace) % 95) for i in range(tokens)]
    else:
        ids = build_input_ids(tokenizer, fixture, tokens)
    maximum = None if starts is None else int(model.config.num_hidden_layers) * len(starts)
    with ProducerCapture(model, tokens, tile_tokens=args.tile_tokens, max_tiles=maximum,
                         selected_starts=starts, on_tile=inline) as hooks:
        captured = runtime.capture(model, ids, device, args.chunk_tokens)
    records, audit, context = hooks.finalize(captured["matrices"])
    if not audit.get("all_stage_targets_exact"):
        raise AssertionError("Actual producer hook targets differ from exported KV")
    for record in records:
        record["split"] = fixture["split"]
    captured["geometry"]["channels"] = captured["geometry"]["columns"]
    captured.update(input_ids_sha256=previous._hash(ids),
        kv_words_sha256=previous._matrices_hash(captured["matrices"]),
        producer_records_sha256=_records_hash(records), producer_audit=audit)
    captured["timings"]["prefill"].update(producer_hooks_included=True,
        scope="Instrumented forward includes actual producer hooks, diagnostic CPU copies and inline learned-side-code emission when supplied. Entropy coding is replayed afterward. No uninstrumented prefill timing comparison.")
    if inline is not None:
        captured["inline_side_codes"] = inline.report()
    return captured, records, context


class InlineCodes:
    """Emit retained codes synchronously while actual attention returns a tile.

    The diagnostic hook already exports original features to CPU. The encoder
    stages these on the selected device and returns only BF16 code words. This
    establishes prefill-time code provenance, not a fused production kernel.
    """
    def __init__(self, models, device):
        self.prepared = {}; self.codes = {}; self.rows = {}
        for name, fitted in models.items():
            prepared, timing = runtime.timed_call(lambda: fitted.prepare(device), device)
            self.prepared[name] = prepared; self.codes[name] = {}
            self.rows[name] = dict(calls=0, feature_bytes=0, wall_seconds=0., preparation=timing, records=[])

    def __call__(self, record, context):
        for name, prepared in self.prepared.items():
            address = (int(record["layer"]), int(record["start"]))
            if address in self.codes[name]:
                raise AssertionError("Inline learned side code emitted more than once")
            words, elapsed = runtime.timed_call(lambda: prepared.encode_features(record), prepared.device)
            words = np.asarray(words)
            if words.dtype != np.uint16 or words.ndim != 2 or len(words) != record["tokens"]:
                raise AssertionError("Inline learned encoder returned invalid code bits")
            words = words.copy(); words.flags.writeable = False
            self.codes[name][address] = words
            row = self.rows[name]; row["calls"] += 1; row["feature_bytes"] += words.nbytes
            row["wall_seconds"] += elapsed["wall_seconds"]
            row["records"].append(dict(layer=address[0], start=address[1], tokens=record["tokens"],
                feature_bytes=words.nbytes, sha256=hashlib.sha256(words.tobytes()).hexdigest()))

    def report(self):
        return dict(methods=self.rows, observed_inside_attention_forward=True,
            scope="Synchronous producer on_tile callback, before attention returns; CPU diagnostic feature copies and selected-device code emission included in instrumented prefill time. No baseline overhead or fused-kernel claim.")

    def release_prepared(self):
        self.prepared.clear(); gc.collect()


def pack_pages(pages, *, layers, tokens, channels, tile_tokens):
    """Persist every byte of an indexed container; pages remain independent."""
    pages = tuple(bytes(page) for page in pages)
    if not pages or any(not page for page in pages):
        raise ValueError("An indexed side cache needs nonempty pages")
    geometry = dict(layers=layers, tokens=tokens, channels=channels, tile_tokens=tile_tokens)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in geometry.values()):
        raise ValueError("Invalid side-cache geometry")
    metadata = _json(dict(format="prefill_side_cache_v1", **geometry))
    offsets = [0]
    for page in pages:
        offsets.append(offsets[-1] + len(page))
    index = struct.pack("<" + "Q" * len(offsets), *offsets)
    body = metadata + index + b"".join(pages)
    return _HEADER.pack(_MAGIC, len(metadata), len(pages), zlib.crc32(body)) + body


def unpack_pages(wire):
    wire = bytes(wire)
    if len(wire) < _HEADER.size:
        raise ValueError("Truncated side-cache container")
    magic, metadata_size, count, crc = _HEADER.unpack_from(wire)
    first = _HEADER.size + metadata_size + 8 * (count + 1)
    if magic != _MAGIC or not count or metadata_size < 2 or first > len(wire):
        raise ValueError("Invalid side-cache header/index")
    body = wire[_HEADER.size:]
    if zlib.crc32(body) != crc:
        raise ValueError("Side-cache container CRC mismatch")
    try:
        metadata = json.loads(body[:metadata_size])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid side-cache metadata") from error
    if (not isinstance(metadata, dict) or _json(metadata) != body[:metadata_size]
            or set(metadata) != {"format", "layers", "tokens", "channels", "tile_tokens"}
            or metadata["format"] != "prefill_side_cache_v1"
            or any(isinstance(metadata[k], bool) or not isinstance(metadata[k], int) or metadata[k] < 1
                   for k in ("layers", "tokens", "channels", "tile_tokens"))):
        raise ValueError("Invalid side-cache geometry/format")
    offsets = struct.unpack_from("<" + "Q" * (count + 1), wire, _HEADER.size + metadata_size)
    if offsets[0] != 0 or offsets[-1] != len(wire) - first or any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Invalid side-cache page offsets")
    pages = tuple(wire[first + a:first + b] for a, b in zip(offsets, offsets[1:]))
    return metadata, pages


def restore_pages(wire, prepared, *, require_full=False, seed=71):
    """Restore from bytes and immutable prepared state only, in shuffled order.

    Source hidden/KV, other pages' decoded tensors, and encoder features are
    never passed to the page decoder. Output allocation is reported separately
    from prerequisite state (zero raw prerequisite pages).
    """
    from . import side_codec as codec
    geometry, pages = unpack_pages(wire)
    layers, tokens, channels, tile = (geometry[k] for k in ("layers", "tokens", "channels", "tile_tokens"))
    addresses = []
    seen = set()
    for page in pages:
        meta = codec.page_metadata(page)
        layer, start, count = (meta[k] for k in ("layer", "start", "tokens"))
        if (any(isinstance(n, bool) or not isinstance(n, int) for n in (layer, start, count))
                or not 0 <= layer < layers or not 0 <= start < tokens or start % tile
                or count != min(tile, tokens - start) or meta.get("channels") != channels
                or (layer, start) in seen):
            raise ValueError("Side page address differs from declared cache geometry")
        seen.add((layer, start)); addresses.append((layer, start, count))
    if require_full:
        expected = {(layer, start) for layer in range(layers) for start in range(0, tokens, tile)}
        if seen != expected:
            raise ValueError("Full side-cache restore requires every layer/token tile")
    order = np.random.default_rng(seed).permutation(len(pages)).tolist()
    if len(order) > 1 and order == list(range(len(pages))):
        order.reverse()
    result = [(np.empty((tokens, channels), np.uint16), np.empty((tokens, channels), np.uint16))
              for _ in range(layers)] if require_full else []
    timing = Counter(); diagnostics = []
    began = time.perf_counter()
    for ordinal in order:
        layer, start, count = addresses[ordinal]
        restored, stages = codec.decode_page(pages[ordinal], prepared, return_stats=True)
        if restored.dtype != np.uint16 or restored.shape != (count, 2 * channels):
            raise AssertionError("Side page decoder returned unexpected output geometry")
        if require_full:
            result[layer][0][start:start + count] = restored[:, :channels]
            result[layer][1][start:start + count] = restored[:, channels:]
        else:
            result.append(dict(layer=layer, start=start, tokens=count,
                               k=restored[:, :channels].copy(), v=restored[:, channels:].copy()))
        for name in ("prediction_seconds", "entropy_decode_seconds", "decode_seconds"):
            timing[name] += stages[name]
        diagnostics.append(dict(layer=layer, start=start, tokens=count, **stages))
    wall = time.perf_counter() - began
    returned = 4 * layers * tokens * channels if require_full else sum(r["k"].nbytes + r["v"].nbytes for r in result)
    return result, dict(records_decoded=len(pages), full_prefix=require_full, wall_seconds=wall,
        **dict(timing), records=diagnostics, decode_order=[[addresses[i][0], addresses[i][1]] for i in order],
        order_is_shuffled=order != list(range(len(pages))) if len(pages) > 1 else None,
        maximum_dependency_depth=0, raw_prerequisite_pages=0, returned_output_bytes=returned,
        source_arrays_used=False, predictor_device=str(prepared.device), entropy_backend="CPU_reference_arithmetic",
        scope="One functional shuffled independent-page restore. Output buffers, neural/coder workspace and frozen weights remain necessary; no other page is a decoding prerequisite.")


def _encode(records, prepared, geometry, args, retained_codes=None):
    from . import side_codec as codec
    pages, traces, parts = [], [], Counter()
    started = time.perf_counter()
    for record in sorted(records, key=lambda r: (r["layer"], r["start"])):
        features = None if retained_codes is None else retained_codes[(record["layer"], record["start"])]
        wire, stats = codec.encode_page(record, prepared, feature_words=features)
        pages.append(wire); traces.append(stats)
        for name in ("feature_bytes", "entropy_payload_bytes", "framing_bytes"):
            parts[name] += stats[name]
        if sum(stats[k] for k in ("feature_bytes", "entropy_payload_bytes", "framing_bytes")) != len(wire):
            raise AssertionError("Side page component accounting mismatch")
    wire = pack_pages(pages, layers=geometry["layers"], tokens=geometry["tokens"],
                      channels=geometry["channels"], tile_tokens=args.tile_tokens)
    parts["container_framing_bytes"] = len(wire) - sum(map(len, pages))
    if sum(parts.values()) != len(wire):
        raise AssertionError("Side container component accounting mismatch")
    raw = sum(row["raw_bytes"] for row in traces)
    values = raw // 2
    groups = {}
    for field in ("layer", "start"):
        grouped = {}
        for trace in traces:
            key = str(trace[field]); row = grouped.setdefault(key, Counter())
            row["raw_bytes"] += trace["raw_bytes"]; row["packet_bytes"] += trace["packet_bytes"]
            row["feature_bytes"] += trace["feature_bytes"]; row["ideal_bits"] += trace["ideal_bits"]
            row["integer_cdf_bits"] += trace["integer_cdf_bits"]; row["records"] += 1
        groups["by_" + field] = {key: dict(row, packet_reduction_pct=100 * (1 - row["packet_bytes"] / row["raw_bytes"]))
                                   for key, row in grouped.items()}
    metrics = dict(raw_sample_bytes=raw, packet_bytes=len(wire), records_measured=len(records),
        records_total=geometry["layers"] * ((geometry["tokens"] + args.tile_tokens - 1) // args.tile_tokens),
        selected_starts=sorted({r["start"] for r in records}), diagnostics=groups,
        component_bytes=dict(parts), records=traces,
        encode_wall_seconds=time.perf_counter() - started,
        feature_encode_seconds=sum(r["feature_encode_seconds"] for r in traces),
        prediction_seconds=sum(r["prediction_seconds"] for r in traces),
        entropy_encode_seconds=sum(r["entropy_encode_seconds"] for r in traces),
        ideal_bits=sum(r["ideal_bits"] for r in traces),
        integer_cdf_bits=sum(r["integer_cdf_bits"] for r in traces),
        side_bits_per_KV_value=8 * parts["feature_bytes"] / values,
        complete_packet_bits_per_KV_value=8 * len(wire) / values,
        maximum_dependency_depth=0, raw_prerequisite_pages=0,
        encoder_placement="Retained code emitted inside attention on_tile callback" if retained_codes is not None else "Post-capture diagnostic replay",
        inline_codes_used=retained_codes is not None,
        scope="Actual page and indexed container bytes; CPU arithmetic coding, selected-device neural prediction. Ideal probability bits are diagnostics, not stored bytes.")
    return wire, metrics


def _check_sample(restored, matrices, starts, tile_tokens):
    seen = set()
    for record in restored:
        layer, start, count = record["layer"], record["start"], record["tokens"]
        if (layer, start) in seen:
            raise AssertionError("Independent restore duplicated a page")
        seen.add((layer, start))
        for kind, matrix in zip(("k", "v"), matrices[layer]):
            if not np.array_equal(record[kind], matrix[start:start + count]):
                raise AssertionError("Independent side page differs from actual captured KV")
    if seen != {(layer, start) for layer in range(len(matrices)) for start in starts}:
        raise AssertionError("Independent restore omitted a requested page")


def _artifact_info(model, output):
    files = []
    for filename in model.artifact_paths:
        path = Path(filename)
        files.append(dict(file=str(path.relative_to(output)), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if sum(row["bytes"] for row in files) != model.state_bytes:
        raise AssertionError("Side model artifact accounting differs from actual files")
    return dict(model_id=model.model_id, serialized_state_bytes=model.state_bytes,
                parameter_count=model.parameter_count, files=files, metadata=model.metadata,
                sharing_scope="Model shared once across compatible contexts; encoder included conservatively. Normal results packages omit models/ and packets/; files are hashed and fully charged.")


def _packet_file(output, name, wire):
    path = output / "packets" / (name + ".bin")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(wire)
    reopened = path.read_bytes()
    if reopened != wire:
        raise AssertionError("Persisted packet bytes changed")
    return reopened, dict(file=str(path.relative_to(output)), bytes=len(wire), sha256=hashlib.sha256(wire).hexdigest())


def _save(output, report):
    write_json(output / "results.json", report)
    fields = ["case", "tokens", "method", "status", "sample_payload_reduction_pct", "net_single_cache_estimated_reduction_pct", "charged_state_bytes", "dependency_depth", "decode_seconds"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in report["cases"]:
            for row in case["results"]:
                acc, measure = row.get("accounting", {}), row.get("measurement", {})
                writer.writerow(dict(case=case["name"], tokens=case["tokens"], method=row["name"], status=row["status"],
                    sample_payload_reduction_pct=acc.get("sample", {}).get("payload_reduction_pct"),
                    net_single_cache_estimated_reduction_pct=acc.get("full_cache_estimate", {}).get("net_reduction_pct"),
                    charged_state_bytes=acc.get("state", {}).get("total_charged_bytes"),
                    dependency_depth=measure.get("maximum_dependency_depth"),
                    decode_seconds=measure.get("independent_packet_decode", {}).get("wall_seconds", measure.get("decode_wall_seconds"))))
    lines = ["# Prefill side-code probability study", "", f"Status: **{report['status']}**", "", report["scope"], "",
        "| Case | Tokens | Method | Packet savings | Net full-cache estimate | Dependency depth |", "|---|---:|---|---:|---:|---:|"]
    for case in report["cases"]:
        for row in case["results"]:
            acc = row.get("accounting", {})
            def fmt(value):
                return "—" if value is None else f"{value:.2f}%"
            lines.append(f"| {case['name']} | {case['tokens']} | {row['name']} ({row['status']}) | {fmt(acc.get('sample', {}).get('payload_reduction_pct'))} | {fmt(acc.get('full_cache_estimate', {}).get('net_reduction_pct'))} | {row.get('measurement', {}).get('maximum_dependency_depth', '—')} |")
    lines += ["", "Local variants none/kv/kv_canonical/hidden share rank, decoder width, update budget and TRAIN-only data. The previous-layer control has separately reported input width, decoder width and training budget; it is not a parameter-matched proof.", "",
        "All side-code bytes, CPU entropy streams, page metadata, actual indexed container and max(serialized,resident) complete model state are charged. Sampled full-context and shared-state concurrency numbers are ESTIMATES. They do not demonstrate a serving gain.", "",
        "Features come from actual normalized hidden states feeding Qwen KV projections; learned BF16 codes are emitted inside attention callbacks and passed verbatim to later entropy encoding. Original model forward still materializes full KV. CPU entropy coding and device neural inference are timed separately. Random-page decode requires own packet plus frozen state; no previous-layer KV chain.", "",
        "Full DEV packet-only generation checks:", ""]
    for check in report["generation_checks"]:
        lines.append(f"- {check['name']}: {check['status']}")
    if report.get("error"):
        lines += ["", "Error: " + report["error"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    from . import side_probability as side, neural_probability as neural, neural_chain as chain
    from .neural_accounting import account
    from .runner import _source_manifest
    parser = _parser(); args = parser.parse_args(argv); _defaults(args)
    positive = [args.rank, args.width, args.steps, args.batch_tokens, args.train_tokens, args.dev_tokens,
                args.chunk_tokens, args.tile_tokens, args.max_pairs, args.threads, args.shared_width,
                args.shared_steps, *args.lengths]
    if any(n < 1 for n in positive) or not 1 <= args.generation_steps <= 64 or args.seed < 0:
        parser.error("Budgets, dimensions, token counts and pair limits must be positive; generation steps in 1..64 and seed nonnegative")
    output = args.output or Path("runs") / ("prefill_side_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True); output = output.resolve()
    report = dict(format="qwen_prefill_side_probability_study_v1", status="running",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        source_manifest=_source_manifest(), model_identity=None, models={}, cases=[], generation_checks=[],
        scope="Exact BF16 page-independent probability coding with no code, a same-page KV-derived code, an inverse-RoPE-aligned KV-derived code, or a producer-hidden-derived code. Four shared models fit a rate proxy on TRAIN only; actual retained BF16 codes and CPU arithmetic streams determine measured bytes. Complete state and framing charged. No compact attention, GPU entropy coding, concurrency scheduler, latency or novelty claim is established.")
    write_json(output / "source_manifest.json", report["source_manifest"]); _save(output, report)
    began = time.perf_counter()
    try:
        device = runtime.select_device(args.device)
        torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
        report["environment"] = runtime.doctor(device, probe=False)
        _note("Loading model for prefill-side probability study on " + str(device))
        if args.tiny:
            from .producer_study import _tiny
            report["input_preflight"] = previous._preflight_inputs(None, args)
            model, tokenizer, identity = _tiny(device, args)
        else:
            def validate_inputs(tokenizer):
                report["input_preflight"] = previous._preflight_inputs(tokenizer, args)
                _note("Prompt-budget preflight passed before weights")
            model, tokenizer, identity = runtime.load_model(args.model, args.revision, device,
                args.local_files_only, tokenizer_validator=validate_inputs)
        report["model_identity"] = identity
        layers, head_dim = int(model.config.num_hidden_layers), int(model.config.head_dim)
        if args.max_pairs < layers:
            raise ValueError(f"--max-pairs {args.max_pairs} must cover all {layers} layers")
        inverse, scaling = previous._rope(model); rope_signature = (inverse.tobytes(), scaling)
        train_records, train_matrices, training_metadata = [], [], []
        for fixture in TRAIN:
            _note("TRAIN producer capture: " + fixture["name"])
            capture, records, context = _capture(model, tokenizer, fixture, args.train_tokens, device, args)
            if not context.get("rope_stable"):
                raise ValueError("Producer side model requires fixed RoPE frequencies")
            train_records.append(records); train_matrices.append(capture["matrices"])
            training_metadata.append(dict(split="train", name=fixture["name"], tokens=args.train_tokens,
                sha256=capture["producer_records_sha256"], producer_records_sha256=capture["producer_records_sha256"],
                input_ids_sha256=capture["input_ids_sha256"], kv_words_sha256=capture["kv_words_sha256"],
                geometry=capture["geometry"], producer_audit=capture["producer_audit"], timings=capture["timings"]))
            del capture, records, context
        channels = train_matrices[0][0][0].shape[1]
        report["training_captures"] = training_metadata
        training_hashes = [_records_hash(records) for records in train_records]
        anchor = previous.fit_anchor(train_matrices, training_metadata, identity, device)
        (output / "calibration").mkdir(exist_ok=True)
        anchor.dump(output / "calibration" / "neural_anchor_state.bin")
        report["anchor_state"] = dict(anchor_codec.state_info(anchor), file="calibration/neural_anchor_state.bin")
        control_batch_tokens = min(args.batch_tokens, args.tile_tokens, args.train_tokens)
        report["training_budget"] = dict(local_variants=["none", "kv", "kv_canonical", "hidden"], updates_each=args.steps,
            rank=args.rank, width=args.width, requested_minibatch_tokens=args.batch_tokens,
            local_maximum_minibatch_tokens=control_batch_tokens, previous_layer_minibatch_tokens=control_batch_tokens,
            minibatch_rule="Local models sample one tile, clipping requested tokens to that record length; chain control clips to min(requested,tile_tokens,train_tokens). Tail tiles may have fewer tokens and realized local examples are logged.",
            previous_layer_control_included=not args.skip_chain_control,
            previous_layer_shared_updates=args.shared_steps, previous_layer_shared_width=args.shared_width,
            matching="Same TRAIN documents, local decoder width, retained rank (kv/kv_canonical/hidden), update count and sampling seed; input dimensions and encoder parameter counts differ and are charged. Previous-layer control width/budget separately disclosed.")
        _note("Fitting TRAIN-only local probability models: none, kv, kv_canonical, hidden")
        fit_started = time.perf_counter()
        models = side.fit(train_records, output / "models", training_metadata=training_metadata,
            model_identity=identity, inverse=inverse, rope_scaling=scaling, head_dim=head_dim,
            rank=args.rank, width=args.width, steps=args.steps, batch_tokens=args.batch_tokens,
            device=device, seed=args.seed)
        for name, fitted in models.items():
            reloaded = side.load(fitted.path, expected_model_identity=identity)
            if reloaded.model_id != fitted.model_id:
                raise AssertionError("Reloaded side model identity differs")
            models[name] = reloaded
            report["models"][name] = _artifact_info(reloaded, output)
        shared = None
        if not args.skip_chain_control:
            _note("Fitting produced-KV previous-layer shared control")
            shared = neural.fit(train_matrices, output / "models", model_identity=identity,
                training_metadata=training_metadata, head_dim=head_dim, rope_inv_freq=inverse,
                rope_scaling=scaling, device=device, width=args.shared_width, steps=args.shared_steps,
                batch_tokens=control_batch_tokens, seed=args.seed, modes=("shared",))["shared"]
            shared = neural.load(shared.path, expected_model_identity=identity, expected_model_id=shared.model_id, expected_mode="shared")
            report["models"]["previous_layer_shared"] = previous._artifact_info(shared, output)
        if training_hashes != [_records_hash(records) for records in train_records]:
            raise AssertionError("TRAIN producer records changed during fitting")
        report["training_wall_seconds"] = time.perf_counter() - fit_started
        freeze = dict(format="prefill_side_probability_freeze_v1", model_identity=identity,
            source_sha256=report["source_manifest"]["sources_sha256"], training_hashes=training_hashes,
            models=report["models"], anchor_state=report["anchor_state"], training_budget=report["training_budget"],
            protocol="Every local variant and optional chain control frozen before DEV/TEST. TEST is never used to choose models or hyperparameters.")
        write_json(output / "frozen_manifest.json", freeze); report["freeze"] = freeze
        del train_records, train_matrices; gc.collect(); _save(output, report)

        _note("DEV complete producer capture and independent packet-only continuation")
        inline = InlineCodes(models, device)
        development, records, context = _capture(model, tokenizer, DEV[0], args.dev_tokens, device, args, inline=inline)
        inline.release_prepared()
        if (context["rope_inv_freq"].tobytes(), float(context["rope_attention_scaling"])) != rope_signature:
            raise ValueError("Producer RoPE context changed after frozen TRAIN")
        for name, fitted in models.items():
            _note("DEV independent packets: " + name)
            prepared = fitted.prepare(device)
            wire, measured = _encode(records, prepared, development["geometry"], args, inline.codes[name])
            wire, artifact = _packet_file(output, "dev_" + name, wire)
            def restore(wire=wire, prepared=prepared):
                return restore_pages(wire, prepared, require_full=True, seed=args.seed)[0]
            restored = restore()
            if previous._matrices_hash(restored) != development["kv_words_sha256"]:
                raise AssertionError("Complete packet-only side cache differs from actual DEV KV")
            del restored
            check = runtime.generation_check(model, development["matrices"], restore, device,
                first_token=development["first_token"], steps=args.generation_steps)
            if not previous._generation_passed(check):
                raise AssertionError("Side-code full DEV finite/bitwise continuation failed")
            report["generation_checks"].append(dict(name="side_" + name, status="ok", result=check,
                complete_prefix_tokens=args.dev_tokens, full_cache_word_hash=development["kv_words_sha256"],
                packet_artifact=artifact, measurement=measured, inline_side_codes=development["inline_side_codes"]["methods"][name],
                decode_inputs="Persisted serialized container and frozen prepared state only; no producer arrays or other decoded-page prerequisite"))
            del restore, prepared, wire; gc.collect(); _save(output, report)
        if shared is not None:
            _note("DEV previous-layer shared control")
            prepared = shared.prepare(device)
            packets, measured = chain.encode_cache(development["matrices"], prepared, anchor,
                tile_tokens=args.tile_tokens, max_pairs=layers * ((args.dev_tokens + args.tile_tokens - 1) // args.tile_tokens), head_dim=head_dim)
            wire, artifact = _packet_file(output, "dev_previous_layer_shared", packets.to_bytes()); del packets
            def restore(wire=wire, prepared=prepared):
                return chain.decode_cache(chain.NeuralCache.from_bytes(wire), prepared, anchor,
                    layers=layers, tokens=args.dev_tokens, channels=channels, require_full=True)
            check = runtime.generation_check(model, development["matrices"], restore, device,
                first_token=development["first_token"], steps=args.generation_steps)
            if not previous._generation_passed(check):
                raise AssertionError("Previous-layer shared DEV continuation failed")
            report["generation_checks"].append(dict(name="previous_layer_shared", status="ok", result=check,
                complete_prefix_tokens=args.dev_tokens, packet_artifact=artifact, measurement=measured))
            del restore, prepared, wire; gc.collect(); _save(output, report)
        del development, records, context, inline

        for length in args.lengths:
            starts = previous._starts(length, args.tile_tokens, layers, args.max_pairs)
            for fixture in TEST[:args.cases]:
                _note(f"TEST producer capture: {fixture['name']}, {length} tokens, starts {starts}")
                inline = InlineCodes(models, device)
                captured, records, context = _capture(model, tokenizer, fixture, length, device, args, starts, inline=inline)
                inline.release_prepared()
                if (context["rope_inv_freq"].tobytes(), float(context["rope_attention_scaling"])) != rope_signature:
                    raise ValueError("Producer RoPE context changed after frozen TRAIN")
                matrices = captured.pop("matrices")
                expected_addresses = {(layer, start) for layer in range(layers) for start in starts}
                if {(row["layer"], row["start"]) for row in records} != expected_addresses:
                    raise AssertionError("Side and control addresses do not match")
                case = dict(name=fixture["name"], split="test", tokens=length, capture=captured,
                    selected_starts=starts, matched_record_addresses=sorted([list(x) for x in expected_addresses]), results=[])
                report["cases"].append(case)
                control = previous._baseline(matrices, anchor, starts, args.tile_tokens, head_dim)
                control["maximum_dependency_depth"] = 0
                case["results"].append(dict(name="frozen_exponent_control", status="ok", measurement=control,
                    accounting=account(control["raw_sample_bytes"], control["packet_bytes"], captured["raw_payload_bytes"],
                                       0, 0, anchor.serialized_bytes, anchor.physical_nbytes)))
                for name, fitted in models.items():
                    _note("TEST independent arithmetic packets: " + name)
                    initial = runtime.reset_peak_memory(device)
                    prepared, preparation = runtime.timed_call(lambda: fitted.prepare(device), device)
                    wire, measured = _encode(records, prepared, captured["geometry"], args, inline.codes[name])
                    wire, artifact = _packet_file(output, f"test_{fixture['name']}_{length}_{name}", wire)
                    restored, decoded = restore_pages(wire, prepared, seed=args.seed)
                    _check_sample(restored, matrices, starts, args.tile_tokens)
                    measured.update(independent_packet_decode=decoded, roundtrip_exact=True,
                        container_roundtrip_verified=True, packet_artifact=artifact, state_preparation=preparation,
                        initial_device_memory=initial, after_device_memory=runtime.memory_snapshot(device))
                    acc = account(measured["raw_sample_bytes"], len(wire), captured["raw_payload_bytes"],
                                  fitted.state_bytes, prepared.physical_nbytes, 0, 0)
                    case["results"].append(dict(name="side_" + name, status="ok", measurement=measured,
                        accounting=acc, execution_contract=getattr(prepared, "execution_contract", None),
                        matched_control_sample_bytes=control["packet_bytes"],
                        actual_stream_byte_delta_vs_control=len(wire) - control["packet_bytes"]))
                    del restored, prepared, wire; gc.collect(); _save(output, report)
                if shared is not None:
                    _note("TEST previous-layer shared arithmetic control")
                    prepared = shared.prepare(device)
                    packets, measured = chain.encode_cache(matrices, prepared, anchor,
                        tile_tokens=args.tile_tokens, max_pairs=args.max_pairs, head_dim=head_dim)
                    wire, artifact = _packet_file(output, f"test_{fixture['name']}_{length}_previous_layer_shared", packets.to_bytes()); del packets
                    checked, decoded = chain.decode_cache(chain.NeuralCache.from_bytes(wire), prepared, anchor,
                        layers=layers, tokens=length, channels=channels, require_full=False, return_metrics=True)
                    _check_sample(checked, matrices, starts, args.tile_tokens)
                    raw_sample = sum(r["k"].nbytes + r["v"].nbytes for r in checked)
                    measured.update(packet_artifact=artifact, independent_packet_decode=decoded,
                                    raw_sample_bytes=raw_sample, roundtrip_exact=True)
                    case["results"].append(dict(name="previous_layer_shared", status="ok", measurement=measured,
                        accounting=account(raw_sample, len(wire), captured["raw_payload_bytes"],
                            shared.state_bytes, prepared.physical_nbytes, anchor.serialized_bytes, anchor.physical_nbytes)))
                    del checked, prepared, wire; gc.collect(); _save(output, report)
                del matrices, records, captured, context, inline; gc.collect()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _note(report["error"])
    report["elapsed_wall_seconds"] = time.perf_counter() - began
    _save(output, report); _note("Report: " + str(output / "summary.md")); _note("Status: " + report["status"])
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
