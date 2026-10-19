"""Forward-producer metadata study with exact paired K/V reconstruction.

Hooks observe real forward intermediates and, by default, build/verify recipes
inside attention forward. Repeated codec timings replay captured producer
features after prefill; the full raw model cache remains necessary.
This module orchestrates capture, TRAIN-only fits, matched microbenchmarks and
full-DEV continuation checks. It does not implement compact-resident attention.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
import zipfile

import numpy as np
import torch

from . import gpu_ops as g, model_runtime as runtime
from .fixtures import TRAIN, DEV, TEST, build_input_ids
from .reporting import checkpoint, write_json


def _note(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default=runtime.DEFAULT_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--tiny", action="store_true", help="No-download random tiny Qwen; functionality only")
    p.add_argument("--producer-preset", choices=("baseline", "frontier"), default="baseline")
    p.add_argument("--lengths", nargs="+", type=int)
    p.add_argument("--cases", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--train-tokens", type=int)
    p.add_argument("--dev-tokens", type=int)
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--tile-tokens", type=int, default=64)
    p.add_argument("--max-tiles", type=int, default=72, help="Selected layer/tile PAIRS; must cover every layer")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--sample-pairs", type=int, default=16, help="Bounded repeated-timing pair count, identical selection for every recipe")
    p.add_argument("--generation-steps", type=int, default=2)
    p.add_argument("--skip-inline", action="store_true", help="Disable recipe encoding inside attention producer callbacks; post-capture replay remains")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--output", type=Path)
    return p


def _defaults(args):
    for name, default in dict(lengths=[64, 96] if args.tiny else [2048, 8192], train_tokens=32 if args.tiny else 512,
                              dev_tokens=24 if args.tiny else 512, chunk_tokens=8 if args.tiny else 512).items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.tiny and args.device == "auto":
        args.device = "cpu"
    if args.revision is None and args.model == runtime.DEFAULT_MODEL:
        args.revision = runtime.DEFAULT_REVISION


def _tiny(device, args):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=max(args.lengths + [args.train_tokens, args.dev_tokens]) + 64,
        attention_dropout=0.)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
    return model, None, dict(model_id="local-random-tiny-qwen3-producer-study", tiny=True,
        seed=args.seed, config=config.to_dict(), dtype="bfloat16", device=str(device),
        model_storage_bytes=runtime.model_storage_bytes(model),
        scope="Random local weights; no pretrained checkpoint, tokenizer, quality or GPU performance claim")


def _capture(model, tokenizer, fixture, tokens, device, args, maximum, inline=None):
    from .producer_hooks import ProducerCapture
    if args.tiny:
        namespace = {"train": 11, "dev": 47, "test": 71}[fixture["split"]]
        topic = sum(fixture["name"].encode())
        ids = [1 + ((index * (namespace + 2) + topic + namespace) % 95) for index in range(tokens)]
    else:
        ids = build_input_ids(tokenizer, fixture, tokens)
    with ProducerCapture(model, len(ids), tile_tokens=args.tile_tokens, max_tiles=maximum,
                         on_tile=inline) as hooks:
        captured = runtime.capture(model, ids, device, args.chunk_tokens)
    records, audit, context = hooks.finalize(captured["matrices"])
    captured["timings"]["prefill"].update(producer_hooks_included=True,
        inline_producer_recipes_included=inline is not None,
        scope="Instrumented model forward includes producer hooks, diagnostic CPU copies and optional inline recipes; no uninstrumented prefill baseline is measured.")
    for record in records:
        record["split"] = fixture["split"]
    captured["input_ids_sha256"] = _digest(ids)
    captured["producer_audit"] = audit
    if inline is not None:
        captured["inline_recipe_metrics"] = inline.report()
    return captured, records, context


def _context_info(context):
    weights = context.get("norm_weights", {})
    frequency = np.asarray(context["rope_inv_freq"])
    return dict(rope_stable=bool(context.get("rope_stable")),
        rope_attention_scaling=float(context["rope_attention_scaling"]),
        rope_inv_freq=dict(dtype=str(frequency.dtype), shape=list(frequency.shape),
            sha256=hashlib.sha256(frequency.tobytes()).hexdigest()),
        norm_eps={str(k): v for k, v in context.get("norm_eps", {}).items()},
        norm_weight_shapes={str(k): list(np.asarray(v).shape) for k, v in weights.items()},
        rule=context.get("position_rule"), head_dim=context.get("head_dim"))


def _check(pair, record):
    if len(pair) != 2:
        raise AssertionError("Producer decoder must return both K and V")
    for actual, name in zip(pair, ("k", "v")):
        value = actual.detach().cpu().numpy()
        if value.dtype != np.uint16 or not np.array_equal(value, record[name]):
            raise AssertionError("Producer reconstructed KV words differ from original " + name)


def _times(values, raw):
    median = float(statistics.median(values))
    return dict(seconds=values, median_seconds=median, repeats=len(values), payload_bytes=raw,
        MiB_per_second=raw / (1 << 20) / median if median else None)


def _state_file(state, codec, output, known):
    wire = bytes(codec.state_bytes(state))
    digest = hashlib.sha256(wire).hexdigest()
    relative = f"calibration/producer_state_{digest}.bin"
    path = output / relative
    if relative not in known:
        path.parent.mkdir(exist_ok=True)
        state.dump(path)
        if path.read_bytes() != wire:
            raise AssertionError("Dumped producer state differs from charged serialized artifact")
        known[relative] = dict(bytes=len(wire), sha256=digest)
    info = dict(codec.state_info(state))
    info.update(file=relative, sha256=digest)
    return info


def _codec_or_default(codec):
    if codec is None:
        from . import producer_codec as codec
    return codec


class _Inline:
    """Bounded verification callback; recipes are built before attention returns."""
    def __init__(self, specs, models, device, codec=None):
        self.specs, self.models, self.device = specs, models, device
        self.codec = _codec_or_default(codec)
        self.states, self.metrics = {}, {}
    def __call__(self, record, context):
        codec = self.codec
        for spec in self.specs:
            name = spec["name"]
            row = self.metrics.setdefault(name, dict(name=name, status="ok", records_checked=0,
                serialized_record_bytes=0, raw_bytes=0, component_bytes=Counter(), records=[],
                state_preparation_seconds=0., input_staging_seconds_total=0.,
                encode_seconds_total=0., decode_seconds_total=0.))
            if row["status"] != "ok":
                continue
            try:
                if name not in self.states:
                    state, timing = runtime.timed_call(lambda: codec.make_state(spec, self.models, context, self.device), self.device)
                    self.states[name] = state
                    row.update(state_preparation_seconds=timing["wall_seconds"], state_id=state.state_id,
                        decoder_state_bytes=state.state_bytes)
                state = self.states[name]
                prepared, stage = runtime.timed_call(lambda: codec.prepare_record(record, self.device, spec=spec), self.device)
                packet, encode = runtime.timed_call(lambda: codec.encode(prepared, spec, state), self.device)
                pair, decode = runtime.timed_call(lambda: codec.decode(packet, state), self.device)
                _check(pair, record)
                stats = packet.stats
                row["records_checked"] += 1
                row["serialized_record_bytes"] += stats["serialized_bytes"]
                row["raw_bytes"] += record["k"].nbytes + record["v"].nbytes
                row["component_bytes"].update(stats["component_bytes"])
                row["records"].append(dict(layer=record["layer"], start=record["start"], tokens=record["tokens"],
                    serialized_bytes=stats["serialized_bytes"], component_bytes=stats["component_bytes"]))
                for key, timing in (("input_staging_seconds_total", stage), ("encode_seconds_total", encode), ("decode_seconds_total", decode)):
                    row[key] += timing["wall_seconds"]
                del pair, packet, prepared
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
    def report(self):
        return dict(enabled=True, methods=list(self.metrics.values()),
            scope="Recipe encoding and exact verification execute inside attention producer callbacks before its forward returns. Hook diagnostics first copy selected features to CPU; preparation stages them back to the selected device. Instrumented prefill includes state preparation, all recipes, synchronization and verification. Packets are discarded; full raw KV remains resident. This is neither zero-copy nor fused encoding, nor the overhead of deploying one recipe.")


def _measure(records, captured, spec, models, context, device, args, output, state_files, codec=None):
    codec = _codec_or_default(codec)
    state = codec.make_state(spec, models, context, device)
    state_info = _state_file(state, codec, output, state_files)
    count = len(records)
    selected = set(np.linspace(0, count - 1, min(count, args.sample_pairs), dtype=np.int64).tolist())
    components = Counter(); traces = []; samples = []
    payload = raw = 0; staging = encoding = decoding = 0.; events = Counter()
    initial = runtime.reset_peak_memory(device)
    started = time.perf_counter()
    for index, record in enumerate(records):
        if hasattr(codec, "prepare_record"):
            prepared, stage = runtime.timed_call(lambda: codec.prepare_record(record, device, spec=spec), device)
            staging += stage["wall_seconds"]
        else:
            prepared = record
        packet, encode = runtime.timed_call(lambda: codec.encode(prepared, spec, state), device)
        pair, decode = runtime.timed_call(lambda: codec.decode(packet, state), device)
        _check(pair, record)  # D2H correctness validation is outside decode timing.
        stats = packet.stats
        parts = stats["component_bytes"]
        size = stats["serialized_bytes"]
        if sum(parts.values()) != size:
            raise AssertionError("Incomplete producer packet byte accounting")
        nraw = record["k"].nbytes + record["v"].nbytes
        payload += size; raw += nraw; components.update(parts)
        encoding += encode["wall_seconds"]; decoding += decode["wall_seconds"]
        events["encode_seconds_total"] += encode["device_event_seconds"] or 0.
        events["decode_seconds_total"] += decode["device_event_seconds"] or 0.
        traces.append(dict(ordinal=index, layer=record["layer"], start=record["start"], tokens=record["tokens"],
            raw_bytes=nraw, serialized_bytes=size, component_bytes=parts,
            encode_seconds=encode["wall_seconds"], decode_seconds=decode["wall_seconds"]))
        if getattr(args, "producer_preset", "baseline") == "frontier" and hasattr(codec, "describe"):
            traces[-1]["recipe_diagnostics"] = codec.describe(packet)
        if index in selected:
            samples.append((g.packet_to_cpu(packet), record))
        del packet, pair, prepared
    shape = captured["geometry"]
    total = shape["layers"] * ((shape["tokens"] + args.tile_tokens - 1) // args.tile_tokens)
    outer = 32 + 8 * (count + 1)
    full_outer = 32 + 8 * (total + 1)
    charged = int(state.state_bytes)
    serialized = payload + outer
    estimated = payload * total / count + full_outer
    full_raw = captured["raw_payload_bytes"]
    if raw <= 0 or count == 0:
        raise ValueError("Empty producer measurement")
    components.update(cache_header=32, pair_offset_table=8 * (count + 1))
    result = dict(execution="producer_tensor_accelerator" if device.type != "cpu" else "producer_tensor_cpu_functional",
        device=str(device), records_total=total, records_measured=count, records_checked=count,
        coverage="full" if count == total else "sampled", size_is_estimated=count != total,
        raw_payload_bytes=raw, full_raw_payload_bytes=full_raw,
        actual_record_payload_bytes=payload, calculated_outer_framing_bytes=outer,
        serialized_bytes=serialized, component_bytes=dict(components),
        decoder_state_bytes=charged, native_state_physical_bytes=int(state.physical_nbytes), decoder_state_info=state_info,
        state_id=state.state_id, estimated_full_cache_serialized_bytes=estimated if count != total else None,
        savings_including_state_percent=100 * (1 - (serialized + charged) / raw),
        estimated_full_savings_including_state_percent=100 * (1 - (estimated + charged) / full_raw) if count != total else None,
        roundtrip_exact=True, exact_verified=True, input_staging_seconds_total=staging,
        encode_seconds_total=encoding, decode_seconds_total=decoding,
        producer_encode_including_feature_staging_seconds=staging + encoding,
        encode_includes_input_staging=not hasattr(codec, "prepare_record"),
        device_event_elapsed=dict(events) if device.type != "cpu" else None,
        memory=dict(initial=initial, after_measurement=runtime.memory_snapshot(device)),
        one_pass_total_wall_seconds=time.perf_counter() - started, record_diagnostics=traces,
        byte_accounting="Actual paired K/V packet fields plus calculated 32-byte cache header and 8*(pair_count+1) index. Complete decoder state charged once under its reported max(serialized,resident) contract. Sampled full-cache estimates scale inner record bytes by total/measured; outer container is not persisted.",
        scope="Producer features observed during actual forward, exported to CPU, then replay encoded on the selected device. Needed-input H2D staging and encode are reported separately when prepare_record is available. Full raw capture remains necessary; no compact attention or serving gain.")
    full_stored = (estimated if count != total else serialized) + charged
    effective_bits = 16 * full_stored / full_raw
    result["effective_bits_per_BF16_word_including_state"] = effective_bits
    result["target_40_percent"] = dict(target_bits_per_word=9.6, effective_bits_per_word=effective_bits,
        gap_bits_per_word=effective_bits - 9.6, met=effective_bits <= 9.6,
        size_is_estimated=count != total, scope="Actual packet/state accounting, extrapolated only where flagged; entropy diagnostics cannot satisfy this target.")
    raw_sample = sum(record["k"].nbytes + record["v"].nbytes for _, record in samples)
    sample_times = []; restore_times = []
    for repeat in range(args.repeats + 1):
        decode_sum = restore_sum = 0.
        for cpu_packet, record in samples:
            packet, transfer = runtime.timed_call(lambda: g.packet_to_device(cpu_packet, device), device)
            pair, elapsed = runtime.timed_call(lambda: codec.decode(packet, state), device)
            _check(pair, record)
            decode_sum += elapsed["wall_seconds"]
            restore_sum += transfer["wall_seconds"] + elapsed["wall_seconds"]
            del packet, pair
        if repeat:
            sample_times.append(decode_sum); restore_times.append(restore_sum)
    result["sample"] = dict(records=len(samples), raw_bytes=raw_sample,
        encoded_bytes=sum(packet.physical_nbytes for packet, _ in samples),
        selection="Same evenly spaced pair ordinals for every recipe; bounded by --sample-pairs")
    result["sample_decode"] = dict(**_times(sample_times, raw_sample),
        scope="Packet on target device before timing; synchronized decode, one warmup; validation D2H excluded")
    result["sample_packet_transfer_then_decode"] = dict(**_times(restore_times, raw_sample),
        scope="Per-field pageable CPU packet copy/validation/allocation + device decode; reconstructed-word D2H validation excluded")
    inline = captured.get("inline_recipe_metrics")
    if inline:
        observed = next(row for row in inline["methods"] if row["name"] == spec["name"])
        expected = {(t["layer"], t["start"], t["tokens"]): (t["serialized_bytes"], t["component_bytes"]) for t in traces}
        actual = {(t["layer"], t["start"], t["tokens"]): (t["serialized_bytes"], t["component_bytes"]) for t in observed["records"]}
        if observed["status"] != "ok" or observed.get("state_id") != state.state_id or actual != expected or observed["records_checked"] != count:
            raise AssertionError("Inside-forward producer bytes/state differ from post-capture replay")
        result["inline_record_payload_matches_replay"] = True
    del samples, state
    return result


def _generation(model, captured, records, specs, models, context, device, args, output, state_files, codec=None):
    codec = _codec_or_default(codec)
    wanted = codec.generation_representatives() if hasattr(codec, "generation_representatives") else ("producer_kv_median", "producer_hidden_exponent", "producer_pre_rope_coset")
    results = []
    for name in wanted:
        row = dict(name=name, track="producer_pair", context_tokens=args.dev_tokens,
            selection="Fixed representative before TEST; full DEV producer coverage independent of sampling")
        results.append(row)
        if args.generation_steps == 0:
            row.update(status="skipped", reason="--generation-steps 0")
            continue
        try:
            spec = next(spec for spec in specs if spec["name"] == name)
            _note("Full DEV continuation: " + name)
            state = codec.make_state(spec, models, context, device)
            _state_file(state, codec, output, state_files)
            packets = [(record, g.packet_to_cpu(codec.encode(record, spec, state))) for record in records]
            def restore():
                matrices = [(np.zeros_like(k), np.zeros_like(v)) for k, v in captured["matrices"]]
                coverage = [np.zeros(k.shape[0], np.uint8) for k, _ in matrices]
                for record, cpu_packet in packets:
                    packet = g.packet_to_device(cpu_packet, device)
                    pair = codec.decode(packet, state)
                    layer, start, length = record["layer"], record["start"], record["tokens"]
                    for target, value in zip(matrices[layer], pair):
                        target[start:start + length] = value.detach().cpu().numpy()
                    coverage[layer][start:start + length] += 1
                if any(not np.all(seen == 1) for seen in coverage):
                    raise AssertionError("Full DEV generation callback has missing or overlapping producer tiles")
                return matrices
            measurement = runtime.generation_check(model, captured["matrices"], restore, device,
                first_token=captured["first_token"], steps=args.generation_steps)
            if not all(measurement[key] for key in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal")):
                raise AssertionError("Producer full DEV continuation differs")
            measurement["producer_restore_scope"] = "GPU decoder outputs return to CPU for existing generation-check cache reconstruction; this roundtrip is correctness plumbing, not a serving latency claim."
            row.update(status="ok", measurement=measurement)
            del packets, state
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
    return results


def _summary(report):
    cases = []
    lines = ["# Forward producer metadata study", "", f"Status: **{report['status']}**", "",
        "Default callbacks build and verify recipes inside actual attention forward, using diagnostic CPU copies staged back to the selected device. Repeated codec timings separately replay captured features after prefill. All payload/state bytes count; full raw KV remains required. CPU tiny runs establish functionality only. No compact attention, concurrency or latency gain is inferred.", "",
        "| Split | Case | Method | Status | Savings incl. state | Delta vs frozen, pp | Bits/word incl. state | 40% target | Records | Decode ms/raw MiB |", "|---|---|---|---|---:|---:|---:|---|---:|---:|"]
    for case in report["cases"]:
        entries = []
        frozen = next((r.get("measurement") for r in case["results"] if r["name"] == "producer_frozen_exponent" and r["status"] == "ok"), None)
        baseline = (frozen.get("estimated_full_savings_including_state_percent") if frozen.get("size_is_estimated") else frozen.get("savings_including_state_percent")) if frozen else None
        for row in case["results"]:
            m = row.get("measurement") or {}
            saving = m.get("estimated_full_savings_including_state_percent") if m.get("size_is_estimated") else m.get("savings_including_state_percent")
            sample = m.get("sample_decode") or {}
            ms = 1000 * sample["median_seconds"] * (1 << 20) / sample["payload_bytes"] if sample.get("payload_bytes") else None
            entries.append(dict(name=row["name"], status=row["status"], savings_percent=saving,
                size_is_estimated=m.get("size_is_estimated"), sample_decode_ms_per_raw_MiB=ms,
                decoder_state_bytes=m.get("decoder_state_bytes"), component_bytes=m.get("component_bytes"),
                raw_payload_bytes=m.get("raw_payload_bytes"), input_staging_seconds_total=m.get("input_staging_seconds_total"),
                encode_seconds_total=m.get("encode_seconds_total"), decode_seconds_total=m.get("decode_seconds_total"),
                effective_bits_per_BF16_word_including_state=m.get("effective_bits_per_BF16_word_including_state"),
                gain_vs_frozen_percentage_points=saving - baseline if saving is not None and baseline is not None else None,
                target_40_percent=m.get("target_40_percent"),
                producer_encode_including_feature_staging_seconds=m.get("producer_encode_including_feature_staging_seconds")))
            number = "—" if saving is None else f"{saving:.3f}%" + (" estimated" if m.get("size_is_estimated") else " measured")
            decode = "—" if ms is None else f"{ms:.3f}"
            gain = "—" if saving is None or baseline is None else f"{saving - baseline:+.3f}"
            bits = m.get("effective_bits_per_BF16_word_including_state")
            ledger = "—" if bits is None else f"{bits:.4f}"
            target = "—" if bits is None else "yes (estimated)" if bits <= 9.6 and m.get("size_is_estimated") else "yes" if bits <= 9.6 else "no"
            lines.append(f"| {case['split']} | {case['name']} | {row['name']} | {row['status']} | {number} | {gain} | {ledger} | {target} | {m.get('records_measured', '—')}/{m.get('records_total', '—')} | {decode} |")
        cases.append(dict(name=case["name"], split=case["split"], tokens=case["tokens"], methods=entries,
            scope="All fixed candidate outcomes; no TEST-selected policy or pooled DEV/TEST ranking"))
    lines += ["", "Continuation checks use complete DEV coverage and compare exact logits, tokens and appended KV. Hook auditing and producer feature copies are included in instrumented prefill capture; that duration is not production overhead for one deployed recipe.", ""]
    if report.get("fatal_error"):
        lines += ["Run error: " + report["fatal_error"], ""]
    return "\n".join(lines), dict(format="qwen_producer_summary_v1", status=report["status"], cases=cases,
        generation_checks=[dict(name=r["name"], status=r["status"]) for r in report["generation_checks"]], scope=report["scope"])


def _bundle(output, state_files):
    names = [name for name in ("results.json", "results.csv", "summary.md", "producer_summary.json", "source_manifest.json", "frozen_manifest.json") if (output / name).is_file()]
    manifest = []
    for relative in names + sorted(state_files):
        data = (output / relative).read_bytes()
        info = dict(file=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if relative in state_files and {k: info[k] for k in ("bytes", "sha256")} != state_files[relative]:
            raise AssertionError("Producer calibration artifact changed before bundle creation")
        manifest.append(info)
    archive = output.with_name(output.name + "_results.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for info in manifest:
            zipped.write(output / info["file"], info["file"])
        zipped.writestr("bundle_manifest.json", json.dumps(dict(format="qwen_producer_bundle_v1", files=manifest,
            raw_KV_or_hidden_features_included=False), indent=2))
    return archive


def main(argv=None):
    p = _parser(); args = p.parse_args(argv); _defaults(args)
    if any(value < 1 for value in args.lengths + [args.train_tokens, args.dev_tokens, args.chunk_tokens, args.tile_tokens, args.max_tiles, args.rank, args.repeats, args.sample_pairs, args.threads]) or not 0 <= args.generation_steps <= 64:
        p.error("Lengths/counts must be positive; generation steps must be 0..64")
    output = (args.output or Path("qwen_producer_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True)
    state_files = {}; started = time.perf_counter(); exit_code = 1
    from .runner import _source_manifest
    report = dict(format="qwen_producer_study_v1", status="running", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, source_manifest=_source_manifest(),
        model_identity=None, calibration=None, freeze=None, cases=[], generation_checks=[], warnings=[],
        scope="Exact paired K/V reconstruction using actual forward-producer metadata. Default inline callbacks build and verify recipes inside attention forward via diagnostic CPU staging; repeated codec timings replay after capture. Full model cache remains necessary. No compact-resident attention, capacity, TTFT/TPOT, concurrency or quality evidence.")
    def save():
        checkpoint(output, report)
        markdown, machine = _summary(report)
        (output / "summary.md").write_text(markdown, encoding="utf-8")
        write_json(output / "producer_summary.json", machine)
    write_json(output / "source_manifest.json", report["source_manifest"])
    save()
    try:
        if args.producer_preset == "frontier":
            from . import frontier_codec as codec, frontier_models as model_api
        else:
            from . import producer_codec as codec, producer_models as model_api
        device = runtime.select_device(args.device)
        torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        report["environment"] = runtime.doctor(str(device))
        _note("Loading producer study model on " + str(device))
        model, tokenizer, identity = _tiny(device, args) if args.tiny else runtime.load_model(args.model, args.revision, device, args.local_files_only)
        report["model_identity"] = identity
        if args.max_tiles < model.config.num_hidden_layers:
            raise ValueError("--max-tiles must cover every model layer")
        if max(args.lengths + [args.train_tokens, args.dev_tokens]) > model.config.max_position_embeddings:
            raise ValueError("Requested producer context exceeds model configuration")
        specs = list(codec.recipes()); report["candidates"] = specs
        training = []; metadata = []
        for fixture in TRAIN:
            _note("Producer TRAIN: " + fixture["name"])
            captured, records, _ = _capture(model, tokenizer, fixture, args.train_tokens, device, args, args.max_tiles)
            training.append(records)
            metadata.append(dict(name=fixture["name"], split="train", input_ids_sha256=captured["input_ids_sha256"], producer_audit=captured["producer_audit"]))
        _note("Fitting producer predictors from TRAIN only")
        models = model_api.fit(training, output / "calibration", rank=args.rank,
            training_metadata=metadata, model_identity=identity, seed=args.seed)
        report["calibration"] = {}
        for name, value in models.items():
            path = Path(value.path)
            raw = path.read_bytes()
            if len(raw) != value.state_bytes or hashlib.sha256(raw).hexdigest() != value.model_id:
                raise AssertionError("Producer fitted model file integrity differs")
            relative = str(path.relative_to(output))
            state_files[relative] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            report["calibration"][name] = dict(file=relative, model_id=value.model_id,
                serialized_bytes=value.state_bytes, metadata=value.metadata)
        del training, records, captured
        _note("Producer DEV: complete coverage for continuation")
        inline = None if args.skip_inline else _Inline(specs, models, device, codec)
        captured, records, context = _capture(model, tokenizer, DEV[0], args.dev_tokens, device, args, None, inline)
        del inline
        dev = dict(name=DEV[0]["name"], split="dev", tokens=args.dev_tokens, fixture=DEV[0],
            capture={k: v for k, v in captured.items() if k != "matrices"}, producer_context=_context_info(context), results=[])
        report["cases"].append(dev)
        def diagnose(case, records):
            if args.producer_preset == "frontier":
                from .frontier_diagnostics import analyze
                started_diagnostics = time.perf_counter()
                case["frontier_diagnostics"] = analyze(records)
                case["frontier_diagnostics_cpu_seconds"] = time.perf_counter() - started_diagnostics
        diagnose(dev, records)
        def measure_case(case, captured, records, context):
            for spec in specs:
                _note(case["split"].upper() + " producer recipe: " + spec["name"])
                row = dict(name=spec["name"], track="producer_pair", spec=spec)
                try:
                    row.update(status="ok", measurement=_measure(records, captured, spec, models, context, device, args, output, state_files, codec))
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    _note(row["error"])
                case["results"].append(row); save()
        measure_case(dev, captured, records, context)
        report["generation_checks"] = _generation(model, captured, records, specs, models, context, device, args, output, state_files, codec)
        del captured, records, context
        frozen = dict(model_identity=identity, source_sha256=report["source_manifest"]["sources_sha256"],
            calibration=report["calibration"], decoder_artifacts=state_files.copy(), candidates=specs,
            configuration=report["config"], selection_source="TRAIN-only fits; fixed recipes and DEV representatives before fresh TEST captures")
        report["freeze"] = dict(manifest=frozen, sha256=_digest(frozen))
        write_json(output / "frozen_manifest.json", report["freeze"]); save()
        for length in args.lengths:
            for fixture in TEST[:args.cases]:
                case = dict(name=f"{fixture['name']}_{length}", split="test", tokens=length, fixture=fixture, results=[])
                report["cases"].append(case)
                try:
                    _note(f"Producer TEST: {fixture['name']}, {length} tokens")
                    inline = None if args.skip_inline else _Inline(specs, models, device, codec)
                    captured, records, context = _capture(model, tokenizer, fixture, length, device, args, args.max_tiles, inline)
                    del inline
                    case["capture"] = {k: v for k, v in captured.items() if k != "matrices"}
                    case["producer_context"] = _context_info(context)
                    diagnose(case, records)
                    measure_case(case, captured, records, context)
                    case["status"] = "completed"; del captured, records, context
                except Exception as error:
                    case.update(status="error", error=f"{type(error).__name__}: {error}")
                save()
        failed = any(c.get("status") == "error" or any(r["status"] != "ok" for r in c["results"]) for c in report["cases"])
        failed = failed or any(r["status"] == "error" for r in report["generation_checks"])
        report["status"] = "failed" if failed else "completed"
        exit_code = 1 if failed else 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"; exit_code = 130
    except Exception as error:
        report.update(status="failed", fatal_error=f"{type(error).__name__}: {error}", fatal_traceback=traceback.format_exc(limit=8))
        _note(report["fatal_error"])
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()
        archive = _bundle(output, state_files)
        _note("Producer report: " + str(output / "summary.md"))
        _note("Share results bundle: " + str(archive))
        _note("Status: " + report["status"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
