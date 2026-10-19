"""CLI orchestration for the self-contained, exact BF16 KV experiment."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--doctor", action="store_true", help="Probe hardware and dependencies without downloading a model")
    p.add_argument("--self-test", action="store_true", help="Run synthetic CPU correctness tests without downloading weights")
    p.add_argument("--list-methods", action="store_true")
    p.add_argument("--device", default="auto", help="auto chooses CUDA then Intel XPU; CPU must be explicit; cuda:1 etc supported")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None, help="Default model uses pinned commit; custom model defaults to its resolved main commit")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--preset", choices=("smoke", "main", "all"), default="all")
    p.add_argument("--grid", action="store_true", help="Expand CPU all preset from 68 to 104 parameter/control options")
    p.add_argument("--native-preset", choices=("main", "all"), default=None)
    p.add_argument("--skip-native", action="store_true")
    p.add_argument("--methods", nargs="+", help="Optional exact CPU method names from --list-methods")
    p.add_argument("--native-methods", nargs="+", help="Optional exact native method names from --list-methods")
    p.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192])
    p.add_argument("--cases", type=int, choices=(1, 2, 3), default=3, help="Fresh test fixtures per context length")
    p.add_argument("--train-tokens", type=int, default=512)
    p.add_argument("--dev-tokens", type=int, default=512)
    p.add_argument("--chunk-tokens", type=int, default=512, help="Prefill chunk size; full prompt KV still resides on device")
    p.add_argument("--tile-tokens", type=int, default=16, help="Independently decodable record length")
    p.add_argument("--max-records", type=int, default=72, help="Evenly spaced record sample per codec/case; full-cache sizes are estimates")
    p.add_argument("--full-cache", action="store_true", help="Encode and verify every record; substantially slower")
    p.add_argument("--dev-records", type=int, default=8, help="Frozen development selection sample")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--measure-io", action="store_true", help="Also benchmark buffered temporary-file write/fsync/read; reads may hit page cache")
    p.add_argument("--io-mib", type=float, default=32, help="Maximum timing sample encoded+raw bytes per method")
    p.add_argument("--generation-steps", type=int, default=2, help="Full development-prefix reconstruction check for one winner per track; 0 disables")
    p.add_argument("--calibration-samples", type=int, default=1024)
    p.add_argument("--calibration-iterations", type=int, default=8)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--threads", type=int, default=4, help="Torch CPU threads, also recorded for fair CPU timings")
    p.add_argument("--memory-reserve-gib", type=float, default=2.0, help="Conservative accelerator workspace reserve; not a measured peak bound")
    p.add_argument("--output", type=Path, default=None, help="Fresh results directory; a shareable ZIP is written beside it")
    return p


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def note(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _release(device):
    import torch
    gc.collect()
    if device.type in ("cuda", "xpu"):
        getattr(torch, device.type).empty_cache()


def _capacity(device, args, tokens, weights_bytes=8_044_936_192, copies=1):
    from .model_runtime import memory_snapshot
    m = memory_snapshot(device)
    free = m.get("free_device_bytes")
    reclaim = max(0, (m.get("reserved_bytes") or 0) - (m.get("allocated_bytes") or 0))
    kv = int(tokens) * 147456 * copies if args.model == DEFAULT_MODEL else None
    need = None if kv is None else int(weights_bytes + kv + args.memory_reserve_gib * 2**30)
    return dict(tokens=tokens, raw_KV_bytes=kv, full_cache_copies=copies,
                required_bytes_estimate=need, available_plus_reclaimable_bytes=None if free is None else free + reclaim,
                can_attempt=None if need is None or free is None else free + reclaim >= need,
                scope="Conservative BF16 Qwen estimate plus requested reserve; not a guarantee against attention/allocator OOM. Custom-model KV cost is unknown before capture.")


def _source_manifest():
    root = Path(__file__).resolve().parent.parent
    names = ("qwen_kv_bench", "kv_smoke", "kv_research", "kv_combinations", "kv_stacked", "kv_coset_fields", "kv_residual", "kv_binary")
    files = {}
    for name in names:
        for path in sorted((root / name).glob("*.py")):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"embedded_payload_sha256": os.environ.get("QWEN_KV_SOURCE_SHA256"), "files": files, "sources_sha256": digest_json(files)}


def _capture_fixture(model, tokenizer, fixture, tokens, device, args):
    from .fixtures import build_input_ids
    from .model_runtime import capture
    ids = build_input_ids(tokenizer, fixture, tokens)
    result = capture(model, ids, device, args.chunk_tokens)
    result["input_ids_sha256"] = digest_json(ids)
    result["fixture_name"] = fixture["name"]
    return result


def _capture_metadata(capture):
    return {k: v for k, v in capture.items() if k != "matrices"}


def _native_state(artifacts, spec, device):
    from .gpu_ops import make_state
    model = artifacts.models.get("learned64") if spec["method"] in ("learned", "learned_e8", "binary") else None
    rotations = None
    if spec["method"] == "binary":
        # Use the identical explicit NumPy reference matrix: changing the
        # entropy packet should not also change the binary predictor basis.
        from kv_binary.binary_codec import make_state as make_reference_state
        o = spec["options"]
        seed = o.get("seed", 104729)
        reference = make_reference_state(model, o["dimension"], o["transform"], seed)
        rotations = {(o["dimension"], o["transform"], seed): reference.rotation.copy()}
    state = make_state(model.codebooks if model else None, device=device,
                       state_id=model.model_id if model else "no_learned_state", rotations=rotations)
    return state


def _status(error):
    text = str(error).lower()
    if isinstance(error, (AssertionError, ValueError)) and any(s in text for s in ("differ", "reconstruct", "changed", "mismatch")):
        return "failed_correctness"
    if isinstance(error, NotImplementedError) or any(s in text for s in ("not implemented", "not supported", "unsupported", "float64")):
        return "unsupported"
    return "error"


def _measure(matrices, adapters, native_specs, artifacts, device, args, output, max_records, on_result):
    from .measurements import measure_reference, measure_native
    kwargs = dict(repeats=args.repeats, io_bytes_limit=int(args.io_mib * 2**20),
                  tile_tokens=args.tile_tokens, max_records=max_records, measure_io=args.measure_io)
    for name, adapter in adapters.items():
        note(f"CPU reference: {name}")
        row = {"name": name, "track": "cpu_reference"}
        try:
            row.update(status="ok", measurement=measure_reference(matrices, adapter, device, output, **kwargs))
        except Exception as error:
            row.update(status=_status(error), error=f"{type(error).__name__}: {error}")
            note(row["status"] + ": " + row["error"])
        on_result(row)
        _release(device)
    for spec in native_specs:
        note(f"Tensor native on {device}: {spec['name']}")
        row = {"name": spec["name"], "track": "native_packet", "spec": spec}
        state = None
        try:
            state = _native_state(artifacts, spec, device)
            row.update(status="ok", measurement=measure_native(matrices, spec, state, device, output, **kwargs))
        except Exception as error:
            row.update(status=_status(error), error=f"{type(error).__name__}: {error}")
            note(row["status"] + ": " + row["error"])
        on_result(row)
        del state
        _release(device)


def _reference_callback(matrices, adapter, tile):
    import numpy as np
    from .measurements import _encode
    packets = []
    for layer, pair in enumerate(matrices):
        records = []
        for kind, words in zip(("K", "V"), pair):
            records.append([_encode(adapter, words[start:start + tile], layer, kind).data
                            for start in range(0, len(words), tile)])
        packets.append(records)
    def decode():
        return [tuple(np.concatenate([adapter.decode(blob, layer, kind) for blob in records])
                      for kind, records in zip(("K", "V"), pair)) for layer, pair in enumerate(packets)]
    return decode


def _native_callback(matrices, spec, state, device, tile):
    import numpy as np
    import torch
    from . import gpu_ops as g
    packets = []
    for layer, pair in enumerate(matrices):
        records = []
        for kind, words in zip(("K", "V"), pair):
            records.append([g.packet_to_cpu(g.encode(torch.from_numpy(words[start:start + tile]).to(device),
                            state, method=spec["method"], key=f"layer_{layer:02d}_{kind}", **spec["options"]))
                            for start in range(0, len(words), tile)])
        packets.append(records)
    def decode():
        result = []
        for pair in packets:
            restored = []
            for records in pair:
                restored.append(np.concatenate([g.decode(g.packet_to_device(p, device), state).cpu().numpy() for p in records]))
            result.append(tuple(restored))
        return result
    return decode


def _generation(model, dev, dev_rows, adapters, native_specs, artifacts, device, args, report, save):
    from .model_runtime import generation_check
    if args.generation_steps == 0:
        return
    check_capacity = _capacity(device, args, args.dev_tokens, weights_bytes=0, copies=2)
    if check_capacity["can_attempt"] is False:
        report["warnings"].append("Development generation correctness comparison skipped: two full caches plus reserve do not fit the preflight estimate.")
        save()
        return
    def score(row):
        m = row["measurement"]
        return m.get("estimated_full_savings_including_state_percent") if m.get("size_is_estimated") else m.get("savings_including_state_percent", -1e30)
    for track in ("cpu_reference", "native_packet"):
        candidates = [r for r in dev_rows if r["track"] == track and r["status"] == "ok"]
        if not candidates:
            continue
        chosen = max(candidates, key=score)
        row = dict(name=chosen["name"], track=track, context_tokens=args.dev_tokens,
                   selection="Highest development estimated/full saving including full state; no test-set tuning")
        state = callback = None
        try:
            note(f"Generation correctness: {row['name']} on complete {args.dev_tokens}-token DEV prefix")
            if track == "cpu_reference":
                callback = _reference_callback(dev["matrices"], adapters[row["name"]], args.tile_tokens)
            else:
                spec = next(s for s in native_specs if s["name"] == row["name"])
                state = _native_state(artifacts, spec, device)
                callback = _native_callback(dev["matrices"], spec, state, device, args.tile_tokens)
            measured = generation_check(model, dev["matrices"], callback, device,
                          first_token=dev["first_token"], steps=args.generation_steps)
            ok = (measured["all_tokens_equal"] and measured["all_logits_bitwise_equal"]
                  and measured["all_appended_KV_bitwise_equal"]
                  and all(step["logits_finite"] for step in measured["per_step"]))
            row.update(status="ok" if ok else "failed_correctness", measurement=measured)
        except Exception as error:
            row.update(status=_status(error), error=f"{type(error).__name__}: {error}")
        report["generation_checks"].append(row)
        save()
        del callback, state
        _release(device)


def _validate(args, p):
    for key in ("train_tokens", "dev_tokens", "chunk_tokens", "tile_tokens", "max_records", "dev_records", "repeats", "calibration_samples", "calibration_iterations", "threads"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if not args.lengths or min(args.lengths) < 1 or not 0 <= args.generation_steps <= 64 or args.io_mib < 0 or args.memory_reserve_gib < 0:
        p.error("Invalid context lengths, generation steps, I/O budget, or reserve")
    if args.grid and args.preset != "all":
        p.error("--grid requires --preset all")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    _validate(args, p)
    if args.revision is None and args.model == DEFAULT_MODEL:
        args.revision = DEFAULT_REVISION
    from . import model_runtime as runtime
    if args.doctor:
        d = runtime.doctor(args.device)
        d["qwen_BF16_preflight"] = None
        if d.get("selected_device"):
            d["qwen_BF16_preflight"] = _capacity(runtime.select_device(args.device), args, max(args.train_tokens, args.dev_tokens, min(args.lengths)))
        print(json.dumps(d, indent=2))
        return 0 if d.get("selected_device") and d.get("bf16_probe", {}).get("success") else 2
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py", top_level_dir=str(Path(__file__).parent))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    from .codec_registry import candidate_specs, required_artifacts, build_registry
    from .gpu_ops import native_options
    specs = candidate_specs(args.preset, args.grid)
    if args.methods:
        unknown = set(args.methods) - set(specs)
        if unknown:
            p.error(f"Unknown or unselected CPU method(s): {sorted(unknown)}; use --preset all [--grid] --list-methods")
        specs = {name: specs[name] for name in args.methods}
    native_specs = [] if args.skip_native else native_options(args.native_preset or ("all" if args.preset == "all" else "main"))
    if args.native_methods:
        selected = {s["name"]: s for s in native_specs}
        if set(args.native_methods) - set(selected):
            p.error("Unknown native method; use --native-preset all --list-methods")
        native_specs = [selected[n] for n in args.native_methods]
    if args.list_methods:
        print(json.dumps({"cpu_reference": specs, "native_packet": native_specs}, indent=2))
        return 0
    import numpy as np
    import torch
    from .calibration import train_calibration
    from .fixtures import TRAIN, DEV, TEST
    from .reporting import checkpoint, bundle, write_json
    try:
        device = runtime.select_device(args.device)
    except (RuntimeError, ValueError) as error:
        p.error(str(error) + "; run --doctor to inspect this machine")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = (args.output or Path("qwen_kv_run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error("--output must be a fresh or empty directory; existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(format="qwen_kv_benchmark_v1", status="running", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment=runtime.doctor(args.device), source_manifest=_source_manifest(),
                  model_identity=None, calibration=None, freeze=None, cases=[], generation_checks=[], warnings=[],
                  scope="Exact BF16 cache representation and stage microbenchmarks. Original entropy codecs execute on CPU. Separately named native packet codecs use eager device tensor operations. No compact-resident attention, high-concurrency serving, admission/TPOT improvement, or physical SSD bandwidth is demonstrated.")
    save = lambda: checkpoint(output, report)
    write_json(output / "source_manifest.json", report["source_manifest"])
    guide = Path(__file__).with_name("README.md")
    if guide.is_file():
        (output / "README.md").write_bytes(guide.read_bytes())
    save()
    started = time.perf_counter()
    exit_code = 0
    try:
        preflight = _capacity(device, args, max(args.train_tokens, args.dev_tokens, min(args.lengths)))
        report["pre_download_preflight"] = preflight
        save()
        if preflight["can_attempt"] is False:
            raise RuntimeError("BF16 Qwen model + minimum context + workspace reserve exceed available accelerator memory before download. Check --doctor; use a larger supported accelerator or explicitly choose a different smaller model. Weight quantization/offload is not enabled by this experiment.")
        if not report["environment"].get("bf16_probe", {}).get("success"):
            raise RuntimeError("BF16 device probe failed before download")
        note(f"Loading {args.model}@{args.revision or 'resolved-main'} on {device}")
        model, tokenizer, identity = runtime.load_model(args.model, args.revision, device, args.local_files_only)
        identity["config"] = {k: getattr(model.config, k, None) for k in ("model_type", "num_hidden_layers", "num_key_value_heads", "num_attention_heads", "head_dim", "hidden_size", "max_position_embeddings")}
        report["model_identity"] = identity
        save()
        max_context = getattr(model.config, "max_position_embeddings", None)
        if max_context and max(args.lengths + [args.train_tokens, args.dev_tokens]) > max_context:
            raise ValueError(f"Requested context exceeds model max_position_embeddings={max_context}")
        train, train_meta = [], []
        for fixture in TRAIN:
            note("Training capture: " + fixture["name"])
            captured = _capture_fixture(model, tokenizer, fixture, args.train_tokens, device, args)
            train.append(captured["matrices"])
            train_meta.append(dict(name=fixture["name"], split="train", input_ids_sha256=captured["input_ids_sha256"], capture=_capture_metadata(captured)))
            del captured
            _release(device)
        requirements = required_artifacts(args.preset, args.grid, names=list(specs))
        if any(s["method"] in ("learned", "learned_e8", "binary") for s in native_specs):
            requirements["magnitude_sizes"] = sorted(set(requirements["magnitude_sizes"]) | {64})
        note("Fitting fresh target-model dictionaries from the three training captures only")
        artifacts = train_calibration(train, output / "calibration", requirements,
                       seed=args.seed, max_samples_per_key=args.calibration_samples,
                       iterations=args.calibration_iterations, training_metadata=train_meta, model_identity=identity)
        report["calibration"] = artifacts.metadata
        del train
        _release(device)
        adapters = build_registry(args.preset, artifacts, output, names=list(specs), grid=args.grid, expected_model_identity=identity)
        note("Development capture; choosing one representative generation check per track")
        dev = _capture_fixture(model, tokenizer, DEV[0], args.dev_tokens, device, args)
        dev_case = dict(name=DEV[0]["name"], split="dev", tokens=args.dev_tokens, fixture=DEV[0], capture=_capture_metadata(dev), results=[])
        report["cases"].append(dev_case)
        def record_dev(row):
            dev_case["results"].append(row)
            save()
        _measure(dev["matrices"], adapters, native_specs, artifacts, device, args, output, args.dev_records, record_dev)
        _generation(model, dev, dev_case["results"], adapters, native_specs, artifacts, device, args, report, save)
        del dev
        _release(device)
        frozen = dict(model_identity=identity, source_sha256=report["source_manifest"]["sources_sha256"],
                      calibration_sha256=artifacts.report_sha256, cpu_candidates=specs, native_candidates=native_specs,
                      configuration=report["config"], selection_source="Development only; all test candidates/settings fixed before test capture")
        report["freeze"] = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), sha256=digest_json(frozen), manifest=frozen)
        write_json(output / "frozen_manifest.json", report["freeze"])
        save()
        for tokens in args.lengths:
            for fixture in TEST[:args.cases]:
                case = dict(name=f"{fixture['name']}_{tokens}", split="test", tokens=tokens, fixture=fixture, results=[])
                report["cases"].append(case)
                _release(device)
                capacity = _capacity(device, args, tokens, weights_bytes=0)
                case["preflight"] = capacity
                if capacity["can_attempt"] is False:
                    case.update(status="skipped_capacity", error="Full raw capture plus reserve exceeds available device memory estimate; compact attention is not integrated")
                    save()
                    continue
                captured = None
                try:
                    note(f"TEST capture: {fixture['name']}, {tokens} tokens")
                    captured = _capture_fixture(model, tokenizer, fixture, tokens, device, args)
                    case["capture"] = _capture_metadata(captured)
                    case["status"] = "captured"
                    save()
                    def record_test(row):
                        case["results"].append(row)
                        save()
                    _measure(captured["matrices"], adapters, native_specs, artifacts, device, args, output,
                             None if args.full_cache else args.max_records, record_test)
                    case["status"] = "completed"
                except Exception as error:
                    case.update(status=_status(error), error=f"{type(error).__name__}: {error}")
                    note(case["error"])
                finally:
                    del captured
                    _release(device)
                    save()
        rows = [r for c in report["cases"] for r in c["results"]] + report["generation_checks"]
        failed = any(r.get("status") in ("error", "failed_correctness") for r in rows) or any(c.get("status") in ("error", "failed_correctness") for c in report["cases"])
        tests = [c for c in report["cases"] if c.get("split") == "test"]
        complete = bool(tests) and all(c.get("status") == "completed" for c in tests) and all(r.get("status") == "ok" for r in rows)
        report["status"] = "failed" if failed else "completed" if complete else "completed_with_skips"
        exit_code = 1 if failed else 0 if complete else 2
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        exit_code = 130
    except Exception as error:
        report.update(status="failed", fatal_error=f"{type(error).__name__}: {error}", fatal_traceback=traceback.format_exc(limit=8))
        note(report["fatal_error"])
        exit_code = 1
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()
        archive = bundle(output)
        note(f"Report: {output / 'summary.md'}")
        note(f"Share results bundle: {archive}")
        note("Status: " + report["status"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
