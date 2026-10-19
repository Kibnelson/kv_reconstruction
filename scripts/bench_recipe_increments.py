#!/usr/bin/env python3
"""Diagnostic incremental recipe sweep over disjoint slices of one old capture.

This is deliberately NOT an independently held-out Qwen experiment: the source
is one SmolLM request, historically a TEST fixture. We assign nonoverlapping
TRAIN/DEV/TEST token ranges only for this diagnostic. All candidate settings
are fixed before fitting; the TEST slice never trains dictionaries. No model
forward pass, live prefill hook, attention integration or concurrency is run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from qwen_kv_bench.calibration import train_calibration, _cache_mapping, _case_identity
from qwen_kv_bench.measurements import measure_native
from qwen_kv_bench.recipe_analysis import analyze_case
from qwen_kv_bench.recipe_sweep import insight_options, train_recipe_artifacts
from qwen_kv_bench.reporting import checkpoint, write_json
from qwen_kv_bench.runner import _native_state, _source_manifest
from qwen_kv_bench import residual_dictionary

_DEFAULT_CAPTURE = Path("/workspace/scratch/724d38606f5c/kv_research/results/sample_raw.npz")
_RANGES = (("train_0", "train", 0, 64), ("train_1", "train", 64, 128),
           ("train_2", "train", 128, 192), ("dev", "dev", 192, 256),
           ("test", "test", 256, 512))
_SCOPE = (
    "Same-capture diagnostic only: all slices originate in one previously captured SmolLM request, "
    "historically labeled TEST. Token ranges are disjoint, but share a causal context and are not "
    "independent prompts or held-out model validation. Synthetic TRAIN/DEV/TEST labels apply only "
    "to this diagnostic. Only the three TRAIN ranges fit fresh dictionaries; DEV is reserved and "
    "unused; all selected candidates/settings are frozen before fitting, with TEST used only for exact "
    "record reconstruction and size analysis. Native tensor execution on CPU establishes functionality "
    "and measured byte sizes, never GPU throughput. No fresh forward pass, live-prefill timing, "
    "attention, concurrency, capacity gain or full cache container is measured. Raw KV is never saved "
    "in the output. Stateful recipes charge their full decoder state once per candidate cache. "
    "The portfolio analysis is a size-only oracle over already encoded stateless packets; it is not "
    "an implemented online prefill selector.")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest_json(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _capture(path, manifest_path=None):
    with np.load(path, allow_pickle=False) as archive:
        matrices = _cache_mapping({key: archive[key].copy() for key in sorted(archive.files)})
    identity = _case_identity(matrices, dict(split="train"), 0)
    if identity["tokens"] != 512:
        raise ValueError("this fixed diagnostic requires exactly 512 source tokens")
    raw_sha = identity["canonical_raw_sha256"]
    provenance = dict(file=str(path.resolve()), file_sha256=_sha(path.read_bytes()),
        canonical_raw_sha256=raw_sha, tokens=512, layers=identity["layers"],
        channels=identity["columns"], raw_payload_bytes=identity["raw_payload_bytes"],
        canonical_order=identity["matrix_keys"], capture_manifest_verified=False)
    manifest_path = manifest_path or path.parent / "test_capture_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        matches = [row for row in manifest.get("cases", []) if row.get("raw_sha256") == raw_sha]
        if len(matches) != 1:
            raise ValueError("source raw SHA256 does not uniquely match its capture manifest")
        case = matches[0]
        if case.get("tokens") != 512 or case.get("raw_payload_bytes") != identity["raw_payload_bytes"]:
            raise ValueError("capture manifest geometry/byte count mismatch")
        provenance.update(model=manifest["model"], revision=manifest["revision"],
            original_source_split=manifest.get("split"), original_case=case["name"],
            capture_runtime=manifest.get("runtime"), capture_manifest_verified=True,
            capture_manifest_file=manifest_path.name, capture_manifest_sha256=_sha(manifest_path.read_bytes()))
    return matrices, provenance


def _state(artifacts, spec, device):
    if spec["method"] == "prefill_dictionary":
        return residual_dictionary.make_state(artifacts.models[spec["recipe_model"]], device)
    return _native_state(artifacts, spec, device)


def _summary(report):
    lines = ["# Prefill recipe increments: same-capture diagnostic", "", _SCOPE, "",
        f"Status: **{report['status']}**. Device: `{report['environment']['selected_device']}`. "
        f"Source SHA256: `{report['provenance']['canonical_raw_sha256']}`.", "",
        "TRAIN slices: `[0,64)`, `[64,128)`, `[128,192)`. Reserved DEV: `[192,256)`. "
        "Measured TEST: `[256,512)`. Ranges denote source token positions; TEST record offsets are local to its slice.", ""]
    for case in report["cases"]:
        lines += [f"## {case['tile_tokens']}-token tiles", "",
            "Savings include actual packet fields, JSON headers, calculated outer framing/index bytes, "
            "and complete state once. Full-cache savings are extrapolated when coverage is sampled.", "",
            "| Representation | Full-cache savings | Coverage | Decode ms/MiB | Exact |",
            "|---|---:|---|---:|---|"]
        for row in case["results"]:
            if row["status"] != "ok":
                lines.append(f"| {row['name']} | error | — | — | failed |")
                continue
            m = row["measurement"]
            saving = (m["estimated_full_savings_including_state_percent"] if m["size_is_estimated"]
                      else m["savings_including_state_percent"])
            decode = 1000 * m["decode_seconds_total"] / (m["raw_payload_bytes"] / (1 << 20))
            coverage = f"{m['records_measured']}/{m['records_total']}"
            lines.append(f"| {row['name']} | {saving:.3f}% | {coverage} | {decode:.3f} | yes |")
        analysis = case.get("recipe_analysis", {})
        oracle = analysis.get("stateless_size_oracle", {})
        if oracle.get("status") == "ok":
            lines += ["", f"Size-only stateless portfolio oracle: **{oracle['full_cache_savings_percent']:.3f}%** "
                f"full-cache savings; {oracle['full_cache_savings_gain_over_best_fixed_percentage_points']:.3f} "
                "percentage points over the best fixed stateless candidate after its extra mode-byte charge. "
                "No selector implementation or selection latency is measured.", ""]
        comparisons = sorted(analysis.get("comparisons", []), key=lambda x: str(x.get("candidate", "")))
        if comparisons:
            lines += [f"Matched incremental comparisons: {len(comparisons)}; see `recipe_analysis.json` "
                "for paired record identities, component byte deltas, K/V breakdowns and state charges.", ""]
    if report.get("error"):
        lines += ["## Failure", "", report["error"], ""]
    lines += ["CPU decode timings are reference implementation costs and do not predict optimized CUDA kernel speed.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=_DEFAULT_CAPTURE)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-tokens", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--max-records", type=int, default=120,
        help="matched uniformly sampled records per method; use --full-records for all")
    parser.add_argument("--full-records", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--calibration-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--io-mib", type=float, default=0,
        help="bounded packet timing sample budget; zero avoids repeated transfers")
    parser.add_argument("--methods", nargs="+", help="optional explicit candidate names; default all 33 insights")
    args = parser.parse_args()
    if (not args.tile_tokens or min(args.tile_tokens) < 1 or args.max_records < 1
            or args.calibration_samples < 64 or args.calibration_iterations < 1
            or args.threads < 1 or args.repeats < 1 or args.io_mib < 0 or args.seed < 0):
        parser.error("positive tile/record/iteration/thread counts and >=64 fitting samples are required")
    if len(set(args.tile_tokens)) != len(args.tile_tokens):
        parser.error("duplicate tile sizes are not supported")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but no actual CUDA device is available")
    if not args.input_npz.is_file():
        parser.error("source capture file does not exist")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be a fresh directory")
    torch.set_num_threads(args.threads)
    specs = insight_options()
    if args.methods:
        known = {spec["name"]: spec for spec in specs}
        if len(set(args.methods)) != len(args.methods) or any(name not in known for name in args.methods):
            parser.error("unknown or duplicate method name")
        specs = [known[name] for name in args.methods]
    matrices, provenance = _capture(args.input_npz, args.capture_manifest)
    slices = {name: {key: value[start:end].copy() for key, value in matrices.items()}
              for name, _, start, end in _RANGES}
    slice_info = []
    for name, split, start, end in _RANGES:
        info = _case_identity(slices[name], dict(split="train", name=name), 0)
        slice_info.append(dict(name=name, diagnostic_split=split, source_token_range=[start, end],
            tokens=end - start, canonical_raw_sha256=info["canonical_raw_sha256"],
            identity_sha256=info["identity_sha256"], raw_payload_bytes=info["raw_payload_bytes"]))
    source = _source_manifest()
    script_sha = _sha(Path(__file__).read_bytes())
    configuration = dict(tile_tokens=args.tile_tokens, max_records=None if args.full_records else args.max_records,
        calibration_samples=args.calibration_samples, calibration_iterations=args.calibration_iterations,
        seed=args.seed, repeats=args.repeats, io_mib=args.io_mib, threads=args.threads,
        candidate_count=len(specs), source_ranges=slice_info)
    frozen = dict(configuration=configuration, candidates=specs,
        source_capture_sha256=provenance["canonical_raw_sha256"], source_manifest_sha256=source["sources_sha256"],
        diagnostic_script_sha256=script_sha, selection="None; every candidate is predetermined before TRAIN fitting")
    report = dict(format="prefill_recipe_increment_same_capture_diagnostic_v1", status="running",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), scope=_SCOPE, provenance=provenance,
        source_manifest=source, diagnostic_script_sha256=script_sha,
        model_identity=dict(model_id=provenance.get("model", "unspecified-existing-capture"),
            revision=provenance.get("revision"), config=dict(num_hidden_layers=provenance["layers"]),
            diagnostic_only=True, original_capture_sha256=provenance["canonical_raw_sha256"]),
        environment=dict(selected_device=args.device, python=platform.python_version(), platform=platform.platform(),
            torch=torch.__version__, numpy=np.__version__, transformers=_version("transformers"),
            threads=args.threads, cuda_runtime=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0) if args.device == "cuda" else None),
        config=configuration, freeze=dict(sha256=_digest_json(frozen), manifest=frozen),
        cases=[], generation_checks=[], calibration_by_tile={},
        warnings=["Same source prompt and causal context across all slices; not an independent held-out benchmark.",
                  "Historical TEST capture is deliberately reused for exploratory TRAIN slices; do not reuse this diagnostic as validation evidence.",
                  "No generation continuation or live prefill hook is tested by this script."])
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "source_manifest.json", source)
    write_json(args.output / "frozen_manifest.json", report["freeze"])
    started = time.perf_counter()
    failure = False
    try:
        train = [slices[f"train_{i}"] for i in range(3)]
        training_metadata = [dict(name=f"same_capture_train_{i}", split="train",
            diagnostic_same_capture=True, source_token_range=[64 * i, 64 * (i + 1)],
            original_capture_sha256=provenance["canonical_raw_sha256"],
            original_source_split=provenance.get("original_source_split", "unspecified")) for i in range(3)]
        needs_base_dictionary = any(spec["method"] in ("learned", "learned_e8", "binary") for spec in specs)
        requirements = dict(magnitude_sizes=[64] if needs_base_dictionary else [], fixed_sizes=[],
            lower_sizes=[], conditional_lower_sizes=[])
        for tile in args.tile_tokens:
            directory = args.output / f"tile_{tile}"
            print(f"Tile {tile}: fitting fresh dictionaries on three disjoint TRAIN ranges", flush=True)
            artifacts = train_calibration(train, directory / "calibration", requirements,
                seed=args.seed, max_samples_per_key=args.calibration_samples,
                iterations=args.calibration_iterations, training_metadata=training_metadata,
                model_identity=report["model_identity"])
            artifacts = train_recipe_artifacts(artifacts, specs, train, directory / "calibration",
                tile_tokens=tile, seed=args.seed, max_samples_per_key=args.calibration_samples,
                iterations=args.calibration_iterations, training_metadata=training_metadata,
                model_identity=report["model_identity"])
            report["calibration_by_tile"][str(tile)] = artifacts.metadata
            case = dict(name=f"same_capture_test_tokens256_512_tile{tile}", split="test", tokens=256,
                tile_tokens=tile, fixture=dict(name="same_capture_tail", split="test", kind="diagnostic"),
                diagnostic_same_capture=True, source_token_range=[256, 512], results=[])
            report["cases"].append(case)
            for spec in specs:
                row = dict(name=spec["name"], spec=spec, track="native_packet", status="running")
                case["results"].append(row)
                try:
                    state = _state(artifacts, spec, args.device)
                    measurement = measure_native(slices["test"], spec, state, args.device, directory,
                        repeats=args.repeats, io_bytes_limit=int(args.io_mib * (1 << 20)),
                        tile_tokens=tile, max_records=configuration["max_records"], record_diagnostics=True)
                    if not measurement["roundtrip_exact"] or not measurement["exact_verified"]:
                        raise AssertionError("missing exact reconstruction verification")
                    row.update(status="ok", measurement=measurement)
                    saving = (measurement["estimated_full_savings_including_state_percent"] if measurement["size_is_estimated"]
                              else measurement["savings_including_state_percent"])
                    print(f"  {spec['name']}: {saving:.3f}% savings, exact {measurement['records_checked']} records", flush=True)
                    del state
                except Exception as error:
                    failure = True
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    print(f"  {spec['name']}: ERROR {row['error']}", flush=True)
                write_json(args.output / "results.json", report)
            case["recipe_analysis"] = analyze_case(case)
            checkpoint(args.output, report)
        report["status"] = "failed" if failure else "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        failure = True
    report["elapsed_seconds"] = time.perf_counter() - started
    checkpoint(args.output, report)
    write_json(args.output / "recipe_analysis.json", dict(scope=_SCOPE,
        source_sha256=provenance["canonical_raw_sha256"],
        cases=[case.get("recipe_analysis", {}) for case in report["cases"]]))
    (args.output / "summary.md").write_text(_summary(report), encoding="utf-8")
    print(f"{report['status']}: {args.output / 'summary.md'}", flush=True)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
