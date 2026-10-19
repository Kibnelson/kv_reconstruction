#!/usr/bin/env python3
"""Diagnostic recipe sweep over an existing BF16 uint16 KV capture.

This reuses a captured sample; it is not a fresh held-out model experiment or
an inline-prefill measurement. The normal Qwen runner provides live observation
with --prefill-recipes. No fitted dictionary is loaded by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from qwen_kv_bench.codec_registry import build_registry
from qwen_kv_bench.gpu_ops import make_state, native_options
from qwen_kv_bench.measurements import measure_native, measure_reference
from qwen_kv_bench.reporting import checkpoint
from qwen_kv_bench.runner import _source_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npz", type=Path, required=True)
    p.add_argument("--capture-manifest", type=Path)
    p.add_argument("--case-name", default="existing_KV_sample")
    p.add_argument("--tile-tokens", nargs="+", type=int, default=[16, 64])
    p.add_argument("--max-records", type=int, default=120)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.tile_tokens) < 1 or args.max_records < 1:
        p.error("tile and record counts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("output must be a fresh directory")
    torch.set_num_threads(4)
    with np.load(args.input_npz, allow_pickle=False) as archive:
        matrices = {key: archive[key].copy() for key in sorted(archive.files)}
    raw = b"".join(words.astype("<u2").tobytes() for words in matrices.values())
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    provenance = dict(file=args.input_npz.name, file_sha256=hashlib.sha256(args.input_npz.read_bytes()).hexdigest(),
                      canonical_raw_sha256=raw_sha256)
    if args.capture_manifest:
        manifest = json.loads(args.capture_manifest.read_text())
        row = next(c for c in manifest["cases"] if c["name"] == args.case_name)
        if row["raw_sha256"] != raw_sha256:
            raise ValueError("capture canonical raw hash mismatch")
        provenance.update(model=manifest["model"], revision=manifest["revision"], case=row["name"],
                          capture_runtime=manifest.get("runtime"), capture_hash_verified=True)
    report = dict(format="prefill_sample_diagnostic_v1", status="running",
        model_identity={"model_id": provenance.get("model", "existing sample, model identity unspecified")},
        environment={"selected_device": "cpu", "torch": torch.__version__, "numpy": np.__version__, "threads": 4},
        source_manifest=_source_manifest(), provenance=provenance, cases=[], generation_checks=[], warnings=[],
        scope="Reused existing sample, post-capture CPU tensor functionality and measured record-size diagnostic only. No new model run, held-out evaluation, GPU timing, or live-prefill timing. All methods/settings fixed before this diagnostic; no selection from these rows is a held-out winner.")
    adapters = build_registry(names=["raw", "exponent_huffman_lower_deflate"], workspace=args.output)
    specs = [s for s in native_options() if s["method"] in ("raw", "exponent", "prefill")]
    for tile in args.tile_tokens:
        case = dict(name=f"{args.case_name}_tile{tile}", split="diagnostic_reused_sample",
                    tokens=next(iter(matrices.values())).shape[0], tile_tokens=tile, results=[])
        report["cases"].append(case)
        for name, adapter in adapters.items():
            m = measure_reference(matrices, adapter, "cpu", args.output, repeats=1,
                tile_tokens=tile, max_records=args.max_records, io_bytes_limit=0)
            case["results"].append(dict(name=name, track="cpu_reference", status="ok", measurement=m))
        for spec in specs:
            m = measure_native(matrices, spec, make_state(), "cpu", args.output, repeats=1,
                tile_tokens=tile, max_records=args.max_records, io_bytes_limit=0)
            case["results"].append(dict(name=spec["name"], track="native_packet", status="ok", spec=spec, measurement=m))
            print(f"tile {tile}: {spec['name']}: {m['estimated_full_savings_including_state_percent'] if m['size_is_estimated'] else m['savings_including_state_percent']:.3f}%", flush=True)
        checkpoint(args.output, report)
    report["status"] = "completed"
    checkpoint(args.output, report)
    print(args.output / "summary.md")


if __name__ == "__main__":
    main()
