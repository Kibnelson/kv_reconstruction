"""Development-only frozen grid for learned residual E8/scalar predictors.

Reads only the named development capture.  It never reads final evaluation
captures.  Run from the package parent with task dependencies on PYTHONPATH.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from kv_research.learned_codec import load_model
from kv_residual.learned_e8 import encode_matrix, decode_matrix


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", default=".kv_research_data/dev_sensor_pipeline_1024.npz")
    parser.add_argument("--model", default="kv_research/models/learned64.bin")
    parser.add_argument("--output", default="kv_residual/results/dev_e8_sweep.json")
    parser.add_argument("--selection", default="kv_residual/e8_selection.json")
    args = parser.parse_args()
    model = load_model(args.model)
    capture = np.load(args.development)
    keys = [f"layer_{layer:02d}_{kind}" for layer in (0, 15, 29) for kind in ("K", "V")]
    common = dict(group_size=3072, force=True, assignment="radius")
    results = []

    def evaluate(family, step, correction):
        total = raw = records = 0
        components, mechanisms, correction_modes = Counter(), Counter(), Counter()
        elapsed = time.perf_counter()
        for key in keys:
            for offset in range(0, 256, 16):
                words = capture[key][offset:offset + 16]
                encoded = encode_matrix(words, model, key, step_log2=step,
                    family=family, correction=correction, **common)
                recovered = decode_matrix(encoded, model)
                if not np.array_equal(words, recovered):
                    raise AssertionError((family, step, correction, key, offset))
                total += len(encoded.data)
                raw += words.nbytes
                records += 1
                components.update(encoded.stats["component_bytes"])
                mechanisms.update(encoded.stats["mechanism_counts"])
                correction_modes.update(encoded.stats["accepted_correction_mode_histogram"])
        assert sum(components.values()) == total
        assert records == 96 and raw == 589824
        result = dict(family=family, step_log2=step, correction=correction, **common,
            raw_payload_bytes=raw, serialized_matrix_bytes=total,
            space_saving_percent=100 * (1 - total / raw), exact_roundtrip=True,
            matrix_records=records, component_bytes=dict(components),
            mechanism_counts=dict(mechanisms), correction_modes=dict(correction_modes),
            cpu_encode_decode_seconds=time.perf_counter() - elapsed)
        results.append(result)
        print(json.dumps({key: result[key] for key in ("family", "step_log2", "correction",
            "serialized_matrix_bytes", "space_saving_percent", "exact_roundtrip")} ), flush=True)
        return result

    for family in ("e8", "scalar"):
        for step in (4, 7, 10):
            evaluate(family, step, "best")
    best_e8 = min((r for r in results if r["family"] == "e8"), key=lambda r: r["serialized_matrix_bytes"])
    best_scalar = min((r for r in results if r["family"] == "scalar"), key=lambda r: r["serialized_matrix_bytes"])
    evaluate("e8", best_e8["step_log2"], "coset")
    provenance = dict(development_file=str(args.development), development_sha256=sha256(args.development),
        model_file=str(args.model), model_sha256=sha256(args.model), model_id=model.model_id,
        source_sha256={str(path): sha256(path) for path in (Path(__file__),
            Path("kv_residual/learned_e8.py"), Path("kv_residual/entropy.py"))})
    report = dict(phase="development only", protocol=dict(keys=keys, first_tokens=256,
        tile_tokens=16, step_grid=[4, 7, 10], **common), provenance=provenance, results=results,
        size_contract="Sum of independently serialized matrices, including all metadata and CRCs; outer cache index and shared model counted separately in final evaluation.",
        timing_scope="CPU Python encode+decode correctness timings; not GPU/serving latency.")
    selection = dict(phase="frozen from development only", model_id=model.model_id,
        common=dict(tile_tokens=16, **common),
        e8=dict(family="e8", step_log2=best_e8["step_log2"], correction="best"),
        scalar=dict(family="scalar", step_log2=best_scalar["step_log2"], correction="best"),
        e8_coset=dict(family="e8", step_log2=best_e8["step_log2"], correction="coset"),
        development_report=str(args.output), development_sha256=provenance["development_sha256"],
        selection_rule="Minimum actual serialized matrix bytes in the fixed development grid.")
    for destination, content in ((args.output, report), (args.selection, selection)):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
