"""Reproduce frozen learned/fixed dictionaries using capture manifests.

Run from the project root:
    python -m kv_research.train_models --data-dir .kv_research_data

Only the three `train_manifest.json` cases enter training. The selected-layer
dev summary is descriptive and does not modify weights. No final/test capture
is loaded. NumPy is the only non-standard-library dependency of this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_research.learned_codec import (LearnedModel, train_model, save_model,
                                      encode_matrix, decode_matrix)
from kv_smoke.codec import encode_matrix as baseline_encode, decode_matrix as baseline_decode


FIXED_CONFIG = {
    "n_centroids": 64, "vector_dim": 8, "trained": False,
    "construction": "For each exponent in(-8,-6,-4,-2,0,2,4,6), take 8D BF16 magnitude center(127+exponent)*128 and add zerooffset or one of7balanced SylvesterH8rows times128. This fixes coordinate powers2^-9..2^7. Identical book for every key. No KVsourcevalues used."
}


def fixed_model(keys):
    """Source-independent 64-codeword dictionary, duplicated per format key."""
    hadamard = np.array([[1]], dtype=np.int32)
    for _ in range(3):
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    patterns = np.vstack([np.zeros(8, dtype=np.int32), hadamard[1:] * 128])
    book = np.vstack([np.full((8, 8), (127 + power) * 128, np.int32) + patterns
                      for power in range(-8, 7, 2)]).astype(np.uint16)
    book.flags.writeable = False
    return LearnedModel({key: book for key in keys}, dict(FIXED_CONFIG),
                        {"trained": False, "source_data_used": False})


def load_capture(path, case):
    with np.load(path) as archive:
        result = {key: archive[key] for key in archive.files}
    raw = b"".join(result[key].astype("<u2").tobytes() for key in sorted(result))
    if len(raw) != case["raw_payload_bytes"] or hashlib.sha256(raw).hexdigest() != case["raw_sha256"]:
        raise ValueError(f"capture does not match manifest: {path}")
    return result


def selected_dev_summary(model, dev):
    rows = []
    for layer in (0, 15, 29):
        for kind in ("K", "V"):
            key = f"layer_{layer:02d}_{kind}"
            words = dev[key][:256]
            start = time.perf_counter()
            encoded = encode_matrix(words, model, key)
            encode_seconds = time.perf_counter() - start
            start = time.perf_counter()
            recovered = decode_matrix(encoded, model)
            decode_seconds = time.perf_counter() - start
            if not np.array_equal(words, recovered):
                raise AssertionError("learned dev recovery mismatch")
            rows.append({"key": key, "tokens": len(words), "raw_bytes": words.nbytes,
                         "serialized_bytes": len(encoded.data), "encode_seconds": encode_seconds,
                         "decode_seconds": decode_seconds, "exact": True, "stats": encoded.stats})
    return rows


def ratio(rows):
    return sum(row["raw_bytes"] for row in rows) / sum(row["serialized_bytes"] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".kv_research_data")
    parser.add_argument("--output-dir", default="kv_research")
    parser.add_argument("--counts", default="16,64,256")
    parser.add_argument("--skip-dev", action="store_true")
    parser.add_argument("--verify-against", help="Existing training report: require identical model SHA256s")
    args = parser.parse_args()
    data_root, output = Path(args.data_dir), Path(args.output_dir)
    (output / "models").mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "train" or len(manifest["cases"]) != 3:
        raise ValueError("expected the three explicit training captures")
    paths = [data_root / case["file"] for case in manifest["cases"]]
    captures = [load_capture(path, case) for path, case in zip(paths, manifest["cases"])]
    training = {key: [capture[key] for capture in captures] for key in captures[0]}
    dev, dev_path = None, None
    if not args.skip_dev:
        dev_manifest = json.loads((data_root / "dev_manifest.json").read_text())
        if dev_manifest["split"] != "dev" or len(dev_manifest["cases"]) != 1:
            raise ValueError("expected one independent dev capture")
        dev_case = dev_manifest["cases"][0]
        dev_path = data_root / dev_case["file"]
        dev = load_capture(dev_path, dev_case)
    expected = json.loads(Path(args.verify_against).read_text()) if args.verify_against else None
    expected_ids = {row["config"]["n_centroids"]: row["model"]["model_id"]
                    for row in expected["configurations"]} if expected else {}
    report = {
        "scope": "Frozen codebooks trained only on3trainingprompts. Dev prefixsubset used solely for model selection; no finalfixtures used.",
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in paths],
        "dev_file": str(dev_path) if dev_path else None,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "configurations": [],
    }
    for count in map(int, args.counts.split(",")):
        config = {"n_centroids": count, "iterations": 10, "seed": 19, "max_samples_per_key": 4096}
        start = time.perf_counter()
        model = train_model(training, config)
        seconds = time.perf_counter() - start
        path = output / "models" / f"learned{count}.bin"
        saved = save_model(model, path)
        if count in expected_ids and saved["model_id"] != expected_ids[count]:
            raise AssertionError(f"reproduced learned{count} model SHA256 differs")
        row = {"config": config, "training_seconds": seconds, "model": saved,
               "model_path": str(path),
               "runtime_prototype_array_bytes": sum(book.nbytes for book in model.codebooks.values()),
               "dev_selected_layers": selected_dev_summary(model, dev) if dev is not None else []}
        if row["dev_selected_layers"]:
            row["dev_subset_ratio_excluding_shared_model_and_outer_index"] = ratio(row["dev_selected_layers"])
        report["configurations"].append(row)
        (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"learned{count}: {saved['serialized_bytes']} bytes; train {seconds:.3f}s; SHA256 {saved['model_id']}", flush=True)
    fixed = fixed_model(training)
    path = output / "models" / "fixed64.bin"
    saved = save_model(fixed, path)
    if expected and saved["model_id"] != expected["fixed64_ablation"]["model"]["model_id"]:
        raise AssertionError("reproduced fixed64 model SHA256 differs")
    ablation = {"construction": dict(FIXED_CONFIG), "model": saved, "model_path": str(path),
                "runtime_prototype_array_bytes_logical_per_key": sum(book.nbytes for book in fixed.codebooks.values()),
                "runtime_unique_numpy_prototype_array_bytes": next(iter(fixed.codebooks.values())).nbytes,
                "dev_selected_layers": selected_dev_summary(fixed, dev) if dev is not None else [],
                "exponent_granularity_controls": []}
    if dev is not None:
        ablation["dev_subset_ratio_excluding_shared_model_and_outer_index"] = ratio(ablation["dev_selected_layers"])
        for row in ablation["dev_selected_layers"]:
            words = dev[row["key"]][:256]
            for group in (8, 64):
                encoded = baseline_encode(words, "exponent", group_size=group)
                if not np.array_equal(words, baseline_decode(encoded)):
                    raise AssertionError("exponent control recovery mismatch")
                ablation["exponent_granularity_controls"].append({
                    "key": row["key"], "group_size": group, "raw_bytes": words.nbytes,
                    "serialized_bytes": len(encoded.data), "exact": True,
                    "component_bytes": encoded.stats["component_bytes"]})
    report["fixed64_ablation"] = ablation
    report["model_sha256_reproduction_verified"] = bool(expected)
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"fixed64: {saved['serialized_bytes']} bytes; SHA256 {saved['model_id']}", flush=True)


if __name__ == "__main__":
    main()
