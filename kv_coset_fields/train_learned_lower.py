"""Reproduce lower-field dictionaries using only the explicit train manifest.

The six selected dev matrices describe candidates; they never modify weights.
No test/final files are loaded. Run: python -m kv_coset_fields.train_learned_lower
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from kv_coset_fields.learned_lower import train_model, save_model, encode_lower, decode_lower


def capture(path, case):
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    raw = b"".join(arrays[key].astype("<u2").tobytes() for key in sorted(arrays))
    if len(raw) != case["raw_payload_bytes"] or hashlib.sha256(raw).hexdigest() != case["raw_sha256"]:
        raise ValueError("capture hash mismatch")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".kv_research_data")
    parser.add_argument("--output-dir", default="kv_coset_fields/models")
    parser.add_argument("--verify-against")
    args = parser.parse_args()
    data_root, output = Path(args.data_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_manifest = json.loads((data_root / "train_manifest.json").read_text())
    dev_manifest = json.loads((data_root / "dev_manifest.json").read_text())
    if train_manifest["split"] != "train" or len(train_manifest["cases"]) != 3 or dev_manifest["split"] != "dev":
        raise ValueError("expected three training captures and independent dev capture")
    paths = [data_root / case["file"] for case in train_manifest["cases"]]
    loaded = [capture(path, case) for path, case in zip(paths, train_manifest["cases"])]
    training = {key: [arrays[key] for arrays in loaded] for key in loaded[0]}
    if len(dev_manifest["cases"]) != 1:
        raise ValueError("expected one explicit sensor dev capture")
    dev_case = dev_manifest["cases"][0]
    dev = capture(data_root / dev_case["file"], dev_case)
    expected = json.loads(Path(args.verify_against).read_text()) if args.verify_against else None
    expected_ids = {row["name"]: row["model"]["model_id"] for row in expected["configurations"]} if expected else {}
    report = {
        "scope": "Circular learned fraction dictionaries plus exact signs/coset corrections. Train only three explicit train captures; selected dev layers do not change weights. No final/test data loaded.",
        "training_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "training_manifest_sha256": hashlib.sha256((data_root / "train_manifest.json").read_bytes()).hexdigest(),
        "dev_file": str(data_root / dev_case["file"]), "configurations": []}
    for count in (16, 64):
        for conditional in (False, True):
            name = f"lower{count}" + ("_conditional" if conditional else "")
            config = {"n_centroids": count, "conditional": conditional, "iterations": 8,
                      "max_samples_per_key": 4096, "seed": 71, "mean_prior_count": 16}
            start = time.perf_counter(); model = train_model(training, config)
            seconds = time.perf_counter() - start
            path = output / (name + ".bin"); saved = save_model(model, path)
            if name in expected_ids and saved["model_id"] != expected_ids[name]:
                raise AssertionError("lower model reproduction SHA256 mismatch")
            row = {"name": name, "config": config, "model": saved, "model_path": str(path),
                   "training_seconds": seconds, "dev_selected_layers": []}
            for layer in (0, 15, 29):
                for kind in ("K", "V"):
                    key = f"layer_{layer:02d}_{kind}"; words = dev[key][:256]
                    lower = (((words >> 8) & 128) | (words & 127)).astype(np.uint8)
                    exponent = ((words >> 7) & 255).astype(np.uint8)
                    start = time.perf_counter(); encoded = encode_lower(lower, exponent, model, key)
                    encode_seconds = time.perf_counter() - start
                    start = time.perf_counter(); recovered = decode_lower(encoded, exponent, model, key)
                    decode_seconds = time.perf_counter() - start
                    if not np.array_equal(recovered, lower):
                        raise AssertionError("lower learned dev recovery mismatch")
                    row["dev_selected_layers"].append({"key": key, "raw_lower_bytes": lower.nbytes,
                        "serialized_bytes": len(encoded.data), "encode_seconds": encode_seconds,
                        "decode_seconds": decode_seconds, "exact": True, "stats": encoded.stats})
            raw = sum(item["raw_lower_bytes"] for item in row["dev_selected_layers"])
            encoded_bytes = sum(item["serialized_bytes"] for item in row["dev_selected_layers"])
            row["dev_subset_lower_saving_percent_excluding_model_and_outer_format"] = 100 * (1 - encoded_bytes / raw)
            row["dev_subset_all_matrices_raw_fallback"] = all(item["stats"]["mode"] == "raw_fallback" for item in row["dev_selected_layers"])
            report["configurations"].append(row)
            report["model_sha256_reproduction_verified"] = bool(expected)
            (output / "learned_lower_training_report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name}: model={saved['serialized_bytes']}B train={seconds:.3f}s devlower={row['dev_subset_lower_saving_percent_excluding_model_and_outer_format']:.4f}% allrawfallback={row['dev_subset_all_matrices_raw_fallback']}", flush=True)


if __name__ == "__main__":
    main()
