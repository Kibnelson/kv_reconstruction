"""Describe where already-measured byte savings occur; never encode or select.

This posthoc report compares per-layer serialized byte counts in a completed
six-case final result. It does not inspect tensors, run codecs, select settings,
or establish a causal mechanism for the differences.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


BASELINE = "exponent_huffman"
METHODS = ("lower_deflate", "split_deflate")


def _layer_bytes(stats: dict) -> dict[tuple[int, str], int]:
    rows = stats["per_layer"]
    result = {(int(row["layer"]), row["kind"]): int(row["serialized_bytes"])
              for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate per-layer entries")
    if any(layer < 0 or kind not in ("K", "V") or value < 0
           for (layer, kind), value in result.items()):
        raise ValueError("invalid per-layer entry")
    return result


def analyze(final: dict, source_sha256: str) -> dict:
    cases = final["cases"]
    names = [case["name"] for case in cases]
    if len(cases) != 6 or len(set(names)) != 6:
        raise ValueError("expected the six unique completed final cases")

    report = {
        "analysis_type": "posthoc_descriptive_byte_attribution",
        "scope": (
            "Describes differences in already measured serialized byte counts. "
            "No reencoding, raw-tensor analysis, parameter selection, or model run "
            "was performed. Concentration of savings is not a causal explanation "
            "and does not demonstrate generalization to other workloads."
        ),
        "source_final_sha256": source_sha256,
        "baseline": BASELINE,
        "compared_methods": list(METHODS),
        "selected_cases": [
            {"name": case["name"], "fixture": case.get("fixture"),
             "kind": case.get("kind"), "tokens": int(case["tokens"])}
            for case in cases
        ],
        "byte_accounting_note": (
            "Per-layer entries count serialized matrix records. Full-cache totals "
            "also include shared outer framing; any difference from the sum of "
            "per-layer savings is explicitly reported. Shares divide signed "
            "byte differences by the full-cache net saving; they can exceed one "
            "if gains elsewhere are offset by expansion."
        ),
        "methods": {},
    }

    for method in METHODS:
        aggregate = defaultdict(lambda: {"baseline_serialized_bytes": 0,
                                         "candidate_serialized_bytes": 0})
        baseline_total = candidate_total = 0
        case_summaries = []
        for case in cases:
            base_stats, candidate_stats = case["codecs"][BASELINE], case["codecs"][method]
            base_layers, candidate_layers = _layer_bytes(base_stats), _layer_bytes(candidate_stats)
            if base_layers.keys() != candidate_layers.keys():
                raise ValueError(f"per-layer keys differ in {case['name']}: {method}")
            if (0, "V") not in base_layers or (0, "K") not in base_layers:
                raise ValueError("first-layer K and V entries are required")
            base_size = int(base_stats["serialized_bytes"])
            candidate_size = int(candidate_stats["serialized_bytes"])
            baseline_total += base_size
            candidate_total += candidate_size
            differences = {key: base_layers[key] - candidate_layers[key] for key in base_layers}
            for key in base_layers:
                aggregate[key]["baseline_serialized_bytes"] += base_layers[key]
                aggregate[key]["candidate_serialized_bytes"] += candidate_layers[key]
            case_summaries.append({
                "name": case["name"],
                "baseline_serialized_bytes": base_size,
                "candidate_serialized_bytes": candidate_size,
                "saved_bytes": base_size - candidate_size,
                "layer0_V_saved_bytes": differences[(0, "V")],
                "layer0_K_saved_bytes": differences[(0, "K")],
                "later_layers_saved_bytes": sum(value for (layer, _), value in differences.items()
                                                if layer > 0),
                "shared_outer_framing_saved_bytes": base_size - candidate_size - sum(differences.values()),
            })

        per_layer = []
        for (layer, kind), counts in sorted(aggregate.items()):
            per_layer.append({"layer": layer, "kind": kind, **counts,
                              "saved_bytes": counts["baseline_serialized_bytes"] - counts["candidate_serialized_bytes"]})
        total_saved = baseline_total - candidate_total
        per_layer_saved = sum(row["saved_bytes"] for row in per_layer)
        layer0_v = next(row["saved_bytes"] for row in per_layer if row["layer"] == 0 and row["kind"] == "V")
        layer0_k = next(row["saved_bytes"] for row in per_layer if row["layer"] == 0 and row["kind"] == "K")
        later_saved = sum(row["saved_bytes"] for row in per_layer if row["layer"] > 0)
        report["methods"][method] = {
            "baseline_total_serialized_bytes": baseline_total,
            "candidate_total_serialized_bytes": candidate_total,
            "total_saved_bytes": total_saved,
            "per_layer_saved_bytes_sum": per_layer_saved,
            "shared_outer_framing_saved_bytes": total_saved - per_layer_saved,
            "layer0_V_saved_bytes": layer0_v,
            "layer0_V_share_of_net_saving": layer0_v / total_saved if total_saved else None,
            "layer0_K_saved_bytes": layer0_k,
            "later_layers_saved_bytes": later_saved,
            "later_layers_share_of_net_saving": later_saved / total_saved if total_saved else None,
            "per_layer": per_layer,
            "per_case": case_summaries,
        }
    return report


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "results" / "final.json")
    parser.add_argument("--output", type=Path, default=root / "results" / "posthoc_attribution.json")
    args = parser.parse_args()
    source = args.input.read_bytes()
    report = analyze(json.loads(source), hashlib.sha256(source).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(report["selected_cases"]),
                      "methods": {name: {key: value for key, value in data.items()
                                         if key not in ("per_layer", "per_case")}
                                  for name, data in report["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
