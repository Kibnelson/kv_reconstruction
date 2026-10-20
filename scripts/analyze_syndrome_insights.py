#!/usr/bin/env python3
"""Analyze recorded prefill-coset metadata effects and optimizer coverage.

Requires NumPy, not Torch or model weights. Reads only recorded evidence; no
model execution, convergence proof, new codec result or information-theoretic
bound is produced. Sampling coverage is emitted only after the reconstructed
complete schedule and all logged checkpoints match their recorded identities.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_report(source):
    path = Path(source)
    if path.is_dir():
        path = next((p for p in (path / "results.json", path / "results.json.gz") if p.is_file()), None)
        if path is None:
            raise ValueError("Input directory needs results.json or results.json.gz")
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    report = json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if report.get("format") != "qwen_prefill_coset_study_v1" or report.get("status") != "completed":
        raise ValueError("Input must be a completed qwen_prefill_coset_study_v1 report")
    return report, hashlib.sha256(data).hexdigest()


def sampling_coverage(report):
    models = report["models"]
    first = models[next(iter(models))]["metadata"]
    geometry, train = first["geometry"], first["training"]
    layers, columns = int(geometry["layers"]), 2 * int(geometry["channels"])
    grouped = {layer: [] for layer in range(layers)}
    available_words = 0
    for capture_index, capture in enumerate(report["training_captures"]):
        audit = capture["producer_audit"]
        selected = audit.get("selected_records")
        if not selected:
            if audit.get("coverage") != "full":
                raise ValueError("Sampling reconstruction requires explicit selected TRAIN records")
            tokens = int(capture["tokens"])
            tile = int(audit.get("tile_tokens", report["config"]["tile_tokens"]))
            selected = [dict(layer=layer, start=start, tokens=min(tile, tokens-start))
                        for layer in range(layers) for start in range(0, tokens, tile)]
        seen = set()
        for record in selected:
            layer, start, tokens = (int(record[k]) for k in ("layer", "start", "tokens"))
            if layer not in grouped or tokens < 1 or (layer, start) in seen:
                raise ValueError("Invalid recorded TRAIN geometry or duplicate record")
            seen.add((layer, start))
            grouped[layer].append((capture_index, start, tokens))
            available_words += tokens * columns
    if any(not values for values in grouped.values()):
        raise ValueError("TRAIN records do not cover every model layer")
    count = int(train["total_optimizer_updates"])
    batch = int(train["batch_tokens"])
    channel_budget = min(int(train["channels_per_step"]), columns)
    budgets = list(map(int, first["config"]["training_residue_bits"]))
    rng = np.random.default_rng(int(train["seed"]))
    seen = np.zeros((layers, columns), dtype=np.int32)
    digest = hashlib.sha256()
    reconstructed = {}
    words = token_examples = 0
    checkpoints = {int(row["update"]): row for row in train["checkpoints"]}
    for update in range(1, count + 1):
        layer = (update - 1) % layers
        ri = int(rng.integers(len(grouped[layer])))
        capture, begin, tokens = grouped[layer][ri]
        sampled_tokens = min(batch, tokens)
        offset = int(rng.integers(tokens - sampled_tokens + 1))
        indices = np.sort(rng.choice(columns, size=channel_budget, replace=False))
        b = budgets[int(rng.integers(len(budgets)))]
        row = dict(update=update, layer=layer, capture=capture, record=ri,
                   offset=offset, start=begin+offset, tokens=sampled_tokens,
                   residue_bits=b, channel_indices=indices.tolist())
        digest.update(canonical(row))
        if update in checkpoints:
            reconstructed[update] = row
        seen[layer, indices] += 1
        words += sampled_tokens * len(indices)
        token_examples += sampled_tokens
    observed = digest.hexdigest()
    for mode, model in models.items():
        recorded = model["metadata"]["training"]
        if observed != recorded["sampling_sha256"]:
            raise ValueError("Reconstructed full sampling digest mismatch: " + mode)
        if words != recorded["total_word_examples"] or token_examples != recorded["total_token_examples"]:
            raise ValueError("Reconstructed TRAIN example counts mismatch: " + mode)
        for checkpoint in recorded["checkpoints"]:
            actual = reconstructed.get(int(checkpoint["update"]))
            if actual is None or any(actual[k] != checkpoint[k] for k in actual):
                raise ValueError("Reconstructed checkpoint fields mismatch: " + mode)
    return dict(schedule_digest_verified=observed, all_model_schedules_verified=True,
        layer_channel_slots=int(seen.size), never_selected_by_optimizer=int(np.sum(seen == 0)),
        never_selected_fraction=float(np.mean(seen == 0)), one_minibatch_slots=int(np.sum(seen == 1)),
        max_minibatches_per_slot=int(seen.max()),
        unique_channels_per_layer_range=[int(np.sum(seen > 0, axis=1).min()), int(np.sum(seen > 0, axis=1).max())],
        actual_word_examples=words, available_TRAIN_words=available_words,
        word_examples_divided_by_available_TRAIN_words=words / available_words,
        selected_TRAIN_records_by_layer={str(k): len(v) for k, v in grouped.items()},
        qualification="TRAIN priors and source normalizers use full selected captures. Never selected refers only to optimizer minibatches, not total data visibility; sampled examples may repeat.")


def row_summary(row):
    measured, account = row["measurement"], row["accounting"]
    out = dict(packet_bytes=measured["packet_bytes"],
        packet_savings_pct=account["sample"]["payload_reduction_pct"],
        net_fullcache_estimate_pct=account["full_cache_estimate"]["net_reduction_pct"],
        state_bytes=account["state"]["total_charged_bytes"],
        components=measured.get("component_bytes"), sample_raw_bytes=measured["raw_sample_bytes"])
    if "coding_summary" in measured:
        t = measured["coding_summary"]["totals"]
        values, groups = t.get("values", 0), t.get("groups", 0)
        ones, predicted, errors = (t.get(k, 0) for k in ("original_ones", "predicted_ones", "bit_errors"))
        out.update(residue_bits=measured["residue_bits"], bit_error_rate=errors / values if values else None,
            constant_zero_diagnostic_error_rate=ones / values if values else None,
            excess_errors_vs_constant_zero=errors - ones,
            predicted_positive_precision=((predicted + ones-errors)/2)/predicted if predicted else None,
            mean_errors_per31group=sum(i*x for i, x in enumerate(t.get("errors_per_group", [])))/groups if groups else None,
            recoverable_groups=t.get("recoverable_groups", 0), retained_short_groups=t.get("retained_short_groups", 0),
            groups=groups, values=values,
            constant_zero_scope="Aggregate hard-bit diagnostic from recorded ones; not an implemented packet codec or continuation check.")
    return out


def analyze(report, source_hash):
    result = dict(format="prefill_coset_metadata_insights_v1",
        scope="Read-only recorded-diagnostic arithmetic, not a GPU rerun, fixed-validation convergence test, new coding bound, or new encoded representation.",
        source_results_sha256=source_hash, selected_residue_bits=report["selection"],
        training_wall_seconds=report["training_wall_seconds"], elapsed_wall_seconds=report["elapsed_wall_seconds"],
        models={}, sampling_coverage=sampling_coverage(report), cases=[])
    for mode, model in report["models"].items():
        t = model["metadata"]["training"]
        grid = report["development_grid"][mode]
        result["models"][mode] = dict(state_bytes=model["serialized_state_bytes"],
            parameter_count=model["parameter_count"], total_optimizer_updates=t["total_optimizer_updates"],
            updates_by_layer_range=[min(t["updates_by_layer"].values()), max(t["updates_by_layer"].values())],
            total_word_examples=t["total_word_examples"], total_token_examples=t["total_token_examples"],
            last100_omitted_BCE_bits=t["final_mean_last_100_batch_losses"], fit_seconds=t["fit_seconds"],
            sampling_sha256=t["sampling_sha256"],
            training_error_by_plane=[x/n if n else None for x, n in zip(t["bit_errors_by_plane"], t["omitted_bit_examples_by_plane"])],
            development_code_summary=report["code_summaries"]["development"][mode],
            DEV_grid=[dict(residue_bits=x["residue_bits"], packet_bytes=x["packet_bytes"],
                bit_error_rate=x["measurement"]["coding_summary"]["totals"]["bit_error_rate"],
                recoverable_groups=x["measurement"]["coding_summary"]["totals"]["recoverable_groups"],
                retained_short_groups=x["measurement"]["coding_summary"]["totals"]["retained_short_groups"])
                for x in grid],
            DEV_ablations=[dict(kind=x["kind"], packet_delta_bytes=x["packet_delta_vs_supplied_code"],
                changed_prediction_fraction=x["prediction_delta"]["changed_fraction"],
                bit_error_rate=x["measurement"]["coding_summary"]["totals"]["bit_error_rate"],
                recoverable_groups=x["measurement"]["coding_summary"]["totals"]["recoverable_groups"])
                for x in report.get("development_code_ablations", {}).get(mode, [])],
            training_loss_qualification="Averages mix batches, layers and residue widths; they are not a fixed-validation convergence curve.")
    for case in report["cases"]:
        rows = {x["name"]: x for x in case["results"] if x.get("status") == "ok"}
        summary = dict(name=case["name"], tokens=case["tokens"],
            common_residue_bits=case["common_residue_bits"], common_residue_methods=case["common_residue_methods"],
            test={k: row_summary(v) for k, v in rows.items() if not k.endswith("_positions")},
            metadata_loss_vs_no_code={}, hypothetical_byte_removal_diagnostics={})
        no_code = rows.get("coset_none_syndrome")
        if no_code is not None:
            raw = no_code["measurement"]["raw_sample_bytes"]
            full = case["capture"]["raw_payload_bytes"]
            for mode in report["config"]["modes"]:
                name = "coset_" + mode + "_syndrome"
                if mode == "none" or name not in rows:
                    continue
                row = rows[name]
                parts = {k: row["measurement"]["component_bytes"].get(k, 0) -
                         no_code["measurement"]["component_bytes"].get(k, 0)
                         for k in set(row["measurement"]["component_bytes"]) | set(no_code["measurement"]["component_bytes"])}
                extra_state = row["accounting"]["state"]["total_charged_bytes"] - no_code["accounting"]["state"]["total_charged_bytes"]
                summary["metadata_loss_vs_no_code"][mode] = dict(
                    widths_match=row["measurement"]["residue_bits"] == no_code["measurement"]["residue_bits"],
                    packet_extra_bytes=row["measurement"]["packet_bytes"]-no_code["measurement"]["packet_bytes"],
                    extra_state_bytes=extra_state, components_delta=parts,
                    component_loss_pp={k: 100*v/raw for k, v in parts.items()}, state_loss_pp=100*extra_state/full,
                    net_loss_pp=no_code["accounting"]["full_cache_estimate"]["net_reduction_pct"] - row["accounting"]["full_cache_estimate"]["net_reduction_pct"],
                    qualification="Selected-width comparison; pure metadata attribution requires matching widths. Common-width row names are recorded separately.")
        for name, row in rows.items():
            if not name.endswith("_syndrome"):
                continue
            m = row["measurement"]
            removable = m.get("component_bytes", {}).get("plane_bytes", 0)
            summary["hypothetical_byte_removal_diagnostics"][name] = dict(
                all_current_omitted_plane_bytes_removed_for_free_estimated_net_pct=
                    row["accounting"]["full_cache_estimate"]["net_reduction_pct"] +
                    100*removable/m["raw_sample_bytes"], removable_sample_plane_bytes=removable,
                qualification="Hypothetical arithmetic only: delete all currently retained omitted-plane payload without replacement; hold side/state/other bytes fixed. Not an encoded result or coding bound.")
        if "field_baseline" in rows:
            summary["hypothetical_byte_removal_diagnostics"]["field_baseline"] = dict(
                one_raw_BF16_mantissa_bit_removed_for_free_estimated_net_pct=
                    rows["field_baseline"]["accounting"]["full_cache_estimate"]["net_reduction_pct"] + 100/16,
                qualification="Hypothetical one-bit-per-original-BF16-value deletion, not an implemented codec or information bound.")
        result["cases"].append(summary)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Run/package directory, results.json or results.json.gz")
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON output file")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a fresh file")
    try:
        report, source_hash = read_report(args.input)
        result = analyze(report, source_hash)
        data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"{type(error).__name__}: {error}")
    print(json.dumps(dict(output=str(args.output), cases=len(result["cases"]), models=len(result["models"]),
        sampling_digest_verified=result["sampling_coverage"]["schedule_digest_verified"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
