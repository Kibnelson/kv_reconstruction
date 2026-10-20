#!/usr/bin/env python3
"""Audit preserved team source and logs without importing/running its model script.

Historical log assertions are reported as historical evidence. New checks cover
source identity, arithmetic, all-word symbol mappings and probability counts;
they do not independently reproduce the original model's compression rate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from fractions import Fraction
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

FORMAT = "team_transfer_source_audit_v1"
AUDITED_COMBINED_SHA256 = "b44a63d32c1d3144436a01205701a26478fed7a63c8bf27777932a3436354cf3"
ROOT = Path(__file__).resolve().parents[1]
HELPERS = {"from_bits", "ordered", "word_of", "cdf_int", "group_num", "mant_num"}
NP_ATTRIBUTES = {"asarray", "uint16", "uint32", "int64", "float32", "float64",
                 "where", "clip", "minimum", "floor", "diff", "arange"}
METHODS = {"astype", "view"}
SAFE_NODES = (ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
              ast.Name, ast.Load, ast.Store, ast.Constant, ast.Call, ast.Attribute,
              ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp, ast.Subscript,
              ast.Tuple, ast.List, ast.Slice, ast.keyword, ast.Add, ast.Sub,
              ast.Mult, ast.Div, ast.Pow, ast.LShift, ast.RShift, ast.BitAnd,
              ast.BitOr, ast.BitXor, ast.Invert, ast.USub, ast.UAdd, ast.Eq,
              ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_relative(name: str) -> Path:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or not p.parts:
        raise ValueError(f"Unsafe source-manifest path: {name!r}")
    return Path(*p.parts)


def validate_helper(node: ast.FunctionDef) -> None:
    """Whitelist pure NumPy expression helpers; reject imports, I/O and builtins."""
    if node.decorator_list or node.args.defaults or node.args.kw_defaults:
        raise ValueError("Helper decorators/default expressions are not allowed")
    if node.returns or any(arg.annotation for arg in node.args.args):
        raise ValueError("Helper annotations are not allowed")
    for child in ast.walk(node):
        if not isinstance(child, SAFE_NODES):
            raise ValueError(f"Unsafe helper AST node: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id.startswith("__"):
            raise ValueError("Private helper names are not allowed")
        if isinstance(child, ast.Assign):
            if any(not isinstance(target, ast.Name) for target in child.targets):
                raise ValueError("Only local-name assignments are allowed")
            if any(target.id in HELPERS | {"np", "__builtins__"} for target in child.targets):
                raise ValueError("Helper cannot replace a shared name")
        if isinstance(child, ast.Attribute):
            if child.attr.startswith("_"):
                raise ValueError("Private attributes are not allowed")
            if isinstance(child.value, ast.Name) and child.value.id == "np":
                if child.attr not in NP_ATTRIBUTES:
                    raise ValueError(f"NumPy attribute is not allowed: {child.attr}")
            elif child.attr not in METHODS:
                raise ValueError(f"Array attribute is not allowed: {child.attr}")
        if isinstance(child, ast.Call):
            fn = child.func
            if isinstance(fn, ast.Name):
                if fn.id not in HELPERS:
                    raise ValueError(f"Helper call is not allowed: {fn.id}")
            elif not isinstance(fn, ast.Attribute):
                raise ValueError("Indirect helper calls are not allowed")


def extract_helpers(source: str) -> dict[str, Any]:
    """Execute only explicitly selected, AST-validated function definitions.

    No module imports, CLI statements, fitting, file access, or top-level source
    expressions are executed. Constants/tables below are reconstructed explicitly.
    """
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    if len(nodes) != len(HELPERS) or {n.name for n in nodes} != HELPERS:
        raise ValueError("Missing or duplicate audited helpers")
    for node in nodes:
        validate_helper(node)
    namespace: dict[str, Any] = {"np": np, "__builtins__": {}}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "<validated-team-cdf-helpers>", "exec"), namespace)
    # These are the original combined.py's documented table parameters, not a
    # replacement codec. Their source identities remain covered by the manifest.
    with np.errstate(invalid="ignore"):
        values = namespace["from_bits"](namespace["word_of"](np.arange(65536))).astype(np.float64)
        edges = np.empty(65537)
        edges[1:65536] = (values[:-1] + values[1:]) / 2
    edges[:129] = -np.inf
    edges[65408:] = np.inf
    table_limit, table_resolution = 32.0, 256
    table_size = int(2 * table_limit * table_resolution) + 1
    grid = -table_limit + np.arange(table_size) / table_resolution
    table = np.rint(2.0 ** 32 / (1 + np.exp(-grid))).astype(np.int64)
    table[0], table[-1] = 0, 2 ** 32
    namespace.update(BD=edges, GB=edges[::128], TT=table_limit,
                     TR=table_resolution, NTAB=table_size, F_TAB=table, ONE16=2 ** 16)
    return namespace


HEADER = re.compile(r"(?m)^(\w+), tokens (\d+)-(\d+), all (\d+) layers, ([\d,]+) values, "
                    r"decoder block (\d+) channels \((\d+) steps/layer\)$")
METRIC = re.compile(r"(?m)^\s*(transfer stream \(real bytes\)|GPU pool, packed \(real bytes\)|"
                    r"current GPU format on same tokens)\s+([\d.]+) bits/value\s+([\d.]+)% smaller")


def parse_logs(text: str) -> list[dict[str, Any]]:
    headers = list(HEADER.finditer(text))
    if not headers:
        raise ValueError("No end-to-end chunk records found")
    rows = []
    keys = {"transfer stream (real bytes)": "transfer", "GPU pool, packed (real bytes)": "pool",
            "current GPU format on same tokens": "baseline_analytical"}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else text.find("Average of the 4 chunks", header.end())
        body = text[header.end():end if end >= 0 else len(text)]
        document, start, stop, layers, count, block, steps = header.groups()
        row: dict[str, Any] = dict(document=document, start=int(start), stop=int(stop),
                                  tokens=int(stop)-int(start)+1, layers=int(layers),
                                  values=int(count.replace(",", "")), block_channels=int(block),
                                  steps_per_layer=int(steps), metrics={})
        for name, bpv, reduction in METRIC.findall(body):
            key = keys[name]
            if key in row["metrics"]:
                raise ValueError(f"Duplicate metric {key}")
            row["metrics"][key] = dict(logged_bpv=float(bpv), logged_reduction_percent=float(reduction),
                                       evidence="historical supplied log; not independently rerun")
        if set(row["metrics"]) != set(keys.values()):
            raise ValueError("Incomplete chunk metrics")
        matches = re.search(r"decoded words == original: (True|False); encoder pass == original: (True|False); "
                            r"pool unpacked == original: (True|False)", body)
        if not matches:
            raise ValueError("Missing historical round-trip assertions")
        row["historical_roundtrip_assertions"] = dict(zip(("decoded", "encoder", "pool"),
                                                               [x == "True" for x in matches.groups()]))
        times = re.search(r"fit ([\d.]+), encode ([\d.]+), decode\+pack ([\d.]+), unpack ([\d.]+)", body)
        if times:
            row["historical_cpu_seconds"] = dict(zip(("fit", "encode", "decode_and_pack", "unpack"),
                                                     map(float, times.groups())))
            row["historical_cpu_raw_restore_bytes_per_second"] = 2*row["values"] / float(times.group(3))
        rows.append(row)
    return rows


def coefficient_accounting(layers: int, channels: int, block: int) -> dict[str, Any]:
    if layers < 1 or channels < 1 or block < 1 or channels % block:
        raise ValueError("Invalid predictor geometry")
    layer_zero = channels * (channels-block) // 2
    later = (layers-1) * (channels*channels + layer_zero)
    active = layer_zero + later
    padded = channels*channels + (layers-1)*2*channels*channels
    return dict(layers=layers, combined_kv_channels=channels, block_channels=block,
                layer_zero_active_coefficients=layer_zero, later_layer_active_coefficients=later,
                active_coefficients=active, hypothetical_compact_int32_coefficient_bytes=4*active,
                actual_padded_int64_coefficient_bytes=8*padded,
                actual_padded_coefficients=padded,
                actual_int64_bias_bytes=8*layers*channels,
                actual_float64_inverse_scale_bytes=8*layers*channels,
                useful_macs_per_value_all_layers=active/(layers*channels),
                useful_macs_per_value_predicted_layers=(channels*channels+layer_zero)/channels)


def amortization(state_bytes: int, bpv: float, layers: int, channels: int,
                 tokens: int = 8192, target_reduction: float = .4) -> dict[str, Any]:
    values = tokens*layers*channels
    inclusive_bpv = bpv + state_bytes*8/values
    # Inputs are reported decimal rates: avoid rounding an exact 10-token
    # boundary up to 11 because 9.6 - 8.0 is represented slightly below 1.6.
    margin = 16*(1-Fraction(str(target_reduction)))-Fraction(str(bpv))
    amount = Fraction(state_bytes*8, layers*channels)/margin if margin > 0 else None
    needed = ((amount.numerator+amount.denominator-1)//amount.denominator) if amount is not None else None
    return dict(state_bytes=state_bytes, tokens=tokens, stream_bpv=bpv,
                state_inclusive_bpv=inclusive_bpv, state_inclusive_reduction_percent=100*(1-inclusive_bpv/16),
                target_reduction_percent=100*target_reduction, minimum_aggregate_tokens_for_target=needed,
                assumptions="Constant observed stream rate; coefficients installed once and shared. Excludes other constants, framing and scratch.")


LIMITATIONS = [
    dict(id="original_model_data_missing", status="unverified", detail="Original GGUF and ACT captures were not supplied; historical compression and word comparisons have not been independently reproduced."),
    dict(id="monolithic_entropy_stream", status="observed", detail="One RangeEncoder is used across tokens, layers and channel groups. Independent predictor tokens do not make the original entropy stream independently decodable."),
    dict(id="external_framing", status="observed", detail="Shape, token positions, fitted constants and tables come from CLI/globals. Their delivery, identities, chunk offsets and checksums are not included in stream bytes."),
    dict(id="full_dense_output", status="observed", detail="transfer allocates a full decoded output with zeros_like(ORIG); the script retains originals and additional encoder/decoded/unpacked copies. Bounded-scratch compressed residency is not demonstrated."),
    dict(id="integer_portability", status="unverified", detail="The predictor uses integers, but standardization and constriction inputs use floating point. Constants/tables are regenerated; cross-machine parity is not tested."),
    dict(id="extreme_feature_conversion", status="unverified", detail="fxv casts very large finite values, infinities and NaNs to int64 without specified fallback/clamping. Positive symbol counts alone do not validate this feature path."),
    dict(id="gpu_performance", status="unmeasured", detail="Supplied execution is CPU only. MAC counts and payload ratios do not establish GPU throughput or serving benefit."),
    dict(id="baseline_size_analytical", status="observed", detail="The original baseline GPU-format size is calculated by a formula; transfer/pool payloads are actually packed."),
    dict(id="hook_coset_extensions", status="absent", detail="The original combined codec conditions on produced KV; it contains no actual upstream hidden metadata or transmitted coset residues."),
]


def review(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    manifest_raw = (source_dir/"source_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, **details: Any) -> None:
        checks.append(dict(check=name, passed=bool(passed), **details))
    identities = []
    for name, expected in sorted(manifest["files"].items()):
        rel = safe_relative(name)
        raw = (source_dir/rel).read_bytes()
        actual = dict(file=name, bytes=len(raw), sha256=sha(raw))
        identities.append(actual)
        check(f"source identity: {name}", len(raw) == expected["bytes"] and sha(raw) == expected["sha256"])
    source_raw = (source_dir/"combined.py").read_bytes()
    check("audited original combined.py version", sha(source_raw) == AUDITED_COMBINED_SHA256)
    source = source_raw.decode()
    log_raw = (source_dir/"combined_results.txt").read_bytes()
    rows = parse_logs(log_raw.decode())
    total = sum(r["values"] for r in rows)
    for row in rows:
        expected = row["tokens"]*row["layers"]*row["block_channels"]*row["steps_per_layer"]
        check(f"logged value geometry: {row['document']}:{row['start']}", expected == row["values"])
        for key, metric in row["metrics"].items():
            calculated = 100*(1-metric["logged_bpv"]/16)
            check(f"rounded historical percentage: {row['document']}:{row['start']}:{key}",
                  abs(calculated-metric["logged_reduction_percent"]) <= .06)
    averages = {key: sum(r["values"]*r["metrics"][key]["logged_bpv"] for r in rows)/total
                for key in rows[0]["metrics"]}
    reference = rows[0]
    channels = reference["block_channels"]*reference["steps_per_layer"]
    check("all historical records share predictor geometry", all(
        r["layers"] == reference["layers"] and r["block_channels"] == reference["block_channels"] and
        r["block_channels"]*r["steps_per_layer"] == channels for r in rows))
    check("historical logs report all roundtrips true", all(all(r["historical_roundtrip_assertions"].values()) for r in rows))
    helpers = extract_helpers(source)
    words = np.arange(65536, dtype=np.uint16)
    check("new test: all BF16 symbol mappings invert exactly", np.array_equal(
        helpers["word_of"](helpers["ordered"](words)), words), tested_words=len(words))
    means = np.array([-100., -1., 0., 1., 100.])
    inverse_scales = np.array([.125, .5, 1., 2., 8.])
    gn = helpers["group_num"](means, inverse_scales)
    check("new test: probability group counts are positive and normalize", (gn > 0).all() and
          (gn.sum(1) == 2**48).all(), means=means.tolist(), inverse_scales=inverse_scales.tolist())
    groups = np.tile(np.arange(512), len(means))
    mn = helpers["mant_num"](np.repeat(means, 512), np.repeat(inverse_scales, 512), groups)
    check("new test: every mantissa group matches its group count", (mn > 0).all() and
          np.array_equal(mn.sum(1), gn.ravel()), tested_word_probabilities=int(mn.size), minimum_word_count=int(mn.min()))
    check("new test: ordered BF16 interval edges are monotone", not np.isnan(helpers["BD"]).any() and
          (helpers["BD"][1:] >= helpers["BD"][:-1]).all())
    coefs = coefficient_accounting(reference["layers"], channels, reference["block_channels"])
    bpv = averages["transfer"]
    amortized = {"hypothetical_compact_int32_coefficients": amortization(
                    coefs["hypothetical_compact_int32_coefficient_bytes"], bpv, reference["layers"], channels),
                 "actual_padded_int64_coefficients": amortization(
                    coefs["actual_padded_int64_coefficient_bytes"], bpv, reference["layers"], channels)}
    observed = dict(layer_zero_earlier_channel_predictor=True,
                    original_target_numeric_input_to_encoder_only=True,
                    decoder_has_shape_only_ORIG_reference=True,
                    source_uses_single_entropy_stream=True,
                    actual_A_dtype="int64", actual_A_storage="zero-padded dense matrices",
                    scope="Manual source observations, not a general static proof or an independent decoder execution")
    return dict(format=FORMAT, scope="Source/log audit and independent helper tests; not original-data compression reproduction",
                all_passed=all(c["passed"] for c in checks), passed=sum(c["passed"] for c in checks), total=len(checks),
                checks=checks,
                provenance=dict(source=manifest.get("source"), original_archive_sha256=manifest.get("source_sha256"),
                                source_manifest_sha256=sha(manifest_raw), files=identities,
                                archive_hash_recomputed=False, archive_hash_scope="Copied from preserved manifest; checks above hash the extracted source files."),
                historical_evidence=dict(records=rows, total_values=total, chunk_count=len(rows),
                                         weighted_average_logged_bpv=averages,
                                         weighted_average_reduction_percent={k:100*(1-v/16) for k,v in averages.items()},
                                         rates_are_rounded_log_values=True,
                                         model_compression_independently_reproduced=False),
                coefficient_accounting=coefs, amortization=amortized,
                source_observations=observed, production_limitations=LIMITATIONS)


def markdown(report: dict[str, Any]) -> str:
    h = report["historical_evidence"]
    c = report["coefficient_accounting"]
    avg = h["weighted_average_logged_bpv"]
    lines = ["# Team transfer source audit", "", report["scope"]+".", "",
             f"New audit checks: **{report['passed']}/{report['total']} passed**. This does not independently confirm the original model's compression rates.", "",
             "## Historical supplied logs", "",
             f"{h['chunk_count']} chunks; {h['total_values']:,} values. Logs report all encoder, decoder and pool word comparisons true.", "",
             "| Metric | Average logged bits/value | Reduction from BF16 |", "|---|---:|---:|"]
    for key, label in (("transfer","Transfer packed stream"),("pool","Pool packed bytes"),("baseline_analytical","Analytical baseline")):
        lines.append(f"| {label} | {avg[key]:.3f} | {100*(1-avg[key]/16):.2f}% |")
    lines += ["", "The averages above use rounded historical log values. Source/data reproduction remains separate.", "",
              "## Independent tests performed now", "",
              "The audit executes only six AST-whitelisted NumPy expression helpers; it never imports or executes the model script. It verifies every BF16 word mapping, positive normalized group counts, all mantissa-group sums for five mean/scale pairs, interval monotonicity, source hashes and log arithmetic.", "",
              "## Predictor state", "",
              f"Layer zero contributes {c['layer_zero_active_coefficients']:,} coefficients. Total active coefficients: **{c['active_coefficients']:,}**.", "",
              f"A hypothetical compact INT32 store costs {c['hypothetical_compact_int32_coefficient_bytes']:,} bytes. The actual original padded INT64 `A` arrays cost **{c['actual_padded_int64_coefficient_bytes']:,} bytes**, before biases, scales and tables.", "",
              f"Useful MACs/value: {c['useful_macs_per_value_all_layers']:.1f} across all layers; {c['useful_macs_per_value_predicted_layers']:.1f} in predicted layers.", "",
              "| State charged once | 8K state-inclusive bpv | 8K reduction | Aggregate tokens for 40% |", "|---|---:|---:|---:|"]
    for key, row in report["amortization"].items():
        lines.append(f"| {key.replace('_',' ')} | {row['state_inclusive_bpv']:.3f} | {row['state_inclusive_reduction_percent']:.2f}% | {row['minimum_aggregate_tokens_for_target']:,} |")
    lines += ["", "Amortization assumes a constant observed 8.360-bpv stream rate and reusable installed coefficients. Other constants, framing and scratch are excluded; these are arithmetic scenarios rather than model measurements.", "", "## Production limitations", ""]
    for row in report["production_limitations"]:
        lines.append(f"- **{row['id']} ({row['status']}):** {row['detail']}")
    lines += ["", "## Next matched comparison", "",
              "Use the same real KV captures, frozen exact coder and TRAIN/DEV/TEST split for previous-layer-only, AQUA-style K/V, and earlier-channel predictors; then add paid KV-derived metadata, paid true-hidden metadata, coset residues, and their combination. Freeze options on DEV. Report packet bytes, shared state, peak scratch, exact packet-only reconstruction and encode/decode timings separately.", "",
              "Coset factorization is bijective and does not create additional ideal entropy savings by itself. Metadata must pay its own bytes; any improvement is a measured gain of the practical model/format.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT/"experiments/team_transfer_original")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be fresh or empty")
    report = review(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    (args.output/"summary.md").write_text(markdown(report))
    print(json.dumps(dict(all_passed=report["all_passed"], passed=report["passed"], total=report["total"],
                          original_compression_reproduced=False, output=str(args.output))))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
