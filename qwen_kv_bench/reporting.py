"""Atomic benchmark checkpoints and a narrow, shareable results bundle.

Reports distinguish original CPU-reference formats from tensor-native packets
and never average development selection with fresh test results. Bundle input
is an allowlist of reports and explicitly described small decoder-state files;
raw KV, checkpoints, spools, extracted source and arbitrary logs are excluded.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
import dataclasses
import datetime
import enum
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return _safe(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _safe(dataclasses.asdict(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_safe(item) for item in sorted(value, key=repr)]
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def write_json(path, obj):
    """Atomically write strict JSON; NumPy/Path values convert, nonfinites null."""
    text = json.dumps(_safe(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return _atomic_text(path, text)


def _fixture(case):
    fixture = case.get("fixture", "")
    return fixture.get("name", fixture.get("kind", "")) if isinstance(fixture, dict) else str(fixture)


def _split(case):
    split = case.get("split")
    if split is None and isinstance(case.get("fixture"), dict):
        split = case["fixture"].get("split")
    return str(split or "unspecified")


def _numeric_leaves(value, prefix="measurement"):
    """Keep nested scalar measurements flat; repeated timing vectors stay JSON."""
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            found.update(_numeric_leaves(item, prefix + "." + str(key)))
    elif value is None or isinstance(value, (bool, int, float)):
        found[prefix] = value
    return found


def _flat_rows(report):
    rows = []
    for case in report.get("cases", []):
        for result in case.get("results", []):
            measurement = result.get("measurement") or {}
            row = dict(split=_split(case), case_name=case.get("name", ""), fixture=_fixture(case),
                       tokens=case.get("tokens"), method_name=result.get("name", ""),
                       track=result.get("track", ""), status=result.get("status", ""),
                       execution=measurement.get("execution", ""), coverage=measurement.get("coverage", ""),
                       device=measurement.get("device", ""), error=result.get("error", ""))
            row.update(_numeric_leaves(measurement))
            rows.append(row)
    return rows


def _csv(report):
    rows = _flat_rows(report)
    base = ["split", "case_name", "fixture", "tokens", "method_name", "track", "status", "execution", "coverage", "device", "error"]
    priority = ["raw_payload_bytes", "serialized_bytes", "decoder_state_bytes", "roundtrip_exact", "records_total", "records_measured",
                "size_is_estimated", "payload_space_saving_percent", "savings_including_state_percent",
                "estimated_full_cache_serialized_bytes", "estimated_full_savings_including_state_percent",
                "encode_seconds_total", "decode_seconds_total", "one_pass_total_wall_seconds",
                "sample_decode.median_seconds", "sample_decode.MiB_per_second",
                "sample_packet_transfer_then_decode.median_seconds", "native_state_physical_bytes"]
    fields = set().union(*(row.keys() for row in rows)) if rows else set()
    preferred = ["measurement." + key for key in priority if "measurement." + key in fields]
    columns = base + preferred + sorted(fields - set(base) - set(preferred))
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return handle.getvalue()


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else "—"


def _summary(report):
    statuses = Counter(result.get("status", "unknown") for case in report.get("cases", []) for result in case.get("results", []))
    environment = report.get("environment") or {}
    identity = report.get("model_identity") or {}
    model = identity.get("model_id", identity.get("requested_model_id", identity.get("model", "not loaded/reported"))) if isinstance(identity, dict) else identity
    lines = ["# Qwen KV benchmark results", "", f"Run status: **{_cell(report.get('status', 'unknown'))}**. Model: `{_cell(model)}`.",
             f"Selected device: `{_cell(environment.get('selected_device', 'not reported'))}`. Method results: "
             + ", ".join(f"{count} {_cell(status)}" for status, count in sorted(statuses.items())) + ".", "",
             "CPU-reference codecs retain their original CPU wire formats. Copying their encoded bytes to a device is staging; it does not create a GPU decoder. Tensor-native packets are separately named formats. `tensor_native_accelerator` reports actual device tensor execution; `tensor_native_cpu_functional` is a CPU functionality check and supplies no GPU performance evidence.", "",
             "Full decoder state is included in the reported savings. Native packet state charges actual distinct tensor storage; original CPU formats charge their serialized decoder files. Those are different accounting contracts. Sampled full-cache estimates are labeled explicitly; they are not full-cache measurements. Development and TEST are separated below and never averaged.", ""]
    splits = ["dev", "test"] + sorted({_split(case) for case in report.get("cases", [])} - {"dev", "test"})
    for split in splits:
        cases = [case for case in report.get("cases", []) if _split(case) == split]
        if not cases:
            continue
        title = "Development — selection and diagnostics only" if split == "dev" else "TEST — frozen comparison" if split == "test" else "Other cases — split " + split
        lines += ["## " + title, "", "| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |",
                  "|---|---:|---|---|---|---:|---:|---:|"]
        for case in cases:
            if not case.get("results"):
                lines.append(f"| {_cell(case.get('name', ''))} | {_cell(case.get('tokens', ''))} | — | — | {_cell(case.get('status', 'no measurements'))} | — | — | — |")
            for result in case.get("results", []):
                measurement = result.get("measurement") or {}
                sampled = bool(measurement.get("size_is_estimated"))
                saving = measurement.get("estimated_full_savings_including_state_percent") if sampled else measurement.get("savings_including_state_percent")
                saving_text = _number(saving) + ("% estimated" if sampled and saving is not None else "% measured" if saving is not None else "")
                count = measurement.get("records_measured")
                records = f"{count}/{measurement.get('records_total')}" if count is not None else "—"
                median = (measurement.get("sample_decode") or {}).get("median_seconds")
                decode = _number(median * 1000) if isinstance(median, (float, int)) else "—"
                lines.append("| " + " | ".join(map(_cell, (case.get("name", ""), case.get("tokens", ""), result.get("name", ""),
                    measurement.get("execution", result.get("track", "")), result.get("status", ""), records, saving_text, decode))) + " |")
        failures = [(case, result) for case in cases for result in case.get("results", []) if result.get("error")]
        if failures:
            lines += ["", "Errors/unsupported methods:", ""]
            lines += [f"- {_cell(case.get('name', ''))} / {_cell(result.get('name', ''))}: {_cell(result['error'])}" for case, result in failures]
        skipped = [case for case in cases if not case.get("results") and case.get("error")]
        if skipped:
            lines += ["", *[f"- {_cell(case.get('name', ''))}: {_cell(case['error'])}" for case in skipped]]
        lines.append("")
    checks = report.get("generation_checks") or []
    if any((case.get("capture") or {}).get("prefill_recipes", {}).get("enabled") for case in report.get("cases", [])):
        lines += ["## Recipes constructed during prefill", "",
                  "Rows with track `prefill_observer` build recipes directly from returned device KV between prefill chunks, before CPU export. Their encode/decode totals cover selected records and one exact verification; they are not repeated restore latency samples. Post-capture `native_packet` rows measure the same method separately. Instrumented capture prefill time includes all observer work. Full raw KV remains resident and compact packets are discarded after verification; this does not establish reduced model-cache memory or compact attention.", ""]
    lines += ["## Model continuation checks", ""]
    if not checks:
        lines += ["No model continuation check is recorded yet. This report does not infer one from cache roundtrips.", ""]
    else:
        lines += ["These checks use the DEV prefix and development-selected candidates. Full-prefix reconstruction for a model check does not establish compact-resident attention.", "",
                  "| Candidate | Track | Status | Exact tokens | Exact logits | Exact appended KV |", "|---|---|---|---|---|---|"]
        for row in checks:
            value = row.get("measurement") or {}
            lines.append("| " + " | ".join(map(_cell, (row.get("name", ""), row.get("track", ""), row.get("status", ""),
                value.get("all_tokens_equal", "not reported"), value.get("all_logits_bitwise_equal", "not reported"),
                value.get("all_appended_KV_bitwise_equal", "not reported")))) + " |")
        lines.append("")
    lines += ["## Measurement scope", "",
              "CPU encode/decode totals cover selected records. Repeated decode, transfer and buffered I/O microbenchmarks use a bounded representative sample. Transfer fields separate pageable/pinned raw copies, encoded-byte staging and CPU decode/host pack followed by a bulk copy. Native packet copies and decoding have separate fields in results.json and results.csv.", "",
              "Device-event durations can include host launch gaps. Native peak allocator deltas exclude the pre-existing model/state baseline and are not total process or reserved-memory peaks. Buffered reads may hit the page cache; write+fsync/read tests do not establish physical SSD bandwidth. This run does not demonstrate attention integration, concurrency, admission capacity, TPOT or TTFT gains.", "",
              "results.json contains all captured metadata, exactness flags, components, transfer timings and errors. results.csv contains one row per method measurement and flat numeric fields; repeated sample timing vectors remain in JSON. Nonfinite JSON numeric values are written as null.", ""]
    if report.get("fatal_error"):
        lines += ["## Run error", "", _cell(report["fatal_error"]), ""]
    if report.get("warnings"):
        lines += ["## Recorded notes", "", *["- " + _cell(note) for note in report["warnings"]], ""]
    return "\n".join(lines)


def checkpoint(outputdir, report):
    """Write authoritative JSON and independently atomic CSV/Markdown views."""
    output = Path(outputdir); output.mkdir(parents=True, exist_ok=True)
    safe = _safe(report)
    paths = {"results_json": write_json(output / "results.json", safe),
             "results_csv": _atomic_text(output / "results.csv", _csv(safe)),
             "summary_md": _atomic_text(output / "summary.md", _summary(safe))}
    return paths


_MODEL_NAME = re.compile(r"(?:learned(?:16|64|256)|fixed64|lower(?:16|64)(?:_conditional)?)\.bin$")
_TRANSFORM_NAME = re.compile(r"(?:identity|dense|hadamard)_d(?:8|32|64)_[0-9a-f]{16}\.bin$")


def bundle(outputdir):
    """Return an atomic sibling ZIP containing only approved reports/state.

    Small actual calibration and referenced binary-transform files are included.
    No directory is copied recursively and no raw KV/weights/spool/source cache
    is selected by a broad extension glob. Omitted optional state is disclosed
    in the bundle manifest; core reports remain available on failed runs.
    """
    output = Path(outputdir).resolve()
    if not output.is_dir():
        raise ValueError("results directory does not exist")
    files, omitted = {}, []
    def include(path, limit, expected_sha=None, expected_bytes=None, magic=None):
        path = Path(path)
        if not path.is_absolute():
            path = output / path
        resolved = path.resolve()
        if not resolved.is_relative_to(output) or any(part in ("_source", "spools", "weights") for part in resolved.relative_to(output).parts):
            omitted.append({"file": path.name, "reason": "outside approved results tree"}); return
        relative = str(resolved.relative_to(output))
        if not resolved.is_file():
            omitted.append({"file": relative, "reason": "optional file missing"}); return
        size = resolved.stat().st_size
        if size > limit:
            omitted.append({"file": relative, "reason": "optional file exceeds size limit"}); return
        data = resolved.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if (expected_bytes is not None and size != expected_bytes) or (expected_sha is not None and digest != expected_sha) or (magic is not None and data[:4] != magic):
            omitted.append({"file": relative, "reason": "state size/hash/format check failed"}); return
        files[relative] = dict(path=resolved, bytes=size, sha256=digest)
    for name, limit in (("results.json", 32 << 20), ("results.csv", 16 << 20), ("summary.md", 4 << 20),
                        ("README.md", 1 << 20), ("source_manifest.json", 1 << 20), ("frozen_manifest.json", 4 << 20)):
        if (output / name).is_file():
            include(name, limit)
    report_path = output / "results.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    if report_path.is_file() and "results.json" not in files:
        raise ValueError("results.json exceeds shareable report size limit")
    for relative in ("calibration_manifest.json", "calibration/calibration_manifest.json", "models/calibration_manifest.json"):
        path = output / relative
        if not path.is_file():
            continue
        include(path, 8 << 20)
        if relative not in files:
            continue
        manifest = json.loads(path.read_text())
        if manifest.get("format") != "qwen_kv_calibration_v1" or manifest.get("calibration_namespace") != "qwen_kv_bench":
            omitted.append({"file": relative, "reason": "unrecognized calibration manifest; model files omitted"}); continue
        for entry in manifest.get("models", {}).values():
            filename = entry.get("file", "")
            if Path(filename).name != filename or not _MODEL_NAME.fullmatch(filename):
                omitted.append({"file": str(filename), "reason": "model name is not an approved calibration artifact"}); continue
            include(path.parent / filename, 8 << 20, entry.get("sha256"), entry.get("serialized_bytes"),
                    b"LFLW" if entry.get("family") == "lower" else b"LKVW")
    for case in report.get("cases", []):
        for row in case.get("results", []):
            info = (row.get("measurement") or {}).get("decoder_state_info") or {}
            path = info.get("transform_path")
            if path and _TRANSFORM_NAME.fullmatch(Path(path).name):
                include(path, 1 << 20, info.get("transform_sha256"), info.get("transform_bytes"), b"KVBQ")
    manifest = dict(format="qwen_kv_results_bundle_v1", files=[dict(file=name, bytes=value["bytes"], sha256=value["sha256"]) for name, value in sorted(files.items())],
        omitted=omitted, scope="Approved reports, small source manifest, calibration model files and explicitly referenced binary transform state only. Raw KV, language-model weights, spools, _source, arbitrary logs and environment files are excluded.")
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    archive = output.parent / (output.name + "_results.zip")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, prefix="." + archive.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for name, value in sorted(files.items()):
                zipped.write(value["path"], name)
            zipped.writestr("bundle_manifest.json", manifest_bytes)
        with zipfile.ZipFile(temporary) as zipped:
            if zipped.testzip() is not None:
                raise AssertionError("shareable results archive integrity failure")
        os.replace(temporary, archive); temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return archive
