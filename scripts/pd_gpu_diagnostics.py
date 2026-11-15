#!/usr/bin/env python3
"""Isolated GPU-copy diagnostics. These runs never establish serving performance.

Each GPU-pair/route runs the pinned pd_link_probe in a fresh process, with
CUDA_LAUNCH_BLOCKING=1. No model download, calibration or vLLM engine is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile


ROUTES = ("host", "host_pipe", "direct")
DEFAULT_PAIRS = "0-1,1-0,1-2,2-1"


def parse_pairs(value):
    if not re.fullmatch(r"[0-3]-[0-3](,[0-3]-[0-3])*", value):
        raise ValueError("pairs must be logical allocated GPU indexes 0..3, e.g. 0-1,1-2")
    pairs = [tuple(map(int, item.split("-"))) for item in value.split(",")]
    if any(i == j for i, j in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("pairs must be distinct and must use two different GPUs")
    return pairs


def validate_case(record, pair, route, mib):
    """A zero child exit alone does not establish a completed exact GPU copy."""
    if record.get("gpu_executed") is not True or record.get("diagnostic_only") is not True:
        raise ValueError("missing actual-GPU or diagnostic-only marker")
    if len(record.get("gpus", [])) != 4:
        raise ValueError("this diagnostic requires the four allocated GPUs to be visible")
    if record.get("torch") != "2.11.0+cu128" or record.get("cuda_runtime") != "12.8":
        raise ValueError("the codec probe must use torch 2.11.0+cu128 / CUDA 12.8")
    if record.get("paths") != [route] or record.get("mib") != mib:
        raise ValueError("child used a different route or payload size")
    rows = record.get("pairs", [])
    if len(rows) != 1 or (rows[0].get("src"), rows[0].get("dst")) != pair:
        raise ValueError("child did not complete exactly the requested pair")
    if rows[0].get("correct") is not True or rows[0].get(route, {}).get("correct") is not True:
        raise ValueError("complete-buffer parity is missing or failed")
    if {row.get("gpu") for row in record.get("host_legs", [])} != set(range(4)):
        raise ValueError("not all allocated GPUs completed their host-leg measurements")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_child(command, logfile, env, timeout):
    """Keep crash output and stop the whole child group on timeout/interruption."""
    with logfile.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            timed_out = True
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        return proc.returncode, timed_out


def summary(report):
    lines = ["# Focused GPU copy diagnostics", "",
             "Diagnostic only: no serving or performance claim.", "",
             f"Status: **{report['status']}**. Actual GPU evidence: **{report['gpu_executed']}**.",
             f"Source: `{report['pin']}`; Slurm job: `{report['slurm_job']}`.",
             f"Payload: {report['mib']} MiB; routes use fresh processes and blocking CUDA diagnostics.", "",
             "| Pair | Route | Status | Exit | Log |", "|---|---|---|---:|---|"]
    for row in report["cases"]:
        lines.append(f"| {row['src']}→{row['dst']} | {row['route']} | {row['status']} | "
                     f"{row['exit_code']} | {row['log']} |")
    lines.extend(["", "Each child first exercises the host legs of all four visible GPUs, then the named pair route.",
                  "The selected pairs cover 0↔1 and 1↔2 by default; this is not the full node survey.",
                  "No model, calibration, engine, real NIC, SSD or serving matrix is tested here.",
                  "A failure must be diagnosed before rerunning the full campaign.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--mib", type=int, default=64)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        pairs = parse_pairs(args.pairs)
    except ValueError as exc:
        parser.error(str(exc))
    if not re.fullmatch(r"[0-9a-f]{40}", args.pin) or min(args.mib, args.reps, args.timeout) < 1:
        parser.error("a full source pin and positive sizes/repetitions/timeout are required")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside the allocated diagnostic launcher")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    report_path = args.output / "diagnostics.json"
    if report_path.exists():
        parser.error("this diagnostic output already ran; use a new tag")
    report = {"format": "t_pd_gpu_diagnostics_v1", "pin": args.pin,
              "started": datetime.now(timezone.utc).isoformat(), "slurm_job": os.environ["SLURM_JOB_ID"],
              "diagnostic_only": True, "performance_validated": False, "serving_validated": False,
              "automatic_merge": False, "gpu_executed": False, "all_selected_copies_passed": False,
              "mib": args.mib, "pairs": pairs, "routes": ROUTES, "status": "started", "cases": []}
    write_json(report_path, report)
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", PYTHONFAULTHANDLER="1", TRITON_INTERPRET="0")
    probe = Path(__file__).with_name("pd_link_probe.py")
    for i, j in pairs:
        for route in ROUTES:
            stem = f"gpu{i}_to_gpu{j}_{route}"
            output, logfile = args.output / (stem + ".json"), args.output / (stem + ".log")
            command = [sys.executable, "-u", str(probe), "--mib", str(args.mib), "--pairs", f"{i}-{j}",
                       "--paths", route, "--reps", str(args.reps), "--output", str(output)]
            print(f"DIAG_BEGIN pair={i}->{j} route={route} mib={args.mib}", flush=True)
            start = time.monotonic()
            code, timeout = run_child(command, logfile, env, args.timeout)
            row = {"src": i, "dst": j, "route": route, "exit_code": code, "timed_out": timeout,
                   "elapsed_seconds": time.monotonic() - start, "log": logfile.name, "status": "failed"}
            try:
                if code != 0 or timeout:
                    raise ValueError("child timed out" if timeout else f"child exited {code}")
                result = json.loads(output.read_text())
                validate_case(result, (i, j), route, args.mib)
                row.update(status="passed", result=output.name)
                report["gpu_executed"] = True
            except (ValueError, OSError) as exc:
                row["reason"] = str(exc)
                print("\n".join(logfile.read_text(errors="replace").splitlines()[-14:]), flush=True)
            report["cases"].append(row)
            write_json(report_path, report)
            print(f"DIAG_{row['status'].upper()} pair={i}->{j} route={route} exit={code}", flush=True)
    report["all_selected_copies_passed"] = all(row["status"] == "passed" for row in report["cases"])
    report["status"] = "passed" if report["all_selected_copies_passed"] else "failed"
    report["finished"] = datetime.now(timezone.utc).isoformat()
    write_json(report_path, report)
    (args.output / "summary.md").write_text(summary(report))
    with zipfile.ZipFile(args.output / "share.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(args.output.iterdir()):
            if file.is_file() and file.name != 'login.log' and file.suffix in (".json", ".log", ".md", ".txt"):
                archive.write(file, file.name)
    print(f"DIAG_STATUS={report['status']}\nREPORT={args.output / 'summary.md'}\nSHARE={args.output / 'share.zip'}", flush=True)
    return 0 if report["all_selected_copies_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
