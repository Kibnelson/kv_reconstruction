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
import platform
import re
import signal
import subprocess
import sys
import time
import zipfile


ROUTES = ("host", "host_pipe", "direct", "cudart_peer")
DEFAULT_PAIRS = "0-1,1-0,1-2,2-1"


def parse_pairs(value):
    if not re.fullmatch(r"[0-3]-[0-3](,[0-3]-[0-3])*", value):
        raise ValueError("pairs must be logical allocated GPU indexes 0..3, e.g. 0-1,1-2")
    pairs = [tuple(map(int, item.split("-"))) for item in value.split(",")]
    if any(i == j for i, j in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("pairs must be distinct and must use two different GPUs")
    return pairs


def validate_case(record, pair, route, mib, reps=None):
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
    if route == 'cudart_peer':
        count = record.get('requested_reps')
        if (record.get('format') != 't_pd_cudart_peer_v1' or record.get('runtime_api_only') is not True
                or record.get('torch_imported') is not False or record.get('status') != 'passed'
                or type(count) is not int or count < 1 or record.get('completed_reps') != count
                or (reps is not None and count != reps) or record.get('cleanup_errors')):
            raise ValueError('incomplete independent CUDA-runtime proof')
        controls, trials = record.get('host_controls', []), record.get('trials', [])
        if (len(controls) != 2 * count or {(x.get('gpu'), x.get('repetition')) for x in controls}
                != {(gpu, rep) for gpu in pair for rep in range(count)}
                or any(x.get('correct') is not True or x.get('mismatched_bytes') != 0
                       or x.get('bytes') != mib << 20 for x in controls)):
            raise ValueError('missing full local CUDA/host controls on both devices')
        if (len(trials) != count or {x.get('repetition') for x in trials} != set(range(count))
                or any(x.get('correct') is not True or x.get('source_unchanged') is not True
                       or x.get('mismatched_bytes') != 0 or x.get('untouched_poison_bytes') != 0
                       or x.get('bytes') != mib << 20 for x in trials)):
            raise ValueError('CUDA peer buffer/source parity is incomplete')
        enabled = record.get('peer_enable', [])
        if (len(enabled) != 2 or {(x.get('src'), x.get('dst')) for x in enabled} != {pair, pair[::-1]}
                or any(x.get('status') not in ('enabled', 'already_enabled') for x in enabled)):
            raise ValueError('both peer directions must be explicitly enabled')
        return
    if {row.get("gpu") for row in record.get("host_legs", [])} != set(range(4)):
        raise ValueError("not all allocated GPUs completed their host-leg measurements")


def platform_context(proc_root=Path('/proc'), sys_root=Path('/sys')):
    """Read-only clues, not a determination of IOMMU/ACS or DMA correctness."""
    result = {'kernel': platform.release(), 'system_settings_changed': False,
              'iommu_cause_established': False}
    try:
        words = (proc_root/'cmdline').read_text().split()
        result['relevant_boot_flags'] = [word for word in words if word.startswith(
            ('iommu=', 'intel_iommu=', 'amd_iommu=', 'pcie_acs_override='))]
    except OSError as exc:
        result['boot_flags_error'] = str(exc)
    try:
        result['iommu_group_count'] = len(list((sys_root/'kernel/iommu_groups').iterdir()))
    except OSError as exc:
        result['iommu_groups_error'] = str(exc)
    commands = {'p2p_read': ['nvidia-smi', 'topo', '-p2p', 'r'],
                'p2p_write': ['nvidia-smi', 'topo', '-p2p', 'w'],
                'pci_bus_ids': ['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader']}
    for name, command in commands.items():
        try:
            child = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = {'exit': child.returncode, 'stdout': child.stdout, 'stderr': child.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {'error': str(exc)}
    result['gpu_iommu_groups'] = []
    for line in result['pci_bus_ids'].get('stdout', '').splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])', line.strip())
        if match:
            bdf = f'{int(match[1], 16):04x}:{match[2].lower()}'
            link = sys_root/'bus/pci/devices'/bdf/'iommu_group'
            try:
                group = link.resolve(strict=True).name if link.is_symlink() else None
                result['gpu_iommu_groups'].append({'pci_bus_id': bdf, 'iommu_group': group})
            except OSError as exc:
                result['gpu_iommu_groups'].append({'pci_bus_id': bdf, 'error': str(exc)})
    return result


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
    lines.extend(["", "Torch children exercise all four GPUs' host legs, then the named pair route.",
                  "cudart_peer imports no Torch: it uses cudaMalloc/cudaMemcpyPeer with explicit peer enable,",
                  "local copy/host controls on both selected devices, new payloads and poisoned destinations.",
                  "Its Torch version is installed-package metadata, not the copy implementation.",
                  "platform_context.json contains read-only peer/IOMMU clues; it establishes no hardware cause.",
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
    report = {"format": "t_pd_gpu_diagnostics_v2", "pin": args.pin,
              "started": datetime.now(timezone.utc).isoformat(), "slurm_job": os.environ["SLURM_JOB_ID"],
              "diagnostic_only": True, "performance_validated": False, "serving_validated": False,
              "automatic_merge": False, "gpu_executed": False, "all_selected_copies_passed": False,
              "mib": args.mib, "pairs": pairs, "routes": ROUTES, "status": "started", "cases": []}
    write_json(report_path, report)
    print('DIAG_PLATFORM_CONTEXT collecting read-only peer/IOMMU details', flush=True)
    write_json(args.output/'platform_context.json', platform_context())
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", PYTHONFAULTHANDLER="1", TRITON_INTERPRET="0")
    probe = Path(__file__).with_name("pd_link_probe.py")
    for i, j in pairs:
        for route in ROUTES:
            stem = f"gpu{i}_to_gpu{j}_{route}"
            output, logfile = args.output / (stem + ".json"), args.output / (stem + ".log")
            worker = probe.with_name('pd_cudart_peer_probe.py') if route == 'cudart_peer' else probe
            command = [sys.executable, "-u", str(worker), "--mib", str(args.mib), "--pairs", f"{i}-{j}",
                       "--paths", route, "--reps", str(args.reps), "--output", str(output)]
            print(f"DIAG_BEGIN pair={i}->{j} route={route} mib={args.mib}", flush=True)
            start = time.monotonic()
            code, timeout = run_child(command, logfile, env, args.timeout)
            row = {"src": i, "dst": j, "route": route, "exit_code": code, "timed_out": timeout,
                   "elapsed_seconds": time.monotonic() - start, "log": logfile.name, "status": "failed"}
            if output.is_file():
                row['result'] = output.name  # Failed raw-runtime parity records are evidence too.
            try:
                if code != 0 or timeout:
                    raise ValueError("child timed out" if timeout else f"child exited {code}")
                result = json.loads(output.read_text())
                validate_case(result, (i, j), route, args.mib, args.reps)
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
