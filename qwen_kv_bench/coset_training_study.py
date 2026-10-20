"""Prespecified 2x2 training-coverage experiment for exact prefill coset pages.

Every arm is reported. Fixed DEV probes observe training without selecting a
checkpoint. Short/long runs share initialization and a cosine horizon; TEST
fixtures are new synthetic holdouts, not a representative serving benchmark.
"""
from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import traceback

from . import syndrome_study as study
from .reporting import write_json

FRESH_TEST = tuple(dict(row, split="test") for row in (
    dict(name="test_training_holdout_canal_gates", kind="text",
         prefix="Read the operating journal for a canal lock.",
         body="Shift <cycle>: Amara checks the upper gate's bearing temperature before opening the sluice. The crew records water levels on both sides, waits for the signal lamp, and compares vibration readings with the previous shift. A repaired pressure gauge stays in the instrument cabinet until its calibration is confirmed.\n",
         suffix="Why should the crew compare water levels before opening the gate?"),
    dict(name="test_training_holdout_interval_merge", kind="code",
         prefix="Review this interval merger and its invariant.",
         body="def merge_<cycle>(intervals):\n    merged = []\n    for lo, hi in sorted(intervals):\n        if lo > hi:\n            raise ValueError('reversed interval')\n        if merged and lo <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], hi)\n        else:\n            merged.append([lo, hi])\n    return merged\n# Each retained interval starts after the previous one ends.\n\n",
         suffix="Why must intervals be sorted before the loop?"),
    dict(name="test_training_holdout_museum_inventory", kind="structured",
         prefix="Read these museum storage records and match identifiers exactly.",
         body="Artifact copper-<cycle>: room N; shelf 23; humidity 44.\nArtifact linen-<cycle>: room P; shelf 58; humidity 39.\nArtifact stone-<cycle>: room Q; shelf 76; humidity 47.\n",
         needle="Artifact amber-926: room V; shelf 84; humidity 42.",
         suffix="Which shelf stores artifact amber-926? Give only the shelf number.", target_answer="84"),
))


def _sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            checksum.update(chunk)
    return checksum.hexdigest()


def _parser():
    parser = study._parser()
    parser.description = __doc__
    parser.add_argument("--short-steps", type=int)
    parser.add_argument("--long-steps", type=int)
    parser.add_argument("--probe-updates", nargs="+", type=int,
                        help="Fixed checkpoints; short/final updates are always included")
    parser.set_defaults(modes=["none", "kv", "hidden"])
    return parser


def _arm_argv(args, arm, output):
    argv = []
    for key in ("device", "model", "revision", "rank", "width", "batch_tokens",
                "channels_per_step", "train_tokens", "dev_tokens", "chunk_tokens",
                "tile_tokens", "max_pairs", "cases", "generation_steps", "repeats",
                "threads", "seed", "diagnostic_tokens"):
        value = getattr(args, key)
        if value is not None:
            argv.extend(("--" + key.replace("_", "-"), str(value)))
    for key in ("tiny", "local_files_only"):
        if getattr(args, key):
            argv.append("--" + key.replace("_", "-"))
    for key in ("lengths", "modes", "residue_bits"):
        argv.extend(["--" + key.replace("_", "-"), *map(str, getattr(args, key))])
    points = sorted({0, args.short_steps, arm["steps"], *args.probe_updates})
    points = [p for p in points if p <= arm["steps"]]
    argv.extend(["--steps", str(arm["steps"]), "--channel-sampling", arm["sampling"],
                 "--lr-horizon", str(args.long_steps), "--diagnostic-updates", *map(str, points),
                 "--fresh-test-fixtures", "--output", str(output)])
    return argv


def _save(output, report):
    import csv
    write_json(output / "results.json", report)
    rows = []
    for arm in report["arms"]:
        for row in arm.get("measurements", []):
            rows.append(dict(arm=arm["name"], **row))
    fields = ["arm", "case", "tokens", "method", "packet_savings_pct", "net_savings_pct", "state_bytes", "short_groups"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Prefill coset training coverage and duration", "", f"Status: **{report['status']}**", "",
             report["scope"], "", "| Arm | Case | Tokens | Method | Packet reduction | Net estimate | Short groups |",
             "|---|---|---:|---|---:|---:|---:|"]
    lines += [f"| {r['arm']} | {r['case']} | {r['tokens']} | {r['method']} | {r['packet_savings_pct']:.3f}% | {r['net_savings_pct']:.3f}% | {r['short_groups']} |" for r in rows]
    lines += ["", "All four arms and final requested updates are prespecified. DEV probes never stop training or select checkpoints. Every child run retains its packet, generation, timing, ablation and complete state-accounting logs.", "",
              "Use scripts/review_coset_training_results.py to verify the four-arm evidence. No serving speedup or research priority is established."]
    if report.get("error"):
        lines += ["", report["error"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    from .runner import _source_manifest
    parser = _parser(); args = parser.parse_args(argv)
    if args.steps is not None or args.lr_horizon is not None or args.diagnostic_updates is not None:
        parser.error("Use --short-steps, --long-steps and --probe-updates for the four-arm plan")
    study._defaults(args)
    args.short_steps = args.short_steps if args.short_steps is not None else (2 if args.tiny else 1000)
    args.long_steps = args.long_steps if args.long_steps is not None else (4 if args.tiny else 8000)
    if args.probe_updates is None:
        args.probe_updates = sorted({args.short_steps, args.long_steps,
            *[p for p in ([1, 2] if args.tiny else [250, 4000]) if p <= args.long_steps]})
    if not 0 < args.short_steps < args.long_steps or any(p < 0 or p > args.long_steps for p in args.probe_updates):
        parser.error("Require 0 < short-steps < long-steps and probe-updates in 0..long-steps")
    study._validate(args, parser)
    output = args.output or Path("runs") / ("coset_training_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True); output = output.resolve()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(channel_sampling=["random", "balanced"], fresh_test_fixtures=True,
                  lr_horizon=args.long_steps, steps=None, diagnostic_updates=None)
    arms = [dict(name=sampling + "_" + label, sampling=sampling, steps=steps)
            for sampling in ("random", "balanced")
            for label, steps in (("short", args.short_steps), ("long", args.long_steps))]
    source = _source_manifest()
    plan = dict(format="prefill_coset_training_plan_v1", config=config, arms=arms,
                lr_horizon=args.long_steps, fresh_test_fixtures=list(FRESH_TEST[:args.cases]),
                protocol="Prespecified full 2x2 plan; final TRAIN update only. Fixed DEV diagnostics observe without optimization or checkpoint selection. All arms reported; TEST never selects a winner or tuning parameter.")
    write_json(output / "source_manifest.json", source)
    write_json(output / "frozen_manifest.json", plan)
    report = dict(format="qwen_prefill_coset_training_study_v1", status="running", config=config,
                  source_manifest=source, plan=plan, arms=[],
                  scope="Fixed rank/width, same TRAIN data, same seed and shared cosine horizon. Random versus coverage-balanced channel sampling crossed with short versus long training. Actual prefill DEV probes; fresh synthetic TEST holdouts. CPU packet coding and selected-device neural work, exact original BF16, complete state charged. No early stopping or TEST-driven selection; sampled storage estimates, no serving/capacity claim.")
    _save(output, report)
    try:
        for arm in arms:
            study._note("Training experiment arm: " + arm["name"])
            directory = output / arm["name"]
            entry = dict(arm, directory=arm["name"], status="running")
            report["arms"].append(entry); _save(output, report)
            code = study.main(_arm_argv(args, arm, directory))
            child_path = directory / "results.json"
            child = json.loads(child_path.read_bytes())
            entry.update(status=child["status"], report_sha256=_sha(child_path))
            if code or child["status"] != "completed":
                raise RuntimeError("Arm failed: " + arm["name"] + ": " + child.get("error", "unknown error"))
            entry["diagnostics_sha256"] = _sha(directory / "training_diagnostics.json")
            entry["measurements"] = [dict(case=case["name"], tokens=case["tokens"], method=row["name"],
                packet_savings_pct=row["accounting"]["sample"]["payload_reduction_pct"],
                net_savings_pct=row["accounting"]["full_cache_estimate"]["net_reduction_pct"],
                state_bytes=row["accounting"]["state"]["total_charged_bytes"],
                short_groups=row["measurement"].get("coding_summary", {}).get("totals", {}).get("retained_short_groups", 0))
                for case in child["cases"] for row in case["results"] if row["status"] == "ok"]
            del child; gc.collect(); _save(output, report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        study._note(report["error"])
    finally:
        _save(output, report)
    study._note("Training experiment report: " + str(output / "summary.md"))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
