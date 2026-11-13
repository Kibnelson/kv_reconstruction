#!/usr/bin/env python3
"""Tests of five ideas for T (kv_compare/t_ideas.py), on the captures of a finished run folder of
scripts/run_a5000.sh (WikiText-2 train calibration: wtrain0-5, wtables; dataset windows ds_<name>_r<i>).

  python scripts/t_ideas.py --run $KV_HOME/runs/llama32_3b --output OUT [--stages capture,adapt,layer0,anchor,classes,gpu,report]

Stages (a finished stage is kept in OUT/t_ideas.json; --redo STAGE measures it again):
  capture  dataset windows still missing (--datasets, --windows per dataset), with the run's model and texts
  adapt    idea 1: T's exact bits per dataset as it is, with forward and backward per-request constants, the float
           one-step predictor, and the oracle refit on each window
  layer0   idea 2: layer 0 recomputed from the token ids on this GPU (bit-identical words, coding cost, time)
  anchor   idea 4: anchors (most recent occurrence, first occurrence, nearest key frame)
  classes  idea 5: token classes (sink, repeated, k-means of layer l-1, encoder's choice of 2 or 4)
  gpu      ideas 2 and 3 on the GPU: every mode's bits and decode time per layer, products alone, coding alone, the
           coding kernel's memory floor; per-layer mode allocation for each link speed
  report   OUT/summary.md with the decision rules of docs/t_ideas.md
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STAGES = ("capture", "adapt", "layer0", "anchor", "classes", "gpu", "report")
DATASETS = ("wikitext2", "humaneval", "gsm8k", "mmlu", "ptb")
NAMES = {"wikitext2": "WikiText-2", "humaneval": "HumanEval", "gsm8k": "GSM8K", "mmlu": "MMLU", "ptb": "PTB"}
LINKS = (1, 3, 10, 25, 100, 400)
MODES = ("dense", "kv", "4g", "hbh", "rank256")
TRAIN_SETS = {"wikitext": ([f"wtrain{i}" for i in range(6)], "wtables"),
              "code": ([f"train{i}" for i in range(6)], "tables")}
LOG = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def windows(cap, wanted, per):
    """{dataset: [window names]} of the captured windows ds_<name>_r<i>, i < per; plus test_text / test_code."""
    out = {}
    for name in wanted:
        ws = [f"ds_{name}_r{i}" for i in range(per) if (cap / f"ds_{name}_r{i}_K_bits.npy").is_file()]
        if ws:
            out[name] = ws
    for t in ("test_text", "test_code"):
        if (cap / f"{t}_K_bits.npy").is_file() and not out:
            out[t] = [t]
    return out


def model_of(args):
    if args.model:
        return args.model, args.revision
    p = Path(args.run) / "model_resolved.json"
    if p.is_file():
        m = json.loads(p.read_text())
        return m["model"], m.get("revision")
    meta = json.loads((Path(args.run) / "captures" / "meta.json").read_text())
    return meta.get("model"), meta.get("revision")


def load_model(args, device):
    from kv_compare.capture import load_model as lm
    mid, rev = model_of(args)
    if not mid:
        raise SystemExit("no model id: pass --model (and --revision)")
    log(f"model {mid} at {rev}")
    tok, model = lm(mid, rev, device)
    return tok, model


def stage_capture(args, res, device):
    from kv_compare.capture import bos_prefix, capture_ids
    cap = Path(args.run) / "captures"
    texts = Path(args.texts) if args.texts else Path(args.run).parent / "dataset_texts"
    need = [(n, i) for n in args.datasets for i in range(args.windows) if not (cap / f"ds_{n}_r{i}_K_bits.npy").is_file()]
    done = res.setdefault("capture", {})
    done.setdefault("captured", [])
    done.setdefault("skipped", [])
    if not need:
        log("capture: every window is there")
        return
    tok, model = load_model(args, device)
    pre = bos_prefix(tok)
    step = args.tokens - len(pre)
    ids_of = {}
    for name, i in need:
        t = texts / f"{name}.txt"
        if not t.is_file():
            done["skipped"].append(f"ds_{name}_r{i}: no {t} (scripts/fetch_eval_texts.py --output {texts})")
            log(done["skipped"][-1])
            continue
        if name not in ids_of:
            ids_of[name] = tok(t.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
        ids = ids_of[name]
        if (i + 1) * step > len(ids):
            done["skipped"].append(f"ds_{name}_r{i}: {name} has {len(ids)} tokens")
            continue
        capture_ids(model, pre + ids[i * step:(i + 1) * step], cap, f"ds_{name}_r{i}", device, chunk_tokens=8192)
        done["captured"].append(f"ds_{name}_r{i}")
        log(f"capture: ds_{name}_r{i}")
    del model


def study_of(args, tests, device):
    from kv_compare.predict import Study
    train, tables = TRAIN_SETS[args.train_set]
    return Study(Path(args.run) / "captures", device, train, tables, tests, args.max_layers, log=lambda m: None)


def consts_of(args, study, out):
    """T's constants (fused.prepare), cached in OUT (a 3B model's take a few minutes to fit)."""
    import torch
    from kv_compare import fused
    p = out / "t_consts.pt"
    key = {"run": str(Path(args.run).resolve()), "train_set": args.train_set, "layers": study.L}
    if p.is_file():
        blob = torch.load(p, map_location=study.device, weights_only=False)
        if blob.get("key") == key:
            return blob["consts"]
    log("fitting T's constants (fused.prepare)")
    consts = fused.prepare(study, log=log)
    torch.save({"key": key, "consts": consts}, p)
    return consts


def stage_adapt(args, res, device, out):
    from kv_compare import t_ideas as ti
    ws = windows(Path(args.run) / "captures", args.datasets, args.windows)
    docs = [w for v in ws.values() for w in v]
    if not docs:
        raise SystemExit("adapt: no dataset windows captured (run the capture stage)")
    study = study_of(args, docs, device)
    consts = consts_of(args, study, out)
    r = res.setdefault("adapt", {})
    r.setdefault("docs", {})
    r["windows"], r["variants"] = ws, ti.ADAPT
    for d in study.tests:
        row = r["docs"].get(d.name)
        if row and "oracle" in row:
            continue
        t0 = time.time()
        per, side = ti.t_costs(study, consts, d)
        row = {v: {"bits": float(per[v].mean()), "side": side[v], "layers": per[v].tolist()} for v in per}
        study._f.clear()
        inside, cross = ti.oracle_costs(study, d, log=lambda m: None)
        row["oracle"] = {"bits": float(cross.mean()), "side": 0.0, "layers": cross.tolist()}
        row["oracle-in-sample"] = {"bits": float(inside.mean()), "side": 0.0}
        r["docs"][d.name] = row
        log(f"adapt {d.name}: T {row['T']['bits']:.4f}, " + ", ".join(
            f"{v} {row[v]['bits'] + row[v]['side']:.4f}" for v in per if v != "T")
            + f", oracle {row['oracle']['bits']:.4f} bits/value ({time.time() - t0:.0f} s)")
        save(res, out)
    if "float" not in r:
        fl, _ = ti.float_costs(study, ti.ANCHORS[0], study.train, study.tables, study.tests, log=lambda m: None)
        r["float"] = {n: float(v.mean()) for n, v in fl.items()}
        log("adapt: float one-step " + ", ".join(f"{n} {v:.4f}" for n, v in r["float"].items()))


def stage_layer0(args, res, device, out):
    import torch
    from kv_compare import t_ideas as ti
    from kv_compare.gpu_speed import timer
    ws = windows(Path(args.run) / "captures", args.datasets, args.windows)
    docs = [v[0] for v in ws.values()]
    study = study_of(args, docs, device)
    t0 = {n: row["T"]["layers"][0] for n, row in res.get("adapt", {}).get("docs", {}).items()}
    _, model = load_model(args, device)
    r = ti.layer0_costs(study, model, study.tests, study.tables, device, chunk_tokens=8192, t_layer0=t0,
                        timer=lambda fn: timer(fn, torch.device(device), warmup=1, reps=3))
    r["windows"] = {n: v[0] for n, v in ws.items()}
    res["layer0"] = r
    for n, row in r["docs"].items():
        log(f"layer0 {n}: {100 * row['equal_fraction']:.3f}% of words recomputed bit for bit, "
            f"{row['bits_per_layer0_value']:.4f} bits per layer-0 value" +
            (f" (T: {row['t_layer0_bits']:.3f}; saving {row['saving_per_value_all_layers']:.4f} bits per value)"
             if "t_layer0_bits" in row else ""))
    del model


def stage_float(args, res, device, out, stage, specs):
    from kv_compare import t_ideas as ti
    ws = windows(Path(args.run) / "captures", args.datasets, args.windows)
    docs = [v[0] for v in ws.values()][: args.float_docs]
    study = study_of(args, docs, device)
    r = res.setdefault(stage, {})
    r.setdefault("specs", {})
    r["windows"] = docs
    for spec in specs:
        if spec.name in r["specs"] and set(r["specs"][spec.name]["bits"]) >= set(docs):
            continue
        t0 = time.time()
        per, info = ti.float_costs(study, spec, study.train, study.tables, study.tests, log=lambda m: None)
        r["specs"][spec.name] = {"note": spec.note, "bits": {n: float(v.mean()) for n, v in per.items()},
                                 "layers": {n: v.tolist() for n, v in per.items()}, "info": info}
        log(f"{stage} {spec.name}: " + ", ".join(
            f"{n} {float(v.mean()) + info[n]['side_bits_per_value']:.4f}" for n, v in per.items())
            + f" ({time.time() - t0:.0f} s)")
        save(res, out)


def stage_gpu(args, res, device, out):
    import torch
    from kv_compare import t_ideas as ti
    from kv_compare.gpu_speed import timer
    from kv_compare.grouped import prepare_groups
    from kv_compare.layerwise import prepare_rank
    dev = torch.device(device)
    if dev.type != "cuda":
        res["gpu"] = {"skipped": "needs a CUDA GPU"}
        log("gpu: skipped (no CUDA)")
        return
    ws = windows(Path(args.run) / "captures", args.datasets, args.windows)
    doc = next(iter(ws.values()))[0]
    study = study_of(args, [doc], device)
    consts = consts_of(args, study, out)
    G = study.W // study.hd
    groups = prepare_groups(study, consts, sorted({2, 4, G}), log=log)
    rank = prepare_rank(study, consts, 256, log)
    rows = ti.mode_rows(study, consts, rank, groups, study.tests[0], args.modes, dev, timer, log=log)
    res["gpu"] = {"doc": doc, "gpu": torch.cuda.get_device_name(dev), "rows": rows,
                  "allocation": [a for a in (ti.allocate(rows, g) for g in LINKS) if a]}
    for a in res["gpu"]["allocation"]:
        p = a["pipelined"]
        log(f"allocation at {a['gbps']} Gb/s: best single {p['best_single']} {p['single_ms']:.2f} ms, per-layer mix "
            f"{p['mix_ms']:.2f} ms")


# ---------------------------------------------------------------------------------------------- report
def ds_of(name):
    m = re.match(r"ds_(.+)_r\d+$", name)
    return m.group(1) if m else name


def by_dataset(values):
    """{doc: v} -> {dataset: mean v}."""
    acc = {}
    for n, v in values.items():
        acc.setdefault(ds_of(n), []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def report(res, out):
    lines = ["# T ideas: measured", ""]
    env = res.get("env", {})
    lines += [f"Run folder `{env.get('run')}`, calibration {env.get('train_set')}, commit `{env.get('commit')}`, "
              f"{env.get('gpu') or 'CPU'}. Bits per value are exact costs (-log2 P of every BF16 word); side bits "
              "are added where a variant sends something. Decision rules: docs/t_ideas.md.", ""]
    verdicts = []
    a = res.get("adapt")
    if a and a.get("docs"):
        docs = a["docs"]
        keys = [k for k in DATASETS if any(ds_of(n) == k for n in docs)] + \
               sorted({ds_of(n) for n in docs} - set(DATASETS))
        names = list(a["variants"]) + ["oracle", "oracle-in-sample"]
        orc = {"oracle": "oracle, cross-fitted: one-step predictor fitted on the other half of the window (not sendable)",
               "oracle-in-sample": "oracle, in-sample: fitted on the whole window (optimistic bound)"}
        net = {v: by_dataset({n: r[v]["bits"] + r[v]["side"] for n, r in docs.items() if v in r}) for v in names}
        side = {v: by_dataset({n: r[v]["side"] for n, r in docs.items() if v in r}) for v in names}
        lines += ["## 1. Per-request adaptation (T's int8 predictor)", "",
                  "| Variant | " + " | ".join(NAMES.get(k, k) for k in keys) + " | side bits |",
                  "| --- |" + " --- |" * (len(keys) + 1)]
        for v in names:
            label = a["variants"].get(v, orc.get(v, v))
            lines.append(f"| {label} | " + " | ".join(fmt(net[v].get(k)) for k in keys) +
                         f" | {fmt(max(side[v].values()) if side[v] else None, 4)} |")
        if a.get("float"):
            fl = by_dataset(a["float"])
            lines.append("| float one-step (T before 8-bit rounding) | " + " | ".join(fmt(fl.get(k)) for k in keys)
                         + " | 0 |")
        lines.append("")
        gap = {k: net["T"][k] - net["oracle"][k] for k in keys}
        sendable = [v for v in names if v not in ("T", "oracle", "oracle-in-sample")]
        best = {k: min(sendable, key=lambda v: net[v][k]) for k in keys}
        lines += ["| | " + " | ".join(NAMES.get(k, k) for k in keys) + " |", "| --- |" + " --- |" * len(keys),
                  "| T minus cross-fitted oracle (room) | " + " | ".join(fmt(gap[k]) for k in keys) + " |",
                  "| best sendable saving | " + " | ".join(fmt(net["T"][k] - net[best[k]][k]) for k in keys) + " |",
                  "| best sendable variant | " + " | ".join(best[k] for k in keys) + " |", ""]
        saving = max(net["T"][k] - net[best[k]][k] for k in keys)
        tbits = sum(net["T"].values()) / len(keys)
        verdicts.append(("1 per-request adaptation", saving >= 0.01 * tbits,
                         f"best sendable saving {saving:.3f} bits/value (rule: >= 1% of T, {0.01 * tbits:.3f})"))
    l0 = res.get("layer0")
    if l0 and l0.get("docs"):
        lines += ["## 2. Layer 0 recomputed on the decode side", "",
                  "| Window | words recomputed bit for bit | bits per layer-0 value | T's layer 0 | saving per value "
                  "(all layers) | recompute time |", "| --- | --- | --- | --- | --- | --- |"]
        for n, r in l0["docs"].items():
            lines.append(f"| {n} | {100 * r['equal_fraction']:.3f}% | {r['bits_per_layer0_value']:.4f} | "
                         f"{fmt(r.get('t_layer0_bits'))} | {fmt(r.get('saving_per_value_all_layers'), 4)} | "
                         f"{fmt(r.get('recompute_ms'), 1)} ms |")
        lines.append("")
        sv = [r["saving_per_value_all_layers"] for r in l0["docs"].values() if "saving_per_value_all_layers" in r]
        eqf = min(r["equal_fraction"] for r in l0["docs"].values())
        if sv:
            verdicts.append(("2 layer 0 on the decode side", min(sv) >= 0.1 and eqf >= 0.99,
                             f"saving {min(sv):.3f} bits/value, {100 * eqf:.2f}% bit-identical "
                             "(rule: >= 0.1 bits/value and >= 99%)"))
    for stage, title, rule in (("anchor", "## 4. Anchors (float one-step predictor)", 0.1),
                               ("classes", "## 5. Token classes (float one-step predictor)", 0.05)):
        f = res.get(stage)
        if not (f and f.get("specs")):
            continue
        docs = f["windows"]
        base = res.get("anchor", {}).get("specs", {}).get("one-step (float)", {}).get("bits", {}) or \
            (res.get("adapt", {}).get("float", {}))
        lines += [title, "", "| Variant | " + " | ".join(docs) + " | saving vs one-step | waves | note |",
                  "| --- |" + " --- |" * (len(docs) + 3)]
        best = 0.0
        for name, s in f["specs"].items():
            vals = {n: s["bits"][n] + s["info"][n]["side_bits_per_value"] for n in docs if n in s["bits"]}
            sav = [base[n] - vals[n] for n in vals if n in base]
            sv = sum(sav) / len(sav) if sav else None
            wv = {s["info"][n].get("waves") for n in docs if n in s["info"]} - {None}
            if sv is not None and name != "one-step (float)" and max(wv, default=1) <= 2:
                best = max(best, sv)                  # many-wave forms are shown, not counted
            lines.append(f"| {name} | " + " | ".join(fmt(vals.get(n)) for n in docs) + f" | {fmt(sv)} | "
                         f"{'/'.join(str(x) for x in sorted(wv)) if wv else 1} | {s['note']} |")
        lines.append("")
        verdicts.append((f"{'4 anchors' if stage == 'anchor' else '5 token classes'}", best >= rule,
                         f"best saving {best:.3f} bits/value (rule: >= {rule}; anchors: 2 waves)"))
    g = res.get("gpu")
    if g and g.get("rows"):
        lines += [f"## 3. Decoder profile and per-layer modes ({g.get('gpu')}, {g['doc']})", "",
                  "| Mode | bits/value | decode | products alone | coding alone | coding floor | coding / floor |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        ratios = []
        for r in g["rows"]:
            if r.get("error"):
                lines.append(f"| {r['mode']} | not measured: {r['error'][:80]} | | | | | |")
                continue
            rt = r["coding_only_ms"] / r["coding_floor_ms"] if r.get("coding_floor_ms") else None
            ratios.append(rt or 0)
            lines.append(f"| {r['mode']} | {r['bits_per_value']:.4f}{'' if r['bit_exact'] else ' (NOT exact)'} | "
                         f"{r['decode_ms']:.1f} ms | {r['products_only_ms']:.1f} ms | {r['coding_only_ms']:.1f} ms | "
                         f"{fmt(r.get('coding_floor_ms'), 1)} ms | {fmt(rt, 2)} |")
        dense = next((r for r in g["rows"] if r["mode"] == "dense" and "exact_model_bits_per_value" in r), None)
        ad = res.get("adapt", {}).get("docs", {}).get(g["doc"])
        if dense and ad:
            lines += ["", f"Check: T's exact-model bits on {g['doc']} from the kernel's own features and priors "
                          f"{dense['exact_model_bits_per_value']:.4f}, from the torch reference of the adapt stage "
                          f"{ad['T']['bits']:.4f} (the cut to a multiple of 256 tokens aside)."]
        lines += ["", "**Per-layer mode allocation** (offline, from the per-layer bits and times above; pipelined: "
                  "max(link, decode) for the whole cache)", "",
                  "| Link | best single mode | per-layer mix | gain | modes used |", "| --- | --- | --- | --- | --- |"]
        gains = []
        for al in g.get("allocation", []):
            p = al["pipelined"]
            gain = 1 - p["mix_ms"] / p["single_ms"]
            gains.append((al["gbps"], gain))
            used = {}
            for m in p["mix_modes"]:
                used[m] = used.get(m, 0) + 1
            lines.append(f"| {al['gbps']} Gb/s | {p['best_single']} {p['single_ms']:.2f} ms | {p['mix_ms']:.2f} ms | "
                         f"{100 * gain:.1f}% | " + ", ".join(f"{m} x{c}" for m, c in used.items()) + " |")
        lines.append("")
        mid = [gn for b, gn in gains if 3 <= b <= 100]
        if mid:
            verdicts.append(("2b per-layer mode allocation", max(mid) >= 0.05,
                             f"best gain {100 * max(mid):.1f}% at 3-100 Gb/s (rule: >= 5%)"))
        if ratios:
            verdicts.append(("3 leaner coding kernel (table-free decode)", min(ratios) >= 2,
                             f"coding / memory floor {min(ratios):.2f}-{max(ratios):.2f} (rule: >= 2)"))
    if verdicts:
        lines += ["## Verdicts", "", "| Idea | worth building? | evidence |", "| --- | --- | --- |"]
        lines += [f"| {n} | {'yes' if ok else 'no'} | {e} |" for n, ok, e in verdicts]
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    return lines


def save(res, out):
    (out / "t_ideas.json").write_text(json.dumps(res, indent=1))


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder with captures/ (scripts/run_a5000.sh)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--stages", default="capture,adapt,layer0,anchor,classes,gpu,report")
    ap.add_argument("--redo", default="", help="stages to measure again")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-set", default="wikitext", choices=sorted(TRAIN_SETS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--windows", type=int, default=2, help="windows per dataset (adapt; others use the first)")
    ap.add_argument("--float-docs", type=int, default=5, help="anchor and classes: windows (one per dataset)")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--texts", default=None, help="dataset texts (default: the run folder's parent/dataset_texts)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-layers", type=int, default=None)
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args()
    args.datasets = [d for d in args.datasets.split(",") if d]
    args.modes = [m for m in args.modes.split(",") if m]
    stages = [s for s in args.stages.split(",") if s]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}; stages are {', '.join(STAGES)}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "console.log"
    path = out / "t_ideas.json"
    res = json.loads(path.read_text()) if path.is_file() else {}
    for s in [s for s in args.redo.split(",") if s]:
        res.pop(s, None)
    import torch
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res["env"] = {"run": str(args.run), "train_set": args.train_set, "commit": commit, "torch": torch.__version__,
                  "python": platform.python_version(),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != "cpu" else None}
    for s in stages:
        if s not in ("report", "capture") and s in res and not res[s].get("partial"):
            log(f"{s}: kept from {path.name}")
            continue
        t0 = time.time()
        log(f"== {s}")
        if s != "report":                  # until the stage finishes; a rerun continues where it stopped
            res.setdefault(s, {})["partial"] = True
        if s == "capture":
            stage_capture(args, res, args.device)
        elif s == "adapt":
            stage_adapt(args, res, args.device, out)
        elif s == "layer0":
            stage_layer0(args, res, args.device, out)
        elif s == "anchor":
            from kv_compare.t_ideas import ANCHORS
            stage_float(args, res, args.device, out, "anchor", ANCHORS)
        elif s == "classes":
            from kv_compare.t_ideas import CLASSES
            stage_float(args, res, args.device, out, "classes", CLASSES)
        elif s == "gpu":
            stage_gpu(args, res, args.device, out)
        elif s == "report":
            for line in report(res, out):
                print(line)
        if s in res:
            res[s].pop("partial", None)
        save(res, out)
        log(f"{s} done in {time.time() - t0:.0f} s")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
