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
  layer0x  layer 0 recomputed the lean way (embedding, norm, K/V projections, RoPE) under other prefill shapes
           (chunks of 2048 and 512, behind 1 or 2048 tokens of another request): bit identity, coding cost, time
  calib    T's constants from other calibration sets (mixed, per domain: code, math, exam, news; texts of
           scripts/fetch_calib_texts.py), on every dataset window, and a per-request pick among the sets
  layer0m  layer 0 under simulated vLLM prefill steps (own tokens with another request's in one product), recomputed
           in one pass, in the same chunks, or padded to the sender's step sizes
  calib_loo  each dataset with its own source left out of the calibration (mixed-all refitted without it; the pick
           without its own set)
  t2       T2 (kv_compare/t2.py, separate from T): round trips with the receiver's layer 0 and the picked set, bits
           against T, times; every mode on one window
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STAGES = ("capture", "adapt", "layer0", "layer0x", "layer0m", "anchor", "classes", "gpu", "calib", "calib_loo", "t2",
          "report")
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


def stage_layer0x(args, res, device, out):
    """Layer 0 recomputed the lean way (no attention, no MLP) under other prefill shapes, and its time."""
    import torch
    from kv_compare import t_ideas as ti
    from kv_compare.gpu_speed import timer
    ws = windows(Path(args.run) / "captures", args.datasets, args.windows)
    docs = [v[0] for v in ws.values()]
    study = study_of(args, docs, device)
    t0 = {n: row["T"]["layers"][0] for n, row in res.get("adapt", {}).get("docs", {}).items()}
    _, model = load_model(args, device)
    dev = torch.device(device)
    r = ti.layer0_robust(study, model, study.tests, study.tables, dev, t_layer0=t0,
                         timer=lambda fn: timer(fn, dev, warmup=2, reps=5))
    res["layer0x"] = r
    for v in r["variants"]:
        rows = [row[v] for row in r["docs"].values()]
        log(f"layer0x {v}: keys {100 * min(x['keys_equal'] for x in rows):.3f}% / values "
            f"{100 * min(x['values_equal'] for x in rows):.3f}% bit for bit (worst window), at most "
            f"{max(x['max_steps'] for x in rows)} BF16 steps apart, "
            f"{max(x['bits_per_layer0_value'] for x in rows):.4f} bits per layer-0 value")
    for n, row in r["docs"].items():
        if "lean_ms" in row:
            log(f"layer0x {n}: lean recompute {row['lean_ms']:.2f} ms, whole first layer {row['first_layer_ms']:.2f} ms")
    del model


def cal_texts(args):
    return Path(args.texts) if args.texts else Path(args.run).parent / "dataset_texts"


def capture_calib(args, res, device, sets):
    """Calibration windows still missing: cal_<domain>_<i> (window i of the domain's text), and cal_mixed_<n> (an
    equal share of WikiText-2 train after the wtrain/wtables windows and of each domain after its window n)."""
    import shutil
    from kv_compare import t_ideas as ti
    from kv_compare.capture import bos_prefix, capture_ids
    cap = Path(args.run) / "captures"
    texts = cal_texts(args)
    have = lambda n: (cap / f"{n}_K_bits.npy").is_file()
    wanted = sorted({d for tr, tb, _ in sets.values() for d in tr + [tb]})
    missing = [n for n in wanted if not have(n)]
    if not missing:
        return []
    per = sum((cap / f"wtables_{p}").stat().st_size for p in ("K_bits.npy", "V_bits.npy", "ids.npy")
              if (cap / f"wtables_{p}").is_file())
    free = shutil.disk_usage(cap).free
    if per and free < per * len(missing) + 2 * 2 ** 30:
        raise SystemExit(f"STOP: {len(missing)} calibration windows need about {per * len(missing) / 2 ** 30:.1f} GiB, "
                         f"{free / 2 ** 30:.1f} GiB free in {cap}")
    tok, model = load_model(args, device)
    pre = bos_prefix(tok)
    step = args.tokens - len(pre)
    ids = {}

    def text_ids(name):
        if name not in ids:
            f = texts / (f"cal_{name}.txt" if name in ti.CAL_DOMAINS else f"{name}.txt")
            ids[name] = tok(f.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"] if f.is_file() else None
        return ids[name]

    def mixed(srcs):
        """An equal share of each (source, offset), filled to a window from the first source's continuation."""
        share = step // len(srcs)
        pieces = []
        for src, off in srcs:
            t = text_ids(src)
            if t is None or len(t) < off + 2 * share:
                return None
            pieces += t[off:off + share]
        src, off = srcs[0]
        return pre + pieces + text_ids(src)[off + share:off + share + step - len(pieces)]

    done = []
    for n in missing:
        if n.startswith("cal_mixedx_") or n.startswith("cal_mixed_"):
            held, k = (n[len("cal_mixedx_"):].rsplit("_", 1) if n.startswith("cal_mixedx_")
                       else (None, n.rsplit("_", 1)[1]))
            k = int(k)                               # after window k of every domain
            srcs = [("wikitext2_train", 7 * step)] + [(d, (k + 1) * step) for d in ti.CAL_DOMAINS]
            srcs = [x for x in srcs if x[0] != held and not (held == "wikitext" and x[0] == "wikitext2_train")]
            window = mixed(srcs)
            if window is None:
                log(f"calib: {n} skipped (a source text is missing or short)")
                continue
        else:
            dom, idx = n[4:].rsplit("_", 1)
            i = int(idx)
            t = text_ids(dom)
            if t is None or len(t) < (i + 1) * step:
                log(f"calib: {n} skipped ({'no text' if t is None else f'{len(t)} tokens'}; "
                    f"scripts/fetch_calib_texts.py --output {texts})")
                continue
            window = pre + t[i * step:(i + 1) * step]
        capture_ids(model, window, cap, n, device, chunk_tokens=8192)
        done.append(n)
        log(f"calib: captured {n}")
    del model
    import gc
    import torch
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return done


def stage_calib(args, res, device, out):
    """T's constants from several calibration sets, measured on every dataset window; and a per-request pick."""
    import torch
    from kv_compare import fused
    from kv_compare import t_ideas as ti
    from kv_compare.predict import Study
    ws = windows(Path(args.run) / "captures", args.datasets, args.windows)
    tests = [w for v in ws.values() for w in v]
    sets = ti.cal_sets(args.cal_windows)
    r = res.setdefault("calib", {})
    r.setdefault("sets", {})
    r["captured"] = r.get("captured", []) + capture_calib(args, res, device, sets)
    cap = Path(args.run) / "captures"
    for name, (train, tables, note) in sets.items():
        missing = [d for d in train + [tables] if not (cap / f"{d}_K_bits.npy").is_file()]
        if missing:
            r["sets"][name] = {"note": note, "skipped": f"missing {', '.join(missing)}"}
            log(f"calib {name}: skipped (missing {', '.join(missing)})")
            continue
        row = r["sets"].get(name, {})
        if row.get("docs") and set(row["docs"]) >= set(tests):
            continue
        t0 = time.time()
        study = Study(cap, device, train, tables, tests, args.max_layers, log=lambda m: None)
        if name == "wikitext" and args.train_set == "wikitext":
            consts = consts_of(args, study, out)
        else:
            log(f"calib {name}: fitting T's constants on {len(train)} documents")
            consts = fused.prepare(study, log=lambda m: None)
        row = {"note": note, "train": train, "tables": tables, "docs": {}}
        for d in study.tests:
            per, side = ti.t_costs(study, consts, d, variants=("T", "fwd-bias+spread/ch"))
            row["docs"][d.name] = {"T": per["T"].tolist(), "fwd": per["fwd-bias+spread/ch"].tolist(),
                                   "fwd_side": side["fwd-bias+spread/ch"], "values": study.L * d.tokens * study.W}
        r["sets"][name] = row
        log(f"calib {name}: " + ", ".join(f"{n} {np.mean(v['T']):.4f}" for n, v in row["docs"].items())
            + f" ({time.time() - t0:.0f} s)")
        save(res, out)
        del consts, study
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def stage_layer0m(args, res, device, out):
    """Layer 0 under simulated vLLM prefill steps, recomputed by receivers that know the sender's step sizes."""
    import torch
    from kv_compare import t_ideas as ti
    ws = windows(Path(args.run) / "captures", args.datasets, args.windows)
    docs = [v[0] for v in ws.values()]
    study = study_of(args, docs, device)
    t0 = {n: row["T"]["layers"][0] for n, row in res.get("adapt", {}).get("docs", {}).items()}
    _, model = load_model(args, device)
    r = ti.layer0_matched(study, model, study.tests, study.tables, torch.device(device), t_layer0=t0)
    res["layer0m"] = r
    first = next(iter(r["docs"].values()))
    for key in first:
        xs = [row[key] for row in r["docs"].values()]
        log(f"layer0m {key}: keys {100 * min(x['keys_equal'] for x in xs):.3f}% / values "
            f"{100 * min(x['values_equal'] for x in xs):.3f}% bit for bit, step coder "
            f"{max(x['step_bits'] for x in xs):.4f}, value coder {max(x['value_bits'] for x in xs):.4f} bits per "
            "layer-0 value (worst window)")
    del model


def stage_calib_loo(args, res, device, out):
    """Leave one source out: mixed-all refitted without it, tested on that source's datasets."""
    import torch
    from kv_compare import fused
    from kv_compare import t_ideas as ti
    from kv_compare.predict import Study
    cap = Path(args.run) / "captures"
    ws = windows(cap, args.datasets, args.windows)
    loo = ti.cal_loo_sets(args.cal_windows)
    r = res.setdefault("calib_loo", {})
    r.setdefault("sets", {})
    r["captured"] = r.get("captured", []) + capture_calib(args, res, device, loo)
    for held, (train, tables, note) in loo.items():
        dsname = next(k for k, v in ti.HELD_OUT.items() if v == held)
        tests = ws.get(dsname, [])
        missing = [d for d in train + [tables] if not (cap / f"{d}_K_bits.npy").is_file()]
        if not tests or missing:
            r["sets"][held] = {"note": note, "skipped": f"missing {', '.join(missing)}" if missing else "no windows"}
            log(f"calib_loo {held}: skipped ({r['sets'][held]['skipped']})")
            continue
        if r["sets"].get(held, {}).get("docs"):
            continue
        t0 = time.time()
        study = Study(cap, device, train, tables, tests, args.max_layers, log=lambda m: None)
        consts = fused.prepare(study, log=lambda m: None)
        row = {"note": note, "dataset": dsname, "train": train, "tables": tables, "docs": {}}
        for d in study.tests:
            per, _ = ti.t_costs(study, consts, d, variants=("T",))
            row["docs"][d.name] = {"T": per["T"].tolist()}
        r["sets"][held] = row
        log(f"calib_loo without {held}: " + ", ".join(f"{n} {np.mean(v['T']):.4f}" for n, v in row["docs"].items())
            + f" ({time.time() - t0:.0f} s)")
        save(res, out)
        del consts, study
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def _int16(w):
    import torch
    return (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)


def t2_consts(args, out, cap, name, train, tables, device):
    """T's constants for one calibration set, cached in OUT."""
    import torch
    from kv_compare import fused
    from kv_compare.predict import Study
    study = Study(cap, device, train, tables, [], args.max_layers, log=lambda m: None)
    if name == "wikitext" and args.train_set == "wikitext":
        return consts_of(args, study, out), study
    p = out / f"t2_consts_{name}.pt"
    key = {"run": str(Path(args.run).resolve()), "train": train, "tables": tables, "layers": study.L}
    if p.is_file():
        blob = torch.load(p, map_location=study.device, weights_only=False)
        if blob.get("key") == key:
            return blob["consts"], study
    log(f"t2: fitting T's constants for {name}")
    consts = fused.prepare(study, log=lambda m: None)
    torch.save({"key": key, "consts": consts}, p)
    return consts, study


def stage_t2(args, res, device, out):
    """T2 (kv_compare/t2.py), kept apart from T: on every dataset window the round trip with the receiver's layer 0,
    its bits against T's, the set it picks, and the times; then every mode on one window."""
    import torch
    from kv_compare import t_ideas as ti
    from kv_compare.gpu_speed import timer
    from kv_compare.grouped import prepare_groups
    from kv_compare.layerwise import prepare_rank
    from kv_compare.t2 import Layer0Recomputed, T2
    dev = torch.device(device)
    if dev.type != "cuda":
        res["t2"] = {"skipped": "needs a CUDA GPU"}
        log("t2: skipped (no CUDA)")
        return
    cap = Path(args.run) / "captures"
    ws = windows(cap, args.datasets, args.windows)
    docs = [v[0] for v in ws.values()]
    names = ["wikitext"] + list(ti.CAL_DOMAINS)
    sets = {n: v for n, v in ti.cal_sets(args.cal_windows).items() if n in names and
            all((cap / f"{d}_K_bits.npy").is_file() for d in v[0] + [v[1]])}
    r = {"sets": list(sets), "docs": {}, "modes": [], "partial": True}      # until the stage finishes
    consts, studies = {}, {}
    for n, (train, tables, _) in sets.items():
        consts[n], studies[n] = t2_consts(args, out, cap, n, train, tables, device)
    st = study_of(args, docs, device)
    L, W, C, hd = st.L, st.W, st.C, st.hd
    _, model = load_model(args, device)

    def builders(T, mode="dense", extra=None):
        extra = extra or {}
        return {n: (lambda nl, rec, tokens=None, n=n: ti.mode_codec(consts[n], *(extra.get(n, (None, None))), mode, C,
                                                                    hd, tokens or T,
                                                                    cls=Layer0Recomputed if rec else None, layers=nl))
                for n in sets}

    def doc_words(d, T):
        w = torch.empty(L, T, W, dtype=torch.int16, device=dev)
        for l in range(L):
            w[l] = _int16(d.words(l, dev, rows=T))
        return w

    for d in st.tests:
        T = d.tokens - d.tokens % 256
        n = L * T * W
        words = doc_words(d, T)
        ids = d.ids[:T].tolist()
        l0 = _int16(torch.cat(ti.layer0_kv(model, ids, dev), 1))
        l0_ok = bool(torch.equal(l0, words[0]))
        codec = T2(builders(T))
        t0 = time.perf_counter()
        name, pbits = codec.pick(words)
        torch.cuda.synchronize(dev)
        pick_ms = 1e3 * (time.perf_counter() - t0)
        variants = {}                                 # the sender's cost of the pick: cheaper looks at the request
        for label, kw in (("layers 1-4, all tokens (codecs kept)", {}), ("layers 1-2, all tokens", {"layers": 2}),
                          ("layers 1-4, first 1024 tokens", {"tokens": 1024}),
                          ("layers 1-2, first 1024 tokens", {"layers": 2, "tokens": 1024})):
            if kw.get("tokens", 0) >= T:
                continue
            codec.pick(words, **kw)                   # builds and warms the codecs
            ms = 1e3 * timer(lambda: codec.pick(words, **kw), dev, warmup=0, reps=3)
            variants[label] = {"set": codec.pick(words, **kw)[0], "ms": ms}
        parts, header = codec.encode(words, l0_ok, name)
        c2 = codec.codec(name, l0_ok)
        dec = torch.empty(L, T, W, dtype=torch.int16, device=dev)
        bufs = c2.buffers(T)
        if l0_ok:
            c2.decode(parts["words"], parts["starts"], T, l0, out=dec, bufs=bufs)
        else:
            c2.decode(parts["words"], parts["starts"], T, out=dec, bufs=bufs)
        row = {"set": name, "pick_bits": {k: v / n for k, v in pbits.items()}, "layer0_recomputed": l0_ok,
               "bit_exact": bool(torch.equal(dec, words)), "t2_bits": codec.bits(parts, header) / n,
               "pick_ms": pick_ms, "pick_variants": variants}
        if l0_ok:
            row["t2_decode_ms"] = 1e3 * timer(lambda: c2.decode(parts["words"], parts["starts"], T, l0, out=dec,
                                                                bufs=bufs), dev)
        else:
            row["t2_decode_ms"] = 1e3 * timer(lambda: c2.decode(parts["words"], parts["starts"], T, out=dec,
                                                                bufs=bufs), dev)
        xa, pa = c2.model(words)
        row["t2_encode_ms"] = 1e3 * timer(lambda: c2.encode(words, xa, pa, out=parts["words"]), dev)
        parts = xa = pa = None
        if name != "wikitext":                        # layer 0 alone: T2 with T's constants
            p0, h0 = codec.encode(words, l0_ok, "wikitext")
            row["t2_wikitext_bits"] = codec.bits(p0, h0) / n
            p0 = None
        else:
            row["t2_wikitext_bits"] = row["t2_bits"]
        tc = ti.mode_codec(consts["wikitext"], None, None, "dense", C, hd, T)
        xa, pa = tc.model(words)
        tb, ts = tc.encode(words, xa, pa)
        row["t_bits"] = tc.bits(ts, T) / n
        tbufs = tc.buffers(T)
        row["t_decode_ms"] = 1e3 * timer(lambda: tc.decode(tb, ts, T, out=dec, bufs=tbufs), dev)
        row["recompute_ms"] = 1e3 * timer(lambda: ti.layer0_kv(model, ids, dev), dev)
        tb = ts = xa = pa = tbufs = bufs = dec = words = None
        torch.cuda.empty_cache()
        r["docs"][d.name] = row
        log(f"t2 {d.name}: set {name}, layer 0 {'recomputed' if l0_ok else 'coded'}, exact={row['bit_exact']}, "
            f"T {row['t_bits']:.4f} -> T2 {row['t2_bits']:.4f} bits/value (T2 with T's constants "
            f"{row['t2_wikitext_bits']:.4f}); decode T {row['t_decode_ms']:.1f} ms, T2 {row['t2_decode_ms']:.1f} ms "
            f"+ recompute {row['recompute_ms']:.1f} ms")
        res["t2"] = r
        save(res, out)
    # every mode on one window (HumanEval's if there, to show a set other than WikiText's)
    d = next((x for x in st.tests if x.name.startswith("ds_humaneval")), st.tests[0])
    name = r["docs"][d.name]["set"]
    G = W // hd
    extra = {}
    for n in {"wikitext", name}:
        extra[n] = (prepare_rank(studies[n], consts[n], 256, lambda m: None),
                    prepare_groups(studies[n], consts[n], sorted({2, 4, G}), log=lambda m: None))
    T = d.tokens - d.tokens % 256
    l0 = _int16(torch.cat(ti.layer0_kv(model, d.ids[:T].tolist(), dev), 1))
    model = None                                     # the modes need the memory (rank 256 ran out beside the model)
    for k in [k for k in consts if k not in ("wikitext", name)]:
        consts[k] = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    words = doc_words(d, T)
    l0_ok = bool(torch.equal(l0, words[0]))
    n = L * T * W
    dec = torch.empty(L, T, W, dtype=torch.int16, device=dev)
    for mode in args.modes:
        row = {"mode": mode, "doc": d.name, "set": name}
        try:
            b = builders(T, mode, extra)
            t = b["wikitext"](None, False)
            xa, pa = t.model(words)
            tb, ts = t.encode(words, xa, pa)
            tbufs = t.buffers(T)
            row["t_bits"] = t.bits(ts, T) / n
            row["t_decode_ms"] = 1e3 * timer(lambda: t.decode(tb, ts, T, out=dec, bufs=tbufs), dev)
            tb = ts = xa = pa = tbufs = None
            c2 = b[name](None, l0_ok)
            xa, pa = c2.model(words)
            cb, cs = c2.encode(words, xa, pa)
            bufs = c2.buffers(T)
            args2 = (cb, cs, T, l0) if l0_ok else (cb, cs, T)
            c2.decode(*args2, out=dec, bufs=bufs)
            row["bit_exact"] = bool(torch.equal(dec, words))
            row["t2_bits"] = c2.bits(cs, T) / n
            row["t2_decode_ms"] = 1e3 * timer(lambda: c2.decode(*args2, out=dec, bufs=bufs), dev)
            cb = cs = xa = pa = bufs = None
        except Exception as exc:
            row["error"] = repr(exc)[:300]
        torch.cuda.empty_cache()
        r["modes"].append(row)
        log(f"t2 mode {mode}: " + (row["error"] if "error" in row else
                                   f"exact={row['bit_exact']}, T {row['t_bits']:.4f} ({row['t_decode_ms']:.1f} ms) -> "
                                   f"T2 {row['t2_bits']:.4f} ({row['t2_decode_ms']:.1f} ms)"))
    res["t2"] = r


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
    lx = res.get("layer0x")
    if lx and lx.get("docs"):
        rows = lx["docs"]
        lines += ["## 2c. Layer 0: the lean recompute under other prefill shapes", "",
                  "The receiver computes layer 0 with the embedding, the first layer's input norm, the K and V "
                  "projections and RoPE only. Worst window of " + str(len(rows)) + " (one per dataset); the cost "
                  "codes each captured word against the recomputed one (difference in BF16 steps, table from the "
                  "scales document under the same shape).", "",
                  "| Prefill shape | keys bit for bit | values bit for bit | largest difference | bits per layer-0 "
                  "value | saving per value (all layers) |", "| --- | --- | --- | --- | --- | --- |"]
        worst = None
        for v, note in lx["variants"].items():
            xs = [r[v] for r in rows.values() if v in r]
            sv = [x["saving_per_value_all_layers"] for x in xs if "saving_per_value_all_layers" in x]
            if sv:
                worst = min(sv) if worst is None else min(worst, min(sv))
            lines.append(f"| {v}: {note} | {100 * min(x['keys_equal'] for x in xs):.3f}% | "
                         f"{100 * min(x['values_equal'] for x in xs):.3f}% | {max(x['max_steps'] for x in xs)} steps | "
                         f"{max(x['bits_per_layer0_value'] for x in xs):.4f} | {fmt(min(sv) if sv else None, 4)} |")
        lean = [r["lean_ms"] for r in rows.values() if "lean_ms" in r]
        first = [r["first_layer_ms"] for r in rows.values() if "first_layer_ms" in r]
        dense = next((r for r in res.get("gpu", {}).get("rows", []) if r.get("mode") == "dense" and "decode_ms" in r),
                     None)
        if lean:
            lines += ["", f"Time per window: lean recompute {min(lean):.2f}-{max(lean):.2f} ms, the whole first layer "
                          f"(attention and MLP too) {min(first):.2f}-{max(first):.2f} ms"
                      + (f"; T's dense decode of the cache takes {dense['decode_ms']:.1f} ms." if dense else ".")]
        lines.append("")
        if worst is not None:
            ok = worst >= 0.3 and (not lean or max(lean) <= 10)
            verdicts.append(("2c layer 0 holds under other prefill shapes", ok,
                             f"worst saving {worst:.3f} bits/value over the shapes"
                             + (f", lean recompute up to {max(lean):.1f} ms" if lean else "")
                             + " (rule: >= 0.3 bits/value in every shape and <= 10 ms)"))
    if res.get("layer0m", {}).get("docs"):
        lines += layer0m_report(res["layer0m"], verdicts)
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
    cb = res.get("calib")
    if cb and any(s.get("docs") for s in cb.get("sets", {}).values()):
        lines += calib_report(res, cb, verdicts)
    if res.get("calib_loo", {}).get("sets"):
        lines += loo_report(res, verdicts)
    if res.get("t2", {}).get("docs"):
        lines += t2_report(res["t2"], verdicts)
    if verdicts:
        lines += ["## Verdicts", "", "| Idea | worth building? | evidence |", "| --- | --- | --- |"]
        lines += [f"| {n} | {'yes' if ok else 'no'} | {e} |" for n, ok, e in verdicts]
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    return lines


def calib_report(res, cb, verdicts):
    """Section 6: T's bits per constant set with layer 0 recomputed, the per-request picks, and the room."""
    import math
    from kv_compare.t_ideas import CAL_DOMAINS, pick_sets
    sets = {n: s for n, s in cb["sets"].items() if s.get("docs")}
    first = next(iter(sets.values()))
    docs = sorted(set.intersection(*(set(s["docs"]) for s in sets.values())))
    keys = [k for k in DATASETS if any(ds_of(n) == k for n in docs)] + sorted({ds_of(n) for n in docs} - set(DATASETS))
    L = len(first["docs"][docs[0]]["T"])
    l0 = 0.0
    if res.get("layer0x", {}).get("docs"):
        l0 = max(r["same"]["bits_per_layer0_value"] for r in res["layer0x"]["docs"].values() if "same" in r)
    elif res.get("layer0", {}).get("docs"):
        l0 = max(r["bits_per_layer0_value"] for r in res["layer0"]["docs"].values())
    values = {d: first["docs"][d]["values"] for d in docs}

    def with_l0(v):
        v = np.asarray(v, dtype=np.float64)
        return float((v[1:].sum() + l0) / L)

    def mean_by(vals):
        m = by_dataset({d: vals[d] for d in docs if d in vals})
        return m, (sum(m.values()) / len(m) if m else None)

    dom = [n for n in ("wikitext",) + tuple(CAL_DOMAINS) if n in sets]
    early = list(range(1, min(5, L)))
    out = ["## 6. Calibration breadth: T's constants from other text", "",
           f"Every row is T (its int8 predictor) with layer 0 recomputed by the receiver (layer 0 at {l0:.4f} bits "
           "per value), in bits per value, mean of the windows of each dataset. Calibration texts are disjoint from "
           "the test windows (scripts/fetch_calib_texts.py). A pick sends the chosen set's number "
           "(log2 of the number of sets per request, counted).", ""]
    verdict_rows = {}
    for title, key, side in (("**T with layer 0 recomputed**", "T", None),
                             ("**... and the request's own bias and spread (sent, counted)**", "fwd", "fwd_side")):
        costs = {n: {d: s["docs"][d][key] for d in s["docs"]} for n, s in sets.items()}
        extra = {d: (first["docs"][d][side] if side else 0.0) for d in docs}
        rows = []
        for n, s in sets.items():
            rows.append((f"{n}: {s['note']}", {d: with_l0(costs[n][d]) + extra[d] for d in docs}, None))
        for label, cands, layers in (("pick per request: WikiText or a domain set, chosen on layers 1-4", dom, early),
                                     ("pick per request: WikiText or a domain set, chosen on the whole cache", dom,
                                      slice(1, None)),
                                     ("pick per request: any set, chosen on layers 1-4", list(sets), early)):
            if len(cands) < 2:
                continue
            p = pick_sets(costs, cands, layers, cost=with_l0,
                          side_bits=lambda d, k=len(cands): extra[d] + math.log2(k) / values[d])
            rows.append((label, {d: b for d, (_, b) in p.items()}, {d: c for d, (c, _) in p.items()}))
        out += [title, "", "| Constants | " + " | ".join(NAMES.get(k, k) for k in keys) + " | mean |",
                "| --- |" + " --- |" * (len(keys) + 1)]
        for label, vals, chosen in rows:
            m, avg = mean_by(vals)
            out.append(f"| {label} | " + " | ".join(fmt(m.get(k)) for k in keys) + f" | {fmt(avg)} |")
            verdict_rows.setdefault(key, {})[label] = (m, avg)
            if chosen and "layers 1-4" in label and "WikiText or" in label:
                picks = {}
                for d, c in chosen.items():
                    picks.setdefault(ds_of(d), []).append(c)
                out.append("| (sets chosen) | " + " | ".join("/".join(picks.get(k, [])) for k in keys) + " | |")
        if key == "T" and res.get("adapt", {}).get("docs"):
            orc = {d: with_l0(r["oracle"]["layers"]) for d, r in res["adapt"]["docs"].items()
                   if d in docs and "layers" in r.get("oracle", {})}
            if orc:
                m, avg = mean_by(orc)
                out.append("| room: one-step predictor fitted on the other half of the window (not sendable) | "
                           + " | ".join(fmt(m.get(k)) for k in keys) + f" | {fmt(avg)} |")
            today = {d: float(np.mean(r["T"]["layers"])) for d, r in res["adapt"]["docs"].items() if d in docs}
            m, avg = mean_by(today)
            out.append("| today: T as it is, layer 0 coded | " + " | ".join(fmt(m.get(k)) for k in keys)
                       + f" | {fmt(avg)} |")
        out.append("")
    base = verdict_rows["T"].get(next(lbl for lbl in verdict_rows["T"] if lbl.startswith("wikitext:")))
    options = {lbl: v for lbl, v in verdict_rows["T"].items() if not lbl.startswith(("wikitext:",))
               and (lbl.startswith(("mixed", "pick")) and "whole cache" not in lbl)}
    if base and options:
        best = min(options, key=lambda lbl: options[lbl][1])
        sav = base[1] - options[best][1]
        wiki = options[best][0].get("wikitext2", 0) - base[0].get("wikitext2", 0)
        verdicts.append(("6 calibration breadth", sav >= 0.01 * base[1],
                         f"best practical option ({best.split(':')[0]}) saves {sav:.3f} bits/value over the five "
                         f"datasets, WikiText {wiki:+.3f} (rule: >= 1% of T, {0.01 * base[1]:.3f})"))
    return out


def layer0m_report(lm, verdicts):
    """Section 2d: layer 0 under simulated prefill steps."""
    docs = lm["docs"]
    first = next(iter(docs.values()))
    out = ["## 2d. Layer 0 when the receiver knows the sender's prefill steps", "",
           "The sender computes layer 0 in vLLM-like steps (its own tokens with another request's tokens in the same "
           "matrix product, at most 2,048 rows); the receiver recomputes it as said. Worst window of "
           f"{len(docs)}. Bits: the better of the step coder and the value coder (a flag, then a logistic around the "
           "recomputed value), tables from the scales document under the same pair.", "",
           "| Sender's steps | receiver | keys bit for bit | values bit for bit | step coder | value coder | saving per "
           "value (all layers) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    padded = []
    for key in first:
        xs = [row[key] for row in docs.values()]
        sch, rec = key.split(" | ")
        sv = [x["saving_per_value_all_layers"] for x in xs if "saving_per_value_all_layers" in x]
        eq = min(min(x["keys_equal"], x["values_equal"]) for x in xs)
        if rec == "padded to the step":
            padded.append(eq)
        out.append(f"| {sch} | {rec} | {100 * min(x['keys_equal'] for x in xs):.3f}% | "
                   f"{100 * min(x['values_equal'] for x in xs):.3f}% | {max(x['step_bits'] for x in xs):.4f} | "
                   f"{max(x['value_bits'] for x in xs):.4f} | {fmt(min(sv) if sv else None, 4)} |")
    out.append("")
    if padded:
        verdicts.append(("2d layer 0 exact when the receiver pads to the sender's steps", min(padded) == 1.0,
                         f"worst share bit for bit with padding {100 * min(padded):.3f}% (rule: 100% in every "
                         "schedule)"))
    return out


def loo_report(res, verdicts):
    """Section 6b: each dataset with its own source left out of the calibration."""
    from kv_compare.t_ideas import CAL_DOMAINS, HELD_OUT, pick_sets
    cb, lo = res.get("calib", {}).get("sets", {}), res["calib_loo"]["sets"]
    L = None
    l0 = 0.0
    if res.get("layer0x", {}).get("docs"):
        l0 = max(r["same"]["bits_per_layer0_value"] for r in res["layer0x"]["docs"].values() if "same" in r)

    def with_l0(v):
        v = np.asarray(v, dtype=np.float64)
        return float((v[1:].sum() + l0) / len(v))

    rows = {"T's constants (WikiText-2 train)": {}, "its own source's set (in-domain, from section 6)": {},
            "mixed-all, which contains its source (section 6)": {}, "mixed-all without its source": {},
            "pick per request, its own source's set left out (chosen on layers 1-4)": {},
            "pick per request with every set (section 6)": {}}
    picked = {}
    for ds, held in HELD_OUT.items():
        s = lo.get(held, {})
        if not s.get("docs"):
            continue
        for d, v in s["docs"].items():
            L = len(v["T"])
            costs = {"mixed-x": {d: v["T"]}}
            for n in ("wikitext",) + tuple(CAL_DOMAINS) + ("mixed-all",):
                if cb.get(n, {}).get("docs", {}).get(d):
                    costs[n] = {d: cb[n]["docs"][d]["T"]}
            if "wikitext" in costs:
                rows["T's constants (WikiText-2 train)"][d] = with_l0(costs["wikitext"][d])
            if held in costs:
                rows["its own source's set (in-domain, from section 6)"][d] = with_l0(costs[held][d])
            if "mixed-all" in costs:
                rows["mixed-all, which contains its source (section 6)"][d] = with_l0(costs["mixed-all"][d])
            rows["mixed-all without its source"][d] = with_l0(v["T"])
            cands = [n for n in costs if n not in (held, "mixed-all")]
            early = list(range(1, min(5, L)))
            p = pick_sets(costs, cands, early, cost=with_l0)
            rows["pick per request, its own source's set left out (chosen on layers 1-4)"][d] = p[d][1]
            picked.setdefault(ds, []).append(p[d][0].replace("mixed-x", "mixed without it"))
            allc = [n for n in costs if n != "mixed-x"]
            if len(allc) > 1:
                rows["pick per request with every set (section 6)"][d] = pick_sets(costs, allc, early, cost=with_l0)[d][1]
    if not any(rows.values()):
        return []
    keys = [k for k in DATASETS if any(ds_of(d) == k for r in rows.values() for d in r)]
    out = ["## 6b. Calibration with the test's own source left out", "",
           "Each dataset is coded with constants that never saw its kind of text: mixed-all refitted without its "
           "source (WikiText-2 for WikiText-2, MBPP for HumanEval, GSM8K train for GSM8K, MMLU dev for MMLU, PTB "
           "train for PTB; its scales document leaves the source out too), and the per-request pick without the "
           "source's own set. Layer 0 recomputed; bits per value.", "",
           "| Constants | " + " | ".join(NAMES.get(k, k) for k in keys) + " | mean |", "| --- |" + " --- |" * (len(keys) + 1)]
    means = {}
    for label, vals in rows.items():
        m = by_dataset(vals)
        avg = sum(m.values()) / len(m) if m else None
        means[label] = (m, avg)
        out.append(f"| {label} | " + " | ".join(fmt(m.get(k)) for k in keys) + f" | {fmt(avg)} |")
        if label.startswith("pick per request, its own"):
            out.append("| (sets chosen) | " + " | ".join("/".join(picked.get(k, [])) for k in keys) + " | |")
    out.append("")
    base = means["T's constants (WikiText-2 train)"]
    honest = min(("mixed-all without its source", "pick per request, its own source's set left out (chosen on layers 1-4)"),
                 key=lambda lbl: means[lbl][1] if means[lbl][1] is not None else 1e9)
    if base[1] is not None and means[honest][1] is not None:
        sav = base[1] - means[honest][1]
        verdicts.append(("6b calibration breadth on unseen text", sav >= 0.01 * base[1],
                         f"{honest.split(' (')[0]} saves {sav:.3f} bits/value over the datasets (rule: >= 1% of T, "
                         f"{0.01 * base[1]:.3f})"))
    return out


def t2_report(t2, verdicts):
    """Section 7: T2, the separate codec, measured."""
    out = ["## 7. T2: T with layer 0 recomputed and a constant set per request (kv_compare/t2.py)", "",
           "T2 is a separate codec on T's kernels; T is unchanged. Dense mode, one window per dataset, cut to a "
           "multiple of 256 tokens. Bits per value of the real streams (data, states, starts). T: WikiText-2 "
           "constants, layer 0 coded. Decode: the codec's GPU time; T2 adds the receiver's layer-0 recompute.", "",
           "| Window | set picked | layer 0 | bit-exact | T | T2, T's constants | T2 | saving | decode T | decode T2 | "
           "recompute |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    ok, savings = True, []
    for d, r in t2["docs"].items():
        ok = ok and r["bit_exact"] and r["t2_bits"] < r["t_bits"]
        savings.append(1 - r["t2_bits"] / r["t_bits"])
        out.append(f"| {d} | {r['set']} | {'recomputed' if r['layer0_recomputed'] else 'coded'} | "
                   f"{'yes' if r['bit_exact'] else 'NO'} | {r['t_bits']:.4f} | {r['t2_wikitext_bits']:.4f} | "
                   f"{r['t2_bits']:.4f} | {100 * savings[-1]:.1f}% | {r['t_decode_ms']:.1f} ms | "
                   f"{r['t2_decode_ms']:.1f} ms | {r['recompute_ms']:.1f} ms |")
    out.append("")
    pv = [r.get("pick_variants", {}) for r in t2["docs"].values()]
    labels = [lbl for lbl in (pv[0] if pv else {})]
    if labels:
        out += ["The sender's pick (GPU time per request; the decision against the pick on layers 1-4 of all tokens):",
                "", "| Pick | time | same set as the full pick |", "| --- | --- | --- |"]
        for lbl in labels:
            xs = [(v[lbl], r["set"]) for v, r in zip(pv, t2["docs"].values()) if lbl in v]
            same = sum(x["set"] == full for x, full in xs)
            out.append(f"| {lbl} | {np.mean([x['ms'] for x, _ in xs]):.1f} ms | {same} of {len(xs)} windows |")
        out.append("")
    if t2.get("modes"):
        out += [f"Every mode on {t2['modes'][0]['doc']} (T2 with its picked set {t2['modes'][0]['set']}):", "",
                "| Mode | bit-exact | T | T2 | saving | decode T | decode T2 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for m in t2["modes"]:
            if "error" in m:
                ok = False
                out.append(f"| {m['mode']} | error: {m['error'][:80]} | | | | | |")
                continue
            ok = ok and m["bit_exact"]
            out.append(f"| {m['mode']} | {'yes' if m['bit_exact'] else 'NO'} | {m['t_bits']:.4f} | {m['t2_bits']:.4f} | "
                       f"{100 * (1 - m['t2_bits'] / m['t_bits']):.1f}% | {m['t_decode_ms']:.1f} ms | "
                       f"{m['t2_decode_ms']:.1f} ms |")
        out.append("")
    if savings:
        verdicts.append(("7 T2 works", ok, f"every round trip bit-exact and smaller than T: {'yes' if ok else 'NO'}; "
                         f"saving {100 * min(savings):.1f}-{100 * max(savings):.1f}% of T's bytes (dense)"))
    return out


def save(res, out):
    (out / "t_ideas.json").write_text(json.dumps(res, indent=1))


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder with captures/ (scripts/run_a5000.sh)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--stages", default="capture,adapt,layer0,layer0x,layer0m,anchor,classes,gpu,calib,calib_loo,t2,"
                                        "report")
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
    ap.add_argument("--cal-windows", type=int, default=3, help="calib: calibration windows per domain")
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
        elif s == "layer0x":
            stage_layer0x(args, res, args.device, out)
        elif s == "calib":
            stage_calib(args, res, args.device, out)
        elif s == "layer0m":
            stage_layer0m(args, res, args.device, out)
        elif s == "calib_loo":
            stage_calib_loo(args, res, args.device, out)
        elif s == "t2":
            stage_t2(args, res, args.device, out)
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
