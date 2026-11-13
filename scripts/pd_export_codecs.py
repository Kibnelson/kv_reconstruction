#!/usr/bin/env python3
"""Export the transfer codecs' constants for the vLLM PD runs (kv_compare/pd/export.py), from a run folder of
scripts/run_a5000.sh (its calcap stage: the WikiText-2 train calibration captures). Runs in the codec environment
($KV_HOME/venv) on the GPU; the vLLM environment only loads the file.

  python scripts/pd_export_codecs.py --run $KV_HOME/runs/llama32_3b --output $KV_HOME/runs/llama32_3b/pd_codecs.pt \
      --splitzip-source $KV_HOME/sources/splitzip [--t2-sets wikitext,code,math,exam,news --texts DIR]

--t2-sets adds T2's constant sets (bundle["t2"]): T's constants, rank-256 and head-group predictors fitted on each
calibration set of kv_compare.t_ideas.cal_sets (WikiText-2 train, and MBPP, GSM8K train, MMLU dev and validation,
PTB train: scripts/fetch_calib_texts.py). Missing calibration windows (cal_<domain>_<i>, about 1 GB each for the
3B) are captured first with the run's model (scripts/t_ideas.py's capture_calib).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder with captures/ (calcap stage)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-set", default="wikitext", help="calibration set of bench_llama_compare (wikitext)")
    ap.add_argument("--groups", default="2,4,16")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--max-positions", type=int, default=32768)
    ap.add_argument("--splitzip-source", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--t2-sets", default="", help="T2's constant sets, e.g. wikitext,code,math,exam,news")
    ap.add_argument("--texts", default=None, help="folder with the calibration texts (cal_<domain>.txt)")
    ap.add_argument("--cal-windows", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=8192)
    a = ap.parse_args()
    import torch
    from bench_llama_compare import TRAIN_SETS
    from kv_compare import fused, layerwise
    from kv_compare.grouped import prepare_groups
    from kv_compare.more_baselines import _exp_hist, adacode_best_bits
    from kv_compare.pd.export import make_bundle
    from kv_compare.predict import Doc, Study, load_meta
    run = Path(a.run)
    train, tables = TRAIN_SETS[a.train_set]
    dev = torch.device(a.device)
    study = Study(run / "captures", a.device, train, tables, [], None, log=print)
    consts = fused.prepare(study, log=print)
    rc = layerwise.prepare_rank(study, consts, a.rank, print) if a.rank else None
    groups = prepare_groups(study, consts, [int(g) for g in a.groups.split(",") if g], print)
    cal = Doc(run / "captures", "wtrain0" if (run / "captures" / "wtrain0_K_bits.npy").is_file() else train[0])
    w = torch.empty(study.L, cal.tokens, study.W, dtype=torch.int16, device=dev)
    for l in range(study.L):
        x = cal.words(l, dev)
        w[l] = (x - 65536 * (x >= 32768).to(torch.int32)).to(torch.int16)
    sz = None
    if a.splitzip_source:
        from kv_compare.mixed import load_splitzip
        sz, _ = load_splitzip(a.splitzip_source, dev)
        cov = sz.calibrate(w.view(torch.bfloat16))
        print(f"SplitZip code book from {cal.name}: top-16 coverage {100 * cov:.2f}%")
    C = study.C
    ada = [[adacode_best_bits(_exp_hist(w[l, :, part]).cpu().tolist()) for part in (slice(0, C), slice(C, None))]
           for l in range(study.L)]
    meta = load_meta(run / "captures")
    b = make_bundle(study, consts, rc, groups, a.max_positions, splitzip=sz, adacode=ada,
                    meta={"model": meta.get("model"), "revision": meta.get("revision"), "train_set": a.train_set,
                          "calibration": [d.name for d in study.train], "splitzip_book_from": cal.name})
    names = [n for n in a.t2_sets.split(",") if n]
    if names:
        b["t2"] = t2_sets(a, run, names, b)
        b["meta"]["t2_sets"] = names
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(b, out)
    print(json.dumps({"written": str(out), **b["meta"]}, indent=1))


def t2_sets(a, run, names, main):
    """{"order", "sets": {name: {consts, rank, groups}}, ...} for T2; WikiText reuses the main constants."""
    import types
    import torch
    import t_ideas as drv
    from kv_compare import fused, layerwise
    from kv_compare import t_ideas as ti
    from kv_compare.grouped import prepare_groups
    from kv_compare.pd.export import _cpu
    from kv_compare.predict import Study
    every = ti.cal_sets(a.cal_windows)
    unknown = [n for n in names if n not in every]
    if unknown:
        raise SystemExit(f"unknown T2 sets {unknown}; known: {sorted(every)}")
    sets = {n: every[n] for n in names}
    args = types.SimpleNamespace(run=str(run), texts=a.texts, tokens=a.tokens, model=None, revision=None)
    got = drv.capture_calib(args, {}, a.device, {n: v for n, v in sets.items() if n != "wikitext"})
    if got:
        print(f"T2: captured {len(got)} calibration windows")
    cap = run / "captures"
    out = {"order": names, "sets": {}, "pick_tokens": 1024, "pick_layers": [1, 2], "cal_windows": a.cal_windows}
    gl = [int(g) for g in a.groups.split(",") if g]
    for n in names:
        train, tables, note = sets[n]
        missing = [d for d in train + [tables] if not (cap / f"{d}_K_bits.npy").is_file()]
        if missing:
            raise SystemExit(f"T2 set {n}: calibration windows {missing} missing in {cap}")
        if n == "wikitext" and a.train_set == "wikitext":
            out["sets"][n] = {"consts": main["consts"], "rank": main["rank"], "groups": main["groups"], "note": note,
                              "train": train, "tables": tables}   # the bundle's own tensors: saved and loaded once
            print(f"T2 set {n}: the main constants")
            continue
        else:
            st = Study(cap, a.device, train, tables, [], None, log=lambda m: None)
            c = fused.prepare(st, log=lambda m: None)
            r = layerwise.prepare_rank(st, c, a.rank, lambda m: None) if a.rank else None
            g = prepare_groups(st, c, gl, lambda m: None)
            del st
        out["sets"][n] = {"consts": _cpu(c), "rank": _cpu(r) if r is not None else None,
                          "groups": {int(k): _cpu(v) for k, v in g.items()}, "note": note, "train": train,
                          "tables": tables}
        print(f"T2 set {n}: {len(train)} calibration documents ({note})")
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    main()
