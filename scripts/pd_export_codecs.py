#!/usr/bin/env python3
"""Export the transfer codecs' constants for the vLLM PD runs (kv_compare/pd/export.py), from a run folder of
scripts/run_a5000.sh (its calcap stage: the WikiText-2 train calibration captures). Runs in the codec environment
($KV_HOME/venv) on the GPU; the vLLM environment only loads the file.

  python scripts/pd_export_codecs.py --run $KV_HOME/runs/llama32_3b --output $KV_HOME/runs/llama32_3b/pd_codecs.pt \
      --splitzip-source $KV_HOME/sources/splitzip
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
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(b, out)
    print(json.dumps({"written": str(out), **b["meta"]}, indent=1))


if __name__ == "__main__":
    main()
