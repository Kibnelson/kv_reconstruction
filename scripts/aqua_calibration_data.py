#!/usr/bin/env python3
"""AQUA-KV's RedPajama calibration samples, made by the authors' own sampler, saved for train_predictors.py.

train_predictors.py --dataset pajama loads togethercomputer/RedPajama-Data-1T-Sample through a loading script,
which the Hub no longer serves to datasets 2.20 ("doesn't exist on the Hub or cannot be accessed"). The authors'
get_loaders also accepts a saved file (--dataset <file.pt>, torch.load(name)[:nsamples]), so this script runs
their sampler unchanged (aquakv/datautils.py get_loaders("pajama", ...): same seed, sample count, length and
tokenizer) with only the download redirected: first the original loader, then the Hub's Parquet conversion of
the same dataset (refs/convert/parquet). The conversion can be partial for a dataset this size, so the drawn
windows may differ from the authors'; the source used is recorded next to the output.

Run with the AQUA-KV environment's python:
    $AQUA_PY scripts/aqua_calibration_data.py --source $AQUA_SRC --model unsloth/Llama-3.2-3B --output cal.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PAJAMA = "togethercomputer/RedPajama-Data-1T-Sample"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="the AQUA-KV checkout")
    p.add_argument("--model", default="unsloth/Llama-3.2-3B")
    p.add_argument("--nsamples", type=int, default=256, help="train_predictors.py --total_nsamples")
    p.add_argument("--seqlen", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    from aquakv import datautils as du
    original = du.load_dataset
    used = {}

    def load(path, *args, **kw):
        if path != PAJAMA:
            return original(path, *args, **kw)
        errors = []
        for rev in (None, "refs/convert/parquet"):
            try:
                ds = original(path, *args, **(dict(kw, revision=rev) if rev else kw))
                used.update(dataset=path, revision=rev or "main (loading script)", rows=len(ds))
                print(f"RedPajama sample loaded ({used['revision']}): {len(ds)} rows", flush=True)
                return ds
            except Exception as exc:              # report every attempt, then give up
                errors.append(f"{rev or 'main'}: {exc!r}"[:600])
                print(f"RedPajama sample, {rev or 'main'}: not loaded ({exc!r})"[:800], flush=True)
        raise RuntimeError("the RedPajama sample is not accessible: " + " | ".join(errors) +
                           " (if the dataset is gated, log in on this machine with huggingface-cli login)")
    du.load_dataset = load
    data = du.get_loaders("pajama", nsamples=a.nsamples, seed=a.seed, seqlen=a.seqlen, model_path=a.model)
    torch.save(data, a.output)
    meta = dict(used, nsamples=len(data), seqlen=a.seqlen, seed=a.seed, model=a.model,
                sampler="aquakv/datautils.py get_loaders('pajama'), unchanged")
    a.output.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {a.output}: {len(data)} samples x {a.seqlen} tokens ({meta['revision']})")


if __name__ == "__main__":
    main()
