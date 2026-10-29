#!/usr/bin/env python3
"""AQUA-KV's RedPajama calibration samples, made by the authors' own sampler, saved for train_predictors.py.

train_predictors.py --dataset pajama loads togethercomputer/RedPajama-Data-1T-Sample, which has been removed from
the Hugging Face Hub ("doesn't exist on the Hub or cannot be accessed"; reported since late 2025). The authors'
get_loaders also accepts a saved file (--dataset <file.pt>, torch.load(name)[:nsamples]), so this script runs
their sampler unchanged (aquakv/datautils.py get_loaders("pajama", ...): same seed, sample count, length and
tokenizer) with only the download redirected: the original repository first, then its Parquet conversion, then
ll922/RedPajama-Data-1T-Sample-Backup, a Parquet mirror of it (3 GB, 930,514 rows, the original's count), loaded
with remote code disabled. The source and its row count are recorded next to the output: with the same rows in
the same order, the authors' seed draws the same windows. (ZengXiangyu/RedPajama-Data-1T-Sample is not used:
902,104 rows and 108 GB, so not the same sample.)

The tokenizer: get_loaders asks for the slow tokenizer (use_fast=False). Llama 3 has no SentencePiece model, and
with their transformers 4.46.3 that call returns False instead of a tokenizer (AttributeError: 'bool' object has
no attribute 'bos_token_id'). Only then is the call repeated with the fast tokenizer, Llama 3's own BPE (the
one their perplexity script uses); which one was used is recorded.

Run with the AQUA-KV environment's python:
    $AQUA_PY scripts/aqua_calibration_data.py --source $AQUA_SRC --model unsloth/Llama-3.2-3B --output cal.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PAJAMA = "togethercomputer/RedPajama-Data-1T-Sample"
MIRRORS = ("ll922/RedPajama-Data-1T-Sample-Backup",)       # Parquet copy (awslabs/ai-on-eks issue 410)
ORIGINAL_ROWS = 930_514


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
    auto = du.AutoTokenizer

    class Tokenizer:                                 # the authors' AutoTokenizer call, fast only if slow fails
        @staticmethod
        def from_pretrained(path, *args, **kw):
            tok = auto.from_pretrained(path, *args, **kw)
            if isinstance(tok, bool) or tok is None:
                print(f"the slow tokenizer of {path} is not available (returned {tok!r}); using the fast one",
                      flush=True)
                tok = auto.from_pretrained(path, *args, **dict(kw, use_fast=True))
                used["tokenizer"] = f"{type(tok).__name__} (fast; the slow tokenizer is not available)"
            else:
                used["tokenizer"] = type(tok).__name__
            return tok
    du.AutoTokenizer = Tokenizer

    def load(path, *args, **kw):
        if path != PAJAMA:
            return original(path, *args, **kw)
        errors = []
        tries = [(path, None, {}), (path, "refs/convert/parquet", {})] + \
            [(m, None, {"trust_remote_code": False}) for m in MIRRORS]
        for repo, rev, extra in tries:
            name = f"{repo}@{rev or 'main'}"
            try:
                ds = original(repo, *args, **dict(kw, **extra, **({"revision": rev} if rev else {})))
                if "text" not in ds.column_names:
                    raise ValueError(f"no text column (columns {ds.column_names})")
                used.update(dataset=repo, revision=rev or "main", rows=len(ds), columns=ds.column_names,
                            same_row_count_as_original=len(ds) == ORIGINAL_ROWS)
                print(f"RedPajama sample loaded from {name}: {len(ds)} rows", flush=True)
                return ds
            except Exception as exc:              # report every attempt, then give up
                errors.append(f"{name}: {exc!r}"[:600])
                print(f"RedPajama sample, {name}: not loaded ({exc!r})"[:800], flush=True)
        raise RuntimeError("the RedPajama sample is not accessible: " + " | ".join(errors) +
                           " (if the dataset is gated, log in on this machine with huggingface-cli login)")
    du.load_dataset = load
    data = du.get_loaders("pajama", nsamples=a.nsamples, seed=a.seed, seqlen=a.seqlen, model_path=a.model)
    torch.save(data, a.output)
    meta = dict(used, nsamples=len(data), seqlen=a.seqlen, seed=a.seed, model=a.model,
                sampler="aquakv/datautils.py get_loaders('pajama'), unchanged")
    a.output.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {a.output}: {len(data)} samples x {a.seqlen} tokens ({meta['dataset']}@{meta['revision']}, "
          f"{meta['rows']} rows)")


if __name__ == "__main__":
    main()
