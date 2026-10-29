#!/usr/bin/env python3
"""Plain-text copies of SplitZip's evaluation datasets (its Table 4) and WikiText-2 train, for the datasets,
context and calcap stages of scripts/bench_llama_compare.py.

    python scripts/fetch_eval_texts.py --output RUN/dataset_texts

Writes into --output:

  wikitext2.txt        WikiText-2 raw test, rows joined with blank lines (as AQUA-KV's evaluation and our context
                       stage read it)
  wikitext2_train.txt  WikiText-2 raw train, the same way (SplitZip's code book and our predictor calibrate on it)
  humaneval.txt        HumanEval (164 problems): each prompt followed by its canonical solution, blank line between
  gsm8k.txt            GSM8K main test: "Question: ...\\nAnswer: ..." per problem, blank line between
  mmlu.txt             MMLU all test, as zero-shot prompts with the answer letter; questions taken round-robin over
                       the 57 subjects (the file is sorted by subject, so a few windows would otherwise cover only
                       the first subjects)
  ptb.txt              Penn Treebank test (Mikolov's text, the file Hugging Face's ptb_text_only loads), sentences
                       joined with spaces as GPTQ-style evaluations do
  manifest.json        sources (Hub dataset and commit, or URL), file names, counts and SHA256 of every text

SplitZip's paper does not say how it turned the datasets into prompts; these are the plainest renderings. Every
text is read from the Hub's Parquet files (no dataset scripts run) or, for PTB, the raw text file.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
HUB = {  # name: (dataset repo, file pattern pieces)
    "wikitext2": ("Salesforce/wikitext", ("wikitext-2-raw-v1/", "test")),
    "wikitext2_train": ("Salesforce/wikitext", ("wikitext-2-raw-v1/", "train")),
    "humaneval": ("openai/openai_humaneval", ("openai_humaneval/", "test")),
    "gsm8k": ("openai/gsm8k", ("main/", "test")),
    "mmlu": ("cais/mmlu", ("all/", "test")),
}


def hub_table(repo, pieces):
    """The Parquet rows of one split of a Hub dataset, plus (commit, files)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    sha = api.dataset_info(repo).sha
    prefix, split = pieces
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=sha)
                   if f.startswith(prefix) and f.endswith(".parquet") and Path(f).name.startswith(split + "-"))
    if not files:
        raise SystemExit(f"{repo}: no Parquet file {prefix}{split}-*.parquet at {sha}")
    tables = [pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=sha)) for f in files]
    rows = [r for t in tables for r in t.to_pylist()]
    return rows, sha, files


def render(name, rows):
    if name in ("wikitext2", "wikitext2_train"):
        return "\n\n".join(r["text"] for r in rows)
    if name == "humaneval":
        return "\n\n".join(r["prompt"] + r["canonical_solution"] for r in rows)
    if name == "gsm8k":
        return "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows)
    if name == "mmlu":
        by = {}
        for r in rows:
            by.setdefault(r["subject"], []).append(r)
        subjects = sorted(by)
        order = [by[s][i] for i in range(max(len(v) for v in by.values())) for s in subjects if i < len(by[s])]
        out = []
        for r in order:
            ch = "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"]))
            out.append(f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n"
                       f"{r['question']}\n{ch}\nAnswer: {'ABCD'[int(r['answer'])]}")
        return "\n\n".join(out)
    raise KeyError(name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", default="", help="comma list of names (default: all)")
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    names = [n for n in a.only.split(",") if n] or list(HUB) + ["ptb"]
    mpath = a.output / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    failed = {}
    for name in names:
        target = a.output / f"{name}.txt"
        if target.is_file() and target.stat().st_size and name in manifest:
            print(f"{name}: kept ({target})")
            continue
        try:
            info, text = fetch(name)
        except Exception as exc:                      # one unreachable source leaves only its dataset out
            failed[name] = repr(exc)[:300]
            print(f"{name}: NOT fetched ({failed[name]})")
            continue
        target.write_text(text, encoding="utf-8")
        info.update(file=target.name, characters=len(text), sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        manifest[name] = info
        mpath.write_text(json.dumps(manifest, indent=1))
        print(f"{name}: {info['rows']} rows, {len(text):,} characters -> {target}")
    print(f"wrote {mpath}")
    needed = {"wikitext2", "wikitext2_train"} & set(names)
    if needed - set(manifest):
        raise SystemExit(f"STOP: {', '.join(sorted(needed - set(manifest)))} could not be fetched (calibration and the "
                         "context sweep need them)")
    if failed:
        print(f"left out: {', '.join(failed)} (the datasets stage skips a dataset without its text)")


def fetch(name):
    """(manifest entry, text) of one dataset."""
    if name == "ptb":
        with urllib.request.urlopen(PTB_URL, timeout=60) as r:
            raw = r.read()
        sentences = [line.strip() for line in io.StringIO(raw.decode("utf-8"))]
        info = {"source": PTB_URL, "raw_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(sentences)}
        return info, " ".join(sentences)
    repo, pieces = HUB[name]
    rows, sha, files = hub_table(repo, pieces)
    info = {"source": f"https://huggingface.co/datasets/{repo}", "commit": sha, "files": files, "rows": len(rows)}
    return info, render(name, rows)

if __name__ == "__main__":
    main()
