#!/usr/bin/env python3
"""Calibration texts for the calib stage of scripts/t_ideas.py: one per domain, disjoint from the evaluation texts of
scripts/fetch_eval_texts.py (which are test splits).

    python scripts/fetch_calib_texts.py --output RUN/dataset_texts

Writes into --output:

  cal_code.txt    MBPP, every split (Python problems with their solutions): HumanEval's kind of text, not HumanEval
  cal_math.txt    GSM8K main train, rendered like the GSM8K test text
  cal_exam.txt    MMLU dev and validation (the test's subjects and prompt format, other questions)
  cal_news.txt    Penn Treebank train (Mikolov's text), sentences joined with spaces like the PTB test text
  cal_manifest.json  sources (Hub dataset and commit, or URL), counts and SHA256 of every text

The texts are read from the Hub's Parquet files (no dataset scripts run) or, for PTB, the raw text file. A source
that cannot be reached is left out (its set is skipped) and the others are still written.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_eval_texts import hub_table, render  # noqa: E402

PTB_TRAIN_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"
SOURCES = {  # name: (dataset repo, [(file prefix, split)], renderer)
    "code": ("google-research-datasets/mbpp", [("full/", s) for s in ("train", "validation", "test", "prompt")], None),
    "math": ("openai/gsm8k", [("main/", "train")], "gsm8k"),
    "exam": ("cais/mmlu", [("all/", "dev"), ("all/", "validation")], "mmlu"),
}


def converted_table(repo, config, split):
    """The rows of one split from the Hub's refs/convert/parquet branch (<config>/<split>/*.parquet)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    rev = "refs/convert/parquet"
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
                   if f.startswith(f"{config}/{split}/") and f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"{repo}: no {config}/{split} Parquet on {rev} either")
    tables = [pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=rev)) for f in files]
    return [r for t in tables for r in t.to_pylist()], rev, files


def render_mbpp(rows):
    return "\n\n".join(f"# {r['text'].strip()}\n{r['code'].replace(chr(13), '').strip()}" for r in rows)


def fetch(name):
    if name == "news":
        with urllib.request.urlopen(PTB_TRAIN_URL, timeout=60) as r:
            raw = r.read()
        sentences = [line.strip() for line in io.StringIO(raw.decode("utf-8"))]
        return {"source": PTB_TRAIN_URL, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "rows": len(sentences)}, " ".join(sentences)
    repo, parts, how = SOURCES[name]
    rows, info = [], {"source": f"https://huggingface.co/datasets/{repo}", "files": []}
    for prefix, split in parts:
        try:
            r, sha, files = hub_table(repo, (prefix, split))
        except (SystemExit, Exception):           # the Hub's own Parquet conversion branch, if main has no Parquet
            r, sha, files = converted_table(repo, prefix.rstrip("/"), split)
        rows += r
        info["commit"] = sha
        info["files"] += files
    info["rows"] = len(rows)
    return info, (render_mbpp(rows) if how is None else render(how, rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", default="", help="comma list of code, math, exam, news (default: all)")
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    names = [n for n in a.only.split(",") if n] or list(SOURCES) + ["news"]
    mpath = a.output / "cal_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    for name in names:
        target = a.output / f"cal_{name}.txt"
        if target.is_file() and target.stat().st_size and name in manifest:
            print(f"cal_{name}: kept ({target})")
            continue
        try:
            info, text = fetch(name)
        except Exception as exc:                      # one unreachable source leaves only its set out
            print(f"cal_{name}: NOT fetched ({repr(exc)[:300]})")
            continue
        target.write_text(text, encoding="utf-8")
        info.update(file=target.name, characters=len(text), sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        manifest[name] = info
        mpath.write_text(json.dumps(manifest, indent=1))
        print(f"cal_{name}: {info['rows']} rows, {len(text):,} characters -> {target}")
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
