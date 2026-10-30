#!/usr/bin/env python3
"""Prompt files for the vLLM PD benchmark (vllm bench serve --dataset-name custom): windows of real text from the
evaluation datasets (scripts/fetch_eval_texts.py: WikiText-2, HumanEval, GSM8K, MMLU, PTB), cut to an exact token
count with the model's tokenizer, in round-robin over the datasets.

  python scripts/pd_prompts.py --texts $KV_RUNS/dataset_texts --model unsloth/Llama-3.2-3B --tokens 2048,8192 \
      --per-length 64 --output-len 128 --out $RUN/pd/prompts

Writes prompts_<tokens>.jsonl ({"prompt", "output_tokens", "dataset"}) and prompts_mixed.jsonl (all lengths
interleaved), plus check.jsonl (a few prompts of uneven lengths for the correctness pass).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ["wikitext2", "humaneval", "gsm8k", "mmlu", "ptb"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tokens", default="2048,8192")
    ap.add_argument("--per-length", type=int, default=64)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--check-lengths", default="1000,3000,5000,8192")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    ids = {}
    for d in DATASETS:
        p = Path(a.texts) / f"{d}.txt"
        if p.is_file():
            ids[d] = tok(p.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    if not ids:
        raise SystemExit(f"no dataset texts in {a.texts} (scripts/fetch_eval_texts.py)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cursor = {d: 0 for d in ids}

    def window(d, n):
        """n tokens of dataset d, wrapping around; decoded and re-encoded until the count is exact."""
        seq = ids[d]
        a0 = cursor[d] % max(1, len(seq) - n)
        cursor[d] = a0 + n
        piece = (seq + seq)[a0:a0 + n] if len(seq) < n else seq[a0:a0 + n]
        text = tok.decode(piece)
        got = tok(text, add_special_tokens=False)["input_ids"]
        return tok.decode(got[:n]) if len(got) > n else text

    mixed = []
    names = list(ids)
    for n in [int(t) for t in a.tokens.split(",")]:
        rows = []
        for i in range(a.per_length):
            d = names[i % len(names)]
            rows.append({"prompt": window(d, n - 1), "output_tokens": a.output_len, "dataset": d, "tokens": n})
        with open(out / f"prompts_{n}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        mixed.append(rows)
    with open(out / "prompts_mixed.jsonl", "w") as f:
        for group in zip(*mixed):
            for r in group:
                f.write(json.dumps(r) + "\n")
    with open(out / "check.jsonl", "w") as f:
        for i, n in enumerate(int(t) for t in a.check_lengths.split(",")):
            d = names[i % len(names)]
            f.write(json.dumps({"prompt": window(d, n - 1), "output_tokens": 24, "dataset": d, "tokens": n}) + "\n")
    print(json.dumps({"out": str(out), "datasets": names, "tokens": a.tokens, "per_length": a.per_length}))


if __name__ == "__main__":
    main()
