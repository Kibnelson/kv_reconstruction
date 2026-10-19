"""Capture the fresh residual-experiment final KV after configuration freeze.

Run from the project root: python -m kv_binary.capture
No training or codec selection occurs here. The pinned pretrained checkpoint
and native BF16 execution match the preceding controlled experiments.
"""

import argparse
import hashlib
import json
import pathlib
import platform
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_binary.fixtures import FINAL, fixture_manifest
from kv_research.model_utils import cache_matrices, mapping
from kv_smoke.download_model import MODEL_ID, REVISION
from kv_smoke.prompts import build_input_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=".smoke_model")
    parser.add_argument("--output", default=".kv_binary_data")
    args = parser.parse_args()
    target = pathlib.Path(args.output)
    target.mkdir(exist_ok=True, parents=True)
    if (target / "final_manifest.json").exists():
        raise RuntimeError("refusing to overwrite the frozen final capture")
    from kv_binary.run_final import verify_freeze
    freeze_sha = verify_freeze("kv_binary/selection.json")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(19)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    manifest = {
        "model": MODEL_ID,
        "revision": REVISION,
        "split": "fresh_binary_final",
        "fixtures": fixture_manifest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "threads": 4,
        },
        "cases": [],
    }

    for fixture in FINAL:
        for tokens in (512, 2048):
            ids = build_input_ids(tokenizer, fixture, tokens)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model(
                    torch.tensor([ids], dtype=torch.long),
                    use_cache=True,
                    logits_to_keep=1,
                )
                matrices = cache_matrices(output.past_key_values)
            prefill_seconds = time.perf_counter() - started
            name = f'{fixture["name"]}_{tokens}'
            filename = name + ".npz"
            np.savez(target / filename, **mapping(matrices))
            raw = b"".join(
                words.astype("<u2").tobytes()
                for pair in matrices
                for words in pair
            )
            manifest["cases"].append(
                {
                    "name": name,
                    "fixture": fixture["name"],
                    "kind": fixture["kind"],
                    "tokens": tokens,
                    "file": filename,
                    "input_token_ids": ids,
                    "first_token": int(output.logits[0, -1].argmax()),
                    "raw_payload_bytes": len(raw),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "prefill_seconds": prefill_seconds,
                }
            )
            print(f"captured {name}", flush=True)

    assert verify_freeze("kv_binary/selection.json") == freeze_sha
    manifest["selection_sha256"] = freeze_sha
    (target / "final_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
