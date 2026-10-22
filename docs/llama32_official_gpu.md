# Llama-3.2-3B: unchanged official baselines on the A4500

The selected common model is the unquantized checkpoint used in AQUA-KV's
README: `unsloth/Llama-3.2-3B`, pinned to
`d4446454d87d51aa42e1fb174f25acc5f8762331`. Its metadata reports
3,212,749,824 BF16 parameters. The test uses the base model, all 28 layers,
eight KV heads, head dimension 128 and its original scaled RoPE. No model
weights, model configuration or baseline source files are modified.

This is a new-model evaluation using author code. Llama-3.2-3B was not
previously established as tested by all three baseline authors. This run
establishes which unchanged implementations actually work; unsupported
methods retain their errors and have no replacement measurements.

## What runs

| Stage | Procedure and output |
| --- | --- |
| Hardware/model | Require actual CUDA before downloads. Fetch the fixed BF16 snapshot and record configuration, revision, device and software. |
| Original KV | Capture all 28 post-RoPE BF16 layers using the pinned model. Our fit uses four 1024-token WikiText2 TRAIN contexts, a separate 1024-token validation context, and TEST uses the first 8192-token context. |
| SplitZip | Run the untouched `bench_codec_throughput.py` with its supported saved-tensor interface. Inputs are the first/last 1024-token slices of that full TEST context, all 28 layers, and a separate TRAIN tensor. Retain chunk1024/warmup10/iters50/repeats5. The official default maximum calibration input is retained and reported by the author script. |
| Ours | Fresh Block32 A fit on this model's TRAIN KV. Both TEST slices must match SplitZip's original-word checksums and decode exactly, including a framed-packet restore. Report payload, complete packet bytes, shared state, wall samples and CUDA intervals. |
| ExANS | Attempt the pinned author's unchanged CMake build and GPU correctness test. The previously patched binary is not used. Local compilation of the unchanged CUDA source confirms the host `unique_ptr` error-path build failure. The official smoke test uses constructed words; even if it passes, its status is `smoke_only`, not a completed Llama comparison, and it produces no throughput row. |
| AQUA-KV | Run untouched `train_predictors.py` with HIGGS-2, first-layer HIGGS-4, Hadamard1024, 256 RedPajama sequences of 8192 tokens, 32 validation sequences, and activation offload. Then run official WikiText2 PPL with raw BF16 and AQUA, prefix4/recent128. No reduced calibration or page adapter fallback. |

AQUA and codec environments must tokenize the complete WikiText2 TEST text
identically before their results are associated. Perplexity uses the full
official TEST stream in 8192-token segments. The codec measurements use two
slices from its first segment, so these are separate metric scopes. Each
method retains its intended calibration procedure. No TEST data fits our
predictors or the SplitZip codebook.

AQUA's nominal HIGGS bit widths are not measured packed/wire bytes. Its
released reference quantizer/cache uses its own CPU/GPU state; running its
official GPU inference does not establish a GPU-only transfer codec. The
report does not rank PPL/inference time against codec-only throughput or our
packet latency. LongBench requires a separate Instruct checkpoint and is not
part of this base-model run. Serving, SSD, network and T-to-P are unmeasured.

## Environments and preservation

Run from the existing project `.venv`. The helper creates separate baseline
environments under ignored `runs/_llama_official/envs/`:

- AQUA: author `requirements.txt`, including Torch2.5.1 and Transformers4.46.3.
- SplitZip: author `requirements.txt`, including Torch2.9.0, Triton3.5.0 and
  Transformers4.57.1. Our codec runs in this environment.
- NumPy1.26.4 and PyArrow20.0.0 constrain unpinned transitive dependencies
  within those requirements; this avoids the removed PyArrow extension API
  used by these datasets versions. Our additional dependency is constriction0.5.0.
- AQUA's fast-Hadamard dependency uses the pinned, unchanged Dao-AILab source
  at `e7706faf8d1c3b9f241e36860640ad1dac644ede`.

Baseline commits are the same as the preceding source audit. Git commit and
tracked-file cleanliness are checked. There is no AST method extraction,
local cache substitute, unpack shim, kernel patch or ExANS build-source fix.
The existing project `.venv` is preserved. The helper reuses/provisions the
project-local CUDA12.8 build tools, without changing the driver. Model files,
source repositories, package installations and calibration require downloads
and disk space. Full AQUA calibration is substantially longer and uses much
more host RAM than the earlier page pilot; it keeps the author's settings.

Our reference predictor now accepts the model's captured rotary inverse
frequencies. Both float calibration features and frozen integer tables use
those same frequencies. Legacy unscaled captures retain their old behavior.
This changes our code, not a baseline or the Llama model.

## Run on the A4500

```bash
(
set -euo pipefail
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama32_official_gpu
git pull --ff-only origin experiment/llama32_official_gpu
source .venv/bin/activate
unset TRITON_INTERPRET

python -u scripts/run_llama_official_a4500.py --publish-results
)
```

The helper runs stages sequentially and prints log locations and a heartbeat
every 30 seconds during subprocesses. It packages and publishes reports even
when a baseline is blocked. Exit1 means a partial/blocked or failed run, not a
successful four-method reproduction. Correctness/data mismatches are failures,
not missing-method results. Full logs and status are preserved.

The default stages are `splitzip,ours,exans,aqua`. `--stages splitzip,ours`
selects only the native codec and ours; it must not be described as the full
comparison. Interrupted or partial runs print their exact resume command:

```bash
python -u scripts/run_llama_official_a4500.py \
  --resume runs/YOUR_LLAMA_RUN --publish-results
```

Completed-stage evidence is checksum-verified. Source/protocol changes require
a new run directory. Reports use an explicit allowlist: raw/compressed KV,
tokens, model weights and predictor tensors are never published. The prior
Qwen reports remain separate and unchanged.
