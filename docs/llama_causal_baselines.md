# Baselines for the completed causal packet run

Run the controls on the A4500 without repeating fitting, Block32/Lite encoding,
or their quality evaluation:

```bash
cd ~/kv_reconstruction
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_llama_baselines_a4500.sh \
  runs/llama_causal_a4500_20261027_184615_1307629
```

The wrapper publishes only the new small report. Raw arrays, packets, model
weights, and fitted tensor state stay on the GPU host. It preserves unrelated
working changes when rebasing. A failed/partial run is published with that status.

## What is compared

| Row | What runs | What it tells us |
|---|---|---|
| Raw BF16 | Framed GPU copy and receiver copy | Packet overhead and minimum work in the current harness |
| SplitZip official, framed | Unchanged pinned author codec, with local transport framing and validation | Matched lossless bytes and per-request wall costs |
| SplitZip author native | Unchanged `bench_codec_throughput.py`, saved-tensor mode, default timing settings | Author's native codec throughput and payload size, reported separately |
| FP8 E4M3FN saturating | Unscaled Torch cast, saturation at ±448, restore to BF16 | Simple lossy size/speed/quality reference; no paper attribution |
| Block32 and Lite, lossless / κ=.5 / κ=1 | Reuse the completed causal run verbatim | Current compression, speed, quality and shared-state tradeoffs |
| AQUA-KV official | Pending a successful official HIGGS/calibration/PPL evaluation | No adapted Quanto or local predictor results substituted |
| ExANS official | Previous official build blocked; no matched result | No different-model or local adaptation results substituted |

Every new row uses the **same original BF16 TEST words** already recovered by
the independently verified lossless receiver: all 28 layers, 512 tokens, K/V
combined in the causal run's layout. Files, word digests and receiver proof are
checked before use. The existing run remains unchanged.

The pinned model is `unsloth/Llama-3.2-3B`, revision
`d4446454d87d51aa42e1fb174f25acc5f8762331`. The team corpus is PEP text and CPython
code, with separated TRAIN/table/TEST documents. This is a matched codec
experiment, not a reproduction of every paper's full evaluation.

SplitZip uses commit `42c743118a29db3270ae6a921a4a4eecb20cad04` from
[the official repository](https://github.com/Intelligent-Microsystems-Lab/SplitZip).
The runner checks the codec, author benchmark and requirements hashes, and
requires the author's pinned packages in its separate environment. Defaults:

- Python: `runs/_llama_official/envs/splitzip/bin/python`
- Source: `runs/_llama_official/sources/splitzip`

Pass `--splitzip-python PATH --splitzip-source PATH` after the base-run argument
if those directories differ. Missing or mismatched environments are reported as
blocked; the runner does not install dependencies or edit baseline code.

The author benchmark supports saved TEST and calibration tensors. Its default
calibration loader takes a **prefix of 4,194,304 values**. We prepare exactly that
many values from the causal run's selected TRAIN data, without repetition. For
the current run this is the first two layers of train0's first 1,024 tokens, in
K||V token-row order. This is explicitly a prefix, not a balanced layer sample.
Native TEST tensors contain exactly the complete request: no repetition or
truncation. The original method and timing loops remain unchanged.

The framed SplitZip row charges its five transmitted fields (including 7-byte
escapes), alignment, directory, geometry/state header and checksum. Construction
scratch is excluded from wire bytes but remains in Torch peak samples. Fixed
LUT state is separately reported. Framing, copies and decoder validation add
work beyond the author native benchmark. The two timing scopes stay separate.

New packets are decoded in fresh receiver processes from wire plus fixed LUT
state only. The FP8 quality check uses the same BF16 model and 127 scored
continuation targets as the base run; exact controls must preserve the reference
NLL. It is a short continuation check, not full WikiText2 PPL or LongBench.

The root environment must match the completed causal run. SplitZip's intended
environment is separately recorded. These results compare the implementations
in their recorded environments, not identical framework versions. Warm codec
time excludes fitting, calibration, model loading, serialized export and upload;
export/upload are recorded separately. No attention, pool conversion, SSD,
network or serving throughput claim follows from these measurements.

The improvement targets are **complete packet bytes, encode/decode wall time,
quality at matched size, and shared decoder state**. In the completed run Lite
lossless averaged 8.87967 bpv (44.50% saved), 330.75 ms encode and 418.62 ms decode,
with 324.11 MiB of shared predictor tensors. SplitZip and raw controls will show
the current latency gap on exactly these words. FP8 will show what is gained by
the more elaborate lossy predictor at its additional compute cost.

The previous official baseline attempt is preserved in
[its report](https://github.com/Kibnelson/kv_reconstruction/blob/cd5df510d42739d11470d50d99f6c1ccf4caec14/benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517/summary.md).
Its WikiText2 metrics are historical context, not matched rows for this corpus.

## Local validation

`tests.test_llama_baselines` checks all BF16 patterns for the raw control, FP8
saturation/rounding, framing/checksum/identity validation, changed input rejection,
separate receiver processes, and tiny BF16-model continuation quality. With the
pinned SplitZip checkout and explicit Triton CPU interpreter it also checks
unaltered native encoding/decoding, odd escape alignment, tails and fixed-state
reload. CPU evidence cannot be packaged as a completed GPU comparison.
