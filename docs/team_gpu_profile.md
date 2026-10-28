# GPU-resident exact KV decoding versus official controls

This experiment starts with compressed packet payloads already on the GPU. It
compares the unchanged official SplitZip and ExANS decode paths, a raw BF16
control, and two one-step predictor configurations. All methods reconstruct the
same original Llama-3.2-3B BF16 cache: all 28 layers, two TEST documents, and
separate 512-token and 8,192-token cases.

The model is `unsloth/Llama-3.2-3B`, revision
`d4446454d87d51aa42e1fb174f25acc5f8762331`. TEST text is PEP 8 and TEST code is
CPython `argparse.py`, with the original TRAIN/table/TEST roles preserved in the
parent capture receipts. No WikiText or AQUA calibration procedure is implied by
this matched cache workload.

The primary metric is synchronized wall latency of the public GPU decode call,
including validation, packet staging, reconstruction and the returned output.
CUDA intervals and native graph replay are diagnostic measurements. Python
launches and validation still contribute to public latency. File reads, wire
checksum parsing, host-to-device payload upload and graph initialization happen
before warm timing. Raw-KV encoding is not measured in this decode experiment.

## Controlled optimization

The reference one-step decoder copies a compact packet into an expanded sparse
GPU arena before running the unchanged native decoder. The candidate maps the
same compact segments directly into stable GPU staging and supplies absolute
starts to that decoder. Predictor coefficients, probability arithmetic, entropy
decoder kernels, transmitted packet bytes and fitted state remain fixed. The
experiment tests whether removing the expanded arena and its expansion pass
reduces latency and workspace. Performance is unmeasured until the A4500 run.

Dense-uniform and rank-256-uniform are separate configurations. Reference and
compact paths consume the same serialized packet within each configuration;
changing predictor rank changes packet size and is not an isolated decoder
optimization. The original six-document, 8,192-token TRAIN fit is reused. No
fitting, model download, source KV capture or new compression is required.

## Controls and evidence

- SplitZip and ExANS use their pinned official implementations, existing fitted
  state and intended runtime settings. ExANS preserves its fixed-slot padding
  and raw overflow accounting. Their original decode APIs remain unchanged.
  This is a matched GPU-resident framed API measurement in a local harness,
  rather than reproduction of each paper's end-to-end evaluation. SplitZip
  retains its original Torch 2.9/Triton 3.5 environment; the one-step codec uses
  Torch 2.11/Triton 3.6. Per-worker versions are recorded.
- Native controls run before and after the one-step comparisons to expose
  drift. Documents and paired one-step arms alternate during measurement.
- Ordinary timings and separate diagnostic profiler processes are distinct.
  Original-word and wire-hash assertions run outside timed/profiled regions.
- A fresh receiver gets fixed codec state and packets, with expected word
  digests used only for assertions. It receives no source KV or predictions.
  Alternating packets also tests returned-output ownership and graph reuse.
- Complete packet bytes include framing, directories and checksums. Shared
  predictors, decoder workspace and staging allocations are additional.
- AQUA-KV is lossy and is not included in this exact-codec comparison. Its
  official evaluation remains separate; no adapted page result substitutes.

## Run

On the existing A4500 host, on `experiment/llama_causal_packet_compare`:

```bash
bash scripts/run_team_gpu_a4500.sh
```

The default parent is
`runs/team_packet_a4500_20261028_122017_2834427`. Resume a partially completed
run with `bash scripts/run_team_gpu_a4500.sh --resume RUN_DIR`. Source identity,
parent receipts, packet/state hashes and workload options must remain identical
when resuming. The wrapper archives its CUDA correctness gate and publishes a
small report. Full profiler traces and original packets remain on the GPU host.

Calculated 1, 100, 200, 300 and 400 Gb/s tables report both serial `wire + decode`
and idealized `max(wire, decode)` values. These are models, not measured network
transfer or overlap. This experiment does not measure SSD I/O, compressed
attention, T-to-P conversion, serving throughput or a resident compressed pool.
