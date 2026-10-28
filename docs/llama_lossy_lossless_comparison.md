# Lossless and lossy KV transfer on Llama-3.2-3B: ours against SplitZip, ExANS, AQUA-KV and KIVI

`scripts/bench_llama_compare.py` captures Llama-3.2-3B KV once and runs every method on the same words:

- **Lossless:** our predictive codec against SplitZip-style and ExANS-style exponent coding, in bits per value.
- **Lossy:** our codec run closed loop with a quantized residual, against AQUA-KV-style, KIVI-style and FP8-cast
  baselines. Each method is scored by bits per value and by how much worse the model continues the text
  when it reads that method's KV.

The implementation is in `kv_compare/`. The protocol and document roles match the SmolLM2-135M study, whose
results are reproduced at the end for reference.

## Run on the NVIDIA NUC

The default checkpoint is `unsloth/Llama-3.2-3B` at revision `d4446454d87d51aa42e1fb174f25acc5f8762331`. It is the
ungated copy of the original BF16 Llama-3.2-3B weights that AQUA-KV's README points to, and the same pin as
`scripts/run_llama_official_a4500.py` on `experiment/llama32_official_gpu`. No Hugging Face login is needed, and a
NUC that ran the official study already has it cached. `meta-llama/Llama-3.2-3B` is gated, so use `--model` only
with an account that has access.

```bash
cd ~/kv_reconstruction
git fetch origin && git checkout llama32-3b-lossy-lossless-compare && git pull
source .venv/bin/activate                      # the CUDA environment from README.md
python -m unittest discover -s kv_compare -p 'test_*.py' -v      # CPU, tiny random Llama, no download
python scripts/bench_llama_compare.py --device cuda --output runs/llama32_3b_compare
```

### If a run stops part-way

Finished stages keep their files. Rerun only the remaining stages in the same `--output`, inside `tmux` so a closed
terminal or dropped SSH session cannot end the run. Keep a console copy:

```bash
tmux new -s kvcmp          # reattach later with: tmux attach -t kvcmp
python -u scripts/bench_llama_compare.py --device cuda --output runs/llama32_3b_compare \
    --stages lossy,report 2>&1 | tee -a runs/llama32_3b_compare/console.log
```

The run needs about 5 GB of GPU memory while fitting predictors and about 8 GB while scoring lossy KV with the
model loaded. It can share the GPU with another job if that much is free; `run.log` records the free memory at the
start and warns when another process holds memory. Timings from a shared GPU are not representative.

`run.log` records each step with host and GPU memory, and on an error the full traceback. If the log simply
stops, the process was killed from outside. Check for the out-of-memory killer with
`sudo dmesg | grep -iE "killed process|out of memory" | tail`. `lossy.json` is written after each test
document, so a partial lossy run still yields results.

### Commit the results to this branch

Package the small result files into `benchmark_results/` and push them to this branch for review. The captured KV stays local.

```bash
NAME=llama32_3b_compare_a4500_$(date +%Y%m%d_%H%M)
python scripts/package_llama_compare.py runs/llama32_3b_compare --destination benchmark_results/$NAME
git add benchmark_results/$NAME
git commit -m "Add Llama-3.2-3B codec comparison results ($NAME)"
git push
```

The package holds `summary.md`, `lossless.json`, `lossy.json`, `run_args.json`, `environment.json` (GPU,
versions, source commit), `run.log` and `capture_meta.json`, plus `manifest.json` with SHA256 hashes. Before
copying, the packager checks that the reports agree with the summary and that every lossy row is measured against
the exact KV.

| Stage | What it does | Approximate size |
| --- | --- | --- |
| `capture` | Prefills 9 documents of 8,192 tokens. Saves BF16 K/V words in cache order, plus q_proj outputs for the six calibration documents (needed for the attention input). | about 8.5 GB of KV and 8.5 GB of queries |
| `lossless` | Exact per-word costs on the two test documents, for whole documents and the four 256-token reference chunks. | — |
| `lossy` | Codes the first 3,072 tokens of each test document with every lossy method. The model then processes the next 512 tokens using that KV. The reference is the captured exact KV restored into the cache; restoring it twice must agree exactly. The model's own prefill of the 3,072-token prompt is reported beside it: on a GPU it can differ slightly (for example +0.05%), because kernels round differently at a different sequence length. | — |
| `frontier` | Optional. About 37 predictor variants fitted from one covariance per layer: input sets, 1–16 steps, reduced-rank inputs and exact 8-bit integer formats. Each gets exact lossless costs and multiply-adds per value (`frontier.json`). | about 30 minutes on an RTX A4500 |
| `gpu_speed` | Optional; needs the GPU to itself. Times one layer of the predictor for the frontier variants in float32, BF16 and exact 8-bit integer arithmetic. Then encodes and decodes four real test layers with the Triton value-slice coder, checking the round trip bit for bit, with both stage-1 decoders. Finally it encodes and decodes the whole first test document (every layer, layer by layer) with the fused one-step codec in several kernel shapes (`gpu_speed.json`). | 10–20 minutes (mostly Triton compiling the fused kernels) |
| `pool` | Optional; needs the GPU to itself. GPU-resident KV pool: how long it takes to bring back a cached 8,192-token prefix by recomputing it (the model's prefill), by copying raw KV from CPU memory, or by decoding our compressed KV; what adding a prefix costs; how many tokens fit (`pool.json`). | 10–15 minutes |
| `report` | Writes `summary.md`. | — |

Useful options:

- `--stages lossless,lossy,report` reuses existing captures in the same `--output`.
- `--revision <commit>` pins the model weights.
- `--no-queries` skips the 8.5 GB query capture. The strong-and-fast configuration then runs without its attention input.
- `--configs`, `--kappas`, `--prompt-tokens` and `--continuation-tokens` change the sweep.
- `--max-layers N --stages capture,lossless` gives a quick partial-layer smoke run.

Each stage's elapsed time is logged.

### Speed run (GPU to itself)

```bash
nvidia-smi                       # nothing else should be running
python -u scripts/bench_llama_compare.py --device cuda --output runs/llama32_3b_compare \
    --stages gpu_speed,report 2>&1 | tee -a runs/llama32_3b_compare/console.log
```

The speed stage reports kernel time only: CUDA-event medians after warm-up, with no host transfers. That matches how SplitZip and ExANS publish GB/s, and GB/s counts raw BF16 bytes. The summary combines the predictor and the coder into a decode rate, shown as decoded values per second and as the link speed that rate keeps up with (values per second x bits per value). It also lists SplitZip's own measurement on the same A4500 for reference. The coder's slice tables are computed once per run, so encoder and decoder share them; for transfers between machines they would ship as constants.

Two kinds of decode rate appear in the summary:

- **Estimates from parts.** The predictor and the coder are timed separately (and overlapped on two CUDA streams) on four layers coded at once. A receiver cannot do that: layer l is predicted from the decoded layer l − 1, so it decodes one layer at a time, with less parallel work. These estimates also leave out the small element-wise steps around the coder (RoPE, the spread, the next layer's features).
- **Measured end to end (per-layer codec).** For each layer, cuBLAS computes the exact 8-bit product (int32 result), then one Triton kernel of one-warp programs (a block of tokens x one head) scales, rotates, codes and makes the next layer's features. Streams stay open across layers, so many small programs cost almost no extra stream state. `--layer-configs` sets tokens per program x token chunks; each chunk decodes on its own CUDA stream, so one chunk's product overlaps another chunk's coding.
- **Measured end to end (fused codec).** One Triton kernel per direction does all of it for the one-step predictor: the 8-bit tensor-core product, RoPE, the spread, the coding and the next layer's 8-bit features, layer by layer over the whole document. Its decode must be bit-exact. The summary's last transfer table uses only these measured numbers. `--fused-configs` sets the kernel shapes tried (tokens per program x rows of lanes x warps), `--fused-modes` the coder modes.

The per-layer rows also time the decode in parts: the products alone, the coding kernels alone (products precomputed) and both as decoded. On the A4500 (`fe1b577`) the parts were 13.3 ms and 15.6 ms but the decode 32.3 ms: the two did not run side by side at all (cuBLAS's product kernels need whole SMs, so the coding cannot share them) and the hand-offs between them cost 3.3 ms more. The speed stage therefore also times the decode captured as a CUDA graph (no launch gaps) and with a smaller Triton int8 product that leaves room on each SM for coding warps.

The next run (`6792614`) showed neither helps: a CUDA graph gave the same 32.5 ms and the small Triton product no overlap (the coding kernel needs every SM's whole register file to hide its memory waits, so nothing runs beside it). The decode is products plus coding, so one of them has to get cheaper. The fast-link mode does that: a reduced-rank product (`--layer-ranks`, default 512): two 8-bit products, previous layer to 512 values and back to 2048, with an exact 8-bit requantization between them, about half the multiply-adds. On Llama it costs about +0.2 bits per value (`scripts/probe_cheap_products.py`), so it only pays on fast links; the measured transfer table picks, for every link speed, the measured mode with the shortest transfer.

Stage 1 of the decoder (finding the slice symbol) has two versions over the same streams. The direct version, now the default, uses a table for the rare tail slots and one comparison elsewhere. The search version, used before, does a binary search that waits for the slowest of a program's lanes. Both are timed.

A cheaper predictor for fusing was ruled out: predicting each KV head only from its own head in the previous layer costs +0.59 bits per value on SmolLM2-135M (+0.38 for keys from keys and values from values), so the fused codec keeps the full one-step product.

### Link speeds in the tables

The transfer tables use link speeds that real prefill/decode deployments have:

| Band | Speeds | Where |
| --- | --- | --- |
| Ordinary cloud and storage | 1–15 Gb/s | between regular cloud servers, remote KV storage (CacheGen, KVServe) |
| Consumer and workstation | 10–100 Gb/s | 10 GbE consumer machines, 50 Gb/s workstations, cross-cluster cloud links (KVServe) |
| Datacenter RDMA, per GPU | 200–400 Gb/s | one InfiniBand/RoCE NIC per GPU: Azure ND H100 v5, AWS P5, Mooncake's H800 nodes |
| Inside one machine | hundreds of GB/s | NVLink; compression does not pay there |

On a slow link the transfer time is the link time for any codec that decodes fast enough, so bits per value decide it. SplitZip codes only the exponents, so it is at most 16/12 = 1.33x faster than raw however slow the link; our ratio is set by the predictor. On a 200–400 Gb/s link, decode speed decides it, and SplitZip's much faster decoder wins on this GPU. Those links come with H100-class GPUs, and each GPU moves its own share of the KV, so the rate that matters there is the per-GPU decode rate.

### GPU-resident pool

The `pool` stage asks whether keeping finished requests' KV compressed in GPU memory pays. The alternatives for a reused prefix are to recompute it, to keep it raw (in GPU memory, or in CPU memory and copy it back over PCIe), or to keep it compressed and decode it. The stage measures each on this machine. The lossy rows of its capacity table use the lossy stage's bits and quality; a GPU decoder for the lossy codec does not exist yet.

```bash
python -u scripts/bench_llama_compare.py --device cuda --output runs/llama32_3b_compare \
    --stages pool,report 2>&1 | tee -a runs/llama32_3b_compare/console.log
```

## Methods

| Method | Lossless? | How it predicts or codes | Rate |
| --- | --- | --- | --- |
| SplitZip-style | yes | One top-16 exponent codebook calibrated on the calibration documents; sign and mantissa byte raw; escapes cost 24 bits | computed |
| ExANS-style | yes | Exponents centred per 256-token block and per layer K/V, block-local entropy model; sign and mantissa byte raw | computed, with model and table overheads |
| Ours, `prev` | yes | Layer l−1 of the same token, whole layer in one step (the "44%" version) | computed −log2 P |
| Ours, `block` | yes | Adds earlier 128-channel groups of the same layer (16 steps) | computed −log2 P |
| Ours, `nohint` | yes | Adds layer 0, layer l−2, the token's earlier occurrence and the adaptive spread | computed −log2 P |
| Ours, `fast` | yes | Layers 0, l−2, l−1 and the recomputed attention of layer l−1 (queries predicted from the token's own KV, window of the last 64 tokens plus token 0); pivoted channel order; 8 steps; adaptive spread | computed −log2 P |
| Ours, step k | no | The `fast` predictor run closed loop on the receiver's reconstructed KV. The residual is quantized with step k × the channel's typical error and entropy-coded. Each value's error is at most half a step plus BF16 rounding. | computed −log2 P |
| AQUA-KV-style, b-bit | no | Closed loop: keys from the previous layer's keys, values from the previous layer's values plus current keys (RoPE removed). b-bit min-max residual per token and 64 channels; layer 0 direct at 4 bits | b + 0.5 |
| KIVI-style, b-bit | no | Keys per channel over 128-token blocks, values per token over 64 channels | b + 0.375 |
| FP8 cast | no | e4m3, no scale | 8 |

Notes:

- The reference range coder matched computed costs within 0.011 bits per value in every SmolLM2 test.
- AQUA-KV quantizes its residual with HIGGS, a vector quantizer. That is more accurate than our scalar stand-in at the same bits, so the AQUA-KV-style rows understate the real method. In particular, the 2-bit stand-in can diverge.

## Reference: SmolLM2-135M (same protocol, NumPy implementation)

Lossless, bits per value, averaged over the four reference chunks (all 30 layers):

| Method | Bits per value |
| --- | --- |
| SplitZip-style | 12.01 |
| ExANS-style | 10.75 |
| Our 44% version | 9.02 (round trip) |
| Our Block32 | 8.36 (round trip) |
| Our strongest no-hint | 7.83 (round trip) |
| Our strong-and-fast | ≈7.77 (cost test on layers 13 and 25) |

Lossy, on a 3,072-token prompt with quality measured on the next 512 tokens:

| Method | Bits per value (text / code) | Perplexity change (text / code) |
| --- | --- | --- |
| Ours, step 0.5 | 4.12 / 3.87 | −0.0% / +0.8% |
| Ours, step 1 | 3.22 / 3.01 | −0.2% / +0.4% |
| Ours, step 2 | 2.50 / 2.36 | +2.0% / +3.7% |
| AQUA-KV-style, 4-bit | 4.50 | +2.2% / +0.5% |
| KIVI-style, 4-bit | 4.38 | −0.3% / +1.2% |
| KIVI-style, 3-bit | 3.38 | +4.9% / +7.1% |
| FP8 cast | 8.00 | −0.2% / +0.4% |

The SmolLM2 lossy predictor used layers 0, l−2 and l−1 with 12 natural-order steps, without the attention input. Perplexity differences under about ±0.5% are within noise for 511 scored tokens.

## What to look for on Llama-3.2-3B

- **Lossless ordering.** Does the SmolLM2 ordering hold (ExANS-style > prev > block > nohint ≥ fast)?
- **Rate-quality frontier.** At equal perplexity change, how many bits does each lossy method need?
- **Compute.** The predictor's multiply-adds per value, listed in `summary.md`. Llama-3.2-3B has a 2,048-wide KV layer, the same as Llama-3.1-8B, so the strong configurations cost about 9,000–15,000 multiply-adds per value. That cost decides whether a GPU kernel can keep up with the link.
