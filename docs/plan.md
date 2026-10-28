# Locked results and plan (Llama-3.2-3B, RTX A4500)

The fixed record of our best results against the baselines, and the order of work to improve them. The numbers
come from `scripts/scoreboard.py`; `summary.md` opens with the same scoreboard in every run.

## How results are locked

1. **The lock run** measures everything once on one commit. It covers every stage except the captures:
   - lossless costs and the predictor frontier;
   - lossy on the two test documents and on WikiText-2;
   - GPU speed, the SplitZip-format codec, the pool and ExANS;
   - the authors' AQUA-KV and KIVI checks and timings.

   It takes about 1.5 hours:

   ```bash
   cd ~/kv_llama32 && git checkout llama32-3b-lossy-lossless-compare \
       && git pull --ff-only origin llama32-3b-lossy-lossless-compare && LOCK=1 bash scripts/run_on_nuc.sh
   ```

2. Its `scoreboard.json` is copied to `benchmark_results/locked_scoreboard.json` and committed. That file is
   the lock.
3. Every later report shows each scoreboard entry with the locked value in brackets. An improvement replaces the
   lock only after a full lock run, so locked numbers always come from one commit.

## Our best so far (provisional until the lock run)

Assembled from runs `7dc5fb6`, `123a83e` and `612a70d`. Test documents: PEP 8 and argparse.py, 8,192 tokens
each, 470M values per document.

| Use | Our best | Baseline |
| --- | --- | --- |
| Lossless, fewest bits (exact cost) | 7.75 bits (layers 0, l−2, l−1 + attention, 16 groups); 7.87 without attention | SplitZip 12.05, ExANS 10.86 |
| Lossless, real GPU stream | 9.00 bits, 33 ms decode (rANS, dense one-step); 8.95 bits at 101 ms (fused, exact mode) | SplitZip 12.05 bits, 3 ms |
| Lossless, fastest of ours | 10.20 bits, 17 ms (SplitZip format, rank 256) | |
| Transfer, up to 100 Gb/s | 1.34-1.35x SplitZip, 25% fewer bytes | |
| Transfer, 200 Gb/s | 1.25x | |
| Transfer, 400 Gb/s | 1.09x (format + SplitZip mix) | |
| Transfer, 800 Gb/s | 1.00x (SplitZip itself) | |
| Lossy at AQUA-KV 3-bit quality (WikiText-2, +0.04%) | 2.21 bits (CPU coder); 2.78 bits (GPU codec) | AQUA-KV 3.07 bits |
| Lossy at AQUA-KV 2-bit quality (+0.92%) | 1.79 bits (CPU coder); 2.18 bits (GPU codec) | AQUA-KV 2.11 bits |
| Lossy decode, one document | 27 ms (GPU codec) | AQUA-KV 78 ms, KIVI 46 ms |
| Pool, cached 8K prefixes in free GPU memory | 25 lossless (504 MiB, 33 ms back); 81 lossy at AQUA-3-bit quality | SplitZip 18, AQUA-KV 3-bit 74 |

What we learned:
- The SplitZip-format code books are within 0.03 bits of the best fixed code book. The rANS coder already sits
  at the entropy of these symbols, so fewer lossless bytes need a stronger prediction.
- At 800 Gb/s no layer of ours beats SplitZip yet.

## Each baseline evaluated its own way

Each method is compared under the baselines' own evaluations, from their papers and released code. For each
baseline we pick the evaluation the NUC (RTX A4500, 20 GB) can run, and run ours in it too.

### AQUA-KV ([paper](https://arxiv.org/abs/2501.19392), goodevening13/aquakv `e744183`)

- **Their evaluation:**
  - **WikiText-2 perplexity** (`evaluate_perplexity.py`), on the base model. The whole test split is cut into
    8,192-token windows, and each window is fed 32 tokens at a time to a compressed cache. The first 4 tokens
    and the most recent tokens (at least 128 before a block is compressed) stay in BF16, and every older token is
    compressed and read back. Bits are counted per compressed value.
  - **LongBench v1:** 14 English tasks, instruct models, inputs not truncated.
  - **Calibration:** 256 x 8,192 RedPajama tokens.
  - **Their Llama-3.2-3B numbers** (Table 1):

    | Method | Bits per value | WikiText-2 perplexity |
    | --- | --- | --- |
    | Exact | 16 | 6.98 |
    | AQUA-KV, HIGGS 2-bit | 2.16 | 7.03 |
    | KIVI-2b-gs128-r128 | 2.25 | 9.33 |
    | HIGGS-2b-gs1024 | 2.02 | 7.47 |
- **On the NUC:** the full perplexity protocol, all 35 windows. Stage `aquappl` (`kv_compare/streaming.py`)
  runs every method in one harness:
  - exact;
  - HIGGS without predictors;
  - KIVI with AQUA-KV's setting (group 128) and the KIVI paper's (group 32);
  - our faithful AQUA-KV re-implementation;
  - our GPU codec, each block coded to a real stream.

  `AQUA_OFFICIAL=1` also runs the authors' own scripts unchanged: no quantization, predictor training on
  RedPajama, then perplexity. The harness is right when its exact, HIGGS and KIVI rows reproduce 6.98, 7.47 and
  9.33, and the authors' run gives 7.03.
- **LongBench:** a subset later. Untruncated inputs take days on the A4500.

### KIVI ([paper](https://arxiv.org/abs/2402.02750), jy-yuan/KIVI `876b4d2`)

- **Their evaluation:**
  - LongBench, 15 tasks: group 32, recent 128 tokens in full precision; LongChat, Mistral and Llama models.
  - LM-Eval: CoQA, TruthfulQA and GSM8K.
  - Passkey retrieval.

  AQUA-KV's paper also runs KIVI in its perplexity protocol (group 128, recent 128).
- **On the NUC:** AQUA-KV's perplexity protocol, with KIVI 2- and 4-bit. A LongBench subset and GSM8K come
  later.

### SplitZip ([paper](https://arxiv.org/abs/2605.01708), Intelligent-Microsystems-Lab/SplitZip `42c7431`)

- **Their evaluation** (lossless):
  - compression ratio including metadata, and encode/decode GB/s;
  - the KV comes from WikiText-2 test prompts and the code book is calibrated on WikiText-2 train
    (`bench_codec_throughput.py`: 268M values, timed after warm-up, bitwise round trip checked);
  - Mooncake transfer time and SGLang TTFT, but the integration is not released;
  - H200, Qwen3-32B.
- **On the NUC:** their script unchanged with Llama-3.2-3B, since Qwen3-32B does not fit in 20 GB. Ours on the
  KV of the same test prompts. Already measured: the authors' codec on our KV, 12.05 bits, 309 GB/s decode.

### ExANS (openlake `de6c38b`, [blog](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache))

- **Their evaluation** (lossless): compression ratio, stored bits per value, and median encode/decode GB/s on a
  real KV snapshot (H100, 256K context), with independent blocks and with position-aligned batches. No
  benchmark script is released.
- **On the NUC:** the authors' codec, built from source, on our captured KV. Done: 10.86 bits, 54 GB/s decode.
  The position-aligned batch mode is still to measure.

**Why this matters for the lossy comparison.** Our lossy tests so far are a transfer protocol: compress a prompt
once, then score its continuation. AQUA-KV's protocol is stricter, because every token is scored against a cache
that is compressed while it grows. It is also the one with published Llama-3.2-3B numbers, so it becomes the
main lossy comparison.

## What is next, in order of expected gain

0. **Run the baselines' own evaluations** (above): stage `aquappl` and `AQUA_OFFICIAL=1` first; then SplitZip's script on Llama-3.2-3B, and LongBench and GSM8K subsets.
1. **A GPU decoder for the 7.87-bit predictor** (layers 0, l−2, l−1, 16 groups, no attention).
   - Gain: 441 MiB per document instead of 504, 13% fewer bytes than our current stream and 35% fewer than
     SplitZip.
   - Products alone measure 53 ms per document, so the expected decode is 55-70 ms. That wins up to 100 Gb/s, at
     about 1.5x SplitZip where the link sets the time, and enlarges the pool by 14%.
   - Target: a bit-exact stream at 7.9-8.0 bits, decoding in 70 ms or less.
2. **The same predictor in the lossy GPU codec.**
   - At AQUA-KV 3-bit quality our CPU coder needs 2.21 bits and the GPU codec 2.78. Most of the gap is the
     predictor.
   - Target: 2.3 bits or less at +0.05% on WikiText-2, decoding in 40 ms or less.
   - Also fix the GPU per-token step: it helps the CPU coder but not yet the GPU codec (2.32 bits, +0.57%).
3. **400-800 Gb/s.** The SplitZip-format decode takes 17 ms: 7 ms of products and 12 ms of coding.
   - It must reach 12 ms or less to beat SplitZip alone at 400 Gb/s.
   - Ideas not yet tried: a two-head program, so the in-kernel product has enough warps; fewer instructions per
     value in the context and word rebuild; the first product's requantization fused into its epilogue output
     layout.
   - 800 Gb/s needs about 4x faster decoding and is not a near-term target.
4. **Baseline fidelity.** One deviation remains: AQUA-KV is calibrated on our calibration documents, not on the
   256 x 8192 RedPajama tokens of the authors' README. Rerun it with their calibration data.
5. **Deployment pieces.**
   - (T) A compact transfer header: model, predictor version, mode, layer and token ranges. The receiver needs
     the predictor's coefficients, about 117 MB per model for one-step (8-bit), shipped once.
   - (P) GPU-pool experiments.
   - A fast GPU encoder for the SplitZip format: the PyTorch packer takes 1 s per document.
