# Milestone report: KV-cache compression on Llama-3.2-3B

29 October 2026 · branch `llama32-3b-lossy-lossless-compare` · latest results `8ae8d22` (run at `331155e`),
code at `fb0fd9c` · NVIDIA RTX A4500 (20 GB)

This report records where the work stands for review while AQUA-KV's own training runs. It covers what we
built, how each method was evaluated, every core result with its size and speed, what failed, the caveats, and
the next steps. Numbers are provisional until the lock run (see [plan.md](plan.md)): they come from several
commits, each named where it is used.

## 1. Summary

**Lossless: our codec is the strongest on size, and wins transfers up to about 400 Gb/s.**
- One 8,192-token document's KV (896 MiB in BF16) becomes **504 MiB** as a real GPU stream (9.00 bits per
  value), against SplitZip's 675 MiB (12.05 bits) and ExANS's 608 MiB (10.86 bits). Every stream decodes bit-exact.
- Our strongest predictor costs 7.75 bits (434 MiB) by exact coding cost, but it has no GPU decoder yet.
- Transfer of one document: 1.34x faster than SplitZip on links up to 100 Gb/s, 1.25x at 200 Gb/s, 1.09x at
  400 Gb/s, and level at 800 Gb/s, where SplitZip itself is the best mode.
- SplitZip decodes 6 to 11 times faster (3 ms against our 17-33 ms). That is our main lossless weakness.

**Lossy: our codec is the fastest, but in AQUA-KV's own evaluation it does not yet match AQUA-KV's quality per
byte at 2-3 bits.**
- In AQUA-KV's protocol AQUA-KV loses +1.51% perplexity at 117 MiB per 8K window. We need 154 MiB for +0.54%,
  and at 122 MiB we lose +11.25%.
- Near-lossless we are about level: -0.03% at 202 MiB, against AQUA-KV 4-bit at -0.09% and 225 MiB.
- We beat KIVI and HIGGS at every rate.
- Decode takes 27 ms per document, against AQUA-KV's 78 ms and KIVI's 46 ms. Encode takes 55 ms, against about 1 s
  and 4.3 s.
- In our own transfer test (compress a prompt once, score its continuation) we beat AQUA-KV: 1.79 bits against
  2.11 at the same quality, with our CPU coder. AQUA-KV's protocol is stricter and the one with published
  numbers, so it decides.

**Diagnosis so far.**
- Our error is about as large as AQUA-KV's but costs about 7 times more perplexity: it is where the error goes.
- At equal bits our keys cost 2.8 times AQUA-KV's and our values 7.6 times.
- Steps shaped by the loss gradient made it worse. Steps that spread the error evenly over the channels are
  queued for the next run.

**Running now:** AQUA-KV's own predictor training on its RedPajama data (paper: perplexity 7.03 at 2.16 bits),
and our evenly spread steps.

## 2. Setup

| Item | Value |
| --- | --- |
| Model | `unsloth/Llama-3.2-3B`, revision `d4446454d87d51aa42e1fb174f25acc5f8762331` (ungated copy of the BF16 weights; the one AQUA-KV's README uses) |
| Shape | 28 layers, 24 query heads, 8 KV heads, head dimension 128: 2,048 cached values per token per layer (1,024 keys after RoPE, 1,024 values) |
| GPU | NVIDIA RTX A4500, 20 GB, Ampere |
| Software | PyTorch 2.11.0+cu128, Triton 3.6.0, Transformers 4.57.6, Python 3.12.3 |
| Calibration documents (ours) | `pep-0484.rst` and five standard-library modules (`typing`, `inspect`, `subprocess`, `tarfile`, `logging`), 8,192 tokens each; spreads from `dataclasses.py` |
| Test documents | `pep-0008.rst` (PEP 8) and `argparse.py`, 8,192 tokens each |
| WikiText-2 | `wikitext-2-raw-v1` test split, joined with blank lines as AQUA-KV reads it |

**Units.**
- One test document is 8,192 tokens x 28 layers x 2,048 = **469,762,048 values**, 896 MiB in BF16.
- **1 bit per value = 56 MiB per document** (and per AQUA-KV 8K window).
- GB/s counts raw BF16 bytes per second, as SplitZip and ExANS report it.
- "Exact cost" is the bits an ideal coder needs for the predicted distribution. Our reference range coder matched
  it within 0.011 bits per value. "Real stream" means encoded, decoded on the GPU, and checked bit for bit.
- **Transfer time** of a document over an ideal link is the longer of the wire time and the decode (decoding
  overlaps arrival). No real network transfer was measured.

## 3. Methods

### 3.1 Our lossless codec

**Prediction.** Each layer's KV of a token is predicted from earlier layers of the same token, then coded
against that prediction. The predictors are named by their inputs:

| Name | Inputs | Exact cost |
| --- | --- | --- |
| P, one step ("44% version") | layer l-1 of the token, the whole layer in one product | 8.96 bits |
| Block | P plus earlier 128-channel groups of the same layer, 16 steps | 8.24 bits |
| No-hint | adds layer 0, layer l-2, the token's earlier occurrence, an adaptive spread | 7.92 bits |
| L0+P2+P+ATT ("strong and fast") | layers 0, l-2, l-1 and the recomputed attention of layer l-1, pivoted channel order, 8 steps | 7.82 bits |
| L0+P2+P+ATT, 16 pivot steps | the same with 16 steps | **7.754 bits** (fewest) |
| L0+P2+P, 16 pivot steps | no attention input | **7.870 bits** |

Costs are the four-chunk averages. 37 variants were measured from one covariance per layer, with their multiply-adds
per value (the Pareto front is in `summary.md`, "Bits against compute").

**The GPU codec** uses the one-step predictor P:
- **Product:** the previous layer's 8-bit features times 8-bit coefficients, an exact integer product (int32
  result), then per-lane scale and bias in float32 and RoPE for keys. Encoder and decoder agree bit for bit.
- **Spread:** per channel, from calibration, times a per-token multiplier derived from the previous layer's
  surprise.
- **Coding:** the "value-slice" coder. A BF16 word is coded against a logistic distribution in two symbols, a
  slice of the predicted range, then the word inside the slice, with 16-bit rANS on one stream per lane. Words
  outside 24 spreads, Inf and NaN travel raw.
- **Layout:** a program is one warp, 128 lanes for the channels of one head, over a block of tokens. Streams stay
  open across layers.

| Mode | What it trades | Bits | Decode |
| --- | --- | --- | --- |
| Per layer, dense product | cuBLAS int8 product, then one Triton kernel per layer | 9.00-9.03 | 31.5-33.3 ms |
| Per layer, reduced rank 256 / 384 / 512 | two smaller 8-bit products (previous layer to r values and back) | 9.41 / 9.30 / 9.22 | 22.8 / 24.4 / 25.3 ms |
| Fused, one kernel per direction | everything in one Triton kernel; "exact" mode codes the word inside a slice by its rounding interval | 8.945 (exact) / 9.004 | 101 / 58.6 ms |
| SplitZip format | fixed-length codes (no entropy coder), the code book chosen by our prediction: 37 contexts, k-bit codes, escapes inline | 10.20 (rank 256) / 9.85 (dense) | 17.2 / 25.5 ms |
| Mix with SplitZip | per layer, ours or the authors' SplitZip codec, whichever the link favours | 9.4-12.05 | 3-31 ms |

### 3.2 Our lossy codec

- **GPU codec** (`kv_compare/lossy_gpu.py`), closed loop:
  - The prediction is the same one-step product, computed from the receiver's reconstruction of the previous
    layer.
  - Each value is quantized to the lattice point prediction + k x step, with step = kappa x the channel's spread,
    rounded to BF16. The error is at most half a step plus BF16 rounding.
  - k is coded with rANS against a logistic model. Out-of-range values travel exact.
  - The decoder reproduces the encoder's reconstruction bit for bit.
- **CPU coder:** the strong predictor (L0+P2+P+ATT) run closed loop, with exact costs, not a real stream.
- **Options:**
  - sinks: the first 4 tokens exact;
  - token step: the step times the token's surprise multiplier;
  - per-lane kappa (new, `fb0fd9c`): any step per layer and channel. The table class follows each lane, so the
    coder's model stays matched.

### 3.3 Baselines

| Baseline | Source | How it runs here | Agreement with the authors' code |
| --- | --- | --- | --- |
| SplitZip | Intelligent-Microsystems-Lab/SplitZip `42c7431` | the authors' GPU codec on our KV, and their benchmark script unchanged | their codec |
| ExANS | openlake `de6c38b` | the authors' CUDA codec, built from source with one compile fix (`fail_cuda(codec, …)` to `fail_cuda(&*codec, …)`); their test passes | their codec |
| AQUA-KV | goodevening13/aquakv `e744183` | re-implemented on our KV from their code (their cache needs older Transformers): HIGGS (Hadamard over 1,024 channels, per-token scale, EDEN grids of 16 / 64 / 256 points), closed-loop linear predictors, layer 0 at 4 bits, the first 4 tokens in BF16. Their own scripts run separately (`AQUA_OFFICIAL=1`) | HIGGS values equal for 99.94-99.99% of entries, the same relative error to 8 digits; regression weights within one BF16 step (`check_aqua.json`) |
| KIVI | jy-yuan/KIVI `876b4d2` | re-implemented from their code: keys per channel over token groups, values per token over channel groups, asymmetric min-max, recent tokens in BF16 | identical outputs (`check_kivi.json`) |
| HIGGS alone | from AQUA-KV's code | no predictor, group 1,024 | as AQUA-KV |

One deviation remains: our AQUA-KV predictors are fitted on our calibration documents, not on the 256 x 8,192
RedPajama tokens the authors use. That run is in progress (section 9).

## 4. How each method is evaluated

The rule: each baseline is compared under its own published evaluation. For each, we picked the one the NUC can run
and ran ours in it too.

| Evaluation | Whose | What it measures | Status |
| --- | --- | --- | --- |
| Exact lossless cost | ours | bits per value on the two test documents (whole, and four 256-token chunks) | done |
| Real streams and speed | ours, as SplitZip and ExANS publish speed | bit-exact round trip; CUDA-event medians after warm-up, kernel time only | done |
| SplitZip's benchmark (`bench_codec_throughput.py`) | SplitZip | code book from WikiText-2 train KV, benchmark on the KV of the first 4 WikiText-2 test prompts repeated to 268M values; ratio and GB/s | done, with Llama-3.2-3B in place of Qwen3-32B (does not fit in 20 GB) |
| ExANS's measurement | ExANS | ratio and median GB/s on a real KV snapshot; independent blocks and position-aligned batches | independent blocks done; batches not yet |
| Our lossy transfer test | ours | compress a prompt's KV once, then the model continues with it. Two documents (3,072-token prompt, next 512 tokens), and 8 WikiText-2 windows (3,072-token prompt, next 1,024 tokens, 8,192 scored tokens) | done |
| **AQUA-KV's WikiText-2 protocol** (`evaluate_perplexity.py`) | AQUA-KV (also how its paper runs KIVI) | see below | done for all methods; the authors' own trained run pending |
| LongBench, LM-Eval (CoQA, TruthfulQA, GSM8K), passkey | AQUA-KV, KIVI | task accuracy | not run (untruncated LongBench takes days on this GPU); subsets planned |
| Mooncake transfer, SGLang time to first token | SplitZip | end-to-end serving | integration not released |

**AQUA-KV's protocol, as implemented in `kv_compare/streaming.py`.**
- The whole test split is tokenized once and cut into windows of 8,192 tokens. That gives 35 windows and 286,685
  scored tokens; the incomplete last window is dropped.
- Each window is fed 32 tokens per forward pass.
- The first 4 tokens stay in BF16. Once at least 128 tokens have accumulated past them, the whole buffer is
  compressed layer by layer, and later passes read the reconstruction.
- Perplexity is computed with the authors' BF16 loss (comparable to the paper). A float32 loss from the same
  logits is used for comparing methods: the BF16 loss resolves only about 0.8% per window. Changes are given as
  the pooled change and as the per-window mean +- standard error.
- Bits are counted per compressed value, as the papers do.

**Harness validation.**
- Our exact perplexity is 6.9865, and the authors' own script on the NUC gives 6.98645 (paper: 6.98).
- HIGGS alone gives 7.554 (paper 7.47).
- KIVI 2-bit, group 128, gives 7.784, much better than the paper's 9.33. So our KIVI is not handicapped.

### Hardware and link speeds in each paper

| | SplitZip | ExANS | AQUA-KV | KIVI | This study |
| --- | --- | --- | --- | --- | --- |
| GPU | H200 server (two Xeon 8468) | H100 | A100 for speed (8B on one, 70B on two); accuracy GPU not named | A100 80 GB for speed and memory | RTX A4500 20 GB, every method |
| Models | Llama-3-8B, Qwen3-30B-A3B, Qwen3-32B | not named | Llama 3.2 3B, 3.1 8B / 70B; Qwen 2.5 | Llama-2 7B / 13B, Falcon-7B, Mistral-7B, Llama-3-8B-Instruct, LongChat-7B | Llama-3.2-3B |
| Context | 512 to 131K tokens | a 256K snapshot | 8K (WikiText-2), up to 128K (LongBench) | 4K-8K (LongBench), about 30K (needle) | 8K |
| Measured | codec GB/s; real KV transfer with Mooncake over RDMA (ConnectX-7; RoCE 4x200G, 700 Gb/s effective); SGLang time to first token | codec GB/s and ratio only | quality; tokens per second | quality; memory; throughput | everything on one GPU; transfers modelled |
| Link speeds discussed | 400 and 800 Gb/s Ethernet (about 50 and 100 GB/s); its codec hides below 463 GB/s | 400 Gb/s InfiniBand (about 50 GB/s) | none: in-memory compression, no transfer | none | 1-800 Gb/s |
| Headline | 1.32x transfer, 1.30x time to first token against uncompressed; 613 / 2,182 GB/s on the H200 | 1.51x; 391 / 638 GB/s on the H100 | 2.16 bits, perplexity 7.03 | 2-bit KV | sections 5-6 |

- **Datacenter fabrics only.** The two lossless papers target datacenter fabrics (400-800 Gb/s) on H100/H200-class
  GPUs. The two lossy papers compress the cache in GPU memory and do not transfer it.
- **GPU-dependent speeds.** On our A4500, ExANS decodes at 54 GB/s, against 638 GB/s on their H100. SplitZip
  decodes at 309 GB/s, against 2,182 GB/s on their H200. Bits per value do not depend on the GPU, but every speed
  and every 400-800 Gb/s conclusion does.

## 5. Lossless results

### 5.1 Size and speed of every real stream (one document, bit-exact)

| Codec | Bits per value | Size | Encode | Decode |
| --- | --- | --- | --- | --- |
| BF16 | 16 | 896 MiB | - | - |
| SplitZip (authors' codec) | 12.05 | 675 MiB | 6.8 ms (139 GB/s) | **3.0 ms (309 GB/s)** |
| ExANS (authors' codec) | 10.86 | 608 MiB | 176 ms (5.3 GB/s) | 17.5 ms (54 GB/s) |
| Ours, SplitZip format, rank 256 | 10.20 | 571 MiB | 1,000 ms (PyTorch packer) | 17.2 ms (55 GB/s) |
| Ours, per layer, rank 256 | 9.41 | 527 MiB | 24 ms (39 GB/s) | 22.8 ms (41 GB/s) |
| Ours, per layer, dense | 9.00 | 504 MiB | 37 ms (25 GB/s) | 33.3 ms (28 GB/s) |
| Ours, fused, exact mode | **8.945** | **501 MiB** | 55 ms (17 GB/s) | 101 ms (9.3 GB/s) |
| Ours, strongest predictor (exact cost only) | 7.754 | 434 MiB | - | no GPU decoder yet |

Per test document (PEP 8 / argparse.py): ExANS 10.864 / 10.874 bits; our strong-and-fast predictor 7.83 / 7.54.

### 5.2 Transfer of one document

| Link | Typical setting | SplitZip | ExANS | Ours (fastest measured mode) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | ordinary cloud, storage | 5,660 ms | 5,103 ms | 4,202 ms, 501 MiB (fused, exact) | 1.35x |
| 10 Gb/s | consumer, cloud Ethernet | 566 ms | 510 ms | 420 ms, 501 MiB | 1.35x |
| 50 Gb/s | workstation | 113 ms | 102 ms | 84 ms, 502 MiB (per layer, dense) | 1.34x |
| 100 Gb/s | cross-cluster | 56.6 ms | 51.0 ms | 42.3 ms, 504 MiB | 1.34x |
| 200 Gb/s | datacenter RDMA per GPU | 28.3 ms | 25.5 ms | 22.7 ms, 541 MiB (24 layers ours, rank 384, rest SplitZip) | 1.25x |
| 400 Gb/s | datacenter RDMA per GPU | 14.1 ms | 17.5 ms | 13.0 ms, 621 MiB (14 layers in our SplitZip format, rest SplitZip) | 1.09x |
| 800 Gb/s | next-generation RDMA | 7.1 ms | 17.5 ms | 7.1 ms, 675 MiB (SplitZip itself; no layer of ours helps) | 1.00x |

At 400 Gb/s, accepting a little more time buys bytes: +10% time (14.3 ms) sends 610 MiB, +25% (16.2 ms) 585 MiB,
+50% (19.2 ms) 567 MiB.

### 5.3 SplitZip's own benchmark (its paper protocol, Llama-3.2-3B)

| Method | Bits per value | Ratio | Encode | Decode |
| --- | --- | --- | --- | --- |
| SplitZip, the authors' script | 12.046 | 1.328 | 136 GB/s | 313 GB/s |
| Ours, one-step predictor (exact cost) | 9.919 | 1.613 | - | - |
| Ours, strong and fast (exact cost) | 8.684 | 1.843 | - | - |

The prompts are short (157-225 tokens), so the bits are a little higher than on 8K documents. GPU speed on prompts
this short is not meaningful; the speeds are in 5.1.

### 5.4 A KV pool in GPU memory (one cached 8K prefix)

| Method | Memory | Prefixes in 12.4 GiB | Bring it back |
| --- | --- | --- | --- |
| Recompute (the model's prefill) | 0 | - | 992 ms |
| Raw BF16 | 896 MiB | 14 | 0 ms (39 ms from CPU memory) |
| SplitZip | 675 MiB | 18 | 3 ms |
| ExANS | 608 MiB | 20 | 18 ms |
| Ours, fused exact | 501 MiB | 25 | 101 ms |
| Ours, per layer, rank 256 | 527 MiB | 24 | 23 ms |

### 5.5 Where the lossless bits go (SplitZip format, dense)

| Part | Bits per value |
| --- | --- |
| Codes | 4.78 |
| Raw bits | 4.08 |
| Escapes (6.1% of values) | 0.97 |
| Total | 9.83 |
| Code books fitted on the test document itself | 9.80 |
| Entropy given the context | 8.96 |

The code books are within 0.03 bits of the best fixed code book. Our rANS codec (9.0 bits) already sits at the
entropy of these symbols, so fewer lossless bytes need a stronger prediction, not better coding.

## 6. Lossy results

### 6.1 Our transfer test (8 WikiText-2 windows; prompt compressed once)

| Method | Bits per value | KV error | Perplexity change (± SE over windows) |
| --- | --- | --- | --- |
| AQUA-KV 2-bit | 2.105 | 11.6% | +0.92% ± 0.21 |
| AQUA-KV 3-bit | 3.068 | 6.2% | +0.04% ± 0.07 |
| AQUA-KV 4-bit | 4.031 | 3.4% | -0.04% ± 0.06 |
| KIVI 2-bit (group 32) | 3.271 | 21.9% | +6.56% ± 0.85 |
| KIVI 4-bit | 5.229 | 4.4% | +0.08% ± 0.09 |
| FP8 cast | 8.000 | 2.6% | +0.08% ± 0.03 |
| Ours, CPU coder, step 4, sinks + token step | **1.791** | 19.8% | +0.80% ± 0.16 |
| Ours, CPU coder, step 2, sinks + token step | **2.209** | 7.3% | +0.04% ± 0.10 |
| Ours, GPU codec, step 4, sinks | 2.180 | 12.9% | +1.11% ± 0.11 |
| Ours, GPU codec, step 2, sinks | 2.782 | 6.5% | +0.04% ± 0.08 |
| Ours, GPU codec, step 4, **no sinks** | 2.162 | 12.9% | +418% |

- **The 4 sink tokens are essential:** without them our codec at about 2.2 bits fails.
- **In this test we beat AQUA-KV.** The CPU coder matches its 3-bit quality at 2.21 bits and its 2-bit quality at
  1.79 bits.
- **The GPU codec matches AQUA-KV 3-bit at 2.78 bits.** It is about level with AQUA-KV 2-bit at 2.18 bits.

### 6.2 AQUA-KV's protocol (35 windows of 8,192 tokens, cache compressed while it grows)

| Method | Bits per compressed value | One window | Perplexity (BF16 loss) | Change (float32 loss) ± SE | Paper (Table 1) |
| --- | --- | --- | --- | --- | --- |
| Exact BF16 cache | 16 | 896 MiB | 6.986 | - | 6.98 |
| HIGGS 2-bit, no predictor | 2.016 | 113 MiB | 7.554 | +8.08% ± 0.27 | 2.02 bits, 7.47 |
| KIVI 2-bit, group 128 | 2.250 | 126 MiB | 7.784 | +11.41% ± 0.72 | 2.25 bits, 9.33 |
| KIVI 2-bit, group 32 | 3.000 | 168 MiB | 7.274 | +4.12% ± 0.23 | - |
| KIVI 4-bit, group 32 | 5.000 | 280 MiB | 6.988 | +0.08% ± 0.01 | - |
| **AQUA-KV 2-bit** (our re-implementation) | 2.087 | 117 MiB | 7.093 | **+1.51% ± 0.13** | 2.16 bits, 7.03 |
| AQUA-KV 3-bit | 3.051 | 171 MiB | 6.991 | +0.08% ± 0.05 | - |
| AQUA-KV 4-bit | 4.016 | 225 MiB | 6.983 | -0.09% ± 0.03 | - |
| Ours, GPU codec, step 4 | 2.172 | 122 MiB | 7.771 | +11.25% ± 0.97 | - |
| Ours, step 2, token step | 2.296 | 129 MiB | 7.160 | +2.53% ± 0.19 | - |
| **Ours, step 2** | 2.757 | 154 MiB | 7.022 | **+0.54% ± 0.06** | - |
| Ours, step 1 | 3.606 | 202 MiB | 6.983 | -0.03% ± 0.03 | - |
| Ours, step 0.5 | 4.563 | 256 MiB | 6.986 | +0.02% ± 0.01 | - |
| Ours, step 4, token step | 2.594 | 145 MiB | 20.74 | +197% | - |

Against each baseline, the fewest bytes we need for at most its perplexity change:

| Baseline | Its size, change | Ours |
| --- | --- | --- |
| HIGGS 2-bit | 113 MiB, +8.08% | 129 MiB, +2.53% (step 2, token step) |
| KIVI 2-bit, group 128 | 126 MiB, +11.41% | 122 MiB, +11.25% (step 4) |
| KIVI 2-bit, group 32 | 168 MiB, +4.12% | 129 MiB, +2.53% |
| KIVI 4-bit | 280 MiB, +0.08% | 202 MiB, -0.03% (step 1) |
| **AQUA-KV 2-bit** | **117 MiB, +1.51%** | 154 MiB, +0.54% (step 2): 32% more bytes |
| **AQUA-KV 3-bit** | **171 MiB, +0.08%** | 202 MiB, -0.03% (step 1): 18% more bytes |
| AQUA-KV 4-bit | 225 MiB, -0.09% | none at this quality yet |

### 6.3 The same settings in the two lossy tests

| Method | Transfer test | AQUA-KV's protocol | Loss grows by |
| --- | --- | --- | --- |
| AQUA-KV 2-bit | 2.11 bits, +0.92% | 2.09 bits, +1.51% | 1.6x |
| AQUA-KV 3-bit | 3.07 bits, +0.04% | 3.05 bits, +0.08% | about 2x |
| Ours, GPU codec, step 2 | 2.78 bits, +0.04% | 2.76 bits, +0.54% | about 13x |
| Ours, GPU codec, step 4 | 2.18 bits, +1.11% | 2.17 bits, +11.25% | 10x |

Our codec degrades far more than AQUA-KV's when the cache is compressed while it grows. Two things are likely to
contribute:
- **Compounding.** In the protocol each new token's KV is computed from hidden states that already read compressed
  values, and is then compressed in turn.
- **Window length.** The protocol's 8K windows put more compressed tokens before each query than the transfer
  test's 3K prompts.

The keys/values test below shows the compounding directly at step 4.

**Which test fits which use.**
- **Prefill/decode disaggregation:** the prompt's KV is computed exactly on the prefill GPU and sent once. The
  decode GPU generates from it and computes every new token's KV itself. That is our transfer test, where we are
  ahead of AQUA-KV.
- **Memory pools, or a cache compressed while decoding:** that is AQUA-KV's protocol, where it is ahead.

### 6.4 Which half costs us (keys only or values only compressed, the other half exact)

| Method | Keys only | Values only | Both | Both, if the halves added up |
| --- | --- | --- | --- | --- |
| AQUA-KV 2-bit (2.09 bits) | +1.10% | +0.30% | +1.51% | +1.40% |
| Ours, step 2 (2.76 bits) | +0.34% | +0.21% | +0.54% | +0.55% |
| Ours, step 4 (2.16 bits) | +3.13% | +2.29% | +11.25% | +5.50% |

- **Both halves are worse than AQUA-KV's.** At about the same bits our keys cost 2.8 times AQUA-KV's and our
  values 7.6 times.
- **At step 4 the two errors compound to twice their sum.** The step is then about 2.2 standard deviations of the
  prediction residual, so most values are rounded to the prediction itself. The keys and the values of a token
  lose the same unforeseen part.
- **The loss is spread over all 35 windows:** the worst three hold 16-17% of it.
- **Error size is not the problem.** In the transfer test our error at 2.2 bits (12.9% of the KV) is about
  AQUA-KV's at 2.1 bits (11.6%). The difference is where the error goes: our step is proportional to each
  channel's spread, so every channel gets the same relative error. AQUA-KV rotates the residual (Hadamard) and
  gives each token one scale, which spreads the error evenly over the channels.

### 6.5 Error shaping, first attempt: steps from the loss gradient (negative)

Each lane's step was made proportional to 1 / sqrt(its diagonal Fisher information) (`kv_compare/sensitivity.py`):
- the gradient of a prefill's loss with respect to the cached KV, on 24 prefills of 2,048 tokens of our
  calibration documents (20 s);
- checked against finite differences;
- multipliers within [1/16, 16].

| Step | Unshaped | Shaped over all lanes | Shaped within each head |
| --- | --- | --- | --- |
| 1 | 202 MiB, -0.03% | 207 MiB, +4.92% | - |
| 2 | 154 MiB, +0.54% | 161 MiB, +10.10% | 157 MiB, +5.85% |
| 3 | - | 139 MiB, +12.76% | - |
| 4 | 122 MiB, +11.25% | 125 MiB, +12.73% | 122 MiB, +6.46% |
| 6 / 8 | - | 110 MiB, +12.69% / 104 MiB, +18.15% | - |

- **Worse at every step.** The loss stays flat from step 3 to step 6, so it comes from the lanes made 16 times
  coarser, which are rounded to the prediction at any step. The gradient called those lanes insensitive; in the
  protocol they are not.
- **The layer ranking was wrong too.** The sensitivity put the early layers first (steps x0.5-0.8 there, x1.5-2.8
  in layers 16-27). The result says the late layers matter more than that.
- **The likely reason:** the gradient is first-order and local. In a prefill it is dominated by nearby queries,
  which in the protocol still read exact recent tokens, and the errors here are not small.

### 6.6 Lossy speed (one document)

| Method | Encode | Decode |
| --- | --- | --- |
| AQUA-KV (authors' code) | 951 ms (2-bit), 1,418 ms (3-bit), 2,434 ms (4-bit) | 78 ms (12 GB/s) |
| KIVI (authors' kernels) | 4,330 ms | 46 ms (20 GB/s) |
| **Ours, GPU codec, dense** | **55 ms** (25.5 quantize + 29.8 code) | **27 ms (35 GB/s)** |
| Ours, GPU codec, rank 256 | 37.5 ms | 18.5 ms (51 GB/s) |

- **Ratios:** we decode 2.9 times faster than AQUA-KV and 1.7 times faster than KIVI, and encode 17 times faster
  than AQUA-KV 2-bit.
- **Our decode time hardly depends on the step:** 26.7 ms at step 4, 28.1 ms at step 0.25.
- **The baselines' encode times are their released code timed on the whole document at once,** which is not what
  those papers optimize.
- **The rank-256 codec has not been through AQUA-KV's protocol.**

## 7. Correctness checks

- **Lossless streams.** Every stream in sections 5.1-5.4 is decoded on the GPU and compared bit for bit with the
  captured BF16 words. That covers SplitZip, ExANS, and all our modes and mixes.
- **Lossy decoder.** It reproduces the encoder's reconstruction bit for bit. Each value's error is checked against
  half its lane's step plus BF16 rounding, and Inf/NaN and the sink tokens travel exact.
- **Unit tests.** All 16 suites pass at `89ab4f2` (CPU, Triton interpreter, a small random Llama). They cover:
  - every codec's round trip;
  - the streaming cache's buffering (blocks of 156, 128 and 128 tokens after the 4 sinks);
  - the gradient against finite differences;
  - the step shaping;
  - packaging.
- **Baselines against the authors' functions:** see the table in 3.3.
- **AQUA-KV protocol harness:** it reproduces the authors' exact perplexity (6.9865 against 6.98645).

## 8. What did not work

| Idea | Result | Kept? |
| --- | --- | --- |
| Fused one-kernel lossless codec | fewest real-stream bits (8.945), but 101 ms decode | as the slow-link mode |
| CUDA graphs; a small Triton product beside the coding | no overlap: the coding kernel needs every SM's registers | no |
| One read per value (`row1` layout) | coding 12.4 ms against 11.7 ms; +0.11 bits | no |
| Second product inside the token-by-token kernel | 34 ms against 18 ms | no |
| 8-bit codes; code books fitted per document | no gain; within 0.03 bits | no |
| Predicting each head only from its own head | +0.59 bits (SmolLM2-135M) | no |
| Per-token step in the GPU codec | helps in the transfer test (with sinks), breaks AQUA-KV's protocol at step 4 (perplexity 20.7): drift raises the surprise, which raises the step | off |
| Steps from the loss gradient (Fisher) | worse at every step (6.5) | no |

## 9. In progress

1. **AQUA-KV's own scripts, unchanged** (`AQUA_OFFICIAL=1`): predictors trained on RedPajama with the 2-bit HIGGS
   grid, then their perplexity evaluation with prefix 4 and buffer 128.
   - **Data:** `togethercomputer/RedPajama-Data-1T-Sample` has been removed from the Hub. The authors' sampler now
     draws from `ll922/RedPajama-Data-1T-Sample-Backup`, a Parquet mirror with the original's 930,514 rows
     (loaded on the NUC in `8ae8d22`). The same rows in the same order give the authors' seed the same samples.
   - **Tokenizer:** their slow-tokenizer call returns `False` for Llama 3 under their Transformers 4.46.3. In that
     case only, the fast tokenizer is used, and the choice is recorded.
   - **Memory:** training keeps about 168 MB per sample in RAM (43 GB for 256 samples). With less free RAM the
     script trains on fewer samples and puts the count in the result label.
   - **What to look for:** the paper reports 7.03 at 2.16 bits, about +0.7%. Our re-implementation gives +1.51% at
     2.09 bits. If the authors' run is near 7.03, the target for our codec gets harder.
2. **Evenly spread steps** in AQUA-KV's protocol:
   - `ours<step>u`: one absolute step for all channels of a head (steps 1-4);
   - `ours<step>w`: one for all keys and one for all values of a layer, AQUA-KV's HIGGS group (steps 2 and 4).

   This is the minimum-error allocation for equally important channels, and close to the error shape after
   AQUA-KV's rotation. It needs no sensitivity estimate.

## 10. Next steps, in order

1. **Close the lossy gap in AQUA-KV's protocol.**
   - Target: +1.5% or less at 117 MiB per window (2.1 bits) or less, and +0.1% at 168 MiB (3.0 bits) or less.
   - a. Read the evenly spread steps (running).
   - b. If they are not enough, quantize the residual in a Hadamard-rotated domain with a scale per token, as
     AQUA-KV does, keeping our entropy coder. About a day of kernel work.
   - c. A sensitivity measured in the protocol itself: errors injected into older tokens only, so nearby queries
     read exact values as they do there.
   - d. The strong predictor in the lossy GPU codec. In the transfer test it saves about 20% of the bits at equal
     quality (2.21 against 2.78 bits).
2. **Lock the results.** One full run on one commit (`LOCK=1`, about 1.5 hours), then
   `benchmark_results/locked_scoreboard.json`. Every later report compares against it.
3. **A GPU decoder for the 7.87-bit predictor** (layers 0, l-2, l-1, 16 groups, no attention).
   - Gain: 441 MiB per document instead of 504.
   - Products alone measure 53 ms per document, so we expect 55-70 ms. That wins up to 100 Gb/s at about 1.5x
     SplitZip.
4. **400-800 Gb/s.** Our fastest decode, 17 ms, splits into 7 ms of products and 12 ms of coding. It must reach
   12 ms or less to beat SplitZip alone at 400 Gb/s.
5. **Baseline coverage.** ExANS position-aligned batches; LongBench and GSM8K subsets for AQUA-KV and KIVI.
6. **Deployment pieces.**
   - A fast GPU encoder for the SplitZip format: the PyTorch packer takes 1 s.
   - A compact transfer header.
   - The predictor's coefficients shipped once: about 117 MB per model, one-step, 8-bit.

## 11. Caveats

- **Provisional numbers.** They are assembled from several commits until the lock run.
- **AQUA-KV is our re-implementation,** calibrated on our documents, until the authors' trained run lands. Checked
  against their functions (3.3).
- **Different hardware and model from the papers.** SplitZip's paper uses an H200 and Qwen3-32B; ExANS's an H100 at
  256K context. Ours is an A4500 with Llama-3.2-3B, an 8K context. The baselines' codecs run on our GPU, so the
  comparison here is like for like, but their speeds here are 5-12 times below their papers', and so would ours
  be against an H100. Results at 400-800 Gb/s hold only for this GPU class.
- **Modelled transfers.** Times are max(wire, decode) on an ideal link; there is no real network or serving
  integration.
- **Exact costs, not streams,** for the strongest lossless predictors (7.75-7.87 bits) and for the lossy CPU coder.
  The CPU coder has not been run in AQUA-KV's protocol.
- **A small transfer test:** 2 documents and 8 WikiText-2 windows. AQUA-KV's protocol has 35 windows, and the
  standard errors are given.
- **No task benchmarks yet:** no LongBench, GSM8K or passkey results.

## 12. Reproduce

On the NUC (folder `~/kv_llama32`, run folder `~/kv_reconstruction/runs/llama32_3b_compare`).
`scripts/run_on_nuc.sh` runs the stages, packages the results into `benchmark_results/<name>` and pushes them.

```bash
cd ~/kv_llama32 && git checkout llama32-3b-lossy-lossless-compare \
    && git pull --ff-only origin llama32-3b-lossy-lossless-compare

# what is running now: AQUA-KV's own training and evaluation, plus the new methods in its protocol
AQUA_OFFICIAL=1 STAGES=aquappl,report bash scripts/run_on_nuc.sh

# the full lock run (every stage except the captures, about 1.5 hours)
LOCK=1 bash scripts/run_on_nuc.sh

# SplitZip's own benchmark
SPLITZIP_OFFICIAL=1 STAGES=report bash scripts/run_on_nuc.sh
```

Without the GPU: `python -m unittest kv_compare.test_kv_compare` (about 20 minutes on a CPU, Triton interpreter).

## 13. Index

| Result folder (`benchmark_results/`) | Commit | What it added |
| --- | --- | --- |
| `llama32_3b_compare_a4500_20261027_1510` | `03c17bc` | lossy first pass; the predictor frontier (37 variants) |
| `..._20261027_2107`, `..._20261028_0620`-`0846` | `f8b4c2d`, `7f9434e`, `fe1b577`, `6792614`, `28ef51d`, `c96f7be` | GPU speed: fused and per-layer codecs, reduced-rank products, product epilogue; the pool |
| `..._20261028_1029` | `474524c` | the authors' baselines (AQUA-KV with HIGGS, KIVI, ExANS's codec); our prediction in SplitZip's format (bits) |
| `..._20261028_1132` | `7dc5fb6` | mixes with SplitZip; the sink tokens; the pool with lossy rows |
| `..._20261028_1349`, `..._1439` | `123a83e`, `610f1d3` | WikiText-2 lossy quality; the SplitZip-format codec (tile, then row kernel) |
| `..._20261028_1548` | `612a70d` | decoder ideas that did not pay; the bit budget; bytes against time |
| `..._20261028_1947`, `..._2013` | `61d6db1`, `7b5bf1a` | AQUA-KV's protocol for every method; SplitZip's benchmark (`2013`: report only) |
| `..._20261028_2119` | `8f8bfd1` | keys-only and values-only diagnostics |
| `..._20261029_0518` | `8ae8d22` | steps from the loss gradient (negative); the RedPajama mirror loads |

| File | Contents |
| --- | --- |
| `summary.md`, `scoreboard.md` (in each result folder) | full tables of that run; the scoreboard first |
| `docs/plan.md` | the locked-results procedure, our best so far, what is next |
| `docs/llama_lossy_lossless_comparison.md` | how to run, methods, results per run |
| `kv_compare/layerwise.py`, `fused.py`, `gpu_coder.py` | lossless GPU codecs |
| `kv_compare/szcodec.py`, `mixed.py` | our prediction in SplitZip's format; mixes with SplitZip |
| `kv_compare/lossy_gpu.py` | the lossy GPU codec (per-lane steps since `89ab4f2`) |
| `kv_compare/streaming.py` | AQUA-KV's perplexity protocol for every method |
| `kv_compare/sensitivity.py` | loss sensitivity of the KV; shaped and uniform steps |
| `kv_compare/baselines.py` | AQUA-KV, KIVI, HIGGS, SplitZip-style and ExANS-style |
| `scripts/bench_llama_compare.py` | all stages and the report |
| `scripts/run_on_nuc.sh` | the NUC driver (official baselines, packaging, push) |
| `scripts/scoreboard.py` | the scoreboard and the lock |
| `scripts/aqua_calibration_data.py` | AQUA-KV's calibration samples from the RedPajama mirror |
