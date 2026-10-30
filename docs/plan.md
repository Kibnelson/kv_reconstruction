# Locked results and plan (Llama-3.2-3B, RTX A4500)

The fixed record of our best results against the baselines, and the order of work to improve them. The numbers
come from `scripts/scoreboard.py`; `summary.md` opens with the same scoreboard in every run.

The full review of where things stand on 29 October 2026, with every core result, the methods and protocols,
what failed and the caveats: [milestone_report_2026-10-29.md](milestone_report_2026-10-29.md).

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
| Lossy, our transfer test, at AQUA-KV 3-bit quality (WikiText-2 windows, +0.04%) | 2.21 bits (CPU coder); 2.78 bits (GPU codec) | AQUA-KV 3.07 bits |
| Lossy, our transfer test, at AQUA-KV 2-bit quality (+0.92%) | 1.79 bits (CPU coder); 2.18 bits (GPU codec) | AQUA-KV 2.11 bits |
| **Lossy, AQUA-KV's own protocol** (GPU codec) | 2.76 bits at +0.54%; 3.61 bits at -0.03% | AQUA-KV 2.09 bits at +1.51%, 3.05 bits at +0.08% |
| Lossy decode, one document | 27 ms (GPU codec) | AQUA-KV 78 ms, KIVI 46 ms |
| Pool, cached 8K prefixes in free GPU memory | 25 lossless (504 MiB, 33 ms back); 81 lossy at AQUA-3-bit quality | SplitZip 18, AQUA-KV 3-bit 74 |

What we learned:
- **In AQUA-KV's own evaluation, AQUA-KV beats our GPU lossy codec at 2-3 bits per value.** There every token is
  scored against a cache compressed while it grows. AQUA-KV needs 2.09 bits for +1.5% perplexity; our codec needs
  2.76 bits for +0.5%, and at 2.17 bits it loses 11%. Near-lossless we are about even (3.61 bits at -0.03%,
  AQUA-KV 4-bit 4.02 bits at -0.09%). We beat KIVI everywhere (KIVI 4-bit needs 5.0 bits for +0.08%) and HIGGS
  alone (2.02 bits, +8.1%). Our earlier "AQUA-KV 3-bit quality at 2.2 bits" holds only in our own transfer test,
  with the CPU coder.
- Our codec's error at 2.2 bits (12.9% of the KV) is about AQUA-KV's at 2.1 bits (11.6%), yet its perplexity
  loss is 7x larger. So it is how the error is spread, not its size: our step is per channel, so channels with
  large values, which drive the attention scores, get large errors. AQUA-KV rotates the residual (Hadamard) and
  scales it per token.
- Keys only or values only (`8f8bfd1`): at about the same bits our keys cost 2.8x AQUA-KV's (+3.13% against
  +1.10%) and our values 7.6x (+2.29% against +0.30%). At step 4 our two errors compound: together +11.25%,
  twice the sum apart. At that step most values are rounded to the prediction, so the keys and the values of a
  token lose the same unforeseen part. At step 2, and for AQUA-KV, the halves add up.
- The per-token step breaks in this protocol at step 4 (perplexity 20.7): drift raises the surprise, which raises
  the step. It stays off.
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

## Focus from 29 October: lossless, for prefill/decode disaggregation on ordinary machines

Lossless is where we lead. The baselines' papers measure datacenter fabrics on H100/H200: SplitZip uses RDMA at
up to 800 Gb/s, and ExANS argues from 400 Gb/s. No one has measured the ordinary case: GPUs in one machine over
PCIe, or machines over 10-100 GbE. The order of work:

1. **Context sweep** (stage `context`): every lossless method from 512 to 16K tokens per request (32K and more on
   a 24 GB GPU), and 16 requests coded together at short lengths, as SplitZip sweeps them. For each: bits, encode,
   decode, and transfer time per link.
2. **(T) A compact KV transfer format, and a real PD transfer** on a 4x RTX A5000 machine.
   - The format: a header (model, predictor version, mode, layers, token range, per-layer offsets, checksum),
     then per-layer streams, so the receiver can decode layer l while layer l+1 arrives.
   - The transfer: prefill on GPU 0, send over PCIe (peer-to-peer and through CPU memory), decode on GPU 1,
     generate. Time to first token for raw, SplitZip, ExANS and ours.
   - First, `scripts/pd_link_probe.py` measures the links between the GPUs.
3. **(P) A GPU-resident pool:** compressed prefixes in a fixed GPU memory budget, evicted least-recently-used.
   A workload of shared prefixes gives hit rates and time to first token against a raw pool and a SplitZip pool.
4. **Lossless speed** (200-400 Gb/s) and the GPU decoder for the 7.87-bit predictor.

Lossy development pauses: the queued runs finish, but nothing new is built.

### Progress on lossless (29 October, runs `9de9185`, `26d7b4b`, `a82e4b4`)

- **Context sweep:** 512 to 32K tokens and batches of 16, every method decoded bit-exact.
  - Up to 100 Gb/s we win from 1K tokens up.
  - Single requests under 2K tokens pay a fixed decode cost of about 3-4 ms; batching removes it.
- **Head-group decoder:** keys then values, 4 head groups, or head by head (`kv_compare/grouped.py`).
- **Calibration on code plus WikiText-2 train** (`--train-set mixed`): prose gains 0.5-0.6 bits; code moves by
  0.01-0.1 bits.
- **WikiText-2 test, 8K tokens, real streams:**

  | Mode | Bits | Decode |
  | --- | --- | --- |
  | 1 step | 9.03 | 31 ms |
  | keys then values | 8.70 | 42 ms |
  | 4 head groups | 8.48 | 53 ms |
  | head by head | 8.35 | 114 ms |

  SplitZip: 12.05 bits, 3 ms. The strong predictor's exact cost on prose is 7.83 bits.
- **Against SplitZip, the best mode per link (8K tokens):**

  | Link | Speed-up |
  | --- | --- |
  | 1-25 Gb/s | 1.44x |
  | 100 Gb/s | 1.35x |
  | 200 Gb/s | 1.24x |
  | 400 Gb/s | 0.82x |
- **Decode speed** (`groupspeed`, `8ba678a`):
  - Below 8K tokens the decode is bound by kernel launches from Python, about 65-70 us each. Replaying it as a
    CUDA graph removes that cost: 512 tokens, one step: 4.0 -> 2.0 ms; head by head: 42 -> 19 ms.
  - At 8K the GPU sets the time: 1 step 31 ms, keys then values 42 ms, 4 groups 50 ms, head by head 75-84 ms.
- **Best mode per link** (mixed calibration, CUDA graphs, against SplitZip):

  | Link | 512 tokens | 2K | 8K | Mode at 8K |
  | --- | --- | --- | --- | --- |
  | 1-25 Gb/s | 1.39-1.42x | 1.43-1.44x | 1.45x | head by head |
  | 50 Gb/s | 1.36x | 1.41x | 1.43x | head by head (8x4) |
  | 100 Gb/s | 1.31x | 1.33x | 1.35x | keys then values |
  | 200 Gb/s | 0.89x | 1.18x | 1.24x | rank 256 |
  | 400 Gb/s | 0.45x | 0.71x | 0.82x | SplitZip format |

  Each mode is best in one band of link speeds, so the codec picks its mode from the link.
- **Next:**
  - CUDA graphs in the context sweep and the transfer tables. Done: `context` and `datasets` time our decoders as
    CUDA graphs too, and the link tables use the faster exact time.
  - (T) the transfer format and a real GPU-to-GPU transfer, once the A5000 link probe is in.

### The baselines' models and datasets on one RTX A5000 ([runbook](a5000_runbook.md))

- **ExANS** names no model or dataset; its blog says only "a real KV snapshot", 256K context, H100.
- **SplitZip** calibrates its code book on WikiText-2 train. Its Table 4 reports top-16 exponent coverage on
  WikiText-2 test, HumanEval, GSM8K, MMLU and PTB, for Qwen3-32B, Qwen3-Coder-30B and Llama-3-8B.
- **On a 24 GB A5000:**
  - Models: Llama-3-8B (SplitZip's own), Qwen3-8B (the Qwen3 family) and Llama-3.2-3B, all calibrated on WikiText-2
    train like SplitZip.
  - New stage `datasets`: the five datasets, 4 x 8K tokens each. It reports every codec bit-exact and timed, our
    exact costs, and SplitZip's coverage beside its Table 4.
  - Context sweep: up to 32K tokens, with 8K-chunked prefill for the long captures.
  - Link probe: this GPU's host legs.
- **Tools:** `scripts/a5000_setup.sh` (environment) and `scripts/run_a5000.sh` (one model: measure, package, push).
  All run inside the one-GPU allocation.
- **Not covered:** Phi-2 (head size 80 with partial RoPE, not handled by our kernels yet), the 30B-80B models,
  and contexts past 32K.
- **The rest of the literature** ([lossless_related_work.md](lossless_related_work.md)): every published exponent
  coder sits at 10.5-13 bits. The comparison now also runs, from the authors' code:
  - AdaCode (EMNLP 2026), exactkv, nvCOMP's GPU codecs and IBP (EuroSys '26);
  - the computed floor of exponent-only coders, 8 + Huffman(exponent);
  - the FP8-model predictor of speculative KV coding (stage `fp8pred`).

  It runs first on the NUC (runbook section 0a), then on the A5000.
- **The P gate** (`scripts/bench_pool_gate.py`, runbook section 0b): is a compressed GPU-resident pool worth
  building the lifecycle around?
  - exactkv is the published P: 4-bit exponent offsets per 16-token page, bit-exact attention in registers,
    -20% memory, +17-23% generation throughput on a 4090. It serves as the backbone; our P1 prototype (13.05 bits
    on Smol) is weaker.
  - Measured on our Llama KV:
    - attention, pool vs dense with identical arithmetic;
    - generation at equal and at maximum batch;
    - our layer-by-layer T-to-P installation (`LayerwiseOneStep.decode(sink=...)`, a two-layer ring) against
      decoding the whole cache first.
  - If the pool pays off, our own P format (a 3-bit exponent window fused into the same attention, about 11.4
    bits on our data) is next.

## What is next, in order of expected gain (before 29 October)

0. **Finish the baselines' own evaluations.** Done (`61d6db1`): AQUA-KV's perplexity protocol for every method.
   Our harness gives the authors' exact perplexity to five digits (6.9865; paper 6.98). SplitZip's own benchmark
   on Llama-3.2-3B: 12.05 bits, 313 GB/s decode; ours on its prompts 9.92 bits (one-step) and 8.68 (strong
   predictor). Still to run:
   - AQUA-KV's own training on RedPajama (`AQUA_OFFICIAL=1`). The original sample is gone from the Hub; the
     script draws the authors' samples from ll922/RedPajama-Data-1T-Sample-Backup, a Parquet mirror with the
     original's 930,514 rows (loaded in `8ae8d22`). Their sampler's slow-tokenizer call returns False for
     Llama 3, so the fast tokenizer is used in that case, recorded. Training keeps about 168 MB per sample in
     RAM; with less free RAM the script trains on fewer samples and says so;
   - LongBench and GSM8K subsets;
   - ExANS's position-aligned batches.
0b. **Lossy error shaping (first for lossy).** Make our error look like AQUA-KV's: quantize the residual in a
    Hadamard-rotated domain and/or weight each channel's step by its attention sensitivity (query magnitude), and
    keep the entropy coder.
    - Diagnosed (`8f8bfd1`): both halves cost more than AQUA-KV's, values most (see what we learned).
    - Tried (`8ae8d22`), worse: the step of each lane proportional to 1 / sqrt(its loss sensitivity), the
      diagonal Fisher information on our calibration documents. Step 2: +10.1% over all lanes, +5.9% within
      heads, against +0.54% unshaped. The gradient is local and first-order; the lanes it made 16x coarser
      carry the loss.
    - Ready to run: one absolute step per head (`ours<step>u`) and per layer's keys / values (`ours<step>w`),
      the error spread evenly as after AQUA-KV's rotation (the GPU codec takes a kappa per lane).
    - Ready to run, our own ways to fewer bits (not AQUA-KV's rotation and grid):
      - a bit allocation measured in the protocol itself, on our calibration text (stage `allocate`, methods
        `ours<step>a`);
      - rate-aware rounding in the encoder (`_r`, decoder unchanged);
      - centre-of-mass reconstruction (`_c`);
      - an estimate of what a transform learned from our data could save (stage `transform`).
    - Later: the strong predictor in the lossy GPU codec; fitting the predictor on the receiver's reconstructions;
      keeping the most-attended tokens exact; low-rank KV.
    - For prefill/decode disaggregation the matching lossy test is our transfer test (the prompt compressed once),
      where we are ahead; AQUA-KV's protocol is the in-memory use.
    - Target, in AQUA-KV's protocol: +1.5% or less at 2.1 bits or less, and +0.1% at 3.0 bits or less.
1. **A GPU decoder for the 7.87-bit predictor** (layers 0, l−2, l−1, 16 groups, no attention).
   - Gain: 441 MiB per document instead of 504, 13% fewer bytes than our current stream and 35% fewer than
     SplitZip.
   - Products alone measure 53 ms per document, so the expected decode is 55-70 ms. That wins up to 100 Gb/s, at
     about 1.5x SplitZip where the link sets the time, and enlarges the pool by 14%.
   - Target: a bit-exact stream at 7.9-8.0 bits, decoding in 70 ms or less.
2. **The same predictor in the lossy GPU codec**, after error shaping.
   - In our transfer test, at AQUA-KV 3-bit quality, our CPU coder needs 2.21 bits and the GPU codec 2.78.
   - Target: judged in AQUA-KV's protocol, decoding in 40 ms or less.
   - Also fix the GPU per-token step: it helps the CPU coder but not yet the GPU codec (2.32 bits, +0.57%).
3. **400-800 Gb/s.** The SplitZip-format decode takes 17 ms: 7 ms of products and 12 ms of coding.
   - It must reach 12 ms or less to beat SplitZip alone at 400 Gb/s.
   - Ideas not yet tried: a two-head program, so the in-kernel product has enough warps; fewer instructions per
     value in the context and word rebuild; the first product's requantization fused into its epilogue output
     layout.
   - 800 Gb/s needs about 4x faster decoding and is not a near-term target.
4. **Baseline fidelity.**
   - Our AQUA-KV re-implementation is calibrated on our documents. In the protocol it gives 7.09 at 2.09 bits
     (paper: 7.03 at 2.16); the authors' own training run will settle that row.
   - Our KIVI is stronger than the paper's (2-bit, group 128: 7.78 against 9.33), so it is not handicapped.
   - HIGGS alone: 7.55 against the paper's 7.47.
5. **Deployment pieces.**
   - (T) A compact transfer header: model, predictor version, mode, layer and token ranges. The receiver needs
     the predictor's coefficients, about 117 MB per model for one-step (8-bit), shipped once.
   - (P) GPU-pool experiments.
   - A fast GPU encoder for the SplitZip format: the PyTorch packer takes 1 s per document.
