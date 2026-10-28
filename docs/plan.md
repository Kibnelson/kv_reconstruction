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

## What is next, in order of expected gain

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
