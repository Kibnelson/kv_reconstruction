# Lossless and lossy KV transfer on Llama-3.2-3B: ours against SplitZip, ExANS, AQUA-KV and KIVI

`scripts/bench_llama_compare.py` captures Llama-3.2-3B KV once and runs every method on the same words:

- **Lossless:** our predictive codec against SplitZip and ExANS (the authors' codecs, plus re-implemented
  "-style" versions of both for exact bit costs), in bits per value and in measured speed.
- **Lossy:** our codec run closed loop with a quantized residual, against AQUA-KV and KIVI (following the
  authors' code, see Methods) and an FP8 cast. Each method is scored by bits per value and by how much worse the
  model continues the text when it reads that method's KV.

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
| `format` | Optional. Bits of our one-step prediction inside SplitZip's kind of format: fixed-length codes (no entropy coder), with the code book of each value chosen by the prediction, for the dense and the rank-256 product (`format.json`, `kv_compare/szformat.py`). | a few minutes |
| `exans` | Optional; needs nvcc, CMake and the nvCOMP SDK (all used by the earlier official run). Builds the official ExANS codec from the authors' source with one compile fix and runs it on both test documents: measured stream size, bit-exact round trip, encode and decode speed (`exans.json`). | about 5 minutes |
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

The first fast-link run (`28ef51d`) decoded bit-exact in 30.0 ms (15.7 G values/s) at 9.21 bits per value (+0.196, as the probe said). At 200 Gb/s that is 0.94x of SplitZip's 28.2 ms, up from 0.87x; slower links are unchanged (1.33-1.34x, dense or exact modes). The products took 11.4 ms instead of the expected 7.5 ms (half of 15.0), and running them with the coding cost 3.1 ms more than one after the other (11.4 + 15.6 = 27.0 ms against 30.1 ms). Part of the 11.4 ms is the int32 intermediate (16 MB written, read back and requantized by an extra kernel in each layer), so the speed stage now also runs the first product as a Triton kernel that requantizes in its epilogue (`requant="epilogue"`, bit for bit the same 8-bit intermediate, three tile shapes), and tries ranks 512, 384 and 256 with 1, 2 and 4 chunks. The coding alone (15.6 ms) is the floor of this design and already slower than SplitZip's 14.1 ms at 400 Gb/s, so 400 Gb/s needs a faster coder, not a cheaper product.

That run (`c96f7be`, commit `f29e3d0`) won 200 Gb/s. Requantizing in the epilogue took the rank-512 products from 11.5 to 8.9 ms (64x128x64 tiles) and 128x128x64 tiles did better still. All decodes bit-exact, 32 tokens per program (now the fastest dense shape too: 31.2 ms), 2 chunks:

| Product | Bits per value | Products alone | Decode, kernel requantization | Decode, epilogue (128x128x64) | 200 Gb/s vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| dense | 9.025 | 14.6 ms | 31.2 ms | - | 0.90x |
| rank 512 | 9.221 (+0.20) | 11.5 → 8.9 ms | 28.2 ms | 25.5 ms | 1.11x |
| rank 384 | 9.296 (+0.27) | 9.6 → 7.3 ms | 26.5 ms | 24.5 ms | 1.15x |
| rank 256 | 9.405 (+0.38) | 7.6 → 5.5 ms | 24.5 ms | 22.9 ms | 1.23x |

(products alone: kernel requantization → epilogue with 64x128x64 tiles; the decode columns are the best of plain and CUDA-graph runs.) Rank 256 decodes at 20.5 G values/s, 22.9 ms for the document, while its 9.405 bits take 22.1 ms on a 200 Gb/s wire: decode and link are now balanced there, so a smaller rank would only add bits. The measured transfer table is 1.34x SplitZip up to 50 Gb/s, 1.33x at 100 Gb/s and 1.23x at 200 Gb/s (rank 256); 0.62x at 400 Gb/s, where the coding alone (15.6-15.7 ms) is the limit.

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
| Ours, GPU codec (one-step), step k | no | The GPU per-layer codec (`kv_compare/lossy_gpu.py`), closed loop: the one-step int8 product of the previous layer's reconstruction, each value quantized to the lattice prediction + j × step (step k × the channel's spread), j coded with rANS against a logistic model on a quarter-octave table of step / spread; values beyond 24 spreads and non-finite values travel exact. The decoder reproduces the encoder's reconstruction bit for bit. Weaker predictor than `fast` (no layer 0, l−2, attention or channel groups), so more bits per step, but a measured stream and decode speed | measured stream |
| ExANS, official codec | yes | The authors' CUDA codec (openlake de6c38b, built with one compile fix): blocks of 499,712 values; exponent bytes with a per-128-value phase shift, coded by nvCOMP's batched ANS; sign and mantissa byte raw; header and phase bitmap | measured stream size |
| AQUA-KV, b-bit | no | The authors' algorithm (goodevening13/aquakv e744183): keys predicted from the previous layer's reconstructed keys, values from the current reconstructed keys plus the previous layer's reconstructed values (RoPE removed), BF16 linear maps fitted layer by layer on the calibration documents' own closed-loop reconstructions (float32, pseudo-inverse, 1e-3 damping); residual quantized with HIGGS (Hadamard transform over 1024 channels, per-token L2 scale, 2-D EDEN grid with 16 / 64 / 256 points); layer 0 HIGGS 4-bit; the first 4 tokens in BF16 | b (4 in layer 0) + one 16-bit scale per 1024 values; 16 for the 4 prefix tokens |
| KIVI, b-bit | no | The authors' code (jy-yuan/KIVI 876b4d2) with the paper's settings: keys per channel over 32-token groups, values per token over 32-channel groups, asymmetric min-max with BF16 scale and minimum; the most recent 128 tokens (values; keys beyond whole 128-token blocks) stay BF16 | b + 1 per quantized value; 16 for the BF16 tokens |
| FP8 cast | no | e4m3, no scale | 8 |

Notes:

- The reference range coder matched computed costs within 0.011 bits per value in every SmolLM2 test.
- Baselines follow the authors' papers and code, so none is weaker than published. AQUA-KV and KIVI are re-implemented on our captured KV from the authors' code (their caches need older Transformers versions) and checked against the authors' own functions with `scripts/check_baselines_official.py`: KIVI's quantizer agrees exactly; AQUA-KV's HIGGS quantizer gives the same values for 99.94-99.99% of entries (the rest are near-ties after a differently ordered Hadamard sum) and the same error, and its regression the same weights within one BF16 step. ExANS and SplitZip run the authors' own codecs.
- One deviation remains, in AQUA-KV's calibration data: it is fitted on this study's calibration documents (the same ones our codec uses), not on the 256 x 8192 RedPajama tokens of the authors' README.
- Earlier runs used stand-ins: "AQUA-KV-style" (a scalar min-max quantizer instead of HIGGS, no BF16 prefix; before commit c96f7be its predictors were fitted on exact KV and its 2- and 3-bit rows diverged on Llama-3.2-3B) and "KIVI-style" (128-token / 64-channel groups, no BF16 recent tokens). Rerun only the baselines with `--stages lossy,report --lossy-methods aqua,kivi`; the other rows of `lossy.json` are kept and the stand-in rows are dropped.

## Results with the authors' baselines (`474524c`, run at commit `1da98b5`)

All on test_text / test_code, RTX A4500. Baselines follow their authors' code (see Methods).

**Lossless, one document (470M values) over an ideal link** (decoding overlapped with arrival; measured decode speeds):

| Link | SplitZip (12.02 bits, 311 GB/s) | ExANS official (10.86 bits, 54 GB/s) | Ours (best measured mode) | Ours vs SplitZip | Ours vs ExANS |
| --- | --- | --- | --- | --- | --- |
| 1-50 Gb/s | | | | 1.34x | 1.21x |
| 100 Gb/s | 56 ms | 51 ms | 42 ms | 1.33x | 1.20x |
| 200 Gb/s | 28 ms | 26 ms | 23 ms | 1.24x | 1.12x |
| 400 Gb/s | 14 ms | 18 ms | 23 ms | 0.62x | 0.77x |
| 800 Gb/s | 7 ms | 18 ms | 23 ms | 0.31x | 0.77x |

The official ExANS codes 10.86-10.87 bits per value here (its re-implementation estimated 10.72) and decodes at 54 GB/s, encodes at 5 GB/s. Per-layer hybrid with SplitZip's format (estimate from measured parts): 1.11x SplitZip at 400 Gb/s and 1.04x at 800 Gb/s. Our prediction inside SplitZip's fixed-length format (bits only): 10.18 bits (dense product) and 10.51 bits (rank 256) against SplitZip's 12.02; our entropy-coded stream is 9.0-9.4.

**Lossy** (first 3,072 tokens; perplexity change of the next 512 tokens):

| Method | Bits per value | Perplexity change |
| --- | --- | --- |
| AQUA-KV 2-bit | 2.11 | +1.46% / -0.35% |
| AQUA-KV 3-bit | 3.07 | +0.06% / -0.23% |
| AQUA-KV 4-bit | 4.03 | -0.08% / -0.10% |
| KIVI 2-bit | 3.27 | +20.79% / +3.46% |
| KIVI 4-bit | 5.23 | +0.28% / +0.23% |
| Ours (`fast` predictor, computed cost), step 1 | 3.31 / 3.04 | +1.26% / +0.28% |
| Ours (`fast`), step 2 | 2.56 / 2.38 | +3.81% / +0.59% |
| Ours, GPU codec (one-step, measured stream), step 0.5 | 4.14 / 3.91 | +0.93% / -0.21% |
| Ours, GPU codec, step 1 | 3.20 / 2.99 | +6.77% / +1.16% |

With HIGGS and its closed-loop fit, AQUA-KV reaches at 2.1 bits the quality our codec needs about 3.2 bits for, and at 3.1 bits better quality than ours at 4 bits: about one bit per value better. It does this with more KV error (5.3% at 3 bits against our 3.8% at 2.6 bits), so mean squared error is not what decides quality here. Two differences are likely: AQUA-KV keeps the first 4 tokens (the attention sinks, which receive much of every query's attention) in BF16, and HIGGS scales each token's error with that token's residual, while our step is the same for every token. Our GPU codec at step 4 (+776% perplexity) points the same way. KIVI is behind both. The GPU codec decodes at 17 G values/s (dense) and 25 G values/s (rank 256).

## Results: both uses, the SplitZip mix and the lossy fixes (`7dc5fb6`, run at commit `7bde391`)

**Transfer, lossless, one document (470M values), time and bytes sent:**

| Link | SplitZip | ExANS | Ours alone | Ours + SplitZip mix (layers ours) | Mix vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1-50 Gb/s | | | 26% fewer bytes | the same (all ours) | 1.34-1.35x |
| 100 Gb/s | 57 ms, 675 MiB | 51 ms, 608 MiB | 42 ms, 505 MiB | 42 ms, 505 MiB (28) | 1.33x |
| 200 Gb/s | 28 ms, 675 MiB | 26 ms, 608 MiB | 23 ms, 527 MiB | 23 ms, 541 MiB (24, rank 384) | 1.25x |
| 400 Gb/s | 14 ms, 675 MiB | 18 ms, 608 MiB | 23 ms, 527 MiB | 13 ms, 628 MiB (8, rank 512) | 1.07x |
| 800 Gb/s | 7 ms, 675 MiB | 18 ms, 608 MiB | 23 ms, 527 MiB | 7 ms, 675 MiB (0) | 1.00x |

SplitZip's own codec, run here: 12.05 bits per value, 309 GB/s decode. Every mix is a real stream decoded bit-exact; SplitZip coded per layer decodes in 6.5 ms for the document (3.0 ms as one tensor). Our codec alone is slower from 400 Gb/s because its decode (23 ms) sets the time; the link is busy for 11 ms and carries 22% fewer bytes.

**Lossy, with the attention sinks kept exact** (perplexity change, test_text / test_code; 2 documents x 512 continuation tokens, so differences of a few tenths of a percent are within noise):

| Method | Bits per value | Perplexity change | Decode (one document) |
| --- | --- | --- | --- |
| AQUA-KV 2-bit | 2.11 | +1.46% / -0.35% | 78 ms (authors' code) |
| AQUA-KV 3-bit | 3.07 | +0.06% / -0.23% | 78 ms |
| KIVI 4-bit | 5.23 | +0.28% / +0.23% | 46 ms (authors' code) |
| Ours (`fast`), step 4, sinks | 2.14 / 2.05 | +0.09% / -0.25% | no GPU decoder |
| Ours (`fast`), step 2, sinks + token step | 2.17 / 2.16 | +0.03% / +0.14% | no GPU decoder |
| Ours (`fast`), step 4, sinks + token step | 1.77 / 1.77 | +1.77% / -0.72% | no GPU decoder |
| Ours, GPU codec, step 4, sinks | 1.96 / 1.88 | +0.98% / +0.68% | 27 ms |
| Ours, GPU codec, step 2, sinks + token step | 2.28 / 2.27 | -0.58% / -1.13% | 27 ms |

Keeping the first 4 tokens exact (0.02 bits per value) is what was missing: without it our codec at 2.1-2.4 bits lost 4-20% perplexity, with it the same streams lose under 1%. With the sinks our lossy codec matches AQUA-KV 3-bit quality at about 2.2 bits and AQUA-KV 2-bit quality at 1.8-1.9 bits; the token step saves about 0.3-0.4 bits per value more but only together with the sinks (without them it fails at large steps).

**Pool (one 8K prefix; recompute 992 ms, raw BF16 896 MiB, 14 prefixes fit):** SplitZip 675 MiB (19 fit), back in 3 ms; ExANS 608 MiB (21), 18 ms; ours lossless 501-527 MiB (24-25), 23 ms (rank 256) to 101 ms (fewest bits); AQUA-KV 2-bit 118 MiB (108), 78 ms, adding it 0.95 s; our GPU codec step 4 with sinks 107 MiB (119), 27 ms, adding it 55 ms.

## Next run: a larger quality test and our prediction in SplitZip's format

- **Lossy quality on WikiText-2** (stage `evalcap`, then `lossy --lossy-docs eval`). The two-document table scores
  1,024 continuation tokens, so differences of a few tenths of a percent are noise. `evalcap` captures 8 windows of
  4,096 consecutive tokens of the WikiText-2 raw test split (joined with blank lines, as AQUA-KV's evaluation reads
  it; each window starts with the BOS token). The lossy stage then runs every method on each window, with a
  3,072-token prompt and 1,024 scored tokens, for 8,192 scored tokens in all. Results go to `lossy_eval.json`. The
  report gives the pooled perplexity change, the per-window mean ± standard error and how many windows got worse.
  `scripts/run_on_nuc.sh` writes the text once with the AQUA-KV environment's `datasets` library.
- **Our prediction in SplitZip's format, as a real codec** (stage `szcodec`, `kv_compare/szcodec.py`). It uses
  fixed-length codes chosen per value by our one-step prediction (the "sign + magnitude" variant the `format`
  probe measured at 10.2 bits dense and 10.5 bits rank 256). Every value of a 32-token × one-head tile is placed by
  two prefix sums, so there is no entropy coder. Escapes now carry the word in place of the raw bits, which costs
  less than the probe's side stream. With the reduced-rank product, the second product runs inside the decoding
  kernel on tensor cores, so the int32 products never go through memory. The stage measures:
  - dense and rank 256 / 512 products, over four tile shapes, each decoded bit-exact and timed, plus a CUDA graph;
  - mixes with the authors' SplitZip per layer;
  - a per-link table against SplitZip, our rANS codec and the rANS mix.

  The aim is to beat SplitZip at 400-800 Gb/s, where our rANS decode (23 ms) is the bottleneck.

```bash
cd ~/kv_llama32 && git checkout llama32-3b-lossy-lossless-compare \
    && git pull --ff-only origin llama32-3b-lossy-lossless-compare \
    && STAGES=evalcap,lossy,szcodec,report \
       EXTRA="--lossy-docs eval --lossy-variants base,sinks,both --kappas 0.5,1,2,4" bash scripts/run_on_nuc.sh
```

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
