# Resume here: T and the "diff" transfer protocol (status 12 November 2026)

This note records what was built and measured up to 30 October 2026, so the work can be picked up again. Nothing
ran between 30 October and 12 November. The work is kept as **two separate ideas**:

| | Idea | One line | Status | Branch |
| --- | --- | --- | --- | --- |
| 1 | **T**: lossless predictive KV transfer format | Predict each layer's KV from the layer before and code the exact BF16 word around the prediction: 8.3-9.0 bits per value, against 10.9-12.1 for the published exact codecs | Measured on Llama-3.2-3B (one dataset complete, four partial) | `llama32-3b-lossy-lossless-compare` |
| 2 | **Diff protocol**: send only what the decode engine lacks | Before sending, find what the decode engine already holds (its GPU cache, its host memory, another decode engine) and send only the rest, fetching evicted prefixes from local DRAM at the same time | Built as a vLLM 0.32 connector; **never run**; phase 2 (measure the opportunity) gates everything else | `pd-vllm-protocol` |

Set aside: **P** (a compressed resident pool read by attention) and the lossy codec work (section 4).

Everything below refers to Llama-3.2-3B (`unsloth/Llama-3.2-3B`, revision `d4446454d87d51aa42e1fb174f25acc5f8762331`;
28 layers, 8 KV heads x 128) on the NUC's RTX A4500 unless it says otherwise. A "window" is 8,192 consecutive
tokens.

---

## 1. Idea 1: T, the transfer format

### What it is

- **The limit of the others.** Every published bit-exact KV codec looks at each BF16 value on its own and compresses
  only its exponent: SplitZip, ExANS, AdaCode, exactkv, IBP, ZipNN, DFloat11, nvCOMP. Sign and mantissa (8 bits)
  travel raw. On our KV, the best such a coder can do with per-layer statistics is **10.69 bits per value**
  (8 + exponent entropy; 10.73 with Huffman), computed.
- **What T does.** It predicts each token's keys and values at layer l from the same token's layer l-1, which the
  decoder already has. The prediction is RoPE removed, 8-bit features, an exact 8-bit integer product with a
  per-layer calibrated matrix, RoPE applied back to keys; layer 0 is predicted by its mean.
  - A per-channel spread, scaled by how surprising the token was one layer down, sets a distribution centred on
    the prediction. The exact word is coded against it with rANS: a coarse interval first, then the word inside it.
    Outliers, Inf and NaN escape raw.
  - Encoder and decoder run identical integer and float32 operations, so the decode is bit-exact; this is
    guaranteed on one GPU type and untested across types.
  - The mantissa becomes compressible, which is why T goes below every exponent-only codec.
- **Modes**, trading bits against decode speed:
  - dense: one predictor;
  - keys then values (kv), 4 head groups (4g), head by head (hbh): later heads are also predicted from heads of the
    same layer already decoded;
  - rank 256: a reduced-rank predictor, the fastest;
  - "SplitZip format": our predictor with SplitZip's layout, for comparison.
- **Decode is layer by layer.** A received request can be installed into the cache one layer at a time, which
  halved the peak memory of installation (measured, below).
- **Calibration**: per model, on WikiText-2 train (wtrain0-5 plus wtables). SplitZip's code book and AdaCode's
  parameters are calibrated on the same KV, following their protocols.

### Closest prior work

`docs/lossless_related_work.md` on the T branch has the literature, checked against the sources. The closest is
**speculative KV coding** (Fergus Finn, a blog note): also a predictor plus arithmetic coding, 5.9-6.7 bits on Qwen3.
- Its predictor is a whole FP8 copy of the model run on both sides, not counted in its cost.
- It has no decoder and no round-trip check.

We emulated its predictor (stage `fp8pred`): 6.49 bits on the GSM8K smoke window, with that predictor's cost not
counted. Ours is a cheap one-step predictor with a measured, bit-exact GPU decoder.

### What is measured

**WikiText-2 test, 4 windows, every row a real stream decoded bit-exact on the A4500.**
Source: `benchmark_results/llama32_3b_lossless_rtxa4500_20261030_1306_partial/`. That run's torch was 2.15 by
accident (see section 5); bits do not depend on it.

| Method | Bits per value | Decode, all 4 windows (ms) |
| --- | --- | --- |
| ours, strong predictor (exact cost; no GPU decoder) | 7.793 | - |
| **ours, head by head** | **8.279** | 452.5 (331.7 as a CUDA graph) |
| ours, 4 head groups | 8.408 | 211.7 |
| ours, keys then values | 8.633 | 167.5 |
| ours, one-step predictor (exact cost) | 8.869 | - |
| ours, dense | 8.972 | 124.3 |
| ours, rank 256 | 9.339 | 91.4 |
| ours, SplitZip format, rank 256 | 10.122 | 70.8 |
| floor of exponent-only coders: 8 + exponent entropy / 8 + Huffman | 10.691 / 10.729 | computed |
| ExANS (authors' codec) | 10.872 | 70.6 |
| AdaCode (authors' codec) | 11.370 | 12.2 |
| SplitZip (authors' codec) | 12.050 | 12.2 |
| exactkv (authors' codec) | 12.575 | 83.2 |
| nvCOMP Zstd / ANS / Cascaded / LZ4 | 12.898 / 13.078 / 16.006 / 16.013 | 258.9 / 20.5 / 26.7 / 32.7 |

**Transfer time.** The time is the longer of wire time and decode, per link, all 4 windows; ours picks its fastest
exact mode per link. Same run.

| Link | Ours vs SplitZip | Ours vs the best other codec |
| --- | --- | --- |
| 1-25 Gb/s | 1.46x faster (head by head) | 1.31x (ExANS) |
| 50 Gb/s | 1.43x | 1.29x (ExANS) |
| 100 Gb/s | 1.35x (keys then values) | 1.22x (ExANS) |
| 200 Gb/s | 1.24x (rank 256) | 1.12x (ExANS) |
| 400 Gb/s | 0.80x | 0.75x (AdaCode, which decodes about 10x faster) |

**Other datasets (exact cost, one-step / strong predictor).** These are from
`..._1240_partial/`, calibrated on WikiText-2 only; the other datasets' stream rows are missing.

| Dataset | One-step | Strong |
| --- | --- | --- |
| WikiText-2 | 8.869 | 7.793 |
| HumanEval | 9.535 | 8.564 |
| GSM8K | 9.179 | 8.201 |
| MMLU | 9.208 | 8.235 |
| PTB | 9.141 | 8.171 |

On HumanEval, head by head reached 8.989 bits against ExANS 10.873 and SplitZip 12.054 (a partial row). The gap
narrows on code, probably because calibration is on text.

**What the bits mean for a PD system.** These are derived from the rows above (8K-token requests, one decode GPU):
- **Requests per second through one link.** We carry 1.9x raw BF16, 1.3-1.45x SplitZip and 1.16-1.31x ExANS.

  | Link | Ours | ExANS | SplitZip | Raw |
  | --- | --- | --- | --- | --- |
  | 10 Gb/s | 2.57 | 1.96 | 1.77 | 1.33 |
  | 100 Gb/s | 23.9 | 19.6 | 17.7 | 13.3 |
- **The GPU's PCIe slot caps the link.** The link probe measured host-to-GPU at 24.0 GB/s, about 192 Gb/s
  (`scripts/pd_link_probe.py`). Any faster network still enters the GPU through that slot. At that cap, ours
  (rank 256) takes 91 ms against ExANS 106, AdaCode 111, SplitZip 118 and raw 157 (all 4 windows): faster at every
  link speed on PCIe Gen4 GPUs. The report's 400 Gb/s row ignores this cap; adding the capped rows to the report
  was offered and not done.
- **Highest link speed each decoder keeps up with**, per GPU: dense 136 Gb/s, rank 256 192 Gb/s, head by head
  47 Gb/s. With 3 decode GPUs sharing the incoming traffic, these triple.
- **Cost.** Decoding takes 23-83 ms of GPU time per 8K request depending on the mode, against about 3 ms for
  SplitZip and AdaCode. Encoding takes 40-106 ms on the prefill GPU, against about 8 ms for SplitZip.

**Other measured results:**
- **Install.** Layer-by-layer install into a pool needs +777 MiB of peak memory, against 1,609 MiB for decoding
  the whole cache first. Time to ready is 1.25-1.7x faster than raw at 100 Gb/s and below, and 1.2-1.3x faster than
  SplitZip. Source: `benchmark_results/llama32_3b_poolgate_rtxa4500_20261030_1251/`.
- **Earlier and lossy work.** The milestone report `docs/milestone_report_2026-10-29.md` has the earlier
  lossless and lossy results, speeds and the full method table.

### What is built (T branch)

- **`kv_compare/`** holds the codec:
  - `fused.py`: the one-step predictor;
  - `layerwise.py`: layer-by-layer codec with a decode sink, reduced rank;
  - `grouped.py`: head groups;
  - `gpu_coder.py`: rANS;
  - `predict.py`: captures and calibration;
  - `context_sweep.py`: the measurement harness, with crash markers;
  - `more_baselines.py`: nvCOMP (in a child process), AdaCode, exactkv, IBP, floors;
  - `exans_official.py`, `mixed.py`: ExANS and SplitZip;
  - `capture.py`: chunked prefill capture;
  - `test_kv_compare.py`: tests.
- **`scripts/bench_llama_compare.py`** runs stages calcap, datasets, context, fp8pred, groupspeed and report.
- **Run scripts:**
  - `scripts/run_a5000.sh` and `scripts/a5000_setup.sh`: a pinned environment (torch 2.11.0+cu128, triton 3.6.0,
    transformers 4.57.6) and the authors' sources at pinned commits;
  - `scripts/fetch_eval_texts.py`: the five datasets;
  - `scripts/pd_link_probe.py`: the link probe.
- **Docs:**
  - `docs/a5000_runbook.md`: Slurm on the A5000 node, plus the NUC rehearsal;
  - `docs/lossless_related_work.md`: the literature;
  - `docs/plan.md`: the plan;
  - `docs/milestone_report_2026-10-29.md`: the milestone report.

### Unfinished

1. **Finish the 3B benchmark.** The full run on the NUC stopped after nvCOMP crashed the process four times (a
   random segfault inside its encode). The fix is commit `d95fd10`:
   - nvCOMP runs in a child process;
   - crash records are kept, and the automatic retries go up to 8;
   - each dataset result records the torch version it was measured with;
   - the run scripts stop if torch is not the pin.

   To do: rerun, then HumanEval, GSM8K, MMLU, PTB, the context sweep (up to 32K+), fp8pred and groupspeed fill in.
2. **The A5000 runs**: Llama-3-8B (SplitZip's model) and Qwen3-8B, per `docs/a5000_runbook.md`.
3. **IBP** does not build; its build log was never inspected.

### Ideas for T not yet tried

These are listed in the order we expected them to pay off.

- **Compute layer 0 on the decode side.** Layer 0's KV depends only on the token IDs and positions, and the decode
  side holds the model weights. Layer 0 is our most expensive layer today (about 10 bits; the average is 8.95). If
  both sides compute it deterministically and use it as the prediction, the average falls to about 8.6 bits (an
  estimate from the per-layer results in `llama32_3b_compare_a4500_20261029_1342/lossless.json`), about 4% fewer
  bytes on every request.
- **A faster decoder.**
  - Profile it first: the 8-bit matrix product is perhaps 40% of the dense mode's time (an estimate).
  - Then try rank 128 and a simpler coding mode.
  - Then mix codecs per layer (some layers in AdaCode) on very fast links: an estimated 49 ms against AdaCode's
    53 at an uncapped 400 Gb/s.
- **Use the base as context.** When the decode side already holds a conversation's earlier KV, adapt the coder to
  it. The gain is unknown and needs measuring.
- **Independent chunked streams** (e.g. 1K tokens), so decoding starts while bytes still arrive. This is also a
  prerequisite of idea 2.

### Resume T

On the NUC, in the clone `~/kv_llama32`, with `KV_HOME=~/kv_a5000`:

```bash
bash <<'NUC'
set -Eeuo pipefail
cd ~/kv_llama32
git fetch -q origin llama32-3b-lossy-lossless-compare
git checkout -q llama32-3b-lossy-lossless-compare && git pull -q --ff-only origin llama32-3b-lossy-lossless-compare
export ALLOW_NO_SLURM=1 KV_HOME=$HOME/kv_a5000
export CUDA_HOME=$HOME/kv_reconstruction/runs/_matched_baselines/cuda12_8/toolkit
bash scripts/a5000_setup.sh                 # puts torch 2.11.0+cu128 back (FlashInfer had replaced it)
MODEL=llama32_3b bash scripts/run_a5000.sh  # resumes; results pushed as benchmark_results/llama32_3b_lossless_...
NUC
```

---

## 2. Idea 2: the diff protocol (send only what is missing)

### What it is

In PD disaggregation, a decode engine often already holds part of a new request's KV:
- a shared system prompt;
- the earlier turns of a chat it served;
- a shared document.

That part may also still be in its host memory after being pushed out of the GPU, or on another decode engine. The
protocol keeps a directory of where every block of KV lives (the decode GPU, the decode node's DRAM, another
engine, or only the prefill side). For each request it plans, per block:
- **skip**: already on the decode GPU;
- **load locally**: in the decode node's DRAM, copied over that node's PCIe;
- **send**: over the network from the prefill side;
- **compute**: layer 0 from token IDs.

The network transfer and the local load run at the same time, and every block is installed layer by layer as it
lands. The request is ready when the slower path finishes. Blocks can travel in any format: raw, or T to shrink
both the network bytes and the DRAM copies.

### Prior work it has to be distinguished from

- **Prefix caching and routing to where the KV already is.** vLLM and SGLang prefix caching, Mooncake, and NVIDIA
  Dynamo's KV-aware routing already avoid recomputing or re-sending a cached prefix.
- **vLLM's own multi-turn PD example** (`examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py`,
  NIXL bidirectional) already reuses the decode engine's blocks across turns.
- **CachedAttention** (ATC '24) keeps multi-turn KV in GPU, DRAM and disk tiers. It loads layers ahead while
  computing and prefetches using the scheduler's queue. Compression is not part of its core.
- **LMCache** does PD transfer with CPU and disk tiers, and a controller that tracks where tokens live. It has a
  compression hook but no lossless method.

So tracking, tiers and prefetching alone are not new. The possible contribution is:
- one lossless compressed format end to end (network, DRAM tier, PCIe);
- planning the network and local PCIe paths together by bytes;
- layer-0 computation on the decode side.

This must be checked against LMCache and CachedAttention in detail before it is claimed.

### What is built (`pd-vllm-protocol`, commits `2fce181`, `4520203`)

The design and runbook are in `docs/pd_protocol.md`. The branch sits on top of the T branch.

- **`kv_compare/pd/connector.py`**: `TKVConnector`, loaded by unmodified vLLM 0.32.0 through
  `kv_connector_module_path`. It works in push mode, 1 prefill and N decode engines.
  - The prefill worker copies a finished prompt's KV out of the paged cache, frees the blocks, encodes the KV and
    queues it.
  - The decode engine holds the request (`WAITING_FOR_REMOTE_KVS`) and installs each arrived transfer layer by
    layer into its blocks.
  - Checksums prove transfers bit-exact in the check pass.
  - Aborted requests, timeouts and unclaimed transfers are handled.
  - It writes one JSON line per transfer. Per request it also logs the decode engine's own prefix-cache hit (event
    `match`) and the prompt's chained block hashes (event `prefill_done`); phase 2 runs on these.
- **`kv_compare/pd/codecs.py`**: raw, ours (dense, kv, 4g, hbh, rank256; only the used stream tails travel),
  SplitZip and AdaCode (the authors' code). It handles any token count. ExANS and nvCOMP are **not** in this path
  yet.
- **`kv_compare/pd/transport.py`**: a first-come-first-served virtual link of 3-400 Gb/s, shared by all
  destinations, over a pinned shared-memory ring (same machine) or TCP. The achieved speed is recorded.
- **Other modules:** `paged.py` (paged cache to words), `wire.py` (bundles, checksums), `export.py` and
  `scripts/pd_export_codecs.py` (codec constants), `selftest.py` (every codec through the transport in vLLM's
  environment).
- **Scripts:**
  - `pd_proxy.py`: routing round-robin, least loaded or by conversation; its `local` mode is the colocated
    baseline;
  - `pd_prompts.py`: prompts from the five datasets;
  - `pd_bench.py`: check, a link x codec x rate sweep with `vllm bench serve`, colocated, multi-turn, report;
  - `pd_opportunity.py`: the phase 2 analysis;
  - `pd_setup.sh`: vLLM 0.32.0 in `$KV_HOME/venv_vllm`, plus the self-test;
  - `run_pd_vllm.sh`: A5000 with 4 GPUs, or the NUC with one GPU shared.
- **Tests:** `kv_compare/test_pd.py`, 22 tests, all passing on the CPU. They cover the wire format, the link
  (pacing, ring wrap, drops), paged gather and scatter, the connector's whole request flow against a stand-in of
  vLLM's interface, the proxy against fake engines, the report, and T through the wire (Triton interpreter).

### Phases and the gate

1. **Everything sent.** Requests/s, goodput at a time-to-first-token target, TTFT p50/p99 and TPOT for links 3, 10,
   25, 100 and 400 Gb/s × 7 codecs × 4 request rates. The same prompts without disaggregation are the baseline.
   This is about 3.5 hours on the A5000 node.
2. **Measure the opportunity; send nothing differently.** vLLM's multi-turn benchmark runs with conversation
   routing. `pd_opportunity.py` splits the KV sent into four parts:
   - already on the decode GPU (vLLM's own count);
   - held before by that engine, which a DRAM tier would have kept (unbounded, or within N GB);
   - held only by another decode engine;
   - new.

   It reports the bytes raw and as coded. **Build phase 3 only if these are large.** The DRAM numbers are
   conservative: they count prompt blocks only, not blocks of answers the decode engine generated itself.
3. **The diff protocol itself**, if phase 2 says so: the directory, the planner, the concurrent DRAM path, and
   layer 0 on the decode side.

### Caveats already known

- **The link is emulated.** On one machine the bytes and all codec work are real, but the network is a pacer on a
  PCIe-bound copy path. Above about 190 Gb/s the machine sets the speed.
- **The link may not be the bottleneck.** One A5000 prefilling the 3B model probably makes only about 7.5 Gb/s of
  raw KV (an estimate: about 1 s per 8K prompt; the runs measure it). At 25 Gb/s and above, the link is likely not
  what limits requests/s; differences show in TTFT and GPU time.
- **Decoding competes with generation.** Our decoding runs inside the decode engine's step. ShadowServe (2025)
  measured 30%+ slowdowns from GPU decompression on the decode GPU. The sweep's TPOT measures this.
- **vLLM 0.32.0 needs CUDA 13**, so NVIDIA driver 580 or newer. `pd_setup.sh` stops otherwise.
- **Link speeds used in the literature**: DistServe 25 Gb/s between nodes, CacheGen 3 Gb/s (sweeps 0.4-400),
  ThunderServe 5 and 40, ShadowServe 10-40, Prefill-as-a-Service about 100. The 1:3 prefill:decode ratio is an
  example only; Dynamo says there is no universal ratio.

### Resume the diff protocol

1. **NUC smoke**, after the T environment's `a5000_setup.sh` has run once. It should print `SELFTEST OK`, then
   `check <codec>: OK`, then `PUSHED ..._smoke`.

   ```bash
   bash <<'NUC'
   set -Eeuo pipefail
   cd ~/kv_llama32
   git fetch -q origin pd-vllm-protocol && git checkout -q pd-vllm-protocol && git pull -q --ff-only origin pd-vllm-protocol
   export ALLOW_NO_SLURM=1 KV_HOME=$HOME/kv_a5000
   bash scripts/pd_setup.sh
   SHARE_GPU=1 NUM_DECODE=1 SMOKE=1 bash scripts/run_pd_vllm.sh
   NUC
   ```

2. **The A5000 run**, inside a 4-GPU allocation, with the commands in `docs/pd_protocol.md`:

   ```bash
   srun --partition=all --account=oana2026 --qos=research12h --nodelist=gpu-grad-01 \
        --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=12:00:00 --job-name=kv-pd --pty bash -l
   ```

   Inside it: `bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh && MODEL=llama32_3b bash scripts/run_pd_vllm.sh`.

### Separating the two ideas in code (not done yet)

`pd-vllm-protocol` contains both today: it sits on the T branch, and T is one of its codecs. A clean split:
- **T**: the T branch, plus `kv_compare/pd/codecs.py`, `wire.py`, `export.py` and `selftest.py`. These are the
  transfer format and its exact round trip, with no vLLM dependency.
- **Diff protocol**: `kv_compare/pd/connector.py`, `transport.py`, `paged.py`, the proxy, bench, opportunity and
  run scripts. Codecs are pluggable: it needs only `raw` to work, and loads T if it is installed.

---

## 3. What did not become a contribution: P, the resident pool

- **The P gate** (`benchmark_results/llama32_3b_poolgate_rtxa4500_20261030_1251/`, `scripts/bench_pool_gate.py`)
  tested exactkv's compressed pool, which attention reads in place. It is bit-identical to dense attention and
  allocates 12.72-12.82 bits per value (12.58 used).
  - Its attention runs at 0.64-0.76x the fastest dense kernel (SDPA or FlashInfer).
  - Generation at the maximum batch:

    | Prompt | Dense (FlashInfer) | Dense (SDPA) | exactkv pool, its larger batch | exactkv pool, the dense batch |
    | --- | --- | --- | --- | --- |
    | 2K + 128 tokens | 1,257 tok/s (49 seqs) | 1,176 tok/s | 879 tok/s (62 seqs) | 1,062 tok/s (49 seqs) |
    | 8K + 128 tokens | 344 tok/s (12 seqs) | 289 tok/s | 259 tok/s (15 seqs) | 258 tok/s (12 seqs) |

  - So the pool fits about 25% more sequences but generates 25-30% fewer tokens/s on the A4500.
- **Our own pool format, P1** (branch `experiment/gpu_pool_p1_source`): widths per channel, escapes per value, each
  head readable on its own, a fixed-size region that rejects cleanly when full.
  - Payload 11.36-11.72 bits on SmolLM pages (CPU check only); 13.05 allocated with its fixed region at 80% of raw.
  - Never run on CUDA; no attention kernel.
  - The cheapest decisive test, not done: run P1's packer on our Llama windows against exactkv's 12.58 bits.
- **Conclusion.** P has no measured unique contribution over exactkv today. Keeping waiting or reused KV in T format
  (8.3-9.3 bits, restored layer by layer) is a T extension, and part of idea 2's DRAM tier.

**Lossy work** (per-layer lossy codec, AQUA-KV and KIVI baselines) is paused. Its results are in
`docs/milestone_report_2026-10-29.md` on the T branch.

---

## 4. Branches, machines, environments

| Branch | What |
| --- | --- |
| `main` | The earlier Qwen / SmolLM studies and this note |
| `llama32-3b-lossy-lossless-compare` | **T**: codec, baselines, benchmark, results, runbook (contains `main`) |
| `pd-vllm-protocol` | **Diff protocol** in vLLM (contains the T branch) |
| `experiment/gpu_pool_p1_source` | The P1 pool source (CPU oracle, CUDA tile reader, never run on a GPU) |
| `wip/head-groups` | An earlier head-group experiment |

**Machines:**
- **NUC**: `gpu-intel-nuc`, one RTX A4500 (20 GB). The clone is `~/kv_llama32`, with
  `KV_HOME=~/kv_a5000` holding the venvs, HF cache, authors' sources, CUDA 12.8 toolkit and runs. Every script runs
  there with `ALLOW_NO_SLURM=1`.
- **A5000 node**: `gpu-grad-01` (4x RTX A5000, 24 GB) through Slurm (`--account=oana2026 --qos=research12h`). Every
  command runs inside the allocation.

**Environments:**
- `$KV_HOME/venv`: torch 2.11.0+cu128, triton 3.6.0, transformers 4.57.6, nvCOMP 5.3.0.16 (cu12; ExANS builds
  against it).
- `$KV_HOME/venv_vllm`: vLLM 0.32.0, torch 2.13 (CUDA 13).

**Pitfalls already hit:**
- Installing FlashInfer unconstrained replaced torch 2.11 with 2.15 + CUDA 13.2 in `$KV_HOME/venv` on the NUC.
  `a5000_setup.sh` now removes what it brought; the run scripts stop on a wrong torch.
- nvCOMP's Python package segfaults at random in encode on full runs; it now runs in a child process.
- ExANS needs nvCOMP 5.x; 6.0 removed the API it calls.
- Triton JIT checks must run from a file, not a heredoc.
- Running third-party repositories' code in Claude's sandbox was refused. All authors' codecs run on the NUC or
  A5000 only.

## 5. Open questions to settle when resuming

1. **T**: is the 400 Gb/s row reported with the PCIe cap? Which models next: Llama-3-8B and Qwen3-8B on the A5000?
2. **T**: try layer 0 on the decode side, which is cheap and helps every request.
3. **Diff protocol**: after the NUC smoke and the A5000 phases 1-2, are the reuse opportunities large enough to build
   phase 3?
4. **Diff protocol**: wire ExANS (strongest other codec) and nvCOMP into the vLLM path?
5. Should this plan also go into the team spec doc ("Strong-and-Fast KV Codec")?
