# Every baseline in real vLLM PD on 4 GPUs (branch `pd-baselines`)

Branch `pd-baselines`, from `pd-vllm-protocol` with `t-ideas` merged in (T2). Goal: the same vLLM, the same
prompts and the same 4 RTX A5000s on `gpu-grad-01`, with the KV moved by each bit-exact codec in turn. That means
the published baselines, then T and T2, and vLLM's own connectors as references. Measured end to end:
- requests per second and goodput, TTFT and TPOT;
- each transfer's or tier load's bytes and times, and a bit-exactness check of every one.

## What is compared

| Codec | What runs | Where it runs |
| --- | --- | --- |
| raw | BF16 as it is | every path |
| splitzip | the authors' SplitZip codec and code book | every path |
| exans | the authors' ExANS library (`libopenlake_expans.so`, CUDA 12.8 build) through ctypes, as their own vLLM connector loads it | every path |
| nvcomp-ans, -zstd, -bitcomp | nVIDIA nvCOMP 6.0 (CUDA 13 Python package), the request's bytes as one buffer | every path, on their own engines |
| adacode | the authors' AdaCode CompressorBitMap | every path |
| exactkv | the authors' exactkv page encoder and tile decoder | every path |
| ibp | the authors' Invariant Bit Packing: compressed in pinned host memory, decompressed on the way to the GPU | the CPU tier only (its design point) |
| ours-hbh, -dense, -rank256 | T | every path |
| t2-hbh, -dense, -rank256 | T2: T with a constant set picked per request and layer 0 recomputed by the receiver | every path |
| nixl | vLLM's NixlConnector, no codec | GPU to GPU reference |
| lmcache | LMCache 0.5.7's CPU offload in vLLM (uncompressed) | CPU-tier reference |

nvCOMP runs on engines of its own in every matrix. Its Python package crashed whole processes at random on the
NUC, so a crash there costs that run only.

## Layouts and paths

- **1 prefill + 3 decode** and **2 prefill + 2 decode** (`LAYOUT=1p3d | 2p2d`). Each prefill engine has its own
  link; the proxy sends each request to the prefill engine with the fewest requests in flight. The QoS
  `research12h` allows 4 GPUs at a time, so these are the two layouts there are.
- **Through host memory at set link speeds** (`TRANSPORT=ring`, 3/10/25/100 Gb/s): the transport of
  `pd-vllm-protocol`, a paced virtual link over pinned host memory.
- **GPU to GPU directly** (`TRANSPORT=p2p`, unpaced). The receiver's landing ring is in its GPU memory, shared with
  the sender by a CUDA IPC handle, so the sender's copy goes peer to peer over PCIe. Every engine sees all the job's
  GPUs, its own first.
  - vLLM's NixlConnector (`CONNECTOR=nixl`) is the production reference for this path. By default each NIXL engine
    sees only its own GPU, as vLLM's own NIXL tests run; `NIXL_ALL_GPUS=1` gives it all of them.
- **A CPU tier** (`LAYOUT=colo4`): 4 engines, each prefilling its own requests, each with a tier in host memory
  (`kv_compare/pd/tier.py`, LMCache's role).
  - Each request's full 256-token chunks are coded and kept when it finishes. A background thread does this,
    outside the engine's step.
  - A later request that starts with the same tokens loads those chunks to the GPU and decodes them into its
    blocks. Each chunk is checked against a checksum taken when it was stored.
  - The budget (`TIER_GB`, 8 GB) limits the coded bytes, so a smaller format keeps more conversations.
  - The engines' GPU cache is limited (`GPU_BLOCKS` 2048, 32K tokens each) so that the conversations outgrow it.
  - The workload is vLLM's multi-turn benchmark on fresh conversations at each point, with conversation routing.
  - LMCache's CPU offload (`CONNECTOR=lmcache`) is the reference, with the same chunk size and budget.

## T2 in vLLM

- **The constant set.** The sender ranks five constant sets per request (WikiText-2, MBPP, GSM8K, MMLU, PTB) by
  T's int8 product on 1,024 sampled tokens of layers 1-2. The header carries the chosen set's name.
  - The sets are fitted by `scripts/pd_export_codecs.py --t2-sets` from calibration windows of
    `scripts/fetch_calib_texts.py`, captured first (about 16 GB, deleted after the export).
- **Layer 0 recomputed** (`kv_compare/pd/layer0.py`). A token's layer-0 K and V depend only on its id and
  position, so the receiver computes them with the engine's own modules: embedding, input norm, fused QKV
  projection, rotary embedding.
  - Bit identity depends on the product's row count, so the prefill engine's scheduler side records each step's
    row count for every request. The recompute pads to it.
  - The sender recomputes too and compares with its own cache. It tries the scheduler's row count, then that count
    rounded up (vLLM pads to CUDA-graph sizes).
  - When the recompute matches, layer 0 stays out of the stream; otherwise it is coded. Either way the transfer is
    bit-exact.
  - In eager engines (`ENGINE_EAGER=1`) the forward runs these same modules. Compiled engines may fuse the norm
    and the product differently. The report shows how often layer 0 was recomputed, and the `t2eager` matrix
    measures the eager case.

## Stage 0: what the node is (`scripts/node_specs.sbatch`)

One 4-GPU job of about 10 minutes. It records:
- the GPUs, PCIe, topology and peer-to-peer matrices;
- CPU, NUMA, memory, disks, quota and network;
- the Slurm limits and the software.

It also measures:
- every GPU's host legs and every pair;
- the layouts with every leg at once, direct and through host memory: 1+3 with each GPU as the prefill one, 2+2 in
  every pairing, and a CPU tier refilling 1-4 GPUs.

It decides which GPUs prefill (`PREFILL_GPUS`, `DECODE_GPUS`) and which link speeds the machine reaches.

## Running it

```bash
module load slurm
cd ~/kv_llama32                        # or your clone
git fetch -q origin pd-baselines && git checkout -q pd-baselines && git pull -q --ff-only origin pd-baselines
export KV_HOME=/a/place/with/90GB/kv_a5000
sbatch scripts/node_specs.sbatch                          # stage 0 (about 10 minutes)
MATRIX=smoke sbatch --time=4:00:00 scripts/pd_matrix.sbatch   # the one-time setup, then every path once
MATRIX=ring1p3d sbatch scripts/pd_matrix.sbatch           # then one job each: ring1p3d ring2p2d direct tier t2eager
squeue --me; tail -f kv-pd-matrix.<job id>.out
```

- The first job also does the setup (`scripts/a5000_setup.sh`, then `scripts/pd_setup.sh`):
  - the codec environment, with the CUDA 12.8 toolkit that builds ExANS;
  - vLLM 0.32.0 with nvCOMP 6.0 for CUDA 13 and NIXL 1.5.2;
  - LMCache 0.5.7 in an environment of its own;
  - IBP, built for vLLM's torch with a CUDA 13 toolkit (one build fix: sm_70 to sm_86);
  - ExANS;
  - the calibration texts and windows, and T2's sets;
  - the self-test of every codec through the transport.

  A rerun skips what is there.
- Each matrix fits one 12-hour job. The QoS runs them one after another. Every run is pushed as its own folder
  (`benchmark_results/llama32_3b_pd_<layout>_<connector>[_<transport>][_eager]_rtxa5000x4_<date>`), and each
  matrix as a combined report (`benchmark_results/pd_matrix_<matrix>_<gpu>_<date>/summary.md`).
- A run that fails is listed in the combined report; the matrix goes on.
- Disk: about 90 GB under `KV_HOME`. That covers the environments (8 + 12 + 12 GB), two CUDA toolkits (8 + 5 GB),
  the model (6 GB), the captures (7 GB, plus 16 GB while T2's sets are fitted), and the caches.

The matrices (the times are estimates until the smoke has measured them):

| Matrix | Runs |
| --- | --- |
| smoke | each path, layout and connector once, with few prompts |
| ring1p3d | 1+3 through host memory at 3/10/25/100 Gb/s: the in-process baselines, T and T2; then nvCOMP |
| ring2p2d | the same with 2+2 |
| direct | GPU to GPU for 1+3 and 2+2, every codec, nvCOMP apart; NIXL for both layouts |
| tier | the CPU tier with every codec, nvCOMP apart; LMCache |
| t2eager | T and T2 in eager engines: host memory at 10 and 100 Gb/s, GPU to GPU, the CPU tier |

## Honest limits

- The links of the ring path are emulated on one machine (as in `pd-vllm-protocol`). The GPU-to-GPU path and the
  CPU tier are the machine's real paths.
- ExANS runs its CUDA 12.8 library beside vLLM's CUDA 13 torch in one process (two CUDA runtimes, one driver). The
  self-test checks it bit-exact before any run.
- IBP has never built on our machines (the NUC's build failed). Its build log is kept; without it, IBP is listed as
  not loaded.
- IBP compresses in place, so its host buffer keeps the raw size. The tier counts that size, not IBP's compressed
  size: IBP saves PCIe time, not capacity.
- T2's layer-0 recompute needs eager engines to match vLLM's cache. In compiled engines T2 may code layer 0,
  which is still exact but larger; the report says how often.
