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
NUC, so a crash there costs that run only. It codes a request in pieces of 128 MiB (zstd ran out of memory on a
whole 8,192-token request, job 30226), each piece starting 16-byte aligned in the coded stream; the receiver
copies a piece that lands unaligned, since nvCOMP's decoder refuses one (job 30252: 3 of 4 transfers failed with
"not aligned to the required alignment of 16 bytes").

## Layouts and paths

- **1 prefill + 3 decode** and **2 prefill + 2 decode** (`LAYOUT=1p3d | 2p2d`). Each prefill engine has its own
  link; the proxy sends each request to the prefill engine with the fewest requests in flight, counted when it
  picks (until job 30210 the count waited for the request's task to run, and a burst went to one engine: 7 of 8
  prompts in that smoke's 2p2d runs of our connector, so those TTFTs are not comparable with NIXL's). The QoS
  `research12h` allows 4 GPUs at a time, so these are the two layouts there are.
- **Every point starts cold.** Before each codec's check and each sweep or multi-turn point, every engine's prefix
  cache is emptied (`POST /reset_prefix_cache`). The points use the same prompts, so before job 30210's fix a
  later codec's prefill hit its own cache (TTFT without the prefill; T2 without steps for layer 0) and the
  decode engines already held most of a prompt (8 or 16 tokens moved of 2048).
- **The link.** Our transport paces a virtual link per prefill engine (kv_compare/pd/transport.py): a chunk of n
  bytes is booked for 8n/speed seconds after what is queued, and the receiver is told of it only when its booking
  ends. The speed is set per point (`LINKS`, 3 to 400 Gb/s, or `native` for unpaced). The report gives what the link
  carried while busy, all its transfers together (job 30280: 9.97-9.99 Gb/s at 10); one transfer's own rate is lower
  whenever transfers share the link (3.7 Gb/s each there). The node's own path bounds the pacing: on the A5000 node,
  about 125 Gb/s GPU to GPU, so faster set speeds measure the node, and the *Carried* row shows where that starts.
- **Greedy sweeps.** `vllm bench serve` in vLLM 0.32 no longer sends temperature 0, so until job 30252 the
  sweeps sampled with the model's defaults (temperature 0.6, top-p 0.9): each codec's run wrote its own text and
  vLLM compiled its top-p kernels at first use. The sweeps now pass `--temperature 0`; the tier and multi-turn runs
  keep the server's sampling, as a chat client would.
- **Warm before measuring.** Before each codec's first point: one short request over every prefill-decode pair (its
  path opens on first use), then a burst of 16 concurrent requests with the point's own sampling (greedy for the
  sweeps, the server's for the tier and multi-turn runs). The report lists vLLM's "JIT compilation during
  inference" warnings with the point each fell in, by the engine log's own position at the point's start and end.
  Job 30252's one compile "inside" a point was in the warm-up just before it: the logs' times are whole seconds,
  and the old one-second slack counted it in.
- **A failed transfer is computed locally.** The engines run with `kv_load_failure_policy: recompute`: under vLLM's
  default the scheduler asserted on a failed request and the engine died (jobs 30176, 30226).
- **Through host memory at set link speeds** (`TRANSPORT=ring`, 3/10/25/100 Gb/s): the transport of
  `pd-vllm-protocol`, a paced virtual link over pinned host memory.
- **GPU to GPU directly** (`TRANSPORT=p2p`, unpaced). The receiver's landing ring is in its GPU memory, shared with
  the sender by a CUDA IPC handle, so the sender's copy goes peer to peer over PCIe. Every engine sees all the job's
  GPUs, its own first.
  - vLLM's NixlConnector (`CONNECTOR=nixl`) is the production reference for this path. By default each NIXL engine
    sees only its own GPU, as vLLM's own NIXL tests run; `NIXL_ALL_GPUS=1` gives it all of them, as our p2p
    engines have. Job 30252 ran NIXL with its own GPU only: TTFT p50 2271 ms against raw p2p's 749 ms. A process
    that cannot see the peer GPU likely cannot use CUDA IPC (peer to peer), so UCX may stage the copy; the engine
    logs now record UCX's choice (`UCX_PROTO_INFO=y`), which settles it. The matrices run NIXL with every GPU
    visible (`_allgpus` in the run name); the direct matrix keeps one 2p2d run with its own GPU only, to compare.
- **A CPU tier** (`LAYOUT=colo4`): 4 engines, each prefilling its own requests, each with a tier in host memory
  (`kv_compare/pd/tier.py`, LMCache's role).
  - Each request's full 256-token chunks are coded and kept when it finishes. A background thread does this,
    outside the engine's step.
  - A later request that starts with the same tokens loads those chunks to the GPU and decodes them into its
    blocks. Each chunk is checked against a checksum taken when it was stored.
  - The budget (`TIER_GB`, 8 GB) limits the coded bytes, so a smaller format keeps more conversations.
  - The engines' GPU cache is limited so that the conversations outgrow it: `GPU_BLOCKS` 640 (10K tokens each) at
    an 8K context, which holds an engine's 4 conversations at 8 clients but not its 8 at 16. In job 30226 the GPU
    prefix cache still served 68% of the smoke's tokens and the tier only 2-5 loads, so the smoke now caps each
    conversation's own start at 2,000 tokens (`TIER_PREFIX_MAX`) and runs 300 blocks (4.8K tokens) at a 4K context.
  - The table gives vLLM's own prefix-cache counters (GPU / external, from each engine's `/metrics`), so LMCache's
    use shows too; the bits are over the whole chunks loaded, with the share of them the requests needed.
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

## Stage 0: what the node is (`scripts/node_specs.py`)

About 10 minutes on the 4 GPUs: in every card run, before its matrix (elsewhere: `scripts/node_specs.sbatch`).
It records:
- the GPUs, PCIe, topology and peer-to-peer matrices;
- CPU, NUMA, memory, disks, quota and network;
- the Slurm limits and the software.

It also measures:
- every GPU's host legs and every pair;
- the layouts with every leg at once, direct and through host memory: 1+3 with each GPU as the prefill one, 2+2 in
  every pairing, and a CPU tier refilling 1-4 GPUs.

It decides which GPUs prefill (`PREFILL_GPUS`, `DECODE_GPUS`) and which link speeds the machine reaches.

## On gpu-grad-01: the card (`scripts/pd_card.sh`)

The cluster runs follow the other team's card workflow: the login host only checks and submits, every stage runs in a
Slurm allocation of its own on gpu-grad-01, everything lives in the node's `/tmp`, and only the results are pushed,
to a branch of their own that we then pull.

```
bash scripts/pd_card.sh STAGE PIN TAG        # PIN: GitHub's pd-baselines head; TAG: pd_<matrix>_<name>
```

| Stage | Allocation | What it does |
| --- | --- | --- |
| (login) | none | The checkout is at the pin and has no changes, the card's bytes are the pin's, GitHub's branch is the pin (for create and run; later stages only need the pin in its history). Then one `srun` with `--chdir=/tmp --export=ALL --propagate=NONE`. |
| create | 2 CPUs, 8 GB, 1 h | The code at the pin in `/tmp/$USER/kvpd-tools/<pin>` (it shares the home checkout's git objects; `benchmark_results` is left out). The campaign in `/tmp/$USER/kvpd-evidence/<tag>/repo`: the pin's tree with only `benchmark_results/pd_card/<tag>/` checked out, and a marker (node, owner, boot ID, pin, the matrix's runs). |
| run | 4 GPUs, 16 CPUs, 192 GB, 12 h (smoke 6 h) | The driver (580 or newer), the 4 GPUs and the disk checked before anything is installed. A fresh runtime in `/tmp/$USER/kvpd/<job>`. Then `scripts/a5000_setup.sh`, `scripts/pd_setup.sh`, `scripts/node_specs.py` and `scripts/pd_matrix.sh` with the runs recorded at create. Results, the log, the setup's records and the disk use go into the campaign. Once per campaign. |
| cleanup | another small allocation | The run job finished (Slurm's record); no process uses the runtime; it deletes exactly `/tmp/$USER/kvpd/<run job>` and writes a receipt. |
| verify-cleanup | a third one | Slurm's records of the run and the cleanup (completed), the receipt, the folder gone. |
| review | small | The combined report rebuilt from the kept results only, and an inventory: every file's size and SHA-256. |
| publish | small | The inventory unchanged, no credentials, no file over 95 MB; one commit on the pin with only the campaign's folder, pushed to `evidence/pd-baselines/<pin12>/<matrix>/job<run job>`. |

On the node (the node's own disk: it stays between jobs, a reboot wipes it):

| Folder | What | Kept |
| --- | --- | --- |
| `/tmp/$USER/kvpd-tools/<pin>` | the code at the pin, read only | yes |
| `/tmp/$USER/kvpd-evidence/<tag>/repo` | the campaign and its results | yes (published) |
| `/tmp/$USER/kvpd/<run job>` | environments, sources, captures, caches, temporary files | deleted by cleanup |
| `/tmp/$USER/kvpd-cache` | uv's package cache, uv's Python 3.12, the model, the CUDA 13 toolkit | yes, reused |

Disk. `/tmp` is the node's system disk, shared with every user (about 58 GB were free in November 2026; the other
repository's folders under `/tmp/$USER` are its own and are never touched). So:
- The environments are made by uv with `UV_LINK_MODE=hardlink`: every environment's files are hard links into the
  one cache, so the codec, vLLM and LMCache environments cost the cache once (about 18 GB) and a fresh set per job
  costs almost nothing.
- ExANS is built with the cluster's CUDA 12.6 (the module): no CUDA 12.8 download. nvCOMP 5.3 links its CUDA
  runtime statically, so the cluster's 12.6 suffices.
- IBP's CUDA 13 toolkit is installed once into the cache and pruned to about 4 GB, only when a matrix uses IBP
  (smoke, tier) and the disk allows it; otherwise IBP is left out and the report says why.
- T2's calibration windows are captured, fitted and deleted one set at a time (4 GB at once, not 16), and every
  capture is deleted after the self-test: the runs read only `pd_codecs.pt`.
- The run stage adds up what it will need (the cache's missing parts, then about 16 GB for the job) and starts
  only if 10 GB stay free; `disk_plan.json` and `disk_after.json` record the plan and the measured sizes.

Every cache of the run points into its runtime or the shared cache (pip's, uv's, Hugging Face's, vLLM's, Triton's,
TorchInductor's, CUDA's JIT cache, Python's bytecode), so nothing is written to the home directory or into the
code.

### The login-host block

From a fixed login host, inside tmux (the run stage holds the terminal for hours; a dropped connection would end
it): `tmux new -s kvpd` (later: `tmux attach -t kvpd`, on the same login host). Then paste:

```bash
bash <<'LOGIN'
set -Eeuo pipefail
umask 077
trap 'printf "PD CAMPAIGN STOPPED: line=%s exit=%s; the campaign and its output are kept\n" "$LINENO" "$?" >&2' ERR
test -z "${SLURM_JOB_ID:-}"
cd ~/kv_reconstruction
test -z "$(git status --porcelain --untracked-files=no)"
git fetch -q --no-tags origin pd-baselines
git switch -q pd-baselines 2>/dev/null || git switch -q -c pd-baselines --track origin/pd-baselines
git merge -q --ff-only origin/pd-baselines
export KVPD_PIN=$(git rev-parse HEAD)
test "$(git ls-remote origin refs/heads/pd-baselines | awk '{print $1}')" = "$KVPD_PIN"
export KVPD_MATRIX=smoke                     # then ring1p3d, ring2p2d, direct, tier, t2eager, links: one campaign each
export KVPD_TAG="pd_${KVPD_MATRIX}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p ~/.kvpd_campaigns
STATE=~/.kvpd_campaigns/$KVPD_TAG.sh
printf 'export KVPD_PIN=%q\nexport KVPD_TAG=%q\n' "$KVPD_PIN" "$KVPD_TAG" > "$STATE"
printf 'PIN=%s\nTAG=%s\nSTATE=%s\n' "$KVPD_PIN" "$KVPD_TAG" "$STATE"
for STAGE in create run cleanup verify-cleanup review publish; do
    bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG"
done
echo "Campaign $KVPD_TAG published (the PUBLISHED line above names its branch)."
LOGIN
```

The run stage goes on through closeout and publication whatever the matrix's outcome (`KVPD_RUN_STATUS`:
completed, completed with failures, setup failed, matrix failed): a failed setup's logs are evidence too. It stops
the sequence only when its checks refuse to start (driver, GPUs, disk, no uv); then close the campaign out with
the block below and start a new tag.

To continue a campaign in a later shell (after a stop, or to close it out):

```bash
cd ~/kv_reconstruction
source ~/.kvpd_campaigns/<tag>.sh            # KVPD_PIN, KVPD_TAG
git switch -q --detach "$KVPD_PIN"           # the card runs only from a checkout at its pin
for STAGE in cleanup verify-cleanup review publish; do bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG" || break; done
git switch -q pd-baselines
```

A stage that is done says so and goes on (cleanup, review and publish are create-once). `run` happens once per
campaign: a new measurement is a new tag. After publish, the results are on GitHub; pull them with the command it
prints (`git fetch origin evidence/pd-baselines/<pin12>/<matrix>/job<id>`), or say the branch name and we merge it
into `pd-baselines`.

Options (exported before the block, passed to the node): `KVPD_TIME` (the run's limit), `KVPD_IBP` (auto, 0, 1),
`RUNS` at create (scripts/pd_matrix.sh's own list instead of the matrix's; the run uses what create recorded).

## Elsewhere: one batch job (`scripts/pd_matrix.sbatch`)

On the NUC or a cluster without the card's rules, the older form runs setup and a matrix as one batch job, with
everything under `KV_HOME`, and pushes each run's folder to `pd-baselines` itself:

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

- The setup (`scripts/a5000_setup.sh`, then `scripts/pd_setup.sh`):
  - the codec environment, with a CUDA 12 toolkit that builds ExANS;
  - vLLM 0.32.0 with nvCOMP 6.0 for CUDA 13 and NIXL 1.5.2;
  - LMCache 0.5.7 in an environment of its own;
  - IBP, built for vLLM's torch with a CUDA 13 toolkit (one build fix: sm_70 to sm_86);
  - ExANS;
  - the calibration texts and windows, and T2's sets;
  - the self-test of every codec through the transport.

  A rerun skips what is there.
- Each matrix fits one 12-hour job. Every run is a folder of its own
  (`<results>/llama32_3b_pd_<layout>_<connector>[_<transport>][_eager]_rtxa5000x4_<date>`), and each matrix has a
  combined report (`<results>/pd_matrix_<matrix>_<gpu>_<date>/summary.md`).
- A run that fails is listed in the combined report; the matrix goes on.
- Disk with pip (no uv): about 90 GB under `KV_HOME`. That covers the environments (8 + 12 + 12 GB), two CUDA
  toolkits (8 + 5 GB), the model (6 GB), the captures (7 GB, plus one T2 set's 4 GB while it is fitted), and the
  caches.

The matrices (the times are estimates until the smoke has measured them):

| Matrix | Runs |
| --- | --- |
| smoke | each path, layout and connector once, with few prompts |
| ring1p3d | 1+3 through host memory at 3/10/25/100 Gb/s: the in-process baselines, T and T2; then nvCOMP |
| ring2p2d | the same with 2+2 |
| direct | GPU to GPU for 1+3 and 2+2, every codec, nvCOMP apart; NIXL for both layouts with every GPU visible, and 2+2 with its own GPU only |
| links | 1+3 over 10, 25, 50, ... 400 Gb/s (steps of 25), through host memory and GPU to GPU (paced, then unpaced), 16 requests at once: raw, SplitZip, ExANS, AdaCode, exactkv, T (hbh) and T2 (hbh, rank 256). One table per path: TTFT p50 per codec and speed, what the link carried, where each codec beats raw, and where a request's time goes |
| tier | the CPU tier with every codec, nvCOMP apart; LMCache |
| t2eager | T and T2 in eager engines: host memory at 10 and 100 Gb/s, GPU to GPU, the CPU tier |

## Honest limits

- gpu-grad-01 is shared: the run has its 4 GPUs to itself, but other jobs on the node share its CPUs, host memory
  bandwidth and PCIe switches. Each card run measures the links (`node_specs/`) in the same allocation, so every
  campaign's numbers come with the node as it was.
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
