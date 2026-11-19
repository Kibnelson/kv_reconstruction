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
| aquakv-2, aquakv-3 | **lossy**, for speed and size only: AQUA-KV (arXiv 2501.19392), the authors' algorithm (goodevening13/aquakv) on the GPU. Keys without RoPE; the first 4 positions as they are; layer 0 HIGGS 4-bit; layer l keys predicted from layer l-1's reconstructed keys, values from this layer's reconstructed keys and layer l-1's values (BF16 linear maps fitted at setup on the calibration documents' own reconstructions, `pd_export_codecs.py --aqua-bits`); the residual HIGGS b-bit (Hadamard over 1,024 channels, one scale per token and group, EDEN grid). It matches `kv_compare.baselines.aqua_official` value for value (test_kv_compare). No checksum: the check reports its KV error instead (relative L2, the sender's), and the outputs' agreement with local prefill | every path |
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

## Workloads: the baselines' own, and ours

Every codec runs every workload: the same prompts (real text from WikiText-2, HumanEval, GSM8K, MMLU and PTB, cut to
an exact token count), the same number of requests, greedy decoding, caches reset before each point. The `workloads`
matrix adds, beside our own serving workload, each baseline's workload as its authors reported it (read at the
sources on 16 November 2026). None reports a request count or an arrival rate for a serving test, so these are their
shapes: how many requests at once, how long, how much output.

| Family | The authors' setup | Ours (`T:B:O:R`: prompt tokens, at once, out, rounds) | Left out on this node |
| --- | --- | --- | --- |
| SplitZip, 1 at a time (`sz1`) | SGLang + Mooncake PD, Qwen3-32B and others, H200: batch 1, sequence 512 to 131K | 512, 1K, 2K, 4K, 8K (8 rounds), 16K (4) and 32K (4) tokens, 1 at a time | 64K and longer: our connector moves a request whole, and its KV then outgrows what the engines leave free (64K: 7.3 GB, held twice while it is coded); a layer-by-layer send would lift that |
| SplitZip, 16 at once (`sz16`) | batch 16, sequence 128 to 64K | 128 to 16K tokens, 16 at once, 2 rounds | 32K and 64K (16 x 32K is 2.5 times what the three decode engines hold at once) |
| SplitZip, 1,024 tokens (`sz1024`) | sequence 1,024, batch 1 to 256 | 1, 2, 4, ... 256 at once | - |
| SplitZip, 32K tokens (`sz32k`) | sequence 32,768, batch 1 to 128 | 1, 2 and 4 at once | 8 and more |
| IBP (`ibp`) | InfiniGen KV offload, OPT, A100: 1,920 tokens in, batch 20, 128 out | 20 at once x 1,920 tokens, 128 out, 2 rounds | - |
| exactkv (`exactkv`) | RTX 4090 decoding loop: 4,096 tokens in, 256 greedy out, as many as fit | 16 at once (one decode engine's cache) x 4,096, 256 out, 2 rounds | - |
| ours | - | 2,048 tokens, 32 requests at 0.5, 1 and 2 per second and all at once; goodput at a 2 s TTFT | - |

ExANS reports a codec benchmark on one 256K-token KV snapshot (no model or data named) and AdaCode accuracy and
memory at batch 1, with no serving or transfer test; nvCOMP is a library. Their codecs run on all of the above.

The `workloads` campaign (about 9 hours):
1. host memory at 10 and 100 Gb/s: raw, T (hbh), T2 (hbh, rank 256), SplitZip, ExANS, AdaCode, and AQUA-KV 2- and
   3-bit (lossy, for speed);
2. GPU to GPU, unpaced: the same codecs;
3. vLLM's NIXL, every GPU visible;
4. nVIDIA nvCOMP (ANS, Zstd, Bitcomp) on its own engines, host memory at 10 and 100 Gb/s;
5. exactkv on its own engines, with raw beside it: host memory at 10 and 100 Gb/s, then GPU to GPU.

The `wlong` campaign (about 4 hours): the biggest workloads, host memory and GPU to GPU, with AQUA-KV 2-bit, then
exactkv apart: 32K-token prompts, 16 at once of 8K and 16K tokens, 128 and 256 at once of 1,024. Where more are at
once than the decode engines hold, vLLM queues them, as it would in service. Apart, so that an engine lost there costs
nothing of the main campaign.

Why exactkv runs apart (job 30334): on 128-token prompts, 16 at once, at 10 Gb/s, an install of exactkv hit an
illegal memory access on all three decode GPUs within half a second (10:38:09), after 5 to 7 installs of the same
shape had succeeded on each; it had passed every earlier point (its check, the 2,048-token sweep at every rate, and
512 and 1,024 tokens one at a time). A CUDA error of that kind poisons the engine's GPU, so every engine died. The
logs do not say whether exactkv's tile decoder read outside the rows it was sent or an earlier kernel on the GPU had
failed. The receiver now checks, before exactkv's decoder runs, that every raw row a page names was sent and that
each buffer has its page shape (one GPU sync for all layers). A bad transfer then fails alone and vLLM computes the
prompt, and a CUDA error at that check is named as one from earlier work on the GPU. Such an error is no longer
retried either, so the record keeps its first message.

A run also stops at a dead engine now. Before each point the bench asks every engine's `/health`, and it asks
again every 30 s while requests stay in flight. A point whose requests are still in flight 10 minutes after its
benchmark ended (`IDLE_TIMEOUT_S`) stops the run too, and so does a benchmark that runs past an hour
(`POINT_TIMEOUT_S`). The points measured before stay, the run's report is written and says where it stopped, and the
run is packaged as `_partial`. The matrix then goes on to its next run, with new engines. In job 30334 the bench
instead waited 30 minutes after each of the next points. Within each run, T and T2 come first and exactkv last, so a
crash costs the others nothing.

What the workloads campaign showed (job 30371, all six runs complete), and what changed after it:
- T's and T2's decoder took a fixed 135-190 ms per request in head by head, from 128 tokens to 8K (the other codecs:
  5-45 ms), while their link time was the lowest of all (half of SplitZip's). The decoder issued about 7,000 small
  launches per request (4 token chunks x 16 head groups x 28 layers, each group with its product of the groups
  before it): launch-bound, not data-bound. It also runs in the decode engine's step, so the time per output token
  of every request went from 12 to 17-18 ms. The receiver now decodes in one chunk (`TKV_DECODE_CHUNKS`, default 1;
  0 restores the encoder's choice). The chunks only split the decoder's launches across CUDA streams; the stream is
  the same, and the round trips are checked bit-exact both ways.
- Every 16K transfer of T and T2 ran out of memory on the prefill GPU (and 5 of 8 of T's at 8K, 100 Gb/s): the
  encoder's scratch is about 4x the request's KV (a stream buffer of twice its words, its features, its padded
  words). Requests longer than 4,096 tokens are now coded in pieces of 4,096 (`TKV_PIECE_TOKENS`), each from its
  own position, with nothing changed in any segment of the stream; the receiver decodes them one after another into
  the request's words and installs those layer by layer.
- At 16K, raw's 1.88 GB transfer was over both GPU budgets (1 GiB to send, 1.5 GiB to receive) and went through
  host memory, while SplitZip's, exactkv's and AdaCode's, just under 1.5 GiB, stayed on the decode GPU: their
  1.3-1.4x over raw there was mostly that. A transfer now stays on the GPU when it is the only one held, whatever
  its size; the reports mark a point "(host)" when any of its transfers still went through host memory.
- ExANS rejected one of its own records ("invalid or incompatible ExpANS record header") twice, on the same
  128-token prompt, once through host memory at 100 Gb/s and once GPU to GPU; the retry failed the same way, and
  vLLM computed that prompt. The sender now sends raw any block whose record header lacks the magic or a sane size
  (counted as `bad_records`), and a failed decode says whether the headers arrived as the sender made them.
- exactkv did not fail at all (the 128-token, 16-at-once workload that killed job 30334 included), nor did its check.
- T2's receiver never recomputed layer 0 (its recompute differs from the compiled engine's cache in about 47 % of the
  words), so T2 coded layer 0 in every request; that is still open.

The `wlours` campaign (about 3.5 hours) measures the changes: raw, T (hbh), T2 (hbh, rank 256), SplitZip and
AdaCode on the same workloads, host memory at 10 and 100 Gb/s and GPU to GPU.

What `wlours` showed (job 30436, both runs complete, no transfer failed):
- One decoder chunk made T's and T2's head-by-head installs 2.4-2.6x faster: 136-163 ms became 40-68 ms from 128
  tokens to 4K. Time to first token fell by about 95 ms (2,048 tokens at 10 Gb/s: 495 -> 400 ms, raw 394), and
  with 16 prompts of 128 tokens at once from 934 to 344 ms. The time per output token under load fell from
  17-18 ms to 14 (raw 12); one request at a time, it is raw's.
- 16K prompts now go through for T and T2 (in four pieces), and raw's 16K transfer stays on the GPU: raw's time to
  first token at 16K went from 3.7-3.9 s to 2.1 s at 100 Gb/s and GPU to GPU, so job 30371's 1.3-1.4x for the
  coded transfers there is gone (SplitZip and AdaCode within 1 % of raw, T2 rank 256 12 % and T 20 % slower; at
  10 Gb/s AdaCode 1.13x, SplitZip 1.12x, T2 rank 256 1.10x, T 1.06x).
- What is left of T's install is about 60 ms for each piece decoded, nearly independent of its size (61 ms at 512
  tokens, 71 at 4K): 8K, in two pieces, takes 147 ms, 16K, in four, 284. T's encode grows with the prompt (33 ms
  at 2K, 127 at 8K) and is now the larger of the two from 8K on.
- Under full load at 10 Gb/s T carries 3.60 requests per second, T2 rank 256 3.68, SplitZip 3.43 and raw 3.00; at
  100 Gb/s T 4.13 against raw's 4.21 and SplitZip's 4.58.

After `wlours` (branch pd-decode until it is measured):
- The receiver decodes T's and T2's stream as sent (wire.compact_starts): a segment's states and words keep their
  distance from its first used word in the dense array, so the S x cap slot buffer (twice the request's KV) and the
  loop of masked copies and host syncs that filled it are gone. The sender packs the tails with one kernel.
- Installs run on a thread and CUDA stream of their own (`async_install`, on in scripts/run_pd_vllm.sh;
  ASYNC_INSTALL=false restores the step): the engine's step goes on while a transfer decodes, and the request is
  reported loaded once its install has finished on the GPU; a request that ends meanwhile is reported after it, so
  vLLM never frees blocks that are being written. T2 transfers whose receiver recomputes layer 0 still install in
  the step (the recompute runs the engine's own modules).
- The `capacity` campaign asks how many requests each codec carries: 96 prompts of 2,048 tokens
  at 1 to 8 per second, and for each codec and link speed the highest rate at which 90 % of the requests meet a
  1 s TTFT target, for raw, T, T2 rank 256, SplitZip, AdaCode, ExANS, AQUA-KV 2-bit (lossy) and exactkv; host
  memory at 10, 25, 50 and 100 Gb/s and GPU to GPU (about 4.5 hours). One A5000 prefill makes KV at about
  8-10 Gb/s, so above that the link is idle most of the time and only the GPUs' work decides; the speeds that
  matter for a datacenter are those per prefill GPU sharing a link.

The report gives one table per path and link: a row per workload, a column per codec, TTFT p50 with its speedup
over raw (SplitZip's measure), then requests per second with the gain over raw.

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
export KVPD_MATRIX=smoke                     # then ring1p3d, ring2p2d, direct, tier, t2eager, links, workloads
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
| workloads | 1+3, every codec on the baselines' published workloads and ours (above, "Workloads"), AQUA-KV 2- and 3-bit beside them: host memory at 10 and 100 Gb/s, GPU to GPU, NIXL, nvCOMP; exactkv in runs of its own, last |
| wlong | the biggest published workloads (32K-token prompts, 16 at once of 8K and 16K, 128 and 256 at once of 1,024): host memory at 10 and 100 Gb/s, GPU to GPU; exactkv apart |
| wlours | the workloads campaign again for T and T2 after job 30371 (decoder in one chunk, pieces over 4K tokens, one transfer over the GPU budget kept on the GPU), with raw, SplitZip and AdaCode: host memory at 10 and 100 Gb/s, GPU to GPU |
| capacity | 96 prompts of 2,048 tokens at 1-8 per second: the highest rate at which 90 % meet a 1 s TTFT target, per codec, host memory at 10, 25, 50 and 100 Gb/s and GPU to GPU |
| links | 1+3 over 10, 25, 50, ... 400 Gb/s (steps of 25), through host memory and GPU to GPU (paced, then unpaced), 16 requests at once: raw, T (hbh), T2 (hbh, rank 256), SplitZip, ExANS, AdaCode, AQUA-KV 2-bit and exactkv (last). One table per path: TTFT p50 per codec and speed, what the link carried, where each codec beats raw, and where a request's time goes |
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
- IBP built on the A5000 node first in job 30316, from the CUDA 13 runfile's extracted files (its installer stops
  at a root-only step). Its preprocess copies the mask it picked with `cudaMemcpy` on the legacy default stream
  while its kernels run on PyTorch's current stream; called from our connector's own (non-blocking) streams, the
  two raced and every tier load failed its check (jobs 30316, 30322), though the self-test, on the default stream,
  was exact. IBP now runs on the default stream, ordered with ours, as its own tests run; its store waits for its
  in-place compression, which returns before its kernel ends. A chunk that fails its check leaves the tier, so vLLM
  computes it instead of asking for it again (job 30316: 18,000 times an engine).
- IBP compresses in place, so its host buffer keeps the raw size. The tier counts that size, not IBP's compressed
  size: IBP saves PCIe time, not capacity.
- T2's layer-0 recompute needs eager engines to match vLLM's cache. In compiled engines T2 may code layer 0,
  which is still exact but larger; the report says how often.
