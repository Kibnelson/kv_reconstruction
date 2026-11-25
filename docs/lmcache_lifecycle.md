# LMCache lifecycle: every block and chunk, from P's GPU to LMCache and to D's GPU

Branch `lmcache-lifecycle` (from `pd-blend`). A clean 1 prefill + 3 decode set-up in which **P and the three D
engines share one LMCache server**. P computes each prompt and its connector stores the prompt's chunks in the
server; the turn's D engine looks the prompt up there and pulls what it does not already hold. Hooks in our own
copies of vLLM and LMCache record every step of every block's and chunk's life, and a report puts them back
together. It runs in two modes: the chunks in the server's memory, or every chunk on disk and read back from disk.

| | |
| --- | --- |
| Model | Llama 3.2 3B (`unsloth/Llama-3.2-3B` at `d4446454`), bf16, 16-token vLLM blocks |
| Engines | vLLM 0.32.0, P on GPU 0 (`kv_both`), D1-D3 on GPUs 1-3 (`kv_consumer`: they only pull) |
| Cache | LMCache 0.5.7.post1 in multi-process mode: one `lmcache server` that sees all four GPUs; each engine's `LMCacheMPConnector` talks to it over ZMQ and the server copies KV to and from each engine's GPU memory over CUDA IPC |
| Chunk | 256 tokens (16 vLLM blocks); LMCache names it by a hash of its tokens and everything before them (blake3) |
| Workload | MTRAG's first 4 conversations, interleaved layout (each turn's prompt is the previous prompt, its answer, the new passages and question); routed by conversation, so a conversation's turns go to one D engine |
| Proxy | store mode: P gets the prompt with one token to generate (its connector stores the chunks), then after `STORE_WAIT_MS` (500 ms) the D engine gets the full request |

## The two modes

| Mode | Server memory (L1) | Disk (L2) | What a D pull reads |
| --- | --- | --- | --- |
| `memory` | 16 GB, LRU | none | memory |
| `disk` | 4 GB, a write buffer only (eviction `noop`) | `fs` adapter, one file per chunk, `O_DIRECT` (no page cache); store policy `skip_l1`: a chunk is deleted from memory as soon as it is on disk; prefetch policy `default`: a chunk read from disk is not kept in memory | disk, every time |
| `tiered` (optional) | 2 GB, LRU | the same files, written through (store policy `default`) | memory while the chunk is there, disk after it is evicted |

`O_DIRECT` matters for the disk mode: without it the kernel's page cache would answer reads from memory.

How a disk-mode run shows that it did what it says (its report's checks, from the trace and the server's own
snapshots at the end):

- every chunk written to memory was then written to disk (`l2.keys.stored`) and only after that removed from
  memory (`l1.keys.evicted`): "on disk then out of memory", N of N;
- every pull read its chunks from disk ("pulls from disk", N of N: the file adapter's read, `lmc.l2.read`, or before
  that hook LMCache's access events), and none was looked up while the chunk was still in memory;
- the server's memory at the end is empty (its `/status`: 0 objects, 0 bytes), and the disk tier holds one file per
  chunk (`server/l2_files.tsv`, listed before the run's folder is deleted).

Job 30728: 114 of 114 on disk then out of memory (each removed within 1 ms of reaching disk, after 189 ms in memory
at the median, 472 at most), 102 of 102 pulls from disk, none while in memory, memory at the end 0 objects and 0
bytes, 114 files of 29,360,128 bytes (one chunk: 256 tokens of 112 KB) on disk.

## Our copies of vLLM and LMCache, and the hooks

`third_party/vllm` (v0.32.0) and `third_party/lmcache` (v0.5.7.post1) are our own copies, imported once and
unchanged (commit `b2189bd`), detached from upstream; `third_party/README.md` says what was imported and how it was
checked against the release wheels. Every change since is a trace hook:

- A hook is one whole added line ending in `# KVTRACE`, for example
  `if kvtrace.ON: kvtrace.hook("vllm.gpu.cached", request=request, blocks=new_full_blocks, ...)  # KVTRACE`.
  It passes the objects at hand and nothing else; `kvtrace/events.py` turns them into a small record. No upstream
  line is changed or removed.
- Removing every line ending in `# KVTRACE` gives back the imported file exactly. `python3 scripts/lmc_overlay.py
  check` verifies that against the import's hashes (`third_party/*.sha256`), and that no file was added.
- To see every hook: `git grep -n "# KVTRACE" -- third_party/`, `git diff b2189bd -- third_party/`, or
  `python3 scripts/lmc_overlay.py hooks`.
- Tracing is on only when `KVTRACE_DIR` is set; otherwise every hook costs one attribute test. A hook never raises:
  an extractor's error becomes a `kvtrace.error` record (at most 20 per event name) and the engine goes on.

How they run: the job installs the release wheels of exactly these versions (for their compiled kernels), then
`scripts/lmc_overlay.py build` makes a per-job tree with the wheels' packages hard-linked file by file and our five
hooked files written over them, plus `kvtrace`; it first checks that every Python file of our copies, as imported,
is byte-identical to the wheel's. LMCache's compiled modules are ours: all four the wheel has (`cuda_ops`,
`lmcache_native`, `lmcache_fs`, `lmcache_redis`), built together from our copy's `csrc/` against this environment's
torch (`scripts/lmc_build_ops.py`; together, because pybind11 shares a type between modules only when they are built
alike: job 30722), with the environment's own CUDA
headers and libraries (its pip packages in `nvidia/cu13/`: the CUDA 13.0 runtime torch was built with) and a
compiler of the same CUDA version from pip (`nvidia-cuda-nvcc`, `-nvvm`, `-crt`, `-cccl` 13.0, about 130 MB in a
folder of their own; `scripts/cuda13_toolkit.sh` joins the two as a toolkit folder of links and checks the versions
agree), because the wheel's do not load beside vLLM 0.32's
torch 2.13 and LMCache's torch fallback crashed the server (job 30694, below). The engines and the server run with
`PYTHONPATH` on that tree, so the only code that differs from the release is the hooks and that build of the
release's own kernel sources. `scripts/lmc_overlay.py verify` (run by the setup) checks that each hooked module,
kvtrace and the CUDA ops are imported from the tree, and stops otherwise.

### Where the hooks are

| File | Function | Line | Event |
| --- | --- | ---: | --- |
| `vllm/v1/core/sched/scheduler.py` | `add_request` | 2636 | `vllm.request.added` |
| `vllm/v1/core/sched/scheduler.py` | `_free_request` | 2743 | `vllm.request.finished` |
| `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` | 322 | `vllm.gpu.prefix_hit` |
| `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 604 | `vllm.gpu.allocated` |
| `vllm/v1/core/kv_cache_manager.py` | `free` | 632 | `vllm.request.blocks_freed` |
| `vllm/v1/core/block_pool.py` | `cache_full_blocks` | 304 | `vllm.gpu.cached` |
| `vllm/v1/core/block_pool.py` | `_maybe_evict_cached_block` | 751 | `vllm.gpu.evicted` |
| `vllm/v1/core/block_pool.py` | `reset_prefix_cache` | 865 | `vllm.gpu.reset` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `get_num_new_matched_tokens` | 1449, 1488 | `lmc.conn.lookup` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `update_state_after_alloc` | 1553 | `lmc.conn.alloc` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `_process_retrieve_requests` | 1834 | `lmc.conn.retrieve_planned` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `_process_new_requests`, `_process_cached_requests` | 1864, 1908 | `lmc.conn.store_planned` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `start_load_kv` | 968 | `lmc.conn.retrieve_submitted` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `wait_for_save` | 1050 | `lmc.conn.store_submitted` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `get_finished` | 1110 | `lmc.conn.finished` |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | `request_finished` | 1707 | `lmc.conn.request_finished` |
| `lmcache/v1/mp_observability/event_bus.py` | `publish` | 223 | `lmc.<event>` (every event LMCache publishes) |
| `lmcache/v1/mp_observability/event_bus.py` | `publish_on_stream` | 195 | `lmc.<event>`, `via: stream` |
| `lmcache/v1/mp_observability/event_bus.py` | `_drain_all` | 348 | `lmc.<event>`, `via: native` |
| `lmcache/v1/mp_observability/event_bus.py` | `has_subscribers` | 183 | none: says yes for `mp.lookup`, `mp.tokens` and `l1.keys.accessed`, which LMCache builds only when someone listens |
| `lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py` | `_execute_load` | 823 | `lmc.l2.read` |
| `lmcache/v1/distributed/storage_manager.py` | `touch_cached_keys` | 648 | `lmc.touch` |

Plus `import kvtrace  # KVTRACE` at the top of each of the seven files: 31 lines in all. (The last two came after
job 30728: LMCache publishes `l2.keys.accessed` both for a real read from disk and for a touch, which only refreshes
the eviction order and moves no bytes, so only these two hooks tell them apart.) Line numbers are in our
copies (`third_party/<copy>/...`).

The event-bus hooks see LMCache's own observability events (`lmcache/v1/mp_observability/event.py`), in the server
and in the engines' connectors, before LMCache's own enabled check and queue (so none is lost to a full queue).
LMCache keeps its own subscribers (metrics, logging) as they were.

## The records

Each process writes `KVTRACE_DIR/kvtrace_<role>_<pid>.jsonl` (a forked child its own file), buffered and flushed
every second and at exit. Every record has:

| Field | Meaning |
| --- | --- |
| `ev` | the event name (below) |
| `t` | wall-clock time, `time.time()` seconds: one clock for every process on the node |
| `role` | `P`, `D1`, `D2`, `D3` (each engine's API server and engine core), `LMC` (the server) |
| `pid` | the process |

Hashes and keys are written as 16 hex characters: the first 8 bytes of vLLM's block hash (sha256; the 4-byte group
id dropped), or of LMCache's chunk hash. Requests are vLLM's request ids, `cmpl-<key>-0-<random>`, where `<key>` is
the proxy's 32-hex key for the turn, the same for P and D (the proxy sends it as `X-Request-Id`).

### vLLM (scheduler and KV cache manager, in each engine's core process)

| Event | When | Fields |
| --- | --- | --- |
| `vllm.request.added` | a request enters the scheduler | `req`; `n_prompt` tokens; `hashes`, the block hash of each full 16-token block of the prompt, in order (block i holds tokens 16i..16i+15); `n_hashes`; `kvp`, the keys of its `kv_transfer_params` |
| `vllm.gpu.prefix_hit` | the engine's own prefix cache is searched | `req`; `tokens` found in its GPU cache; `blocks`, their GPU block ids |
| `vllm.gpu.allocated` | GPU blocks are given to a request | `req`; `new` tokens to compute; `local_hit`; `external` tokens to come from LMCache; `delay_cache` (true while a load is pending: the blocks are cached when it lands); `blocks`, the new GPU block ids; `n_computed` |
| `vllm.gpu.cached` | full blocks enter the prefix cache (their hash now finds them): the block's KV is complete on this GPU, computed or loaded | `req`; `index`, each block's place in the request; `blocks`, its GPU block id; `hashes`; `first`; `group`; `block_size` |
| `vllm.gpu.evicted` | a cached block's hash is dropped so the block can be reused | `block` (GPU id); `hashes` |
| `vllm.gpu.reset` | the prefix cache emptied | none |
| `vllm.request.blocks_freed` | a finished request's blocks go back to the free queue (they stay cached until reused) | `req`; `blocks` |
| `vllm.request.finished` | the request leaves the scheduler | `req`; `status`; `n_out`; `n_tokens`; `delay_free` (true while a store is still in flight) |

### LMCache's connector inside each engine

| Event | When | Fields |
| --- | --- | --- |
| `lmc.conn.lookup` | the connector learns how much of the prompt LMCache holds | `req`; `vllm_hit` (tokens on the GPU already); `lmc_hit` (tokens LMCache holds, a multiple of 256); `load` (tokens it will load); `n_tokens` |
| `lmc.conn.alloc` | vLLM gave the request its blocks | `req`; `new_blocks`; `external`; `state` (the connector's state for it) |
| `lmc.conn.retrieve_planned` | the scheduler plans a load for this step | `req`; `start`, `end` (tokens; `start` is the GPU hit rounded down to a chunk); `skip` (tokens at the start already on the GPU, not written); `blocks`, the GPU block ids for tokens `start..end` (block k holds tokens `start + 16k ...`); `n_blocks` |
| `lmc.conn.store_planned` | the scheduler plans a store for this step | the same fields, for the chunks to store (whole chunks of the prompt not stored yet) |
| `lmc.conn.retrieve_submitted` | the worker sends the step's loads to the server | `reqs` |
| `lmc.conn.store_submitted` | the worker sends the step's stores to the server (after the step's forward pass) | `reqs` |
| `lmc.conn.finished` | the worker reports finished transfers | `sending` (stores done), `receiving` (loads done): request ids |
| `lmc.conn.request_finished` | the connector's last word on a request | `req`; `vllm_hit`; `lmc_hit`; `stored` tokens; `delay_free` |

### LMCache's server (its event bus)

Every event LMCache publishes is recorded as `lmc.<its name>` with `session` (the vLLM request id, for request
events) and its metadata, keys as hashes (`keys`, with `n_keys`), token lists as their length and a CRC-32 per
chunk. Left out: `trace.call`, `mp.transfer.phase_samples`, `sm.capacity.changed`, `l1.eviction.loop_tick`. Events
LMCache schedules on a CUDA stream (the store's and retrieve's start and end) are seen when queued
(`via: stream`) and again when the GPU reached them (`via: native`, from LMCache's native recorder, or published
again by the host callback); the report keeps the second, with `t_queued`. A `native` record's `t` is the
recorder's time and `t_hook` when the bus drained it.

The ones the report reads:

| Event | Meaning | Fields |
| --- | --- | --- |
| `lmc.mp.tokens` | a store bound chunks to their tokens | `session` (the storing request); `chunk_hashes`; `token_offsets` (each chunk's first token in the prompt); `chunk_tokens`; `chunk_crc` (CRC-32 of the chunk's token ids as int32); `parent_hashes` (the chunk before it) |
| `lmc.mp.store.submitted`, `.start`, `.end` | a store's arrival, start and end on the server | `session`; `transfer_key`; at the end `stored_count`, `total_bytes`, `num_tokens`, `engine_id` |
| `lmc.l1.write.reserved`, `lmc.l1.write.finished` | memory reserved for chunks, then written | `keys`; `sizes` (bytes, when finished) |
| `lmc.l2.store.submitted`, `.completed` | a write to disk submitted and done | `task_id`, `key_count`, `total_bytes`; `succeeded_count`, `failed_count` |
| `lmc.l2.keys.stored` | chunks on disk | `keys`, `sizes`, `backend` |
| `lmc.l1.keys.evicted` | chunks removed from memory: evicted (LRU), or, in the disk mode, deleted once on disk | `keys`, `sizes` |
| `lmc.mp.lookup` | an engine's lookup of a prompt | `session`; `chunk_hashes` (every whole chunk of the prompt, in order); `seq_len`; `n_token_ids` |
| `lmc.mp.lookup_prefetch.start`, `.end` | the lookup and its prefetch: at the end, how many chunks were found and where | `found_count` (chunks from the start); `hit_tokens`, `l1_hit_tokens`, `l2_hit_tokens`; `requested_tokens`; `engine_hit_tokens` |
| `lmc.l2.prefetch.lookup.submitted`, `lmc.l2.prefetch.load.submitted`, `.completed`, `lmc.l2.load_task.*` | the disk part of a prefetch | `request_id`, `key_count`, `loaded_count`, `failed_count` |
| `lmc.l2.keys.accessed` | chunks read from disk, or touched (eviction order refreshed, no I/O): LMCache publishes both alike | `keys` |
| `lmc.l2.read` (our hook) | the file adapter read these chunks from disk | `keys`, `n_keys`, `task_id`, `requested` |
| `lmc.touch` (our hook) | the storage manager refreshed these chunks' eviction order in memory and on disk, no I/O | `keys`, `n_keys` |
| `lmc.sm.read.prefetched_finished` | prefetched chunks ready in memory | `succeeded_keys`, `failed_keys` |
| `lmc.l1.read.reserved`, `.finished` | chunks read from memory (for a copy to a GPU) | `keys` |
| `lmc.mp.retrieve.submitted`, `.start`, `.end` | the copy to the engine's GPU | `session`; `transfer_key`; at the end `retrieved_count`, `total_bytes`, `num_tokens` |
| `lmc.l1.keys.accessed` | chunks touched in memory (LRU) | `keys` |
| `lmc.mp.request.start`, `.end`, `lmc.mp.vllm.end_session` | the server's view of a request's life | `session` |
| `lmc.mp.vllm.block_allocation` | an engine's connector told the server of vLLM's new blocks (LMCache's own view, a cross-check of `vllm.gpu.allocated`) | `instance_id`; `records`: per request `req`, `blocks`, `n_tokens` |

## How the report joins them (`scripts/lmc_lifecycle.py`)

The keys that tie everything together:

| From | To | By |
| --- | --- | --- |
| a turn of MTRAG (`mtrag.jsonl`) | the proxy's request (`proxy.jsonl`) | the turn's tag |
| the proxy's request | P's and D's vLLM requests | the 32-hex key inside the vLLM request id |
| a vLLM request | the server's records of it | the server's `session` is the vLLM request id |
| a stored chunk | its tokens, its place in the prompt and its MTRAG pieces (system, document, question, answer) | `mp.tokens` (`token_offsets`, `chunk_crc`) and the turn's prompt pieces |
| a stored chunk | P's GPU blocks it was made from | the storing request's block hashes for tokens `offset .. offset+256`, and P's `vllm.gpu.cached` of those hashes |
| a stored chunk | memory and disk | its chunk hash in `keys` of the `l1.*` and `l2.*` events |
| a D lookup | the chunks it asked for, found or missed | `mp.lookup`'s `chunk_hashes` (chunk j is tokens `256j ..`) and `mp.lookup_prefetch.end`'s `found_count` |
| a pulled chunk | D's GPU blocks it went into | `lmc.conn.retrieve_planned`'s `blocks`, and D's `vllm.gpu.cached` of the same block indexes |
| a pulled chunk | whether it came from disk | `lmc.l2.read` for its key between D's lookup and the retrieve's end (traces before that hook: `l2.keys.accessed`) |

What it reports for each chunk (`chunks.jsonl.gz`): its hash, conversation, turn, tokens, pieces, bytes; P's blocks
(index, GPU block, hash, when cached, which request computed it); when the store was planned and submitted; when it
was bound, reserved and written in memory, written to disk, removed from memory, read from disk, deleted; and every
lookup of it: which engine and turn, the outcome (`pulled`, `held (on GPU already)`, `hit, not loaded`,
`miss: not yet stored`, `miss: evicted`, `miss: never stored`), the tier it was pulled from, when, and D's blocks
(index, GPU block, whether written, its hash and whether it equals P's).

For each engine request (`requests.jsonl.gz`): prompt tokens, its own GPU hit, LMCache's hit, tokens loaded and
computed, its chunks by outcome and tier, the retrieve's range and bytes, its stores, and the times (added, lookup,
found, retrieve start and end, load seen by the worker, store planned and ended, finished, blocks freed; the proxy's
arrival, P's answer and D's first byte), and three durations: `prefetch_s` (the server's lookup to its prefetch
done: where a chunk on disk is read into memory), `retrieve_s` (the copy to the GPU) and `pull_s` (the server's
lookup to the copy done). For D: whether its lookup finished before P's store of the same turn had ended, and by
how much.

For each engine request and prompt block (`blocks.jsonl.gz`): the block's hash, its chunk, where it came from
(`local prefix cache`, `LMCache`, `computed`), its GPU block, when cached and when evicted.

`results.json` has the totals and medians; `summary.md` says what happened, then tells one chunk's whole life and
one D request's, as timelines. `conversations/` has one page per conversation, from its prompt's pieces to its
chunks and blocks, turn by turn (`docs/lmcache_views/`: job 30728's, both modes): each turn's prompt as text (turn 0
whole; on later turns only what is new after the previous prompt, which is checked to be its prefix id for id), how
the prompt grows (the previous answer, the documents and how many were sent before, the question, the framing, the
whole chunks), then its chunks. The text is the driver's record of what it sent (`texts` in `mtrag.jsonl`, with
`--trace`, the answers too), or for an older run, rebuilt by the driver's `build()` from the MTRAG file
(`--conversations`), every piece but the answers, and only where it comes out the run's pieces kind for kind and id
for id. Checks it makes: every chunk's tokens against the prompt (CRC), every pulled block's
hash against P's, every stored chunk with its whole chain (computed on P, in memory). `compare.md` puts the modes
side by side.

## Options

`scripts/lmc_run.sh` (environment): `LMC_MODE` (memory, disk, tiered), `LMC_L1_GB`, `LMC_CHUNK` (256),
`LMC_ODIRECT` (1), `LMC_WORKERS` (4), `LMC_SAVE_DECODE` (0), `LMC_D_ROLE` (kv_consumer; kv_both lets D store
what it computes), `MTRAG_LIMIT` (4), `MTRAG_LAYOUT` (interleaved), `MTRAG_CONCURRENCY` (4), `MAX_TOKENS` (128),
`STORE_WAIT_MS` (500; 0 shows how often D looks up before P's store has landed), `GPU_MEM` (0.80), `GPU_BLOCKS`
(a small number shrinks every engine's GPU cache, so blocks are evicted from it and come back from LMCache),
`D_PREFIX_CACHE` (1; 0 makes every turn pull its whole prefix), `KEEP_L2` (0), `ENGINE_EAGER` (0).
`scripts/lmc_setup.sh`: `LMC_TORCH_BASELINE=1` skips the CUDA ops build (the wheel's ops, or LMCache's torch
fallback, as job 30694 ran: not for real runs), `CUDA13_HOME` (a toolkit already there, used only if the environment
has no CUDA 13 compiler of its own; never downloaded by this setup).

The matrix `lmctrace` (`scripts/pd_matrix.sh`) runs one per mode; set before the campaign starts:
`KVPD_LMC_MODES` (memory,disk), `KVPD_LMC_CONVS` (4), `KVPD_LMC_STORE_WAIT_MS` (500). Its setup is
`scripts/lmc_setup.sh` only (the LMCache environment, LMCache's CUDA ops, the overlay, MTRAG, the model): no codec
builds, no toolkit download. The run stage defaults to 3 hours. The disk plan reserves 3 GB for the setup, 1.5 GB a
conversation for the disk tier (job 30694's first 4 conversations: 3.4 GB in all) and 0.25 GB a run for its records,
and keeps 10 GB free; each run deletes its disk tier after its report, keeping the file list in `server/l2_files.tsv`.

To report again on a published run (no GPU, standard library only), from a checkout with the results:

```bash
python3 -I scripts/lmc_lifecycle.py report \
    --results benchmark_results/pd_card/<tag>/matrix/llama32_3b_lmc_<mode>_rtxa5000_<time> --output /tmp/re_<mode>
python3 -I scripts/lmc_lifecycle.py compare --dirs /tmp/re_memory,/tmp/re_disk --output /tmp/re_compare
```

## Runbook (gpu-grad-01, the card)

The same card as every campaign (`docs/pd_baselines.md`, "On gpu-grad-01: the card"): the login host only checks
and submits; the run stage builds everything itself in its allocation (the LMCache environment from the shared uv
cache, our overlay, MTRAG, the model), runs both modes, and the results go to their own branch. From the login
host, inside tmux:

```bash
module load slurm
bash <<'LOGIN'
set -Eeuo pipefail
umask 077
trap 'printf "PD CAMPAIGN STOPPED: line=%s exit=%s; the campaign and its output are kept\n" "$LINENO" "$?" >&2' ERR
test -z "${SLURM_JOB_ID:-}"
cd ~/kv_reconstruction
test -z "$(git status --porcelain --untracked-files=no)"
git fetch -q --no-tags origin lmcache-lifecycle
git switch -q lmcache-lifecycle 2>/dev/null || git switch -q -c lmcache-lifecycle --track origin/lmcache-lifecycle
git merge -q --ff-only origin/lmcache-lifecycle
export KVPD_BRANCH=lmcache-lifecycle
export KVPD_PIN=$(git rev-parse HEAD)
test "$(git ls-remote origin refs/heads/lmcache-lifecycle | awk '{print $1}')" = "$KVPD_PIN"
export KVPD_MATRIX=lmctrace
export KVPD_TAG="pd_${KVPD_MATRIX}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p ~/.kvpd_campaigns
STATE=~/.kvpd_campaigns/$KVPD_TAG.sh
printf 'export KVPD_BRANCH=%q\nexport KVPD_PIN=%q\nexport KVPD_TAG=%q\n' "$KVPD_BRANCH" "$KVPD_PIN" "$KVPD_TAG" > "$STATE"
printf 'PIN=%s\nTAG=%s\nSTATE=%s\n' "$KVPD_PIN" "$KVPD_TAG" "$STATE"
for STAGE in create run cleanup verify-cleanup review publish; do
    bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG"
done
echo "Campaign $KVPD_TAG published (the PUBLISHED line above names its branch)."
LOGIN
```

Options go before `bash <<'LOGIN'`, exported: for example `export KVPD_LMC_MODES=memory,disk,tiered` or
`export KVPD_LMC_CONVS=8`. To resume after a stop: `source ~/.kvpd_campaigns/<tag>.sh`, `git switch -q --detach
"$KVPD_PIN"`, then the remaining stages as in `docs/pd_baselines.md`.

The published branch is `evidence/lmcache-lifecycle/<pin12>/lmctrace/job<run job>`. In it,
`benchmark_results/pd_card/<tag>/matrix/` has one folder per mode (`llama32_3b_lmc_<mode>_rtxa5000_<time>/`:
`summary.md`, `results.json`, `chunks.jsonl.gz`, `requests.jsonl.gz`, `blocks.jsonl.gz`, `kvtrace.tar.gz` with every
raw record, `proxy.jsonl.gz`, `mtrag.jsonl.gz`, `env.json`, `server/` snapshots of the server's status, config,
objects and metrics, `logs/` with each process's log tail) and `pd_matrix_lmctrace_*/compare.md`.

## Runs

**Job 30688** (pin `ef21c33`): stopped in setup, before any engine: the import check ran from the checkout and
picked up its `kvtrace/` instead of the overlay's. Fixed in `29a277b` (`verify`, run with `python -P` from a folder
of its own).

**Job 30694** (pin `29a277b`; `benchmark_results/pd_card/pd_lmctrace_20261125T203203Z/`): both modes ran, and in both
the LMCache server crashed about 3 seconds into the workload with a segmentation fault, with the wheel's compiled
CUDA ops not loaded. The memory run's log shows where: LMCache's torch fallback (`torch_ops/mp_mem_kernels.py`,
`_transfer_per_layer_fused`), copying a chunk from server memory to an engine's GPU; the disk run's kept tail ends
before the crashing thread (runs now keep every log whole, gzipped). The engines gave up on the server 20 seconds in ("degraded
mode") and served the rest without it: 44 of the 60 engine requests ran without LMCache. What was traced before
the crash is whole: 4 stores (10 chunks, 293.6 MB), 4 pulls (all 10 chunks: from memory in the memory mode, from
disk in the disk mode, 40.7 ms median for the server's lookup and read from disk, 66.3 ms to the copy on D's GPU
done), every chunk's tokens equal to the prompt's, and all 160 pulled blocks with the same vLLM block hash on D as
on P. Fixed in the next commit: LMCache's CUDA ops built from our copy, and the setup stops unless they load.

**Job 30700** (pin `96f5833`): stopped before setup by the disk check: 18.4 GB free, and the plan asked for 23.4 GB
plus 10 GB to spare, most of it for a CUDA 13 toolkit download and the codec setup's peak. Neither is needed: the
environment's own pip packages carry a CUDA 13 compiler, and this matrix's setup is small; the plan now asks for
about 9.5 GB.

**Job 30716** (pin `f0dac4a`): the CUDA ops did not build. The environment's own compiler is CUDA 13.5
(`nvidia-cuda-nvcc 13.5.37`, pulled in by another package) and its runtime headers CUDA 13.0 (`nvidia-cuda-runtime
13.0.96`, torch's), and CCCL refuses that pair at the first `#include`: "CUDA compiler and CUDA toolkit headers are
incompatible". The setup now installs the compiler of the headers' version (13.0) into a folder of its own, and
the toolkit script stops early if the two ever disagree.

**Job 30722** (pin `4f9fc89`; `benchmark_results/pd_card/pd_lmctrace_20261125T220106Z/`): the CUDA ops built and
loaded, both modes ran to the end, and the server stayed up, but every store failed: "Cannot store keys due to
exception: DirectCopyGroupSpec(): incompatible constructor arguments" (30 in each mode). The descriptor a store
passes (`PageBufferShapeDesc`) is a type of `lmcache_native`, still the wheel's, and our `cuda_ops`, built with torch
2.13's pybind11, does not recognise the wheel's types. So nothing reached the server's memory or disk, every D
lookup missed, and D computed what it did not hold itself (26,305 tokens; 101,648 from its own GPU cache). What the
trace did show: P's stores bound 114 chunks to their tokens (all 114 equal to the prompt's), 758 lookups found the
chunk on the engine's GPU already. Fixed in the next commit: all four of LMCache's compiled modules built from our
copy, the setup checks that `cuda_ops` takes `lmcache_native`'s descriptor, and a run whose server logs errors
fails, with the errors at the top of its report.

**Job 30728** (pin `f159ef6`; `benchmark_results/pd_card/pd_lmctrace_20261125T221525Z/`): the first whole run. All
four of LMCache's compiled modules ours, the server up and without errors in both modes, and every chunk's life
traced end to end:

| | memory | disk |
| --- | ---: | ---: |
| chunks P's stores wrote to the server (all bound to their tokens, all equal to the prompt's) | 114 (3.35 GB) | 114, then to disk and out of memory |
| pulled into a D engine | 102, from memory | 102, from disk |
| never pulled | 12 | 12 |
| D's tokens: from its own GPU / loaded from LMCache / computed | 101,632 / 23,040 / 3,281 | the same |
| pulled blocks with the same vLLM hash on D as on P | 1,632 of 1,632 | 1,632 of 1,632 |
| a pull (median: 4 chunks, 117 MB): the server's lookup and prefetch | 2.2 ms | 47 ms (the disk read, about 2 GB/s) |
| then the copy to D's GPU | 5.1 ms (about 22 GB/s) | 4.8 ms |
| the server's lookup to the copy done | 13.9 ms | 70.9 ms (p90 97.7) |
| D's request taken to its first token at the proxy | 50 ms | 115 ms |
| P's stores, planned to stored (median) | 129 ms | 116 ms; memory to disk 188 ms more |
| D lookups that came before P's store had landed | 0 | 0 |

What it shows. In the disk mode every chunk P stores is written to disk and dropped from memory within about a
third of a second, and every pull reads it back from disk; the disk read costs about 45 ms a pull here, the copy
to the GPU the same 5 ms as from memory. D's own GPU cache serves most of every prompt (the conversation's earlier
turns), and LMCache supplies what is new this turn: 23,040 tokens, nearly all of what D did not hold (D computed
3,281: each turn's last partial chunk). The 12 chunks never pulled are all ones that hold an earlier answer: D
generated that answer itself and holds its blocks, while P never saw it and had to prefill it, and stored the
chunk. The 500 ms wait was always enough: no lookup came before its store had landed. The chunk timelines of this
run show "disk tier accessed" at every later lookup of a chunk D already held; most of those are touches (LMCache
refreshing eviction order, no I/O), which this trace could not tell from reads; the two hooks added after it do.

## What to look out for

- **The store can still be in flight when D looks up.** P answers once its single token is out; its store is
  submitted in the same step and lands on the server a little later. The proxy waits `STORE_WAIT_MS` for it; the
  report counts the D lookups that finished before P's store ended, and the chunks they missed (`miss: not yet
  stored`).
- **D holds earlier turns itself.** A conversation stays on one D engine, so D's own GPU prefix cache serves the
  turns before; D pulls only what is new (the chunks from the end of what it holds). `held (on GPU already)` counts
  those. `D_PREFIX_CACHE=0` or a small `GPU_BLOCKS` makes D pull more.
- **Only whole chunks are stored.** The prompt's last partial chunk is computed by D; it is stored by P on the
  next turn, when the prompt has grown past it.
- **One clock.** Times compare across processes because they are all `time.time()` on one node.
- **LMCache's compiled CUDA ops.** The release wheel's `lmcache.cuda_ops` does not load beside vLLM 0.32's torch
  2.13 (`undefined symbol: c10::cuda::CUDAErrorLogCapture`: it was built against another torch). LMCache then falls
  back to plain torch, and that crashed the server in job 30694. The setup builds them from our copy instead and
  stops unless they load; `lmc_verify.json` (with the setup's records, and in each run's `server/`) says which ops
  ran, and `lmc_ops_build.json` what they were built from. With them, the store's and retrieve's start and end are
  timed when the GPU reached them (LMCache's CUDA event recorder); without, when queued.
- **The server's health.** If the LMCache server stops, the engines' connectors say "entering degraded mode" and
  go on without it; if a store or load fails, the server logs an error and goes on. Either way the run fails, and
  the report opens by saying so: when the server stopped, when each engine gave up on it and how many requests
  came after, or how many errors the server logged, of which kinds.
- **The setup's import check** runs with `python -P` from a folder of its own: run from the checkout, the
  checkout's own `kvtrace/` came first on the path and the check stopped (job 30688). The engines and the server
  run in the run's folder for the same reason.
- **Hashes are cut to 8 bytes** in the records; for a few thousand chunks a clash is not a concern.
