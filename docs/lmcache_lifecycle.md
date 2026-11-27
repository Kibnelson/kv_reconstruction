# LMCache lifecycle: every block and chunk, from P's GPU to LMCache and to D's GPU

Branch `lmcache-lifecycle` (from `pd-blend`). A clean 1 prefill + 3 decode set-up in which **P and the three D
engines share one LMCache server**. P computes each prompt and its connector stores the prompt's chunks in the
server; the turn's D engine looks the prompt up there and pulls what it does not already hold. Hooks in our own
copies of vLLM and LMCache record every step of every block's and chunk's life, and a report puts them back
together. It runs in two modes: the chunks in the server's memory, or every chunk on disk and read back from disk.
It can also run as LMCache's own P/D guide has it: P's KV reaches D over NIXL, with LMCache beside it, on one shared
server or a server per engine ("Transfer and servers", below); the table describes the default, the store path.

| | |
| --- | --- |
| Model | Llama 3.2 3B (`unsloth/Llama-3.2-3B` at `d4446454`), bf16, 16-token vLLM blocks |
| Engines | vLLM 0.32.0, P on GPU 0 (`kv_both`), D1-D3 on GPUs 1-3 (`kv_consumer`: they only pull) |
| Cache | LMCache 0.5.7.post1 in multi-process mode: one `lmcache server` that sees all four GPUs; each engine's `LMCacheMPConnector` talks to it over ZMQ and the server copies KV to and from each engine's GPU memory over CUDA IPC |
| Chunk | 256 tokens (16 vLLM blocks); LMCache names it by a hash of its tokens and everything before them (blake3) |
| Workload | MTRAG's first 4 conversations, interleaved layout (each turn's prompt is the previous prompt, its answer, the new passages and question); routed by conversation, so a conversation's turns go to one D engine |
| Proxy | store mode: P gets the prompt with one token to generate (its connector stores the chunks), then after `STORE_WAIT_MS` (500 ms) the D engine gets the full request |

## Transfer and servers

Two settings choose how P's KV reaches D and how many LMCache servers there are (`scripts/lmc_run.sh`; for a
campaign `KVPD_LMC_TRANSFERS` and `KVPD_LMC_SERVERS`, every combination with every mode in `KVPD_LMC_MODES`):

| | `LMC_SERVERS=shared` (default): one server for P, D1, D2, D3 | `LMC_SERVERS=per_engine`: a server per engine, as LMCache's guide |
| --- | --- | --- |
| `LMC_TRANSFER=lmcache` (default) | the store path: P stores its chunks, D loads them from the server; the proxy waits `STORE_WAIT_MS`; each conversation on one D | not run: D could not load what P stored on its own server (the run stops, the campaign skips it) |
| `LMC_TRANSFER=nixl` | D reads P's GPU blocks over NIXL; every engine's LMCache connector stores to the one server | LMCache's P/D guide: D reads P's GPU blocks over NIXL; each engine stores to its own server |

**`nixl`, as LMCache's guide sets it up** ([Disaggregated Prefill, MP mode](https://docs.lmcache.ai/mp/disaggregated_prefill.html)):

- Engines: vLLM's `MultiConnector` with two connectors, in this order: `NixlConnector` (P `kv_producer`, D
  `kv_consumer`, `kv_load_failure_policy: fail`) for the handoff, and LMCache's `LMCacheMPConnector` (`kv_role:
  kv_both` on both) to store and load. `MultiConnector` gives a request's load to the first connector that has
  tokens for it (vLLM 0.32, `multi_connector.py`), so on D NIXL loads the prompt and LMCache's lookup loads nothing
  (it is still asked, and D waits for its answer); every connector stores. Each engine's NIXL side channel has its
  own port (`VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`), and each engine sees its own GPU only, as in the guide and vLLM's
  NIXL tests.
- The proxy, as `vllm-router --vllm-pd-disaggregation` does it for NIXL's pull mode (read in its source:
  `src/routers/http/vllm_pd_router.rs` at `c132dbc0cfdf` (2026-11-26)): P gets the request with `max_tokens` and
  `min_tokens` 1 (`max_completion_tokens` 1 if set), `stream` off and no `stream_options`, and `kv_transfer_params`
  `{do_remote_decode: true, do_remote_prefill: false, remote_engine_id, remote_block_ids, remote_host, remote_port:
  null}`; P's reply's `kv_transfer_params` (its blocks, its NIXL agent) go to D as they are, with the original
  request, as soon as P answers: no wait. D engines round-robin. This is `scripts/pd_proxy.py --mode nixl --policy
  round_robin`, which also keeps the per-turn times the timelines are built from (`vllm-router` keeps none).
- Servers: `per_engine` starts four, `LMC_P` and `LMC_D1`-`LMC_D3` (`--instance-id` so named), each seeing its
  engine's GPU and holding `LMC_L1_GB` (16 GB in memory mode, so 64 GB in all); the run checks the node's free memory
  first (`LMC_HOST_MARGIN_GB`, 16, for the engines) and stops if they do not fit. Each has its own disk tier folder
  and its snapshots in `server/<role>/`.
- What the trace shows: P's request holding its blocks for D (`nixl.request_finished`), D's match and its blocks
  (`nixl.match`, `nixl.recv_planned`), the handshake (once per pair), D's read of P's blocks (`nixl.read`), D having
  the KV and P freeing its blocks (`nixl.done`), and every engine's LMCache stores, on its server.
- What to expect: D never loads from LMCache (NIXL has the tokens first). Its lookup still runs, and in the disk mode
  the server's lookup reads what it finds from disk into memory (its prefetch) even though nothing loads it; P loads
  from LMCache only what its own GPU cache no longer holds (`GPU_BLOCKS` makes that happen). With a server per engine each server keeps its own copy, and with D engines
  round-robin one conversation's chunks can end up on all three D servers.

### NIXL's transport

NIXL's default backend is UCX ([vLLM's NIXL guide](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)),
and UCX picks its transports itself (`UCX_TLS` unset means all). Between two GPUs of one node it should take
`cuda_ipc`, a peer-to-peer copy over PCIe or NVLink: vLLM's guide says KV reads "should show `cuda_ipc`, not `tcp`".
When `cuda_ipc` cannot open the peer GPU's memory and there is no RDMA card, UCX falls back to software emulation
over TCP: GPU to host, TCP, host to GPU, at about 0.56 GB/s
([vLLM issue 62080](https://github.com/vllm-project/vllm/issues/62080), on GPUs without peer access). The requests
still succeed, and nothing warns at the default log level.

Job 30763 took that path. UCX's protocol tables in every engine's log name `tcp/bond0`, `cuda_copy` and `cma`,
and vLLM's own transfer metrics give 540-580 MB/s for every read (0.59 GB/s in the report), though the node's GPUs
report peer reads and writes ("OK" in `nvidia-smi topo -p2p`).

**Why (job 30774, UCX's log at debug).** UCX does set up `cuda_ipc` between the engines (`device(cuda_ipc/cuda)` in
its lanes), but it turns `cuda_ipc` reads off when it finds no NVLink: `cuda-ipc get zcopy disabled as no nvlinks
detected`. These A5000s are joined by PCIe only. NIXL's pull mode is a read (`ucp_get`), so its table says `remote
memory read by ucp_get*(multi) into cuda/GPU0 from cuda/dev[4]`: `software emulation | tcp/bond0`. UCX's setting
for it is `UCX_CUDA_IPC_ENABLE_GET_ZCOPY` (`auto` by default: "Enable get operations except for platforms known to
have slower performance", [UCX's cuda_ipc
interface](https://github.com/openucx/ucx/blob/master/src/uct/cuda/cuda_ipc/cuda_ipc_iface.c)); `on` allows the reads
over PCIe. The 0.56 GB/s in vLLM issue 62080 is the same fallback on GPUs with no peer access at all, where `on` does
not help; here the GPUs have it.

The report says how D read P's memory (`nixl.read_path`, from that table when the log has it), which transports
UCX used (`nixl.ucx_transports`), how fast the reads were (`nixl.read_gb_s_median`), whether they were GPU to GPU
(`nixl.gpu_to_gpu`) and UCX's CUDA IPC messages. `export KVPD_LMC_PRESET=nixlcheck` runs the first conversation,
memory, shared server, carry-forward, with `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on` and UCX's log at `debug`: once with each
engine seeing its own GPU and once with every GPU visible (`NIXL_ALL_GPUS=1`). Its runs compare with job 30763's
`memory_nixl_shared_carry_all` run and job 30774's (`auto`). Switches: `KVPD_LMC_UCX_GET_ZCOPY` (`auto`, `on`, `off`,
by commas), `KVPD_LMC_ALL_GPUS` (0 and/or 1), `KVPD_LMC_UCX_TLS`, `KVPD_LMC_UCX_LOG`; for one run
`NIXL_UCX_GET_ZCOPY`, `NIXL_ALL_GPUS`, `NIXL_UCX_TLS`, `NIXL_UCX_LOG`. (Job 30774's every-GPU run did not get every
GPU: the run script's switch was overwritten by its own list of the GPUs; fixed.)

**Do the reads deliver the right KV?** The transfer layer says yes in every run: every read done, none failed,
P freed its blocks after each, no UCX error in any log (the tracebacks in some engine logs come at shutdown, after
the stop signal). The answers do not agree, though. The first turn's prompt is the same 515 tokens in jobs 30763,
30774 and 30780, and D answered it three ways: "I don't know. I don't know what you mean by "in the air"..."
(30763, TCP), " I am a helpful assistant. Answer the user's last question from the documents..." (30774, TCP, UCX
logging at debug) and " other other other..." (30780, cuda_ipc reads). Within each job every run answered the
same. Job 30786 (the eight, 1 conversation, UCX's defaults, `eea6fcf`): all eight runs gave the same answers, and 3 of 5
(carry-forward) or 4 of 5 (chat) were one plain engine's token for token. Turn 0 parts from it at the second token
(" I don't know..." for " I am a helpful assistant..."), carry-forward's turn 1 at the tenth; turns 3 and 4, whose
prefix D1 and D2 took from their own GPU caches of blocks NIXL brought on turns 0 and 1, are the engine's exactly,
and so is turn 2, D3's first read. Job 30774's D gave the engine's answer to turn 0. Item 4 of "What each of the
eight must show" now checks the bytes themselves and a second engine with NIXL's KV layout.

**Yes (job 30793, the eight, 5 conversations, UCX's defaults, `133bef9`).** Every block D read is P's, bit for bit,
in every layer, checked when each read was done: 4,541 of 4,541 blocks in each carry-forward run, 2,318 of 2,318
in each chat run, all eight runs. (The published summaries say 220 and 132: the report paired each read's blocks
from the front, while vLLM drops the front of P's list, the blocks D's own cache has, and reads into P's last ones;
fixed.) The answers: 26 of 33 turns are one plain engine's token for token and 24 of 33 the NIXL-layout engine's,
and those two single engines, neither doing P/D, disagree with each other on 3 (carry-forward) and 5 (chat) of
the 33. So the answers that differ come from the computation (where the prompt's last token is computed, which KV
layout the attention reads), not from the transfer: greedy decoding flips at near ties, mostly many tokens in.

### Differences from LMCache's documented P/D, in the `nixl` runs

What the `nixl` runs do differently from the guide, kept here to change later:

| | the guide | our `nixl` runs | why |
| --- | --- | --- | --- |
| LMCache's connector class | `"kv_connector": "LMCacheMPConnector"` only, which in vLLM 0.32 is vLLM's own copy | LMCache 0.5.7's own class (`kv_connector_module_path: lmcache.integration.vllm.lmcache_mp_connector`) | matched to the server's version; our hooks are in it (the store path uses it too) |
| connector to server | `grpc://` (current LMCache) | `tcp://` (ZMQ, LMCache 0.5.7's default) | the pinned version |
| hosts | P and D on their own hosts | one node, 4 GPUs (P on the first, D1-D3 on the others) | the card's node |
| LMCache memory | `--l1-size-gb 100` | 16 GB (memory mode), 4 GB (disk mode) per server; the eight configurations: 8 GB shared, 2 GB per engine | the node's memory |
| disk tier | none | disk mode: `fs` adapter with O_DIRECT, `skip_l1`, eviction `noop`; the eight configurations: prefetch policy `retain`, eviction LRU | to see the disk path (memory mode has none) |
| `--max-workers`, `--l1-init-size-gb` | defaults (1, 20) | 4, equal to the memory size | as the store-path runs |
| proxy | `vllm-router` | `scripts/pd_proxy.py --mode nixl` with the router's requests (above); our request id (`X-Request-Id`, the same on P and D) in place of the router's `___prefill_addr_..._decode_addr_...` ids, which only other connectors read | the per-turn times |
| vLLM settings | defaults | `--gpu-memory-utilization 0.80` (0.90), `--max-model-len 32768`, `--max-num-batched-tokens 8192`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_SERVER_DEV_MODE=1`, `PYTHONHASHSEED=0` (vLLM's SHA-256 block hashes already match across engines without it), `UCX_PROTO_INFO=y` (a log line) | as the store-path runs |
| LMCache's compiled modules | the wheel's | built from our copy's sources | the wheel's do not load beside vLLM 0.32's torch (job 30694) |
| trace hooks | none | 41 lines in 10 files, recording only | the trace |
| workload | | MTRAG's first 4 conversations at once, 128-token answers; the eight configurations: the ten by id, one at a time, every passage | the lifecycle's workload |

### What a `nixl` run must show (to check it kept to the plan)

1. `env.json`: `transfer: nixl`, `servers` as asked, `routing: round_robin`, `store_wait_ms: 0`, the
   `kv_transfer_config` of P and D (`MultiConnector`: `NixlConnector` first, `LMCacheMPConnector` `kv_both`), and
   `engine_server` (all `LMC`, or `LMC_P`, `LMC_D1`...).
2. The run's log: "proxy (NIXL mode, as vllm-router: D engines round-robin, no wait)", one "LMCache server" line per
   server, and with `per_engine` the host memory check.
3. The report: every D request read its prompt from P over NIXL (`nixl_read_tokens`, `requests_reading_nixl` equal
   to the D requests), LMCache loaded nothing on D, no failed reads, one handshake per P-D pair; P's blocks freed
   after each read; D's chunks stored on its server (`stored_by`, and with `per_engine` `per_server`).
4. The pages: a conversation's turns on different D engines; D's chunks `N` (read from P over NIXL); the step by
   step showing P holding its blocks, the proxy sending D at once, D's read of P's blocks into its own, P freeing.

## The two modes

| Mode | Server memory (L1) | Disk (L2) | What a D pull reads |
| --- | --- | --- | --- |
| `memory` | 16 GB, LRU | none | memory |
| `disk` | 4 GB, a write buffer only (eviction `noop`) | `fs` adapter, one file per chunk, `O_DIRECT` (no page cache); store policy `skip_l1`: a chunk is deleted from memory as soon as it is on disk; prefetch policy `default`: a chunk read from disk is not kept in memory (`LMC_DISK_READS=retain`: it is, until evicted; eviction then LRU) | disk, every time (with `retain`: disk the first time, memory after) |
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
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` | `get_num_new_matched_tokens` | 65 | `nixl.match` |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` | `update_state_after_alloc` | 180 | `nixl.recv_planned` |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` | `request_finished` | 281 | `nixl.request_finished` |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py` | `_read_blocks` | 417 | `nixl.read` |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` | `_nixl_handshake` | 997, 1141 | `nixl.handshake` (start, done) |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` | `get_transfer_results` | 2991 | `nixl.done` |

Plus `import kvtrace  # KVTRACE` at the top of each of the ten files: 41 lines in all. (`lmc.l2.read` and `lmc.touch`
came after job 30728: LMCache publishes `l2.keys.accessed` both for a real read from disk and for a touch, which only
refreshes the eviction order and moves no bytes, so only these two hooks tell them apart. The `nixl.*` hooks came
with the `nixl` transfer: vLLM's NIXL connector, pull mode, its scheduler side in each engine's core and its worker
side.) Line numbers are in our
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
whole chunks), a timeline of every turn on one clock (P's prefill, its chunks in memory and on disk, the store wait, D's find, copy, compute and decode: `conversation_<k>_timeline.svg` and a table), each turn step by step (every move of its KV in time order), then its chunks. How the KV moves: each engine registers its GPU KV cache with the LMCache server at start (CUDA IPC handles; LMCache's default transfer mode on CUDA, LMCache-driven), and the server does every copy itself: out of P's GPU slots into its memory when P stores, out of its memory into D's GPU slots when D loads. No NIXL: LMCache uses it only between LMCache servers (P2P), and this setup has one server. The text is the driver's record of what it sent (`texts` in `mtrag.jsonl`, with
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
`D_PREFIX_CACHE` (1; 0 makes every turn pull its whole prefix), `KEEP_L2` (0), `ENGINE_EAGER` (0),
`LMC_TRANSFER` (lmcache; nixl), `LMC_SERVERS` (shared; per_engine), `LMC_HOST_MARGIN_GB` (16: the host memory kept for
the engines beside the servers'), `MTRAG_DOCS` (conversations; rag, reference, all), `MTRAG_CONV_IDS` (MTRAG ids by
commas, in run order), `LMC_DISK_READS` (temporary; retain).
`scripts/lmc_setup.sh`: `LMC_TORCH_BASELINE=1` skips the CUDA ops build (the wheel's ops, or LMCache's torch
fallback, as job 30694 ran: not for real runs), `CUDA13_HOME` (a toolkit already there, used only if the environment
has no CUDA 13 compiler of its own; never downloaded by this setup).

The matrix `lmctrace` (`scripts/pd_matrix.sh`) runs one per mode; set before the campaign starts:
`KVPD_LMC_MODES` (memory,disk), `KVPD_LMC_CONVS` (4), `KVPD_LMC_STORE_WAIT_MS` (500), `KVPD_LMC_TRANSFERS` (lmcache;
nixl; or both), `KVPD_LMC_SERVERS` (shared; per_engine; or both), `KVPD_LMC_LAYOUTS` (interleaved; history_first;
or both), `KVPD_LMC_DOCS` (conversations; rag, reference, all, by commas), `KVPD_LMC_CONV_IDS`,
`KVPD_LMC_CONCURRENCY`, `KVPD_LMC_DISK_READS`, `KVPD_LMC_L1_SHARED_GB`, `KVPD_LMC_L1_PER_ENGINE_GB`, and
`KVPD_LMC_PRESET=eight` ("The eight configurations"). Its setup is
`scripts/lmc_setup.sh` only (the LMCache environment, LMCache's CUDA ops, the overlay, MTRAG, the model): no codec
builds, no toolkit download; NIXL 1.5.2 is installed in the LMCache environment for the `nixl` runs. The run stage
defaults to 3 hours. The disk plan reserves 3.5 GB for the setup, 1 GB a conversation for the disk tier (job 30728's
first 4 conversations: 3.35 GB in all; 0.75 GB for MTRAG's task files, `rag`, `reference`, `all`: job 30763), for a
run with a server per engine 3.6 times that with the carry-forward layout (P's copy and one on each D's server, less
what they share) and 2.3 times with the chat one, for the largest run, and 0.1 GB a run for its records, and keeps 10 GB free; each run deletes its disk tier after its report, keeping the file list in `server/l2_files.tsv`.

To report again on a published run (no GPU, standard library only), from a checkout with the results:

```bash
python3 -I scripts/lmc_lifecycle.py report \
    --results benchmark_results/pd_card/<tag>/matrix/llama32_3b_lmc_<mode>_rtxa5000_<time> --output /tmp/re_<mode>
python3 -I scripts/lmc_lifecycle.py compare --dirs /tmp/re_memory,/tmp/re_disk --output /tmp/re_compare
```

## The eight configurations

Every run hands each turn over the same way: P computes the turn's prompt (one token) and holds its GPU blocks; the
proxy sends the request on at once to the next D engine (round-robin), with P's blocks named in it; D checks its own
GPU prefix cache, reads the blocks it does not have from P's GPU over NIXL, and generates the answer (128 tokens).
Every engine's LMCache connector stores the prompt's chunks on its server. What changes is where those chunks live
and what the prompts look like:

| | setting | the two values |
| --- | --- | --- |
| topology | `LMC_SERVERS` | `shared`: one LMCache server for P, D1, D2, D3, 8 GB of memory · `per_engine`: four, `LMC_P` and `LMC_D1`-`LMC_D3`, 2 GB each |
| storage | `LMC_MODE` | `memory`: chunks in the server's memory, LRU · `disk`: each chunk written to disk (`fs`, O_DIRECT) and then removed from memory (`skip_l1`); a read from disk brings it back into memory, where it stays until evicted (`LMC_DISK_READS=retain`: prefetch policy `retain`, eviction LRU) |
| prompt layout | `MTRAG_LAYOUT` | chat RAG (`history_first`): Instructions, Q1 A1, this turn's passages, Q2, cue · carry-forward (`interleaved`): Instructions, turn 1's passages, Q1 A1, turn 2's passages, Q2, cue |

2 × 2 × 2 = 8 runs. In all eight:

- Conversations: the ten below, by MTRAG id, the first `KVPD_LMC_CONVS` of them (1, 5 or 10: three campaigns),
  one conversation at a time (`MTRAG_CONCURRENCY=1`), in this order.
- Passages (`MTRAG_DOCS=all`): each turn's passages marked relevant (`reference.jsonl`) and then the retriever's top
  5 (`RAG.jsonl`) not among them, each once: 5 to 9 a turn for these ten. `KVPD_LMC_DOCS` picks others: `rag` (the
  top 5 only), `reference` (the relevant ones only: 0 to 6, some turns none), `conversations`
  (`conversations.json`'s), or several by commas (8 more runs each). MTRAG's own `reference+RAG.jsonl` has only
  some turns of each conversation (2 of a2698b29's 9), so it is not used.
- Earlier answers: D's own (what the model generated for the turn before), as a chat system sends them.
- Routing: round-robin over D1-D3 across every request of the run, so a conversation's turns go D1, D2, D3, D1...
- Each run starts empty: new engines and servers, empty GPU caches, empty LMCache memory and disk. Within a run each
  D keeps its GPU prefix cache from one turn to the next.

| # | MTRAG conversation | turns | passages a turn (`all`) |
| --- | --- | ---: | --- |
| 1 | `1534a095279f2cb888fb0bea17bd70da` | 5 | 5,5,5,5,5 |
| 2 | `0208bf26ec357a803445290fa88a2e9e` | 7 | 5,6,5,5,6,5,5 |
| 3 | `a2698b2973ea7db1ee5adb5e70ec02e4` | 9 | 6,7,7,5,8,7,6,6,7 |
| 4 | `c1e53df98a28f19448f4af244304b89a` | 6 | 5,5,6,8,7,6 |
| 5 | `cfbcfa3e382f12aed450b10f87a72aeb` | 6 | 5,5,5,6,5,6 |
| 6 | `3f5fa378239f7475baac89fa40288aaa` | 6 | 5,6,5,5,5,6 |
| 7 | `5c13daa9432149f6243aa1115629cd1a` | 7 | 5,6,5,9,8,5,6 |
| 8 | `694e275f1a01ad0e8ac448ad809f7930` | 8 | 5,5,7,7,7,7,6,6 |
| 9 | `aea2634b0e5f1c2444550d31c41adc94` | 8 | 6,5,7,5,6,5,5,6 |
| 10 | `79f0d0539d9ec0acbf90cb3388b30c17` | 7 | 5,7,6,6,7,8,6 |

All ten are ClapNQ conversations. `scripts/lmc_setup.sh` fetches `conversations.json`, `RAG.jsonl` and
`reference.jsonl` from MTRAG's repository at commit `e0970de1`, each checked by its SHA-256.

**What LMCache does in these runs.** vLLM's `MultiConnector` asks every connector for a request's tokens and gives
the load to the first that has some: on D, NIXL always does, so LMCache never loads on D. Its lookup still runs, and
D waits for it; in the disk mode that lookup reads the chunks it finds from disk into memory (its prefetch), and
with `retain` they stay there. P's requests carry no blocks to read, so on P LMCache loads what P's own GPU cache no
longer holds. With a server per engine each server keeps the chunks of the requests its engine ran.

**Disk.** The disk tier lives in the run's folder in node-local `/tmp` and is deleted after the run's report, so a
campaign needs room for its largest run: the per-engine carry-forward run on disk. Job 30763 measured the first
conversation: 12 chunks on the shared server (0.35 GB), 40 across the per-engine servers carry-forward (1.17 GB) and
26 chat (0.76 GB). The other nine are estimated from their text, scaled to that run's tokens (the estimate gives 12,
40 and 27 for the first). The campaign's disk plan (`scripts/pd_campaign.py`) reserves 0.75 GB a conversation, 3.6
copies for per-engine carry-forward and 2.3 for per-engine chat, 3.5 GB of setup and 0.1 GB a run, and keeps 10 GB
free. The setup's caches (16 GB: the packages 9.6, the model 6.4) stay on the node between campaigns.

| conversations | shared server | per engine, chat | per engine, carry-forward | free disk the plan asks for (caches there) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.35 GB (measured) | 0.76 GB (measured) | 1.17 GB (measured) | 17 GB |
| 5 | 3.1 GB | 6.8 GB | 10.7 GB | 28 GB |
| 10 | 6.6 GB | 14.3 GB | 23.3 GB | 41 GB |

The campaign's create stage stops before anything runs if the node's `/tmp` lacks that room.

### Running them

The runbook below, once per campaign, with these exported before `bash <<'LOGIN'`:

```bash
export KVPD_LMC_PRESET=eight KVPD_LMC_CONVS=1     # then 5, then 10: one campaign each
```

The preset sets `KVPD_LMC_TRANSFERS=nixl`, `KVPD_LMC_SERVERS=shared,per_engine`, `KVPD_LMC_MODES=memory,disk`,
`KVPD_LMC_LAYOUTS=history_first,interleaved`, `KVPD_LMC_DOCS=all`, `KVPD_LMC_CONV_IDS` (the ten),
`KVPD_LMC_CONCURRENCY=1`, `KVPD_LMC_DISK_READS=retain`, `KVPD_LMC_L1_SHARED_GB=8` and
`KVPD_LMC_L1_PER_ENGINE_GB=2`; each of them, exported too, wins over the preset. To see a campaign's runs first:
`MATRIX=lmctrace KVPD_LMC_PRESET=eight KVPD_LMC_CONVS=5 PRINT_RUNS=1 bash scripts/pd_matrix.sh`. Each run's folder
is `llama32_3b_lmc_<mode>_nixl_<servers>_<chat|carry>_<docs>_rtxa5000_<time>/`.

### What each of the eight must show

1. `env.json`: `transfer: nixl`; `servers`; `layout` and `layout_label` (`chat` or `carry`); `mtrag_docs: all`;
   `conv_ids` (the ten) and `mtrag_limit` (1, 5 or 10); `concurrency: 1`; `routing: round_robin`;
   `store_wait_ms: 0`; `l1_gb` 8 (shared) or 2 (per engine); in the disk mode `store_policy: skip_l1`,
   `prefetch_policy: retain`, `eviction: LRU`.
2. `mtrag.jsonl`: the conversations in the order above, each turn's `docs` from `reference.jsonl+RAG.jsonl`, and
   each prompt's earlier answers the ones D generated.
3. The report: every D request read the tokens its own GPU lacked from P over NIXL; LMCache loaded nothing on D; no
   failed reads; D's own GPU hits on the turns a D had seen the conversation before (carry-forward: the whole
   earlier prompt; chat: the instructions and the history up to the earlier answers); the chunks on each server
   (`per_server`, per engine).
4. The bytes and the answers. The bytes: kvtrace checksums every block NIXL moved when the read is done, D's copy
   on D's stream (as the model's next step reads it) and P's original it still holds (`nixl.done`'s `kv_sums`,
   `KVTRACE_KV_SUMS=1`); the report pairs each block D read with P's (`nixl.bytes_same_blocks` of
   `nixl.bytes_checked_blocks`, "NIXL blocks = P's bytes") and must find every one the same. The answers
   (`local_check.json`, `local_check_nixl.json`): after the run, one engine on P's GPU, no P/D and no trace, answers
   every turn's exact prompt at temperature 0, once with no connector (vLLM's own KV layout) and once with
   NixlConnector set and unused (the LBHNC layout NIXL sets on P and D); D's answers should be the same, token for
   token. With the bytes the same, a different answer is the computation's, not the transfer's: D computes the
   prompt's last token itself, in a step of its own, where one engine computes it inside its prefill.
   `LMC_LOCAL_CHECK=0` leaves the engines out; `LMC_LOCAL_CHECK_REFS` picks them.
5. Disk mode: every chunk on disk and then out of memory; "disk reads kept in memory": N of N still in memory at the
   end unless LRU evicted them (with the default policy they leave about 80 ms after the read: job 30728, 102 of
   102); the server's memory at the end holds what was read back.

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
for v in $(compgen -e | grep '^KVPD_LMC_' || true); do printf 'export %s=%q\n' "$v" "${!v}"; done >> "$STATE"
cat "$STATE"
printf 'PIN=%s\nTAG=%s\nSTATE=%s\n' "$KVPD_PIN" "$KVPD_TAG" "$STATE"
for STAGE in create run cleanup verify-cleanup review publish; do
    bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG"
done
echo "Campaign $KVPD_TAG published (the PUBLISHED line above names its branch)."
LOGIN
```

Options go before `bash <<'LOGIN'`, exported: for example `export KVPD_LMC_MODES=memory,disk,tiered` or
`export KVPD_LMC_CONVS=8`. LMCache's documented P/D, with a shared server and with a server per engine, in memory and
on disk (four runs): `export KVPD_LMC_TRANSFERS=nixl KVPD_LMC_SERVERS=shared,per_engine KVPD_LMC_MODES=memory,disk`. To resume after a stop: `source ~/.kvpd_campaigns/<tag>.sh` (it keeps the `KVPD_LMC_*` options too), `git switch -q --detach
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
