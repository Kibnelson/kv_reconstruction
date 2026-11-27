# LMCache lifecycle: memory mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 0447555704eb.

Trace: 1073 records from 5 processes (D1 308, D2 310, D3 155, LMC 222, P 78); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | – |

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 216746.060 ms, p90 228660.420 ms
- P held its blocks for D from its request's end to D's read done: median 216920.550 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.001 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, cuda_ipc, tcp — GPU to GPU (cuda_ipc); every GPU visible to each engine: yes; UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on
- the read of P's GPU memory (UCX's ucp_get table): D1 zero-copy: cuda_ipc/cuda; D2 zero-copy: cuda_ipc/cuda; D3 zero-copy: cuda_ipc/cuda

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 92.630 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 88.310 ms, p90 92.040 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded (LMCache / NIXL) | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 0 | 0 | 515 | 0 | 0 | 0 / 0 | 515 | new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 0 | 515 | 0 | 512 | 0 / 515 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 0 | 1 | 1131 | 512 | 512 | 0 / 0 | 619 | held (on GPU already) 2, new: stored by this request 2 | – | – | 512 |
| D2 | 0 | 1 | 1131 | 0 | 1024 | 0 / 1131 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 0 | 2 | 1847 | 1120 | 1024 | 0 / 0 | 727 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D3 | 0 | 2 | 1847 | 0 | 1792 | 0 / 1847 | 0 | hit, not loaded 7 | – | – | 0 |
| P | 0 | 3 | 2531 | 1840 | 1792 | 0 / 0 | 691 | held (on GPU already) 7, new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 3 | 2531 | 640 | 2304 | 0 / 1891 | 0 | held (on GPU already) 2, hit, not loaded 7 | – | – | 0 |
| P | 0 | 4 | 3205 | 2528 | 2304 | 0 / 0 | 677 | held (on GPU already) 9, new: stored by this request 3 | – | – | 768 |
| D2 | 0 | 4 | 3205 | 1248 | 3072 | 0 / 1957 | 0 | held (on GPU already) 4, hit, not loaded 8 | – | – | 0 |

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0029 | P looked it up (turn 0): new: stored by this request |
| 0.0051 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request addc6b1d) |
| 0.0051 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request addc6b1d) |
| 0.0053 | P's connector planned the store |
| 0.0178 | P's worker sent it to the server |
| 0.0192 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0197 | server memory (L1) reserved |
| 0.0772 | eviction order refreshed (a touch: no I/O) |
| 0.0809 | written to memory (L1) |
| 0.1022 | D1 looked it up (turn 0): hit, not loaded |
| 64.6586 | eviction order refreshed (a touch: no I/O) |
| 64.6740 | P looked it up (turn 1): held (on GPU already) |
| 64.6740 | eviction order refreshed (a touch: no I/O) |
| 64.7545 | eviction order refreshed (a touch: no I/O) |
| 64.7726 | D2 looked it up (turn 1): hit, not loaded |
| 200.0715 | eviction order refreshed (a touch: no I/O) |
| 200.0921 | P looked it up (turn 2): held (on GPU already) |
| 200.0921 | eviction order refreshed (a touch: no I/O) |
| 200.1821 | eviction order refreshed (a touch: no I/O) |
| 200.2050 | D3 looked it up (turn 2): hit, not loaded |
| 418.5367 | eviction order refreshed (a touch: no I/O) |
| 418.5527 | P looked it up (turn 3): held (on GPU already) |
| 418.5528 | eviction order refreshed (a touch: no I/O) |
| 418.6429 | eviction order refreshed (a touch: no I/O) |
| 418.6590 | D1 looked it up (turn 3): held (on GPU already) |
| 418.6591 | eviction order refreshed (a touch: no I/O) |
| 641.2961 | eviction order refreshed (a touch: no I/O) |
| 641.3161 | P looked it up (turn 4): held (on GPU already) |
| 641.3163 | eviction order refreshed (a touch: no I/O) |
| 641.4096 | eviction order refreshed (a touch: no I/O) |
| 641.4292 | D2 looked it up (turn 4): held (on GPU already) |
| 641.4293 | eviction order refreshed (a touch: no I/O) |
| 871.5535 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0802 | P answered (stored) |
| 0.0996 | D1 took the request |
| 0.1022 | the server got its lookup |
| 0.1034 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1038 | its connector had the lookup's answer |
| 63.2923 | the first token reached the proxy |
| 64.6580 | finished |
| 64.6613 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
