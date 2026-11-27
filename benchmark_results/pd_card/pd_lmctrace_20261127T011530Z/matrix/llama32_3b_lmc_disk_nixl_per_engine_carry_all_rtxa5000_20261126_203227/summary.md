# LMCache lifecycle: disk mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1235 records from 8 processes (D1 320, D2 322, D3 161, LMC_D1 77, LMC_D2 77, LMC_D3 37, LMC_P 157, P 84); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | 7168 |

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 3 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; one engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 351.370 ms, p90 364.910 ms
- P held its blocks for D from its request's end to D's read done: median 378.090 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.597 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 40 chunks (by {'P': 12, 'D': 28}; per server {'LMC_P': 12, 'LMC_D1': 9, 'LMC_D2': 12, 'LMC_D3': 7}); written to memory 40 (1174.4 MB); to disk 40; removed from memory 40
- pulled by D or P: 0 times, from –; never pulled: 40 (15 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 40, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 40}
- median time from a chunk's last block computed on P to the chunk in memory: 38.940 ms; memory to disk: 265.960 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 70.470 ms, p90 73.710 ms.

## Checks

- chunk tokens match prompt: 40 of 40
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 40 of 40
- d answers same as one engine: 3 of 5
- disk mode on disk then out of memory: 40 of 40
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 0 of 0 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end (LMC P): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC P): 12 files, 0.35 GB
- disk mode server memory at end (LMC D1): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D1): 9 files, 0.26 GB
- disk mode server memory at end (LMC D2): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D2): 12 files, 0.35 GB
- disk mode server memory at end (LMC D3): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D3): 7 files, 0.21 GB

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded (LMCache / NIXL) | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 0 | 0 | 515 | 0 | 0 | 0 / 0 | 515 | new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 0 | 515 | 0 | 0 | 0 / 515 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 0 | 1 | 1131 | 512 | 512 | 0 / 0 | 619 | held (on GPU already) 2, new: stored by this request 2 | – | – | 512 |
| D2 | 0 | 1 | 1131 | 0 | 0 | 0 / 1131 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 0 | 2 | 1847 | 1120 | 1024 | 0 / 0 | 727 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D3 | 0 | 2 | 1847 | 0 | 0 | 0 / 1847 | 0 | new: stored by this request 7 | – | – | 1792 |
| P | 0 | 3 | 2531 | 1840 | 1792 | 0 / 0 | 691 | held (on GPU already) 7, new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 3 | 2531 | 640 | 512 | 0 / 1891 | 0 | held (on GPU already) 2, new: stored by this request 7 | – | – | 1792 |
| P | 0 | 4 | 3205 | 2528 | 2304 | 0 / 0 | 677 | held (on GPU already) 9, new: stored by this request 3 | – | – | 768 |
| D2 | 0 | 4 | 3205 | 1248 | 1024 | 0 / 1957 | 0 | held (on GPU already) 4, new: stored by this request 8 | – | – | 2048 |

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0029 | P looked it up (turn 0): new: stored by this request |
| 0.0052 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request a182baf6) |
| 0.0052 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request a182baf6) |
| 0.0053 | P's connector planned the store |
| 0.0181 | P's worker sent it to the server |
| 0.0195 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0201 | server memory (L1) reserved |
| 0.0774 | eviction order refreshed (a touch: no I/O) |
| 0.0830 | written to memory (L1) |
| 0.1684 | written to disk (L2) |
| 0.1688 | removed from memory (L1) |
| 1.7483 | P looked it up (turn 1): held (on GPU already) |
| 1.7484 | eviction order refreshed (a touch: no I/O) |
| 1.8092 | eviction order refreshed (a touch: no I/O) |
| 3.6104 | P looked it up (turn 2): held (on GPU already) |
| 3.6105 | eviction order refreshed (a touch: no I/O) |
| 3.6834 | eviction order refreshed (a touch: no I/O) |
| 5.6470 | P looked it up (turn 3): held (on GPU already) |
| 5.6471 | eviction order refreshed (a touch: no I/O) |
| 5.7215 | eviction order refreshed (a touch: no I/O) |
| 7.5551 | P looked it up (turn 4): held (on GPU already) |
| 7.5552 | eviction order refreshed (a touch: no I/O) |
| 7.6294 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0804 | P answered (stored) |
| 0.0924 | D1 took the request |
| 0.0956 | the server got its lookup |
| 0.0982 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.0994 | its connector had the lookup's answer |
| 0.3691 | the first token reached the proxy |
| 1.7368 | finished |
| 1.7404 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
