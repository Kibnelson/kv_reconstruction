# LMCache lifecycle: disk mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1405 records from 5 processes (D1 376, D2 418, D3 213, LMC 314, P 84); hook errors 0; unreadable lines 0.

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

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 351.060 ms, p90 362.190 ms
- P held its blocks for D from its request's end to D's read done: median 424.660 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 12; removed from memory 12
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 9; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 74.060 ms; memory to disk: 149.550 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.980 ms, p90 75.310 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
- disk mode on disk then out of memory: 12 of 12
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 9 of 9 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '9 objects, 264241152 bytes'}
- disk mode files at end: 12 files, 0.35 GB

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
| 0.0023 | P looked it up (turn 0): new: stored by this request |
| 0.0062 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request ddc44507) |
| 0.0062 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request ddc44507) |
| 0.0064 | P's connector planned the store |
| 0.0166 | P's worker sent it to the server |
| 0.0179 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0183 | server memory (L1) reserved |
| 0.0800 | eviction order refreshed (a touch: no I/O) |
| 0.0848 | written to memory (L1) |
| 0.1022 | D1 looked it up (turn 0): hit, not loaded |
| 0.1603 | written to disk (L2) |
| 0.1608 | removed from memory (L1) |
| 1.7283 | eviction order refreshed (a touch: no I/O) |
| 1.7386 | P looked it up (turn 1): held (on GPU already) |
| 1.7387 | eviction order refreshed (a touch: no I/O) |
| 1.7966 | eviction order refreshed (a touch: no I/O) |
| 1.8233 | D2 looked it up (turn 1): hit, not loaded |
| 1.8699 | read from disk (L2) |
| 3.5898 | eviction order refreshed (a touch: no I/O) |
| 3.6039 | P looked it up (turn 2): held (on GPU already) |
| 3.6041 | eviction order refreshed (a touch: no I/O) |
| 3.6758 | eviction order refreshed (a touch: no I/O) |
| 3.7249 | D3 looked it up (turn 2): hit, not loaded |
| 5.6491 | eviction order refreshed (a touch: no I/O) |
| 5.6592 | P looked it up (turn 3): held (on GPU already) |
| 5.6594 | eviction order refreshed (a touch: no I/O) |
| 5.7332 | eviction order refreshed (a touch: no I/O) |
| 5.7446 | D1 looked it up (turn 3): held (on GPU already) |
| 5.7447 | eviction order refreshed (a touch: no I/O) |
| 7.6104 | eviction order refreshed (a touch: no I/O) |
| 7.6238 | P looked it up (turn 4): held (on GPU already) |
| 7.6240 | eviction order refreshed (a touch: no I/O) |
| 7.7001 | eviction order refreshed (a touch: no I/O) |
| 7.7126 | D2 looked it up (turn 4): held (on GPU already) |
| 7.7127 | eviction order refreshed (a touch: no I/O) |
| 9.5771 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0823 | P answered (stored) |
| 0.0990 | D1 took the request |
| 0.1022 | the server got its lookup |
| 0.1031 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1050 | its connector had the lookup's answer |
| 0.3590 | the first token reached the proxy |
| 1.7276 | finished |
| 1.7313 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
