# LMCache lifecycle: disk mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1173 records from 5 processes (D1 348, D2 311, D3 157, LMC 274, P 83); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 4238 | 4238 |
| from the engine's own GPU prefix cache | 928 | 176 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 4062 |
| computed | 3310 | 0 |
| stored to LMCache (tokens) | 3072 | – |

## NIXL

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 178.740 ms, p90 189.710 ms
- P held its blocks for D from its request's end to D's read done: median 278.060 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 12; removed from memory 12
- pulled by D or P: 0 times, from –; never pulled: 12 (6 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 1; touches (eviction order refreshed, no I/O): 30
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 14, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 68.130 ms; memory to disk: 108.680 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 63.580 ms, p90 75.930 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
- disk mode on disk then out of memory: 12 of 12
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 1 of 1 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '1 objects, 29360128 bytes'}
- disk mode files at end: 12 files, 0.35 GB

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded (LMCache / NIXL) | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 0 | 0 | 515 | 0 | 0 | 0 / 0 | 515 | new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 0 | 515 | 0 | 512 | 0 / 515 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 0 | 1 | 657 | 16 | 0 | 0 / 0 | 641 | new: stored by this request 2 | – | – | 512 |
| D2 | 0 | 1 | 657 | 0 | 512 | 0 / 657 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 0 | 2 | 899 | 160 | 0 | 0 / 0 | 739 | new: stored by this request 3 | – | – | 768 |
| D3 | 0 | 2 | 899 | 0 | 768 | 0 / 899 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 0 | 3 | 1015 | 304 | 256 | 0 / 0 | 711 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 3 | 1015 | 16 | 768 | 0 / 999 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 0 | 4 | 1152 | 448 | 256 | 0 / 0 | 704 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D2 | 0 | 4 | 1152 | 160 | 1024 | 0 / 992 | 0 | hit, not loaded 4 | – | – | 0 |

## One chunk's life: 803ab0bea97e045e

Conversation 0, turn 2, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0068 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 302295e1) |
| 3.5098 | P looked it up (turn 2): new: stored by this request |
| 3.5139 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 418aa524) |
| 3.5141 | P's connector planned the store |
| 3.5255 | P's worker sent it to the server |
| 3.5266 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5271 | server memory (L1) reserved |
| 3.5746 | eviction order refreshed (a touch: no I/O) |
| 3.5784 | written to memory (L1) |
| 3.6000 | D3 looked it up (turn 2): hit, not loaded |
| 3.6871 | written to disk (L2) |
| 3.6877 | removed from memory (L1) |
| 5.3166 | eviction order refreshed (a touch: no I/O) |
| 5.3256 | P looked it up (turn 3): held (on GPU already) |
| 5.3257 | eviction order refreshed (a touch: no I/O) |
| 5.3929 | eviction order refreshed (a touch: no I/O) |
| 5.4004 | D1 looked it up (turn 3): hit, not loaded |
| 5.4265 | read from disk (L2) |
| 7.0249 | eviction order refreshed (a touch: no I/O) |
| 7.0356 | P looked it up (turn 4): held (on GPU already) |
| 7.0357 | eviction order refreshed (a touch: no I/O) |
| 7.1024 | eviction order refreshed (a touch: no I/O) |
| 7.1114 | D2 looked it up (turn 4): hit, not loaded |
| 8.7125 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0025 | the proxy received the turn |
| 0.0829 | P answered (stored) |
| 0.1031 | D1 took the request |
| 0.1057 | the server got its lookup |
| 0.1082 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1088 | its connector had the lookup's answer |
| 0.3781 | the first token reached the proxy |
| 1.7454 | finished |
| 1.7488 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
