# LMCache lifecycle: disk mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1214 records from 8 processes (D1 318, D2 317, D3 161, LMC_D1 74, LMC_D2 73, LMC_D3 37, LMC_P 151, P 83); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 4238 | 4238 |
| from the engine's own GPU prefix cache | 928 | 176 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 4062 |
| computed | 3310 | 0 |
| stored to LMCache (tokens) | 3072 | 3584 |

## NIXL

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 173.910 ms, p90 187.870 ms
- P held its blocks for D from its request's end to D's read done: median 267.880 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 26 chunks (by {'P': 12, 'D': 14}; per server {'LMC_P': 12, 'LMC_D1': 5, 'LMC_D2': 6, 'LMC_D3': 3}); written to memory 26 (763.4 MB); to disk 26; removed from memory 26
- pulled by D or P: 0 times, from –; never pulled: 26 (14 of them hold an earlier answer, which D generated itself and holds)
- reads from disk and touches are one event in this trace (before the read hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)
- every lookup's outcome per chunk: {'new: stored by this request': 26, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 26}
- median time from a chunk's last block computed on P to the chunk in memory: 31.320 ms; memory to disk: 109.430 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 65.230 ms, p90 75.490 ms.

## Checks

- chunk tokens match prompt: 26 of 26
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 26 of 26
- disk mode on disk then out of memory: 26 of 26
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 0 of 0 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end (LMC P): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC P): 12 files, 0.35 GB
- disk mode server memory at end (LMC D1): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D1): 5 files, 0.15 GB
- disk mode server memory at end (LMC D2): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D2): 6 files, 0.18 GB
- disk mode server memory at end (LMC D3): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D3): 3 files, 0.09 GB

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded (LMCache / NIXL) | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 0 | 0 | 515 | 0 | 0 | 0 / 0 | 515 | new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 0 | 515 | 0 | 0 | 0 / 515 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 0 | 1 | 657 | 16 | 0 | 0 / 0 | 641 | new: stored by this request 2 | – | – | 512 |
| D2 | 0 | 1 | 657 | 0 | 0 | 0 / 657 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 0 | 2 | 899 | 160 | 0 | 0 / 0 | 739 | new: stored by this request 3 | – | – | 768 |
| D3 | 0 | 2 | 899 | 0 | 0 | 0 / 899 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 0 | 3 | 1015 | 304 | 256 | 0 / 0 | 711 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 3 | 1015 | 16 | 0 | 0 / 999 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 0 | 4 | 1152 | 448 | 256 | 0 / 0 | 704 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D2 | 0 | 4 | 1152 | 160 | 0 | 0 / 992 | 0 | new: stored by this request 4 | – | – | 1024 |

## One chunk's life: 803ab0bea97e045e

Conversation 0, turn 2, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0062 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request f1e029e8) |
| 3.5049 | P looked it up (turn 2): new: stored by this request |
| 3.5096 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 0e113f58) |
| 3.5099 | P's connector planned the store |
| 3.5290 | P's worker sent it to the server |
| 3.5301 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5304 | server memory (L1) reserved |
| 3.5724 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 3.5768 | written to memory (L1) |
| 3.7011 | written to disk (L2) |
| 3.7017 | removed from memory (L1) |
| 5.3314 | P looked it up (turn 3): held (on GPU already) |
| 5.3316 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 5.3973 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 7.0168 | P looked it up (turn 4): held (on GPU already) |
| 7.0170 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 7.0852 | disk tier accessed (a read from disk, or a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0825 | P answered (stored) |
| 0.0934 | D1 took the request |
| 0.0968 | the server got its lookup |
| 0.0993 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1002 | its connector had the lookup's answer |
| 0.3681 | the first token reached the proxy |
| 1.7379 | finished |
| 1.7414 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
