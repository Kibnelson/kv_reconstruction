# LMCache lifecycle: memory mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1069 records from 5 processes (D1 312, D2 309, D3 155, LMC 213, P 80); hook errors 0; unreadable lines 0.

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

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 177.920 ms, p90 187.560 ms
- P held its blocks for D from its request's end to D's read done: median 288.960 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 12 (6 of them hold an earlier answer, which D generated itself and holds)
- reads from disk and touches are one event in this trace (before the read hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 14, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 65.990 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 65.080 ms, p90 73.120 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12

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
| 0.0051 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 4debcbbf) |
| 3.5228 | P looked it up (turn 2): new: stored by this request |
| 3.5261 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 9d9bfd76) |
| 3.5264 | P's connector planned the store |
| 3.5464 | P's worker sent it to the server |
| 3.5475 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5478 | server memory (L1) reserved |
| 3.5921 | written to memory (L1) |
| 3.6159 | D3 looked it up (turn 2): hit, not loaded |
| 5.3412 | P looked it up (turn 3): held (on GPU already) |
| 5.4137 | D1 looked it up (turn 3): hit, not loaded |
| 7.0222 | P looked it up (turn 4): held (on GPU already) |
| 7.0961 | D2 looked it up (turn 4): hit, not loaded |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0782 | P answered (stored) |
| 0.1188 | D1 took the request |
| 0.1217 | the server got its lookup |
| 0.1226 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1244 | its connector had the lookup's answer |
| 0.3851 | the first token reached the proxy |
| 1.7518 | finished |
| 1.7553 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
