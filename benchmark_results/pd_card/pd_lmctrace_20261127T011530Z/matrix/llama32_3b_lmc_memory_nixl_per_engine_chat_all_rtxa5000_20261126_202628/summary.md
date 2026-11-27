# LMCache lifecycle: memory mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1112 records from 8 processes (D1 316, D2 315, D3 159, LMC_D1 56, LMC_D2 55, LMC_D3 28, LMC_P 104, P 79); hook errors 0; unreadable lines 0.

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

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 4 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`

## NIXL

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 173.240 ms, p90 187.740 ms
- P held its blocks for D from its request's end to D's read done: median 271.100 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.595 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 26 chunks (by {'P': 12, 'D': 14}; per server {'LMC_P': 12, 'LMC_D1': 5, 'LMC_D2': 6, 'LMC_D3': 3}); written to memory 26 (763.4 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 26 (14 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 30
- every lookup's outcome per chunk: {'new: stored by this request': 26, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 26}
- median time from a chunk's last block computed on P to the chunk in memory: 33.360 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 65.080 ms, p90 75.440 ms.

## Checks

- chunk tokens match prompt: 26 of 26
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 26 of 26
- d answers same as one engine: 4 of 5

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
| 0.0034 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 7dc8ad88) |
| 3.5153 | P looked it up (turn 2): new: stored by this request |
| 3.5185 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 520d0993) |
| 3.5187 | P's connector planned the store |
| 3.5387 | P's worker sent it to the server |
| 3.5399 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5401 | server memory (L1) reserved |
| 3.5810 | eviction order refreshed (a touch: no I/O) |
| 3.5848 | written to memory (L1) |
| 5.3317 | P looked it up (turn 3): held (on GPU already) |
| 5.3318 | eviction order refreshed (a touch: no I/O) |
| 5.3980 | eviction order refreshed (a touch: no I/O) |
| 7.0137 | P looked it up (turn 4): held (on GPU already) |
| 7.0138 | eviction order refreshed (a touch: no I/O) |
| 7.0803 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0792 | P answered (stored) |
| 0.0985 | D1 took the request |
| 0.1015 | the server got its lookup |
| 0.1024 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1042 | its connector had the lookup's answer |
| 0.3721 | the first token reached the proxy |
| 1.7416 | finished |
| 1.7453 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
