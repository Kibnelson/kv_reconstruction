# LMCache lifecycle: disk mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 7719 records from 8 processes (D1 1594, D2 1612, D3 1751, LMC_D1 401, LMC_D2 404, LMC_D3 417, LMC_P 992, P 548); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 33 | 33 |
| prompt tokens | 40904 | 40904 |
| from the engine's own GPU prefix cache | 10592 | 4080 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 36824 |
| computed | 30312 | 0 |
| stored to LMCache (tokens) | 29184 | 34560 |

**D's answers against one plain engine (no P/D, no connector: vLLM's own KV layout)** (the same prompts, temperature 0; `local_check.json`): 26 of 33 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 1/0, from token 39: D `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`; the engine `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`
- turn 1/1, from token 19: D `' Guinea pigs are herbivores and eat a variety of grasses and plants. They also eat hay, which is a t'`; the engine `' Guinea pigs are herbivores and eat a variety of grasses and plants. They also eat their own feces, '`
- turn 1/3, from token 22: D `' Yes, they have predators. Some of the most common predators of guinea pigs are cats, dogs, and bird'`; the engine `' Yes, they have predators. Some of the most common predators of guinea pigs are cats, dogs, and othe'`
- turn 1/4, from token 19: D `' Yes, guinea pigs are raised for human consumption. They are a popular food in South America and are'`; the engine `' Yes, guinea pigs are raised for human consumption. They are a popular food in South America, especi'`

**D's answers against one engine with NIXL's KV layout (NixlConnector set, unused; no P/D)** (the same prompts, temperature 0; `local_check_nixl.json`): 24 of 33 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 1/0, from token 39: D `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`; the engine `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`
- turn 1/1, from token 19: D `' Guinea pigs are herbivores and eat a variety of grasses and plants. They also eat hay, which is a t'`; the engine `' Guinea pigs are herbivores and eat a variety of grasses and plants. They also eat their own feces, '`
- turn 1/4, from token 19: D `' Yes, guinea pigs are raised for human consumption. They are a popular food in South America and are'`; the engine `' Yes, guinea pigs are raised for human consumption. They are a popular food in South America, especi'`
- turn 2/1, from token 17: D `' A catchbasin is a small reservoir that collects water and debris from the street and directs it int'`; the engine `' A catchbasin is a small reservoir that collects water and debris from the street and prevents the p'`

## NIXL

- D requests that read from P: 33, 36824 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 199.820 ms, p90 286.330 ms
- P held its blocks for D from its request's end to D's read done: median 235.140 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.613 GB/s (the KV it read over its time)
- the bytes: 132 of 2318 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 33; DIFFERS: D1 block 43 from P's 34; DIFFERS: D1 block 44 from P's 35; DIFFERS: D1 block 45 from P's 36; DIFFERS: D1 block 46 from P's 37; DIFFERS: D1 block 47 from P's 38; DIFFERS: D1 block 48 from P's 39
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 249 chunks (by {'P': 114, 'D': 135}; per server {'LMC_P': 114, 'LMC_D1': 43, 'LMC_D2': 44, 'LMC_D3': 48}); written to memory 249 (7310.7 MB); to disk 249; removed from memory 249
- pulled by D or P: 0 times, from –; never pulled: 249 (99 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 323
- every lookup's outcome per chunk: {'new: stored by this request': 249, 'held (on GPU already)': 37}
- pulls per stored chunk: {0: 249}
- median time from a chunk's last block computed on P to the chunk in memory: 37.500 ms; memory to disk: 168.590 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 82.920 ms, p90 121.780 ms.

## Checks

- chunk tokens match prompt: 249 of 249
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 249 of 249
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33
- disk mode on disk then out of memory: 249 of 249
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 0 of 0 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end (LMC P): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC P): 114 files, 3.35 GB
- disk mode server memory at end (LMC D1): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D1): 43 files, 1.26 GB
- disk mode server memory at end (LMC D2): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D2): 44 files, 1.29 GB
- disk mode server memory at end (LMC D3): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D3): 48 files, 1.41 GB

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
| P | 1 | 0 | 1087 | 16 | 0 | 0 / 0 | 1071 | new: stored by this request 4 | – | – | 1024 |
| D3 | 1 | 0 | 1087 | 16 | 0 | 0 / 1071 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 1 | 1 | 1412 | 16 | 0 | 0 / 0 | 1396 | new: stored by this request 5 | – | – | 1280 |
| D1 | 1 | 1 | 1412 | 16 | 0 | 0 / 1396 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 1 | 2 | 1347 | 160 | 0 | 0 / 0 | 1187 | new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 2 | 1347 | 16 | 0 | 0 / 1331 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 1 | 3 | 1221 | 304 | 256 | 0 / 0 | 917 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D3 | 1 | 3 | 1221 | 16 | 0 | 0 / 1205 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 1 | 4 | 1850 | 336 | 256 | 0 / 0 | 1514 | held (on GPU already) 1, new: stored by this request 6 | – | – | 1536 |
| D1 | 1 | 4 | 1850 | 160 | 0 | 0 / 1690 | 0 | new: stored by this request 7 | – | – | 1792 |
| P | 1 | 5 | 1759 | 480 | 256 | 0 / 0 | 1279 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 5 | 1759 | 304 | 256 | 0 / 1455 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 1 | 6 | 1923 | 624 | 512 | 0 / 0 | 1299 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 1 | 6 | 1923 | 336 | 256 | 0 / 1587 | 0 | held (on GPU already) 1, new: stored by this request 6 | – | – | 1536 |
| P | 2 | 0 | 641 | 16 | 0 | 0 / 0 | 625 | new: stored by this request 2 | – | – | 512 |
| D1 | 2 | 0 | 641 | 16 | 0 | 0 / 625 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 2 | 1 | 687 | 16 | 0 | 0 / 0 | 671 | new: stored by this request 2 | – | – | 512 |
| D2 | 2 | 1 | 687 | 16 | 0 | 0 / 671 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 2 | 2 | 851 | 48 | 0 | 0 / 0 | 803 | new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 2 | 851 | 16 | 0 | 0 / 835 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 2 | 3 | 957 | 192 | 0 | 0 / 0 | 765 | new: stored by this request 3 | – | – | 768 |
| D1 | 2 | 3 | 957 | 16 | 0 | 0 / 941 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 2 | 4 | 1516 | 336 | 256 | 0 / 0 | 1180 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 2 | 4 | 1516 | 48 | 0 | 0 / 1468 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 2 | 5 | 1270 | 496 | 256 | 0 / 0 | 774 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 5 | 1270 | 192 | 0 | 0 / 1078 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 2 | 6 | 1646 | 672 | 512 | 0 / 0 | 974 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D1 | 2 | 6 | 1646 | 336 | 256 | 0 / 1310 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 2 | 7 | 1608 | 816 | 768 | 0 / 0 | 792 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 2 | 7 | 1608 | 496 | 256 | 0 / 1112 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 2 | 8 | 2175 | 960 | 768 | 0 / 0 | 1215 | held (on GPU already) 3, new: stored by this request 5 | – | – | 1280 |
| D3 | 2 | 8 | 2175 | 672 | 512 | 0 / 1503 | 0 | held (on GPU already) 2, new: stored by this request 6 | – | – | 1536 |
| P | 3 | 0 | 633 | 16 | 0 | 0 / 0 | 617 | new: stored by this request 2 | – | – | 512 |
| D1 | 3 | 0 | 633 | 16 | 0 | 0 / 617 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 3 | 1 | 750 | 16 | 0 | 0 / 0 | 734 | new: stored by this request 2 | – | – | 512 |
| D2 | 3 | 1 | 750 | 16 | 0 | 0 / 734 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 3 | 2 | 1060 | 144 | 0 | 0 / 0 | 916 | new: stored by this request 4 | – | – | 1024 |
| D3 | 3 | 2 | 1060 | 16 | 0 | 0 / 1044 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 3 | 3 | 1237 | 288 | 256 | 0 / 0 | 949 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D1 | 3 | 3 | 1237 | 16 | 0 | 0 / 1221 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 3 | 4 | 1132 | 432 | 256 | 0 / 0 | 700 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D2 | 3 | 4 | 1132 | 144 | 0 | 0 / 988 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 3 | 5 | 1336 | 576 | 512 | 0 / 0 | 760 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 5 | 1336 | 288 | 256 | 0 / 1048 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 4 | 0 | 1087 | 1072 | 1024 | 0 / 0 | 15 | held (on GPU already) 4 | – | – | 0 |
| D1 | 4 | 0 | 1087 | 16 | 0 | 0 / 1071 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 4 | 1 | 1256 | 80 | 0 | 0 / 0 | 1176 | new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 1256 | 80 | 0 | 0 / 1176 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 4 | 2 | 1041 | 160 | 0 | 0 / 0 | 881 | new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 1041 | 80 | 0 | 0 / 961 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 4 | 3 | 1667 | 320 | 256 | 0 / 0 | 1347 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 1667 | 80 | 0 | 0 / 1587 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 4 | 4 | 1676 | 464 | 256 | 0 / 0 | 1212 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 1676 | 160 | 0 | 0 / 1516 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 4 | 5 | 1841 | 608 | 512 | 0 / 0 | 1233 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 4 | 5 | 1841 | 320 | 256 | 0 / 1521 | 0 | held (on GPU already) 1, new: stored by this request 6 | – | – | 1536 |

## One chunk's life: aa54d2c3db5e0d3a

Conversation 2, turn 3, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1, newline 1, user 2, assistant 2, answer 2); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0053 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 8d7072e4) |
| 23.8273 | P looked it up (turn 3): new: stored by this request |
| 23.8306 | P computed and cached block 15 (GPU block 878, hash 5c2a3032cc6dadfb, request 21d9f8d9) |
| 23.8309 | P's connector planned the store |
| 23.8496 | P's worker sent it to the server |
| 23.8510 | the server bound the chunk to its tokens (mp.tokens) |
| 23.8515 | server memory (L1) reserved |
| 23.8933 | eviction order refreshed (a touch: no I/O) |
| 23.8966 | written to memory (L1) |
| 24.0130 | written to disk (L2) |
| 24.0135 | removed from memory (L1) |
| 25.5177 | P looked it up (turn 4): held (on GPU already) |
| 25.5178 | eviction order refreshed (a touch: no I/O) |
| 25.6235 | eviction order refreshed (a touch: no I/O) |
| 27.3538 | P looked it up (turn 5): held (on GPU already) |
| 27.3538 | eviction order refreshed (a touch: no I/O) |
| 27.4305 | eviction order refreshed (a touch: no I/O) |
| 29.0950 | P looked it up (turn 6): held (on GPU already) |
| 29.0950 | eviction order refreshed (a touch: no I/O) |
| 29.1828 | eviction order refreshed (a touch: no I/O) |
| 30.8920 | P looked it up (turn 7): held (on GPU already) |
| 30.8921 | eviction order refreshed (a touch: no I/O) |
| 30.9706 | eviction order refreshed (a touch: no I/O) |
| 32.6455 | P looked it up (turn 8): held (on GPU already) |
| 32.6457 | eviction order refreshed (a touch: no I/O) |
| 32.7558 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0793 | P answered (stored) |
| 0.0973 | D1 took the request |
| 0.1014 | the server got its lookup |
| 0.1036 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1048 | its connector had the lookup's answer |
| 0.4030 | the first token reached the proxy |
| 1.7707 | finished |
| 1.7741 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
