# LMCache lifecycle: disk mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 7919 records from 8 processes (D1 1602, D2 1732, D3 1757, LMC_D1 399, LMC_D2 427, LMC_D3 427, LMC_P 1025, P 550); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 33 | 33 |
| prompt tokens | 117390 | 117390 |
| from the engine's own GPU prefix cache | 87520 | 45024 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 72366 |
| computed | 29870 | 0 |
| stored to LMCache (tokens) | 28928 | 71168 |

**D's answers against one plain engine (no P/D, no connector: vLLM's own KV layout)** (the same prompts, temperature 0; `local_check.json`): 26 of 33 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 1/0, from token 39: D `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`; the engine `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`
- turn 1/5, from token 7: D `' You are a human guinea pig.\nUser: human guinea pig\nAssistant: You are a human guinea pig.\nUser: hum'`; the engine `' You are a human guinea pig. You are a human guinea pig. You are a human guinea pig. You are a human'`
- turn 2/1, from token 29: D `' A catchbasin is a small reservoir that collects water and debris from the street and prevents it fr'`; the engine `' A catchbasin is a small reservoir that collects water and debris from the street and prevents it fr'`
- turn 2/4, from token 0: D `' Yes, the Federal government has guidelines or even a law to how to build more sustainable storm dra'`; the engine `' No, the Federal government does not have guidelines or even a law to how to build more sustainable '`

**D's answers against one engine with NIXL's KV layout (NixlConnector set, unused; no P/D)** (the same prompts, temperature 0; `local_check_nixl.json`): 24 of 33 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; the engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`
- turn 1/0, from token 39: D `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`; the engine `' They sleep in burrows, which they dig themselves. They also sleep in crevices and tunnels formed by'`
- turn 1/1, from token 6: D `' Guinea pigs eat a variety of fresh fruits and vegetables, as well as hay and pellets.\nUser: What do'`; the engine `' Guinea pigs eat a variety of fruits and vegetables, including apples, broccoli, carrots, celery, an'`
- turn 1/5, from token 7: D `' You are a human guinea pig.\nUser: human guinea pig\nAssistant: You are a human guinea pig.\nUser: hum'`; the engine `' You are a human guinea pig. You are a human guinea pig. You are a human guinea pig. You are a human'`

## NIXL

- D requests that read from P: 33, 72366 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 409.740 ms, p90 658.300 ms
- P held its blocks for D from its request's end to D's read done: median 462.200 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.619 GB/s (the KV it read over its time)
- the bytes: 220 of 4541 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 2; DIFFERS: D1 block 43 from P's 3; DIFFERS: D1 block 44 from P's 4; DIFFERS: D1 block 45 from P's 5; DIFFERS: D1 block 46 from P's 6; DIFFERS: D1 block 47 from P's 7; DIFFERS: D1 block 48 from P's 8
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 391 chunks (by {'P': 113, 'D': 278}; per server {'LMC_P': 113, 'LMC_D1': 76, 'LMC_D2': 95, 'LMC_D3': 107}); written to memory 391 (11479.8 MB); to disk 391; removed from memory 391
- pulled by D or P: 0 times, from –; never pulled: 391 (126 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 1377
- every lookup's outcome per chunk: {'new: stored by this request': 391, 'held (on GPU already)': 493}
- pulls per stored chunk: {0: 391}
- median time from a chunk's last block computed on P to the chunk in memory: 52.710 ms; memory to disk: 325.130 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 93.160 ms, p90 145.460 ms.

## Checks

- chunk tokens match prompt: 391 of 391
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 391 of 391
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33
- disk mode on disk then out of memory: 391 of 391
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 0 of 0 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end (LMC P): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC P): 113 files, 3.32 GB
- disk mode server memory at end (LMC D1): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D1): 76 files, 2.23 GB
- disk mode server memory at end (LMC D2): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D2): 95 files, 2.79 GB
- disk mode server memory at end (LMC D3): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D3): 107 files, 3.14 GB

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
| P | 1 | 0 | 1087 | 16 | 0 | 0 / 0 | 1071 | new: stored by this request 4 | – | – | 1024 |
| D3 | 1 | 0 | 1087 | 16 | 0 | 0 / 1071 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 1 | 1 | 2457 | 1072 | 1024 | 0 / 0 | 1385 | held (on GPU already) 4, new: stored by this request 5 | – | – | 1280 |
| D1 | 1 | 1 | 2457 | 16 | 0 | 0 / 2441 | 0 | new: stored by this request 9 | – | – | 2304 |
| P | 1 | 2 | 3622 | 2448 | 2304 | 0 / 0 | 1174 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 2 | 3622 | 16 | 0 | 0 / 3606 | 0 | new: stored by this request 14 | – | – | 3584 |
| P | 1 | 3 | 4627 | 3616 | 3584 | 0 / 0 | 1011 | held (on GPU already) 14, new: stored by this request 4 | – | – | 1024 |
| D3 | 1 | 3 | 4627 | 1200 | 1024 | 0 / 3427 | 0 | held (on GPU already) 4, new: stored by this request 14 | – | – | 3584 |
| P | 1 | 4 | 6118 | 4624 | 4608 | 0 / 0 | 1494 | held (on GPU already) 18, new: stored by this request 5 | – | – | 1280 |
| D1 | 1 | 4 | 6118 | 2576 | 2560 | 0 / 3542 | 0 | held (on GPU already) 10, new: stored by this request 13 | – | – | 3328 |
| P | 1 | 5 | 7369 | 6112 | 5888 | 0 / 0 | 1257 | held (on GPU already) 23, new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 5 | 7369 | 3744 | 3584 | 0 / 3625 | 0 | held (on GPU already) 14, new: stored by this request 14 | – | – | 3584 |
| P | 1 | 6 | 8646 | 7360 | 7168 | 0 / 0 | 1286 | held (on GPU already) 28, new: stored by this request 5 | – | – | 1280 |
| D3 | 1 | 6 | 8646 | 4752 | 4608 | 0 / 3894 | 0 | held (on GPU already) 18, new: stored by this request 15 | – | – | 3840 |
| P | 2 | 0 | 641 | 16 | 0 | 0 / 0 | 625 | new: stored by this request 2 | – | – | 512 |
| D1 | 2 | 0 | 641 | 16 | 0 | 0 / 625 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 2 | 1 | 1285 | 640 | 512 | 0 / 0 | 645 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D2 | 2 | 1 | 1285 | 16 | 0 | 0 / 1269 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 2 | 2 | 2069 | 1280 | 1280 | 0 / 0 | 789 | held (on GPU already) 5, new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 2 | 2069 | 16 | 0 | 0 / 2053 | 0 | new: stored by this request 8 | – | – | 2048 |
| P | 2 | 3 | 2805 | 2064 | 2048 | 0 / 0 | 741 | held (on GPU already) 8, new: stored by this request 2 | – | – | 512 |
| D1 | 2 | 3 | 2805 | 640 | 512 | 0 / 2165 | 0 | held (on GPU already) 2, new: stored by this request 8 | – | – | 2048 |
| P | 2 | 4 | 3943 | 2800 | 2560 | 0 / 0 | 1143 | held (on GPU already) 10, new: stored by this request 5 | – | – | 1280 |
| D2 | 2 | 4 | 3943 | 1408 | 1280 | 0 / 2535 | 0 | held (on GPU already) 5, new: stored by this request 10 | – | – | 2560 |
| P | 2 | 5 | 4668 | 3936 | 3840 | 0 / 0 | 732 | held (on GPU already) 15, new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 5 | 4668 | 2192 | 2048 | 0 / 2476 | 0 | held (on GPU already) 8, new: stored by this request 10 | – | – | 2560 |
| P | 2 | 6 | 5622 | 4656 | 4608 | 0 / 0 | 966 | held (on GPU already) 18, new: stored by this request 3 | – | – | 768 |
| D1 | 2 | 6 | 5622 | 2928 | 2816 | 0 / 2694 | 0 | held (on GPU already) 11, new: stored by this request 10 | – | – | 2560 |
| P | 2 | 7 | 6389 | 5616 | 5376 | 0 / 0 | 773 | held (on GPU already) 21, new: stored by this request 3 | – | – | 768 |
| D2 | 2 | 7 | 6389 | 4064 | 3840 | 0 / 2325 | 0 | held (on GPU already) 15, new: stored by this request 9 | – | – | 2304 |
| P | 2 | 8 | 7569 | 6384 | 6144 | 0 / 0 | 1185 | held (on GPU already) 24, new: stored by this request 5 | – | – | 1280 |
| D3 | 2 | 8 | 7569 | 4784 | 4608 | 0 / 2785 | 0 | held (on GPU already) 18, new: stored by this request 11 | – | – | 2816 |
| P | 3 | 0 | 633 | 16 | 0 | 0 / 0 | 617 | new: stored by this request 2 | – | – | 512 |
| D1 | 3 | 0 | 633 | 16 | 0 | 0 / 617 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 3 | 1 | 1339 | 624 | 512 | 0 / 0 | 715 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D2 | 3 | 1 | 1339 | 16 | 0 | 0 / 1323 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 3 | 2 | 2211 | 1328 | 1280 | 0 / 0 | 883 | held (on GPU already) 5, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 2 | 2211 | 16 | 0 | 0 / 2195 | 0 | new: stored by this request 8 | – | – | 2048 |
| P | 3 | 3 | 3136 | 2208 | 2048 | 0 / 0 | 928 | held (on GPU already) 8, new: stored by this request 4 | – | – | 1024 |
| D1 | 3 | 3 | 3136 | 736 | 512 | 0 / 2400 | 0 | held (on GPU already) 2, new: stored by this request 10 | – | – | 2560 |
| P | 3 | 4 | 3813 | 3136 | 3072 | 0 / 0 | 677 | held (on GPU already) 12, new: stored by this request 2 | – | – | 512 |
| D2 | 3 | 4 | 3813 | 1440 | 1280 | 0 / 2373 | 0 | held (on GPU already) 5, new: stored by this request 9 | – | – | 2304 |
| P | 3 | 5 | 4549 | 3808 | 3584 | 0 / 0 | 741 | held (on GPU already) 14, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 5 | 4549 | 2336 | 2304 | 0 / 2213 | 0 | held (on GPU already) 9, new: stored by this request 8 | – | – | 2048 |
| P | 4 | 0 | 1087 | 1072 | 1024 | 0 / 0 | 15 | held (on GPU already) 4 | – | – | 0 |
| D1 | 4 | 0 | 1087 | 1072 | 1024 | 0 / 15 | 0 | held (on GPU already) 4 | – | – | 0 |
| P | 4 | 1 | 2301 | 1120 | 1024 | 0 / 0 | 1181 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 2301 | 1120 | 1024 | 0 / 1181 | 0 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| P | 4 | 2 | 3149 | 2288 | 2048 | 0 / 0 | 861 | held (on GPU already) 8, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 3149 | 1120 | 1024 | 0 / 2029 | 0 | held (on GPU already) 4, new: stored by this request 8 | – | – | 2048 |
| P | 4 | 3 | 4481 | 3136 | 3072 | 0 / 0 | 1345 | held (on GPU already) 12, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 4481 | 1200 | 1024 | 0 / 3281 | 0 | held (on GPU already) 4, new: stored by this request 13 | – | – | 3328 |
| P | 4 | 4 | 5667 | 4480 | 4352 | 0 / 0 | 1187 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 5667 | 2416 | 2304 | 0 / 3251 | 0 | held (on GPU already) 9, new: stored by this request 13 | – | – | 3328 |
| P | 4 | 5 | 6878 | 5664 | 5632 | 0 / 0 | 1214 | held (on GPU already) 22, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 5 | 6878 | 3264 | 3072 | 0 / 3614 | 0 | held (on GPU already) 12, new: stored by this request 14 | – | – | 3584 |

## One chunk's life: 5e868214004923dc

Conversation 1, turn 0, tokens 0-256 (system, doc 850931827_11086-12414-0-1328); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0069 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 928763b8) |
| 9.6071 | P looked it up (turn 0): new: stored by this request |
| 9.6105 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request ee2de12d) |
| 9.6108 | P's connector planned the store |
| 9.6299 | P's worker sent it to the server |
| 9.6312 | the server bound the chunk to its tokens (mp.tokens) |
| 9.6315 | server memory (L1) reserved |
| 9.7003 | eviction order refreshed (a touch: no I/O) |
| 9.7052 | written to memory (L1) |
| 9.8632 | written to disk (L2) |
| 9.8639 | removed from memory (L1) |
| 11.3621 | P looked it up (turn 1): held (on GPU already) |
| 11.3622 | eviction order refreshed (a touch: no I/O) |
| 11.4902 | eviction order refreshed (a touch: no I/O) |
| 13.4321 | P looked it up (turn 2): held (on GPU already) |
| 13.4323 | eviction order refreshed (a touch: no I/O) |
| 13.5509 | eviction order refreshed (a touch: no I/O) |
| 15.7568 | P looked it up (turn 3): held (on GPU already) |
| 15.7570 | eviction order refreshed (a touch: no I/O) |
| 15.8632 | eviction order refreshed (a touch: no I/O) |
| 18.0547 | P looked it up (turn 4): held (on GPU already) |
| 18.0549 | eviction order refreshed (a touch: no I/O) |
| 18.2210 | eviction order refreshed (a touch: no I/O) |
| 20.4839 | P looked it up (turn 5): held (on GPU already) |
| 20.4841 | eviction order refreshed (a touch: no I/O) |
| 20.6316 | eviction order refreshed (a touch: no I/O) |
| 22.9310 | P looked it up (turn 6): held (on GPU already) |
| 22.9313 | eviction order refreshed (a touch: no I/O) |
| 23.1028 | eviction order refreshed (a touch: no I/O) |
| 53.3583 | P looked it up (turn 0): held (on GPU already) |
| 53.3585 | eviction order refreshed (a touch: no I/O) |
| 53.3788 | eviction order refreshed (a touch: no I/O) |
| 54.8894 | P looked it up (turn 1): held (on GPU already) |
| 54.8895 | eviction order refreshed (a touch: no I/O) |
| 55.0002 | eviction order refreshed (a touch: no I/O) |
| 56.7146 | P looked it up (turn 2): held (on GPU already) |
| 56.7148 | eviction order refreshed (a touch: no I/O) |
| 56.8073 | eviction order refreshed (a touch: no I/O) |
| 58.6896 | P looked it up (turn 3): held (on GPU already) |
| 58.6898 | eviction order refreshed (a touch: no I/O) |
| 58.8331 | eviction order refreshed (a touch: no I/O) |
| 60.9937 | P looked it up (turn 4): held (on GPU already) |
| 60.9939 | eviction order refreshed (a touch: no I/O) |
| 61.1278 | eviction order refreshed (a touch: no I/O) |
| 63.3210 | P looked it up (turn 5): held (on GPU already) |
| 63.3212 | eviction order refreshed (a touch: no I/O) |
| 63.4631 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0808 | P answered (stored) |
| 0.1006 | D1 took the request |
| 0.1039 | the server got its lookup |
| 0.1067 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1079 | its connector had the lookup's answer |
| 0.3967 | the first token reached the proxy |
| 1.7647 | finished |
| 1.7676 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
