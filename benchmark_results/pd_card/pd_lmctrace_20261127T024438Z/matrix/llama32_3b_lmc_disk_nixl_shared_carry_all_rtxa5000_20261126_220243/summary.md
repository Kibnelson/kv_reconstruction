# LMCache lifecycle: disk mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 9849 records from 5 processes (D1 2114, D2 2502, D3 2613, LMC 2073, P 547); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 33 | 33 |
| prompt tokens | 117390 | 117390 |
| from the engine's own GPU prefix cache | 87520 | 45024 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 72366 |
| computed | 29870 | 0 |
| stored to LMCache (tokens) | 28928 | – |

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

- D requests that read from P: 33, 72366 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 411.050 ms, p90 659.680 ms
- P held its blocks for D from its request's end to D's read done: median 525.700 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.618 GB/s (the KV it read over its time)
- the bytes: 220 of 4541 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 2; DIFFERS: D1 block 43 from P's 3; DIFFERS: D1 block 44 from P's 4; DIFFERS: D1 block 45 from P's 5; DIFFERS: D1 block 46 from P's 6; DIFFERS: D1 block 47 from P's 7; DIFFERS: D1 block 48 from P's 8
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 113 chunks (by {'P': 113}); written to memory 113 (3317.7 MB); to disk 113; removed from memory 113
- pulled by D or P: 0 times, from –; never pulled: 113 (39 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 93; touches (eviction order refreshed, no I/O): 1377
- every lookup's outcome per chunk: {'new: stored by this request': 113, 'hit, not loaded': 278, 'held (on GPU already)': 493}
- pulls per stored chunk: {0: 113}
- median time from a chunk's last block computed on P to the chunk in memory: 108.470 ms; memory to disk: 228.390 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 93.080 ms, p90 144.250 ms.

## Checks

- chunk tokens match prompt: 113 of 113
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 113 of 113
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33
- disk mode on disk then out of memory: 113 of 113
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 93 of 93 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '93 objects, 2730491904 bytes'}
- disk mode files at end: 113 files, 3.32 GB

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
| P | 1 | 0 | 1087 | 16 | 0 | 0 / 0 | 1071 | new: stored by this request 4 | – | – | 1024 |
| D3 | 1 | 0 | 1087 | 16 | 1024 | 0 / 1071 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 1 | 1 | 2457 | 1072 | 1024 | 0 / 0 | 1385 | held (on GPU already) 4, new: stored by this request 5 | – | – | 1280 |
| D1 | 1 | 1 | 2457 | 16 | 2304 | 0 / 2441 | 0 | hit, not loaded 9 | – | – | 0 |
| P | 1 | 2 | 3622 | 2448 | 2304 | 0 / 0 | 1174 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 2 | 3622 | 16 | 3584 | 0 / 3606 | 0 | hit, not loaded 14 | – | – | 0 |
| P | 1 | 3 | 4627 | 3616 | 3584 | 0 / 0 | 1011 | held (on GPU already) 14, new: stored by this request 4 | – | – | 1024 |
| D3 | 1 | 3 | 4627 | 1200 | 4608 | 0 / 3427 | 0 | held (on GPU already) 4, hit, not loaded 14 | – | – | 0 |
| P | 1 | 4 | 6118 | 4624 | 4608 | 0 / 0 | 1494 | held (on GPU already) 18, new: stored by this request 5 | – | – | 1280 |
| D1 | 1 | 4 | 6118 | 2576 | 5888 | 0 / 3542 | 0 | held (on GPU already) 10, hit, not loaded 13 | – | – | 0 |
| P | 1 | 5 | 7369 | 6112 | 5888 | 0 / 0 | 1257 | held (on GPU already) 23, new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 5 | 7369 | 3744 | 7168 | 0 / 3625 | 0 | held (on GPU already) 14, hit, not loaded 14 | – | – | 0 |
| P | 1 | 6 | 8646 | 7360 | 7168 | 0 / 0 | 1286 | held (on GPU already) 28, new: stored by this request 5 | – | – | 1280 |
| D3 | 1 | 6 | 8646 | 4752 | 8448 | 0 / 3894 | 0 | held (on GPU already) 18, hit, not loaded 15 | – | – | 0 |
| P | 2 | 0 | 641 | 16 | 0 | 0 / 0 | 625 | new: stored by this request 2 | – | – | 512 |
| D1 | 2 | 0 | 641 | 16 | 512 | 0 / 625 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 2 | 1 | 1285 | 640 | 512 | 0 / 0 | 645 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D2 | 2 | 1 | 1285 | 16 | 1280 | 0 / 1269 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 2 | 2 | 2069 | 1280 | 1280 | 0 / 0 | 789 | held (on GPU already) 5, new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 2 | 2069 | 16 | 2048 | 0 / 2053 | 0 | hit, not loaded 8 | – | – | 0 |
| P | 2 | 3 | 2805 | 2064 | 2048 | 0 / 0 | 741 | held (on GPU already) 8, new: stored by this request 2 | – | – | 512 |
| D1 | 2 | 3 | 2805 | 640 | 2560 | 0 / 2165 | 0 | held (on GPU already) 2, hit, not loaded 8 | – | – | 0 |
| P | 2 | 4 | 3943 | 2800 | 2560 | 0 / 0 | 1143 | held (on GPU already) 10, new: stored by this request 5 | – | – | 1280 |
| D2 | 2 | 4 | 3943 | 1408 | 3840 | 0 / 2535 | 0 | held (on GPU already) 5, hit, not loaded 10 | – | – | 0 |
| P | 2 | 5 | 4668 | 3936 | 3840 | 0 / 0 | 732 | held (on GPU already) 15, new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 5 | 4668 | 2192 | 4608 | 0 / 2476 | 0 | held (on GPU already) 8, hit, not loaded 10 | – | – | 0 |
| P | 2 | 6 | 5622 | 4656 | 4608 | 0 / 0 | 966 | held (on GPU already) 18, new: stored by this request 3 | – | – | 768 |
| D1 | 2 | 6 | 5622 | 2928 | 5376 | 0 / 2694 | 0 | held (on GPU already) 11, hit, not loaded 10 | – | – | 0 |
| P | 2 | 7 | 6389 | 5616 | 5376 | 0 / 0 | 773 | held (on GPU already) 21, new: stored by this request 3 | – | – | 768 |
| D2 | 2 | 7 | 6389 | 4064 | 6144 | 0 / 2325 | 0 | held (on GPU already) 15, hit, not loaded 9 | – | – | 0 |
| P | 2 | 8 | 7569 | 6384 | 6144 | 0 / 0 | 1185 | held (on GPU already) 24, new: stored by this request 5 | – | – | 1280 |
| D3 | 2 | 8 | 7569 | 4784 | 7424 | 0 / 2785 | 0 | held (on GPU already) 18, hit, not loaded 11 | – | – | 0 |
| P | 3 | 0 | 633 | 16 | 0 | 0 / 0 | 617 | new: stored by this request 2 | – | – | 512 |
| D1 | 3 | 0 | 633 | 16 | 512 | 0 / 617 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 3 | 1 | 1339 | 624 | 512 | 0 / 0 | 715 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D2 | 3 | 1 | 1339 | 16 | 1280 | 0 / 1323 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 3 | 2 | 2211 | 1328 | 1280 | 0 / 0 | 883 | held (on GPU already) 5, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 2 | 2211 | 16 | 2048 | 0 / 2195 | 0 | hit, not loaded 8 | – | – | 0 |
| P | 3 | 3 | 3136 | 2208 | 2048 | 0 / 0 | 928 | held (on GPU already) 8, new: stored by this request 4 | – | – | 1024 |
| D1 | 3 | 3 | 3136 | 736 | 3072 | 0 / 2400 | 0 | held (on GPU already) 2, hit, not loaded 10 | – | – | 0 |
| P | 3 | 4 | 3813 | 3136 | 3072 | 0 / 0 | 677 | held (on GPU already) 12, new: stored by this request 2 | – | – | 512 |
| D2 | 3 | 4 | 3813 | 1440 | 3584 | 0 / 2373 | 0 | held (on GPU already) 5, hit, not loaded 9 | – | – | 0 |
| P | 3 | 5 | 4549 | 3808 | 3584 | 0 / 0 | 741 | held (on GPU already) 14, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 5 | 4549 | 2336 | 4352 | 0 / 2213 | 0 | held (on GPU already) 9, hit, not loaded 8 | – | – | 0 |
| P | 4 | 0 | 1087 | 1072 | 1024 | 0 / 0 | 15 | held (on GPU already) 4 | – | – | 0 |
| D1 | 4 | 0 | 1087 | 1072 | 1024 | 0 / 15 | 0 | held (on GPU already) 4 | – | – | 0 |
| P | 4 | 1 | 2301 | 1120 | 1024 | 0 / 0 | 1181 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 2301 | 1120 | 2048 | 0 / 1181 | 0 | held (on GPU already) 4, hit, not loaded 4 | – | – | 0 |
| P | 4 | 2 | 3149 | 2288 | 2048 | 0 / 0 | 861 | held (on GPU already) 8, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 3149 | 1120 | 3072 | 0 / 2029 | 0 | held (on GPU already) 4, hit, not loaded 8 | – | – | 0 |
| P | 4 | 3 | 4481 | 3136 | 3072 | 0 / 0 | 1345 | held (on GPU already) 12, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 4481 | 1200 | 4352 | 0 / 3281 | 0 | held (on GPU already) 4, hit, not loaded 13 | – | – | 0 |
| P | 4 | 4 | 5667 | 4480 | 4352 | 0 / 0 | 1187 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 5667 | 2416 | 5632 | 0 / 3251 | 0 | held (on GPU already) 9, hit, not loaded 13 | – | – | 0 |
| P | 4 | 5 | 6878 | 5664 | 5632 | 0 / 0 | 1214 | held (on GPU already) 22, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 5 | 6878 | 3264 | 6656 | 0 / 3614 | 0 | held (on GPU already) 12, hit, not loaded 14 | – | – | 0 |

## One chunk's life: 5e868214004923dc

Conversation 1, turn 0, tokens 0-256 (system, doc 850931827_11086-12414-0-1328); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0066 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 4f34adaf) |
| 9.6971 | P looked it up (turn 0): new: stored by this request |
| 9.7025 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request 1e5e64df) |
| 9.7028 | P's connector planned the store |
| 9.7209 | P's worker sent it to the server |
| 9.7223 | the server bound the chunk to its tokens (mp.tokens) |
| 9.7228 | server memory (L1) reserved |
| 9.7920 | eviction order refreshed (a touch: no I/O) |
| 9.7976 | written to memory (L1) |
| 9.7993 | D3 looked it up (turn 0): hit, not loaded |
| 9.9387 | written to disk (L2) |
| 9.9392 | removed from memory (L1) |
| 11.4153 | eviction order refreshed (a touch: no I/O) |
| 11.4332 | P looked it up (turn 1): held (on GPU already) |
| 11.4333 | eviction order refreshed (a touch: no I/O) |
| 11.5622 | eviction order refreshed (a touch: no I/O) |
| 11.5747 | D1 looked it up (turn 1): hit, not loaded |
| 11.6574 | read from disk (L2) |
| 13.5709 | eviction order refreshed (a touch: no I/O) |
| 13.5833 | P looked it up (turn 2): held (on GPU already) |
| 13.5834 | eviction order refreshed (a touch: no I/O) |
| 13.7016 | eviction order refreshed (a touch: no I/O) |
| 13.7147 | D2 looked it up (turn 2): hit, not loaded |
| 15.9970 | eviction order refreshed (a touch: no I/O) |
| 16.0154 | P looked it up (turn 3): held (on GPU already) |
| 16.0156 | eviction order refreshed (a touch: no I/O) |
| 16.1205 | eviction order refreshed (a touch: no I/O) |
| 16.1336 | D3 looked it up (turn 3): held (on GPU already) |
| 16.1338 | eviction order refreshed (a touch: no I/O) |
| 18.3964 | eviction order refreshed (a touch: no I/O) |
| 18.4147 | P looked it up (turn 4): held (on GPU already) |
| 18.4149 | eviction order refreshed (a touch: no I/O) |
| 18.5823 | eviction order refreshed (a touch: no I/O) |
| 18.5998 | D1 looked it up (turn 4): held (on GPU already) |
| 18.5999 | eviction order refreshed (a touch: no I/O) |
| 20.8845 | eviction order refreshed (a touch: no I/O) |
| 20.9099 | P looked it up (turn 5): held (on GPU already) |
| 20.9100 | eviction order refreshed (a touch: no I/O) |
| 21.0544 | eviction order refreshed (a touch: no I/O) |
| 21.0729 | D2 looked it up (turn 5): held (on GPU already) |
| 21.0730 | eviction order refreshed (a touch: no I/O) |
| 23.4394 | eviction order refreshed (a touch: no I/O) |
| 23.4576 | P looked it up (turn 6): held (on GPU already) |
| 23.4578 | eviction order refreshed (a touch: no I/O) |
| 23.6303 | eviction order refreshed (a touch: no I/O) |
| 23.6574 | D3 looked it up (turn 6): held (on GPU already) |
| 23.6575 | eviction order refreshed (a touch: no I/O) |
| 26.0862 | eviction order refreshed (a touch: no I/O) |
| 54.7861 | P looked it up (turn 0): held (on GPU already) |
| 54.7863 | eviction order refreshed (a touch: no I/O) |
| 54.8067 | eviction order refreshed (a touch: no I/O) |
| 54.8148 | D1 looked it up (turn 0): held (on GPU already) |
| 54.8149 | eviction order refreshed (a touch: no I/O) |
| 56.2566 | eviction order refreshed (a touch: no I/O) |
| 56.2689 | P looked it up (turn 1): held (on GPU already) |
| 56.2690 | eviction order refreshed (a touch: no I/O) |
| 56.3782 | eviction order refreshed (a touch: no I/O) |
| 56.3896 | D2 looked it up (turn 1): held (on GPU already) |
| 56.3897 | eviction order refreshed (a touch: no I/O) |
| 58.0668 | eviction order refreshed (a touch: no I/O) |
| 58.0857 | P looked it up (turn 2): held (on GPU already) |
| 58.0859 | eviction order refreshed (a touch: no I/O) |
| 58.1765 | eviction order refreshed (a touch: no I/O) |
| 58.1909 | D3 looked it up (turn 2): held (on GPU already) |
| 58.1911 | eviction order refreshed (a touch: no I/O) |
| 60.1235 | eviction order refreshed (a touch: no I/O) |
| 60.1399 | P looked it up (turn 3): held (on GPU already) |
| 60.1400 | eviction order refreshed (a touch: no I/O) |
| 60.2828 | eviction order refreshed (a touch: no I/O) |
| 60.2986 | D1 looked it up (turn 3): held (on GPU already) |
| 60.2988 | eviction order refreshed (a touch: no I/O) |
| 62.5009 | eviction order refreshed (a touch: no I/O) |
| 62.5238 | P looked it up (turn 4): held (on GPU already) |
| 62.5240 | eviction order refreshed (a touch: no I/O) |
| 62.6572 | eviction order refreshed (a touch: no I/O) |
| 62.6746 | D2 looked it up (turn 4): held (on GPU already) |
| 62.6748 | eviction order refreshed (a touch: no I/O) |
| 64.9257 | eviction order refreshed (a touch: no I/O) |
| 64.9455 | P looked it up (turn 5): held (on GPU already) |
| 64.9457 | eviction order refreshed (a touch: no I/O) |
| 65.0846 | eviction order refreshed (a touch: no I/O) |
| 65.1059 | D3 looked it up (turn 5): held (on GPU already) |
| 65.1060 | eviction order refreshed (a touch: no I/O) |
| 67.4362 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0023 | the proxy received the turn |
| 0.0822 | P answered (stored) |
| 0.0924 | D1 took the request |
| 0.0952 | the server got its lookup |
| 0.0961 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.0979 | its connector had the lookup's answer |
| 0.3893 | the first token reached the proxy |
| 1.7568 | finished |
| 1.7602 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
