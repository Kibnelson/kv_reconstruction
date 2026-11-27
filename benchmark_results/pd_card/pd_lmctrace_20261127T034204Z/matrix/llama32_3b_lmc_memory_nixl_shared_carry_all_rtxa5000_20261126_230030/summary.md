# LMCache lifecycle: memory mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 14697 records from 5 processes (D1 3416, D2 3536, D3 3566, LMC 3080, P 1099); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 258844 | 258844 |
| from the engine's own GPU prefix cache | 195248 | 101344 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 157500 |
| computed | 63596 | 0 |
| stored to LMCache (tokens) | 61952 | 1024 |

**D's answers against one plain engine (no P/D, no connector: vLLM's own KV layout)** (the same prompts, temperature 0; `local_check.json`): 62 of 69 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; the engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`
- turn 2/1, from token 32: D `' A catchbasin is a small reservoir that collects water and debris from the street and prevents it fr'`; the engine `' A catchbasin is a small reservoir that collects water and debris from the street and prevents it fr'`
- turn 4/2, from token 17: D `' guinea pig appearance or body is a small mammal with a short tail and large ears.\nUser: more about '`; the engine `' guinea pig appearance or body is a small mammal with a short tail and large head.\nUser: more about '`
- turn 5/2, from token 3: D `' The moon is about 4.5 billion years old. This is based on the age of the solar system, which is est'`; the engine `' The moon is approximately 4.5 billion years old. This is based on the age of the solar system, whic'`

**D's answers against one engine with NIXL's KV layout (NixlConnector set, unused; no P/D)** (the same prompts, temperature 0; `local_check_nixl.json`): 62 of 69 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; the engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`
- turn 2/1, from token 32: D `' A catchbasin is a small reservoir that collects water and debris from the street and prevents it fr'`; the engine `' A catchbasin is a small reservoir that collects water and debris from the street and prevents it fr'`
- turn 4/2, from token 17: D `' guinea pig appearance or body is a small mammal with a short tail and large ears.\nUser: more about '`; the engine `' guinea pig appearance or body is a small mammal with a short tail and large head.\nUser: more about '`
- turn 5/2, from token 3: D `' The moon is about 4.5 billion years old. This is based on the age of the solar system, which is est'`; the engine `' The moon is approximately 4.5 billion years old. This is based on the age of the solar system, whic'`

## NIXL

- D requests that read from P: 69, 157500 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 445.590 ms, p90 626.580 ms
- P held its blocks for D from its request's end to D's read done: median 466.970 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.618 GB/s (the KV it read over its time)
- the bytes: 9879 of 9879 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 242 chunks (by {'P': 242}); written to memory 242 (7105.2 MB); to disk 0; removed from memory 47
- pulled by D or P: 0 times, from –; never pulled: 242 (81 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 3069
- every lookup's outcome per chunk: {'new: stored by this request': 242, 'hit, not loaded': 605, 'held (on GPU already)': 1109, 'miss: not yet stored': 4}
- pulls per stored chunk: {0: 242}
- median time from a chunk's last block computed on P to the chunk in memory: 104.410 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 94.120 ms, p90 135.060 ms.

## Checks

- chunk tokens match prompt: 242 of 242
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 242 of 242
- d answers same as one engine: 62 of 69
- d answers same as one engine nixl layout: 62 of 69

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
| P | 4 | 1 | 2301 | 1216 | 1024 | 0 / 0 | 1085 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 2301 | 1216 | 2048 | 0 / 1085 | 0 | held (on GPU already) 4, hit, not loaded 4 | – | – | 0 |
| P | 4 | 2 | 3149 | 2288 | 2048 | 0 / 0 | 861 | held (on GPU already) 8, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 3149 | 1216 | 3072 | 0 / 1933 | 0 | held (on GPU already) 4, hit, not loaded 8 | – | – | 0 |
| P | 4 | 3 | 4481 | 3136 | 3072 | 0 / 0 | 1345 | held (on GPU already) 12, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 4481 | 1216 | 4352 | 0 / 3265 | 0 | held (on GPU already) 4, hit, not loaded 13 | – | – | 0 |
| P | 4 | 4 | 5667 | 4480 | 4352 | 0 / 0 | 1187 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 5667 | 2416 | 5632 | 0 / 3251 | 0 | held (on GPU already) 9, hit, not loaded 13 | – | – | 0 |
| P | 4 | 5 | 6878 | 5664 | 5632 | 0 / 0 | 1214 | held (on GPU already) 22, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 5 | 6878 | 3264 | 6656 | 0 / 3614 | 0 | held (on GPU already) 12, hit, not loaded 14 | – | – | 0 |
| P | 5 | 0 | 1035 | 16 | 0 | 0 / 0 | 1019 | new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 0 | 1035 | 16 | 1024 | 0 / 1019 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 5 | 1 | 1944 | 1024 | 1024 | 0 / 0 | 920 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D2 | 5 | 1 | 1944 | 16 | 1792 | 0 / 1928 | 0 | hit, not loaded 7 | – | – | 0 |
| P | 5 | 2 | 2959 | 1936 | 1792 | 0 / 0 | 1023 | held (on GPU already) 7, new: stored by this request 4 | – | – | 1024 |
| D3 | 5 | 2 | 2959 | 16 | 2816 | 0 / 2943 | 0 | hit, not loaded 11 | – | – | 0 |
| P | 5 | 3 | 3848 | 2944 | 2816 | 0 / 0 | 904 | held (on GPU already) 11, new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 3 | 3848 | 1152 | 3840 | 0 / 2696 | 0 | held (on GPU already) 4, hit, not loaded 11 | – | – | 0 |
| P | 5 | 4 | 4996 | 3840 | 3840 | 0 / 0 | 1156 | held (on GPU already) 15, new: stored by this request 4 | – | – | 1024 |
| D2 | 5 | 4 | 4996 | 2064 | 4864 | 0 / 2932 | 0 | held (on GPU already) 8, hit, not loaded 11 | – | – | 0 |
| P | 5 | 5 | 5662 | 4992 | 4864 | 0 / 0 | 670 | held (on GPU already) 19, new: stored by this request 3 | – | – | 768 |
| D3 | 5 | 5 | 5662 | 3072 | 5632 | 0 / 2590 | 0 | held (on GPU already) 12, hit, not loaded 10 | – | – | 0 |
| P | 6 | 0 | 814 | 16 | 0 | 0 / 0 | 798 | new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 0 | 814 | 16 | 768 | 0 / 798 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 6 | 1 | 1585 | 800 | 768 | 0 / 0 | 785 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 1 | 1585 | 16 | 1536 | 0 / 1569 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 6 | 2 | 2124 | 1584 | 1536 | 0 / 0 | 540 | held (on GPU already) 6, new: stored by this request 2 | – | – | 512 |
| D3 | 6 | 2 | 2124 | 16 | 2048 | 0 / 2108 | 0 | hit, not loaded 8 | – | – | 0 |
| P | 6 | 3 | 3426 | 2112 | 2048 | 0 / 0 | 1314 | held (on GPU already) 8, new: stored by this request 5 | – | – | 1280 |
| D1 | 6 | 3 | 3426 | 928 | 3328 | 0 / 2498 | 0 | held (on GPU already) 3, hit, not loaded 10 | – | – | 0 |
| P | 6 | 4 | 4297 | 3424 | 3328 | 0 / 0 | 873 | held (on GPU already) 13, new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 4 | 4297 | 1712 | 4096 | 0 / 2585 | 0 | held (on GPU already) 6, hit, not loaded 10 | – | – | 0 |
| P | 6 | 5 | 5161 | 4288 | 4096 | 0 / 0 | 873 | held (on GPU already) 16, new: stored by this request 4 | – | – | 1024 |
| D3 | 6 | 5 | 5161 | 2240 | 5120 | 0 / 2921 | 0 | held (on GPU already) 8, hit, not loaded 12 | – | – | 0 |
| P | 6 | 6 | 5949 | 5152 | 5120 | 0 / 0 | 797 | held (on GPU already) 20, new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 6 | 5949 | 3552 | 5888 | 0 / 2397 | 0 | held (on GPU already) 13, hit, not loaded 10 | – | – | 0 |
| P | 7 | 0 | 1136 | 16 | 0 | 0 / 0 | 1120 | new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 0 | 1136 | 16 | 1024 | 0 / 1120 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 7 | 1 | 1909 | 1136 | 1024 | 0 / 0 | 773 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 1 | 1909 | 16 | 1792 | 0 / 1893 | 0 | hit, not loaded 7 | – | – | 0 |
| P | 7 | 2 | 3115 | 1904 | 1792 | 0 / 0 | 1211 | held (on GPU already) 7, new: stored by this request 5 | – | – | 1280 |
| D1 | 7 | 2 | 3115 | 16 | 3072 | 0 / 3099 | 0 | hit, not loaded 12 | – | – | 0 |
| P | 7 | 3 | 4099 | 3104 | 3072 | 0 / 0 | 995 | held (on GPU already) 12, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 3 | 4099 | 1248 | 4096 | 0 / 2851 | 0 | held (on GPU already) 4, hit, not loaded 12 | – | – | 0 |
| P | 7 | 4 | 5352 | 4096 | 4096 | 0 / 0 | 1256 | held (on GPU already) 16, new: stored by this request 4 | – | – | 1024 |
| D3 | 7 | 4 | 5352 | 2032 | 5120 | 0 / 3320 | 0 | held (on GPU already) 7, hit, not loaded 13 | – | – | 0 |
| P | 7 | 5 | 6094 | 5344 | 5120 | 0 / 0 | 750 | held (on GPU already) 20, new: stored by this request 3 | – | – | 768 |
| D1 | 7 | 5 | 6094 | 3232 | 5888 | 0 / 2862 | 0 | held (on GPU already) 12, hit, not loaded 11 | – | – | 0 |
| P | 7 | 6 | 7005 | 6080 | 5888 | 0 / 0 | 925 | held (on GPU already) 23, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 6 | 7005 | 4224 | 6912 | 0 / 2781 | 0 | held (on GPU already) 16, hit, not loaded 11 | – | – | 0 |
| P | 7 | 7 | 7916 | 6992 | 6912 | 0 / 0 | 924 | held (on GPU already) 27, new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 7 | 7916 | 5472 | 7680 | 0 / 2444 | 0 | held (on GPU already) 21, hit, not loaded 9 | – | – | 0 |
| P | 8 | 0 | 1156 | 16 | 0 | 0 / 0 | 1140 | new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 0 | 1156 | 16 | 1024 | 0 / 1140 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 8 | 1 | 2338 | 1152 | 1024 | 0 / 0 | 1186 | held (on GPU already) 4, new: stored by this request 5 | – | – | 1280 |
| D2 | 8 | 1 | 2338 | 16 | 2304 | 0 / 2322 | 0 | hit, not loaded 9 | – | – | 0 |
| P | 8 | 2 | 3605 | 2336 | 2304 | 0 / 0 | 1269 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D3 | 8 | 2 | 3605 | 16 | 3584 | 0 / 3589 | 0 | hit, not loaded 14 | – | – | 0 |
| P | 8 | 3 | 4188 | 3600 | 3584 | 0 / 0 | 588 | held (on GPU already) 14, new: stored by this request 2 | – | – | 512 |
| D1 | 8 | 3 | 4188 | 1280 | 4096 | 0 / 2908 | 0 | held (on GPU already) 5, hit, not loaded 11 | – | – | 0 |
| P | 8 | 4 | 4833 | 4176 | 4096 | 0 / 0 | 657 | held (on GPU already) 16, new: stored by this request 2 | – | – | 512 |
| D2 | 8 | 4 | 4833 | 2464 | 4608 | 0 / 2369 | 0 | held (on GPU already) 9, hit, not loaded 9 | – | – | 0 |
| P | 8 | 5 | 5699 | 4832 | 4608 | 0 / 0 | 867 | held (on GPU already) 18, new: stored by this request 4 | – | – | 1024 |
| D3 | 8 | 5 | 5699 | 3728 | 5632 | 0 / 1971 | 0 | held (on GPU already) 14, hit, not loaded 8 | – | – | 0 |
| P | 8 | 6 | 6579 | 5696 | 5632 | 0 / 0 | 883 | held (on GPU already) 22, new: stored by this request 3 | – | – | 768 |
| D1 | 8 | 6 | 6579 | 4304 | 6400 | 0 / 2275 | 0 | held (on GPU already) 16, hit, not loaded 9 | – | – | 0 |
| P | 8 | 7 | 7624 | 6576 | 6400 | 0 / 0 | 1048 | held (on GPU already) 25, new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 7 | 7624 | 4960 | 7424 | 0 / 2664 | 0 | held (on GPU already) 19, hit, not loaded 10 | – | – | 0 |
| P | 9 | 0 | 464 | 16 | 0 | 0 / 0 | 448 | new: stored by this request 1 | – | – | 256 |
| D3 | 9 | 0 | 464 | 16 | 256 | 0 / 448 | 0 | hit, not loaded 1 | – | – | 0 |
| P | 9 | 1 | 1314 | 464 | 256 | 0 / 0 | 850 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D1 | 9 | 1 | 1314 | 16 | 256 | 0 / 1298 | 0 | hit, not loaded 1, miss: not yet stored 4 | – | – | 1024 |
| P | 9 | 2 | 2331 | 1312 | 1280 | 0 / 0 | 1019 | held (on GPU already) 5, new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 2 | 2331 | 16 | 2304 | 0 / 2315 | 0 | hit, not loaded 9 | – | – | 0 |
| P | 9 | 3 | 3627 | 2320 | 2304 | 0 / 0 | 1307 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D3 | 9 | 3 | 3627 | 576 | 3584 | 0 / 3051 | 0 | held (on GPU already) 2, hit, not loaded 12 | – | – | 0 |
| P | 9 | 4 | 4891 | 3616 | 3584 | 0 / 0 | 1275 | held (on GPU already) 14, new: stored by this request 5 | – | – | 1280 |
| D1 | 9 | 4 | 4891 | 1440 | 4864 | 0 / 3451 | 0 | held (on GPU already) 5, hit, not loaded 14 | – | – | 0 |
| P | 9 | 5 | 5845 | 4880 | 4864 | 0 / 0 | 965 | held (on GPU already) 19, new: stored by this request 3 | – | – | 768 |
| D2 | 9 | 5 | 5845 | 2448 | 5632 | 0 / 3397 | 0 | held (on GPU already) 9, hit, not loaded 13 | – | – | 0 |
| P | 9 | 6 | 6534 | 5840 | 5632 | 0 / 0 | 694 | held (on GPU already) 22, new: stored by this request 3 | – | – | 768 |
| D3 | 9 | 6 | 6534 | 3744 | 6400 | 0 / 2790 | 0 | held (on GPU already) 14, hit, not loaded 11 | – | – | 0 |

## One chunk's life: 5e868214004923dc

Conversation 1, turn 0, tokens 0-256 (system, doc 850931827_11086-12414-0-1328); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0048 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 206606fa) |
| 9.5677 | P looked it up (turn 0): new: stored by this request |
| 9.5704 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request fa46d21c) |
| 9.5707 | P's connector planned the store |
| 9.5892 | P's worker sent it to the server |
| 9.5905 | the server bound the chunk to its tokens (mp.tokens) |
| 9.5910 | server memory (L1) reserved |
| 9.6595 | eviction order refreshed (a touch: no I/O) |
| 9.6651 | written to memory (L1) |
| 9.6708 | D3 looked it up (turn 0): hit, not loaded |
| 11.2905 | eviction order refreshed (a touch: no I/O) |
| 11.3135 | P looked it up (turn 1): held (on GPU already) |
| 11.3136 | eviction order refreshed (a touch: no I/O) |
| 11.4399 | eviction order refreshed (a touch: no I/O) |
| 11.4517 | D1 looked it up (turn 1): hit, not loaded |
| 13.3648 | eviction order refreshed (a touch: no I/O) |
| 13.3862 | P looked it up (turn 2): held (on GPU already) |
| 13.3863 | eviction order refreshed (a touch: no I/O) |
| 13.5032 | eviction order refreshed (a touch: no I/O) |
| 13.5155 | D2 looked it up (turn 2): hit, not loaded |
| 15.6987 | eviction order refreshed (a touch: no I/O) |
| 15.7180 | P looked it up (turn 3): held (on GPU already) |
| 15.7182 | eviction order refreshed (a touch: no I/O) |
| 15.8213 | eviction order refreshed (a touch: no I/O) |
| 15.8366 | D3 looked it up (turn 3): held (on GPU already) |
| 15.8367 | eviction order refreshed (a touch: no I/O) |
| 17.9801 | eviction order refreshed (a touch: no I/O) |
| 18.0033 | P looked it up (turn 4): held (on GPU already) |
| 18.0035 | eviction order refreshed (a touch: no I/O) |
| 18.1656 | eviction order refreshed (a touch: no I/O) |
| 18.1868 | D1 looked it up (turn 4): held (on GPU already) |
| 18.1869 | eviction order refreshed (a touch: no I/O) |
| 20.3919 | eviction order refreshed (a touch: no I/O) |
| 20.4203 | P looked it up (turn 5): held (on GPU already) |
| 20.4205 | eviction order refreshed (a touch: no I/O) |
| 20.5644 | eviction order refreshed (a touch: no I/O) |
| 20.5888 | D2 looked it up (turn 5): held (on GPU already) |
| 20.5890 | eviction order refreshed (a touch: no I/O) |
| 22.8325 | eviction order refreshed (a touch: no I/O) |
| 22.8506 | P looked it up (turn 6): held (on GPU already) |
| 22.8508 | eviction order refreshed (a touch: no I/O) |
| 23.0190 | eviction order refreshed (a touch: no I/O) |
| 23.0492 | D3 looked it up (turn 6): held (on GPU already) |
| 23.0494 | eviction order refreshed (a touch: no I/O) |
| 25.3668 | eviction order refreshed (a touch: no I/O) |
| 53.1020 | P looked it up (turn 0): held (on GPU already) |
| 53.1022 | eviction order refreshed (a touch: no I/O) |
| 53.1225 | eviction order refreshed (a touch: no I/O) |
| 53.1306 | D1 looked it up (turn 0): held (on GPU already) |
| 53.1307 | eviction order refreshed (a touch: no I/O) |
| 54.6155 | eviction order refreshed (a touch: no I/O) |
| 54.6286 | P looked it up (turn 1): held (on GPU already) |
| 54.6287 | eviction order refreshed (a touch: no I/O) |
| 54.7273 | eviction order refreshed (a touch: no I/O) |
| 54.7377 | D2 looked it up (turn 1): held (on GPU already) |
| 54.7378 | eviction order refreshed (a touch: no I/O) |
| 56.3952 | eviction order refreshed (a touch: no I/O) |
| 56.4102 | P looked it up (turn 2): held (on GPU already) |
| 56.4103 | eviction order refreshed (a touch: no I/O) |
| 56.4972 | eviction order refreshed (a touch: no I/O) |
| 56.5097 | D3 looked it up (turn 2): held (on GPU already) |
| 56.5097 | eviction order refreshed (a touch: no I/O) |
| 58.3400 | eviction order refreshed (a touch: no I/O) |
| 58.3572 | P looked it up (turn 3): held (on GPU already) |
| 58.3573 | eviction order refreshed (a touch: no I/O) |
| 58.5008 | eviction order refreshed (a touch: no I/O) |
| 58.5154 | D1 looked it up (turn 3): held (on GPU already) |
| 58.5154 | eviction order refreshed (a touch: no I/O) |
| 60.6289 | eviction order refreshed (a touch: no I/O) |
| 60.6512 | P looked it up (turn 4): held (on GPU already) |
| 60.6514 | eviction order refreshed (a touch: no I/O) |
| 60.7838 | eviction order refreshed (a touch: no I/O) |
| 60.8006 | D2 looked it up (turn 4): held (on GPU already) |
| 60.8007 | eviction order refreshed (a touch: no I/O) |
| 62.9379 | eviction order refreshed (a touch: no I/O) |
| 62.9582 | P looked it up (turn 5): held (on GPU already) |
| 62.9584 | eviction order refreshed (a touch: no I/O) |
| 63.0976 | eviction order refreshed (a touch: no I/O) |
| 63.1171 | D3 looked it up (turn 5): held (on GPU already) |
| 63.1172 | eviction order refreshed (a touch: no I/O) |
| 65.3517 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0810 | P answered (stored) |
| 0.1027 | D1 took the request |
| 0.1056 | the server got its lookup |
| 0.1066 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1070 | its connector had the lookup's answer |
| 0.3884 | the first token reached the proxy |
| 1.7548 | finished |
| 1.7577 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
