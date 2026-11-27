# LMCache lifecycle: disk mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 16245 records from 5 processes (D1 3682, D2 3613, D3 3834, LMC 3968, P 1148); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 87764 | 87764 |
| from the engine's own GPU prefix cache | 23248 | 10096 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 77668 |
| computed | 64516 | 0 |
| stored to LMCache (tokens) | 62208 | 3840 |

**D's answers against one plain engine (no P/D, no connector: vLLM's own KV layout)** (the same prompts, temperature 0; `local_check.json`): 52 of 69 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/3, from token 40: D `' I don\'t know. I don\'t know what you mean by "air-to-air". I don\'t know what you mean by "ground-to-'`; the engine `' I don\'t know. I don\'t know what you mean by "air-to-air". I don\'t know what you mean by "ground-to-'`
- turn 1/5, from token 22: D `' A human guinea pig is a person who is used as a test subject in medical experiments . The term is o'`; the engine `' A human guinea pig is a person who is used as a test subject in medical experiments . The term is a'`
- turn 2/1, from token 17: D `' A catchbasin is a small reservoir that collects water and debris from the street and directs it int'`; the engine `' A catchbasin is a small reservoir that collects water and debris from the street and prevents the p'`
- turn 2/5, from token 14: D `' The storm drain works in the other parts of the world by collecting the rainwater and directing it '`; the engine `' The storm drain works in the other parts of the world by collecting the water from the streets, roa'`

**D's answers against one engine with NIXL's KV layout (NixlConnector set, unused; no P/D)** (the same prompts, temperature 0; `local_check_nixl.json`): 52 of 69 the same, token for token; the first that differs: turn 0/0.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; the engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/3, from token 40: D `' I don\'t know. I don\'t know what you mean by "air-to-air". I don\'t know what you mean by "ground-to-'`; the engine `' I don\'t know. I don\'t know what you mean by "air-to-air". I don\'t know what you mean by "ground-to-'`
- turn 1/5, from token 22: D `' A human guinea pig is a person who is used as a test subject in medical experiments . The term is o'`; the engine `' A human guinea pig is a person who is used as a test subject in medical experiments . The term is a'`
- turn 2/1, from token 17: D `' A catchbasin is a small reservoir that collects water and debris from the street and directs it int'`; the engine `' A catchbasin is a small reservoir that collects water and debris from the street and prevents the p'`
- turn 2/5, from token 14: D `' The storm drain works in the other parts of the world by collecting the rainwater and directing it '`; the engine `' The storm drain works in the other parts of the world by collecting the water from the streets, roa'`

## NIXL

- D requests that read from P: 69, 77668 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 213.020 ms, p90 286.530 ms
- P held its blocks for D from its request's end to D's read done: median 244.860 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.607 GB/s (the KV it read over its time)
- the bytes: 4883 of 4883 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 243 chunks (by {'P': 243}); written to memory 243 (7134.5 MB); to disk 243; removed from memory 243
- pulled by D or P: 0 times, from –; never pulled: 243 (87 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 26; touches (eviction order refreshed, no I/O): 699
- every lookup's outcome per chunk: {'new: stored by this request': 243, 'hit, not loaded': 271, 'held (on GPU already)': 85, 'miss: not yet stored': 15}
- pulls per stored chunk: {0: 243}
- median time from a chunk's last block computed on P to the chunk in memory: 94.590 ms; memory to disk: 149.370 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 82.820 ms, p90 108.750 ms.

## Checks

- chunk tokens match prompt: 243 of 243
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 243 of 243
- d answers same as one engine: 52 of 69
- d answers same as one engine nixl layout: 52 of 69
- disk mode on disk then out of memory: 243 of 243
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 26 of 26 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '26 objects, 763363328 bytes'}
- disk mode files at end: 243 files, 7.13 GB

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
| P | 1 | 0 | 1087 | 16 | 0 | 0 / 0 | 1071 | new: stored by this request 4 | – | – | 1024 |
| D3 | 1 | 0 | 1087 | 16 | 1024 | 0 / 1071 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 1 | 1 | 1412 | 16 | 0 | 0 / 0 | 1396 | new: stored by this request 5 | – | – | 1280 |
| D1 | 1 | 1 | 1412 | 16 | 1280 | 0 / 1396 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 1 | 2 | 1347 | 160 | 0 | 0 / 0 | 1187 | new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 2 | 1347 | 16 | 0 | 0 / 1331 | 0 | miss: not yet stored 5 | – | – | 1280 |
| P | 1 | 3 | 1221 | 304 | 256 | 0 / 0 | 917 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D3 | 1 | 3 | 1221 | 16 | 1024 | 0 / 1205 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 1 | 4 | 1850 | 336 | 256 | 0 / 0 | 1514 | held (on GPU already) 1, new: stored by this request 6 | – | – | 1536 |
| D1 | 1 | 4 | 1850 | 160 | 1792 | 0 / 1690 | 0 | hit, not loaded 7 | – | – | 0 |
| P | 1 | 5 | 1759 | 480 | 256 | 0 / 0 | 1279 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 1 | 5 | 1759 | 304 | 1536 | 0 / 1455 | 0 | held (on GPU already) 1, hit, not loaded 5 | – | – | 0 |
| P | 1 | 6 | 1923 | 624 | 512 | 0 / 0 | 1299 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 1 | 6 | 1923 | 336 | 1792 | 0 / 1587 | 0 | held (on GPU already) 1, hit, not loaded 6 | – | – | 0 |
| P | 2 | 0 | 641 | 16 | 0 | 0 / 0 | 625 | new: stored by this request 2 | – | – | 512 |
| D1 | 2 | 0 | 641 | 16 | 512 | 0 / 625 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 2 | 1 | 687 | 16 | 0 | 0 / 0 | 671 | new: stored by this request 2 | – | – | 512 |
| D2 | 2 | 1 | 687 | 16 | 512 | 0 / 671 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 2 | 2 | 851 | 48 | 0 | 0 / 0 | 803 | new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 2 | 851 | 16 | 768 | 0 / 835 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 2 | 3 | 957 | 192 | 0 | 0 / 0 | 765 | new: stored by this request 3 | – | – | 768 |
| D1 | 2 | 3 | 957 | 16 | 768 | 0 / 941 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 2 | 4 | 1516 | 336 | 256 | 0 / 0 | 1180 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 2 | 4 | 1516 | 48 | 1280 | 0 / 1468 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 2 | 5 | 1270 | 496 | 256 | 0 / 0 | 774 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 5 | 1270 | 192 | 1024 | 0 / 1078 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 2 | 6 | 1646 | 672 | 512 | 0 / 0 | 974 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D1 | 2 | 6 | 1646 | 336 | 1536 | 0 / 1310 | 0 | held (on GPU already) 1, hit, not loaded 5 | – | – | 0 |
| P | 2 | 7 | 1608 | 816 | 768 | 0 / 0 | 792 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 2 | 7 | 1608 | 496 | 1536 | 0 / 1112 | 0 | held (on GPU already) 1, hit, not loaded 5 | – | – | 0 |
| P | 2 | 8 | 2175 | 960 | 768 | 0 / 0 | 1215 | held (on GPU already) 3, new: stored by this request 5 | – | – | 1280 |
| D3 | 2 | 8 | 2175 | 672 | 2048 | 0 / 1503 | 0 | held (on GPU already) 2, hit, not loaded 6 | – | – | 0 |
| P | 3 | 0 | 633 | 16 | 0 | 0 / 0 | 617 | new: stored by this request 2 | – | – | 512 |
| D1 | 3 | 0 | 633 | 16 | 512 | 0 / 617 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 3 | 1 | 750 | 16 | 0 | 0 / 0 | 734 | new: stored by this request 2 | – | – | 512 |
| D2 | 3 | 1 | 750 | 16 | 512 | 0 / 734 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 3 | 2 | 1060 | 144 | 0 | 0 / 0 | 916 | new: stored by this request 4 | – | – | 1024 |
| D3 | 3 | 2 | 1060 | 16 | 1024 | 0 / 1044 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 3 | 3 | 1237 | 288 | 256 | 0 / 0 | 949 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D1 | 3 | 3 | 1237 | 16 | 1024 | 0 / 1221 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 3 | 4 | 1132 | 432 | 256 | 0 / 0 | 700 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D2 | 3 | 4 | 1132 | 144 | 1024 | 0 / 988 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 3 | 5 | 1336 | 576 | 512 | 0 / 0 | 760 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 5 | 1336 | 288 | 1280 | 0 / 1048 | 0 | held (on GPU already) 1, hit, not loaded 4 | – | – | 0 |
| P | 4 | 0 | 1087 | 1072 | 1024 | 0 / 0 | 15 | held (on GPU already) 4 | – | – | 0 |
| D1 | 4 | 0 | 1087 | 16 | 1024 | 0 / 1071 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 4 | 1 | 1256 | 176 | 0 | 0 / 0 | 1080 | new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 1256 | 160 | 1024 | 0 / 1096 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 4 | 2 | 1041 | 160 | 0 | 0 / 0 | 881 | new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 1041 | 160 | 1024 | 0 / 881 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 4 | 3 | 1667 | 320 | 256 | 0 / 0 | 1347 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 1667 | 160 | 1536 | 0 / 1507 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 4 | 4 | 1676 | 464 | 256 | 0 / 0 | 1212 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 1676 | 160 | 1536 | 0 / 1516 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 4 | 5 | 1841 | 608 | 512 | 0 / 0 | 1233 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 4 | 5 | 1841 | 320 | 1792 | 0 / 1521 | 0 | held (on GPU already) 1, hit, not loaded 6 | – | – | 0 |
| P | 5 | 0 | 1035 | 16 | 0 | 0 / 0 | 1019 | new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 0 | 1035 | 16 | 1024 | 0 / 1019 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 5 | 1 | 950 | 16 | 0 | 0 / 0 | 934 | new: stored by this request 3 | – | – | 768 |
| D2 | 5 | 1 | 950 | 16 | 768 | 0 / 934 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 5 | 2 | 1199 | 160 | 0 | 0 / 0 | 1039 | new: stored by this request 4 | – | – | 1024 |
| D3 | 5 | 2 | 1199 | 16 | 1024 | 0 / 1183 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 5 | 3 | 1212 | 304 | 256 | 0 / 0 | 908 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D1 | 5 | 3 | 1212 | 16 | 1024 | 0 / 1196 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 5 | 4 | 1613 | 448 | 256 | 0 / 0 | 1165 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 5 | 4 | 1613 | 160 | 1536 | 0 / 1453 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 5 | 5 | 1272 | 592 | 512 | 0 / 0 | 680 | held (on GPU already) 2, new: stored by this request 2 | – | – | 512 |
| D3 | 5 | 5 | 1272 | 304 | 1024 | 0 / 968 | 0 | held (on GPU already) 1, hit, not loaded 3 | – | – | 0 |
| P | 6 | 0 | 814 | 16 | 0 | 0 / 0 | 798 | new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 0 | 814 | 16 | 768 | 0 / 798 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 6 | 1 | 816 | 16 | 0 | 0 / 0 | 800 | new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 1 | 816 | 16 | 768 | 0 / 800 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 6 | 2 | 726 | 160 | 0 | 0 / 0 | 566 | new: stored by this request 2 | – | – | 512 |
| D3 | 6 | 2 | 726 | 16 | 512 | 0 / 710 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 6 | 3 | 1629 | 304 | 256 | 0 / 0 | 1325 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 6 | 3 | 1629 | 16 | 1536 | 0 / 1613 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 6 | 4 | 1336 | 448 | 256 | 0 / 0 | 888 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 6 | 4 | 1336 | 160 | 1280 | 0 / 1176 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 6 | 5 | 1466 | 592 | 512 | 0 / 0 | 874 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 6 | 5 | 1466 | 304 | 1280 | 0 / 1162 | 0 | held (on GPU already) 1, hit, not loaded 4 | – | – | 0 |
| P | 6 | 6 | 1530 | 720 | 512 | 0 / 0 | 810 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 6 | 1530 | 448 | 1280 | 0 / 1082 | 0 | held (on GPU already) 1, hit, not loaded 4 | – | – | 0 |
| P | 7 | 0 | 1136 | 16 | 0 | 0 / 0 | 1120 | new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 0 | 1136 | 16 | 1024 | 0 / 1120 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 7 | 1 | 815 | 16 | 0 | 0 / 0 | 799 | new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 1 | 815 | 16 | 768 | 0 / 799 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 7 | 2 | 1405 | 160 | 0 | 0 / 0 | 1245 | new: stored by this request 5 | – | – | 1280 |
| D1 | 7 | 2 | 1405 | 16 | 1280 | 0 / 1389 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 7 | 3 | 1353 | 320 | 256 | 0 / 0 | 1033 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 3 | 1353 | 16 | 1280 | 0 / 1337 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 7 | 4 | 1770 | 496 | 256 | 0 / 0 | 1274 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D3 | 7 | 4 | 1770 | 160 | 1536 | 0 / 1610 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 7 | 5 | 1403 | 640 | 512 | 0 / 0 | 763 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D1 | 7 | 5 | 1403 | 320 | 1280 | 0 / 1083 | 0 | held (on GPU already) 1, hit, not loaded 4 | – | – | 0 |
| P | 7 | 6 | 1738 | 784 | 768 | 0 / 0 | 954 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 7 | 6 | 1738 | 496 | 1536 | 0 / 1242 | 0 | held (on GPU already) 1, hit, not loaded 5 | – | – | 0 |
| P | 7 | 7 | 1886 | 944 | 768 | 0 / 0 | 942 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D3 | 7 | 7 | 1886 | 640 | 1792 | 0 / 1246 | 0 | held (on GPU already) 2, hit, not loaded 5 | – | – | 0 |
| P | 8 | 0 | 1156 | 16 | 0 | 0 / 0 | 1140 | new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 0 | 1156 | 16 | 1024 | 0 / 1140 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 8 | 1 | 1225 | 16 | 0 | 0 / 0 | 1209 | new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 1 | 1225 | 16 | 1024 | 0 / 1209 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 8 | 2 | 1448 | 160 | 0 | 0 / 0 | 1288 | new: stored by this request 5 | – | – | 1280 |
| D3 | 8 | 2 | 1448 | 16 | 0 | 0 / 1432 | 0 | miss: not yet stored 5 | – | – | 1280 |
| P | 8 | 3 | 902 | 304 | 256 | 0 / 0 | 598 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| D1 | 8 | 3 | 902 | 16 | 768 | 0 / 886 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 8 | 4 | 1111 | 448 | 256 | 0 / 0 | 663 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D2 | 8 | 4 | 1111 | 160 | 1024 | 0 / 951 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 8 | 5 | 1473 | 592 | 512 | 0 / 0 | 881 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 8 | 5 | 1473 | 304 | 1280 | 0 / 1169 | 0 | held (on GPU already) 1, hit, not loaded 4 | – | – | 0 |
| P | 8 | 6 | 1627 | 736 | 512 | 0 / 0 | 891 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 6 | 1627 | 448 | 1536 | 0 / 1179 | 0 | held (on GPU already) 1, hit, not loaded 5 | – | – | 0 |
| P | 8 | 7 | 1932 | 864 | 768 | 0 / 0 | 1068 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 7 | 1932 | 592 | 1792 | 0 / 1340 | 0 | held (on GPU already) 2, hit, not loaded 5 | – | – | 0 |
| P | 9 | 0 | 464 | 16 | 0 | 0 / 0 | 448 | new: stored by this request 1 | – | – | 256 |
| D3 | 9 | 0 | 464 | 16 | 256 | 0 / 448 | 0 | hit, not loaded 1 | – | – | 0 |
| P | 9 | 1 | 892 | 32 | 0 | 0 / 0 | 860 | new: stored by this request 3 | – | – | 768 |
| D1 | 9 | 1 | 892 | 32 | 768 | 0 / 860 | 0 | hit, not loaded 3 | – | – | 0 |
| P | 9 | 2 | 1197 | 160 | 0 | 0 / 0 | 1037 | new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 2 | 1197 | 32 | 1024 | 0 / 1165 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 9 | 3 | 1615 | 304 | 256 | 0 / 0 | 1311 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D3 | 9 | 3 | 1615 | 32 | 256 | 0 / 1583 | 0 | hit, not loaded 1, miss: not yet stored 5 | – | – | 1280 |
| P | 9 | 4 | 1719 | 448 | 256 | 0 / 0 | 1271 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 9 | 4 | 1719 | 160 | 1536 | 0 / 1559 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 9 | 5 | 1557 | 576 | 512 | 0 / 0 | 981 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 5 | 1557 | 304 | 1536 | 0 / 1253 | 0 | held (on GPU already) 1, hit, not loaded 5 | – | – | 0 |
| P | 9 | 6 | 1438 | 720 | 512 | 0 / 0 | 718 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 9 | 6 | 1438 | 448 | 1280 | 0 / 990 | 0 | held (on GPU already) 1, hit, not loaded 4 | – | – | 0 |

## One chunk's life: f315a89b1e0bc2f2

Conversation 8, turn 2, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0061 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request c1363bca) |
| 95.8443 | P looked it up (turn 2): new: stored by this request |
| 95.8491 | P computed and cached block 15 (GPU block 3258, hash 253acfd8be53f6d2, request 92747eb3) |
| 95.8494 | P's connector planned the store |
| 95.8677 | P's worker sent it to the server |
| 95.8691 | the server bound the chunk to its tokens (mp.tokens) |
| 95.8698 | server memory (L1) reserved |
| 95.9629 | eviction order refreshed (a touch: no I/O) |
| 95.9723 | D3 looked it up (turn 2): miss: not yet stored |
| 95.9735 | written to memory (L1) |
| 96.1780 | written to disk (L2) |
| 96.1786 | removed from memory (L1) |
| 96.2923 | removed from memory (L1) |
| 97.6799 | eviction order refreshed (a touch: no I/O) |
| 97.6968 | P looked it up (turn 3): held (on GPU already) |
| 97.6968 | eviction order refreshed (a touch: no I/O) |
| 97.7532 | eviction order refreshed (a touch: no I/O) |
| 97.7615 | D1 looked it up (turn 3): hit, not loaded |
| 97.7876 | read from disk (L2) |
| 99.3705 | eviction order refreshed (a touch: no I/O) |
| 99.3847 | P looked it up (turn 4): held (on GPU already) |
| 99.3849 | eviction order refreshed (a touch: no I/O) |
| 99.4496 | eviction order refreshed (a touch: no I/O) |
| 99.4581 | D2 looked it up (turn 4): hit, not loaded |
| 101.0589 | eviction order refreshed (a touch: no I/O) |
| 101.0784 | P looked it up (turn 5): held (on GPU already) |
| 101.0785 | eviction order refreshed (a touch: no I/O) |
| 101.1602 | eviction order refreshed (a touch: no I/O) |
| 101.1704 | D3 looked it up (turn 5): held (on GPU already) |
| 101.1705 | eviction order refreshed (a touch: no I/O) |
| 102.8553 | eviction order refreshed (a touch: no I/O) |
| 102.8700 | P looked it up (turn 6): held (on GPU already) |
| 102.8702 | eviction order refreshed (a touch: no I/O) |
| 102.9542 | eviction order refreshed (a touch: no I/O) |
| 102.9645 | D1 looked it up (turn 6): held (on GPU already) |
| 102.9646 | eviction order refreshed (a touch: no I/O) |
| 104.6269 | eviction order refreshed (a touch: no I/O) |
| 104.6390 | P looked it up (turn 7): held (on GPU already) |
| 104.6391 | eviction order refreshed (a touch: no I/O) |
| 104.7364 | eviction order refreshed (a touch: no I/O) |
| 104.7473 | D2 looked it up (turn 7): held (on GPU already) |
| 104.7473 | eviction order refreshed (a touch: no I/O) |
| 106.4632 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0815 | P answered (stored) |
| 0.1031 | D1 took the request |
| 0.1058 | the server got its lookup |
| 0.1070 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1087 | its connector had the lookup's answer |
| 0.4077 | the first token reached the proxy |
| 1.7750 | finished |
| 1.7785 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
