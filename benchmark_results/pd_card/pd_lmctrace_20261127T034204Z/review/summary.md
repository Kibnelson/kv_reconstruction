# PD matrix lmctrace on gpu-grad-01, job 30806 (completed), reviewed from the kept results


# LMCache lifecycle: the runs side by side

| | disk, NIXL, per-engine, carry-forward | disk, NIXL, per-engine, chat RAG | disk, NIXL, shared, carry-forward | disk, NIXL, shared, chat RAG | memory, NIXL, per-engine, carry-forward | memory, NIXL, per-engine, chat RAG | memory, NIXL, shared, carry-forward | memory, NIXL, shared, chat RAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LMCache server lasted the run | True | True | True | True | True | True | True | True |
| LMCache server errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P to D | nixl | nixl | nixl | nixl | nixl | nixl | nixl | nixl |
| LMCache servers | per_engine | per_engine | shared | shared | per_engine | per_engine | shared | shared |
| prompt layout | interleaved | history_first | interleaved | history_first | interleaved | history_first | interleaved | history_first |
| passages | all | all | all | all | all | all | all | all |
| conversations | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| disk reads | retain | retain | retain | retain | retain | retain | retain | retain |
| chunks bound by the engines' stores | 851 | 529 | 242 | 243 | 851 | 529 | 242 | 243 |
| chunks stored by | {'P': 242, 'D': 609} | {'P': 243, 'D': 286} | {'P': 242} | {'P': 243} | {'P': 242, 'D': 609} | {'P': 243, 'D': 286} | {'P': 242} | {'P': 243} |
| chunks written to memory | 851 | 529 | 242 | 243 | 851 | 529 | 242 | 243 |
| stored MB | 24985.500 | 15531.500 | 7105.200 | 7134.500 | 24985.500 | 15531.500 | 7105.200 | 7134.500 |
| chunks to disk | 851 | 529 | 242 | 243 | 0 | 0 | 0 | 0 |
| chunks removed from memory | 851 | 529 | 242 | 243 | 638 | 318 | 47 | 47 |
| pulls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pulled from | {} | {} | {} | {} | {} | {} | {} | {} |
| never pulled | 851 | 529 | 242 | 243 | 851 | 529 | 242 | 243 |
| D tokens from own GPU | 101344 | 10096 | 101344 | 10096 | 101344 | 10096 | 101344 | 10096 |
| D tokens loaded from LMCache | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D tokens read from P over NIXL | 157500 | 77668 | 157500 | 77668 | 157500 | 77668 | 157500 | 77668 |
| NIXL read ms (median) | 449.470 | 211.100 | 446.130 | 213.020 | 441.670 | 207.150 | 445.590 | 209.220 |
| NIXL read ms (p90) | 646.750 | 288.440 | 632.480 | 286.530 | 626.900 | 280.750 | 626.580 | 286.470 |
| NIXL read GB/s (median) | 0.607 | 0.604 | 0.612 | 0.607 | 0.619 | 0.615 | 0.618 | 0.610 |
| NIXL blocks = P's bytes | 9879 of 9879 | 4883 of 4883 | 9879 of 9879 | 4883 of 4883 | 9879 of 9879 | 4883 of 4883 | 9879 of 9879 | 4883 of 4883 |
| D's answers = one engine's | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 |
| D's answers = one engine's, NIXL's layout | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 | 62 of 69 | 52 of 69 |
| UCX transports | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp |
| P held its blocks for D, ms (median) | 476.550 | 236.360 | 540.490 | 244.860 | 464.640 | 232.770 | 466.970 | 234.690 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 94.120 | 83.000 | 94.380 | 82.820 | 94.530 | 82.240 | 94.120 | 83.060 |
| memory to disk ms (median) | 356.270 | 163.230 | 212.690 | 149.370 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 851 of 851; 0 of 0; 851 of 851; 62 of 69; 62 of 69; 851 of 851; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 242 files, 7.11 GB; {'_default': '0 objects, 0 bytes'}; 180 files, 5.28 GB; {'_default': '0 objects, 0 bytes'}; 206 files, 6.05 GB; {'_default': '0 objects, 0 bytes'}; 223 files, 6.55 GB | 529 of 529; 0 of 0; 529 of 529; 52 of 69; 52 of 69; 529 of 529; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 243 files, 7.13 GB; {'_default': '0 objects, 0 bytes'}; 94 files, 2.76 GB; {'_default': '0 objects, 0 bytes'}; 97 files, 2.85 GB; {'_default': '0 objects, 0 bytes'}; 95 files, 2.79 GB | 242 of 242; 0 of 0; 242 of 242; 62 of 69; 62 of 69; 242 of 242; 0 of 0; 0; 206 of 206 disk reads still in memory at the end (prefetch policy retain); {'_default': '206 objects, 6048186368 bytes'}; 242 files, 7.11 GB | 243 of 243; 0 of 0; 243 of 243; 52 of 69; 52 of 69; 243 of 243; 0 of 0; 0; 26 of 26 disk reads still in memory at the end (prefetch policy retain); {'_default': '26 objects, 763363328 bytes'}; 243 files, 7.13 GB | 851 of 851; 0 of 0; 851 of 851; 62 of 69; 62 of 69 | 529 of 529; 0 of 0; 529 of 529; 52 of 69; 52 of 69 | 242 of 242; 0 of 0; 242 of 242; 62 of 69; 62 of 69 | 243 of 243; 0 of 0; 243 of 243; 52 of 69; 52 of 69 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_234104, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_232500, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_230851, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_225243, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_233246, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_231716, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_230030, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_224358


# LMCache lifecycle: disk mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 16808 records from 8 processes (D1 3518, D2 3652, D3 3678, LMC_D1 864, LMC_D2 890, LMC_D3 892, LMC_P 2162, P 1152); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 258844 | 258844 |
| from the engine's own GPU prefix cache | 195248 | 101344 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 157500 |
| computed | 63596 | 0 |
| stored to LMCache (tokens) | 61952 | 155904 |

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

- D requests that read from P: 69, 157500 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 449.470 ms, p90 646.750 ms
- P held its blocks for D from its request's end to D's read done: median 476.550 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.607 GB/s (the KV it read over its time)
- the bytes: 9879 of 9879 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 851 chunks (by {'P': 242, 'D': 609}; per server {'LMC_P': 242, 'LMC_D1': 180, 'LMC_D2': 206, 'LMC_D3': 223}); written to memory 851 (24985.5 MB); to disk 851; removed from memory 851
- pulled by D or P: 0 times, from –; never pulled: 851 (265 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 3069
- every lookup's outcome per chunk: {'new: stored by this request': 851, 'held (on GPU already)': 1109}
- pulls per stored chunk: {0: 851}
- median time from a chunk's last block computed on P to the chunk in memory: 51.500 ms; memory to disk: 356.270 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 94.120 ms, p90 133.890 ms.

## Checks

- chunk tokens match prompt: 851 of 851
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 851 of 851
- d answers same as one engine: 62 of 69
- d answers same as one engine nixl layout: 62 of 69
- disk mode on disk then out of memory: 851 of 851
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 0 of 0 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end (LMC P): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC P): 242 files, 7.11 GB
- disk mode server memory at end (LMC D1): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D1): 180 files, 5.28 GB
- disk mode server memory at end (LMC D2): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D2): 206 files, 6.05 GB
- disk mode server memory at end (LMC D3): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D3): 223 files, 6.55 GB

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
| P | 4 | 1 | 2301 | 1216 | 1024 | 0 / 0 | 1085 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 2301 | 1216 | 1024 | 0 / 1085 | 0 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| P | 4 | 2 | 3149 | 2288 | 2048 | 0 / 0 | 861 | held (on GPU already) 8, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 3149 | 1216 | 1024 | 0 / 1933 | 0 | held (on GPU already) 4, new: stored by this request 8 | – | – | 2048 |
| P | 4 | 3 | 4481 | 3136 | 3072 | 0 / 0 | 1345 | held (on GPU already) 12, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 4481 | 1216 | 1024 | 0 / 3265 | 0 | held (on GPU already) 4, new: stored by this request 13 | – | – | 3328 |
| P | 4 | 4 | 5667 | 4480 | 4352 | 0 / 0 | 1187 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 5667 | 2416 | 2304 | 0 / 3251 | 0 | held (on GPU already) 9, new: stored by this request 13 | – | – | 3328 |
| P | 4 | 5 | 6878 | 5664 | 5632 | 0 / 0 | 1214 | held (on GPU already) 22, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 5 | 6878 | 3264 | 3072 | 0 / 3614 | 0 | held (on GPU already) 12, new: stored by this request 14 | – | – | 3584 |
| P | 5 | 0 | 1035 | 16 | 0 | 0 / 0 | 1019 | new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 0 | 1035 | 16 | 0 | 0 / 1019 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 1 | 1944 | 1024 | 1024 | 0 / 0 | 920 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D2 | 5 | 1 | 1944 | 16 | 0 | 0 / 1928 | 0 | new: stored by this request 7 | – | – | 1792 |
| P | 5 | 2 | 2959 | 1936 | 1792 | 0 / 0 | 1023 | held (on GPU already) 7, new: stored by this request 4 | – | – | 1024 |
| D3 | 5 | 2 | 2959 | 16 | 0 | 0 / 2943 | 0 | new: stored by this request 11 | – | – | 2816 |
| P | 5 | 3 | 3848 | 2944 | 2816 | 0 / 0 | 904 | held (on GPU already) 11, new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 3 | 3848 | 1152 | 1024 | 0 / 2696 | 0 | held (on GPU already) 4, new: stored by this request 11 | – | – | 2816 |
| P | 5 | 4 | 4996 | 3840 | 3840 | 0 / 0 | 1156 | held (on GPU already) 15, new: stored by this request 4 | – | – | 1024 |
| D2 | 5 | 4 | 4996 | 2064 | 2048 | 0 / 2932 | 0 | held (on GPU already) 8, new: stored by this request 11 | – | – | 2816 |
| P | 5 | 5 | 5662 | 4992 | 4864 | 0 / 0 | 670 | held (on GPU already) 19, new: stored by this request 3 | – | – | 768 |
| D3 | 5 | 5 | 5662 | 3072 | 3072 | 0 / 2590 | 0 | held (on GPU already) 12, new: stored by this request 10 | – | – | 2560 |
| P | 6 | 0 | 814 | 16 | 0 | 0 / 0 | 798 | new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 0 | 814 | 16 | 0 | 0 / 798 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 6 | 1 | 1585 | 800 | 768 | 0 / 0 | 785 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 1 | 1585 | 16 | 0 | 0 / 1569 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 6 | 2 | 2124 | 1584 | 1536 | 0 / 0 | 540 | held (on GPU already) 6, new: stored by this request 2 | – | – | 512 |
| D3 | 6 | 2 | 2124 | 16 | 0 | 0 / 2108 | 0 | new: stored by this request 8 | – | – | 2048 |
| P | 6 | 3 | 3426 | 2112 | 2048 | 0 / 0 | 1314 | held (on GPU already) 8, new: stored by this request 5 | – | – | 1280 |
| D1 | 6 | 3 | 3426 | 928 | 768 | 0 / 2498 | 0 | held (on GPU already) 3, new: stored by this request 10 | – | – | 2560 |
| P | 6 | 4 | 4297 | 3424 | 3328 | 0 / 0 | 873 | held (on GPU already) 13, new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 4 | 4297 | 1712 | 1536 | 0 / 2585 | 0 | held (on GPU already) 6, new: stored by this request 10 | – | – | 2560 |
| P | 6 | 5 | 5161 | 4288 | 4096 | 0 / 0 | 873 | held (on GPU already) 16, new: stored by this request 4 | – | – | 1024 |
| D3 | 6 | 5 | 5161 | 2240 | 2048 | 0 / 2921 | 0 | held (on GPU already) 8, new: stored by this request 12 | – | – | 3072 |
| P | 6 | 6 | 5949 | 5152 | 5120 | 0 / 0 | 797 | held (on GPU already) 20, new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 6 | 5949 | 3552 | 3328 | 0 / 2397 | 0 | held (on GPU already) 13, new: stored by this request 10 | – | – | 2560 |
| P | 7 | 0 | 1136 | 16 | 0 | 0 / 0 | 1120 | new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 0 | 1136 | 16 | 0 | 0 / 1120 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 7 | 1 | 1909 | 1136 | 1024 | 0 / 0 | 773 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 1 | 1909 | 16 | 0 | 0 / 1893 | 0 | new: stored by this request 7 | – | – | 1792 |
| P | 7 | 2 | 3115 | 1904 | 1792 | 0 / 0 | 1211 | held (on GPU already) 7, new: stored by this request 5 | – | – | 1280 |
| D1 | 7 | 2 | 3115 | 16 | 0 | 0 / 3099 | 0 | new: stored by this request 12 | – | – | 3072 |
| P | 7 | 3 | 4099 | 3104 | 3072 | 0 / 0 | 995 | held (on GPU already) 12, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 3 | 4099 | 1248 | 1024 | 0 / 2851 | 0 | held (on GPU already) 4, new: stored by this request 12 | – | – | 3072 |
| P | 7 | 4 | 5352 | 4096 | 4096 | 0 / 0 | 1256 | held (on GPU already) 16, new: stored by this request 4 | – | – | 1024 |
| D3 | 7 | 4 | 5352 | 2032 | 1792 | 0 / 3320 | 0 | held (on GPU already) 7, new: stored by this request 13 | – | – | 3328 |
| P | 7 | 5 | 6094 | 5344 | 5120 | 0 / 0 | 750 | held (on GPU already) 20, new: stored by this request 3 | – | – | 768 |
| D1 | 7 | 5 | 6094 | 3232 | 3072 | 0 / 2862 | 0 | held (on GPU already) 12, new: stored by this request 11 | – | – | 2816 |
| P | 7 | 6 | 7005 | 6080 | 5888 | 0 / 0 | 925 | held (on GPU already) 23, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 6 | 7005 | 4224 | 4096 | 0 / 2781 | 0 | held (on GPU already) 16, new: stored by this request 11 | – | – | 2816 |
| P | 7 | 7 | 7916 | 6992 | 6912 | 0 / 0 | 924 | held (on GPU already) 27, new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 7 | 7916 | 5472 | 5376 | 0 / 2444 | 0 | held (on GPU already) 21, new: stored by this request 9 | – | – | 2304 |
| P | 8 | 0 | 1156 | 16 | 0 | 0 / 0 | 1140 | new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 0 | 1156 | 16 | 0 | 0 / 1140 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 1 | 2338 | 1152 | 1024 | 0 / 0 | 1186 | held (on GPU already) 4, new: stored by this request 5 | – | – | 1280 |
| D2 | 8 | 1 | 2338 | 16 | 0 | 0 / 2322 | 0 | new: stored by this request 9 | – | – | 2304 |
| P | 8 | 2 | 3605 | 2336 | 2304 | 0 / 0 | 1269 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D3 | 8 | 2 | 3605 | 16 | 0 | 0 / 3589 | 0 | new: stored by this request 14 | – | – | 3584 |
| P | 8 | 3 | 4188 | 3600 | 3584 | 0 / 0 | 588 | held (on GPU already) 14, new: stored by this request 2 | – | – | 512 |
| D1 | 8 | 3 | 4188 | 1280 | 1280 | 0 / 2908 | 0 | held (on GPU already) 5, new: stored by this request 11 | – | – | 2816 |
| P | 8 | 4 | 4833 | 4176 | 4096 | 0 / 0 | 657 | held (on GPU already) 16, new: stored by this request 2 | – | – | 512 |
| D2 | 8 | 4 | 4833 | 2464 | 2304 | 0 / 2369 | 0 | held (on GPU already) 9, new: stored by this request 9 | – | – | 2304 |
| P | 8 | 5 | 5699 | 4832 | 4608 | 0 / 0 | 867 | held (on GPU already) 18, new: stored by this request 4 | – | – | 1024 |
| D3 | 8 | 5 | 5699 | 3728 | 3584 | 0 / 1971 | 0 | held (on GPU already) 14, new: stored by this request 8 | – | – | 2048 |
| P | 8 | 6 | 6579 | 5696 | 5632 | 0 / 0 | 883 | held (on GPU already) 22, new: stored by this request 3 | – | – | 768 |
| D1 | 8 | 6 | 6579 | 4304 | 4096 | 0 / 2275 | 0 | held (on GPU already) 16, new: stored by this request 9 | – | – | 2304 |
| P | 8 | 7 | 7624 | 6576 | 6400 | 0 / 0 | 1048 | held (on GPU already) 25, new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 7 | 7624 | 4960 | 4864 | 0 / 2664 | 0 | held (on GPU already) 19, new: stored by this request 10 | – | – | 2560 |
| P | 9 | 0 | 464 | 16 | 0 | 0 / 0 | 448 | new: stored by this request 1 | – | – | 256 |
| D3 | 9 | 0 | 464 | 16 | 0 | 0 / 448 | 0 | new: stored by this request 1 | – | – | 256 |
| P | 9 | 1 | 1314 | 464 | 256 | 0 / 0 | 850 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D1 | 9 | 1 | 1314 | 16 | 0 | 0 / 1298 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 9 | 2 | 2331 | 1312 | 1280 | 0 / 0 | 1019 | held (on GPU already) 5, new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 2 | 2331 | 16 | 0 | 0 / 2315 | 0 | new: stored by this request 9 | – | – | 2304 |
| P | 9 | 3 | 3627 | 2320 | 2304 | 0 / 0 | 1307 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D3 | 9 | 3 | 3627 | 576 | 512 | 0 / 3051 | 0 | held (on GPU already) 2, new: stored by this request 12 | – | – | 3072 |
| P | 9 | 4 | 4891 | 3616 | 3584 | 0 / 0 | 1275 | held (on GPU already) 14, new: stored by this request 5 | – | – | 1280 |
| D1 | 9 | 4 | 4891 | 1440 | 1280 | 0 / 3451 | 0 | held (on GPU already) 5, new: stored by this request 14 | – | – | 3584 |
| P | 9 | 5 | 5845 | 4880 | 4864 | 0 / 0 | 965 | held (on GPU already) 19, new: stored by this request 3 | – | – | 768 |
| D2 | 9 | 5 | 5845 | 2448 | 2304 | 0 / 3397 | 0 | held (on GPU already) 9, new: stored by this request 13 | – | – | 3328 |
| P | 9 | 6 | 6534 | 5840 | 5632 | 0 / 0 | 694 | held (on GPU already) 22, new: stored by this request 3 | – | – | 768 |
| D3 | 9 | 6 | 6534 | 3744 | 3584 | 0 / 2790 | 0 | held (on GPU already) 14, new: stored by this request 11 | – | – | 2816 |

## One chunk's life: 5e868214004923dc

Conversation 1, turn 0, tokens 0-256 (system, doc 850931827_11086-12414-0-1328); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0065 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 18163722) |
| 9.5678 | P looked it up (turn 0): new: stored by this request |
| 9.5728 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request 47b4eaf2) |
| 9.5731 | P's connector planned the store |
| 9.5913 | P's worker sent it to the server |
| 9.5926 | the server bound the chunk to its tokens (mp.tokens) |
| 9.5931 | server memory (L1) reserved |
| 9.6623 | eviction order refreshed (a touch: no I/O) |
| 9.6668 | written to memory (L1) |
| 9.8305 | written to disk (L2) |
| 9.8311 | removed from memory (L1) |
| 11.3283 | P looked it up (turn 1): held (on GPU already) |
| 11.3284 | eviction order refreshed (a touch: no I/O) |
| 11.4570 | eviction order refreshed (a touch: no I/O) |
| 13.4135 | P looked it up (turn 2): held (on GPU already) |
| 13.4136 | eviction order refreshed (a touch: no I/O) |
| 13.5325 | eviction order refreshed (a touch: no I/O) |
| 15.7570 | P looked it up (turn 3): held (on GPU already) |
| 15.7572 | eviction order refreshed (a touch: no I/O) |
| 15.8636 | eviction order refreshed (a touch: no I/O) |
| 18.0749 | P looked it up (turn 4): held (on GPU already) |
| 18.0751 | eviction order refreshed (a touch: no I/O) |
| 18.2371 | eviction order refreshed (a touch: no I/O) |
| 20.5222 | P looked it up (turn 5): held (on GPU already) |
| 20.5225 | eviction order refreshed (a touch: no I/O) |
| 20.6644 | eviction order refreshed (a touch: no I/O) |
| 22.9843 | P looked it up (turn 6): held (on GPU already) |
| 22.9846 | eviction order refreshed (a touch: no I/O) |
| 23.1551 | eviction order refreshed (a touch: no I/O) |
| 53.4468 | P looked it up (turn 0): held (on GPU already) |
| 53.4469 | eviction order refreshed (a touch: no I/O) |
| 53.4669 | eviction order refreshed (a touch: no I/O) |
| 54.9732 | P looked it up (turn 1): held (on GPU already) |
| 54.9733 | eviction order refreshed (a touch: no I/O) |
| 55.0749 | eviction order refreshed (a touch: no I/O) |
| 56.7627 | P looked it up (turn 2): held (on GPU already) |
| 56.7629 | eviction order refreshed (a touch: no I/O) |
| 56.8528 | eviction order refreshed (a touch: no I/O) |
| 58.7191 | P looked it up (turn 3): held (on GPU already) |
| 58.7193 | eviction order refreshed (a touch: no I/O) |
| 58.8639 | eviction order refreshed (a touch: no I/O) |
| 61.0229 | P looked it up (turn 4): held (on GPU already) |
| 61.0231 | eviction order refreshed (a touch: no I/O) |
| 61.1550 | eviction order refreshed (a touch: no I/O) |
| 63.3497 | P looked it up (turn 5): held (on GPU already) |
| 63.3500 | eviction order refreshed (a touch: no I/O) |
| 63.4910 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0025 | the proxy received the turn |
| 0.0819 | P answered (stored) |
| 0.0935 | D1 took the request |
| 0.0966 | the server got its lookup |
| 0.0993 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1006 | its connector had the lookup's answer |
| 0.3990 | the first token reached the proxy |
| 1.7657 | finished |
| 1.7691 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 16501 records from 8 processes (D1 3494, D2 3513, D3 3656, LMC_D1 854, LMC_D2 857, LMC_D3 875, LMC_P 2099, P 1153); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 87764 | 87764 |
| from the engine's own GPU prefix cache | 23248 | 10096 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 77668 |
| computed | 64516 | 0 |
| stored to LMCache (tokens) | 62208 | 73216 |

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

- D requests that read from P: 69, 77668 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 211.100 ms, p90 288.440 ms
- P held its blocks for D from its request's end to D's read done: median 236.360 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.604 GB/s (the KV it read over its time)
- the bytes: 4883 of 4883 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 529 chunks (by {'P': 243, 'D': 286}; per server {'LMC_P': 243, 'LMC_D1': 94, 'LMC_D2': 97, 'LMC_D3': 95}); written to memory 529 (15531.5 MB); to disk 529; removed from memory 529
- pulled by D or P: 0 times, from –; never pulled: 529 (213 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 699
- every lookup's outcome per chunk: {'new: stored by this request': 529, 'held (on GPU already)': 85}
- pulls per stored chunk: {0: 529}
- median time from a chunk's last block computed on P to the chunk in memory: 37.120 ms; memory to disk: 163.230 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 83.000 ms, p90 109.090 ms.

## Checks

- chunk tokens match prompt: 529 of 529
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 529 of 529
- d answers same as one engine: 52 of 69
- d answers same as one engine nixl layout: 52 of 69
- disk mode on disk then out of memory: 529 of 529
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 0 of 0 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end (LMC P): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC P): 243 files, 7.13 GB
- disk mode server memory at end (LMC D1): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D1): 94 files, 2.76 GB
- disk mode server memory at end (LMC D2): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D2): 97 files, 2.85 GB
- disk mode server memory at end (LMC D3): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D3): 95 files, 2.79 GB

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
| P | 4 | 1 | 1256 | 176 | 0 | 0 / 0 | 1080 | new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 1256 | 160 | 0 | 0 / 1096 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 4 | 2 | 1041 | 160 | 0 | 0 / 0 | 881 | new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 1041 | 160 | 0 | 0 / 881 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 4 | 3 | 1667 | 320 | 256 | 0 / 0 | 1347 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 1667 | 160 | 0 | 0 / 1507 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 4 | 4 | 1676 | 464 | 256 | 0 / 0 | 1212 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 1676 | 160 | 0 | 0 / 1516 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 4 | 5 | 1841 | 608 | 512 | 0 / 0 | 1233 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 4 | 5 | 1841 | 320 | 256 | 0 / 1521 | 0 | held (on GPU already) 1, new: stored by this request 6 | – | – | 1536 |
| P | 5 | 0 | 1035 | 16 | 0 | 0 / 0 | 1019 | new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 0 | 1035 | 16 | 0 | 0 / 1019 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 1 | 950 | 16 | 0 | 0 / 0 | 934 | new: stored by this request 3 | – | – | 768 |
| D2 | 5 | 1 | 950 | 16 | 0 | 0 / 934 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 5 | 2 | 1199 | 160 | 0 | 0 / 0 | 1039 | new: stored by this request 4 | – | – | 1024 |
| D3 | 5 | 2 | 1199 | 16 | 0 | 0 / 1183 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 3 | 1212 | 304 | 256 | 0 / 0 | 908 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D1 | 5 | 3 | 1212 | 16 | 0 | 0 / 1196 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 4 | 1613 | 448 | 256 | 0 / 0 | 1165 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 5 | 4 | 1613 | 160 | 0 | 0 / 1453 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 5 | 5 | 1272 | 592 | 512 | 0 / 0 | 680 | held (on GPU already) 2, new: stored by this request 2 | – | – | 512 |
| D3 | 5 | 5 | 1272 | 304 | 256 | 0 / 968 | 0 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| P | 6 | 0 | 814 | 16 | 0 | 0 / 0 | 798 | new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 0 | 814 | 16 | 0 | 0 / 798 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 6 | 1 | 816 | 16 | 0 | 0 / 0 | 800 | new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 1 | 816 | 16 | 0 | 0 / 800 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 6 | 2 | 726 | 160 | 0 | 0 / 0 | 566 | new: stored by this request 2 | – | – | 512 |
| D3 | 6 | 2 | 726 | 16 | 0 | 0 / 710 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 6 | 3 | 1629 | 304 | 256 | 0 / 0 | 1325 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 6 | 3 | 1629 | 16 | 0 | 0 / 1613 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 6 | 4 | 1336 | 448 | 256 | 0 / 0 | 888 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 6 | 4 | 1336 | 160 | 0 | 0 / 1176 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 6 | 5 | 1466 | 592 | 512 | 0 / 0 | 874 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 6 | 5 | 1466 | 304 | 256 | 0 / 1162 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 6 | 6 | 1530 | 720 | 512 | 0 / 0 | 810 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 6 | 1530 | 448 | 256 | 0 / 1082 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 7 | 0 | 1136 | 16 | 0 | 0 / 0 | 1120 | new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 0 | 1136 | 16 | 0 | 0 / 1120 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 7 | 1 | 815 | 16 | 0 | 0 / 0 | 799 | new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 1 | 815 | 16 | 0 | 0 / 799 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 7 | 2 | 1405 | 160 | 0 | 0 / 0 | 1245 | new: stored by this request 5 | – | – | 1280 |
| D1 | 7 | 2 | 1405 | 16 | 0 | 0 / 1389 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 7 | 3 | 1353 | 320 | 256 | 0 / 0 | 1033 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 3 | 1353 | 16 | 0 | 0 / 1337 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 7 | 4 | 1770 | 496 | 256 | 0 / 0 | 1274 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D3 | 7 | 4 | 1770 | 160 | 0 | 0 / 1610 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 7 | 5 | 1403 | 640 | 512 | 0 / 0 | 763 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D1 | 7 | 5 | 1403 | 320 | 256 | 0 / 1083 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 7 | 6 | 1738 | 784 | 768 | 0 / 0 | 954 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 7 | 6 | 1738 | 496 | 256 | 0 / 1242 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 7 | 7 | 1886 | 944 | 768 | 0 / 0 | 942 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D3 | 7 | 7 | 1886 | 640 | 512 | 0 / 1246 | 0 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| P | 8 | 0 | 1156 | 16 | 0 | 0 / 0 | 1140 | new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 0 | 1156 | 16 | 0 | 0 / 1140 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 1 | 1225 | 16 | 0 | 0 / 0 | 1209 | new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 1 | 1225 | 16 | 0 | 0 / 1209 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 2 | 1448 | 160 | 0 | 0 / 0 | 1288 | new: stored by this request 5 | – | – | 1280 |
| D3 | 8 | 2 | 1448 | 16 | 0 | 0 / 1432 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 8 | 3 | 902 | 304 | 256 | 0 / 0 | 598 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| D1 | 8 | 3 | 902 | 16 | 0 | 0 / 886 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 8 | 4 | 1111 | 448 | 256 | 0 / 0 | 663 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D2 | 8 | 4 | 1111 | 160 | 0 | 0 / 951 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 5 | 1473 | 592 | 512 | 0 / 0 | 881 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 8 | 5 | 1473 | 304 | 256 | 0 / 1169 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 8 | 6 | 1627 | 736 | 512 | 0 / 0 | 891 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 6 | 1627 | 448 | 256 | 0 / 1179 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 8 | 7 | 1932 | 864 | 768 | 0 / 0 | 1068 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 7 | 1932 | 592 | 512 | 0 / 1340 | 0 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| P | 9 | 0 | 464 | 16 | 0 | 0 / 0 | 448 | new: stored by this request 1 | – | – | 256 |
| D3 | 9 | 0 | 464 | 16 | 0 | 0 / 448 | 0 | new: stored by this request 1 | – | – | 256 |
| P | 9 | 1 | 892 | 32 | 0 | 0 / 0 | 860 | new: stored by this request 3 | – | – | 768 |
| D1 | 9 | 1 | 892 | 32 | 0 | 0 / 860 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 9 | 2 | 1197 | 160 | 0 | 0 / 0 | 1037 | new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 2 | 1197 | 32 | 0 | 0 / 1165 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 9 | 3 | 1615 | 304 | 256 | 0 / 0 | 1311 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D3 | 9 | 3 | 1615 | 32 | 0 | 0 / 1583 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 9 | 4 | 1719 | 448 | 256 | 0 / 0 | 1271 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 9 | 4 | 1719 | 160 | 0 | 0 / 1559 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 9 | 5 | 1557 | 576 | 512 | 0 / 0 | 981 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 5 | 1557 | 304 | 256 | 0 / 1253 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 9 | 6 | 1438 | 720 | 512 | 0 / 0 | 718 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 9 | 6 | 1438 | 448 | 256 | 0 / 990 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |

## One chunk's life: aa54d2c3db5e0d3a

Conversation 2, turn 3, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1, newline 1, user 2, assistant 2, answer 2); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0064 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 5d1a25c9) |
| 23.8044 | P looked it up (turn 3): new: stored by this request |
| 23.8092 | P computed and cached block 15 (GPU block 878, hash 5c2a3032cc6dadfb, request 20b8f40a) |
| 23.8095 | P's connector planned the store |
| 23.8286 | P's worker sent it to the server |
| 23.8298 | the server bound the chunk to its tokens (mp.tokens) |
| 23.8303 | server memory (L1) reserved |
| 23.8718 | eviction order refreshed (a touch: no I/O) |
| 23.8796 | written to memory (L1) |
| 24.0097 | written to disk (L2) |
| 24.0103 | removed from memory (L1) |
| 25.4891 | P looked it up (turn 4): held (on GPU already) |
| 25.4892 | eviction order refreshed (a touch: no I/O) |
| 25.5943 | eviction order refreshed (a touch: no I/O) |
| 27.3292 | P looked it up (turn 5): held (on GPU already) |
| 27.3293 | eviction order refreshed (a touch: no I/O) |
| 27.4090 | eviction order refreshed (a touch: no I/O) |
| 29.0749 | P looked it up (turn 6): held (on GPU already) |
| 29.0751 | eviction order refreshed (a touch: no I/O) |
| 29.1632 | eviction order refreshed (a touch: no I/O) |
| 30.8714 | P looked it up (turn 7): held (on GPU already) |
| 30.8716 | eviction order refreshed (a touch: no I/O) |
| 30.9523 | eviction order refreshed (a touch: no I/O) |
| 32.6212 | P looked it up (turn 8): held (on GPU already) |
| 32.6214 | eviction order refreshed (a touch: no I/O) |
| 32.7300 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0819 | P answered (stored) |
| 0.0942 | D1 took the request |
| 0.0970 | the server got its lookup |
| 0.0996 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1008 | its connector had the lookup's answer |
| 0.3856 | the first token reached the proxy |
| 1.7526 | finished |
| 1.7562 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 20733 records from 5 processes (D1 4696, D2 5212, D3 5292, LMC 4404, P 1129); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 258844 | 258844 |
| from the engine's own GPU prefix cache | 195248 | 101344 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 157500 |
| computed | 63596 | 0 |
| stored to LMCache (tokens) | 61952 | 2048 |

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

- D requests that read from P: 69, 157500 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 446.130 ms, p90 632.480 ms
- P held its blocks for D from its request's end to D's read done: median 540.490 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.612 GB/s (the KV it read over its time)
- the bytes: 9879 of 9879 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 242 chunks (by {'P': 242}); written to memory 242 (7105.2 MB); to disk 242; removed from memory 242
- pulled by D or P: 0 times, from –; never pulled: 242 (81 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 206; touches (eviction order refreshed, no I/O): 3069
- every lookup's outcome per chunk: {'new: stored by this request': 242, 'hit, not loaded': 601, 'held (on GPU already)': 1109, 'miss: not yet stored': 8}
- pulls per stored chunk: {0: 242}
- median time from a chunk's last block computed on P to the chunk in memory: 104.320 ms; memory to disk: 212.690 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 94.380 ms, p90 134.060 ms.

## Checks

- chunk tokens match prompt: 242 of 242
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 242 of 242
- d answers same as one engine: 62 of 69
- d answers same as one engine nixl layout: 62 of 69
- disk mode on disk then out of memory: 242 of 242
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 206 of 206 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '206 objects, 6048186368 bytes'}
- disk mode files at end: 242 files, 7.11 GB

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
| D1 | 5 | 0 | 1035 | 16 | 0 | 0 / 1019 | 0 | miss: not yet stored 4 | – | – | 1024 |
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
| D1 | 8 | 0 | 1156 | 16 | 0 | 0 / 1140 | 0 | miss: not yet stored 4 | – | – | 1024 |
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
| D1 | 9 | 1 | 1314 | 16 | 1280 | 0 / 1298 | 0 | hit, not loaded 5 | – | – | 0 |
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
| 0.0069 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 38e3f852) |
| 9.6682 | P looked it up (turn 0): new: stored by this request |
| 9.6734 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request 50aaad9e) |
| 9.6737 | P's connector planned the store |
| 9.6918 | P's worker sent it to the server |
| 9.6931 | the server bound the chunk to its tokens (mp.tokens) |
| 9.6935 | server memory (L1) reserved |
| 9.7623 | eviction order refreshed (a touch: no I/O) |
| 9.7710 | written to memory (L1) |
| 9.7731 | D3 looked it up (turn 0): hit, not loaded |
| 9.9099 | written to disk (L2) |
| 9.9106 | removed from memory (L1) |
| 11.3988 | eviction order refreshed (a touch: no I/O) |
| 11.4206 | P looked it up (turn 1): held (on GPU already) |
| 11.4207 | eviction order refreshed (a touch: no I/O) |
| 11.5484 | eviction order refreshed (a touch: no I/O) |
| 11.5601 | D1 looked it up (turn 1): hit, not loaded |
| 11.6426 | read from disk (L2) |
| 13.5594 | eviction order refreshed (a touch: no I/O) |
| 13.5819 | P looked it up (turn 2): held (on GPU already) |
| 13.5820 | eviction order refreshed (a touch: no I/O) |
| 13.7011 | eviction order refreshed (a touch: no I/O) |
| 13.7150 | D2 looked it up (turn 2): hit, not loaded |
| 15.9815 | eviction order refreshed (a touch: no I/O) |
| 16.0011 | P looked it up (turn 3): held (on GPU already) |
| 16.0012 | eviction order refreshed (a touch: no I/O) |
| 16.1058 | eviction order refreshed (a touch: no I/O) |
| 16.1226 | D3 looked it up (turn 3): held (on GPU already) |
| 16.1228 | eviction order refreshed (a touch: no I/O) |
| 18.3777 | eviction order refreshed (a touch: no I/O) |
| 18.4004 | P looked it up (turn 4): held (on GPU already) |
| 18.4006 | eviction order refreshed (a touch: no I/O) |
| 18.5666 | eviction order refreshed (a touch: no I/O) |
| 18.5858 | D1 looked it up (turn 4): held (on GPU already) |
| 18.5859 | eviction order refreshed (a touch: no I/O) |
| 20.8701 | eviction order refreshed (a touch: no I/O) |
| 20.8973 | P looked it up (turn 5): held (on GPU already) |
| 20.8976 | eviction order refreshed (a touch: no I/O) |
| 21.0446 | eviction order refreshed (a touch: no I/O) |
| 21.0690 | D2 looked it up (turn 5): held (on GPU already) |
| 21.0691 | eviction order refreshed (a touch: no I/O) |
| 23.4167 | eviction order refreshed (a touch: no I/O) |
| 23.4307 | P looked it up (turn 6): held (on GPU already) |
| 23.4308 | eviction order refreshed (a touch: no I/O) |
| 23.6006 | eviction order refreshed (a touch: no I/O) |
| 23.6247 | D3 looked it up (turn 6): held (on GPU already) |
| 23.6248 | eviction order refreshed (a touch: no I/O) |
| 26.0484 | eviction order refreshed (a touch: no I/O) |
| 54.6447 | P looked it up (turn 0): held (on GPU already) |
| 54.6448 | eviction order refreshed (a touch: no I/O) |
| 54.6626 | eviction order refreshed (a touch: no I/O) |
| 54.6705 | D1 looked it up (turn 0): held (on GPU already) |
| 54.6707 | eviction order refreshed (a touch: no I/O) |
| 56.1273 | eviction order refreshed (a touch: no I/O) |
| 56.1394 | P looked it up (turn 1): held (on GPU already) |
| 56.1396 | eviction order refreshed (a touch: no I/O) |
| 56.2409 | eviction order refreshed (a touch: no I/O) |
| 56.2530 | D2 looked it up (turn 1): held (on GPU already) |
| 56.2532 | eviction order refreshed (a touch: no I/O) |
| 57.9192 | eviction order refreshed (a touch: no I/O) |
| 57.9378 | P looked it up (turn 2): held (on GPU already) |
| 57.9379 | eviction order refreshed (a touch: no I/O) |
| 58.0267 | eviction order refreshed (a touch: no I/O) |
| 58.0406 | D3 looked it up (turn 2): held (on GPU already) |
| 58.0407 | eviction order refreshed (a touch: no I/O) |
| 59.9564 | eviction order refreshed (a touch: no I/O) |
| 59.9729 | P looked it up (turn 3): held (on GPU already) |
| 59.9730 | eviction order refreshed (a touch: no I/O) |
| 60.1177 | eviction order refreshed (a touch: no I/O) |
| 60.1319 | D1 looked it up (turn 3): held (on GPU already) |
| 60.1320 | eviction order refreshed (a touch: no I/O) |
| 62.3331 | eviction order refreshed (a touch: no I/O) |
| 62.3543 | P looked it up (turn 4): held (on GPU already) |
| 62.3545 | eviction order refreshed (a touch: no I/O) |
| 62.4861 | eviction order refreshed (a touch: no I/O) |
| 62.5052 | D2 looked it up (turn 4): held (on GPU already) |
| 62.5054 | eviction order refreshed (a touch: no I/O) |
| 64.7678 | eviction order refreshed (a touch: no I/O) |
| 64.7875 | P looked it up (turn 5): held (on GPU already) |
| 64.7877 | eviction order refreshed (a touch: no I/O) |
| 64.9280 | eviction order refreshed (a touch: no I/O) |
| 64.9480 | D3 looked it up (turn 5): held (on GPU already) |
| 64.9482 | eviction order refreshed (a touch: no I/O) |
| 67.2777 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0838 | P answered (stored) |
| 0.0943 | D1 took the request |
| 0.0968 | the server got its lookup |
| 0.0979 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.0997 | its connector had the lookup's answer |
| 0.3878 | the first token reached the proxy |
| 1.7539 | finished |
| 1.7573 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


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


# LMCache lifecycle: memory mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 15290 records from 8 processes (D1 3460, D2 3584, D3 3610, LMC_D1 661, LMC_D2 681, LMC_D3 685, LMC_P 1505, P 1104); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 258844 | 258844 |
| from the engine's own GPU prefix cache | 195248 | 101344 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 157500 |
| computed | 63596 | 0 |
| stored to LMCache (tokens) | 61952 | 155904 |

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

- D requests that read from P: 69, 157500 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 441.670 ms, p90 626.900 ms
- P held its blocks for D from its request's end to D's read done: median 464.640 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.619 GB/s (the KV it read over its time)
- the bytes: 9879 of 9879 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 851 chunks (by {'P': 242, 'D': 609}; per server {'LMC_P': 242, 'LMC_D1': 180, 'LMC_D2': 206, 'LMC_D3': 223}); written to memory 851 (24985.5 MB); to disk 0; removed from memory 638
- pulled by D or P: 0 times, from –; never pulled: 851 (265 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 3069
- every lookup's outcome per chunk: {'new: stored by this request': 851, 'held (on GPU already)': 1109}
- pulls per stored chunk: {0: 851}
- median time from a chunk's last block computed on P to the chunk in memory: 50.360 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 94.530 ms, p90 134.580 ms.

## Checks

- chunk tokens match prompt: 851 of 851
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 851 of 851
- d answers same as one engine: 62 of 69
- d answers same as one engine nixl layout: 62 of 69

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
| P | 4 | 1 | 2301 | 1216 | 1024 | 0 / 0 | 1085 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 2301 | 1216 | 1024 | 0 / 1085 | 0 | held (on GPU already) 4, new: stored by this request 4 | – | – | 1024 |
| P | 4 | 2 | 3149 | 2288 | 2048 | 0 / 0 | 861 | held (on GPU already) 8, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 3149 | 1216 | 1024 | 0 / 1933 | 0 | held (on GPU already) 4, new: stored by this request 8 | – | – | 2048 |
| P | 4 | 3 | 4481 | 3136 | 3072 | 0 / 0 | 1345 | held (on GPU already) 12, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 4481 | 1216 | 1024 | 0 / 3265 | 0 | held (on GPU already) 4, new: stored by this request 13 | – | – | 3328 |
| P | 4 | 4 | 5667 | 4480 | 4352 | 0 / 0 | 1187 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 5667 | 2416 | 2304 | 0 / 3251 | 0 | held (on GPU already) 9, new: stored by this request 13 | – | – | 3328 |
| P | 4 | 5 | 6878 | 5664 | 5632 | 0 / 0 | 1214 | held (on GPU already) 22, new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 5 | 6878 | 3264 | 3072 | 0 / 3614 | 0 | held (on GPU already) 12, new: stored by this request 14 | – | – | 3584 |
| P | 5 | 0 | 1035 | 16 | 0 | 0 / 0 | 1019 | new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 0 | 1035 | 16 | 0 | 0 / 1019 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 1 | 1944 | 1024 | 1024 | 0 / 0 | 920 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D2 | 5 | 1 | 1944 | 16 | 0 | 0 / 1928 | 0 | new: stored by this request 7 | – | – | 1792 |
| P | 5 | 2 | 2959 | 1936 | 1792 | 0 / 0 | 1023 | held (on GPU already) 7, new: stored by this request 4 | – | – | 1024 |
| D3 | 5 | 2 | 2959 | 16 | 0 | 0 / 2943 | 0 | new: stored by this request 11 | – | – | 2816 |
| P | 5 | 3 | 3848 | 2944 | 2816 | 0 / 0 | 904 | held (on GPU already) 11, new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 3 | 3848 | 1152 | 1024 | 0 / 2696 | 0 | held (on GPU already) 4, new: stored by this request 11 | – | – | 2816 |
| P | 5 | 4 | 4996 | 3840 | 3840 | 0 / 0 | 1156 | held (on GPU already) 15, new: stored by this request 4 | – | – | 1024 |
| D2 | 5 | 4 | 4996 | 2064 | 2048 | 0 / 2932 | 0 | held (on GPU already) 8, new: stored by this request 11 | – | – | 2816 |
| P | 5 | 5 | 5662 | 4992 | 4864 | 0 / 0 | 670 | held (on GPU already) 19, new: stored by this request 3 | – | – | 768 |
| D3 | 5 | 5 | 5662 | 3072 | 3072 | 0 / 2590 | 0 | held (on GPU already) 12, new: stored by this request 10 | – | – | 2560 |
| P | 6 | 0 | 814 | 16 | 0 | 0 / 0 | 798 | new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 0 | 814 | 16 | 0 | 0 / 798 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 6 | 1 | 1585 | 800 | 768 | 0 / 0 | 785 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 1 | 1585 | 16 | 0 | 0 / 1569 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 6 | 2 | 2124 | 1584 | 1536 | 0 / 0 | 540 | held (on GPU already) 6, new: stored by this request 2 | – | – | 512 |
| D3 | 6 | 2 | 2124 | 16 | 0 | 0 / 2108 | 0 | new: stored by this request 8 | – | – | 2048 |
| P | 6 | 3 | 3426 | 2112 | 2048 | 0 / 0 | 1314 | held (on GPU already) 8, new: stored by this request 5 | – | – | 1280 |
| D1 | 6 | 3 | 3426 | 928 | 768 | 0 / 2498 | 0 | held (on GPU already) 3, new: stored by this request 10 | – | – | 2560 |
| P | 6 | 4 | 4297 | 3424 | 3328 | 0 / 0 | 873 | held (on GPU already) 13, new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 4 | 4297 | 1712 | 1536 | 0 / 2585 | 0 | held (on GPU already) 6, new: stored by this request 10 | – | – | 2560 |
| P | 6 | 5 | 5161 | 4288 | 4096 | 0 / 0 | 873 | held (on GPU already) 16, new: stored by this request 4 | – | – | 1024 |
| D3 | 6 | 5 | 5161 | 2240 | 2048 | 0 / 2921 | 0 | held (on GPU already) 8, new: stored by this request 12 | – | – | 3072 |
| P | 6 | 6 | 5949 | 5152 | 5120 | 0 / 0 | 797 | held (on GPU already) 20, new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 6 | 5949 | 3552 | 3328 | 0 / 2397 | 0 | held (on GPU already) 13, new: stored by this request 10 | – | – | 2560 |
| P | 7 | 0 | 1136 | 16 | 0 | 0 / 0 | 1120 | new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 0 | 1136 | 16 | 0 | 0 / 1120 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 7 | 1 | 1909 | 1136 | 1024 | 0 / 0 | 773 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 1 | 1909 | 16 | 0 | 0 / 1893 | 0 | new: stored by this request 7 | – | – | 1792 |
| P | 7 | 2 | 3115 | 1904 | 1792 | 0 / 0 | 1211 | held (on GPU already) 7, new: stored by this request 5 | – | – | 1280 |
| D1 | 7 | 2 | 3115 | 16 | 0 | 0 / 3099 | 0 | new: stored by this request 12 | – | – | 3072 |
| P | 7 | 3 | 4099 | 3104 | 3072 | 0 / 0 | 995 | held (on GPU already) 12, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 3 | 4099 | 1248 | 1024 | 0 / 2851 | 0 | held (on GPU already) 4, new: stored by this request 12 | – | – | 3072 |
| P | 7 | 4 | 5352 | 4096 | 4096 | 0 / 0 | 1256 | held (on GPU already) 16, new: stored by this request 4 | – | – | 1024 |
| D3 | 7 | 4 | 5352 | 2032 | 1792 | 0 / 3320 | 0 | held (on GPU already) 7, new: stored by this request 13 | – | – | 3328 |
| P | 7 | 5 | 6094 | 5344 | 5120 | 0 / 0 | 750 | held (on GPU already) 20, new: stored by this request 3 | – | – | 768 |
| D1 | 7 | 5 | 6094 | 3232 | 3072 | 0 / 2862 | 0 | held (on GPU already) 12, new: stored by this request 11 | – | – | 2816 |
| P | 7 | 6 | 7005 | 6080 | 5888 | 0 / 0 | 925 | held (on GPU already) 23, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 6 | 7005 | 4224 | 4096 | 0 / 2781 | 0 | held (on GPU already) 16, new: stored by this request 11 | – | – | 2816 |
| P | 7 | 7 | 7916 | 6992 | 6912 | 0 / 0 | 924 | held (on GPU already) 27, new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 7 | 7916 | 5472 | 5376 | 0 / 2444 | 0 | held (on GPU already) 21, new: stored by this request 9 | – | – | 2304 |
| P | 8 | 0 | 1156 | 16 | 0 | 0 / 0 | 1140 | new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 0 | 1156 | 16 | 0 | 0 / 1140 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 1 | 2338 | 1152 | 1024 | 0 / 0 | 1186 | held (on GPU already) 4, new: stored by this request 5 | – | – | 1280 |
| D2 | 8 | 1 | 2338 | 16 | 0 | 0 / 2322 | 0 | new: stored by this request 9 | – | – | 2304 |
| P | 8 | 2 | 3605 | 2336 | 2304 | 0 / 0 | 1269 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D3 | 8 | 2 | 3605 | 16 | 0 | 0 / 3589 | 0 | new: stored by this request 14 | – | – | 3584 |
| P | 8 | 3 | 4188 | 3600 | 3584 | 0 / 0 | 588 | held (on GPU already) 14, new: stored by this request 2 | – | – | 512 |
| D1 | 8 | 3 | 4188 | 1280 | 1280 | 0 / 2908 | 0 | held (on GPU already) 5, new: stored by this request 11 | – | – | 2816 |
| P | 8 | 4 | 4833 | 4176 | 4096 | 0 / 0 | 657 | held (on GPU already) 16, new: stored by this request 2 | – | – | 512 |
| D2 | 8 | 4 | 4833 | 2464 | 2304 | 0 / 2369 | 0 | held (on GPU already) 9, new: stored by this request 9 | – | – | 2304 |
| P | 8 | 5 | 5699 | 4832 | 4608 | 0 / 0 | 867 | held (on GPU already) 18, new: stored by this request 4 | – | – | 1024 |
| D3 | 8 | 5 | 5699 | 3728 | 3584 | 0 / 1971 | 0 | held (on GPU already) 14, new: stored by this request 8 | – | – | 2048 |
| P | 8 | 6 | 6579 | 5696 | 5632 | 0 / 0 | 883 | held (on GPU already) 22, new: stored by this request 3 | – | – | 768 |
| D1 | 8 | 6 | 6579 | 4304 | 4096 | 0 / 2275 | 0 | held (on GPU already) 16, new: stored by this request 9 | – | – | 2304 |
| P | 8 | 7 | 7624 | 6576 | 6400 | 0 / 0 | 1048 | held (on GPU already) 25, new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 7 | 7624 | 4960 | 4864 | 0 / 2664 | 0 | held (on GPU already) 19, new: stored by this request 10 | – | – | 2560 |
| P | 9 | 0 | 464 | 16 | 0 | 0 / 0 | 448 | new: stored by this request 1 | – | – | 256 |
| D3 | 9 | 0 | 464 | 16 | 0 | 0 / 448 | 0 | new: stored by this request 1 | – | – | 256 |
| P | 9 | 1 | 1314 | 464 | 256 | 0 / 0 | 850 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D1 | 9 | 1 | 1314 | 16 | 0 | 0 / 1298 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 9 | 2 | 2331 | 1312 | 1280 | 0 / 0 | 1019 | held (on GPU already) 5, new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 2 | 2331 | 16 | 0 | 0 / 2315 | 0 | new: stored by this request 9 | – | – | 2304 |
| P | 9 | 3 | 3627 | 2320 | 2304 | 0 / 0 | 1307 | held (on GPU already) 9, new: stored by this request 5 | – | – | 1280 |
| D3 | 9 | 3 | 3627 | 576 | 512 | 0 / 3051 | 0 | held (on GPU already) 2, new: stored by this request 12 | – | – | 3072 |
| P | 9 | 4 | 4891 | 3616 | 3584 | 0 / 0 | 1275 | held (on GPU already) 14, new: stored by this request 5 | – | – | 1280 |
| D1 | 9 | 4 | 4891 | 1440 | 1280 | 0 / 3451 | 0 | held (on GPU already) 5, new: stored by this request 14 | – | – | 3584 |
| P | 9 | 5 | 5845 | 4880 | 4864 | 0 / 0 | 965 | held (on GPU already) 19, new: stored by this request 3 | – | – | 768 |
| D2 | 9 | 5 | 5845 | 2448 | 2304 | 0 / 3397 | 0 | held (on GPU already) 9, new: stored by this request 13 | – | – | 3328 |
| P | 9 | 6 | 6534 | 5840 | 5632 | 0 / 0 | 694 | held (on GPU already) 22, new: stored by this request 3 | – | – | 768 |
| D3 | 9 | 6 | 6534 | 3744 | 3584 | 0 / 2790 | 0 | held (on GPU already) 14, new: stored by this request 11 | – | – | 2816 |

## One chunk's life: 5e868214004923dc

Conversation 1, turn 0, tokens 0-256 (system, doc 850931827_11086-12414-0-1328); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0048 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request ca1c7483) |
| 9.5742 | P looked it up (turn 0): new: stored by this request |
| 9.5774 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request c54cbee7) |
| 9.5777 | P's connector planned the store |
| 9.5964 | P's worker sent it to the server |
| 9.5975 | the server bound the chunk to its tokens (mp.tokens) |
| 9.5978 | server memory (L1) reserved |
| 9.6659 | eviction order refreshed (a touch: no I/O) |
| 9.6729 | written to memory (L1) |
| 11.3179 | P looked it up (turn 1): held (on GPU already) |
| 11.3180 | eviction order refreshed (a touch: no I/O) |
| 11.4456 | eviction order refreshed (a touch: no I/O) |
| 13.3915 | P looked it up (turn 2): held (on GPU already) |
| 13.3916 | eviction order refreshed (a touch: no I/O) |
| 13.5091 | eviction order refreshed (a touch: no I/O) |
| 15.7189 | P looked it up (turn 3): held (on GPU already) |
| 15.7191 | eviction order refreshed (a touch: no I/O) |
| 15.8238 | eviction order refreshed (a touch: no I/O) |
| 18.0246 | P looked it up (turn 4): held (on GPU already) |
| 18.0249 | eviction order refreshed (a touch: no I/O) |
| 18.1879 | eviction order refreshed (a touch: no I/O) |
| 20.4363 | P looked it up (turn 5): held (on GPU already) |
| 20.4366 | eviction order refreshed (a touch: no I/O) |
| 20.5802 | eviction order refreshed (a touch: no I/O) |
| 22.8742 | P looked it up (turn 6): held (on GPU already) |
| 22.8747 | eviction order refreshed (a touch: no I/O) |
| 23.0432 | eviction order refreshed (a touch: no I/O) |
| 53.2167 | P looked it up (turn 0): held (on GPU already) |
| 53.2168 | eviction order refreshed (a touch: no I/O) |
| 53.2365 | eviction order refreshed (a touch: no I/O) |
| 54.7425 | P looked it up (turn 1): held (on GPU already) |
| 54.7426 | eviction order refreshed (a touch: no I/O) |
| 54.8432 | eviction order refreshed (a touch: no I/O) |
| 56.5334 | P looked it up (turn 2): held (on GPU already) |
| 56.5336 | eviction order refreshed (a touch: no I/O) |
| 56.6235 | eviction order refreshed (a touch: no I/O) |
| 58.4839 | P looked it up (turn 3): held (on GPU already) |
| 58.4840 | eviction order refreshed (a touch: no I/O) |
| 58.6264 | eviction order refreshed (a touch: no I/O) |
| 60.7740 | P looked it up (turn 4): held (on GPU already) |
| 60.7742 | eviction order refreshed (a touch: no I/O) |
| 60.9051 | eviction order refreshed (a touch: no I/O) |
| 63.0971 | P looked it up (turn 5): held (on GPU already) |
| 63.0973 | eviction order refreshed (a touch: no I/O) |
| 63.2375 | eviction order refreshed (a touch: no I/O) |
| 96.1344 | removed from memory (L1) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0829 | P answered (stored) |
| 0.0926 | D1 took the request |
| 0.0961 | the server got its lookup |
| 0.0974 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.0983 | its connector had the lookup's answer |
| 0.3844 | the first token reached the proxy |
| 1.7507 | finished |
| 1.7537 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 15075 records from 8 processes (D1 3464, D2 3473, D3 3614, LMC_D1 646, LMC_D2 649, LMC_D3 663, LMC_P 1463, P 1103); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 87764 | 87764 |
| from the engine's own GPU prefix cache | 23248 | 10096 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 77668 |
| computed | 64516 | 0 |
| stored to LMCache (tokens) | 62208 | 73216 |

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

- D requests that read from P: 69, 77668 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 207.150 ms, p90 280.750 ms
- P held its blocks for D from its request's end to D's read done: median 232.770 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.615 GB/s (the KV it read over its time)
- the bytes: 4883 of 4883 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 529 chunks (by {'P': 243, 'D': 286}; per server {'LMC_P': 243, 'LMC_D1': 94, 'LMC_D2': 97, 'LMC_D3': 95}); written to memory 529 (15531.5 MB); to disk 0; removed from memory 318
- pulled by D or P: 0 times, from –; never pulled: 529 (213 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 699
- every lookup's outcome per chunk: {'new: stored by this request': 529, 'held (on GPU already)': 85}
- pulls per stored chunk: {0: 529}
- median time from a chunk's last block computed on P to the chunk in memory: 37.500 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 82.240 ms, p90 107.640 ms.

## Checks

- chunk tokens match prompt: 529 of 529
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 529 of 529
- d answers same as one engine: 52 of 69
- d answers same as one engine nixl layout: 52 of 69

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
| P | 4 | 1 | 1256 | 176 | 0 | 0 / 0 | 1080 | new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 1256 | 160 | 0 | 0 / 1096 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 4 | 2 | 1041 | 160 | 0 | 0 / 0 | 881 | new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 1041 | 160 | 0 | 0 / 881 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 4 | 3 | 1667 | 320 | 256 | 0 / 0 | 1347 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 1667 | 160 | 0 | 0 / 1507 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 4 | 4 | 1676 | 464 | 256 | 0 / 0 | 1212 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 1676 | 160 | 0 | 0 / 1516 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 4 | 5 | 1841 | 608 | 512 | 0 / 0 | 1233 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 4 | 5 | 1841 | 320 | 256 | 0 / 1521 | 0 | held (on GPU already) 1, new: stored by this request 6 | – | – | 1536 |
| P | 5 | 0 | 1035 | 16 | 0 | 0 / 0 | 1019 | new: stored by this request 4 | – | – | 1024 |
| D1 | 5 | 0 | 1035 | 16 | 0 | 0 / 1019 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 1 | 950 | 16 | 0 | 0 / 0 | 934 | new: stored by this request 3 | – | – | 768 |
| D2 | 5 | 1 | 950 | 16 | 0 | 0 / 934 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 5 | 2 | 1199 | 160 | 0 | 0 / 0 | 1039 | new: stored by this request 4 | – | – | 1024 |
| D3 | 5 | 2 | 1199 | 16 | 0 | 0 / 1183 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 3 | 1212 | 304 | 256 | 0 / 0 | 908 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D1 | 5 | 3 | 1212 | 16 | 0 | 0 / 1196 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 5 | 4 | 1613 | 448 | 256 | 0 / 0 | 1165 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 5 | 4 | 1613 | 160 | 0 | 0 / 1453 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 5 | 5 | 1272 | 592 | 512 | 0 / 0 | 680 | held (on GPU already) 2, new: stored by this request 2 | – | – | 512 |
| D3 | 5 | 5 | 1272 | 304 | 256 | 0 / 968 | 0 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| P | 6 | 0 | 814 | 16 | 0 | 0 / 0 | 798 | new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 0 | 814 | 16 | 0 | 0 / 798 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 6 | 1 | 816 | 16 | 0 | 0 / 0 | 800 | new: stored by this request 3 | – | – | 768 |
| D2 | 6 | 1 | 816 | 16 | 0 | 0 / 800 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 6 | 2 | 726 | 160 | 0 | 0 / 0 | 566 | new: stored by this request 2 | – | – | 512 |
| D3 | 6 | 2 | 726 | 16 | 0 | 0 / 710 | 0 | new: stored by this request 2 | – | – | 512 |
| P | 6 | 3 | 1629 | 304 | 256 | 0 / 0 | 1325 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 6 | 3 | 1629 | 16 | 0 | 0 / 1613 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 6 | 4 | 1336 | 448 | 256 | 0 / 0 | 888 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 6 | 4 | 1336 | 160 | 0 | 0 / 1176 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 6 | 5 | 1466 | 592 | 512 | 0 / 0 | 874 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 6 | 5 | 1466 | 304 | 256 | 0 / 1162 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 6 | 6 | 1530 | 720 | 512 | 0 / 0 | 810 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D1 | 6 | 6 | 1530 | 448 | 256 | 0 / 1082 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 7 | 0 | 1136 | 16 | 0 | 0 / 0 | 1120 | new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 0 | 1136 | 16 | 0 | 0 / 1120 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 7 | 1 | 815 | 16 | 0 | 0 / 0 | 799 | new: stored by this request 3 | – | – | 768 |
| D3 | 7 | 1 | 815 | 16 | 0 | 0 / 799 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 7 | 2 | 1405 | 160 | 0 | 0 / 0 | 1245 | new: stored by this request 5 | – | – | 1280 |
| D1 | 7 | 2 | 1405 | 16 | 0 | 0 / 1389 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 7 | 3 | 1353 | 320 | 256 | 0 / 0 | 1033 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 7 | 3 | 1353 | 16 | 0 | 0 / 1337 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 7 | 4 | 1770 | 496 | 256 | 0 / 0 | 1274 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D3 | 7 | 4 | 1770 | 160 | 0 | 0 / 1610 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 7 | 5 | 1403 | 640 | 512 | 0 / 0 | 763 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D1 | 7 | 5 | 1403 | 320 | 256 | 0 / 1083 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 7 | 6 | 1738 | 784 | 768 | 0 / 0 | 954 | held (on GPU already) 3, new: stored by this request 3 | – | – | 768 |
| D2 | 7 | 6 | 1738 | 496 | 256 | 0 / 1242 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 7 | 7 | 1886 | 944 | 768 | 0 / 0 | 942 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D3 | 7 | 7 | 1886 | 640 | 512 | 0 / 1246 | 0 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| P | 8 | 0 | 1156 | 16 | 0 | 0 / 0 | 1140 | new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 0 | 1156 | 16 | 0 | 0 / 1140 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 1 | 1225 | 16 | 0 | 0 / 0 | 1209 | new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 1 | 1225 | 16 | 0 | 0 / 1209 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 2 | 1448 | 160 | 0 | 0 / 0 | 1288 | new: stored by this request 5 | – | – | 1280 |
| D3 | 8 | 2 | 1448 | 16 | 0 | 0 / 1432 | 0 | new: stored by this request 5 | – | – | 1280 |
| P | 8 | 3 | 902 | 304 | 256 | 0 / 0 | 598 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| D1 | 8 | 3 | 902 | 16 | 0 | 0 / 886 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 8 | 4 | 1111 | 448 | 256 | 0 / 0 | 663 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D2 | 8 | 4 | 1111 | 160 | 0 | 0 / 951 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 8 | 5 | 1473 | 592 | 512 | 0 / 0 | 881 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 8 | 5 | 1473 | 304 | 256 | 0 / 1169 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| P | 8 | 6 | 1627 | 736 | 512 | 0 / 0 | 891 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D1 | 8 | 6 | 1627 | 448 | 256 | 0 / 1179 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 8 | 7 | 1932 | 864 | 768 | 0 / 0 | 1068 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D2 | 8 | 7 | 1932 | 592 | 512 | 0 / 1340 | 0 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| P | 9 | 0 | 464 | 16 | 0 | 0 / 0 | 448 | new: stored by this request 1 | – | – | 256 |
| D3 | 9 | 0 | 464 | 16 | 0 | 0 / 448 | 0 | new: stored by this request 1 | – | – | 256 |
| P | 9 | 1 | 892 | 32 | 0 | 0 / 0 | 860 | new: stored by this request 3 | – | – | 768 |
| D1 | 9 | 1 | 892 | 32 | 0 | 0 / 860 | 0 | new: stored by this request 3 | – | – | 768 |
| P | 9 | 2 | 1197 | 160 | 0 | 0 / 0 | 1037 | new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 2 | 1197 | 32 | 0 | 0 / 1165 | 0 | new: stored by this request 4 | – | – | 1024 |
| P | 9 | 3 | 1615 | 304 | 256 | 0 / 0 | 1311 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D3 | 9 | 3 | 1615 | 32 | 0 | 0 / 1583 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 9 | 4 | 1719 | 448 | 256 | 0 / 0 | 1271 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 9 | 4 | 1719 | 160 | 0 | 0 / 1559 | 0 | new: stored by this request 6 | – | – | 1536 |
| P | 9 | 5 | 1557 | 576 | 512 | 0 / 0 | 981 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 5 | 1557 | 304 | 256 | 0 / 1253 | 0 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| P | 9 | 6 | 1438 | 720 | 512 | 0 / 0 | 718 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 9 | 6 | 1438 | 448 | 256 | 0 / 990 | 0 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |

## One chunk's life: aa54d2c3db5e0d3a

Conversation 2, turn 3, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1, newline 1, user 2, assistant 2, answer 2); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0048 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 4873f2a5) |
| 23.7602 | P looked it up (turn 3): new: stored by this request |
| 23.7634 | P computed and cached block 15 (GPU block 878, hash 5c2a3032cc6dadfb, request 5472fd56) |
| 23.7637 | P's connector planned the store |
| 23.7832 | P's worker sent it to the server |
| 23.7846 | the server bound the chunk to its tokens (mp.tokens) |
| 23.7850 | server memory (L1) reserved |
| 23.8260 | eviction order refreshed (a touch: no I/O) |
| 23.8322 | written to memory (L1) |
| 25.4409 | P looked it up (turn 4): held (on GPU already) |
| 25.4410 | eviction order refreshed (a touch: no I/O) |
| 25.5444 | eviction order refreshed (a touch: no I/O) |
| 27.2692 | P looked it up (turn 5): held (on GPU already) |
| 27.2693 | eviction order refreshed (a touch: no I/O) |
| 27.3459 | eviction order refreshed (a touch: no I/O) |
| 29.0119 | P looked it up (turn 6): held (on GPU already) |
| 29.0121 | eviction order refreshed (a touch: no I/O) |
| 29.0988 | eviction order refreshed (a touch: no I/O) |
| 30.8063 | P looked it up (turn 7): held (on GPU already) |
| 30.8064 | eviction order refreshed (a touch: no I/O) |
| 30.8852 | eviction order refreshed (a touch: no I/O) |
| 32.5528 | P looked it up (turn 8): held (on GPU already) |
| 32.5529 | eviction order refreshed (a touch: no I/O) |
| 32.6594 | eviction order refreshed (a touch: no I/O) |
| 55.3089 | removed from memory (L1) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0799 | P answered (stored) |
| 0.0917 | D1 took the request |
| 0.0949 | the server got its lookup |
| 0.0960 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.0971 | its connector had the lookup's answer |
| 0.3975 | the first token reached the proxy |
| 1.7637 | finished |
| 1.7670 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


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


# LMCache lifecycle: memory mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, 3f5fa378, 5c13daa9, 694e275f, aea2634b, 79f0d053, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 268a9a96a324.

Trace: 14486 records from 5 processes (D1 3422, D2 3427, D3 3560, LMC 2981, P 1096); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 69 | 69 |
| prompt tokens | 87764 | 87764 |
| from the engine's own GPU prefix cache | 23248 | 10096 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 77668 |
| computed | 64516 | 0 |
| stored to LMCache (tokens) | 62208 | – |

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

- D requests that read from P: 69, 77668 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 209.220 ms, p90 286.470 ms
- P held its blocks for D from its request's end to D's read done: median 234.690 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.610 GB/s (the KV it read over its time)
- the bytes: 4883 of 4883 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 243 chunks (by {'P': 243}); written to memory 243 (7134.5 MB); to disk 0; removed from memory 47
- pulled by D or P: 0 times, from –; never pulled: 243 (87 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 699
- every lookup's outcome per chunk: {'new: stored by this request': 243, 'hit, not loaded': 286, 'held (on GPU already)': 85}
- pulls per stored chunk: {0: 243}
- median time from a chunk's last block computed on P to the chunk in memory: 92.180 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 83.060 ms, p90 109.350 ms.

## Checks

- chunk tokens match prompt: 243 of 243
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 243 of 243
- d answers same as one engine: 52 of 69
- d answers same as one engine nixl layout: 52 of 69

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
| D2 | 1 | 2 | 1347 | 16 | 1280 | 0 / 1331 | 0 | hit, not loaded 5 | – | – | 0 |
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
| D3 | 8 | 2 | 1448 | 16 | 1280 | 0 / 1432 | 0 | hit, not loaded 5 | – | – | 0 |
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
| D3 | 9 | 3 | 1615 | 32 | 1536 | 0 / 1583 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 9 | 4 | 1719 | 448 | 256 | 0 / 0 | 1271 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 9 | 4 | 1719 | 160 | 1536 | 0 / 1559 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 9 | 5 | 1557 | 576 | 512 | 0 / 0 | 981 | held (on GPU already) 2, new: stored by this request 4 | – | – | 1024 |
| D2 | 9 | 5 | 1557 | 304 | 1536 | 0 / 1253 | 0 | held (on GPU already) 1, hit, not loaded 5 | – | – | 0 |
| P | 9 | 6 | 1438 | 720 | 512 | 0 / 0 | 718 | held (on GPU already) 2, new: stored by this request 3 | – | – | 768 |
| D3 | 9 | 6 | 1438 | 448 | 1280 | 0 / 990 | 0 | held (on GPU already) 1, hit, not loaded 4 | – | – | 0 |

## One chunk's life: aa54d2c3db5e0d3a

Conversation 2, turn 3, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1, newline 1, user 2, assistant 2, answer 2); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0048 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 3daecb5d) |
| 23.8529 | P looked it up (turn 3): new: stored by this request |
| 23.8562 | P computed and cached block 15 (GPU block 878, hash 5c2a3032cc6dadfb, request 893e90d7) |
| 23.8564 | P's connector planned the store |
| 23.8755 | P's worker sent it to the server |
| 23.8769 | the server bound the chunk to its tokens (mp.tokens) |
| 23.8774 | server memory (L1) reserved |
| 23.9190 | eviction order refreshed (a touch: no I/O) |
| 23.9239 | written to memory (L1) |
| 23.9267 | D1 looked it up (turn 3): hit, not loaded |
| 25.5136 | eviction order refreshed (a touch: no I/O) |
| 25.5335 | P looked it up (turn 4): held (on GPU already) |
| 25.5336 | eviction order refreshed (a touch: no I/O) |
| 25.6373 | eviction order refreshed (a touch: no I/O) |
| 25.6478 | D2 looked it up (turn 4): hit, not loaded |
| 27.3463 | eviction order refreshed (a touch: no I/O) |
| 27.3632 | P looked it up (turn 5): held (on GPU already) |
| 27.3632 | eviction order refreshed (a touch: no I/O) |
| 27.4405 | eviction order refreshed (a touch: no I/O) |
| 27.4497 | D3 looked it up (turn 5): hit, not loaded |
| 29.0794 | eviction order refreshed (a touch: no I/O) |
| 29.1020 | P looked it up (turn 6): held (on GPU already) |
| 29.1021 | eviction order refreshed (a touch: no I/O) |
| 29.1886 | eviction order refreshed (a touch: no I/O) |
| 29.1980 | D1 looked it up (turn 6): held (on GPU already) |
| 29.1981 | eviction order refreshed (a touch: no I/O) |
| 30.8698 | eviction order refreshed (a touch: no I/O) |
| 30.8916 | P looked it up (turn 7): held (on GPU already) |
| 30.8917 | eviction order refreshed (a touch: no I/O) |
| 30.9709 | eviction order refreshed (a touch: no I/O) |
| 30.9821 | D2 looked it up (turn 7): held (on GPU already) |
| 30.9821 | eviction order refreshed (a touch: no I/O) |
| 32.6191 | eviction order refreshed (a touch: no I/O) |
| 32.6457 | P looked it up (turn 8): held (on GPU already) |
| 32.6458 | eviction order refreshed (a touch: no I/O) |
| 32.7541 | eviction order refreshed (a touch: no I/O) |
| 32.7654 | D3 looked it up (turn 8): held (on GPU already) |
| 32.7654 | eviction order refreshed (a touch: no I/O) |
| 34.5052 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0778 | P answered (stored) |
| 0.1379 | D1 took the request |
| 0.1407 | the server got its lookup |
| 0.1418 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1422 | its connector had the lookup's answer |
| 0.4419 | the first token reached the proxy |
| 1.8100 | finished |
| 1.8134 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
