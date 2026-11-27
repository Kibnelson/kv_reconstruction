# PD matrix lmctrace on gpu-grad-01, job 30793 (completed), reviewed from the kept results


# LMCache lifecycle: the runs side by side

| | disk, NIXL, per-engine, carry-forward | disk, NIXL, per-engine, chat RAG | disk, NIXL, shared, carry-forward | disk, NIXL, shared, chat RAG | memory, NIXL, per-engine, carry-forward | memory, NIXL, per-engine, chat RAG | memory, NIXL, shared, carry-forward | memory, NIXL, shared, chat RAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LMCache server lasted the run | True | True | True | True | True | True | True | True |
| LMCache server errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P to D | nixl | nixl | nixl | nixl | nixl | nixl | nixl | nixl |
| LMCache servers | per_engine | per_engine | shared | shared | per_engine | per_engine | shared | shared |
| prompt layout | interleaved | history_first | interleaved | history_first | interleaved | history_first | interleaved | history_first |
| passages | all | all | all | all | all | all | all | all |
| conversations | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| disk reads | retain | retain | retain | retain | retain | retain | retain | retain |
| chunks bound by the engines' stores | 391 | 249 | 113 | 114 | 391 | 249 | 113 | 114 |
| chunks stored by | {'P': 113, 'D': 278} | {'P': 114, 'D': 135} | {'P': 113} | {'P': 114} | {'P': 113, 'D': 278} | {'P': 114, 'D': 135} | {'P': 113} | {'P': 114} |
| chunks written to memory | 391 | 249 | 113 | 114 | 391 | 249 | 113 | 114 |
| stored MB | 11479.800 | 7310.700 | 3317.700 | 3347.100 | 11479.800 | 7310.700 | 3317.700 | 3347.100 |
| chunks to disk | 391 | 249 | 113 | 114 | 0 | 0 | 0 | 0 |
| chunks removed from memory | 391 | 249 | 113 | 114 | 173 | 58 | 0 | 0 |
| pulls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pulled from | {} | {} | {} | {} | {} | {} | {} | {} |
| never pulled | 391 | 249 | 113 | 114 | 391 | 249 | 113 | 114 |
| D tokens from own GPU | 45024 | 4080 | 45024 | 4080 | 45024 | 4080 | 45024 | 4080 |
| D tokens loaded from LMCache | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D tokens read from P over NIXL | 72366 | 36824 | 72366 | 36824 | 72366 | 36824 | 72366 | 36824 |
| NIXL read ms (median) | 409.740 | 199.820 | 411.050 | 201.640 | 415.840 | 200.800 | 414.830 | 200.620 |
| NIXL read ms (p90) | 658.300 | 286.330 | 659.680 | 293.820 | 661.630 | 281.900 | 662.970 | 281.520 |
| NIXL read GB/s (median) | 0.619 | 0.613 | 0.618 | 0.610 | 0.614 | 0.616 | 0.614 | 0.615 |
| NIXL blocks = P's bytes | 220 of 4541 | 132 of 2318 | 220 of 4541 | 132 of 2318 | 220 of 4541 | 132 of 2318 | 220 of 4541 | 132 of 2318 |
| D's answers = one engine's | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 | 26 of 33 |
| D's answers = one engine's, NIXL's layout | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 | 24 of 33 |
| UCX transports | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp |
| P held its blocks for D, ms (median) | 462.200 | 235.140 | 525.700 | 252.340 | 460.670 | 234.000 | 460.940 | 231.630 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 93.160 | 82.920 | 93.080 | 82.590 | 92.550 | 81.740 | 91.890 | 82.490 |
| memory to disk ms (median) | 325.130 | 168.590 | 228.390 | 169.430 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 391 of 391; 0 of 0; 391 of 391; 26 of 33; 24 of 33; 391 of 391; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 113 files, 3.32 GB; {'_default': '0 objects, 0 bytes'}; 76 files, 2.23 GB; {'_default': '0 objects, 0 bytes'}; 95 files, 2.79 GB; {'_default': '0 objects, 0 bytes'}; 107 files, 3.14 GB | 249 of 249; 0 of 0; 249 of 249; 26 of 33; 24 of 33; 249 of 249; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 114 files, 3.35 GB; {'_default': '0 objects, 0 bytes'}; 43 files, 1.26 GB; {'_default': '0 objects, 0 bytes'}; 44 files, 1.29 GB; {'_default': '0 objects, 0 bytes'}; 48 files, 1.41 GB | 113 of 113; 0 of 0; 113 of 113; 26 of 33; 24 of 33; 113 of 113; 0 of 0; 0; 93 of 93 disk reads still in memory at the end (prefetch policy retain); {'_default': '93 objects, 2730491904 bytes'}; 113 files, 3.32 GB | 114 of 114; 0 of 0; 114 of 114; 26 of 33; 24 of 33; 114 of 114; 0 of 0; 0; 14 of 14 disk reads still in memory at the end (prefetch policy retain); {'_default': '14 objects, 411041792 bytes'}; 114 files, 3.35 GB | 391 of 391; 0 of 0; 391 of 391; 26 of 33; 24 of 33 | 249 of 249; 0 of 0; 249 of 249; 26 of 33; 24 of 33 | 113 of 113; 0 of 0; 113 of 113; 26 of 33; 24 of 33 | 114 of 114; 0 of 0; 114 of 114; 26 of 33; 24 of 33 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_222259, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_221252, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_220243, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_215232, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_221747, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_220758, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_215729, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_214637


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


# LMCache lifecycle: disk mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 7614 records from 5 processes (D1 1710, D2 1636, D3 1841, LMC 1881, P 546); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 33 | 33 |
| prompt tokens | 40904 | 40904 |
| from the engine's own GPU prefix cache | 10592 | 4080 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 36824 |
| computed | 30312 | 0 |
| stored to LMCache (tokens) | 29184 | 1536 |

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

- D requests that read from P: 33, 36824 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 201.640 ms, p90 293.820 ms
- P held its blocks for D from its request's end to D's read done: median 252.340 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.610 GB/s (the KV it read over its time)
- the bytes: 132 of 2318 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 33; DIFFERS: D1 block 43 from P's 34; DIFFERS: D1 block 44 from P's 35; DIFFERS: D1 block 45 from P's 36; DIFFERS: D1 block 46 from P's 37; DIFFERS: D1 block 47 from P's 38; DIFFERS: D1 block 48 from P's 39
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 114 chunks (by {'P': 114}); written to memory 114 (3347.1 MB); to disk 114; removed from memory 114
- pulled by D or P: 0 times, from –; never pulled: 114 (41 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 14; touches (eviction order refreshed, no I/O): 323
- every lookup's outcome per chunk: {'new: stored by this request': 114, 'hit, not loaded': 129, 'held (on GPU already)': 37, 'miss: not yet stored': 6}
- pulls per stored chunk: {0: 114}
- median time from a chunk's last block computed on P to the chunk in memory: 94.120 ms; memory to disk: 169.430 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 82.590 ms, p90 121.830 ms.

## Checks

- chunk tokens match prompt: 114 of 114
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 114 of 114
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33
- disk mode on disk then out of memory: 114 of 114
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 14 of 14 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '14 objects, 411041792 bytes'}
- disk mode files at end: 114 files, 3.35 GB

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
| D1 | 2 | 3 | 957 | 16 | 0 | 0 / 941 | 0 | miss: not yet stored 3 | – | – | 768 |
| P | 2 | 4 | 1516 | 336 | 256 | 0 / 0 | 1180 | held (on GPU already) 1, new: stored by this request 4 | – | – | 1024 |
| D2 | 2 | 4 | 1516 | 48 | 1280 | 0 / 1468 | 0 | hit, not loaded 5 | – | – | 0 |
| P | 2 | 5 | 1270 | 496 | 256 | 0 / 0 | 774 | held (on GPU already) 1, new: stored by this request 3 | – | – | 768 |
| D3 | 2 | 5 | 1270 | 192 | 256 | 0 / 1078 | 0 | hit, not loaded 1, miss: not yet stored 3 | – | – | 768 |
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
| P | 4 | 1 | 1256 | 80 | 0 | 0 / 0 | 1176 | new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 1256 | 80 | 1024 | 0 / 1176 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 4 | 2 | 1041 | 160 | 0 | 0 / 0 | 881 | new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 1041 | 80 | 1024 | 0 / 961 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 4 | 3 | 1667 | 320 | 256 | 0 / 0 | 1347 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 1667 | 80 | 1536 | 0 / 1587 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 4 | 4 | 1676 | 464 | 256 | 0 / 0 | 1212 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 1676 | 160 | 1536 | 0 / 1516 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 4 | 5 | 1841 | 608 | 512 | 0 / 0 | 1233 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 4 | 5 | 1841 | 320 | 1792 | 0 / 1521 | 0 | held (on GPU already) 1, hit, not loaded 6 | – | – | 0 |

## One chunk's life: aa54d2c3db5e0d3a

Conversation 2, turn 3, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1, newline 1, user 2, assistant 2, answer 2); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0068 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request c45b7731) |
| 23.8641 | P looked it up (turn 3): new: stored by this request |
| 23.8674 | P computed and cached block 15 (GPU block 878, hash 5c2a3032cc6dadfb, request c88e2d73) |
| 23.8677 | P's connector planned the store |
| 23.8871 | P's worker sent it to the server |
| 23.8883 | the server bound the chunk to its tokens (mp.tokens) |
| 23.8888 | server memory (L1) reserved |
| 23.9300 | eviction order refreshed (a touch: no I/O) |
| 23.9360 | D1 looked it up (turn 3): miss: not yet stored |
| 23.9375 | written to memory (L1) |
| 24.0435 | written to disk (L2) |
| 24.0438 | removed from memory (L1) |
| 24.1527 | removed from memory (L1) |
| 25.5295 | eviction order refreshed (a touch: no I/O) |
| 25.5450 | P looked it up (turn 4): held (on GPU already) |
| 25.5451 | eviction order refreshed (a touch: no I/O) |
| 25.6502 | eviction order refreshed (a touch: no I/O) |
| 25.6597 | D2 looked it up (turn 4): hit, not loaded |
| 25.6859 | read from disk (L2) |
| 27.3834 | eviction order refreshed (a touch: no I/O) |
| 27.3960 | P looked it up (turn 5): held (on GPU already) |
| 27.3960 | eviction order refreshed (a touch: no I/O) |
| 27.4724 | eviction order refreshed (a touch: no I/O) |
| 27.4794 | D3 looked it up (turn 5): hit, not loaded |
| 29.1191 | eviction order refreshed (a touch: no I/O) |
| 29.1359 | P looked it up (turn 6): held (on GPU already) |
| 29.1360 | eviction order refreshed (a touch: no I/O) |
| 29.2227 | eviction order refreshed (a touch: no I/O) |
| 29.2306 | D1 looked it up (turn 6): held (on GPU already) |
| 29.2307 | eviction order refreshed (a touch: no I/O) |
| 30.9344 | eviction order refreshed (a touch: no I/O) |
| 30.9556 | P looked it up (turn 7): held (on GPU already) |
| 30.9557 | eviction order refreshed (a touch: no I/O) |
| 31.0330 | eviction order refreshed (a touch: no I/O) |
| 31.0432 | D2 looked it up (turn 7): held (on GPU already) |
| 31.0433 | eviction order refreshed (a touch: no I/O) |
| 32.7099 | eviction order refreshed (a touch: no I/O) |
| 32.7334 | P looked it up (turn 8): held (on GPU already) |
| 32.7335 | eviction order refreshed (a touch: no I/O) |
| 32.8420 | eviction order refreshed (a touch: no I/O) |
| 32.8520 | D3 looked it up (turn 8): held (on GPU already) |
| 32.8522 | eviction order refreshed (a touch: no I/O) |
| 34.6102 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0815 | P answered (stored) |
| 0.0939 | D1 took the request |
| 0.0970 | the server got its lookup |
| 0.0985 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1017 | its connector had the lookup's answer |
| 0.4001 | the first token reached the proxy |
| 1.7692 | finished |
| 1.7727 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 7206 records from 8 processes (D1 1574, D2 1708, D3 1735, LMC_D1 303, LMC_D2 324, LMC_D3 325, LMC_P 712, P 525); hook errors 0; unreadable lines 0.

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

- D requests that read from P: 33, 72366 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 415.840 ms, p90 661.630 ms
- P held its blocks for D from its request's end to D's read done: median 460.670 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.614 GB/s (the KV it read over its time)
- the bytes: 220 of 4541 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 2; DIFFERS: D1 block 43 from P's 3; DIFFERS: D1 block 44 from P's 4; DIFFERS: D1 block 45 from P's 5; DIFFERS: D1 block 46 from P's 6; DIFFERS: D1 block 47 from P's 7; DIFFERS: D1 block 48 from P's 8
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 391 chunks (by {'P': 113, 'D': 278}; per server {'LMC_P': 113, 'LMC_D1': 76, 'LMC_D2': 95, 'LMC_D3': 107}); written to memory 391 (11479.8 MB); to disk 0; removed from memory 173
- pulled by D or P: 0 times, from –; never pulled: 391 (126 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 1377
- every lookup's outcome per chunk: {'new: stored by this request': 391, 'held (on GPU already)': 493}
- pulls per stored chunk: {0: 391}
- median time from a chunk's last block computed on P to the chunk in memory: 52.900 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 92.550 ms, p90 144.110 ms.

## Checks

- chunk tokens match prompt: 391 of 391
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 391 of 391
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33

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
| 0.0053 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 5510ce95) |
| 9.5825 | P looked it up (turn 0): new: stored by this request |
| 9.5857 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request 32d2466d) |
| 9.5860 | P's connector planned the store |
| 9.6044 | P's worker sent it to the server |
| 9.6057 | the server bound the chunk to its tokens (mp.tokens) |
| 9.6064 | server memory (L1) reserved |
| 9.6748 | eviction order refreshed (a touch: no I/O) |
| 9.6830 | written to memory (L1) |
| 11.3279 | P looked it up (turn 1): held (on GPU already) |
| 11.3280 | eviction order refreshed (a touch: no I/O) |
| 11.4564 | eviction order refreshed (a touch: no I/O) |
| 13.4076 | P looked it up (turn 2): held (on GPU already) |
| 13.4077 | eviction order refreshed (a touch: no I/O) |
| 13.5256 | eviction order refreshed (a touch: no I/O) |
| 15.7246 | P looked it up (turn 3): held (on GPU already) |
| 15.7247 | eviction order refreshed (a touch: no I/O) |
| 15.8296 | eviction order refreshed (a touch: no I/O) |
| 18.0277 | P looked it up (turn 4): held (on GPU already) |
| 18.0279 | eviction order refreshed (a touch: no I/O) |
| 18.1908 | eviction order refreshed (a touch: no I/O) |
| 20.4465 | P looked it up (turn 5): held (on GPU already) |
| 20.4467 | eviction order refreshed (a touch: no I/O) |
| 20.5921 | eviction order refreshed (a touch: no I/O) |
| 22.8813 | P looked it up (turn 6): held (on GPU already) |
| 22.8815 | eviction order refreshed (a touch: no I/O) |
| 23.0512 | eviction order refreshed (a touch: no I/O) |
| 53.3402 | P looked it up (turn 0): held (on GPU already) |
| 53.3403 | eviction order refreshed (a touch: no I/O) |
| 53.3603 | eviction order refreshed (a touch: no I/O) |
| 54.8697 | P looked it up (turn 1): held (on GPU already) |
| 54.8699 | eviction order refreshed (a touch: no I/O) |
| 54.9782 | eviction order refreshed (a touch: no I/O) |
| 56.6874 | P looked it up (turn 2): held (on GPU already) |
| 56.6876 | eviction order refreshed (a touch: no I/O) |
| 56.7762 | eviction order refreshed (a touch: no I/O) |
| 58.6608 | P looked it up (turn 3): held (on GPU already) |
| 58.6610 | eviction order refreshed (a touch: no I/O) |
| 58.8036 | eviction order refreshed (a touch: no I/O) |
| 60.9642 | P looked it up (turn 4): held (on GPU already) |
| 60.9644 | eviction order refreshed (a touch: no I/O) |
| 61.0948 | eviction order refreshed (a touch: no I/O) |
| 63.2850 | P looked it up (turn 5): held (on GPU already) |
| 63.2853 | eviction order refreshed (a touch: no I/O) |
| 63.4252 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0023 | the proxy received the turn |
| 0.0810 | P answered (stored) |
| 0.0914 | D1 took the request |
| 0.0944 | the server got its lookup |
| 0.0952 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.0956 | its connector had the lookup's answer |
| 0.3872 | the first token reached the proxy |
| 1.7551 | finished |
| 1.7587 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 7039 records from 8 processes (D1 1580, D2 1598, D3 1731, LMC_D1 301, LMC_D2 303, LMC_D3 314, LMC_P 690, P 522); hook errors 0; unreadable lines 0.

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

- D requests that read from P: 33, 36824 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 200.800 ms, p90 281.900 ms
- P held its blocks for D from its request's end to D's read done: median 234.000 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.616 GB/s (the KV it read over its time)
- the bytes: 132 of 2318 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 33; DIFFERS: D1 block 43 from P's 34; DIFFERS: D1 block 44 from P's 35; DIFFERS: D1 block 45 from P's 36; DIFFERS: D1 block 46 from P's 37; DIFFERS: D1 block 47 from P's 38; DIFFERS: D1 block 48 from P's 39
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 249 chunks (by {'P': 114, 'D': 135}; per server {'LMC_P': 114, 'LMC_D1': 43, 'LMC_D2': 44, 'LMC_D3': 48}); written to memory 249 (7310.7 MB); to disk 0; removed from memory 58
- pulled by D or P: 0 times, from –; never pulled: 249 (99 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 323
- every lookup's outcome per chunk: {'new: stored by this request': 249, 'held (on GPU already)': 37}
- pulls per stored chunk: {0: 249}
- median time from a chunk's last block computed on P to the chunk in memory: 38.090 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 81.740 ms, p90 122.250 ms.

## Checks

- chunk tokens match prompt: 249 of 249
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 249 of 249
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33

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
| 0.0036 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 2c451c47) |
| 23.8077 | P looked it up (turn 3): new: stored by this request |
| 23.8102 | P computed and cached block 15 (GPU block 878, hash 5c2a3032cc6dadfb, request ab764723) |
| 23.8105 | P's connector planned the store |
| 23.8303 | P's worker sent it to the server |
| 23.8317 | the server bound the chunk to its tokens (mp.tokens) |
| 23.8322 | server memory (L1) reserved |
| 23.8727 | eviction order refreshed (a touch: no I/O) |
| 23.8762 | written to memory (L1) |
| 25.4848 | P looked it up (turn 4): held (on GPU already) |
| 25.4849 | eviction order refreshed (a touch: no I/O) |
| 25.5878 | eviction order refreshed (a touch: no I/O) |
| 27.3129 | P looked it up (turn 5): held (on GPU already) |
| 27.3130 | eviction order refreshed (a touch: no I/O) |
| 27.3890 | eviction order refreshed (a touch: no I/O) |
| 29.0458 | P looked it up (turn 6): held (on GPU already) |
| 29.0459 | eviction order refreshed (a touch: no I/O) |
| 29.1335 | eviction order refreshed (a touch: no I/O) |
| 30.8350 | P looked it up (turn 7): held (on GPU already) |
| 30.8352 | eviction order refreshed (a touch: no I/O) |
| 30.9132 | eviction order refreshed (a touch: no I/O) |
| 32.5826 | P looked it up (turn 8): held (on GPU already) |
| 32.5827 | eviction order refreshed (a touch: no I/O) |
| 32.6902 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0789 | P answered (stored) |
| 0.0889 | D1 took the request |
| 0.0918 | the server got its lookup |
| 0.0928 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.0944 | its connector had the lookup's answer |
| 0.3918 | the first token reached the proxy |
| 1.7602 | finished |
| 1.7630 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 6941 records from 5 processes (D1 1556, D2 1690, D3 1713, LMC 1458, P 524); hook errors 0; unreadable lines 0.

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

- D requests that read from P: 33, 72366 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 414.830 ms, p90 662.970 ms
- P held its blocks for D from its request's end to D's read done: median 460.940 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.614 GB/s (the KV it read over its time)
- the bytes: 220 of 4541 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 2; DIFFERS: D1 block 43 from P's 3; DIFFERS: D1 block 44 from P's 4; DIFFERS: D1 block 45 from P's 5; DIFFERS: D1 block 46 from P's 6; DIFFERS: D1 block 47 from P's 7; DIFFERS: D1 block 48 from P's 8
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 113 chunks (by {'P': 113}); written to memory 113 (3317.7 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 113 (39 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 1377
- every lookup's outcome per chunk: {'new: stored by this request': 113, 'hit, not loaded': 278, 'held (on GPU already)': 493}
- pulls per stored chunk: {0: 113}
- median time from a chunk's last block computed on P to the chunk in memory: 107.380 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 91.890 ms, p90 144.490 ms.

## Checks

- chunk tokens match prompt: 113 of 113
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 113 of 113
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33

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
| 0.0049 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 3ad4b739) |
| 9.5898 | P looked it up (turn 0): new: stored by this request |
| 9.5931 | P computed and cached block 15 (GPU block 215, hash 93446f65e6cddea9, request a4e27212) |
| 9.5934 | P's connector planned the store |
| 9.6120 | P's worker sent it to the server |
| 9.6133 | the server bound the chunk to its tokens (mp.tokens) |
| 9.6138 | server memory (L1) reserved |
| 9.6814 | eviction order refreshed (a touch: no I/O) |
| 9.6872 | written to memory (L1) |
| 9.6899 | D3 looked it up (turn 0): hit, not loaded |
| 11.3147 | eviction order refreshed (a touch: no I/O) |
| 11.3407 | P looked it up (turn 1): held (on GPU already) |
| 11.3408 | eviction order refreshed (a touch: no I/O) |
| 11.4690 | eviction order refreshed (a touch: no I/O) |
| 11.4843 | D1 looked it up (turn 1): hit, not loaded |
| 13.4019 | eviction order refreshed (a touch: no I/O) |
| 13.4155 | P looked it up (turn 2): held (on GPU already) |
| 13.4156 | eviction order refreshed (a touch: no I/O) |
| 13.5325 | eviction order refreshed (a touch: no I/O) |
| 13.5463 | D2 looked it up (turn 2): hit, not loaded |
| 15.7158 | eviction order refreshed (a touch: no I/O) |
| 15.7349 | P looked it up (turn 3): held (on GPU already) |
| 15.7351 | eviction order refreshed (a touch: no I/O) |
| 15.8396 | eviction order refreshed (a touch: no I/O) |
| 15.8563 | D3 looked it up (turn 3): held (on GPU already) |
| 15.8564 | eviction order refreshed (a touch: no I/O) |
| 18.0094 | eviction order refreshed (a touch: no I/O) |
| 18.0324 | P looked it up (turn 4): held (on GPU already) |
| 18.0326 | eviction order refreshed (a touch: no I/O) |
| 18.1966 | eviction order refreshed (a touch: no I/O) |
| 18.2188 | D1 looked it up (turn 4): held (on GPU already) |
| 18.2190 | eviction order refreshed (a touch: no I/O) |
| 20.4205 | eviction order refreshed (a touch: no I/O) |
| 20.4483 | P looked it up (turn 5): held (on GPU already) |
| 20.4485 | eviction order refreshed (a touch: no I/O) |
| 20.5930 | eviction order refreshed (a touch: no I/O) |
| 20.6175 | D2 looked it up (turn 5): held (on GPU already) |
| 20.6177 | eviction order refreshed (a touch: no I/O) |
| 22.8705 | eviction order refreshed (a touch: no I/O) |
| 22.8890 | P looked it up (turn 6): held (on GPU already) |
| 22.8892 | eviction order refreshed (a touch: no I/O) |
| 23.0584 | eviction order refreshed (a touch: no I/O) |
| 23.0845 | D3 looked it up (turn 6): held (on GPU already) |
| 23.0847 | eviction order refreshed (a touch: no I/O) |
| 25.4095 | eviction order refreshed (a touch: no I/O) |
| 53.2495 | P looked it up (turn 0): held (on GPU already) |
| 53.2497 | eviction order refreshed (a touch: no I/O) |
| 53.2676 | eviction order refreshed (a touch: no I/O) |
| 53.2746 | D1 looked it up (turn 0): held (on GPU already) |
| 53.2746 | eviction order refreshed (a touch: no I/O) |
| 54.7643 | eviction order refreshed (a touch: no I/O) |
| 54.7755 | P looked it up (turn 1): held (on GPU already) |
| 54.7755 | eviction order refreshed (a touch: no I/O) |
| 54.8848 | eviction order refreshed (a touch: no I/O) |
| 54.8949 | D2 looked it up (turn 1): held (on GPU already) |
| 54.8950 | eviction order refreshed (a touch: no I/O) |
| 56.5729 | eviction order refreshed (a touch: no I/O) |
| 56.5897 | P looked it up (turn 2): held (on GPU already) |
| 56.5898 | eviction order refreshed (a touch: no I/O) |
| 56.6791 | eviction order refreshed (a touch: no I/O) |
| 56.6904 | D3 looked it up (turn 2): held (on GPU already) |
| 56.6905 | eviction order refreshed (a touch: no I/O) |
| 58.5467 | eviction order refreshed (a touch: no I/O) |
| 58.5621 | P looked it up (turn 3): held (on GPU already) |
| 58.5624 | eviction order refreshed (a touch: no I/O) |
| 58.7052 | eviction order refreshed (a touch: no I/O) |
| 58.7206 | D1 looked it up (turn 3): held (on GPU already) |
| 58.7208 | eviction order refreshed (a touch: no I/O) |
| 60.8420 | eviction order refreshed (a touch: no I/O) |
| 60.8640 | P looked it up (turn 4): held (on GPU already) |
| 60.8642 | eviction order refreshed (a touch: no I/O) |
| 60.9951 | eviction order refreshed (a touch: no I/O) |
| 61.0137 | D2 looked it up (turn 4): held (on GPU already) |
| 61.0139 | eviction order refreshed (a touch: no I/O) |
| 63.1589 | eviction order refreshed (a touch: no I/O) |
| 63.1787 | P looked it up (turn 5): held (on GPU already) |
| 63.1790 | eviction order refreshed (a touch: no I/O) |
| 63.3188 | eviction order refreshed (a touch: no I/O) |
| 63.3376 | D3 looked it up (turn 5): held (on GPU already) |
| 63.3377 | eviction order refreshed (a touch: no I/O) |
| 65.5706 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0850 | P answered (stored) |
| 0.1048 | D1 took the request |
| 0.1074 | the server got its lookup |
| 0.1084 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1091 | its connector had the lookup's answer |
| 0.3937 | the first token reached the proxy |
| 1.7616 | finished |
| 1.7644 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, 0208bf26, a2698b29, c1e53df9, cfbcfa3e, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 133bef914eaf.

Trace: 6764 records from 5 processes (D1 1558, D2 1572, D3 1707, LMC 1408, P 519); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 33 | 33 |
| prompt tokens | 40904 | 40904 |
| from the engine's own GPU prefix cache | 10592 | 4080 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 36824 |
| computed | 30312 | 0 |
| stored to LMCache (tokens) | 29184 | 1024 |

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

- D requests that read from P: 33, 36824 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 200.620 ms, p90 281.520 ms
- P held its blocks for D from its request's end to D's read done: median 231.630 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.615 GB/s (the KV it read over its time)
- the bytes: 132 of 2318 blocks D read are P's, bit for bit, every layer (checksums when each read was done, on D's stream, and of P's blocks it held); DIFFERS: D1 block 41 from P's 1; DIFFERS: D1 block 42 from P's 33; DIFFERS: D1 block 43 from P's 34; DIFFERS: D1 block 44 from P's 35; DIFFERS: D1 block 45 from P's 36; DIFFERS: D1 block 46 from P's 37; DIFFERS: D1 block 47 from P's 38; DIFFERS: D1 block 48 from P's 39
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 114 chunks (by {'P': 114}); written to memory 114 (3347.1 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 114 (41 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 323
- every lookup's outcome per chunk: {'new: stored by this request': 114, 'hit, not loaded': 131, 'held (on GPU already)': 37, 'miss: not yet stored': 4}
- pulls per stored chunk: {0: 114}
- median time from a chunk's last block computed on P to the chunk in memory: 93.950 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 82.490 ms, p90 120.680 ms.

## Checks

- chunk tokens match prompt: 114 of 114
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 114 of 114
- d answers same as one engine: 26 of 33
- d answers same as one engine nixl layout: 24 of 33

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
| P | 4 | 1 | 1256 | 80 | 0 | 0 / 0 | 1176 | new: stored by this request 4 | – | – | 1024 |
| D2 | 4 | 1 | 1256 | 80 | 1024 | 0 / 1176 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 4 | 2 | 1041 | 160 | 0 | 0 / 0 | 881 | new: stored by this request 4 | – | – | 1024 |
| D3 | 4 | 2 | 1041 | 80 | 0 | 0 / 961 | 0 | miss: not yet stored 4 | – | – | 1024 |
| P | 4 | 3 | 1667 | 320 | 256 | 0 / 0 | 1347 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D1 | 4 | 3 | 1667 | 80 | 1536 | 0 / 1587 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 4 | 4 | 1676 | 464 | 256 | 0 / 0 | 1212 | held (on GPU already) 1, new: stored by this request 5 | – | – | 1280 |
| D2 | 4 | 4 | 1676 | 160 | 1536 | 0 / 1516 | 0 | hit, not loaded 6 | – | – | 0 |
| P | 4 | 5 | 1841 | 608 | 512 | 0 / 0 | 1233 | held (on GPU already) 2, new: stored by this request 5 | – | – | 1280 |
| D3 | 4 | 5 | 1841 | 320 | 1792 | 0 / 1521 | 0 | held (on GPU already) 1, hit, not loaded 6 | – | – | 0 |

## One chunk's life: aa54d2c3db5e0d3a

Conversation 2, turn 3, tokens 0-256 (system, user 0, assistant 0, answer 0, newline 0, user 1, assistant 1, answer 1, newline 1, user 2, assistant 2, answer 2); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0052 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request d4be425c) |
| 23.7601 | P looked it up (turn 3): new: stored by this request |
| 23.7633 | P computed and cached block 15 (GPU block 878, hash 5c2a3032cc6dadfb, request d44f29d6) |
| 23.7635 | P's connector planned the store |
| 23.7822 | P's worker sent it to the server |
| 23.7834 | the server bound the chunk to its tokens (mp.tokens) |
| 23.7836 | server memory (L1) reserved |
| 23.8259 | eviction order refreshed (a touch: no I/O) |
| 23.8297 | written to memory (L1) |
| 23.8339 | D1 looked it up (turn 3): hit, not loaded |
| 25.4213 | eviction order refreshed (a touch: no I/O) |
| 25.4392 | P looked it up (turn 4): held (on GPU already) |
| 25.4393 | eviction order refreshed (a touch: no I/O) |
| 25.5430 | eviction order refreshed (a touch: no I/O) |
| 25.5521 | D2 looked it up (turn 4): hit, not loaded |
| 27.2525 | eviction order refreshed (a touch: no I/O) |
| 27.2658 | P looked it up (turn 5): held (on GPU already) |
| 27.2660 | eviction order refreshed (a touch: no I/O) |
| 27.3437 | eviction order refreshed (a touch: no I/O) |
| 27.3539 | D3 looked it up (turn 5): hit, not loaded |
| 28.9878 | eviction order refreshed (a touch: no I/O) |
| 29.0089 | P looked it up (turn 6): held (on GPU already) |
| 29.0090 | eviction order refreshed (a touch: no I/O) |
| 29.0961 | eviction order refreshed (a touch: no I/O) |
| 29.1057 | D1 looked it up (turn 6): held (on GPU already) |
| 29.1059 | eviction order refreshed (a touch: no I/O) |
| 30.7798 | eviction order refreshed (a touch: no I/O) |
| 30.7994 | P looked it up (turn 7): held (on GPU already) |
| 30.7995 | eviction order refreshed (a touch: no I/O) |
| 30.8788 | eviction order refreshed (a touch: no I/O) |
| 30.8892 | D2 looked it up (turn 7): held (on GPU already) |
| 30.8893 | eviction order refreshed (a touch: no I/O) |
| 32.5263 | eviction order refreshed (a touch: no I/O) |
| 32.5484 | P looked it up (turn 8): held (on GPU already) |
| 32.5485 | eviction order refreshed (a touch: no I/O) |
| 32.6568 | eviction order refreshed (a touch: no I/O) |
| 32.6691 | D3 looked it up (turn 8): held (on GPU already) |
| 32.6692 | eviction order refreshed (a touch: no I/O) |
| 34.4136 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0023 | the proxy received the turn |
| 0.0790 | P answered (stored) |
| 0.1353 | D1 took the request |
| 0.1384 | the server got its lookup |
| 0.1393 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1412 | its connector had the lookup's answer |
| 0.4317 | the first token reached the proxy |
| 1.8001 | finished |
| 1.8035 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
