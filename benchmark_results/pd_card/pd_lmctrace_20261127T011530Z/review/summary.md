# PD matrix lmctrace on gpu-grad-01, job 30786 (completed), reviewed from the kept results


# LMCache lifecycle: the runs side by side

| | disk, NIXL, per-engine, carry-forward | disk, NIXL, per-engine, chat RAG | disk, NIXL, shared, carry-forward | disk, NIXL, shared, chat RAG | memory, NIXL, per-engine, carry-forward | memory, NIXL, per-engine, chat RAG | memory, NIXL, shared, carry-forward | memory, NIXL, shared, chat RAG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LMCache server lasted the run | True | True | True | True | True | True | True | True |
| LMCache server errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P to D | nixl | nixl | nixl | nixl | nixl | nixl | nixl | nixl |
| LMCache servers | per_engine | per_engine | shared | shared | per_engine | per_engine | shared | shared |
| prompt layout | interleaved | history_first | interleaved | history_first | interleaved | history_first | interleaved | history_first |
| passages | all | all | all | all | all | all | all | all |
| conversations | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| disk reads | retain | retain | retain | retain | retain | retain | retain | retain |
| chunks bound by the engines' stores | 40 | 26 | 12 | 12 | 40 | 26 | 12 | 12 |
| chunks stored by | {'P': 12, 'D': 28} | {'P': 12, 'D': 14} | {'P': 12} | {'P': 12} | {'P': 12, 'D': 28} | {'P': 12, 'D': 14} | {'P': 12} | {'P': 12} |
| chunks written to memory | 40 | 26 | 12 | 12 | 40 | 26 | 12 | 12 |
| stored MB | 1174.400 | 763.400 | 352.300 | 352.300 | 1174.400 | 763.400 | 352.300 | 352.300 |
| chunks to disk | 40 | 26 | 12 | 12 | 0 | 0 | 0 | 0 |
| chunks removed from memory | 40 | 26 | 12 | 12 | 0 | 0 | 0 | 0 |
| pulls | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pulled from | {} | {} | {} | {} | {} | {} | {} | {} |
| never pulled | 40 | 26 | 12 | 12 | 40 | 26 | 12 | 12 |
| D tokens from own GPU | 1888 | 176 | 1888 | 176 | 1888 | 176 | 1888 | 176 |
| D tokens loaded from LMCache | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D tokens read from P over NIXL | 7341 | 4062 | 7341 | 4062 | 7341 | 4062 | 7341 | 4062 |
| NIXL read ms (median) | 351.370 | 175.350 | 351.980 | 176.470 | 352.140 | 173.240 | 355.350 | 176.220 |
| NIXL read ms (p90) | 364.910 | 187.370 | 365.650 | 188.680 | 379.500 | 187.740 | 366.620 | 186.790 |
| NIXL read GB/s (median) | 0.597 | 0.588 | 0.599 | 0.584 | 0.591 | 0.595 | 0.600 | 0.585 |
| D's answers = one engine's | 3 of 5 | 4 of 5 | 3 of 5 | 4 of 5 | 3 of 5 | 4 of 5 | 3 of 5 | 4 of 5 |
| UCX transports | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp | cma, cuda_copy, tcp |
| P held its blocks for D, ms (median) | 378.090 | 260.690 | 427.630 | 256.700 | 394.860 | 271.100 | 382.890 | 294.130 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 70.470 | 64.410 | 69.680 | 64.750 | 69.880 | 65.080 | 69.420 | 65.350 |
| memory to disk ms (median) | 265.960 | 110.010 | 149.590 | 109.740 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 40 of 40; 0 of 0; 40 of 40; 3 of 5; 40 of 40; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 9 files, 0.26 GB; {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 7 files, 0.21 GB | 26 of 26; 0 of 0; 26 of 26; 4 of 5; 26 of 26; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 5 files, 0.15 GB; {'_default': '0 objects, 0 bytes'}; 6 files, 0.18 GB; {'_default': '0 objects, 0 bytes'}; 3 files, 0.09 GB | 12 of 12; 0 of 0; 12 of 12; 3 of 5; 12 of 12; 0 of 0; 0; 9 of 9 disk reads still in memory at the end (prefetch policy retain); {'_default': '9 objects, 264241152 bytes'}; 12 files, 0.35 GB | 12 of 12; 0 of 0; 12 of 12; 4 of 5; 12 of 12; 0 of 0; 0; 1 of 1 disk reads still in memory at the end (prefetch policy retain); {'_default': '1 objects, 29360128 bytes'}; 12 files, 0.35 GB | 40 of 40; 0 of 0; 40 of 40; 3 of 5 | 26 of 26; 0 of 0; 26 of 26; 4 of 5 | 12 of 12; 0 of 0; 12 of 12; 3 of 5 | 12 of 12; 0 of 0; 12 of 12; 4 of 5 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_203227, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_202827, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_202424, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_202019, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_203026, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_202628, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_202221, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_201728


# LMCache lifecycle: disk mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1235 records from 8 processes (D1 320, D2 322, D3 161, LMC_D1 77, LMC_D2 77, LMC_D3 37, LMC_P 157, P 84); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | 7168 |

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 3 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; one engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 351.370 ms, p90 364.910 ms
- P held its blocks for D from its request's end to D's read done: median 378.090 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.597 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 40 chunks (by {'P': 12, 'D': 28}; per server {'LMC_P': 12, 'LMC_D1': 9, 'LMC_D2': 12, 'LMC_D3': 7}); written to memory 40 (1174.4 MB); to disk 40; removed from memory 40
- pulled by D or P: 0 times, from –; never pulled: 40 (15 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 40, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 40}
- median time from a chunk's last block computed on P to the chunk in memory: 38.940 ms; memory to disk: 265.960 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 70.470 ms, p90 73.710 ms.

## Checks

- chunk tokens match prompt: 40 of 40
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 40 of 40
- d answers same as one engine: 3 of 5
- disk mode on disk then out of memory: 40 of 40
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 0 of 0 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end (LMC P): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC P): 12 files, 0.35 GB
- disk mode server memory at end (LMC D1): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D1): 9 files, 0.26 GB
- disk mode server memory at end (LMC D2): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D2): 12 files, 0.35 GB
- disk mode server memory at end (LMC D3): {'_default': '0 objects, 0 bytes'}
- disk mode files at end (LMC D3): 7 files, 0.21 GB

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

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0029 | P looked it up (turn 0): new: stored by this request |
| 0.0052 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request a182baf6) |
| 0.0052 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request a182baf6) |
| 0.0053 | P's connector planned the store |
| 0.0181 | P's worker sent it to the server |
| 0.0195 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0201 | server memory (L1) reserved |
| 0.0774 | eviction order refreshed (a touch: no I/O) |
| 0.0830 | written to memory (L1) |
| 0.1684 | written to disk (L2) |
| 0.1688 | removed from memory (L1) |
| 1.7483 | P looked it up (turn 1): held (on GPU already) |
| 1.7484 | eviction order refreshed (a touch: no I/O) |
| 1.8092 | eviction order refreshed (a touch: no I/O) |
| 3.6104 | P looked it up (turn 2): held (on GPU already) |
| 3.6105 | eviction order refreshed (a touch: no I/O) |
| 3.6834 | eviction order refreshed (a touch: no I/O) |
| 5.6470 | P looked it up (turn 3): held (on GPU already) |
| 5.6471 | eviction order refreshed (a touch: no I/O) |
| 5.7215 | eviction order refreshed (a touch: no I/O) |
| 7.5551 | P looked it up (turn 4): held (on GPU already) |
| 7.5552 | eviction order refreshed (a touch: no I/O) |
| 7.6294 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0804 | P answered (stored) |
| 0.0924 | D1 took the request |
| 0.0956 | the server got its lookup |
| 0.0982 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.0994 | its connector had the lookup's answer |
| 0.3691 | the first token reached the proxy |
| 1.7368 | finished |
| 1.7404 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1220 records from 8 processes (D1 320, D2 319, D3 161, LMC_D1 74, LMC_D2 73, LMC_D3 37, LMC_P 151, P 85); hook errors 0; unreadable lines 0.

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

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 175.350 ms, p90 187.370 ms
- P held its blocks for D from its request's end to D's read done: median 260.690 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.588 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 26 chunks (by {'P': 12, 'D': 14}; per server {'LMC_P': 12, 'LMC_D1': 5, 'LMC_D2': 6, 'LMC_D3': 3}); written to memory 26 (763.4 MB); to disk 26; removed from memory 26
- pulled by D or P: 0 times, from –; never pulled: 26 (14 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 30
- every lookup's outcome per chunk: {'new: stored by this request': 26, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 26}
- median time from a chunk's last block computed on P to the chunk in memory: 29.530 ms; memory to disk: 110.010 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 64.410 ms, p90 75.560 ms.

## Checks

- chunk tokens match prompt: 26 of 26
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 26 of 26
- d answers same as one engine: 4 of 5
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
| 0.0063 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 5a64eeda) |
| 3.5070 | P looked it up (turn 2): new: stored by this request |
| 3.5114 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request d086ece9) |
| 3.5116 | P's connector planned the store |
| 3.5309 | P's worker sent it to the server |
| 3.5322 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5327 | server memory (L1) reserved |
| 3.5737 | eviction order refreshed (a touch: no I/O) |
| 3.5794 | written to memory (L1) |
| 3.7048 | written to disk (L2) |
| 3.7053 | removed from memory (L1) |
| 5.3241 | P looked it up (turn 3): held (on GPU already) |
| 5.3242 | eviction order refreshed (a touch: no I/O) |
| 5.3908 | eviction order refreshed (a touch: no I/O) |
| 7.0095 | P looked it up (turn 4): held (on GPU already) |
| 7.0097 | eviction order refreshed (a touch: no I/O) |
| 7.0765 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0821 | P answered (stored) |
| 0.0955 | D1 took the request |
| 0.0988 | the server got its lookup |
| 0.1008 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1026 | its connector had the lookup's answer |
| 0.3607 | the first token reached the proxy |
| 1.7295 | finished |
| 1.7331 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1425 records from 5 processes (D1 384, D2 422, D3 221, LMC 314, P 84); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | – |

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 3 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; one engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 351.980 ms, p90 365.650 ms
- P held its blocks for D from its request's end to D's read done: median 427.630 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.599 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 12; removed from memory 12
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 9; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 72.260 ms; memory to disk: 149.590 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.680 ms, p90 75.990 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
- d answers same as one engine: 3 of 5
- disk mode on disk then out of memory: 12 of 12
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 9 of 9 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '9 objects, 264241152 bytes'}
- disk mode files at end: 12 files, 0.35 GB

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

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0030 | P looked it up (turn 0): new: stored by this request |
| 0.0071 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 0bc7902e) |
| 0.0071 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request 0bc7902e) |
| 0.0073 | P's connector planned the store |
| 0.0200 | P's worker sent it to the server |
| 0.0214 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0217 | server memory (L1) reserved |
| 0.0815 | eviction order refreshed (a touch: no I/O) |
| 0.0882 | written to memory (L1) |
| 0.0963 | D1 looked it up (turn 0): hit, not loaded |
| 0.1662 | written to disk (L2) |
| 0.1667 | removed from memory (L1) |
| 1.7248 | eviction order refreshed (a touch: no I/O) |
| 1.7347 | P looked it up (turn 1): held (on GPU already) |
| 1.7348 | eviction order refreshed (a touch: no I/O) |
| 1.7921 | eviction order refreshed (a touch: no I/O) |
| 1.8192 | D2 looked it up (turn 1): hit, not loaded |
| 1.8656 | read from disk (L2) |
| 3.5824 | eviction order refreshed (a touch: no I/O) |
| 3.5933 | P looked it up (turn 2): held (on GPU already) |
| 3.5933 | eviction order refreshed (a touch: no I/O) |
| 3.6622 | eviction order refreshed (a touch: no I/O) |
| 3.6889 | D3 looked it up (turn 2): hit, not loaded |
| 5.6211 | eviction order refreshed (a touch: no I/O) |
| 5.6301 | P looked it up (turn 3): held (on GPU already) |
| 5.6301 | eviction order refreshed (a touch: no I/O) |
| 5.7018 | eviction order refreshed (a touch: no I/O) |
| 5.7126 | D1 looked it up (turn 3): held (on GPU already) |
| 5.7126 | eviction order refreshed (a touch: no I/O) |
| 7.5827 | eviction order refreshed (a touch: no I/O) |
| 7.5942 | P looked it up (turn 4): held (on GPU already) |
| 7.5944 | eviction order refreshed (a touch: no I/O) |
| 7.6707 | eviction order refreshed (a touch: no I/O) |
| 7.6820 | D2 looked it up (turn 4): held (on GPU already) |
| 7.6821 | eviction order refreshed (a touch: no I/O) |
| 9.5503 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0834 | P answered (stored) |
| 0.0935 | D1 took the request |
| 0.0963 | the server got its lookup |
| 0.0976 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.0990 | its connector had the lookup's answer |
| 0.3562 | the first token reached the proxy |
| 1.7242 | finished |
| 1.7271 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1172 records from 5 processes (D1 348, D2 311, D3 155, LMC 274, P 84); hook errors 0; unreadable lines 0.

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

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 4 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`

## NIXL

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 176.470 ms, p90 188.680 ms
- P held its blocks for D from its request's end to D's read done: median 256.700 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.584 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 12; removed from memory 12
- pulled by D or P: 0 times, from –; never pulled: 12 (6 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 1; touches (eviction order refreshed, no I/O): 30
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 14, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 68.350 ms; memory to disk: 109.740 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 64.750 ms, p90 75.260 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
- d answers same as one engine: 4 of 5
- disk mode on disk then out of memory: 12 of 12
- disk mode pulls from disk: 0 of 0
- disk mode pulls while still in memory: 0
- disk mode reads kept in memory: 1 of 1 disk reads still in memory at the end (prefetch policy retain)
- disk mode server memory at end: {'_default': '1 objects, 29360128 bytes'}
- disk mode files at end: 12 files, 0.35 GB

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
| 0.0063 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 35b953d1) |
| 3.5090 | P looked it up (turn 2): new: stored by this request |
| 3.5135 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 8116aa8a) |
| 3.5138 | P's connector planned the store |
| 3.5327 | P's worker sent it to the server |
| 3.5341 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5344 | server memory (L1) reserved |
| 3.5758 | eviction order refreshed (a touch: no I/O) |
| 3.5816 | written to memory (L1) |
| 3.6009 | D3 looked it up (turn 2): hit, not loaded |
| 3.6914 | written to disk (L2) |
| 3.6917 | removed from memory (L1) |
| 5.3211 | eviction order refreshed (a touch: no I/O) |
| 5.3300 | P looked it up (turn 3): held (on GPU already) |
| 5.3301 | eviction order refreshed (a touch: no I/O) |
| 5.3960 | eviction order refreshed (a touch: no I/O) |
| 5.4041 | D1 looked it up (turn 3): hit, not loaded |
| 5.4302 | read from disk (L2) |
| 7.0264 | eviction order refreshed (a touch: no I/O) |
| 7.0370 | P looked it up (turn 4): held (on GPU already) |
| 7.0371 | eviction order refreshed (a touch: no I/O) |
| 7.1044 | eviction order refreshed (a touch: no I/O) |
| 7.1130 | D2 looked it up (turn 4): hit, not loaded |
| 8.7112 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0814 | P answered (stored) |
| 0.0928 | D1 took the request |
| 0.0960 | the server got its lookup |
| 0.0975 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.0983 | its connector had the lookup's answer |
| 0.3540 | the first token reached the proxy |
| 1.7217 | finished |
| 1.7246 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1123 records from 8 processes (D1 316, D2 316, D3 159, LMC_D1 58, LMC_D2 58, LMC_D3 28, LMC_P 108, P 80); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | 7168 |

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 3 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; one engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 352.140 ms, p90 379.500 ms
- P held its blocks for D from its request's end to D's read done: median 394.860 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.591 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 40 chunks (by {'P': 12, 'D': 28}; per server {'LMC_P': 12, 'LMC_D1': 9, 'LMC_D2': 12, 'LMC_D3': 7}); written to memory 40 (1174.4 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 40 (15 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 40, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 40}
- median time from a chunk's last block computed on P to the chunk in memory: 38.860 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.880 ms, p90 75.320 ms.

## Checks

- chunk tokens match prompt: 40 of 40
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 40 of 40
- d answers same as one engine: 3 of 5

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

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0029 | P looked it up (turn 0): new: stored by this request |
| 0.0053 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request af658d89) |
| 0.0053 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request af658d89) |
| 0.0054 | P's connector planned the store |
| 0.0182 | P's worker sent it to the server |
| 0.0196 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0201 | server memory (L1) reserved |
| 0.0790 | eviction order refreshed (a touch: no I/O) |
| 0.0849 | written to memory (L1) |
| 1.7535 | P looked it up (turn 1): held (on GPU already) |
| 1.7536 | eviction order refreshed (a touch: no I/O) |
| 1.8123 | eviction order refreshed (a touch: no I/O) |
| 3.6305 | P looked it up (turn 2): held (on GPU already) |
| 3.6306 | eviction order refreshed (a touch: no I/O) |
| 3.6980 | eviction order refreshed (a touch: no I/O) |
| 5.6454 | P looked it up (turn 3): held (on GPU already) |
| 5.6456 | eviction order refreshed (a touch: no I/O) |
| 5.7175 | eviction order refreshed (a touch: no I/O) |
| 7.5431 | P looked it up (turn 4): held (on GPU already) |
| 7.5433 | eviction order refreshed (a touch: no I/O) |
| 7.6181 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0803 | P answered (stored) |
| 0.0997 | D1 took the request |
| 0.1028 | the server got its lookup |
| 0.1036 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1058 | its connector had the lookup's answer |
| 0.3749 | the first token reached the proxy |
| 1.7422 | finished |
| 1.7458 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


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


# LMCache lifecycle: memory mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1079 records from 5 processes (D1 310, D2 310, D3 157, LMC 222, P 80); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | – |

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 3 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`
- turn 0/1, from token 9: D `' I don\'t know what you mean by "photos in the air". I don\'t know what you mean by "photos". I don\'t '`; one engine `' I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I don\'t know wh'`

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 355.350 ms, p90 366.620 ms
- P held its blocks for D from its request's end to D's read done: median 382.890 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.600 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 74.270 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.420 ms, p90 75.320 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
- d answers same as one engine: 3 of 5

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

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0024 | P looked it up (turn 0): new: stored by this request |
| 0.0049 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 0bec70b0) |
| 0.0049 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request 0bec70b0) |
| 0.0050 | P's connector planned the store |
| 0.0153 | P's worker sent it to the server |
| 0.0167 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0172 | server memory (L1) reserved |
| 0.0788 | eviction order refreshed (a touch: no I/O) |
| 0.0847 | written to memory (L1) |
| 0.0931 | D1 looked it up (turn 0): hit, not loaded |
| 1.7319 | eviction order refreshed (a touch: no I/O) |
| 1.7421 | P looked it up (turn 1): held (on GPU already) |
| 1.7422 | eviction order refreshed (a touch: no I/O) |
| 1.8006 | eviction order refreshed (a touch: no I/O) |
| 1.8261 | D2 looked it up (turn 1): hit, not loaded |
| 3.5978 | eviction order refreshed (a touch: no I/O) |
| 3.6110 | P looked it up (turn 2): held (on GPU already) |
| 3.6111 | eviction order refreshed (a touch: no I/O) |
| 3.6813 | eviction order refreshed (a touch: no I/O) |
| 3.7081 | D3 looked it up (turn 2): hit, not loaded |
| 5.6325 | eviction order refreshed (a touch: no I/O) |
| 5.6429 | P looked it up (turn 3): held (on GPU already) |
| 5.6430 | eviction order refreshed (a touch: no I/O) |
| 5.7145 | eviction order refreshed (a touch: no I/O) |
| 5.7266 | D1 looked it up (turn 3): held (on GPU already) |
| 5.7268 | eviction order refreshed (a touch: no I/O) |
| 7.5346 | eviction order refreshed (a touch: no I/O) |
| 7.5475 | P looked it up (turn 4): held (on GPU already) |
| 7.5477 | eviction order refreshed (a touch: no I/O) |
| 7.6225 | eviction order refreshed (a touch: no I/O) |
| 7.6341 | D2 looked it up (turn 4): held (on GPU already) |
| 7.6343 | eviction order refreshed (a touch: no I/O) |
| 9.4575 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0808 | P answered (stored) |
| 0.0903 | D1 took the request |
| 0.0931 | the server got its lookup |
| 0.0939 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.0946 | its connector had the lookup's answer |
| 0.3632 | the first token reached the proxy |
| 1.7313 | finished |
| 1.7349 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit eea6fcf20beb.

Trace: 1070 records from 5 processes (D1 312, D2 311, D3 155, LMC 213, P 79); hook errors 0; unreadable lines 0.

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

**D's answers against one engine's own** (the same prompts to one plain engine, no P/D, no connector, temperature 0; `local_check.json`): 4 of 5 the same, token for token; the first that differs: turn 0/0, so D's KV was not its prompt's.

- turn 0/0, from token 1: D `' I don\'t know. I don\'t know what you mean by "in the air". I don\'t know what you mean by "photos". I'`; one engine `" I am a helpful assistant. Answer the user's last question from the documents. If they do not hold t"`

## NIXL

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 176.220 ms, p90 186.790 ms
- P held its blocks for D from its request's end to D's read done: median 294.130 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.585 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all); UCX_CUDA_IPC_ENABLE_GET_ZCOPY unset (auto: no cuda_ipc read without NVLink)

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 12 (6 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 30
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 14, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 68.950 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 65.350 ms, p90 73.070 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
- d answers same as one engine: 4 of 5

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
| 0.0048 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 6b0f398a) |
| 3.5263 | P looked it up (turn 2): new: stored by this request |
| 3.5291 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 6f8c1671) |
| 3.5294 | P's connector planned the store |
| 3.5490 | P's worker sent it to the server |
| 3.5503 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5508 | server memory (L1) reserved |
| 3.5921 | eviction order refreshed (a touch: no I/O) |
| 3.5986 | written to memory (L1) |
| 3.6107 | D3 looked it up (turn 2): hit, not loaded |
| 5.3248 | eviction order refreshed (a touch: no I/O) |
| 5.3330 | P looked it up (turn 3): held (on GPU already) |
| 5.3331 | eviction order refreshed (a touch: no I/O) |
| 5.3987 | eviction order refreshed (a touch: no I/O) |
| 5.4063 | D1 looked it up (turn 3): hit, not loaded |
| 7.0038 | eviction order refreshed (a touch: no I/O) |
| 7.0163 | P looked it up (turn 4): held (on GPU already) |
| 7.0163 | eviction order refreshed (a touch: no I/O) |
| 7.0807 | eviction order refreshed (a touch: no I/O) |
| 7.0934 | D2 looked it up (turn 4): hit, not loaded |
| 8.6907 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0776 | P answered (stored) |
| 0.1191 | D1 took the request |
| 0.1220 | the server got its lookup |
| 0.1230 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1250 | its connector had the lookup's answer |
| 0.3890 | the first token reached the proxy |
| 1.7568 | finished |
| 1.7602 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
