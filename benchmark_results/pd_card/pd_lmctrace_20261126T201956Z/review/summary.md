# PD matrix lmctrace on gpu-grad-01, job 30763 (completed), reviewed from the kept results


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
| NIXL read ms (median) | 353.110 | 173.910 | 351.060 | 178.740 | 350.280 | 174.250 | 349.970 | 177.920 |
| NIXL read ms (p90) | 377.930 | 187.870 | 362.190 | 189.710 | 366.320 | 186.510 | 363.870 | 187.560 |
| P held its blocks for D, ms (median) | 396.790 | 267.880 | 424.660 | 278.060 | 382.350 | 272.420 | 382.810 | 288.960 |
| D tokens computed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pull ms, memory (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (median) | – | – | – | – | – | – | – | – |
| pull ms, disk (p90) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, memory (median) | – | – | – | – | – | – | – | – |
| lookup and prefetch ms, disk (median) | – | – | – | – | – | – | – | – |
| retrieve ms (median) | – | – | – | – | – | – | – | – |
| P store ms (median) | 69.970 | 65.230 | 69.980 | 63.580 | 69.340 | 64.950 | 69.190 | 65.080 |
| memory to disk ms (median) | 266.780 | 109.430 | 149.550 | 108.680 | – | – | – | – |
| D lookups before P's store ended | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| checks | 40 of 40; 0 of 0; 40 of 40; 40 of 40; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 9 files, 0.26 GB; {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 7 files, 0.21 GB | 26 of 26; 0 of 0; 26 of 26; 26 of 26; 0 of 0; 0; 0 of 0 disk reads still in memory at the end (prefetch policy retain); {'_default': '0 objects, 0 bytes'}; 12 files, 0.35 GB; {'_default': '0 objects, 0 bytes'}; 5 files, 0.15 GB; {'_default': '0 objects, 0 bytes'}; 6 files, 0.18 GB; {'_default': '0 objects, 0 bytes'}; 3 files, 0.09 GB | 12 of 12; 0 of 0; 12 of 12; 12 of 12; 0 of 0; 0; 9 of 9 disk reads still in memory at the end (prefetch policy retain); {'_default': '9 objects, 264241152 bytes'}; 12 files, 0.35 GB | 12 of 12; 0 of 0; 12 of 12; 12 of 12; 0 of 0; 0; 1 of 1 disk reads still in memory at the end (prefetch policy retain); {'_default': '1 objects, 29360128 bytes'}; 12 files, 0.35 GB | 40 of 40; 0 of 0; 40 of 40 | 26 of 26; 0 of 0; 26 of 26 | 12 of 12; 0 of 0; 12 of 12 | 12 of 12; 0 of 0; 12 of 12 |

Runs: llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_153205, llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_152940, llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_152713, llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_152443, llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_153052, llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_152828, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_152557, llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_152252


# LMCache lifecycle: disk mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1236 records from 8 processes (D1 322, D2 320, D3 161, LMC_D1 77, LMC_D2 77, LMC_D3 37, LMC_P 157, P 85); hook errors 0; unreadable lines 0.

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

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 353.110 ms, p90 377.930 ms
- P held its blocks for D from its request's end to D's read done: median 396.790 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 40 chunks (by {'P': 12, 'D': 28}; per server {'LMC_P': 12, 'LMC_D1': 9, 'LMC_D2': 12, 'LMC_D3': 7}); written to memory 40 (1174.4 MB); to disk 40; removed from memory 40
- pulled by D or P: 0 times, from –; never pulled: 40 (15 of them hold an earlier answer, which D generated itself and holds)
- reads from disk and touches are one event in this trace (before the read hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)
- every lookup's outcome per chunk: {'new: stored by this request': 40, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 40}
- median time from a chunk's last block computed on P to the chunk in memory: 38.360 ms; memory to disk: 266.780 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.970 ms, p90 75.720 ms.

## Checks

- chunk tokens match prompt: 40 of 40
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 40 of 40
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
| 0.0024 | P looked it up (turn 0): new: stored by this request |
| 0.0064 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 2aa5abf5) |
| 0.0064 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request 2aa5abf5) |
| 0.0066 | P's connector planned the store |
| 0.0172 | P's worker sent it to the server |
| 0.0185 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0190 | server memory (L1) reserved |
| 0.0812 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 0.0855 | written to memory (L1) |
| 0.1702 | written to disk (L2) |
| 0.1706 | removed from memory (L1) |
| 1.7604 | P looked it up (turn 1): held (on GPU already) |
| 1.7606 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 1.8206 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 3.6441 | P looked it up (turn 2): held (on GPU already) |
| 3.6443 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 3.7155 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 5.6815 | P looked it up (turn 3): held (on GPU already) |
| 5.6818 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 5.7559 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 7.5891 | P looked it up (turn 4): held (on GPU already) |
| 7.5893 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 7.6666 | disk tier accessed (a read from disk, or a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0023 | the proxy received the turn |
| 0.0836 | P answered (stored) |
| 0.1033 | D1 took the request |
| 0.1075 | the server got its lookup |
| 0.1094 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1111 | its connector had the lookup's answer |
| 0.3817 | the first token reached the proxy |
| 1.7495 | finished |
| 1.7523 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **disk** (memory 2.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1214 records from 8 processes (D1 318, D2 317, D3 161, LMC_D1 74, LMC_D2 73, LMC_D3 37, LMC_P 151, P 83); hook errors 0; unreadable lines 0.

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

## NIXL

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 173.910 ms, p90 187.870 ms
- P held its blocks for D from its request's end to D's read done: median 267.880 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 26 chunks (by {'P': 12, 'D': 14}; per server {'LMC_P': 12, 'LMC_D1': 5, 'LMC_D2': 6, 'LMC_D3': 3}); written to memory 26 (763.4 MB); to disk 26; removed from memory 26
- pulled by D or P: 0 times, from –; never pulled: 26 (14 of them hold an earlier answer, which D generated itself and holds)
- reads from disk and touches are one event in this trace (before the read hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)
- every lookup's outcome per chunk: {'new: stored by this request': 26, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 26}
- median time from a chunk's last block computed on P to the chunk in memory: 31.320 ms; memory to disk: 109.430 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 65.230 ms, p90 75.490 ms.

## Checks

- chunk tokens match prompt: 26 of 26
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 26 of 26
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
| 0.0062 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request f1e029e8) |
| 3.5049 | P looked it up (turn 2): new: stored by this request |
| 3.5096 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 0e113f58) |
| 3.5099 | P's connector planned the store |
| 3.5290 | P's worker sent it to the server |
| 3.5301 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5304 | server memory (L1) reserved |
| 3.5724 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 3.5768 | written to memory (L1) |
| 3.7011 | written to disk (L2) |
| 3.7017 | removed from memory (L1) |
| 5.3314 | P looked it up (turn 3): held (on GPU already) |
| 5.3316 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 5.3973 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 7.0168 | P looked it up (turn 4): held (on GPU already) |
| 7.0170 | disk tier accessed (a read from disk, or a touch: no I/O) |
| 7.0852 | disk tier accessed (a read from disk, or a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0825 | P answered (stored) |
| 0.0934 | D1 took the request |
| 0.0968 | the server got its lookup |
| 0.0993 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1002 | its connector had the lookup's answer |
| 0.3681 | the first token reached the proxy |
| 1.7379 | finished |
| 1.7414 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1405 records from 5 processes (D1 376, D2 418, D3 213, LMC 314, P 84); hook errors 0; unreadable lines 0.

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

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 351.060 ms, p90 362.190 ms
- P held its blocks for D from its request's end to D's read done: median 424.660 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 12; removed from memory 12
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 9; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 74.060 ms; memory to disk: 149.550 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.980 ms, p90 75.310 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
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
| 0.0023 | P looked it up (turn 0): new: stored by this request |
| 0.0062 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request ddc44507) |
| 0.0062 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request ddc44507) |
| 0.0064 | P's connector planned the store |
| 0.0166 | P's worker sent it to the server |
| 0.0179 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0183 | server memory (L1) reserved |
| 0.0800 | eviction order refreshed (a touch: no I/O) |
| 0.0848 | written to memory (L1) |
| 0.1022 | D1 looked it up (turn 0): hit, not loaded |
| 0.1603 | written to disk (L2) |
| 0.1608 | removed from memory (L1) |
| 1.7283 | eviction order refreshed (a touch: no I/O) |
| 1.7386 | P looked it up (turn 1): held (on GPU already) |
| 1.7387 | eviction order refreshed (a touch: no I/O) |
| 1.7966 | eviction order refreshed (a touch: no I/O) |
| 1.8233 | D2 looked it up (turn 1): hit, not loaded |
| 1.8699 | read from disk (L2) |
| 3.5898 | eviction order refreshed (a touch: no I/O) |
| 3.6039 | P looked it up (turn 2): held (on GPU already) |
| 3.6041 | eviction order refreshed (a touch: no I/O) |
| 3.6758 | eviction order refreshed (a touch: no I/O) |
| 3.7249 | D3 looked it up (turn 2): hit, not loaded |
| 5.6491 | eviction order refreshed (a touch: no I/O) |
| 5.6592 | P looked it up (turn 3): held (on GPU already) |
| 5.6594 | eviction order refreshed (a touch: no I/O) |
| 5.7332 | eviction order refreshed (a touch: no I/O) |
| 5.7446 | D1 looked it up (turn 3): held (on GPU already) |
| 5.7447 | eviction order refreshed (a touch: no I/O) |
| 7.6104 | eviction order refreshed (a touch: no I/O) |
| 7.6238 | P looked it up (turn 4): held (on GPU already) |
| 7.6240 | eviction order refreshed (a touch: no I/O) |
| 7.7001 | eviction order refreshed (a touch: no I/O) |
| 7.7126 | D2 looked it up (turn 4): held (on GPU already) |
| 7.7127 | eviction order refreshed (a touch: no I/O) |
| 9.5771 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0823 | P answered (stored) |
| 0.0990 | D1 took the request |
| 0.1022 | the server got its lookup |
| 0.1031 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1050 | its connector had the lookup's answer |
| 0.3590 | the first token reached the proxy |
| 1.7276 | finished |
| 1.7313 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: disk mode, NIXL, shared servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 8.0 GB each, eviction LRU, store policy skip_l1, chunk 256 tokens, prefetch policy retain: what a disk read brings into memory stays until evicted); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1173 records from 5 processes (D1 348, D2 311, D3 157, LMC 274, P 83); hook errors 0; unreadable lines 0.

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

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 178.740 ms, p90 189.710 ms
- P held its blocks for D from its request's end to D's read done: median 278.060 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 12; removed from memory 12
- pulled by D or P: 0 times, from –; never pulled: 12 (6 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 1; touches (eviction order refreshed, no I/O): 30
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 14, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 68.130 ms; memory to disk: 108.680 ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 63.580 ms, p90 75.930 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12
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
| 0.0068 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 302295e1) |
| 3.5098 | P looked it up (turn 2): new: stored by this request |
| 3.5139 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 418aa524) |
| 3.5141 | P's connector planned the store |
| 3.5255 | P's worker sent it to the server |
| 3.5266 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5271 | server memory (L1) reserved |
| 3.5746 | eviction order refreshed (a touch: no I/O) |
| 3.5784 | written to memory (L1) |
| 3.6000 | D3 looked it up (turn 2): hit, not loaded |
| 3.6871 | written to disk (L2) |
| 3.6877 | removed from memory (L1) |
| 5.3166 | eviction order refreshed (a touch: no I/O) |
| 5.3256 | P looked it up (turn 3): held (on GPU already) |
| 5.3257 | eviction order refreshed (a touch: no I/O) |
| 5.3929 | eviction order refreshed (a touch: no I/O) |
| 5.4004 | D1 looked it up (turn 3): hit, not loaded |
| 5.4265 | read from disk (L2) |
| 7.0249 | eviction order refreshed (a touch: no I/O) |
| 7.0356 | P looked it up (turn 4): held (on GPU already) |
| 7.0357 | eviction order refreshed (a touch: no I/O) |
| 7.1024 | eviction order refreshed (a touch: no I/O) |
| 7.1114 | D2 looked it up (turn 4): hit, not loaded |
| 8.7125 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0025 | the proxy received the turn |
| 0.0829 | P answered (stored) |
| 0.1031 | D1 took the request |
| 0.1057 | the server got its lookup |
| 0.1082 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1088 | its connector had the lookup's answer |
| 0.3781 | the first token reached the proxy |
| 1.7454 | finished |
| 1.7488 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, per-engine servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1121 records from 8 processes (D1 314, D2 316, D3 159, LMC_D1 58, LMC_D2 58, LMC_D3 28, LMC_P 108, P 80); hook errors 0; unreadable lines 0.

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

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 350.280 ms, p90 366.320 ms
- P held its blocks for D from its request's end to D's read done: median 382.350 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 40 chunks (by {'P': 12, 'D': 28}; per server {'LMC_P': 12, 'LMC_D1': 9, 'LMC_D2': 12, 'LMC_D3': 7}); written to memory 40 (1174.4 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 40 (15 of them hold an earlier answer, which D generated itself and holds)
- reads from disk and touches are one event in this trace (before the read hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)
- every lookup's outcome per chunk: {'new: stored by this request': 40, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 40}
- median time from a chunk's last block computed on P to the chunk in memory: 38.850 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.340 ms, p90 75.090 ms.

## Checks

- chunk tokens match prompt: 40 of 40
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 40 of 40

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
| 0.0028 | P looked it up (turn 0): new: stored by this request |
| 0.0051 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 60bd820f) |
| 0.0051 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request 60bd820f) |
| 0.0053 | P's connector planned the store |
| 0.0177 | P's worker sent it to the server |
| 0.0192 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0196 | server memory (L1) reserved |
| 0.0809 | written to memory (L1) |
| 1.7503 | P looked it up (turn 1): held (on GPU already) |
| 3.6183 | P looked it up (turn 2): held (on GPU already) |
| 5.6401 | P looked it up (turn 3): held (on GPU already) |
| 7.5409 | P looked it up (turn 4): held (on GPU already) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0800 | P answered (stored) |
| 0.0997 | D1 took the request |
| 0.1025 | the server got its lookup |
| 0.1032 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1038 | its connector had the lookup's answer |
| 0.3694 | the first token reached the proxy |
| 1.7394 | finished |
| 1.7428 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, per-engine servers, chat RAG, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), an LMCache server per engine (LMC_P, LMC_D1-3); mode **memory** (memory 2.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, history_first (chat RAG), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1111 records from 8 processes (D1 314, D2 315, D3 159, LMC_D1 56, LMC_D2 55, LMC_D3 28, LMC_P 104, P 80); hook errors 0; unreadable lines 0.

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

## NIXL

- D requests that read from P: 5, 4062 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 174.250 ms, p90 186.510 ms
- P held its blocks for D from its request's end to D's read done: median 272.420 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 26 chunks (by {'P': 12, 'D': 14}; per server {'LMC_P': 12, 'LMC_D1': 5, 'LMC_D2': 6, 'LMC_D3': 3}); written to memory 26 (763.4 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 26 (14 of them hold an earlier answer, which D generated itself and holds)
- reads from disk and touches are one event in this trace (before the read hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)
- every lookup's outcome per chunk: {'new: stored by this request': 26, 'held (on GPU already)': 2}
- pulls per stored chunk: {0: 26}
- median time from a chunk's last block computed on P to the chunk in memory: 28.410 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 64.950 ms, p90 75.660 ms.

## Checks

- chunk tokens match prompt: 26 of 26
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 26 of 26

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
| 0.0053 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 371cf817) |
| 3.5077 | P looked it up (turn 2): new: stored by this request |
| 3.5109 | P computed and cached block 15 (GPU block 78, hash cb4b47bc1a771a98, request 00f14c4b) |
| 3.5111 | P's connector planned the store |
| 3.5300 | P's worker sent it to the server |
| 3.5313 | the server bound the chunk to its tokens (mp.tokens) |
| 3.5318 | server memory (L1) reserved |
| 3.5779 | written to memory (L1) |
| 5.3237 | P looked it up (turn 3): held (on GPU already) |
| 7.0070 | P looked it up (turn 4): held (on GPU already) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0814 | P answered (stored) |
| 0.1010 | D1 took the request |
| 0.1039 | the server got its lookup |
| 0.1047 | the server found 0 of 2 chunks (memory 0, disk 0 tokens) |
| 0.1053 | its connector had the lookup's answer |
| 0.3723 | the first token reached the proxy |
| 1.7417 | finished |
| 1.7452 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit bbc499a66260.

Trace: 1081 records from 5 processes (D1 312, D2 310, D3 157, LMC 222, P 80); hook errors 0; unreadable lines 0.

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

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 349.970 ms, p90 363.870 ms
- P held its blocks for D from its request's end to D's read done: median 382.810 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk and touches are one event in this trace (before the read hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 73.610 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.190 ms, p90 75.500 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12

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
| 0.0053 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request b9e259af) |
| 0.0053 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request b9e259af) |
| 0.0054 | P's connector planned the store |
| 0.0184 | P's worker sent it to the server |
| 0.0198 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0205 | server memory (L1) reserved |
| 0.0834 | written to memory (L1) |
| 0.0952 | D1 looked it up (turn 0): hit, not loaded |
| 1.7338 | P looked it up (turn 1): held (on GPU already) |
| 1.8185 | D2 looked it up (turn 1): hit, not loaded |
| 3.6024 | P looked it up (turn 2): held (on GPU already) |
| 3.7024 | D3 looked it up (turn 2): hit, not loaded |
| 5.6295 | P looked it up (turn 3): held (on GPU already) |
| 5.7110 | D1 looked it up (turn 3): held (on GPU already) |
| 7.5271 | P looked it up (turn 4): held (on GPU already) |
| 7.6151 | D2 looked it up (turn 4): held (on GPU already) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0806 | P answered (stored) |
| 0.0927 | D1 took the request |
| 0.0952 | the server got its lookup |
| 0.0960 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.0980 | its connector had the lookup's answer |
| 0.3543 | the first token reached the proxy |
| 1.7234 | finished |
| 1.7271 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


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
