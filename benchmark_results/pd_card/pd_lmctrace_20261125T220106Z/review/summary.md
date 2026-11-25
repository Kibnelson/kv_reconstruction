# PD matrix lmctrace on gpu-grad-01, job 30722 (completed), reviewed from the kept results


# LMCache lifecycle: the modes side by side

| | disk | memory |
| --- | ---: | ---: |
| LMCache server lasted the run | True | True |
| stored chunks | 114 | 114 |
| stored MB | 0.000 | 0.000 |
| chunks to disk | 0 | 0 |
| chunks removed from memory | 0 | 0 |
| pulls | 0 | 0 |
| pulled from | {} | {} |
| never pulled | 114 | 114 |
| D tokens from own GPU | 101648 | 101648 |
| D tokens loaded | 0 | 0 |
| D tokens computed | 26305 | 26305 |
| pull ms, memory (median) | – | – |
| pull ms, disk (median) | – | – |
| pull ms, disk (p90) | – | – |
| lookup and prefetch ms, memory (median) | – | – |
| lookup and prefetch ms, disk (median) | – | – |
| retrieve ms (median) | – | – |
| P store ms (median) | 120.330 | 123.930 |
| memory to disk ms (median) | – | – |
| D lookups before P's store ended | 0 | 0 |
| checks | 114 of 114; 0 of 0; 0 of 114 | 114 of 114; 0 of 0; 0 of 114 |

Runs: llama32_3b_lmc_disk_rtxa5000_20261125_170432, llama32_3b_lmc_memory_rtxa5000_20261125_170227


# LMCache lifecycle: disk

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 4.0 GB, eviction noop, store policy skip_l1, chunk 256 tokens); MTRAG's first 4 conversations, interleaved; the proxy waits 500.0 ms between P's answer and D's request. D's role kv_consumer. Commit 4f9fc89248f9.

Trace: 6070 records from 5 processes (D1 2350, D2 1030, D3 1028, LMC 1217, P 445); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 30 | 30 |
| prompt tokens | 127953 | 127953 |
| from the engine's own GPU prefix cache | 98400 | 101648 |
| loaded from LMCache | 0 | 0 |
| computed | 29553 | 26305 |
| stored to LMCache (tokens) | 29184 | – |

## Chunks

- stored: 114 chunks, 0.0 MB; to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 114
- every lookup's outcome per chunk: {'new: stored by this request': 114, 'miss: never stored': 102, 'held (on GPU already)': 758}
- pulls per stored chunk: {0: 114}
- median time from a chunk's last block computed on P to the chunk in memory: – ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

## Pulls

- D requests that loaded from LMCache: 0; bytes 0.0 MB
- the server's lookup to the copy on D's GPU done, all from memory: median – ms, p90 – ms; with chunks from disk: median – ms, p90 – ms
- of which the server's lookup and prefetch (where a disk read happens): median – ms from memory, – ms with disk
- and its copy to D's GPU (retrieve start to end): median – ms
- D requests whose lookup finished before P's store of the same turn: 0
- P's stores (planned to stored): median 120.330 ms, p90 197.160 ms

## Checks

- chunk tokens match prompt: 114 of 114
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 0 of 114

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 1 | 0 | 925 | 0 | 0 | 0 | 925 | new: stored by this request 3 | – | – | 768 |
| P | 2 | 0 | 949 | 32 | 0 | 0 | 917 | new: stored by this request 3 | – | – | 768 |
| P | 3 | 0 | 337 | 16 | 0 | 0 | 321 | new: stored by this request 1 | – | – | 256 |
| P | 0 | 0 | 817 | 16 | 0 | 0 | 801 | new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 0 | 337 | 0 | 0 | 0 | 337 | miss: never stored 1 | – | – | 0 |
| D2 | 2 | 0 | 949 | 0 | 0 | 0 | 949 | miss: never stored 3 | – | – | 0 |
| D1 | 1 | 0 | 925 | 0 | 0 | 0 | 925 | miss: never stored 3 | – | – | 0 |
| D1 | 0 | 0 | 817 | 16 | 0 | 0 | 801 | miss: never stored 3 | – | – | 0 |
| P | 3 | 1 | 846 | 336 | 256 | 0 | 510 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| P | 2 | 1 | 2023 | 944 | 768 | 0 | 1079 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| P | 1 | 1 | 2336 | 912 | 768 | 0 | 1424 | held (on GPU already) 3, new: stored by this request 6 | – | – | 1536 |
| P | 0 | 1 | 1799 | 816 | 768 | 0 | 983 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D3 | 3 | 1 | 846 | 464 | 256 | 0 | 382 | held (on GPU already) 1, miss: never stored 2 | – | – | 0 |
| D2 | 2 | 1 | 2023 | 1072 | 1024 | 0 | 951 | held (on GPU already) 4, miss: never stored 3 | – | – | 0 |
| D1 | 1 | 1 | 2336 | 1040 | 1024 | 0 | 1296 | held (on GPU already) 4, miss: never stored 5 | – | – | 0 |
| D1 | 0 | 1 | 1799 | 944 | 768 | 0 | 855 | held (on GPU already) 3, miss: never stored 4 | – | – | 0 |
| P | 3 | 2 | 1226 | 832 | 768 | 0 | 394 | held (on GPU already) 3, new: stored by this request 1 | – | – | 256 |
| P | 2 | 2 | 3246 | 2016 | 1792 | 0 | 1230 | held (on GPU already) 7, new: stored by this request 5 | – | – | 1280 |
| P | 1 | 2 | 3921 | 2336 | 2304 | 0 | 1585 | held (on GPU already) 9, new: stored by this request 6 | – | – | 1536 |
| P | 0 | 2 | 3494 | 1792 | 1792 | 0 | 1702 | held (on GPU already) 7, new: stored by this request 6 | – | – | 1536 |
| D3 | 3 | 2 | 1226 | 960 | 768 | 0 | 266 | held (on GPU already) 3, miss: never stored 1 | – | – | 0 |
| D2 | 2 | 2 | 3246 | 2144 | 2048 | 0 | 1102 | held (on GPU already) 8, miss: never stored 4 | – | – | 0 |
| D1 | 1 | 2 | 3921 | 2448 | 2304 | 0 | 1473 | held (on GPU already) 9, miss: never stored 6 | – | – | 0 |
| D1 | 0 | 2 | 3494 | 1920 | 1792 | 0 | 1574 | held (on GPU already) 7, miss: never stored 6 | – | – | 0 |
| P | 3 | 3 | 1639 | 1216 | 1024 | 0 | 423 | held (on GPU already) 4, new: stored by this request 2 | – | – | 512 |
| P | 2 | 3 | 4343 | 3232 | 3072 | 0 | 1111 | held (on GPU already) 12, new: stored by this request 4 | – | – | 1024 |
| D3 | 3 | 3 | 1639 | 1344 | 1280 | 0 | 295 | held (on GPU already) 5, miss: never stored 1 | – | – | 0 |
| D2 | 2 | 3 | 4343 | 3360 | 3328 | 0 | 983 | held (on GPU already) 13, miss: never stored 3 | – | – | 0 |
| P | 1 | 3 | 5032 | 3920 | 3840 | 0 | 1112 | held (on GPU already) 15, new: stored by this request 4 | – | – | 1024 |
| P | 0 | 3 | 4561 | 3488 | 3328 | 0 | 1073 | held (on GPU already) 13, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 3 | 5032 | 4048 | 3840 | 0 | 984 | held (on GPU already) 15, miss: never stored 4 | – | – | 0 |
| D1 | 0 | 3 | 4561 | 3616 | 3584 | 0 | 945 | held (on GPU already) 14, miss: never stored 3 | – | – | 0 |
| P | 3 | 4 | 2336 | 1632 | 1536 | 0 | 704 | held (on GPU already) 6, new: stored by this request 3 | – | – | 768 |
| P | 2 | 4 | 5393 | 4336 | 4096 | 0 | 1057 | held (on GPU already) 16, new: stored by this request 5 | – | – | 1280 |
| D3 | 3 | 4 | 2336 | 1760 | 1536 | 0 | 576 | held (on GPU already) 6, miss: never stored 3 | – | – | 0 |
| D2 | 2 | 4 | 5393 | 4464 | 4352 | 0 | 929 | held (on GPU already) 17, miss: never stored 4 | – | – | 0 |
| P | 1 | 4 | 6995 | 5024 | 4864 | 0 | 1971 | held (on GPU already) 19, new: stored by this request 8 | – | – | 2048 |
| P | 0 | 4 | 5813 | 4560 | 4352 | 0 | 1253 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| P | 3 | 5 | 2718 | 2336 | 2304 | 0 | 382 | held (on GPU already) 9, new: stored by this request 1 | – | – | 256 |
| D1 | 1 | 4 | 6995 | 5152 | 5120 | 0 | 1843 | held (on GPU already) 20, miss: never stored 7 | – | – | 0 |
| D1 | 0 | 4 | 5813 | 4688 | 4608 | 0 | 1125 | held (on GPU already) 18, miss: never stored 4 | – | – | 0 |
| D3 | 3 | 5 | 2718 | 2448 | 2304 | 0 | 270 | held (on GPU already) 9, miss: never stored 1 | – | – | 0 |
| P | 2 | 5 | 6497 | 5392 | 5376 | 0 | 1105 | held (on GPU already) 21, new: stored by this request 4 | – | – | 1024 |
| D2 | 2 | 5 | 6497 | 5520 | 5376 | 0 | 977 | held (on GPU already) 21, miss: never stored 4 | – | – | 0 |
| P | 1 | 5 | 7585 | 6992 | 6912 | 0 | 593 | held (on GPU already) 27, new: stored by this request 2 | – | – | 512 |
| P | 0 | 5 | 6925 | 5808 | 5632 | 0 | 1117 | held (on GPU already) 22, new: stored by this request 5 | – | – | 1280 |
| P | 3 | 6 | 3102 | 2704 | 2560 | 0 | 398 | held (on GPU already) 10, new: stored by this request 2 | – | – | 512 |
| D1 | 1 | 5 | 7585 | 7120 | 6912 | 0 | 465 | held (on GPU already) 27, miss: never stored 2 | – | – | 0 |
| D1 | 0 | 5 | 6925 | 5936 | 5888 | 0 | 989 | held (on GPU already) 23, miss: never stored 4 | – | – | 0 |
| D3 | 3 | 6 | 3102 | 2832 | 2816 | 0 | 270 | held (on GPU already) 11, miss: never stored 1 | – | – | 0 |
| P | 2 | 6 | 7503 | 6496 | 6400 | 0 | 1007 | held (on GPU already) 25, new: stored by this request 4 | – | – | 1024 |
| D2 | 2 | 6 | 7503 | 6624 | 6400 | 0 | 879 | held (on GPU already) 25, miss: never stored 4 | – | – | 0 |
| P | 1 | 6 | 8705 | 7584 | 7424 | 0 | 1121 | held (on GPU already) 29, new: stored by this request 5 | – | – | 1280 |
| P | 0 | 6 | 8025 | 6912 | 6912 | 0 | 1113 | held (on GPU already) 27, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 6 | 8705 | 7712 | 7680 | 0 | 993 | held (on GPU already) 30, miss: never stored 4 | – | – | 0 |
| D1 | 0 | 6 | 8025 | 7040 | 6912 | 0 | 985 | held (on GPU already) 27, miss: never stored 4 | – | – | 0 |
| P | 1 | 7 | 9765 | 8704 | 8704 | 0 | 1061 | held (on GPU already) 34, new: stored by this request 4 | – | – | 1024 |
| P | 0 | 7 | 9097 | 8016 | 7936 | 0 | 1081 | held (on GPU already) 31, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 7 | 9765 | 8832 | 8704 | 0 | 933 | held (on GPU already) 34, miss: never stored 4 | – | – | 0 |
| D1 | 0 | 7 | 9097 | 8144 | 7936 | 0 | 953 | held (on GPU already) 31, miss: never stored 4 | – | – | 0 |

## One chunk's life: 1dd495890c85343c

Conversation 1, turn 0, tokens 0-256 (system, doc c8db6e06ff46669e-50302-52227); – bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0036 | P looked it up (turn 0): new: stored by this request |
| 0.0095 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 50c88ee1) |
| 0.0095 | P computed and cached block 15 (GPU block 16, hash 549e5b84eeeceb71, request 50c88ee1) |
| 0.0100 | P's connector planned the store |
| 0.0208 | P's worker sent it to the server |
| 0.0223 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0227 | server memory (L1) reserved |
| 0.1776 | read from disk (L2) |
| 0.7094 | D1 looked it up (turn 0): miss: never stored |
| 2.2354 | read from disk (L2) |
| 2.2999 | P looked it up (turn 1): held (on GPU already) |
| 2.3000 | read from disk (L2) |
| 2.5093 | read from disk (L2) |
| 3.0233 | D1 looked it up (turn 1): held (on GPU already) |
| 3.0235 | read from disk (L2) |
| 4.6555 | read from disk (L2) |
| 4.6804 | P looked it up (turn 2): held (on GPU already) |
| 4.6807 | read from disk (L2) |
| 4.8376 | read from disk (L2) |
| 5.3517 | D1 looked it up (turn 2): held (on GPU already) |
| 5.3520 | read from disk (L2) |
| 7.1379 | read from disk (L2) |
| 7.1594 | P looked it up (turn 3): held (on GPU already) |
| 7.1596 | read from disk (L2) |
| 7.2772 | read from disk (L2) |
| 7.7936 | D1 looked it up (turn 3): held (on GPU already) |
| 7.7938 | read from disk (L2) |
| 9.5359 | read from disk (L2) |
| 9.5654 | P looked it up (turn 4): held (on GPU already) |
| 9.5657 | read from disk (L2) |
| 9.7813 | read from disk (L2) |
| 10.3033 | D1 looked it up (turn 4): held (on GPU already) |
| 10.3036 | read from disk (L2) |
| 12.2371 | read from disk (L2) |
| 12.2544 | P looked it up (turn 5): held (on GPU already) |
| 12.2547 | read from disk (L2) |
| 12.3380 | read from disk (L2) |
| 12.8616 | D1 looked it up (turn 5): held (on GPU already) |
| 12.8623 | read from disk (L2) |
| 14.6807 | read from disk (L2) |
| 14.7068 | P looked it up (turn 6): held (on GPU already) |
| 14.7072 | read from disk (L2) |
| 14.8508 | read from disk (L2) |
| 15.3764 | D1 looked it up (turn 6): held (on GPU already) |
| 15.3768 | read from disk (L2) |
| 17.3093 | read from disk (L2) |
| 17.3356 | P looked it up (turn 7): held (on GPU already) |
| 17.3359 | read from disk (L2) |
| 17.4832 | read from disk (L2) |
| 18.0106 | D1 looked it up (turn 7): held (on GPU already) |
| 18.0110 | read from disk (L2) |
| 19.9948 | read from disk (L2) |

## One D request: conversation 3, turn 0 on D3

| s | what |
| ---: | --- |
| -0.0012 | the proxy received the turn |
| 0.1834 | P answered (stored) |
| 0.7060 | D3 took the request |
| 0.7107 | the server got its lookup |
| 0.7136 | the server found 0 of 1 chunks (memory 0, disk 0 tokens) |
| 0.7146 | its connector had the lookup's answer |
| 0.7533 | the first token reached the proxy |
| 2.1245 | finished |
| 2.1245 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).


# LMCache lifecycle: memory

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 16.0 GB, eviction LRU, store policy default, chunk 256 tokens); MTRAG's first 4 conversations, interleaved; the proxy waits 500.0 ms between P's answer and D's request. D's role kv_consumer. Commit 4f9fc89248f9.

Trace: 5828 records from 5 processes (D1 2329, D2 1024, D3 1021, LMC 1045, P 409); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 30 | 30 |
| prompt tokens | 127953 | 127953 |
| from the engine's own GPU prefix cache | 98400 | 101648 |
| loaded from LMCache | 0 | 0 |
| computed | 29553 | 26305 |
| stored to LMCache (tokens) | 29184 | – |

## Chunks

- stored: 114 chunks, 0.0 MB; to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 114
- every lookup's outcome per chunk: {'new: stored by this request': 114, 'miss: never stored': 102, 'held (on GPU already)': 758}
- pulls per stored chunk: {0: 114}
- median time from a chunk's last block computed on P to the chunk in memory: – ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

## Pulls

- D requests that loaded from LMCache: 0; bytes 0.0 MB
- the server's lookup to the copy on D's GPU done, all from memory: median – ms, p90 – ms; with chunks from disk: median – ms, p90 – ms
- of which the server's lookup and prefetch (where a disk read happens): median – ms from memory, – ms with disk
- and its copy to D's GPU (retrieve start to end): median – ms
- D requests whose lookup finished before P's store of the same turn: 0
- P's stores (planned to stored): median 123.930 ms, p90 252.240 ms

## Checks

- chunk tokens match prompt: 114 of 114
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 0 of 114

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 1 | 0 | 925 | 0 | 0 | 0 | 925 | new: stored by this request 3 | – | – | 768 |
| P | 2 | 0 | 949 | 32 | 0 | 0 | 917 | new: stored by this request 3 | – | – | 768 |
| P | 3 | 0 | 337 | 16 | 0 | 0 | 321 | new: stored by this request 1 | – | – | 256 |
| P | 0 | 0 | 817 | 16 | 0 | 0 | 801 | new: stored by this request 3 | – | – | 768 |
| D1 | 1 | 0 | 925 | 0 | 0 | 0 | 925 | miss: never stored 3 | – | – | 0 |
| D1 | 0 | 0 | 817 | 16 | 0 | 0 | 801 | miss: never stored 3 | – | – | 0 |
| D3 | 3 | 0 | 337 | 0 | 0 | 0 | 337 | miss: never stored 1 | – | – | 0 |
| D2 | 2 | 0 | 949 | 0 | 0 | 0 | 949 | miss: never stored 3 | – | – | 0 |
| P | 3 | 1 | 846 | 336 | 256 | 0 | 510 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| P | 1 | 1 | 2336 | 912 | 768 | 0 | 1424 | held (on GPU already) 3, new: stored by this request 6 | – | – | 1536 |
| P | 0 | 1 | 1799 | 816 | 768 | 0 | 983 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| P | 2 | 1 | 2023 | 944 | 768 | 0 | 1079 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D3 | 3 | 1 | 846 | 464 | 256 | 0 | 382 | held (on GPU already) 1, miss: never stored 2 | – | – | 0 |
| D1 | 1 | 1 | 2336 | 1040 | 1024 | 0 | 1296 | held (on GPU already) 4, miss: never stored 5 | – | – | 0 |
| D1 | 0 | 1 | 1799 | 944 | 768 | 0 | 855 | held (on GPU already) 3, miss: never stored 4 | – | – | 0 |
| D2 | 2 | 1 | 2023 | 1072 | 1024 | 0 | 951 | held (on GPU already) 4, miss: never stored 3 | – | – | 0 |
| P | 3 | 2 | 1226 | 832 | 768 | 0 | 394 | held (on GPU already) 3, new: stored by this request 1 | – | – | 256 |
| P | 1 | 2 | 3921 | 2336 | 2304 | 0 | 1585 | held (on GPU already) 9, new: stored by this request 6 | – | – | 1536 |
| P | 2 | 2 | 3246 | 2016 | 1792 | 0 | 1230 | held (on GPU already) 7, new: stored by this request 5 | – | – | 1280 |
| P | 0 | 2 | 3494 | 1792 | 1792 | 0 | 1702 | held (on GPU already) 7, new: stored by this request 6 | – | – | 1536 |
| D3 | 3 | 2 | 1226 | 960 | 768 | 0 | 266 | held (on GPU already) 3, miss: never stored 1 | – | – | 0 |
| D1 | 1 | 2 | 3921 | 2448 | 2304 | 0 | 1473 | held (on GPU already) 9, miss: never stored 6 | – | – | 0 |
| D1 | 0 | 2 | 3494 | 1920 | 1792 | 0 | 1574 | held (on GPU already) 7, miss: never stored 6 | – | – | 0 |
| D2 | 2 | 2 | 3246 | 2144 | 2048 | 0 | 1102 | held (on GPU already) 8, miss: never stored 4 | – | – | 0 |
| P | 3 | 3 | 1639 | 1216 | 1024 | 0 | 423 | held (on GPU already) 4, new: stored by this request 2 | – | – | 512 |
| D3 | 3 | 3 | 1639 | 1344 | 1280 | 0 | 295 | held (on GPU already) 5, miss: never stored 1 | – | – | 0 |
| P | 1 | 3 | 5032 | 3920 | 3840 | 0 | 1112 | held (on GPU already) 15, new: stored by this request 4 | – | – | 1024 |
| P | 2 | 3 | 4343 | 3232 | 3072 | 0 | 1111 | held (on GPU already) 12, new: stored by this request 4 | – | – | 1024 |
| P | 0 | 3 | 4561 | 3488 | 3328 | 0 | 1073 | held (on GPU already) 13, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 3 | 5032 | 4048 | 3840 | 0 | 984 | held (on GPU already) 15, miss: never stored 4 | – | – | 0 |
| D2 | 2 | 3 | 4343 | 3360 | 3328 | 0 | 983 | held (on GPU already) 13, miss: never stored 3 | – | – | 0 |
| D1 | 0 | 3 | 4561 | 3616 | 3584 | 0 | 945 | held (on GPU already) 14, miss: never stored 3 | – | – | 0 |
| P | 3 | 4 | 2336 | 1632 | 1536 | 0 | 704 | held (on GPU already) 6, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 4 | 2336 | 1760 | 1536 | 0 | 576 | held (on GPU already) 6, miss: never stored 3 | – | – | 0 |
| P | 2 | 4 | 5393 | 4336 | 4096 | 0 | 1057 | held (on GPU already) 16, new: stored by this request 5 | – | – | 1280 |
| P | 1 | 4 | 6995 | 5024 | 4864 | 0 | 1971 | held (on GPU already) 19, new: stored by this request 8 | – | – | 2048 |
| P | 0 | 4 | 5813 | 4560 | 4352 | 0 | 1253 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| D2 | 2 | 4 | 5393 | 4464 | 4352 | 0 | 929 | held (on GPU already) 17, miss: never stored 4 | – | – | 0 |
| D1 | 1 | 4 | 6995 | 5152 | 5120 | 0 | 1843 | held (on GPU already) 20, miss: never stored 7 | – | – | 0 |
| P | 3 | 5 | 2718 | 2336 | 2304 | 0 | 382 | held (on GPU already) 9, new: stored by this request 1 | – | – | 256 |
| D1 | 0 | 4 | 5813 | 4688 | 4608 | 0 | 1125 | held (on GPU already) 18, miss: never stored 4 | – | – | 0 |
| D3 | 3 | 5 | 2718 | 2448 | 2304 | 0 | 270 | held (on GPU already) 9, miss: never stored 1 | – | – | 0 |
| P | 2 | 5 | 6497 | 5392 | 5376 | 0 | 1105 | held (on GPU already) 21, new: stored by this request 4 | – | – | 1024 |
| P | 1 | 5 | 7585 | 6992 | 6912 | 0 | 593 | held (on GPU already) 27, new: stored by this request 2 | – | – | 512 |
| P | 0 | 5 | 6925 | 5808 | 5632 | 0 | 1117 | held (on GPU already) 22, new: stored by this request 5 | – | – | 1280 |
| D2 | 2 | 5 | 6497 | 5520 | 5376 | 0 | 977 | held (on GPU already) 21, miss: never stored 4 | – | – | 0 |
| P | 3 | 6 | 3102 | 2704 | 2560 | 0 | 398 | held (on GPU already) 10, new: stored by this request 2 | – | – | 512 |
| D1 | 1 | 5 | 7585 | 7120 | 6912 | 0 | 465 | held (on GPU already) 27, miss: never stored 2 | – | – | 0 |
| D1 | 0 | 5 | 6925 | 5936 | 5888 | 0 | 989 | held (on GPU already) 23, miss: never stored 4 | – | – | 0 |
| D3 | 3 | 6 | 3102 | 2832 | 2816 | 0 | 270 | held (on GPU already) 11, miss: never stored 1 | – | – | 0 |
| P | 2 | 6 | 7503 | 6496 | 6400 | 0 | 1007 | held (on GPU already) 25, new: stored by this request 4 | – | – | 1024 |
| D2 | 2 | 6 | 7503 | 6624 | 6400 | 0 | 879 | held (on GPU already) 25, miss: never stored 4 | – | – | 0 |
| P | 1 | 6 | 8705 | 7584 | 7424 | 0 | 1121 | held (on GPU already) 29, new: stored by this request 5 | – | – | 1280 |
| P | 0 | 6 | 8025 | 6912 | 6912 | 0 | 1113 | held (on GPU already) 27, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 6 | 8705 | 7712 | 7680 | 0 | 993 | held (on GPU already) 30, miss: never stored 4 | – | – | 0 |
| D1 | 0 | 6 | 8025 | 7040 | 6912 | 0 | 985 | held (on GPU already) 27, miss: never stored 4 | – | – | 0 |
| P | 1 | 7 | 9765 | 8704 | 8704 | 0 | 1061 | held (on GPU already) 34, new: stored by this request 4 | – | – | 1024 |
| P | 0 | 7 | 9097 | 8016 | 7936 | 0 | 1081 | held (on GPU already) 31, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 7 | 9765 | 8832 | 8704 | 0 | 933 | held (on GPU already) 34, miss: never stored 4 | – | – | 0 |
| D1 | 0 | 7 | 9097 | 8144 | 7936 | 0 | 953 | held (on GPU already) 31, miss: never stored 4 | – | – | 0 |

## One chunk's life: 1dd495890c85343c

Conversation 1, turn 0, tokens 0-256 (system, doc c8db6e06ff46669e-50302-52227); – bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0037 | P looked it up (turn 0): new: stored by this request |
| 0.0058 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 700ae106) |
| 0.0058 | P computed and cached block 15 (GPU block 16, hash 549e5b84eeeceb71, request 700ae106) |
| 0.0060 | P's connector planned the store |
| 0.0164 | P's worker sent it to the server |
| 0.0207 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0211 | server memory (L1) reserved |
| 0.6577 | D1 looked it up (turn 0): miss: never stored |
| 2.2191 | P looked it up (turn 1): held (on GPU already) |
| 2.8922 | D1 looked it up (turn 1): held (on GPU already) |
| 4.5438 | P looked it up (turn 2): held (on GPU already) |
| 5.2152 | D1 looked it up (turn 2): held (on GPU already) |
| 7.0159 | P looked it up (turn 3): held (on GPU already) |
| 7.6538 | D1 looked it up (turn 3): held (on GPU already) |
| 9.4763 | P looked it up (turn 4): held (on GPU already) |
| 10.2137 | D1 looked it up (turn 4): held (on GPU already) |
| 12.1596 | P looked it up (turn 5): held (on GPU already) |
| 12.7685 | D1 looked it up (turn 5): held (on GPU already) |
| 14.6113 | P looked it up (turn 6): held (on GPU already) |
| 15.2779 | D1 looked it up (turn 6): held (on GPU already) |
| 17.2350 | P looked it up (turn 7): held (on GPU already) |
| 17.9049 | D1 looked it up (turn 7): held (on GPU already) |

## One D request: conversation 1, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0045 | the proxy received the turn |
| 0.0872 | P answered (stored) |
| 0.6540 | D1 took the request |
| 0.6577 | the server got its lookup |
| 0.6589 | the server found 0 of 3 chunks (memory 0, disk 0 tokens) |
| 0.6606 | its connector had the lookup's answer |
| 0.7421 | the first token reached the proxy |
| 2.1774 | finished |
| 2.1775 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).
