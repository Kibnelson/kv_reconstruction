# PD matrix lmctrace on gpu-grad-01, job 30694 (completed), reviewed from the kept results


# LMCache lifecycle: the modes side by side

| | disk | memory |
| --- | ---: | ---: |
| stored chunks | 10 | 10 |
| stored MB | 293.600 | 293.600 |
| chunks to disk | 10 | 0 |
| chunks removed from memory | 10 | 0 |
| pulls | 10 | 10 |
| pulled from | {'L2': 10} | {'L1': 10} |
| never pulled | 0 | 0 |
| D tokens from own GPU | 100624 | 100624 |
| D tokens loaded | 2864 | 2864 |
| D tokens computed | 23825 | 23825 |
| pull ms, memory (median) | – | 71.600 |
| pull ms, disk (median) | 66.340 | – |
| pull ms, disk (p90) | 76.790 | – |
| lookup and prefetch ms, memory (median) | – | 1.600 |
| lookup and prefetch ms, disk (median) | 40.740 | – |
| retrieve ms (median) | 21.530 | 64.410 |
| P store ms (median) | 232.180 | 319.460 |
| memory to disk ms (median) | 331.360 | – |
| D lookups before P's store ended | 0 | 0 |
| checks | 10 of 10; 160 of 160; 10 of 10 | 10 of 10; 160 of 160; 10 of 10 |

Runs: llama32_3b_lmc_disk_rtxa5000_20261125_153456, llama32_3b_lmc_memory_rtxa5000_20261125_153237


# LMCache lifecycle: disk

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **disk** (memory 4.0 GB, eviction noop, store policy skip_l1, chunk 256 tokens); MTRAG's first 4 conversations, interleaved; the proxy waits 500.0 ms between P's answer and D's request. D's role kv_consumer. Commit 29a277bf1ed9.

Trace: 39097 records from 5 processes (D1 24656, D2 12893, D3 927, LMC 230, P 391); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 30 | 30 |
| prompt tokens | 127313 | 127313 |
| from the engine's own GPU prefix cache | 97888 | 100624 |
| loaded from LMCache | 0 | 2864 |
| computed | 29425 | 23825 |
| stored to LMCache (tokens) | 121344 | – |

## Chunks

- stored: 10 chunks, 293.6 MB; to disk 10; removed from memory 10
- pulled by D or P: 10 times, from {'L2': 10}; never pulled: 0
- every lookup's outcome per chunk: {'miss: not yet stored': 10, 'pulled': 10}
- pulls per stored chunk: {1: 10}
- median time from a chunk's last block computed on P to the chunk in memory: 235.880 ms; memory to disk: 331.360 ms
- lookups of chunks never seen stored: 0

## Pulls

- D requests that loaded from LMCache: 5; bytes 293.6 MB
- the server's lookup to the copy on D's GPU done, all from memory: median – ms, p90 – ms; with chunks from disk: median 66.340 ms, p90 76.790 ms
- of which the server's lookup and prefetch (where a disk read happens): median – ms from memory, 40.740 ms with disk
- and its copy to D's GPU (retrieve start to end): median 21.530 ms
- D requests whose lookup finished before P's store of the same turn: 0
- P's stores (planned to stored): median 232.180 ms, p90 243.030 ms

## Checks

- chunk tokens match prompt: 10 of 10
- pulled blocks same hash as p: 160 of 160
- chunks with full chain: 10 of 10

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 1 | 0 | 925 | 32 | 0 | 0 | 893 | miss: not yet stored 3 | – | – | 768 |
| P | 2 | 0 | 949 | 0 | 0 | 0 | 949 | miss: not yet stored 3 | – | – | 768 |
| P | 3 | 0 | 337 | 16 | 0 | 0 | 321 | miss: not yet stored 1 | – | – | 256 |
| P | 0 | 0 | 817 | 16 | 0 | 0 | 801 | miss: not yet stored 3 | – | – | 768 |
| D2 | 2 | 0 | 949 | 0 | 768 | 768 | 181 | pulled 3 | L2 3 | 63.200 | 0 |
| D3 | 3 | 0 | 337 | 0 | 256 | 256 | 81 | pulled 1 | L2 1 | 47.240 | 0 |
| D1 | 1 | 0 | 925 | 0 | 768 | 768 | 157 | pulled 3 | L2 3 | 66.340 | 0 |
| D1 | 0 | 0 | 817 | 0 | 768 | 768 | 49 | pulled 3 | L2 3 | 76.790 | 0 |
| P | 3 | 1 | 846 | 336 | 256 | 0 | 510 | – | – | – | 512 |
| P | 2 | 1 | 2023 | 944 | 768 | 0 | 1079 | – | – | – | 1024 |
| P | 1 | 1 | 2336 | 912 | 768 | 0 | 1424 | – | – | – | 1536 |
| P | 0 | 1 | 1799 | 816 | 768 | 0 | 983 | – | – | – | 1024 |
| D3 | 3 | 1 | 846 | 464 | 768 | 304 | 78 | – | – | – | 0 |
| D2 | 2 | 1 | 2023 | 1072 | 0 | 0 | 951 | – | – | – | 0 |
| D1 | 1 | 1 | 2336 | 1040 | 0 | 0 | 1296 | – | – | – | 0 |
| D1 | 0 | 1 | 1799 | 944 | 0 | 0 | 855 | – | – | – | 0 |
| P | 3 | 2 | 1098 | 832 | 0 | 0 | 266 | – | – | – | 1024 |
| D3 | 3 | 2 | 1098 | 464 | 0 | 0 | 634 | – | – | – | 0 |
| P | 2 | 2 | 3246 | 2016 | 0 | 0 | 1230 | – | – | – | 3072 |
| P | 1 | 2 | 3921 | 2336 | 0 | 0 | 1585 | – | – | – | 3840 |
| P | 0 | 2 | 3494 | 1792 | 0 | 0 | 1702 | – | – | – | 3328 |
| P | 3 | 3 | 1511 | 1088 | 0 | 0 | 423 | – | – | – | 1280 |
| D2 | 2 | 2 | 3246 | 2144 | 0 | 0 | 1102 | – | – | – | 0 |
| D1 | 1 | 2 | 3921 | 2448 | 0 | 0 | 1473 | – | – | – | 0 |
| D1 | 0 | 2 | 3494 | 1920 | 0 | 0 | 1574 | – | – | – | 0 |
| D3 | 3 | 3 | 1511 | 1216 | 0 | 0 | 295 | – | – | – | 0 |
| P | 2 | 3 | 4343 | 3232 | 0 | 0 | 1111 | – | – | – | 4096 |
| P | 3 | 4 | 2208 | 1504 | 0 | 0 | 704 | – | – | – | 2048 |
| P | 1 | 3 | 5032 | 3920 | 0 | 0 | 1112 | – | – | – | 4864 |
| D2 | 2 | 3 | 4343 | 3360 | 0 | 0 | 983 | – | – | – | 0 |
| P | 0 | 3 | 4561 | 3488 | 0 | 0 | 1073 | – | – | – | 4352 |
| D3 | 3 | 4 | 2208 | 1632 | 0 | 0 | 576 | – | – | – | 0 |
| D1 | 1 | 3 | 5032 | 4048 | 0 | 0 | 984 | – | – | – | 0 |
| D1 | 0 | 3 | 4561 | 3616 | 0 | 0 | 945 | – | – | – | 0 |
| P | 2 | 4 | 5393 | 4336 | 0 | 0 | 1057 | – | – | – | 5376 |
| P | 3 | 5 | 2590 | 2208 | 0 | 0 | 382 | – | – | – | 2560 |
| D2 | 2 | 4 | 5393 | 4464 | 0 | 0 | 929 | – | – | – | 0 |
| P | 1 | 4 | 6995 | 5024 | 0 | 0 | 1971 | – | – | – | 6912 |
| D3 | 3 | 5 | 2590 | 2320 | 0 | 0 | 270 | – | – | – | 0 |
| P | 0 | 4 | 5813 | 4560 | 0 | 0 | 1253 | – | – | – | 5632 |
| D1 | 1 | 4 | 6995 | 5152 | 0 | 0 | 1843 | – | – | – | 0 |
| D1 | 0 | 4 | 5813 | 4688 | 0 | 0 | 1125 | – | – | – | 0 |
| P | 2 | 5 | 6497 | 5392 | 0 | 0 | 1105 | – | – | – | 6400 |
| P | 3 | 6 | 2974 | 2576 | 0 | 0 | 398 | – | – | – | 2816 |
| D2 | 2 | 5 | 6497 | 5520 | 0 | 0 | 977 | – | – | – | 0 |
| D3 | 3 | 6 | 2974 | 2704 | 0 | 0 | 270 | – | – | – | 0 |
| P | 1 | 5 | 7585 | 6992 | 0 | 0 | 593 | – | – | – | 7424 |
| P | 0 | 5 | 6925 | 5808 | 0 | 0 | 1117 | – | – | – | 6912 |
| D1 | 1 | 5 | 7585 | 7120 | 0 | 0 | 465 | – | – | – | 0 |
| D1 | 0 | 5 | 6925 | 5936 | 0 | 0 | 989 | – | – | – | 0 |
| P | 2 | 6 | 7503 | 6496 | 0 | 0 | 1007 | – | – | – | 7424 |
| D2 | 2 | 6 | 7503 | 6624 | 0 | 0 | 879 | – | – | – | 0 |
| P | 1 | 6 | 8705 | 7584 | 0 | 0 | 1121 | – | – | – | 8704 |
| P | 0 | 6 | 8025 | 6912 | 0 | 0 | 1113 | – | – | – | 7936 |
| D1 | 1 | 6 | 8705 | 7712 | 0 | 0 | 993 | – | – | – | 0 |
| D1 | 0 | 6 | 8025 | 7040 | 0 | 0 | 985 | – | – | – | 0 |
| P | 1 | 7 | 9765 | 8704 | 0 | 0 | 1061 | – | – | – | 9728 |
| P | 0 | 7 | 9097 | 8016 | 0 | 0 | 1081 | – | – | – | 8960 |
| D1 | 1 | 7 | 9765 | 8832 | 0 | 0 | 933 | – | – | – | 0 |
| D1 | 0 | 7 | 9097 | 8144 | 0 | 0 | 953 | – | – | – | 0 |

## One chunk's life: 85f8dd3d9f0a22f8

Conversation 2, turn 0, tokens 0-256 (system, doc 62964e3a38783558-42267-44322); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0040 | P looked it up (turn 0): miss: not yet stored |
| 0.0094 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request fd622ca4) |
| 0.0094 | P computed and cached block 15 (GPU block 16, hash f948ec801d8ffca5, request fd622ca4) |
| 0.0098 | P's connector planned the store |
| 0.0199 | P's worker sent it to the server |
| 0.0213 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0216 | server memory (L1) reserved |
| 0.1059 | read from disk (L2) |
| 0.1892 | written to memory (L1) |
| 0.5299 | written to disk (L2) |
| 0.5306 | removed from memory (L1) |
| 0.6316 | D2 looked it up (turn 0): pulled, from L2 |
| 0.6686 | read from disk (L2) |
| 0.6948 | copied to D2's GPU |
| 0.6968 | removed from memory (L1) |
| 0.6983 | D2 cached block 0 (GPU block 1; same hash as P's: True) |

## One D request: conversation 2, turn 0 on D2

| s | what |
| ---: | --- |
| -0.0043 | the proxy received the turn |
| 0.1091 | P answered (stored) |
| 0.6282 | D2 took the request |
| 0.6316 | the server got its lookup |
| 0.6698 | the server found 3 of 3 chunks (memory 0, disk 768 tokens) |
| 0.6708 | its connector had the lookup's answer |
| 0.6733 | the server's copy to the GPU began |
| 0.6948 | copied (88080384 bytes) |
| 0.6958 | the worker saw the load finished |
| 0.7220 | the first token reached the proxy |
| 2.1035 | finished |
| 2.1035 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).


# LMCache lifecycle: memory

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 16.0 GB, eviction LRU, store policy default, chunk 256 tokens); MTRAG's first 4 conversations, interleaved; the proxy waits 500.0 ms between P's answer and D's request. D's role kv_consumer. Commit 29a277bf1ed9.

Trace: 36308 records from 5 processes (D1 22494, D2 12392, D3 885, LMC 159, P 378); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 30 | 30 |
| prompt tokens | 127313 | 127313 |
| from the engine's own GPU prefix cache | 97888 | 100624 |
| loaded from LMCache | 0 | 2864 |
| computed | 29425 | 23825 |
| stored to LMCache (tokens) | 121344 | – |

## Chunks

- stored: 10 chunks, 293.6 MB; to disk 0; removed from memory 0
- pulled by D or P: 10 times, from {'L1': 10}; never pulled: 0
- every lookup's outcome per chunk: {'miss: not yet stored': 10, 'pulled': 10}
- pulls per stored chunk: {1: 10}
- median time from a chunk's last block computed on P to the chunk in memory: 316.700 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

## Pulls

- D requests that loaded from LMCache: 5; bytes 293.6 MB
- the server's lookup to the copy on D's GPU done, all from memory: median 71.600 ms, p90 75.620 ms; with chunks from disk: median – ms, p90 – ms
- of which the server's lookup and prefetch (where a disk read happens): median 1.600 ms from memory, – ms with disk
- and its copy to D's GPU (retrieve start to end): median 64.410 ms
- D requests whose lookup finished before P's store of the same turn: 0
- P's stores (planned to stored): median 319.460 ms, p90 329.500 ms

## Checks

- chunk tokens match prompt: 10 of 10
- pulled blocks same hash as p: 160 of 160
- chunks with full chain: 10 of 10

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 1 | 0 | 925 | 0 | 0 | 0 | 925 | miss: not yet stored 3 | – | – | 768 |
| P | 2 | 0 | 949 | 32 | 0 | 0 | 917 | miss: not yet stored 3 | – | – | 768 |
| P | 3 | 0 | 337 | 16 | 0 | 0 | 321 | miss: not yet stored 1 | – | – | 256 |
| P | 0 | 0 | 817 | 16 | 0 | 0 | 801 | miss: not yet stored 3 | – | – | 768 |
| D2 | 2 | 0 | 949 | 0 | 768 | 768 | 181 | pulled 3 | L1 3 | 71.600 | 0 |
| D1 | 1 | 0 | 925 | 0 | 768 | 768 | 157 | pulled 3 | L1 3 | 69.890 | 0 |
| D1 | 0 | 0 | 817 | 0 | 768 | 768 | 49 | pulled 3 | L1 3 | 75.620 | 0 |
| D3 | 3 | 0 | 337 | 0 | 256 | 256 | 81 | pulled 1 | L1 1 | 66.540 | 0 |
| P | 3 | 1 | 846 | 336 | 256 | 0 | 510 | – | – | – | 512 |
| P | 2 | 1 | 2023 | 944 | 768 | 0 | 1079 | – | – | – | 1024 |
| P | 1 | 1 | 2336 | 912 | 768 | 0 | 1424 | – | – | – | 1536 |
| P | 0 | 1 | 1799 | 816 | 768 | 0 | 983 | – | – | – | 1024 |
| D3 | 3 | 1 | 846 | 464 | 768 | 304 | 78 | – | – | – | 0 |
| D2 | 2 | 1 | 2023 | 1072 | 0 | 0 | 951 | – | – | – | 0 |
| D1 | 1 | 1 | 2336 | 1040 | 0 | 0 | 1296 | – | – | – | 0 |
| D1 | 0 | 1 | 1799 | 944 | 0 | 0 | 855 | – | – | – | 0 |
| P | 3 | 2 | 1098 | 832 | 0 | 0 | 266 | – | – | – | 1024 |
| D3 | 3 | 2 | 1098 | 464 | 0 | 0 | 634 | – | – | – | 0 |
| P | 2 | 2 | 3246 | 2016 | 0 | 0 | 1230 | – | – | – | 3072 |
| P | 1 | 2 | 3921 | 2336 | 0 | 0 | 1585 | – | – | – | 3840 |
| P | 0 | 2 | 3494 | 1792 | 0 | 0 | 1702 | – | – | – | 3328 |
| P | 3 | 3 | 1511 | 1088 | 0 | 0 | 423 | – | – | – | 1280 |
| D2 | 2 | 2 | 3246 | 2144 | 0 | 0 | 1102 | – | – | – | 0 |
| D1 | 1 | 2 | 3921 | 2448 | 0 | 0 | 1473 | – | – | – | 0 |
| D1 | 0 | 2 | 3494 | 1920 | 0 | 0 | 1574 | – | – | – | 0 |
| D3 | 3 | 3 | 1511 | 1216 | 0 | 0 | 295 | – | – | – | 0 |
| P | 2 | 3 | 4343 | 3232 | 0 | 0 | 1111 | – | – | – | 4096 |
| P | 3 | 4 | 2208 | 1504 | 0 | 0 | 704 | – | – | – | 2048 |
| P | 1 | 3 | 5032 | 3920 | 0 | 0 | 1112 | – | – | – | 4864 |
| P | 0 | 3 | 4561 | 3488 | 0 | 0 | 1073 | – | – | – | 4352 |
| D2 | 2 | 3 | 4343 | 3360 | 0 | 0 | 983 | – | – | – | 0 |
| D3 | 3 | 4 | 2208 | 1632 | 0 | 0 | 576 | – | – | – | 0 |
| D1 | 1 | 3 | 5032 | 4048 | 0 | 0 | 984 | – | – | – | 0 |
| D1 | 0 | 3 | 4561 | 3616 | 0 | 0 | 945 | – | – | – | 0 |
| P | 2 | 4 | 5393 | 4336 | 0 | 0 | 1057 | – | – | – | 5376 |
| P | 3 | 5 | 2590 | 2208 | 0 | 0 | 382 | – | – | – | 2560 |
| P | 1 | 4 | 6995 | 5024 | 0 | 0 | 1971 | – | – | – | 6912 |
| D2 | 2 | 4 | 5393 | 4464 | 0 | 0 | 929 | – | – | – | 0 |
| P | 0 | 4 | 5813 | 4560 | 0 | 0 | 1253 | – | – | – | 5632 |
| D3 | 3 | 5 | 2590 | 2320 | 0 | 0 | 270 | – | – | – | 0 |
| D1 | 1 | 4 | 6995 | 5152 | 0 | 0 | 1843 | – | – | – | 0 |
| D1 | 0 | 4 | 5813 | 4688 | 0 | 0 | 1125 | – | – | – | 0 |
| P | 2 | 5 | 6497 | 5392 | 0 | 0 | 1105 | – | – | – | 6400 |
| P | 3 | 6 | 2974 | 2576 | 0 | 0 | 398 | – | – | – | 2816 |
| D2 | 2 | 5 | 6497 | 5520 | 0 | 0 | 977 | – | – | – | 0 |
| D3 | 3 | 6 | 2974 | 2704 | 0 | 0 | 270 | – | – | – | 0 |
| P | 1 | 5 | 7585 | 6992 | 0 | 0 | 593 | – | – | – | 7424 |
| P | 0 | 5 | 6925 | 5808 | 0 | 0 | 1117 | – | – | – | 6912 |
| D1 | 1 | 5 | 7585 | 7120 | 0 | 0 | 465 | – | – | – | 0 |
| D1 | 0 | 5 | 6925 | 5936 | 0 | 0 | 989 | – | – | – | 0 |
| P | 2 | 6 | 7503 | 6496 | 0 | 0 | 1007 | – | – | – | 7424 |
| D2 | 2 | 6 | 7503 | 6624 | 0 | 0 | 879 | – | – | – | 0 |
| P | 1 | 6 | 8705 | 7584 | 0 | 0 | 1121 | – | – | – | 8704 |
| P | 0 | 6 | 8025 | 6912 | 0 | 0 | 1113 | – | – | – | 7936 |
| D1 | 1 | 6 | 8705 | 7712 | 0 | 0 | 993 | – | – | – | 0 |
| D1 | 0 | 6 | 8025 | 7040 | 0 | 0 | 985 | – | – | – | 0 |
| P | 1 | 7 | 9765 | 8704 | 0 | 0 | 1061 | – | – | – | 9728 |
| P | 0 | 7 | 9097 | 8016 | 0 | 0 | 1081 | – | – | – | 8960 |
| D1 | 1 | 7 | 9765 | 8832 | 0 | 0 | 933 | – | – | – | 0 |
| D1 | 0 | 7 | 9097 | 8144 | 0 | 0 | 953 | – | – | – | 0 |

## One chunk's life: 1dd495890c85343c

Conversation 1, turn 0, tokens 0-256 (system, doc c8db6e06ff46669e-50302-52227); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0035 | P looked it up (turn 0): miss: not yet stored |
| 0.0068 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 81e63790) |
| 0.0068 | P computed and cached block 15 (GPU block 16, hash 549e5b84eeeceb71, request 81e63790) |
| 0.0075 | P's connector planned the store |
| 0.0185 | P's worker sent it to the server |
| 0.0252 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0256 | server memory (L1) reserved |
| 0.3133 | written to memory (L1) |
| 0.7960 | D1 looked it up (turn 0): pulled, from L1 |
| 0.8659 | copied to D1's GPU |
| 0.8719 | D1 cached block 0 (GPU block 1; same hash as P's: True) |

## One D request: conversation 2, turn 0 on D2

| s | what |
| ---: | --- |
| -0.0040 | the proxy received the turn |
| 0.2362 | P answered (stored) |
| 0.7883 | D2 took the request |
| 0.7917 | the server got its lookup |
| 0.7933 | the server found 3 of 3 chunks (memory 768, disk 0 tokens) |
| 0.7938 | its connector had the lookup's answer |
| 0.7954 | the server's copy to the GPU began |
| 0.8633 | copied (88080384 bytes) |
| 0.8646 | the worker saw the load finished |
| 0.8906 | the first token reached the proxy |
| 2.2712 | finished |
| 2.2713 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).
