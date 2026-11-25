# LMCache lifecycle: memory

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 16.0 GB, eviction LRU, store policy default, chunk 256 tokens); MTRAG's first 4 conversations, interleaved; the proxy waits 500.0 ms between P's answer and D's request. D's role kv_consumer. Commit f159ef6cab84.

Trace: 6376 records from 5 processes (D1 2428, D2 1069, D3 1068, LMC 1389, P 422); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 30 | 30 |
| prompt tokens | 127953 | 127953 |
| from the engine's own GPU prefix cache | 98400 | 101632 |
| loaded from LMCache | 0 | 23040 |
| computed | 29553 | 3281 |
| stored to LMCache (tokens) | 29184 | – |

## Chunks

- bound to their tokens by P's stores: 114 chunks; written to the server's memory 114 (3347.1 MB); to disk 0; removed from memory 0
- pulled by D or P: 102 times, from {'L1': 102}; never pulled: 12
- every lookup's outcome per chunk: {'new: stored by this request': 114, 'pulled': 102, 'held (on GPU already)': 758}
- pulls per stored chunk: {0: 12, 1: 102}
- median time from a chunk's last block computed on P to the chunk in memory: 136.350 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

## Pulls

- D requests that loaded from LMCache: 30; bytes 2994.7 MB
- the server's lookup to the copy on D's GPU done, all from memory: median 13.930 ms, p90 37.620 ms; with chunks from disk: median – ms, p90 – ms
- of which the server's lookup and prefetch (where a disk read happens): median 2.190 ms from memory, – ms with disk
- and its copy to D's GPU (retrieve start to end): median 5.020 ms
- D requests whose lookup finished before P's store of the same turn: 0
- P's stores (planned to stored): median 128.630 ms, p90 283.970 ms

## Checks

- chunk tokens match prompt: 114 of 114
- pulled blocks same hash as p: 1632 of 1632
- chunks with full chain: 114 of 114

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 1 | 0 | 925 | 0 | 0 | 0 | 925 | new: stored by this request 3 | – | – | 768 |
| P | 2 | 0 | 949 | 32 | 0 | 0 | 917 | new: stored by this request 3 | – | – | 768 |
| P | 3 | 0 | 337 | 16 | 0 | 0 | 321 | new: stored by this request 1 | – | – | 256 |
| P | 0 | 0 | 817 | 16 | 0 | 0 | 801 | new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 0 | 337 | 0 | 256 | 256 | 81 | pulled 1 | L1 1 | 18.250 | 0 |
| D1 | 1 | 0 | 925 | 0 | 768 | 768 | 157 | pulled 3 | L1 3 | 21.140 | 0 |
| D2 | 2 | 0 | 949 | 0 | 768 | 768 | 181 | pulled 3 | L1 3 | 19.410 | 0 |
| D1 | 0 | 0 | 817 | 0 | 768 | 768 | 49 | pulled 3 | L1 3 | 24.260 | 0 |
| P | 3 | 1 | 846 | 336 | 256 | 0 | 510 | held (on GPU already) 1, new: stored by this request 2 | – | – | 512 |
| P | 2 | 1 | 2023 | 944 | 768 | 0 | 1079 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| P | 1 | 1 | 2336 | 912 | 768 | 0 | 1424 | held (on GPU already) 3, new: stored by this request 6 | – | – | 1536 |
| P | 0 | 1 | 1799 | 816 | 768 | 0 | 983 | held (on GPU already) 3, new: stored by this request 4 | – | – | 1024 |
| D3 | 3 | 1 | 846 | 464 | 768 | 304 | 78 | held (on GPU already) 1, pulled 2 | L1 2 | 8.760 | 0 |
| D2 | 2 | 1 | 2023 | 1072 | 1792 | 720 | 231 | held (on GPU already) 4, pulled 3 | L1 3 | 11.440 | 0 |
| D1 | 1 | 1 | 2336 | 1040 | 2304 | 1264 | 32 | held (on GPU already) 4, pulled 5 | L1 5 | 16.310 | 0 |
| D1 | 0 | 1 | 1799 | 944 | 1792 | 848 | 7 | held (on GPU already) 3, pulled 4 | L1 4 | 18.560 | 0 |
| P | 3 | 2 | 1226 | 832 | 768 | 0 | 394 | held (on GPU already) 3, new: stored by this request 1 | – | – | 256 |
| P | 2 | 2 | 3246 | 2016 | 1792 | 0 | 1230 | held (on GPU already) 7, new: stored by this request 5 | – | – | 1280 |
| P | 1 | 2 | 3921 | 2336 | 2304 | 0 | 1585 | held (on GPU already) 9, new: stored by this request 6 | – | – | 1536 |
| P | 0 | 2 | 3494 | 1792 | 1792 | 0 | 1702 | held (on GPU already) 7, new: stored by this request 6 | – | – | 1536 |
| D3 | 3 | 2 | 1226 | 960 | 1024 | 64 | 202 | held (on GPU already) 3, pulled 1 | L1 1 | 7.260 | 0 |
| D2 | 2 | 2 | 3246 | 2144 | 3072 | 928 | 174 | held (on GPU already) 8, pulled 4 | L1 4 | 12.040 | 0 |
| D1 | 1 | 2 | 3921 | 2448 | 3840 | 1392 | 81 | held (on GPU already) 9, pulled 6 | L1 6 | 15.210 | 0 |
| D1 | 0 | 2 | 3494 | 1920 | 3328 | 1408 | 166 | held (on GPU already) 7, pulled 6 | L1 6 | 38.790 | 0 |
| P | 3 | 3 | 1639 | 1216 | 1024 | 0 | 423 | held (on GPU already) 4, new: stored by this request 2 | – | – | 512 |
| P | 2 | 3 | 4343 | 3232 | 3072 | 0 | 1111 | held (on GPU already) 12, new: stored by this request 4 | – | – | 1024 |
| D3 | 3 | 3 | 1639 | 1344 | 1536 | 192 | 103 | held (on GPU already) 5, pulled 1 | L1 1 | 8.430 | 0 |
| P | 1 | 3 | 5032 | 3920 | 3840 | 0 | 1112 | held (on GPU already) 15, new: stored by this request 4 | – | – | 1024 |
| P | 0 | 3 | 4561 | 3488 | 3328 | 0 | 1073 | held (on GPU already) 13, new: stored by this request 4 | – | – | 1024 |
| D2 | 2 | 3 | 4343 | 3360 | 4096 | 736 | 247 | held (on GPU already) 13, pulled 3 | L1 3 | 11.140 | 0 |
| D1 | 1 | 3 | 5032 | 4048 | 4864 | 816 | 168 | held (on GPU already) 15, pulled 4 | L1 4 | 10.310 | 0 |
| D1 | 0 | 3 | 4561 | 3616 | 4352 | 736 | 209 | held (on GPU already) 14, pulled 3 | L1 3 | 35.680 | 0 |
| P | 3 | 4 | 2336 | 1632 | 1536 | 0 | 704 | held (on GPU already) 6, new: stored by this request 3 | – | – | 768 |
| D3 | 3 | 4 | 2336 | 1760 | 2304 | 544 | 32 | held (on GPU already) 6, pulled 3 | L1 3 | 8.670 | 0 |
| P | 2 | 4 | 5393 | 4336 | 4096 | 0 | 1057 | held (on GPU already) 16, new: stored by this request 5 | – | – | 1280 |
| P | 1 | 4 | 6995 | 5024 | 4864 | 0 | 1971 | held (on GPU already) 19, new: stored by this request 8 | – | – | 2048 |
| P | 0 | 4 | 5813 | 4560 | 4352 | 0 | 1253 | held (on GPU already) 17, new: stored by this request 5 | – | – | 1280 |
| D2 | 2 | 4 | 5393 | 4464 | 5376 | 912 | 17 | held (on GPU already) 17, pulled 4 | L1 4 | 12.560 | 0 |
| D1 | 1 | 4 | 6995 | 5152 | 6912 | 1760 | 83 | held (on GPU already) 20, pulled 7 | L1 7 | 14.260 | 0 |
| D1 | 0 | 4 | 5813 | 4688 | 5632 | 944 | 181 | held (on GPU already) 18, pulled 4 | L1 4 | 37.720 | 0 |
| P | 3 | 5 | 2718 | 2336 | 2304 | 0 | 382 | held (on GPU already) 9, new: stored by this request 1 | – | – | 256 |
| D3 | 3 | 5 | 2718 | 2448 | 2560 | 112 | 158 | held (on GPU already) 9, pulled 1 | L1 1 | 8.670 | 0 |
| P | 2 | 5 | 6497 | 5392 | 5376 | 0 | 1105 | held (on GPU already) 21, new: stored by this request 4 | – | – | 1024 |
| P | 1 | 5 | 7585 | 6992 | 6912 | 0 | 593 | held (on GPU already) 27, new: stored by this request 2 | – | – | 512 |
| D2 | 2 | 5 | 6497 | 5520 | 6400 | 880 | 97 | held (on GPU already) 21, pulled 4 | L1 4 | 11.220 | 0 |
| P | 0 | 5 | 6925 | 5808 | 5632 | 0 | 1117 | held (on GPU already) 22, new: stored by this request 5 | – | – | 1280 |
| D1 | 1 | 5 | 7585 | 7120 | 7424 | 304 | 161 | held (on GPU already) 27, pulled 2 | L1 2 | 10.650 | 0 |
| P | 3 | 6 | 3102 | 2704 | 2560 | 0 | 398 | held (on GPU already) 10, new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 5 | 6925 | 5936 | 6912 | 976 | 13 | held (on GPU already) 23, pulled 4 | L1 4 | 37.940 | 0 |
| D3 | 3 | 6 | 3102 | 2832 | 3072 | 240 | 30 | held (on GPU already) 11, pulled 1 | L1 1 | 9.400 | 0 |
| P | 2 | 6 | 7503 | 6496 | 6400 | 0 | 1007 | held (on GPU already) 25, new: stored by this request 4 | – | – | 1024 |
| P | 1 | 6 | 8705 | 7584 | 7424 | 0 | 1121 | held (on GPU already) 29, new: stored by this request 5 | – | – | 1280 |
| D2 | 2 | 6 | 7503 | 6624 | 7424 | 800 | 79 | held (on GPU already) 25, pulled 4 | L1 4 | 13.040 | 0 |
| P | 0 | 6 | 8025 | 6912 | 6912 | 0 | 1113 | held (on GPU already) 27, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 6 | 8705 | 7712 | 8704 | 992 | 1 | held (on GPU already) 30, pulled 4 | L1 4 | 14.070 | 0 |
| D1 | 0 | 6 | 8025 | 7040 | 7936 | 896 | 89 | held (on GPU already) 27, pulled 4 | L1 4 | 34.920 | 0 |
| P | 1 | 7 | 9765 | 8704 | 8704 | 0 | 1061 | held (on GPU already) 34, new: stored by this request 4 | – | – | 1024 |
| P | 0 | 7 | 9097 | 8016 | 7936 | 0 | 1081 | held (on GPU already) 31, new: stored by this request 4 | – | – | 1024 |
| D1 | 1 | 7 | 9765 | 8832 | 9728 | 896 | 37 | held (on GPU already) 34, pulled 4 | L1 4 | 13.930 | 0 |
| D1 | 0 | 7 | 9097 | 8144 | 8960 | 816 | 137 | held (on GPU already) 31, pulled 4 | L1 4 | 37.620 | 0 |

## One chunk's life: 1dd495890c85343c

Conversation 1, turn 0, tokens 0-256 (system, doc c8db6e06ff46669e-50302-52227); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0041 | P looked it up (turn 0): new: stored by this request |
| 0.0070 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 2d665407) |
| 0.0070 | P computed and cached block 15 (GPU block 16, hash 549e5b84eeeceb71, request 2d665407) |
| 0.0077 | P's connector planned the store |
| 0.0187 | P's worker sent it to the server |
| 0.0238 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0242 | server memory (L1) reserved |
| 0.2378 | written to memory (L1) |
| 0.8082 | D1 looked it up (turn 0): pulled, from L1 |
| 0.8293 | copied to D1's GPU |
| 0.8344 | D1 cached block 0 (GPU block 1; same hash as P's: True) |
| 2.2873 | P looked it up (turn 1): held (on GPU already) |
| 3.0935 | D1 looked it up (turn 1): held (on GPU already) |
| 4.5966 | P looked it up (turn 2): held (on GPU already) |
| 5.3577 | D1 looked it up (turn 2): held (on GPU already) |
| 6.9281 | P looked it up (turn 3): held (on GPU already) |
| 7.5630 | D1 looked it up (turn 3): held (on GPU already) |
| 9.1936 | P looked it up (turn 4): held (on GPU already) |
| 9.9263 | D1 looked it up (turn 4): held (on GPU already) |
| 11.6009 | P looked it up (turn 5): held (on GPU already) |
| 12.2110 | D1 looked it up (turn 5): held (on GPU already) |
| 13.9215 | P looked it up (turn 6): held (on GPU already) |
| 14.5873 | D1 looked it up (turn 6): held (on GPU already) |
| 16.3289 | P looked it up (turn 7): held (on GPU already) |
| 17.0004 | D1 looked it up (turn 7): held (on GPU already) |

## One D request: conversation 1, turn 4 on D1

| s | what |
| ---: | --- |
| 9.1827 | the proxy received the turn |
| 9.4138 | P answered (stored) |
| 9.9234 | D1 took the request |
| 9.9263 | the server got its lookup |
| 9.9284 | the server found 27 of 27 chunks (memory 1792, disk 0 tokens) |
| 9.9293 | its connector had the lookup's answer |
| 9.9318 | the server's copy to the GPU began |
| 9.9405 | copied (205520896 bytes) |
| 9.9405 | the worker saw the load finished |
| 9.9735 | the first token reached the proxy |
| 11.5825 | finished |
| 11.5826 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).
