# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |
| ours-hbh | 4/4 | 3/4 | 9.042 |
| t2-hbh | 4/4 | 3/4 | 8.516 |
| t2-rank256 | 4/4 | 3/4 | 9.597 |
| splitzip | 4/4 | 3/4 | 12.044 |
| exans | 4/4 | 3/4 | 10.924 |
| adacode | 4/4 | 3/4 | 11.372 |
| aquakv-2 | 0/4 | 0/4 | 2.126 |
| aquakv-3 | 0/4 | 2/4 | 3.088 |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5 | 0.49 | 0.49 | 215 / 524 | 11.2 | 16.000 | 1.1 / 2.6 | 11.4 | 142.5 |
| raw | 1.0 | 0.95 | 0.95 | 217 / 667 | 11.2 | 16.000 | 1.1 / 2.7 | 12.8 | 130.5 |
| raw | 2.0 | 1.81 | 1.81 | 250 / 668 | 11.3 | 16.000 | 1.2 / 2.7 | 14.2 | 118.6 |
| raw | inf | 4.40 | 0.14 | 4728 / 5677 | 12.8 | 16.000 | 2.1 / 2.9 | 46.3 | 71.9 |
| ours-hbh | 0.5 | 0.49 | 0.49 | 403 / 750 | 11.2 | 8.671 | 33.1 / 166.2 | 5.6 | 140.7 |
| ours-hbh | 1.0 | 0.95 | 0.95 | 402 / 955 | 11.2 | 8.671 | 33.0 / 161.5 | 5.7 | 138.0 |
| ours-hbh | 2.0 | 1.79 | 1.79 | 421 / 939 | 11.9 | 8.671 | 33.1 / 156.3 | 5.6 | 138.7 |
| ours-hbh | inf | 3.94 | 0.00 | 5625 / 6426 | 16.9 | 8.671 | 32.0 / 146.7 | 6.0 | 131.6 |
| t2-hbh | 0.5 | 0.49 | 0.49 | 413 / 786 | 11.2 | 8.146 | 44.5 / 163.9 | 5.4 | 143.2 |
| t2-hbh | 1.0 | 0.95 | 0.95 | 414 / 970 | 11.2 | 8.146 | 44.6 / 162.2 | 5.6 | 139.7 |
| t2-hbh | 2.0 | 1.79 | 1.79 | 429 / 997 | 11.9 | 8.146 | 44.5 / 157.0 | 5.4 | 141.2 |
| t2-hbh | inf | 3.79 | 0.00 | 5603 / 6786 | 17.7 | 8.146 | 49.4 / 147.0 | 5.9 | 131.7 |
| t2-rank256 | 0.5 | 0.49 | 0.49 | 267 / 611 | 11.2 | 9.229 | 33.1 / 26.1 | 6.8 | 141.7 |
| t2-rank256 | 1.0 | 0.95 | 0.95 | 268 / 715 | 11.2 | 9.229 | 33.1 / 25.9 | 7.1 | 136.5 |
| t2-rank256 | 2.0 | 1.81 | 1.81 | 293 / 773 | 11.4 | 9.229 | 33.1 / 24.3 | 7.1 | 134.4 |
| t2-rank256 | inf | 4.03 | 0.13 | 4923 / 6330 | 13.1 | 9.229 | 37.6 / 21.6 | 7.7 | 91.7 |
| splitzip | 0.5 | 0.49 | 0.49 | 223 / 526 | 11.2 | 12.041 | 7.8 / 5.2 | 9.0 | 139.2 |
| splitzip | 1.0 | 0.95 | 0.95 | 223 / 662 | 11.2 | 12.041 | 7.7 / 5.4 | 9.2 | 136.2 |
| splitzip | 2.0 | 1.81 | 1.81 | 251 / 670 | 11.3 | 12.041 | 7.3 / 5.3 | 9.2 | 137.0 |
| splitzip | inf | 4.62 | 0.00 | 4662 / 5276 | 13.1 | 12.041 | 6.0 / 4.9 | 10.0 | 124.5 |
| exans | 0.5 | 0.49 | 0.49 | 271 / 677 | 11.2 | 10.883 | 53.2 / 12.1 | 9.8 | 139.2 |
| exans | 1.0 | 0.95 | 0.95 | 271 / 768 | 11.2 | 10.883 | 52.6 / 11.9 | 11.2 | 127.0 |
| exans | 2.0 | 1.80 | 1.80 | 302 / 825 | 11.3 | 10.883 | 52.6 / 11.9 | 11.4 | 120.4 |
| exans | inf | 3.87 | 0.00 | 5351 / 6667 | 12.7 | 10.883 | 48.4 / 11.8 | 43.9 | 34.2 |
| adacode | 0.5 | 0.49 | 0.49 | 249 / 552 | 11.2 | 11.372 | 27.6 / 8.1 | 7.3 | 135.5 |
| adacode | 1.0 | 0.95 | 0.95 | 249 / 674 | 11.2 | 11.372 | 25.4 / 8.1 | 7.6 | 131.6 |
| adacode | 2.0 | 1.81 | 1.81 | 273 / 698 | 11.3 | 11.372 | 24.2 / 7.4 | 7.4 | 133.2 |
| adacode | inf | 4.45 | 0.00 | 4815 / 5566 | 13.0 | 11.372 | 14.3 / 6.5 | 7.9 | 128.2 |
| aquakv-2 | 0.5 | 0.49 | 0.49 | 388 / 977 | 11.2 | 2.130 | 147.0 / 43.4 | 0.0 | 122.1 |
| aquakv-2 | 1.0 | 0.95 | 0.95 | 388 / 1114 | 11.2 | 2.130 | 146.8 / 42.8 | 0.0 | 117.9 |
| aquakv-2 | 2.0 | 1.79 | 1.79 | 430 / 1191 | 11.6 | 2.130 | 147.2 / 43.1 | 0.0 | 122.4 |
| aquakv-2 | inf | 2.83 | 0.00 | 7209 / 9783 | 12.2 | 2.130 | 144.9 / 43.5 | 0.0 | 111.3 |
| aquakv-3 | 0.5 | 0.49 | 0.49 | 481 / 1304 | 11.2 | 3.092 | 233.6 / 48.1 | 1.9 | 125.3 |
| aquakv-3 | 1.0 | 0.94 | 0.94 | 481 / 1444 | 11.2 | 3.092 | 233.5 / 47.4 | 2.1 | 119.6 |
| aquakv-3 | 2.0 | 1.75 | 1.75 | 712 / 1678 | 11.6 | 3.092 | 234.1 / 47.9 | 2.0 | 113.5 |
| aquakv-3 | inf | 2.28 | 0.00 | 8464 / 12516 | 11.8 | 3.092 | 232.2 / 48.1 | 2.5 | 109.1 |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 512 | 1 | 128 | 8 | sz1 | raw | 82 / 84 | 0.68 | 16.000 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | raw | 123 / 128 | 0.66 | 16.000 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | raw | 226 / 249 | 9.90 | 16.000 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | raw | 202 / 206 | 1.25 | 16.000 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | raw | 216 / 219 | 0.61 | 16.000 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | raw | 368 / 395 | 9.01 | 16.000 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | raw | 354 / 399 | 2.25 | 16.000 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | raw | 413 / 417 | 0.53 | 16.000 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | raw | 670 / 718 | 7.56 | 16.000 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | raw | 661 / 713 | 3.80 | 16.000 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | raw | 876 / 880 | 0.41 | 16.000 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | raw | 1252 / 1343 | 5.74 | 16.000 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | raw | 3852 / 4012 | 0.18 | 16.000 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | raw | 2130 / 2908 | 7.84 | 16.000 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | raw | 2423 / 2764 | 3.84 | 16.000 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | raw | 2984 / 3131 | 4.26 | 16.000 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | raw | 3434 / 6961 | 10.79 | 16.000 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | raw | 4692 / 9215 | 2.19 | 16.000 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | raw | 3957 / 6976 | 1.84 | 16.000 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | ours-hbh | 250 / 254 | 0.62 | 8.532 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | ours-hbh | 298 / 303 | 0.59 | 8.682 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | ours-hbh | 766 / 1066 | 6.84 | 29.719 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | ours-hbh | 374 / 396 | 1.12 | 8.682 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | ours-hbh | 400 / 405 | 0.55 | 8.503 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | ours-hbh | 990 / 1174 | 6.69 | 8.742 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | ours-hbh | 536 / 662 | 2.03 | 8.682 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | ours-hbh | 626 / 630 | 0.48 | 8.521 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | ours-hbh | 1379 / 1482 | 5.69 | 8.715 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | ours-hbh | 1018 / 1122 | 3.22 | 8.691 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | ours-hbh | 1135 / 1589 | 0.36 | 8.601 | 1 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | ours-hbh | 1986 / 2141 | 4.54 | 8.689 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | ours-hbh | 3979 / 4015 | 0.18 | - | 4 |
| native | 1024 | 32 | 128 | 64 | sz1024 | ours-hbh | 3751 / 4009 | 5.90 | 8.680 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | ours-hbh | 3146 / 3407 | 3.28 | 8.671 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | ours-hbh | 3839 / 4020 | 3.61 | 10.070 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | ours-hbh | 5363 / 7124 | 7.48 | 8.693 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | ours-hbh | 4560 / 8511 | 2.19 | 8.668 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | ours-hbh | 4758 / 6920 | 1.71 | 8.668 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | t2-hbh | 260 / 264 | 0.61 | 8.278 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | t2-hbh | 310 / 317 | 0.59 | 8.095 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | t2-hbh | 906 / 1018 | 6.94 | 29.399 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | t2-hbh | 387 / 415 | 1.12 | 8.095 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | t2-hbh | 412 / 418 | 0.54 | 8.185 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | t2-hbh | 1070 / 1211 | 6.45 | 8.297 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | t2-hbh | 543 / 690 | 2.00 | 8.095 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | t2-hbh | 633 / 640 | 0.47 | 8.191 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | t2-hbh | 1380 / 1520 | 5.64 | 8.241 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | t2-hbh | 1052 / 1145 | 3.18 | 8.153 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | t2-hbh | 1150 / 1163 | 0.37 | 8.103 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | t2-hbh | 1997 / 2176 | 4.55 | 8.156 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | t2-hbh | 4023 / 4063 | 0.17 | - | 4 |
| native | 1024 | 32 | 128 | 64 | sz1024 | t2-hbh | 3730 / 4085 | 5.84 | 8.170 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | t2-hbh | 3069 / 3520 | 3.23 | 8.146 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | t2-hbh | 3622 / 4234 | 3.53 | 9.553 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | t2-hbh | 5034 / 7901 | 7.42 | 8.175 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | t2-hbh | 4874 / 9819 | 2.10 | 8.148 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | t2-hbh | 4469 / 7212 | 1.72 | 8.148 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | t2-rank256 | 121 / 123 | 0.67 | 9.349 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | t2-rank256 | 168 / 171 | 0.64 | 9.175 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | t2-rank256 | 283 / 481 | 9.08 | 30.615 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | t2-rank256 | 244 / 266 | 1.22 | 9.175 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | t2-rank256 | 268 / 273 | 0.59 | 9.257 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | t2-rank256 | 391 / 649 | 8.39 | 9.379 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | t2-rank256 | 417 / 491 | 2.16 | 9.176 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | t2-rank256 | 481 / 485 | 0.51 | 9.250 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | t2-rank256 | 722 / 970 | 7.04 | 9.321 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | t2-rank256 | 723 / 906 | 3.55 | 9.228 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | t2-rank256 | 985 / 1000 | 0.39 | 9.188 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | t2-rank256 | 1329 / 1751 | 5.30 | 9.245 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | t2-rank256 | 3923 / 3994 | 0.18 | - | 4 |
| native | 1024 | 32 | 128 | 64 | sz1024 | t2-rank256 | 2487 / 3488 | 7.04 | 9.254 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | t2-rank256 | 2488 / 3211 | 3.60 | 9.229 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | t2-rank256 | 2940 / 3978 | 3.98 | 10.642 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | t2-rank256 | 4768 / 9368 | 8.31 | 9.258 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | t2-rank256 | 4643 / 9265 | 2.24 | 9.220 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | t2-rank256 | 3717 / 6685 | 1.86 | 9.220 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | splitzip | 94 / 95 | 0.68 | 12.038 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | splitzip | 135 / 139 | 0.65 | 12.039 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | splitzip | 258 / 313 | 9.60 | 12.046 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | splitzip | 208 / 219 | 1.24 | 12.039 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | splitzip | 225 / 226 | 0.61 | 12.039 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | splitzip | 388 / 457 | 8.83 | 12.041 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | splitzip | 366 / 404 | 2.23 | 12.039 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | splitzip | 419 / 425 | 0.53 | 12.041 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | splitzip | 670 / 792 | 7.46 | 12.039 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | splitzip | 662 / 729 | 3.78 | 12.039 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | splitzip | 876 / 880 | 0.41 | 12.046 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | splitzip | 1278 / 1420 | 5.63 | 12.040 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | splitzip | 2716 / 2778 | 0.22 | 12.049 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | splitzip | 2419 / 2687 | 7.74 | 12.039 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | splitzip | 2385 / 2718 | 3.84 | 12.041 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | splitzip | 2817 / 3150 | 4.35 | 12.041 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | splitzip | 3792 / 6615 | 10.50 | 12.040 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | splitzip | 3916 / 7428 | 2.51 | 12.043 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | splitzip | 3844 / 5817 | 2.00 | 12.043 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | exans | 104 / 105 | 0.68 | 11.020 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | exans | 160 / 215 | 0.64 | 10.929 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | exans | 271 / 359 | 9.19 | 11.212 | 1 |
| native | 1024 | 2 | 128 | 8 | sz1024 | exans | 237 / 265 | 1.22 | 10.929 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | exans | 274 / 279 | 0.59 | 10.881 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | exans | 405 / 511 | 8.58 | 11.025 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | exans | 433 / 510 | 2.14 | 10.929 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | exans | 523 / 537 | 0.50 | 10.893 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | exans | 722 / 934 | 7.11 | 11.021 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | exans | 721 / 914 | 3.57 | 10.927 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | exans | 1089 / 1181 | 0.38 | 10.907 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | exans | 1344 / 1751 | 5.28 | 10.928 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | exans | 3088 / 3284 | 0.21 | 10.907 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | exans | 2558 / 3338 | 7.12 | 10.928 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | exans | 2529 / 3381 | 3.54 | 10.883 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | exans | 2919 / 4132 | 3.98 | 10.890 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | exans | 4685 / 9101 | 8.64 | 10.929 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | exans | 5039 / 9771 | 2.06 | 10.894 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | exans | 3792 / 7761 | 1.81 | 10.894 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | adacode | 119 / 121 | 0.67 | 11.383 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | adacode | 162 / 167 | 0.64 | 11.375 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | adacode | 274 / 474 | 9.10 | 11.416 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | adacode | 223 / 234 | 1.23 | 11.375 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | adacode | 250 / 258 | 0.60 | 11.372 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | adacode | 420 / 615 | 8.36 | 11.393 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | adacode | 385 / 444 | 2.20 | 11.375 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | adacode | 441 / 449 | 0.52 | 11.371 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | adacode | 706 / 935 | 7.11 | 11.381 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | adacode | 680 / 816 | 3.67 | 11.375 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | adacode | 893 / 904 | 0.41 | 11.369 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | adacode | 1287 / 1523 | 5.52 | 11.375 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | adacode | 2751 / 2805 | 0.22 | 11.369 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | adacode | 2481 / 2991 | 7.42 | 11.375 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | adacode | 2377 / 2844 | 3.79 | 11.372 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | adacode | 2814 / 3451 | 4.26 | 11.372 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | adacode | 4104 / 8416 | 9.56 | 11.375 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | adacode | 4115 / 7652 | 2.45 | 11.370 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | adacode | 3950 / 6070 | 1.92 | 11.370 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | aquakv-2 | 151 / 154 | 0.65 | 2.211 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | aquakv-2 | 226 / 231 | 0.62 | 2.157 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | aquakv-2 | 337 / 725 | 8.25 | 2.537 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | aquakv-2 | 233 / 379 | 1.19 | 2.157 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | aquakv-2 | 391 / 396 | 0.55 | 2.130 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | aquakv-2 | 471 / 872 | 7.65 | 2.320 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | aquakv-2 | 491 / 709 | 1.99 | 2.157 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | aquakv-2 | 755 / 759 | 0.45 | 2.116 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | aquakv-2 | 828 / 1365 | 6.30 | 2.211 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | aquakv-2 | 798 / 1324 | 3.21 | 2.157 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | aquakv-2 | 1553 / 1559 | 0.32 | 2.109 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | aquakv-2 | 1469 / 2525 | 4.60 | 2.157 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | aquakv-2 | 3428 / 3436 | 0.19 | 2.106 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | aquakv-2 | 3305 / 5147 | 5.59 | 2.157 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | aquakv-2 | 3272 / 5110 | 2.82 | 2.130 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | aquakv-2 | 4065 / 6789 | 3.07 | 2.132 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | aquakv-2 | 7354 / 12344 | 6.00 | 2.157 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | aquakv-2 | 7583 / 12964 | 1.48 | 2.116 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | aquakv-2 | 6179 / 12789 | 1.37 | 2.116 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | aquakv-3 | 178 / 182 | 0.64 | 3.168 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | aquakv-3 | 275 / 277 | 0.60 | 3.117 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | aquakv-3 | 394 / 790 | 7.98 | 3.471 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | aquakv-3 | 285 / 471 | 1.15 | 3.117 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | aquakv-3 | 481 / 484 | 0.52 | 3.092 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | aquakv-3 | 522 / 980 | 7.34 | 3.269 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | aquakv-3 | 544 / 890 | 1.89 | 3.117 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | aquakv-3 | 941 / 945 | 0.41 | 3.080 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | aquakv-3 | 864 / 1701 | 5.86 | 3.168 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | aquakv-3 | 852 / 1657 | 2.97 | 3.117 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | aquakv-3 | 1922 / 1932 | 0.29 | 3.073 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | aquakv-3 | 1676 / 3202 | 4.08 | 3.117 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | aquakv-3 | 4167 / 4174 | 0.17 | 3.070 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | aquakv-3 | 4659 / 6840 | 4.52 | 3.117 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | aquakv-3 | 4573 / 6742 | 2.28 | 3.092 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | aquakv-3 | 5670 / 8408 | 2.46 | 3.094 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | aquakv-3 | 10285 / 15100 | 4.75 | 3.117 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | aquakv-3 | 10505 / 15558 | 1.17 | 3.080 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | aquakv-3 | 9038 / 15739 | 1.11 | 3.080 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

14 in the engine logs, 0 inside a measured point.

- 17:52:42 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 17:52:42 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 17:53:07 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:53:07 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:53:07 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:53:07 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:53:07 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:53:07 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:53:07 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:53:07 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:53:07 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:53:07 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:53:08 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:53:08 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

