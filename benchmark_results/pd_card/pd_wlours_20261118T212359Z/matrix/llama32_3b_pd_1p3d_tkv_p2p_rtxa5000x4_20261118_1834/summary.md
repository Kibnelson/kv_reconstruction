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
| adacode | 4/4 | 3/4 | 11.372 |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5 | 0.49 | 0.49 | 215 / 526 | 11.2 | 16.000 | 1.1 / 2.7 | 11.4 | 141.6 |
| raw | 1.0 | 0.95 | 0.95 | 215 / 664 | 11.2 | 16.000 | 1.2 / 2.9 | 12.7 | 130.4 |
| raw | 2.0 | 1.81 | 1.81 | 248 / 669 | 11.3 | 16.000 | 1.2 / 2.8 | 13.8 | 120.3 |
| raw | inf | 4.40 | 0.00 | 4671 / 5666 | 13.0 | 16.000 | 2.5 / 2.9 | 44.5 | 63.2 |
| ours-hbh | 0.5 | 0.49 | 0.49 | 304 / 652 | 11.2 | 8.671 | 33.1 / 67.4 | 5.6 | 140.0 |
| ours-hbh | 1.0 | 0.95 | 0.95 | 303 / 753 | 11.2 | 8.671 | 32.9 / 64.6 | 5.8 | 135.9 |
| ours-hbh | 2.0 | 1.80 | 1.80 | 323 / 792 | 11.4 | 8.671 | 33.0 / 62.2 | 5.9 | 133.1 |
| ours-hbh | inf | 4.12 | 0.13 | 5005 / 6161 | 14.3 | 8.671 | 31.9 / 51.8 | 6.5 | 122.1 |
| t2-hbh | 0.5 | 0.49 | 0.49 | 314 / 685 | 11.2 | 8.146 | 44.2 / 66.8 | 5.5 | 141.6 |
| t2-hbh | 1.0 | 0.95 | 0.95 | 314 / 780 | 11.2 | 8.146 | 44.3 / 65.5 | 5.8 | 134.9 |
| t2-hbh | 2.0 | 1.80 | 1.80 | 337 / 844 | 11.4 | 8.146 | 44.5 / 60.8 | 5.8 | 131.4 |
| t2-hbh | inf | 3.85 | 0.12 | 5325 / 6734 | 13.9 | 8.146 | 49.5 / 54.1 | 6.7 | 113.1 |
| t2-rank256 | 0.5 | 0.49 | 0.49 | 254 / 610 | 11.2 | 9.229 | 31.9 / 17.2 | 6.9 | 139.7 |
| t2-rank256 | 1.0 | 0.95 | 0.95 | 257 / 709 | 11.2 | 9.229 | 33.4 / 16.9 | 7.2 | 134.0 |
| t2-rank256 | 2.0 | 1.81 | 1.81 | 285 / 754 | 11.4 | 9.229 | 32.7 / 16.7 | 7.4 | 127.8 |
| t2-rank256 | inf | 4.06 | 0.00 | 5212 / 6285 | 12.9 | 9.229 | 37.1 / 16.8 | 8.8 | 86.2 |
| splitzip | 0.5 | 0.49 | 0.49 | 223 / 525 | 11.2 | 12.041 | 8.2 / 5.1 | 9.0 | 139.6 |
| splitzip | 1.0 | 0.95 | 0.95 | 224 / 661 | 11.2 | 12.041 | 7.9 / 5.1 | 9.4 | 134.9 |
| splitzip | 2.0 | 1.81 | 1.81 | 251 / 667 | 11.3 | 12.041 | 7.2 / 5.3 | 9.8 | 130.2 |
| splitzip | inf | 4.61 | 0.14 | 4531 / 5297 | 13.1 | 12.041 | 6.3 / 4.9 | 10.6 | 120.0 |
| adacode | 0.5 | 0.49 | 0.49 | 247 / 552 | 11.2 | 11.372 | 26.7 / 7.9 | 7.3 | 135.9 |
| adacode | 1.0 | 0.95 | 0.95 | 249 / 672 | 11.2 | 11.372 | 25.5 / 8.1 | 7.7 | 131.4 |
| adacode | 2.0 | 1.81 | 1.81 | 272 / 691 | 11.3 | 11.372 | 24.1 / 8.1 | 7.9 | 125.7 |
| adacode | inf | 4.42 | 0.14 | 4390 / 5589 | 13.3 | 11.372 | 14.0 / 6.5 | 8.5 | 119.7 |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 512 | 1 | 128 | 8 | sz1 | raw | 83 / 85 | 0.69 | 16.000 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | raw | 124 / 127 | 0.66 | 16.000 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | raw | 226 / 239 | 9.94 | 16.000 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | raw | 200 / 207 | 1.25 | 16.000 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | raw | 215 / 219 | 0.61 | 16.000 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | raw | 365 / 395 | 9.03 | 16.000 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | raw | 354 / 401 | 2.26 | 16.000 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | raw | 410 / 413 | 0.53 | 16.000 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | raw | 659 / 712 | 7.60 | 16.000 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | raw | 646 / 710 | 3.82 | 16.000 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | raw | 873 / 880 | 0.41 | 16.000 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | raw | 1255 / 1327 | 5.75 | 16.000 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | raw | 2099 / 2190 | 0.26 | 16.000 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | raw | 2359 / 2621 | 7.87 | 16.000 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | raw | 2334 / 2782 | 3.88 | 16.000 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | raw | 2824 / 3339 | 4.27 | 16.000 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | raw | 3694 / 5556 | 10.78 | 16.000 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | raw | 4559 / 9037 | 2.22 | 16.000 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | raw | 3466 / 7202 | 1.92 | 16.000 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | ours-hbh | 154 / 155 | 0.65 | 8.531 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | ours-hbh | 200 / 203 | 0.63 | 8.682 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | ours-hbh | 430 / 453 | 8.83 | 29.719 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | ours-hbh | 274 / 286 | 1.19 | 8.682 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | ours-hbh | 305 / 312 | 0.58 | 8.503 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | ours-hbh | 569 / 612 | 8.08 | 8.742 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | ours-hbh | 460 / 470 | 2.15 | 8.682 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | ours-hbh | 531 / 541 | 0.50 | 8.522 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | ours-hbh | 888 / 904 | 6.90 | 8.715 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | ours-hbh | 775 / 876 | 3.55 | 8.691 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | ours-hbh | 1112 / 1125 | 0.38 | 8.649 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | ours-hbh | 1454 / 1598 | 5.25 | 8.689 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | ours-hbh | 2544 / 2556 | 0.23 | 8.668 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | ours-hbh | 2755 / 3134 | 7.03 | 8.680 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | ours-hbh | 2635 / 3159 | 3.54 | 8.671 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | ours-hbh | 3021 / 3875 | 3.95 | 10.070 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | ours-hbh | 4192 / 7679 | 9.26 | 8.692 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | ours-hbh | 4745 / 9203 | 2.21 | 8.668 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | ours-hbh | 3913 / 6833 | 1.82 | 8.668 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | t2-hbh | 161 / 165 | 0.65 | 8.278 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | t2-hbh | 210 / 213 | 0.62 | 8.095 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | t2-hbh | 420 / 490 | 8.74 | 29.399 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | t2-hbh | 289 / 317 | 1.19 | 8.095 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | t2-hbh | 314 / 318 | 0.58 | 8.185 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | t2-hbh | 579 / 634 | 8.00 | 8.297 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | t2-hbh | 472 / 537 | 2.10 | 8.095 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | t2-hbh | 543 / 552 | 0.50 | 8.191 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | t2-hbh | 867 / 1001 | 6.75 | 8.241 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | t2-hbh | 808 / 980 | 3.44 | 8.153 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | t2-hbh | 1136 / 1140 | 0.37 | 8.103 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | t2-hbh | 1412 / 1922 | 5.06 | 8.157 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | t2-hbh | 2608 / 2624 | 0.23 | 8.119 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | t2-hbh | 2777 / 3771 | 6.58 | 8.170 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | t2-hbh | 2654 / 3425 | 3.44 | 8.146 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | t2-hbh | 3011 / 4283 | 3.82 | 9.553 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | t2-hbh | 5224 / 9729 | 7.90 | 8.175 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | t2-hbh | 4867 / 9664 | 2.12 | 8.148 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | t2-hbh | 3844 / 7106 | 1.80 | 8.148 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | t2-rank256 | 110 / 113 | 0.67 | 9.349 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | t2-rank256 | 159 / 162 | 0.64 | 9.176 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | t2-rank256 | 269 / 480 | 9.15 | 30.615 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | t2-rank256 | 232 / 253 | 1.23 | 9.176 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | t2-rank256 | 257 / 261 | 0.59 | 9.257 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | t2-rank256 | 420 / 606 | 8.38 | 9.379 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | t2-rank256 | 408 / 486 | 2.17 | 9.176 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | t2-rank256 | 477 / 481 | 0.51 | 9.250 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | t2-rank256 | 691 / 980 | 7.11 | 9.321 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | t2-rank256 | 709 / 884 | 3.58 | 9.228 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | t2-rank256 | 1008 / 1018 | 0.39 | 9.188 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | t2-rank256 | 1326 / 1743 | 5.31 | 9.245 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | t2-rank256 | 2376 / 2381 | 0.24 | 9.190 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | t2-rank256 | 2446 / 3467 | 7.10 | 9.254 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | t2-rank256 | 2462 / 3193 | 3.62 | 9.229 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | t2-rank256 | 2990 / 3950 | 3.95 | 10.642 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | t2-rank256 | 5054 / 9425 | 8.28 | 9.258 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | t2-rank256 | 4660 / 9265 | 2.24 | 9.220 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | t2-rank256 | 3714 / 6693 | 1.87 | 9.220 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | splitzip | 95 / 96 | 0.68 | 12.038 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | splitzip | 136 / 146 | 0.65 | 12.039 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | splitzip | 248 / 298 | 9.67 | 12.046 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | splitzip | 209 / 217 | 1.24 | 12.039 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | splitzip | 225 / 230 | 0.61 | 12.039 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | splitzip | 411 / 455 | 8.74 | 12.041 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | splitzip | 359 / 415 | 2.24 | 12.039 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | splitzip | 419 / 424 | 0.53 | 12.041 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | splitzip | 689 / 781 | 7.40 | 12.039 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | splitzip | 656 / 717 | 3.79 | 12.039 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | splitzip | 877 / 883 | 0.41 | 12.046 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | splitzip | 1264 / 1380 | 5.68 | 12.040 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | splitzip | 2059 / 2067 | 0.26 | 12.049 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | splitzip | 2397 / 2681 | 7.77 | 12.039 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | splitzip | 2402 / 2713 | 3.83 | 12.041 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | splitzip | 2839 / 3180 | 4.32 | 12.041 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | splitzip | 3625 / 6847 | 10.51 | 12.040 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | splitzip | 3931 / 7447 | 2.50 | 12.043 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | splitzip | 3983 / 5934 | 1.93 | 12.043 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | adacode | 120 / 127 | 0.67 | 11.383 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | adacode | 159 / 168 | 0.64 | 11.375 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | adacode | 299 / 464 | 9.05 | 11.416 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | adacode | 220 / 237 | 1.24 | 11.375 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | adacode | 249 / 252 | 0.60 | 11.372 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | adacode | 417 / 616 | 8.38 | 11.393 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | adacode | 378 / 440 | 2.21 | 11.375 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | adacode | 441 / 450 | 0.52 | 11.371 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | adacode | 700 / 920 | 7.15 | 11.381 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | adacode | 675 / 832 | 3.67 | 11.375 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | adacode | 896 / 915 | 0.41 | 11.369 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | adacode | 1293 / 1535 | 5.51 | 11.375 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | adacode | 2102 / 2107 | 0.26 | 11.369 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | adacode | 2414 / 2973 | 7.52 | 11.375 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | adacode | 2398 / 2839 | 3.78 | 11.372 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | adacode | 2826 / 3315 | 4.27 | 11.372 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | adacode | 4044 / 8349 | 9.57 | 11.375 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | adacode | 4091 / 7586 | 2.45 | 11.370 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | adacode | 3623 / 6062 | 2.00 | 11.370 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

14 in the engine logs, 0 inside a measured point.

- 18:35:29 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 18:35:29 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 18:35:37 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 18:35:37 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 18:35:37 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 18:35:37 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 18:35:37 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 18:35:37 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 18:35:38 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 18:35:38 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 18:35:38 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 18:35:38 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 18:35:38 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 18:35:38 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

