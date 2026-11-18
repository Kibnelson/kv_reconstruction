# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |
| nvcomp-ans | 4/4 | 3/4 | 13.369 |
| nvcomp-zstd | 4/4 | 3/4 | 12.898 |
| nvcomp-bitcomp | 4/4 | 3/4 | 13.791 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5 | 0.49 | 0.49 | 393 / 1028 | 11.2 | 16.000 | 1.1 / 2.7 | 219.1 | 9.3 |
| raw | 1.0 | 0.95 | 0.95 | 394 / 1254 | 11.2 | 16.000 | 1.1 / 2.6 | 239.1 | 8.7 |
| raw | 2.0 | 1.79 | 1.79 | 426 / 1418 | 11.3 | 16.000 | 1.1 / 2.7 | 259.2 | 8.1 |
| raw | inf | 3.03 | 0.00 | 6334 / 9079 | 11.8 | 16.000 | 1.1 / 2.9 | 480.7 | 3.7 |
| nvcomp-ans | 0.5 | 0.49 | 0.49 | 390 / 966 | 11.2 | 13.363 | 20.6 / 11.2 | 180.9 | 9.3 |
| nvcomp-ans | 1.0 | 0.95 | 0.95 | 392 / 1160 | 11.2 | 13.363 | 20.2 / 11.1 | 198.8 | 8.7 |
| nvcomp-ans | 2.0 | 1.79 | 1.79 | 421 / 1315 | 11.3 | 13.363 | 20.1 / 11.1 | 215.2 | 8.1 |
| nvcomp-ans | inf | 3.12 | 0.10 | 6457 / 8749 | 12.2 | 13.363 | 17.1 / 11.2 | 394.3 | 3.6 |
| nvcomp-zstd | 0.5 | 0.49 | 0.49 | 520 / 1311 | 11.2 | 12.892 | 141.5 / 27.9 | 172.3 | 9.4 |
| nvcomp-zstd | 1.0 | 0.94 | 0.94 | 520 / 1439 | 11.2 | 12.892 | 141.7 / 26.9 | 192.3 | 8.6 |
| nvcomp-zstd | 2.0 | 1.78 | 1.78 | 557 / 1626 | 11.4 | 12.892 | 141.9 / 26.9 | 214.4 | 8.0 |
| nvcomp-zstd | inf | 2.70 | 0.00 | 7520 / 10335 | 12.0 | 12.892 | 142.2 / 27.6 | 438.6 | 3.5 |
| nvcomp-bitcomp | 0.5 | 0.49 | 0.49 | 393 / 1008 | 11.2 | 13.786 | 17.9 / 11.9 | 189.1 | 9.3 |
| nvcomp-bitcomp | 1.0 | 0.95 | 0.95 | 393 / 1190 | 11.2 | 13.786 | 17.4 / 11.8 | 208.2 | 8.7 |
| nvcomp-bitcomp | 2.0 | 1.79 | 1.79 | 422 / 1353 | 11.3 | 13.786 | 17.3 / 11.7 | 226.7 | 8.1 |
| nvcomp-bitcomp | inf | 3.19 | 0.00 | 6161 / 8498 | 12.2 | 13.786 | 15.0 / 11.9 | 420.9 | 3.7 |

## Link 100 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5 | 0.49 | 0.49 | 222 / 544 | 11.2 | 16.000 | 1.1 / 2.7 | 21.6 | 89.2 |
| raw | 1.0 | 0.95 | 0.95 | 223 / 681 | 11.2 | 16.000 | 1.1 / 2.7 | 23.9 | 83.4 |
| raw | 2.0 | 1.81 | 1.81 | 253 / 696 | 11.3 | 16.000 | 1.1 / 2.7 | 24.4 | 81.2 |
| raw | inf | 4.21 | 0.00 | 5087 / 6034 | 12.6 | 16.000 | 1.2 / 2.7 | 89.0 | 33.6 |
| nvcomp-ans | 0.5 | 0.49 | 0.49 | 248 / 590 | 11.2 | 13.363 | 20.6 / 11.2 | 17.0 | 90.8 |
| nvcomp-ans | 1.0 | 0.95 | 0.95 | 249 / 700 | 11.2 | 13.363 | 20.2 / 10.9 | 18.2 | 86.6 |
| nvcomp-ans | 2.0 | 1.81 | 1.81 | 281 / 720 | 11.3 | 13.363 | 20.2 / 10.9 | 18.8 | 84.3 |
| nvcomp-ans | inf | 4.37 | 0.00 | 5214 / 5689 | 13.1 | 13.363 | 17.7 / 10.6 | 26.9 | 50.2 |
| nvcomp-zstd | 0.5 | 0.49 | 0.49 | 382 / 1001 | 11.2 | 12.892 | 141.5 / 27.6 | 24.1 | 88.1 |
| nvcomp-zstd | 1.0 | 0.95 | 0.95 | 382 / 1150 | 11.3 | 12.892 | 141.5 / 26.8 | 37.0 | 79.4 |
| nvcomp-zstd | 2.0 | 1.79 | 1.79 | 422 / 1223 | 11.4 | 12.892 | 141.9 / 26.9 | 40.9 | 76.4 |
| nvcomp-zstd | inf | 2.88 | 0.00 | 7465 / 9633 | 12.0 | 12.892 | 141.3 / 27.3 | 158.2 | 17.6 |
| nvcomp-bitcomp | 0.5 | 0.49 | 0.49 | 247 / 585 | 11.2 | 13.786 | 17.9 / 11.8 | 18.1 | 89.6 |
| nvcomp-bitcomp | 1.0 | 0.95 | 0.95 | 248 / 700 | 11.2 | 13.786 | 17.6 / 11.7 | 18.6 | 87.4 |
| nvcomp-bitcomp | 2.0 | 1.81 | 1.81 | 274 / 719 | 11.3 | 13.786 | 17.4 / 11.6 | 19.7 | 83.5 |
| nvcomp-bitcomp | inf | 4.41 | 0.14 | 4558 / 5610 | 13.4 | 13.786 | 15.5 / 11.5 | 35.7 | 44.6 |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 512 | 1 | 128 | 8 | sz1 | raw | 128 / 129 | 0.66 | 16.000 | 0 |
| 10 | 1024 | 1 | 128 | 8 | sz1/sz1024 | raw | 213 / 215 | 0.62 | 16.000 | 0 |
| 10 | 128 | 16 | 128 | 32 | sz16 | raw | 207 / 408 | 9.49 | 16.000 | 0 |
| 10 | 1024 | 2 | 128 | 8 | sz1024 | raw | 371 / 382 | 1.13 | 16.000 | 0 |
| 10 | 2048 | 1 | 128 | 8 | sz1 | raw | 395 / 397 | 0.55 | 16.000 | 0 |
| 10 | 256 | 16 | 128 | 32 | sz16 | raw | 309 / 746 | 8.44 | 16.000 | 0 |
| 10 | 1024 | 4 | 128 | 8 | sz1024 | raw | 613 / 737 | 1.96 | 16.000 | 0 |
| 10 | 4096 | 1 | 128 | 8 | sz1 | raw | 766 / 769 | 0.45 | 16.000 | 0 |
| 10 | 512 | 16 | 128 | 32 | sz16 | raw | 534 / 1409 | 6.99 | 16.000 | 0 |
| 10 | 1024 | 8 | 128 | 16 | sz1024 | raw | 805 / 1419 | 3.28 | 16.000 | 0 |
| 10 | 8192 | 1 | 128 | 8 | sz1 | raw | 1581 / 1585 | 0.32 | 16.000 | 0 |
| 10 | 1024 | 16 | 128 | 32 | sz16/sz1024 | raw | 1051 / 2734 | 5.13 | 16.000 | 0 |
| 10 | 16384 | 1 | 128 | 4 | sz1 | raw | 4504 / 4629 | 0.16 | 16.000 | 0 |
| 10 | 1024 | 32 | 128 | 64 | sz1024 | raw | 2195 / 5553 | 6.44 | 16.000 | 0 |
| 10 | 2048 | 16 | 128 | 32 | sz16 | raw | 2193 / 5330 | 3.21 | 16.000 | 0 |
| 10 | 1920 | 20 | 128 | 40 | ibp | raw | 2660 / 6499 | 3.54 | 16.000 | 0 |
| 10 | 1024 | 64 | 128 | 128 | sz1024 | raw | 4667 / 9261 | 7.63 | 16.000 | 0 |
| 10 | 4096 | 16 | 128 | 32 | sz16 | raw | 5792 / 11083 | 1.72 | 16.000 | 0 |
| 10 | 4096 | 16 | 256 | 32 | exactkv | raw | 3903 / 10034 | 1.67 | 16.000 | 0 |
| 10 | 512 | 1 | 128 | 8 | sz1 | nvcomp-ans | 129 / 130 | 0.66 | 13.349 | 0 |
| 10 | 1024 | 1 | 128 | 8 | sz1/sz1024 | nvcomp-ans | 213 / 218 | 0.62 | 13.359 | 0 |
| 10 | 128 | 16 | 128 | 32 | sz16 | nvcomp-ans | 243 / 374 | 9.47 | 13.352 | 0 |
| 10 | 1024 | 2 | 128 | 8 | sz1024 | nvcomp-ans | 344 / 364 | 1.15 | 13.359 | 0 |
| 10 | 2048 | 1 | 128 | 8 | sz1 | nvcomp-ans | 391 / 394 | 0.55 | 13.361 | 0 |
| 10 | 256 | 16 | 128 | 32 | sz16 | nvcomp-ans | 330 / 686 | 8.48 | 13.351 | 0 |
| 10 | 1024 | 4 | 128 | 8 | sz1024 | nvcomp-ans | 553 / 697 | 2.00 | 13.359 | 0 |
| 10 | 4096 | 1 | 128 | 8 | sz1 | nvcomp-ans | 755 / 757 | 0.45 | 13.365 | 0 |
| 10 | 512 | 16 | 128 | 32 | sz16 | nvcomp-ans | 624 / 1297 | 6.89 | 13.353 | 0 |
| 10 | 1024 | 8 | 128 | 16 | sz1024 | nvcomp-ans | 756 / 1311 | 3.38 | 13.358 | 0 |
| 10 | 8192 | 1 | 128 | 8 | sz1 | nvcomp-ans | 1534 / 1538 | 0.32 | 13.373 | 0 |
| 10 | 1024 | 16 | 128 | 32 | sz16/sz1024 | nvcomp-ans | 1031 / 2512 | 5.23 | 13.358 | 0 |
| 10 | 16384 | 1 | 128 | 4 | sz1 | nvcomp-ans | 4115 / 4245 | 0.17 | 13.379 | 0 |
| 10 | 1024 | 32 | 128 | 64 | sz1024 | nvcomp-ans | 1895 / 5050 | 7.09 | 13.358 | 0 |
| 10 | 2048 | 16 | 128 | 32 | sz16 | nvcomp-ans | 1864 / 4946 | 3.44 | 13.363 | 0 |
| 10 | 1920 | 20 | 128 | 40 | ibp | nvcomp-ans | 2307 / 5895 | 3.83 | 13.363 | 0 |
| 10 | 1024 | 64 | 128 | 128 | sz1024 | nvcomp-ans | 4519 / 9137 | 7.99 | 13.358 | 0 |
| 10 | 4096 | 16 | 128 | 32 | sz16 | nvcomp-ans | 5569 / 10471 | 1.84 | 13.368 | 0 |
| 10 | 4096 | 16 | 256 | 32 | exactkv | nvcomp-ans | 3849 / 9057 | 1.67 | 13.368 | 0 |
| 10 | 512 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 163 / 166 | 0.65 | 12.891 | 0 |
| 10 | 1024 | 1 | 128 | 8 | sz1/sz1024 | nvcomp-zstd | 278 / 279 | 0.60 | 12.894 | 0 |
| 10 | 128 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 276 / 399 | 9.35 | 12.882 | 0 |
| 10 | 1024 | 2 | 128 | 8 | sz1024 | nvcomp-zstd | 427 / 468 | 1.09 | 12.894 | 0 |
| 10 | 2048 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 522 / 523 | 0.51 | 12.898 | 0 |
| 10 | 256 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 443 / 697 | 8.12 | 12.882 | 0 |
| 10 | 1024 | 4 | 128 | 8 | sz1024 | nvcomp-zstd | 669 / 821 | 1.86 | 12.894 | 0 |
| 10 | 4096 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 1018 / 1021 | 0.40 | 12.898 | 0 |
| 10 | 512 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 806 / 1362 | 6.38 | 12.887 | 0 |
| 10 | 1024 | 8 | 128 | 16 | sz1024 | nvcomp-zstd | 976 / 1427 | 3.05 | 12.891 | 0 |
| 10 | 8192 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 2076 / 2090 | 0.28 | 12.900 | 0 |
| 10 | 1024 | 16 | 128 | 32 | sz16/sz1024 | nvcomp-zstd | 1587 / 2630 | 4.49 | 12.888 | 0 |
| 10 | 16384 | 1 | 128 | 4 | sz1 | nvcomp-zstd | 5165 / 5248 | 0.14 | 12.906 | 0 |
| 10 | 1024 | 32 | 128 | 64 | sz1024 | nvcomp-zstd | 3096 / 5164 | 5.68 | 12.889 | 0 |
| 10 | 2048 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 3200 / 5586 | 2.74 | 12.892 | 0 |
| 10 | 1920 | 20 | 128 | 40 | ibp | nvcomp-zstd | 3939 / 6677 | 2.95 | 12.894 | 0 |
| 10 | 1024 | 64 | 128 | 128 | sz1024 | nvcomp-zstd | 7103 / 12286 | 6.12 | 12.887 | 0 |
| 10 | 4096 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 8872 / 14867 | 1.31 | 12.897 | 0 |
| 10 | 4096 | 16 | 256 | 32 | exactkv | nvcomp-zstd | 7766 / 14618 | 1.23 | 12.897 | 0 |
| 10 | 512 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 130 / 132 | 0.66 | 13.755 | 0 |
| 10 | 1024 | 1 | 128 | 8 | sz1/sz1024 | nvcomp-bitcomp | 215 / 216 | 0.62 | 13.781 | 0 |
| 10 | 128 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 221 / 387 | 9.51 | 13.721 | 0 |
| 10 | 1024 | 2 | 128 | 8 | sz1024 | nvcomp-bitcomp | 370 / 373 | 1.13 | 13.781 | 0 |
| 10 | 2048 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 395 / 396 | 0.55 | 13.783 | 0 |
| 10 | 256 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 342 / 688 | 8.42 | 13.745 | 0 |
| 10 | 1024 | 4 | 128 | 8 | sz1024 | nvcomp-bitcomp | 602 / 703 | 1.98 | 13.782 | 0 |
| 10 | 4096 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 760 / 770 | 0.45 | 13.791 | 0 |
| 10 | 512 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 627 / 1317 | 6.83 | 13.762 | 0 |
| 10 | 1024 | 8 | 128 | 16 | sz1024 | nvcomp-bitcomp | 808 / 1342 | 3.32 | 13.781 | 0 |
| 10 | 8192 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 1556 / 1561 | 0.32 | 13.800 | 0 |
| 10 | 1024 | 16 | 128 | 32 | sz16/sz1024 | nvcomp-bitcomp | 1102 / 2576 | 5.16 | 13.778 | 0 |
| 10 | 16384 | 1 | 128 | 4 | sz1 | nvcomp-bitcomp | 4288 / 4408 | 0.17 | 13.804 | 0 |
| 10 | 1024 | 32 | 128 | 64 | sz1024 | nvcomp-bitcomp | 1981 / 5157 | 7.00 | 13.776 | 0 |
| 10 | 2048 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 1957 / 5049 | 3.39 | 13.786 | 0 |
| 10 | 1920 | 20 | 128 | 40 | ibp | nvcomp-bitcomp | 2357 / 5970 | 3.84 | 13.787 | 0 |
| 10 | 1024 | 64 | 128 | 128 | sz1024 | nvcomp-bitcomp | 4454 / 8840 | 8.07 | 13.776 | 0 |
| 10 | 4096 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 5792 / 9672 | 1.77 | 13.795 | 0 |
| 10 | 4096 | 16 | 256 | 32 | exactkv | nvcomp-bitcomp | 3679 / 9250 | 1.67 | 13.795 | 0 |
| 100 | 512 | 1 | 128 | 8 | sz1 | raw | 86 / 86 | 0.68 | 16.000 | 0 |
| 100 | 1024 | 1 | 128 | 8 | sz1/sz1024 | raw | 129 / 130 | 0.65 | 16.000 | 0 |
| 100 | 128 | 16 | 128 | 32 | sz16 | raw | 230 / 246 | 9.89 | 16.000 | 0 |
| 100 | 1024 | 2 | 128 | 8 | sz1024 | raw | 211 / 215 | 1.24 | 16.000 | 0 |
| 100 | 2048 | 1 | 128 | 8 | sz1 | raw | 226 / 229 | 0.61 | 16.000 | 0 |
| 100 | 256 | 16 | 128 | 32 | sz16 | raw | 366 / 396 | 9.02 | 16.000 | 0 |
| 100 | 1024 | 4 | 128 | 8 | sz1024 | raw | 378 / 412 | 2.23 | 16.000 | 0 |
| 100 | 4096 | 1 | 128 | 8 | sz1 | raw | 432 / 434 | 0.53 | 16.000 | 0 |
| 100 | 512 | 16 | 128 | 32 | sz16 | raw | 675 / 750 | 7.50 | 16.000 | 0 |
| 100 | 1024 | 8 | 128 | 16 | sz1024 | raw | 655 / 743 | 3.76 | 16.000 | 0 |
| 100 | 8192 | 1 | 128 | 8 | sz1 | raw | 903 / 907 | 0.41 | 16.000 | 0 |
| 100 | 1024 | 16 | 128 | 32 | sz16/sz1024 | raw | 1250 / 1497 | 5.65 | 16.000 | 0 |
| 100 | 16384 | 1 | 128 | 4 | sz1 | raw | 3743 / 3943 | 0.18 | 16.000 | 0 |
| 100 | 1024 | 32 | 128 | 64 | sz1024 | raw | 2066 / 3298 | 7.60 | 16.000 | 0 |
| 100 | 2048 | 16 | 128 | 32 | sz16 | raw | 2367 / 3289 | 3.69 | 16.000 | 0 |
| 100 | 1920 | 20 | 128 | 40 | ibp | raw | 2812 / 3893 | 4.10 | 16.000 | 0 |
| 100 | 1024 | 64 | 128 | 128 | sz1024 | raw | 3663 / 7060 | 10.34 | 16.000 | 0 |
| 100 | 4096 | 16 | 128 | 32 | sz16 | raw | 4729 / 9238 | 2.14 | 16.000 | 0 |
| 100 | 4096 | 16 | 256 | 32 | exactkv | raw | 3237 / 7162 | 1.90 | 16.000 | 0 |
| 100 | 512 | 1 | 128 | 8 | sz1 | nvcomp-ans | 93 / 96 | 0.68 | 13.349 | 0 |
| 100 | 1024 | 1 | 128 | 8 | sz1/sz1024 | nvcomp-ans | 142 / 150 | 0.65 | 13.359 | 0 |
| 100 | 128 | 16 | 128 | 32 | sz16 | nvcomp-ans | 232 / 253 | 9.86 | 13.353 | 0 |
| 100 | 1024 | 2 | 128 | 8 | sz1024 | nvcomp-ans | 218 / 229 | 1.23 | 13.359 | 0 |
| 100 | 2048 | 1 | 128 | 8 | sz1 | nvcomp-ans | 248 / 252 | 0.60 | 13.361 | 0 |
| 100 | 256 | 16 | 128 | 32 | sz16 | nvcomp-ans | 387 / 422 | 8.91 | 13.351 | 0 |
| 100 | 1024 | 4 | 128 | 8 | sz1024 | nvcomp-ans | 381 / 428 | 2.21 | 13.359 | 0 |
| 100 | 4096 | 1 | 128 | 8 | sz1 | nvcomp-ans | 469 / 472 | 0.52 | 13.365 | 0 |
| 100 | 512 | 16 | 128 | 32 | sz16 | nvcomp-ans | 686 / 759 | 7.43 | 13.353 | 0 |
| 100 | 1024 | 8 | 128 | 16 | sz1024 | nvcomp-ans | 680 / 779 | 3.72 | 13.358 | 0 |
| 100 | 8192 | 1 | 128 | 8 | sz1 | nvcomp-ans | 967 / 971 | 0.40 | 13.373 | 0 |
| 100 | 1024 | 16 | 128 | 32 | sz16/sz1024 | nvcomp-ans | 1286 / 1468 | 5.57 | 13.358 | 0 |
| 100 | 16384 | 1 | 128 | 4 | sz1 | nvcomp-ans | 2982 / 3069 | 0.21 | 13.379 | 0 |
| 100 | 1024 | 32 | 128 | 64 | sz1024 | nvcomp-ans | 2466 / 2841 | 7.58 | 13.358 | 0 |
| 100 | 2048 | 16 | 128 | 32 | sz16 | nvcomp-ans | 2464 / 2912 | 3.72 | 13.363 | 0 |
| 100 | 1920 | 20 | 128 | 40 | ibp | nvcomp-ans | 2902 / 3381 | 4.21 | 13.363 | 0 |
| 100 | 1024 | 64 | 128 | 128 | sz1024 | nvcomp-ans | 3882 / 7903 | 10.01 | 13.358 | 0 |
| 100 | 4096 | 16 | 128 | 32 | sz16 | nvcomp-ans | 4785 / 9394 | 2.19 | 13.368 | 0 |
| 100 | 4096 | 16 | 256 | 32 | exactkv | nvcomp-ans | 4219 / 7200 | 1.81 | 13.368 | 0 |
| 100 | 512 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 130 / 132 | 0.66 | 12.891 | 0 |
| 100 | 1024 | 1 | 128 | 8 | sz1/sz1024 | nvcomp-zstd | 208 / 210 | 0.62 | 12.894 | 0 |
| 100 | 128 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 252 / 389 | 9.33 | 12.882 | 0 |
| 100 | 1024 | 2 | 128 | 8 | sz1024 | nvcomp-zstd | 349 / 360 | 1.14 | 12.894 | 0 |
| 100 | 2048 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 385 / 386 | 0.55 | 12.898 | 0 |
| 100 | 256 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 419 / 647 | 8.27 | 12.882 | 0 |
| 100 | 1024 | 4 | 128 | 8 | sz1024 | nvcomp-zstd | 513 / 683 | 2.00 | 12.894 | 0 |
| 100 | 4096 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 745 / 746 | 0.45 | 12.898 | 0 |
| 100 | 512 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 786 / 1304 | 6.47 | 12.887 | 0 |
| 100 | 1024 | 8 | 128 | 16 | sz1024 | nvcomp-zstd | 835 / 1276 | 3.23 | 12.891 | 0 |
| 100 | 8192 | 1 | 128 | 8 | sz1 | nvcomp-zstd | 1525 / 1530 | 0.33 | 12.900 | 0 |
| 100 | 1024 | 16 | 128 | 32 | sz16/sz1024 | nvcomp-zstd | 1444 / 2482 | 4.68 | 12.888 | 0 |
| 100 | 16384 | 1 | 128 | 4 | sz1 | nvcomp-zstd | 4073 / 4148 | 0.17 | 12.906 | 0 |
| 100 | 1024 | 32 | 128 | 64 | sz1024 | nvcomp-zstd | 3176 / 4926 | 5.77 | 12.889 | 0 |
| 100 | 2048 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 3193 / 5072 | 2.86 | 12.892 | 0 |
| 100 | 1920 | 20 | 128 | 40 | ibp | nvcomp-zstd | 3938 / 5853 | 3.13 | 12.894 | 0 |
| 100 | 1024 | 64 | 128 | 128 | sz1024 | nvcomp-zstd | 7002 / 11996 | 6.21 | 12.887 | 0 |
| 100 | 4096 | 16 | 128 | 32 | sz16 | nvcomp-zstd | 7970 / 13854 | 1.42 | 12.897 | 0 |
| 100 | 4096 | 16 | 256 | 32 | exactkv | nvcomp-zstd | 6936 / 13839 | 1.32 | 12.897 | 0 |
| 100 | 512 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 93 / 94 | 0.68 | 13.755 | 0 |
| 100 | 1024 | 1 | 128 | 8 | sz1/sz1024 | nvcomp-bitcomp | 142 / 143 | 0.65 | 13.781 | 0 |
| 100 | 128 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 238 / 258 | 9.82 | 13.721 | 0 |
| 100 | 1024 | 2 | 128 | 8 | sz1024 | nvcomp-bitcomp | 219 / 229 | 1.24 | 13.781 | 0 |
| 100 | 2048 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 247 / 248 | 0.60 | 13.783 | 0 |
| 100 | 256 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 363 / 414 | 8.93 | 13.746 | 0 |
| 100 | 1024 | 4 | 128 | 8 | sz1024 | nvcomp-bitcomp | 377 / 426 | 2.22 | 13.782 | 0 |
| 100 | 4096 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 467 / 469 | 0.52 | 13.791 | 0 |
| 100 | 512 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 685 / 767 | 7.43 | 13.763 | 0 |
| 100 | 1024 | 8 | 128 | 16 | sz1024 | nvcomp-bitcomp | 671 / 773 | 3.72 | 13.781 | 0 |
| 100 | 8192 | 1 | 128 | 8 | sz1 | nvcomp-bitcomp | 965 / 970 | 0.40 | 13.800 | 0 |
| 100 | 1024 | 16 | 128 | 32 | sz16/sz1024 | nvcomp-bitcomp | 1300 / 1453 | 5.56 | 13.778 | 0 |
| 100 | 16384 | 1 | 128 | 4 | sz1 | nvcomp-bitcomp | 3611 / 3803 | 0.19 | 13.804 | 0 |
| 100 | 1024 | 32 | 128 | 64 | sz1024 | nvcomp-bitcomp | 2395 / 2839 | 7.61 | 13.776 | 0 |
| 100 | 2048 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 2432 / 2858 | 3.76 | 13.786 | 0 |
| 100 | 1920 | 20 | 128 | 40 | ibp | nvcomp-bitcomp | 2891 / 3356 | 4.21 | 13.787 | 0 |
| 100 | 1024 | 64 | 128 | 128 | sz1024 | nvcomp-bitcomp | 3817 / 7256 | 10.19 | 13.776 | 0 |
| 100 | 4096 | 16 | 128 | 32 | sz16 | nvcomp-bitcomp | 5223 / 10133 | 2.08 | 13.795 | 0 |
| 100 | 4096 | 16 | 256 | 32 | exactkv | nvcomp-bitcomp | 3909 / 7029 | 1.82 | 13.795 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 19:36:11 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:11 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:11 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:36:11 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:11 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:11 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:36:11 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:11 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:11 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:36:11 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:11 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:11 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

