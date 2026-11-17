# PD matrix workloads on gpu-grad-01, job 30334 (ended without a final status (Slurm: FAILED)), reviewed from the kept results

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 1 runs: `20261117_0859_1p3d_tkv_ring_salvaged`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | raw | 4/4 | 1/4 | yes | 16.000 |
| 1p3d, host memory | splitzip | 4/4 | 2/4 | yes | 12.044 |
| 1p3d, host memory | exans | 4/4 | 2/4 | yes | 10.924 |
| 1p3d, host memory | adacode | 4/4 | 1/4 | yes | 11.372 |
| 1p3d, host memory | exactkv | 4/4 | 2/4 | yes | 12.629 |
| 1p3d, host memory | ours-hbh | 4/4 | 2/4 | yes | 9.042 |
| 1p3d, host memory | t2-hbh | 4/4 | 1/4 | yes | 8.516 |
| 1p3d, host memory | t2-rank256 | 4/4 | 2/4 | yes | 9.597 |
| 1p3d, host memory | aquakv-2 (lossy) | lossy (KV error 10.46%) | 0/4 | yes | 2.126 |
| 1p3d, host memory | aquakv-3 (lossy) | lossy (KV error 5.57%) | 2/4 | yes | 3.088 |

## 1p3d, host memory: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate 0.5 | rate 1.0 | rate 2.0 | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 16.000 | 0.49 (0.49); 393 / 1033 | 0.95 (0.95); 393 / 1253 | 1.79 (1.79); 426 / 1418 | 3.03 (0.00); 6318 / 9080 | 11.8 |
| splitzip | 12.041 | 0.49 (0.49); 359 / 927 | 0.95 (0.95); 359 / 1100 | 1.80 (1.80); 388 / 1246 | 3.47 (0.00); 5649 / 7746 | 12.1 |
| exans | 10.883 | 0.49 (0.49); 393 / 906 | 0.95 (0.95); 393 / 1103 | 1.79 (1.79); 421 / 1200 | 3.45 (0.00); 5914 / 7784 | 12.2 |
| adacode | 11.372 | 0.49 (0.49); 378 / 883 | 0.95 (0.95); 380 / 1064 | 1.79 (1.79); 407 / 1199 | 3.47 (0.00); 5826 / 7726 | 12.2 |
| exactkv | 12.591 | 0.49 (0.49); 391 / 948 | 0.95 (0.95); 391 / 1129 | 1.79 (1.79); 421 / 1271 | 3.38 (0.00); 5789 / 7978 | 12.1 |
| ours-hbh | 8.671 | 0.49 (0.49); 500 / 941 | 0.94 (0.94); 500 / 1108 | 1.78 (1.78); 517 / 1203 | 3.55 (0.00); 5552 / 7392 | 18.5 |
| t2-hbh | 8.146 | 0.49 (0.49); 501 / 948 | 0.94 (0.94); 504 / 1085 | 1.78 (1.78); 522 / 1190 | 3.51 (0.00); 5486 / 7508 | 18.3 |
| t2-rank256 | 9.229 | 0.49 (0.49); 369 / 810 | 0.95 (0.95); 370 / 998 | 1.79 (1.79); 396 / 1097 | 3.66 (0.00); 5592 / 7234 | 12.9 |
| aquakv-2 (lossy) | 2.130 | 0.49 (0.49); 413 / 1002 | 0.95 (0.95); 412 / 1145 | 1.79 (1.79); 457 / 1222 | 2.83 (0.00); 6898 / 9772 | 12.3 |
| aquakv-3 (lossy) | 3.092 | 0.49 (0.49); 516 / 1330 | 0.94 (0.94); 516 / 1482 | 1.75 (1.75); 772 / 1713 | 2.28 (0.07); 7842 / 12495 | 12.1 |

Best by goodput, then TTFT p50: rate 0.5: splitzip (goodput 0.49/s, TTFT p50 359 ms); rate 1.0: splitzip (goodput 0.95/s, TTFT p50 359 ms); rate 2.0: splitzip (goodput 1.80/s, TTFT p50 388 ms); rate inf: aquakv-3 (lossy) (goodput 0.07/s, TTFT p50 7842 ms).

## 1p3d, host memory: link 100 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate 0.5 | rate 1.0 | rate 2.0 | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 16.000 | 0.49 (0.49); 224 / 541 | 0.95 (0.95); 225 / 683 | 1.81 (1.81); 252 / 696 | 4.13 (0.13); 4938 / 6160 | 12.6 |
| splitzip | 12.041 | 0.49 (0.49); 230 / 537 | 0.95 (0.95); 231 / 667 | 1.81 (1.81); 259 / 679 | 4.55 (0.00); 4928 / 5396 | 13.0 |
| exans | 10.883 | 0.49 (0.49); 278 / 691 | 0.95 (0.95); 280 / 777 | 1.80 (1.80); 307 / 833 | 3.85 (0.00); 5649 / 6738 | 12.5 |
| adacode | 11.372 | 0.49 (0.49); 255 / 561 | 0.95 (0.95); 255 / 680 | 1.81 (1.81); 280 / 703 | 4.41 (0.00); 4845 / 5639 | 13.0 |
| exactkv | 12.591 | 0.49 (0.49); 255 / 569 | 0.95 (0.95); 256 / 684 | 1.81 (1.81); 278 / 710 | 4.37 (0.00); 5177 / 5695 | 13.0 |
| ours-hbh | 8.671 | 0.49 (0.49); 407 / 755 | 0.95 (0.95); 412 / 961 | 1.79 (1.79); 423 / 982 | 3.93 (0.12); 5067 / 6429 | 17.7 |
| t2-hbh | 8.146 | 0.49 (0.49); 415 / 782 | 0.95 (0.95); 419 / 964 | 1.79 (1.79); 434 / 995 | 3.78 (0.00); 5609 / 6806 | 17.6 |
| t2-rank256 | 9.229 | 0.49 (0.49); 272 / 619 | 0.95 (0.95); 272 / 723 | 1.80 (1.80); 294 / 766 | 4.05 (0.13); 5104 / 6319 | 13.1 |
| aquakv-2 (lossy) | 2.130 | 0.49 (0.49); 391 / 980 | 0.95 (0.95); 392 / 1119 | 1.79 (1.79); 434 / 1199 | 2.83 (0.00); 7198 / 9772 | 12.1 |
| aquakv-3 (lossy) | 3.092 | 0.49 (0.49); 483 / 1305 | 0.94 (0.94); 483 / 1451 | 1.75 (1.75); 741 / 1682 | 2.28 (0.00); 8466 / 12514 | 11.8 |

Best by goodput, then TTFT p50: rate 0.5: raw (goodput 0.49/s, TTFT p50 224 ms); rate 1.0: raw (goodput 0.95/s, TTFT p50 225 ms); rate 2.0: raw (goodput 1.81/s, TTFT p50 252 ms); rate inf: raw (goodput 0.13/s, TTFT p50 4938 ms).

## Published workloads: 1p3d, host memory, link 10 Gb/s

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's).

| Workload | Prompt tokens x at once, out | raw | splitzip | exans | adacode | exactkv |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | **128** | 129 (1.00x) | 134 (0.96x) | 153 (0.84x) | 152 (0.85x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | 214 | **203** (1.05x) | 219 (0.98x) | 226 (0.95x) | 229 (0.93x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | 395 | **358** (1.10x) | 392 (1.01x) | 377 (1.05x) | - |
| SplitZip, 1 at a time | 4096 x 1, 128 | 770 | **688** (1.12x) | 761 (1.01x) | 691 (1.11x) | - |
| SplitZip, 1 at a time | 8192 x 1, 128 | 1582 | 1413 (1.12x) | 1569 (1.01x) | **1391** (1.14x) | - |
| SplitZip, 1 at a time | 16384 x 1, 128 | 4461 | 3821 (1.17x) | 4081 (1.09x) | **3766** (1.18x) | - |
| SplitZip, 16 at once | 128 x 16, 128 | **208** | 333 (0.62x) | 266 (0.78x) | 269 (0.77x) | 337 (0.62x) (failed) |
| SplitZip, 16 at once | 256 x 16, 128 | **294** | 346 (0.85x) | 404 (0.73x) | 422 (0.70x) | - |
| SplitZip, 16 at once | 512 x 16, 128 | **545** | 676 (0.81x) | 685 (0.80x) | 701 (0.78x) | - |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | **1047** | 1119 (0.94x) | 1236 (0.85x) | 1249 (0.84x) | - |
| SplitZip, 16 at once | 2048 x 16, 128 | 2208 | **1725** (1.28x) | 2177 (1.01x) | 1859 (1.19x) | - |
| SplitZip, 16 at once | 4096 x 16, 128 | 5906 | 5225 (1.13x) | 5613 (1.05x) | **4659** (1.27x) | - |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | 372 | 332 (1.12x) | **322** (1.16x) | 336 (1.11x) | - |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | 613 | 544 (1.13x) | **466** (1.32x) | 507 (1.21x) | - |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | 811 | 751 (1.08x) | 744 (1.09x) | **737** (1.10x) | - |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | 2204 | **1764** (1.25x) | 2107 (1.05x) | 1846 (1.19x) | - |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | 4663 | **3851** (1.21x) | 4902 (0.95x) | 4279 (1.09x) | - |
| IBP | 1920 x 20, 128 | 2837 | 2260 (1.26x) | 2441 (1.16x) | **2253** (1.26x) | - |
| exactkv | 4096 x 16, 256 | 4297 | **3033** (1.42x) | 4147 (1.04x) | 3165 (1.36x) | - |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | splitzip | exans | adacode | exactkv |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | **0.66** | 0.65 (0.98x) | 0.66 (1.00x) | 0.65 (0.99x) | 0.65 (0.98x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | **0.62** | 0.61 (0.99x) | 0.62 (1.00x) | 0.62 (0.99x) | 0.62 (0.99x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | 0.55 | **0.56** (1.02x) | 0.55 (1.00x) | 0.56 (1.01x) | - |
| SplitZip, 1 at a time | 4096 x 1, 128 | 0.45 | 0.46 (1.03x) | 0.45 (1.00x) | **0.46** (1.04x) | - |
| SplitZip, 1 at a time | 8192 x 1, 128 | 0.32 | 0.34 (1.06x) | 0.32 (1.00x) | **0.34** (1.07x) | - |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.16 | 0.18 (1.12x) | 0.17 (1.07x) | **0.18** (1.13x) | - |
| SplitZip, 16 at once | 128 x 16, 128 | 9.50 | 8.96 (0.94x) | 9.52 (1.00x) | 9.09 (0.96x) | **13.44** (1.42x) (failed) |
| SplitZip, 16 at once | 256 x 16, 128 | **8.50** | 8.48 (1.00x) | 8.34 (0.98x) | 8.24 (0.97x) | - |
| SplitZip, 16 at once | 512 x 16, 128 | **6.98** | 6.82 (0.98x) | 6.82 (0.98x) | 6.75 (0.97x) | - |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | 5.13 | **5.21** (1.02x) | 5.04 (0.98x) | 5.05 (0.99x) | - |
| SplitZip, 16 at once | 2048 x 16, 128 | 3.26 | **3.49** (1.07x) | 3.30 (1.01x) | 3.44 (1.06x) | - |
| SplitZip, 16 at once | 4096 x 16, 128 | 1.74 | 1.93 (1.11x) | 1.85 (1.07x) | **2.06** (1.18x) | - |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | 1.13 | 1.16 (1.02x) | **1.16** (1.02x) | 1.15 (1.02x) | - |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | 1.96 | 2.03 (1.03x) | 2.04 (1.04x) | **2.04** (1.04x) | - |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | 3.28 | 3.36 (1.03x) | 3.38 (1.03x) | **3.41** (1.04x) | - |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | 6.44 | **7.09** (1.10x) | 6.64 (1.03x) | 6.91 (1.07x) | - |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | 7.72 | **8.68** (1.12x) | 7.99 (1.03x) | 8.40 (1.09x) | - |
| IBP | 1920 x 20, 128 | 3.55 | **3.92** (1.10x) | 3.71 (1.05x) | 3.82 (1.08x) | - |
| exactkv | 4096 x 16, 256 | 1.63 | **1.80** (1.10x) | 1.70 (1.04x) | 1.79 (1.10x) | - |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | t2-hbh | 10 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 10 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 10 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 10 | inf | 0 | 32 | {'exam': 6, 'math': 6, 'wikitext': 7, 'news': 6, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | inf | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | inf | 0 | 32 | {'math': 6, 'exam': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | inf | 0 | 32 | {'exam': 6, 'math': 6, 'wikitext': 7, 'news': 6, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |

## Note on the lossy reference

aquakv-<b> is AQUA-KV (arXiv 2501.19392, the authors' algorithm on the GPU), lossy: the KV it delivers differs from the original (its error is in the correctness table). It is here for speed and size only, beside the bit-exact codecs.

