# PD matrix wlours on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 2 runs: `llama32_3b_pd_1p3d_tkv_p2p_rtxa5000x4_20261118_1834`, `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261118_1635`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, GPU to GPU | raw | 4/4 | 3/4 | yes | 16.000 |
| 1p3d, GPU to GPU | ours-hbh | 4/4 | 3/4 | yes | 9.042 |
| 1p3d, GPU to GPU | t2-hbh | 4/4 | 3/4 | yes | 8.516 |
| 1p3d, GPU to GPU | t2-rank256 | 4/4 | 3/4 | yes | 9.597 |
| 1p3d, GPU to GPU | splitzip | 4/4 | 3/4 | yes | 12.044 |
| 1p3d, GPU to GPU | adacode | 4/4 | 3/4 | yes | 11.372 |
| 1p3d, host memory | raw | 4/4 | 1/4 | yes | 16.000 |
| 1p3d, host memory | ours-hbh | 4/4 | 3/4 | yes | 9.042 |
| 1p3d, host memory | t2-hbh | 4/4 | 3/4 | yes | 8.516 |
| 1p3d, host memory | t2-rank256 | 4/4 | 1/4 | yes | 9.597 |
| 1p3d, host memory | splitzip | 4/4 | 3/4 | yes | 12.044 |
| 1p3d, host memory | adacode | 4/4 | 3/4 | yes | 11.372 |

## 1p3d, GPU to GPU: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate 0.5 | rate 1.0 | rate 2.0 | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 16.000 | 0.49 (0.49); 215 / 526 | 0.95 (0.95); 215 / 664 | 1.81 (1.81); 248 / 669 | 4.40 (0.00); 4671 / 5666 | 13.0 |
| ours-hbh | 8.671 | 0.49 (0.49); 304 / 652 | 0.95 (0.95); 303 / 753 | 1.80 (1.80); 323 / 792 | 4.12 (0.13); 5005 / 6161 | 14.3 |
| t2-hbh | 8.146 | 0.49 (0.49); 314 / 685 | 0.95 (0.95); 314 / 780 | 1.80 (1.80); 337 / 844 | 3.85 (0.12); 5325 / 6734 | 13.9 |
| t2-rank256 | 9.229 | 0.49 (0.49); 254 / 610 | 0.95 (0.95); 257 / 709 | 1.81 (1.81); 285 / 754 | 4.06 (0.00); 5212 / 6285 | 12.9 |
| splitzip | 12.041 | 0.49 (0.49); 223 / 525 | 0.95 (0.95); 224 / 661 | 1.81 (1.81); 251 / 667 | 4.61 (0.14); 4531 / 5297 | 13.1 |
| adacode | 11.372 | 0.49 (0.49); 247 / 552 | 0.95 (0.95); 249 / 672 | 1.81 (1.81); 272 / 691 | 4.42 (0.14); 4390 / 5589 | 13.3 |

Best by goodput, then TTFT p50: rate 0.5: raw (goodput 0.49/s, TTFT p50 215 ms); rate 1.0: raw (goodput 0.95/s, TTFT p50 215 ms); rate 2.0: splitzip (goodput 1.81/s, TTFT p50 251 ms); rate inf: splitzip (goodput 0.14/s, TTFT p50 4531 ms).

## 1p3d, host memory: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate 0.5 | rate 1.0 | rate 2.0 | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 16.000 | 0.49 (0.49); 394 / 1027 | 0.95 (0.95); 394 / 1251 | 1.79 (1.79); 424 / 1420 | 3.00 (0.00); 6438 / 9129 | 11.7 |
| ours-hbh | 8.671 | 0.49 (0.49); 402 / 843 | 0.95 (0.95); 401 / 1020 | 1.79 (1.79); 421 / 1107 | 3.60 (0.00); 5647 / 7301 | 14.3 |
| t2-hbh | 8.146 | 0.49 (0.49); 406 / 851 | 0.95 (0.95); 408 / 999 | 1.79 (1.79); 424 / 1063 | 3.62 (0.00); 5599 / 7295 | 13.9 |
| t2-rank256 | 9.229 | 0.49 (0.49); 361 / 799 | 0.95 (0.95); 361 / 993 | 1.80 (1.80); 390 / 1110 | 3.68 (0.00); 5582 / 7164 | 12.9 |
| splitzip | 12.041 | 0.49 (0.49); 356 / 923 | 0.95 (0.95); 356 / 1095 | 1.80 (1.80); 387 / 1242 | 3.43 (0.00); 5788 / 7797 | 12.1 |
| adacode | 11.372 | 0.49 (0.49); 375 / 877 | 0.95 (0.95); 376 / 1060 | 1.80 (1.80); 407 / 1193 | 3.52 (0.00); 5619 / 7590 | 12.2 |

Best by goodput, then TTFT p50: rate 0.5: splitzip (goodput 0.49/s, TTFT p50 356 ms); rate 1.0: t2-rank256 (goodput 0.95/s, TTFT p50 361 ms); rate 2.0: splitzip (goodput 1.80/s, TTFT p50 387 ms); rate inf: t2-rank256 (goodput 0.00/s, TTFT p50 5582 ms).

## 1p3d, host memory: link 100 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate 0.5 | rate 1.0 | rate 2.0 | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 16.000 | 0.49 (0.49); 222 / 546 | 0.95 (0.95); 224 / 684 | 1.81 (1.81); 251 / 695 | 4.21 (0.00); 5004 / 6017 | 12.6 |
| ours-hbh | 8.671 | 0.49 (0.49); 309 / 653 | 0.95 (0.95); 306 / 756 | 1.80 (1.80); 326 / 790 | 4.13 (0.00); 5137 / 6136 | 14.1 |
| t2-hbh | 8.146 | 0.49 (0.49); 316 / 680 | 0.95 (0.95); 319 / 784 | 1.80 (1.80); 335 / 844 | 3.86 (0.00); 5449 / 6695 | 13.9 |
| t2-rank256 | 9.229 | 0.49 (0.49); 262 / 609 | 0.95 (0.95); 261 / 712 | 1.81 (1.81); 290 / 758 | 4.05 (0.13); 4594 / 6305 | 13.3 |
| splitzip | 12.041 | 0.49 (0.49); 230 / 530 | 0.95 (0.95); 233 / 673 | 1.81 (1.81); 259 / 677 | 4.58 (0.14); 4581 / 5355 | 13.0 |
| adacode | 11.372 | 0.49 (0.49); 255 / 562 | 0.95 (0.95); 259 / 681 | 1.81 (1.81); 281 / 715 | 4.39 (0.14); 4534 / 5649 | 13.1 |

Best by goodput, then TTFT p50: rate 0.5: raw (goodput 0.49/s, TTFT p50 222 ms); rate 1.0: raw (goodput 0.95/s, TTFT p50 224 ms); rate 2.0: splitzip (goodput 1.81/s, TTFT p50 259 ms); rate inf: splitzip (goodput 0.14/s, TTFT p50 4581 ms).

## Published workloads: 1p3d, GPU to GPU, link unpaced (the machine's own path)

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer over the GPU budget while others were held), which costs copies the others did not pay.

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip | adacode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | **83** | 154 (0.54x) | 161 (0.51x) | 110 (0.75x) | 95 (0.87x) | 120 (0.69x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | **124** | 200 (0.62x) | 210 (0.59x) | 159 (0.78x) | 136 (0.91x) | 159 (0.78x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | **215** | 305 (0.70x) | 314 (0.69x) | 257 (0.84x) | 225 (0.96x) | 249 (0.86x) |
| SplitZip, 1 at a time | 4096 x 1, 128 | **410** | 531 (0.77x) | 543 (0.76x) | 477 (0.86x) | 419 (0.98x) | 441 (0.93x) |
| SplitZip, 1 at a time | 8192 x 1, 128 | **873** | 1112 (0.78x) | 1136 (0.77x) | 1008 (0.87x) | 877 (1.00x) | 896 (0.97x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 2099 | 2544 (0.83x) | 2608 (0.80x) | 2376 (0.88x) | **2059** (1.02x) | 2102 (1.00x) |
| SplitZip, 16 at once | 128 x 16, 128 | **226** | 430 (0.53x) | 420 (0.54x) | 269 (0.84x) | 248 (0.91x) | 299 (0.76x) |
| SplitZip, 16 at once | 256 x 16, 128 | **365** | 569 (0.64x) | 579 (0.63x) | 420 (0.87x) | 411 (0.89x) | 417 (0.88x) |
| SplitZip, 16 at once | 512 x 16, 128 | **659** | 888 (0.74x) | 867 (0.76x) | 691 (0.95x) | 689 (0.96x) | 700 (0.94x) |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | **1255** | 1454 (0.86x) | 1412 (0.89x) | 1326 (0.95x) | 1264 (0.99x) | 1293 (0.97x) |
| SplitZip, 16 at once | 2048 x 16, 128 | **2334** (host) | 2635 (0.89x) | 2654 (0.88x) | 2462 (0.95x) | 2402 (0.97x) | 2398 (0.97x) |
| SplitZip, 16 at once | 4096 x 16, 128 | 4559 (host) | 4745 (0.96x) | 4867 (0.94x) | 4660 (0.98x) | **3931** (1.16x) | 4091 (1.11x) |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | **200** | 274 (0.73x) | 289 (0.69x) | 232 (0.86x) | 209 (0.96x) | 220 (0.91x) |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | **354** | 460 (0.77x) | 472 (0.75x) | 408 (0.87x) | 359 (0.99x) | 378 (0.94x) |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | **646** | 775 (0.83x) | 808 (0.80x) | 709 (0.91x) | 656 (0.98x) | 675 (0.96x) |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | **2359** | 2755 (0.86x) | 2777 (0.85x) | 2446 (0.96x) | 2397 (0.98x) | 2414 (0.98x) |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | 3694 | 4192 (0.88x) | 5224 (0.71x) | 5054 (0.73x) | **3625** (1.02x) | 4044 (0.91x) |
| IBP | 1920 x 20, 128 | **2824** (host) | 3021 (0.93x) | 3011 (0.94x) | 2990 (0.94x) | 2839 (0.99x) | 2826 (1.00x) |
| exactkv | 4096 x 16, 256 | **3466** (host) | 3913 (0.89x) | 3844 (0.90x) | 3714 (0.93x) | 3983 (0.87x) | 3623 (0.96x) |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip | adacode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | **0.69** | 0.65 (0.95x) | 0.65 (0.95x) | 0.67 (0.98x) | 0.68 (0.99x) | 0.67 (0.98x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | **0.66** | 0.63 (0.95x) | 0.62 (0.95x) | 0.64 (0.98x) | 0.65 (0.99x) | 0.64 (0.98x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | **0.61** | 0.58 (0.95x) | 0.58 (0.94x) | 0.59 (0.97x) | 0.61 (0.99x) | 0.60 (0.98x) |
| SplitZip, 1 at a time | 4096 x 1, 128 | **0.53** | 0.50 (0.94x) | 0.50 (0.93x) | 0.51 (0.97x) | 0.53 (1.00x) | 0.52 (0.98x) |
| SplitZip, 1 at a time | 8192 x 1, 128 | **0.41** | 0.38 (0.91x) | 0.37 (0.90x) | 0.39 (0.95x) | 0.41 (1.00x) | 0.41 (0.99x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.26 | 0.23 (0.90x) | 0.23 (0.88x) | 0.24 (0.94x) | **0.26** (1.02x) | 0.26 (1.00x) |
| SplitZip, 16 at once | 128 x 16, 128 | **9.94** | 8.83 (0.89x) | 8.74 (0.88x) | 9.15 (0.92x) | 9.67 (0.97x) | 9.05 (0.91x) |
| SplitZip, 16 at once | 256 x 16, 128 | **9.03** | 8.08 (0.89x) | 8.00 (0.89x) | 8.38 (0.93x) | 8.74 (0.97x) | 8.38 (0.93x) |
| SplitZip, 16 at once | 512 x 16, 128 | **7.60** | 6.90 (0.91x) | 6.75 (0.89x) | 7.11 (0.94x) | 7.40 (0.97x) | 7.15 (0.94x) |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | **5.75** | 5.25 (0.91x) | 5.06 (0.88x) | 5.31 (0.92x) | 5.68 (0.99x) | 5.51 (0.96x) |
| SplitZip, 16 at once | 2048 x 16, 128 | **3.88** (host) | 3.54 (0.91x) | 3.44 (0.89x) | 3.62 (0.93x) | 3.83 (0.99x) | 3.78 (0.97x) |
| SplitZip, 16 at once | 4096 x 16, 128 | 2.22 (host) | 2.21 (0.99x) | 2.12 (0.96x) | 2.24 (1.01x) | **2.50** (1.13x) | 2.45 (1.10x) |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | **1.25** | 1.19 (0.95x) | 1.19 (0.95x) | 1.23 (0.98x) | 1.24 (0.99x) | 1.24 (0.99x) |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | **2.26** | 2.15 (0.95x) | 2.10 (0.93x) | 2.17 (0.96x) | 2.24 (0.99x) | 2.21 (0.98x) |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | **3.82** | 3.55 (0.93x) | 3.44 (0.90x) | 3.58 (0.94x) | 3.79 (0.99x) | 3.67 (0.96x) |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | **7.87** | 7.03 (0.89x) | 6.58 (0.84x) | 7.10 (0.90x) | 7.77 (0.99x) | 7.52 (0.96x) |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | **10.78** | 9.26 (0.86x) | 7.90 (0.73x) | 8.28 (0.77x) | 10.51 (0.98x) | 9.57 (0.89x) |
| IBP | 1920 x 20, 128 | 4.27 (host) | 3.95 (0.93x) | 3.82 (0.90x) | 3.95 (0.93x) | **4.32** (1.01x) | 4.27 (1.00x) |
| exactkv | 4096 x 16, 256 | 1.92 (host) | 1.82 (0.95x) | 1.80 (0.94x) | 1.87 (0.97x) | 1.93 (1.01x) | **2.00** (1.04x) |

## Published workloads: 1p3d, host memory, link 10 Gb/s

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer over the GPU budget while others were held), which costs copies the others did not pay.

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip | adacode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | 129 | 178 (0.72x) | 185 (0.70x) | 136 (0.94x) | **129** (1.00x) | 151 (0.85x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | 214 | 249 (0.86x) | 257 (0.83x) | 210 (1.02x) | **203** (1.05x) | 223 (0.96x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | 394 | 400 (0.98x) | 408 (0.96x) | 361 (1.09x) | **356** (1.11x) | 375 (1.05x) |
| SplitZip, 1 at a time | 4096 x 1, 128 | 768 | 722 (1.07x) | 727 (1.06x) | **682** (1.13x) | 685 (1.12x) | 690 (1.11x) |
| SplitZip, 1 at a time | 8192 x 1, 128 | 1582 | 1487 (1.06x) | 1500 (1.05x) | 1422 (1.11x) | 1408 (1.12x) | **1392** (1.14x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 3506 | 3300 (1.06x) | 3333 (1.05x) | 3176 (1.10x) | 3142 (1.12x) | **3096** (1.13x) |
| SplitZip, 16 at once | 128 x 16, 128 | **212** | 344 (0.62x) | 408 (0.52x) | 221 (0.96x) | 330 (0.64x) | 259 (0.82x) |
| SplitZip, 16 at once | 256 x 16, 128 | **297** | 576 (0.52x) | 579 (0.51x) | 404 (0.74x) | 359 (0.83x) | 422 (0.70x) |
| SplitZip, 16 at once | 512 x 16, 128 | **536** | 837 (0.64x) | 862 (0.62x) | 701 (0.76x) | 629 (0.85x) | 684 (0.78x) |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | **1056** (host) | 1511 (0.70x) | 1512 (0.70x) | 1251 (0.84x) | 1206 (0.88x) (host) | 1248 (0.85x) (host) |
| SplitZip, 16 at once | 2048 x 16, 128 | 2205 (host) | 2436 (0.91x) (host) | 2567 (0.86x) | 2167 (1.02x) (host) | 1969 (1.12x) (host) | **1830** (1.20x) (host) |
| SplitZip, 16 at once | 4096 x 16, 128 | 5805 (host) | 5043 (1.15x) (host) | 5169 (1.12x) (host) | 5077 (1.14x) (host) | **4323** (1.34x) (host) | 4648 (1.25x) (host) |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | 370 | 354 (1.05x) | 334 (1.11x) | 333 (1.11x) | **328** (1.13x) | 335 (1.11x) |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | 610 | 535 (1.14x) | 506 (1.21x) | **454** (1.35x) | 534 (1.14x) | 508 (1.20x) |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | 802 | 857 (0.94x) | 860 (0.93x) | **722** (1.11x) | 754 (1.06x) | 737 (1.09x) |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | 2026 (host) | 2590 (0.78x) (host) | 2737 (0.74x) | 2178 (0.93x) | 1670 (1.21x) (host) | **1656** (1.22x) (host) |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | 4602 (host) | 4287 (1.07x) (host) | 4985 (0.92x) | 5117 (0.90x) | **3714** (1.24x) (host) | 4238 (1.09x) (host) |
| IBP | 1920 x 20, 128 | 2538 (host) | 2793 (0.91x) (host) | 2854 (0.89x) (host) | 2412 (1.05x) (host) | 2271 (1.12x) (host) | **2086** (1.22x) (host) |
| exactkv | 4096 x 16, 256 | 3883 (host) | 4525 (0.86x) (host) | 4034 (0.96x) (host) | 4204 (0.92x) (host) | **3552** (1.09x) (host) | 3803 (1.02x) (host) |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip | adacode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | **0.66** | 0.64 (0.97x) | 0.64 (0.97x) | 0.66 (0.99x) | 0.65 (0.98x) | 0.66 (0.99x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | 0.62 | 0.61 (0.98x) | 0.60 (0.97x) | **0.62** (1.00x) | 0.61 (0.99x) | 0.62 (0.99x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | 0.55 | 0.55 (1.00x) | 0.55 (1.00x) | 0.56 (1.02x) | **0.56** (1.02x) | 0.56 (1.01x) |
| SplitZip, 1 at a time | 4096 x 1, 128 | 0.45 | 0.46 (1.02x) | 0.46 (1.02x) | **0.47** (1.04x) | 0.46 (1.03x) | 0.46 (1.04x) |
| SplitZip, 1 at a time | 8192 x 1, 128 | 0.32 | 0.33 (1.03x) | 0.33 (1.03x) | 0.34 (1.05x) | 0.34 (1.06x) | **0.34** (1.06x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.19 | 0.20 (1.04x) | 0.20 (1.04x) | 0.20 (1.07x) | 0.20 (1.07x) | **0.21** (1.09x) |
| SplitZip, 16 at once | 128 x 16, 128 | **9.47** | 8.56 (0.90x) | 8.40 (0.89x) | 8.95 (0.94x) | 8.98 (0.95x) | 9.04 (0.95x) |
| SplitZip, 16 at once | 256 x 16, 128 | **8.50** | 8.04 (0.95x) | 7.99 (0.94x) | 8.38 (0.99x) | 8.44 (0.99x) | 8.25 (0.97x) |
| SplitZip, 16 at once | 512 x 16, 128 | **7.02** | 6.64 (0.95x) | 6.60 (0.94x) | 6.89 (0.98x) | 6.97 (0.99x) | 6.79 (0.97x) |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | **5.11** (host) | 4.82 (0.94x) | 4.84 (0.95x) | 5.10 (1.00x) | 5.07 (0.99x) (host) | 5.06 (0.99x) (host) |
| SplitZip, 16 at once | 2048 x 16, 128 | 3.26 (host) | 3.31 (1.01x) (host) | 3.27 (1.00x) | 3.40 (1.04x) (host) | 3.33 (1.02x) (host) | **3.49** (1.07x) (host) |
| SplitZip, 16 at once | 4096 x 16, 128 | 1.73 (host) | 2.06 (1.19x) (host) | 2.02 (1.17x) (host) | 2.05 (1.18x) (host) | **2.07** (1.20x) (host) | 2.06 (1.19x) (host) |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | 1.13 | 1.14 (1.01x) | 1.15 (1.02x) | **1.16** (1.02x) | 1.16 (1.02x) | 1.15 (1.02x) |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | 1.97 | 2.01 (1.02x) | 2.03 (1.03x) | **2.07** (1.05x) | 2.02 (1.03x) | 2.03 (1.03x) |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | 3.27 | 3.29 (1.01x) | 3.31 (1.01x) | **3.42** (1.05x) | 3.37 (1.03x) | 3.41 (1.04x) |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | 6.52 (host) | 6.51 (1.00x) (host) | 6.44 (0.99x) | 6.83 (1.05x) | **7.29** (1.12x) (host) | 7.03 (1.08x) (host) |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | 7.76 (host) | 8.62 (1.11x) (host) | 7.76 (1.00x) | 8.01 (1.03x) | **8.73** (1.12x) (host) | 8.55 (1.10x) (host) |
| IBP | 1920 x 20, 128 | 3.55 (host) | 3.60 (1.01x) (host) | 3.62 (1.02x) (host) | 3.79 (1.07x) (host) | **3.99** (1.12x) (host) | 3.94 (1.11x) (host) |
| exactkv | 4096 x 16, 256 | 1.67 (host) | 1.69 (1.01x) (host) | 1.73 (1.04x) (host) | 1.72 (1.03x) (host) | **1.77** (1.06x) (host) | 1.72 (1.03x) (host) |

## Published workloads: 1p3d, host memory, link 100 Gb/s

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer over the GPU budget while others were held), which costs copies the others did not pay.

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip | adacode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | **86** | 155 (0.55x) | 163 (0.53x) | 112 (0.77x) | 97 (0.89x) | 122 (0.71x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | **128** | 203 (0.63x) | 213 (0.60x) | 162 (0.79x) | 139 (0.92x) | 163 (0.79x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | **223** | 311 (0.72x) | 320 (0.70x) | 265 (0.84x) | 230 (0.97x) | 255 (0.87x) |
| SplitZip, 1 at a time | 4096 x 1, 128 | **430** | 539 (0.80x) | 551 (0.78x) | 489 (0.88x) | 431 (1.00x) | 453 (0.95x) |
| SplitZip, 1 at a time | 8192 x 1, 128 | 907 | 1126 (0.81x) | 1153 (0.79x) | 1028 (0.88x) | **899** (1.01x) | 921 (0.99x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 2129 | 2564 (0.83x) | 2643 (0.81x) | 2395 (0.89x) | **2114** (1.01x) | 2137 (1.00x) |
| SplitZip, 16 at once | 128 x 16, 128 | **229** | 426 (0.54x) | 439 (0.52x) | 273 (0.84x) | 265 (0.86x) | 275 (0.83x) |
| SplitZip, 16 at once | 256 x 16, 128 | **366** | 556 (0.66x) | 566 (0.65x) | 393 (0.93x) | 407 (0.90x) | 422 (0.87x) |
| SplitZip, 16 at once | 512 x 16, 128 | 670 | 878 (0.76x) | 866 (0.77x) | 703 (0.95x) | **669** (1.00x) | 710 (0.94x) |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | **1254** (host) | 1468 (0.85x) | 1476 (0.85x) | 1299 (0.97x) | 1273 (0.98x) | 1279 (0.98x) |
| SplitZip, 16 at once | 2048 x 16, 128 | 2478 (host) | 2644 (0.94x) | 2669 (0.93x) | 2472 (1.00x) | **2369** (1.05x) | 2434 (1.02x) |
| SplitZip, 16 at once | 4096 x 16, 128 | 4693 (host) | 4684 (1.00x) | 4909 (0.96x) | 4647 (1.01x) | **3941** (1.19x) (host) | 4177 (1.12x) |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | **208** | 281 (0.74x) | 291 (0.72x) | 236 (0.88x) | 210 (0.99x) | 226 (0.92x) |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | **367** | 462 (0.80x) | 470 (0.78x) | 421 (0.87x) | 372 (0.99x) | 385 (0.96x) |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | 660 | 801 (0.82x) | 794 (0.83x) | 723 (0.91x) | **655** (1.01x) | 684 (0.97x) |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | **2369** (host) | 2744 (0.86x) | 2707 (0.88x) | 2444 (0.97x) | 2408 (0.98x) | 2409 (0.98x) |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | **3700** (host) | 4216 (0.88x) | 4906 (0.75x) | 4976 (0.74x) | 3723 (0.99x) | 4133 (0.90x) |
| IBP | 1920 x 20, 128 | **2759** (host) | 3143 (0.88x) | 3113 (0.89x) | 2892 (0.95x) | 2843 (0.97x) | 2844 (0.97x) |
| exactkv | 4096 x 16, 256 | **3600** (host) | 3909 (0.92x) | 3846 (0.94x) | 3674 (0.98x) | 3817 (0.94x) (host) | 3943 (0.91x) |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip | adacode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 512 x 1, 128 | **0.68** | 0.65 (0.96x) | 0.65 (0.95x) | 0.67 (0.98x) | 0.68 (0.99x) | 0.67 (0.98x) |
| SplitZip, 1 at a time + SplitZip, 1,024 tokens | 1024 x 1, 128 | **0.66** | 0.63 (0.95x) | 0.62 (0.95x) | 0.64 (0.98x) | 0.65 (0.99x) | 0.64 (0.98x) |
| SplitZip, 1 at a time | 2048 x 1, 128 | **0.61** | 0.58 (0.95x) | 0.57 (0.95x) | 0.59 (0.98x) | 0.60 (1.00x) | 0.60 (0.98x) |
| SplitZip, 1 at a time | 4096 x 1, 128 | 0.53 | 0.50 (0.95x) | 0.49 (0.94x) | 0.51 (0.97x) | **0.53** (1.00x) | 0.52 (0.99x) |
| SplitZip, 1 at a time | 8192 x 1, 128 | 0.41 | 0.37 (0.92x) | 0.37 (0.91x) | 0.39 (0.95x) | **0.41** (1.00x) | 0.40 (0.99x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.26 | 0.23 (0.90x) | 0.23 (0.88x) | 0.24 (0.93x) | **0.26** (1.00x) | 0.26 (1.00x) |
| SplitZip, 16 at once | 128 x 16, 128 | **9.91** | 8.85 (0.89x) | 8.74 (0.88x) | 9.11 (0.92x) | 9.59 (0.97x) | 9.11 (0.92x) |
| SplitZip, 16 at once | 256 x 16, 128 | **9.04** | 8.19 (0.91x) | 8.03 (0.89x) | 8.40 (0.93x) | 8.80 (0.97x) | 8.36 (0.92x) |
| SplitZip, 16 at once | 512 x 16, 128 | **7.52** | 6.94 (0.92x) | 6.76 (0.90x) | 7.10 (0.94x) | 7.44 (0.99x) | 7.12 (0.95x) |
| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 16, 128 | **5.66** (host) | 5.25 (0.93x) | 5.04 (0.89x) | 5.34 (0.94x) | 5.65 (1.00x) | 5.53 (0.98x) |
| SplitZip, 16 at once | 2048 x 16, 128 | 3.63 (host) | 3.54 (0.97x) | 3.43 (0.95x) | 3.63 (1.00x) | **3.84** (1.06x) | 3.75 (1.03x) |
| SplitZip, 16 at once | 4096 x 16, 128 | 2.16 (host) | 2.20 (1.02x) | 2.12 (0.98x) | 2.24 (1.04x) | **2.47** (1.14x) (host) | 2.43 (1.13x) |
| SplitZip, 1,024 tokens | 1024 x 2, 128 | **1.25** | 1.19 (0.95x) | 1.18 (0.95x) | 1.22 (0.98x) | 1.24 (1.00x) | 1.23 (0.99x) |
| SplitZip, 1,024 tokens | 1024 x 4, 128 | **2.24** | 2.14 (0.96x) | 2.10 (0.94x) | 2.16 (0.96x) | 2.22 (0.99x) | 2.20 (0.98x) |
| SplitZip, 1,024 tokens | 1024 x 8, 128 | 3.76 | 3.55 (0.94x) | 3.45 (0.92x) | 3.56 (0.95x) | **3.79** (1.01x) | 3.67 (0.98x) |
| SplitZip, 1,024 tokens | 1024 x 32, 128 | 7.60 (host) | 7.01 (0.92x) | 6.67 (0.88x) | 7.10 (0.93x) | **7.76** (1.02x) | 7.48 (0.99x) |
| SplitZip, 1,024 tokens | 1024 x 64, 128 | 10.25 (host) | 9.17 (0.89x) | 7.91 (0.77x) | 8.34 (0.81x) | **10.35** (1.01x) | 9.52 (0.93x) |
| IBP | 1920 x 20, 128 | 4.12 (host) | 3.95 (0.96x) | 3.80 (0.92x) | 3.97 (0.97x) | **4.30** (1.04x) | 4.23 (1.03x) |
| exactkv | 4096 x 16, 256 | 1.91 (host) | 1.86 (0.97x) | 1.80 (0.94x) | 1.88 (0.99x) | **1.95** (1.02x) (host) | 1.92 (1.00x) |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, GPU to GPU | t2-hbh | native | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, GPU to GPU | t2-hbh | native | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, GPU to GPU | t2-hbh | native | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, GPU to GPU | t2-hbh | native | inf | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, GPU to GPU | t2-rank256 | native | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, GPU to GPU | t2-rank256 | native | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, GPU to GPU | t2-rank256 | native | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, GPU to GPU | t2-rank256 | native | inf | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 10 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 10 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 10 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 10 | inf | 0 | 32 | {'math': 6, 'exam': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 10 | inf | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-hbh | 100 | inf | 0 | 32 | {'exam': 6, 'math': 6, 'wikitext': 7, 'news': 6, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | 0.5 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | 1.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | 2.0 | 0 | 32 | {'exam': 6, 'math': 6, 'news': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |
| 1p3d, host memory | t2-rank256 | 100 | inf | 0 | 32 | {'news': 6, 'math': 6, 'exam': 6, 'wikitext': 7, 'code': 7} | differs in 1978893 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the sc |

