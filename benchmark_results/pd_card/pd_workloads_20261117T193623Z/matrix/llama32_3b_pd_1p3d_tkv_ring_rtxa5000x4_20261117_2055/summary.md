# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |
| exactkv | 4/4 | 3/4 | 12.629 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5 | 0.49 | 0.49 | 392 / 1025 | 11.2 | 16.000 | 1.1 / 2.7 | 219.1 | 9.3 |
| raw | 1.0 | 0.95 | 0.95 | 393 / 1258 | 11.2 | 16.000 | 1.1 / 2.6 | 239.1 | 8.7 |
| raw | 2.0 | 1.79 | 1.79 | 427 / 1420 | 11.3 | 16.000 | 1.1 / 2.7 | 259.1 | 8.1 |
| raw | inf | 2.99 | 0.00 | 6415 / 9135 | 11.8 | 16.000 | 1.1 / 2.7 | 484.1 | 3.7 |
| exactkv | 0.5 | 0.49 | 0.49 | 388 / 942 | 11.2 | 12.591 | 25.0 / 10.2 | 171.0 | 9.3 |
| exactkv | 1.0 | 0.95 | 0.95 | 390 / 1122 | 11.2 | 12.591 | 23.2 / 10.2 | 188.1 | 8.6 |
| exactkv | 2.0 | 1.79 | 1.79 | 423 / 1280 | 11.3 | 12.591 | 23.7 / 10.1 | 204.0 | 8.1 |
| exactkv | inf | 3.24 | 0.10 | 6154 / 8374 | 12.2 | 12.591 | 14.3 / 10.3 | 375.1 | 3.8 |

## Link 100 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5 | 0.49 | 0.49 | 223 / 544 | 11.2 | 16.000 | 1.1 / 2.6 | 21.6 | 89.3 |
| raw | 1.0 | 0.95 | 0.95 | 224 / 682 | 11.2 | 16.000 | 1.1 / 2.6 | 23.7 | 83.5 |
| raw | 2.0 | 1.81 | 1.81 | 251 / 693 | 11.3 | 16.000 | 1.1 / 2.6 | 24.3 | 81.3 |
| raw | inf | 4.21 | 0.00 | 5070 / 6015 | 12.6 | 16.000 | 1.2 / 2.9 | 89.1 | 31.2 |
| exactkv | 0.5 | 0.49 | 0.49 | 255 / 565 | 11.2 | 12.591 | 25.2 / 10.1 | 15.5 | 89.6 |
| exactkv | 1.0 | 0.95 | 0.95 | 257 / 687 | 11.2 | 12.591 | 24.4 / 10.0 | 15.5 | 89.7 |
| exactkv | 2.0 | 1.81 | 1.81 | 283 / 712 | 11.3 | 12.591 | 23.4 / 10.0 | 15.6 | 88.5 |
| exactkv | inf | 4.35 | 0.00 | 5067 / 5729 | 13.0 | 12.591 | 17.7 / 10.0 | 15.7 | 88.3 |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 512 | 1 | 128 | 8 | sz1 | raw | 128 / 129 | 0.66 | 16.000 | 0 |
| 10 | 1024 | 1 | 128 | 8 | sz1/sz1024 | raw | 214 / 215 | 0.62 | 16.000 | 0 |
| 10 | 128 | 16 | 128 | 32 | sz16 | raw | 210 / 410 | 9.49 | 16.000 | 0 |
| 10 | 1024 | 2 | 128 | 8 | sz1024 | raw | 371 / 384 | 1.13 | 16.000 | 0 |
| 10 | 2048 | 1 | 128 | 8 | sz1 | raw | 393 / 395 | 0.55 | 16.000 | 0 |
| 10 | 256 | 16 | 128 | 32 | sz16 | raw | 295 / 737 | 8.50 | 16.000 | 0 |
| 10 | 1024 | 4 | 128 | 8 | sz1024 | raw | 603 / 744 | 1.96 | 16.000 | 0 |
| 10 | 4096 | 1 | 128 | 8 | sz1 | raw | 770 / 773 | 0.45 | 16.000 | 0 |
| 10 | 512 | 16 | 128 | 32 | sz16 | raw | 583 / 1417 | 6.91 | 16.000 | 0 |
| 10 | 1024 | 8 | 128 | 16 | sz1024 | raw | 808 / 1432 | 3.27 | 16.000 | 0 |
| 10 | 8192 | 1 | 128 | 8 | sz1 | raw | 1585 / 1592 | 0.32 | 16.000 | 0 |
| 10 | 1024 | 16 | 128 | 32 | sz16/sz1024 | raw | 1046 / 2735 | 5.12 | 16.000 | 0 |
| 10 | 16384 | 1 | 128 | 4 | sz1 | raw | 4429 / 4573 | 0.16 | 16.000 | 0 |
| 10 | 1024 | 32 | 128 | 64 | sz1024 | raw | 2201 / 5566 | 6.43 | 16.000 | 0 |
| 10 | 2048 | 16 | 128 | 32 | sz16 | raw | 2193 / 5353 | 3.26 | 16.000 | 0 |
| 10 | 1920 | 20 | 128 | 40 | ibp | raw | 2631 / 6571 | 3.53 | 16.000 | 0 |
| 10 | 1024 | 64 | 128 | 128 | sz1024 | raw | 4590 / 9357 | 7.60 | 16.000 | 0 |
| 10 | 4096 | 16 | 128 | 32 | sz16 | raw | 5585 / 9694 | 1.79 | 16.000 | 0 |
| 10 | 4096 | 16 | 256 | 32 | exactkv | raw | 4201 / 10204 | 1.64 | 16.000 | 0 |
| 10 | 512 | 1 | 128 | 8 | sz1 | exactkv | 152 / 155 | 0.65 | 12.600 | 0 |
| 10 | 1024 | 1 | 128 | 8 | sz1/sz1024 | exactkv | 228 / 230 | 0.61 | 12.593 | 0 |
| 10 | 128 | 16 | 128 | 32 | sz16 | exactkv | 289 / 450 | 9.12 | 12.620 | 0 |
| 10 | 1024 | 2 | 128 | 8 | sz1024 | exactkv | 345 / 360 | 1.15 | 12.593 | 0 |
| 10 | 2048 | 1 | 128 | 8 | sz1 | exactkv | 390 / 392 | 0.55 | 12.592 | 0 |
| 10 | 256 | 16 | 128 | 32 | sz16 | exactkv | 396 / 676 | 8.25 | 12.605 | 0 |
| 10 | 1024 | 4 | 128 | 8 | sz1024 | exactkv | 511 / 691 | 2.02 | 12.593 | 0 |
| 10 | 4096 | 1 | 128 | 8 | sz1 | exactkv | 730 / 733 | 0.45 | 12.590 | 0 |
| 10 | 512 | 16 | 128 | 32 | sz16 | exactkv | 651 / 1279 | 6.80 | 12.598 | 0 |
| 10 | 1024 | 8 | 128 | 16 | sz1024 | exactkv | 734 / 1290 | 3.37 | 12.593 | 0 |
| 10 | 8192 | 1 | 128 | 8 | sz1 | exactkv | 1468 / 1469 | 0.33 | 12.588 | 0 |
| 10 | 1024 | 16 | 128 | 32 | sz16/sz1024 | exactkv | 1132 / 2438 | 5.16 | 12.593 | 0 |
| 10 | 16384 | 1 | 128 | 4 | sz1 | exactkv | 3928 / 4030 | 0.18 | 12.587 | 0 |
| 10 | 1024 | 32 | 128 | 64 | sz1024 | exactkv | 1802 / 4898 | 7.03 | 12.593 | 0 |
| 10 | 2048 | 16 | 128 | 32 | sz16 | exactkv | 1918 / 4788 | 3.36 | 12.591 | 0 |
| 10 | 1920 | 20 | 128 | 40 | ibp | exactkv | 2411 / 5990 | 3.83 | 12.591 | 0 |
| 10 | 1024 | 64 | 128 | 128 | sz1024 | exactkv | 4208 / 8785 | 8.37 | 12.593 | 0 |
| 10 | 4096 | 16 | 128 | 32 | sz16 | exactkv | 5649 / 9511 | 1.82 | 12.589 | 0 |
| 10 | 4096 | 16 | 256 | 32 | exactkv | exactkv | 3403 / 8729 | 1.76 | 12.589 | 0 |
| 100 | 512 | 1 | 128 | 8 | sz1 | raw | 85 / 88 | 0.68 | 16.000 | 0 |
| 100 | 1024 | 1 | 128 | 8 | sz1/sz1024 | raw | 128 / 130 | 0.66 | 16.000 | 0 |
| 100 | 128 | 16 | 128 | 32 | sz16 | raw | 220 / 241 | 9.92 | 16.000 | 0 |
| 100 | 1024 | 2 | 128 | 8 | sz1024 | raw | 209 / 213 | 1.25 | 16.000 | 0 |
| 100 | 2048 | 1 | 128 | 8 | sz1 | raw | 222 / 224 | 0.61 | 16.000 | 0 |
| 100 | 256 | 16 | 128 | 32 | sz16 | raw | 370 / 419 | 9.01 | 16.000 | 0 |
| 100 | 1024 | 4 | 128 | 8 | sz1024 | raw | 370 / 414 | 2.24 | 16.000 | 0 |
| 100 | 4096 | 1 | 128 | 8 | sz1 | raw | 427 / 434 | 0.53 | 16.000 | 0 |
| 100 | 512 | 16 | 128 | 32 | sz16 | raw | 663 / 735 | 7.52 | 16.000 | 0 |
| 100 | 1024 | 8 | 128 | 16 | sz1024 | raw | 653 / 735 | 3.78 | 16.000 | 0 |
| 100 | 8192 | 1 | 128 | 8 | sz1 | raw | 904 / 907 | 0.41 | 16.000 | 0 |
| 100 | 1024 | 16 | 128 | 32 | sz16/sz1024 | raw | 1257 / 1509 | 5.60 | 16.000 | 0 |
| 100 | 16384 | 1 | 128 | 4 | sz1 | raw | 3762 / 3936 | 0.18 | 16.000 | 0 |
| 100 | 1024 | 32 | 128 | 64 | sz1024 | raw | 2056 / 3252 | 7.62 | 16.000 | 0 |
| 100 | 2048 | 16 | 128 | 32 | sz16 | raw | 2236 / 3297 | 3.73 | 16.000 | 0 |
| 100 | 1920 | 20 | 128 | 40 | ibp | raw | 2932 / 3800 | 4.07 | 16.000 | 0 |
| 100 | 1024 | 64 | 128 | 128 | sz1024 | raw | 3764 / 6763 | 10.24 | 16.000 | 0 |
| 100 | 4096 | 16 | 128 | 32 | sz16 | raw | 4708 / 9204 | 2.15 | 16.000 | 0 |
| 100 | 4096 | 16 | 256 | 32 | exactkv | raw | 3223 / 7141 | 1.90 | 16.000 | 0 |
| 100 | 512 | 1 | 128 | 8 | sz1 | exactkv | 120 / 122 | 0.67 | 12.600 | 0 |
| 100 | 1024 | 1 | 128 | 8 | sz1/sz1024 | exactkv | 163 / 166 | 0.64 | 12.593 | 0 |
| 100 | 128 | 16 | 128 | 32 | sz16 | exactkv | 281 / 459 | 9.15 | 12.621 | 0 |
| 100 | 1024 | 2 | 128 | 8 | sz1024 | exactkv | 225 / 238 | 1.23 | 12.593 | 0 |
| 100 | 2048 | 1 | 128 | 8 | sz1 | exactkv | 259 / 265 | 0.59 | 12.592 | 0 |
| 100 | 256 | 16 | 128 | 32 | sz16 | exactkv | 416 / 591 | 8.43 | 12.606 | 0 |
| 100 | 1024 | 4 | 128 | 8 | sz1024 | exactkv | 385 / 458 | 2.19 | 12.593 | 0 |
| 100 | 4096 | 1 | 128 | 8 | sz1 | exactkv | 464 / 465 | 0.52 | 12.590 | 0 |
| 100 | 512 | 16 | 128 | 32 | sz16 | exactkv | 710 / 906 | 7.17 | 12.597 | 0 |
| 100 | 1024 | 8 | 128 | 16 | sz1024 | exactkv | 686 / 818 | 3.67 | 12.593 | 0 |
| 100 | 8192 | 1 | 128 | 8 | sz1 | exactkv | 940 / 952 | 0.40 | 12.588 | 0 |
| 100 | 1024 | 16 | 128 | 32 | sz16/sz1024 | exactkv | 1288 / 1536 | 5.50 | 12.593 | 0 |
| 100 | 16384 | 1 | 128 | 4 | sz1 | exactkv | 2855 / 2964 | 0.22 | 12.587 | 0 |
| 100 | 1024 | 32 | 128 | 64 | sz1024 | exactkv | 2401 / 3135 | 7.39 | 12.593 | 0 |
| 100 | 2048 | 16 | 128 | 32 | sz16 | exactkv | 2412 / 2923 | 3.74 | 12.591 | 0 |
| 100 | 1920 | 20 | 128 | 40 | ibp | exactkv | 2860 / 3413 | 4.21 | 12.591 | 0 |
| 100 | 1024 | 64 | 128 | 128 | sz1024 | exactkv | 4074 / 8412 | 9.48 | 12.593 | 0 |
| 100 | 4096 | 16 | 128 | 32 | sz16 | exactkv | 4130 / 7655 | 2.42 | 12.589 | 0 |
| 100 | 4096 | 16 | 256 | 32 | exactkv | exactkv | 4008 / 6152 | 1.90 | 12.589 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

13 in the engine logs, 0 inside a measured point.

- 20:56:45 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _encode_pages_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:56:50 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:56:50 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:56:50 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:56:50 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:56:50 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

