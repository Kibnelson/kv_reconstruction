# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |
| exactkv | 4/4 | 3/4 | 12.629 |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5 | 0.49 | 0.49 | 216 / 524 | 11.2 | 16.000 | 1.2 / 2.7 | 11.4 | 142.0 |
| raw | 1.0 | 0.95 | 0.95 | 215 / 663 | 11.2 | 16.000 | 1.2 / 2.6 | 12.5 | 131.8 |
| raw | 2.0 | 1.81 | 1.81 | 248 / 672 | 11.3 | 16.000 | 1.2 / 2.7 | 14.3 | 118.3 |
| raw | inf | 4.51 | 0.00 | 4626 / 5452 | 12.9 | 16.000 | 2.8 / 2.7 | 39.3 | 64.0 |
| exactkv | 0.5 | 0.49 | 0.49 | 249 / 560 | 11.2 | 12.591 | 25.4 / 10.2 | 9.3 | 132.0 |
| exactkv | 1.0 | 0.95 | 0.95 | 250 / 679 | 11.2 | 12.591 | 24.1 / 9.9 | 9.4 | 131.0 |
| exactkv | 2.0 | 1.81 | 1.81 | 279 / 705 | 11.3 | 12.591 | 23.9 / 10.3 | 9.4 | 131.3 |
| exactkv | inf | 4.41 | 0.00 | 4765 / 5607 | 13.4 | 12.591 | 15.6 / 10.0 | 10.7 | 118.2 |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 512 | 1 | 128 | 8 | sz1 | raw | 83 / 84 | 0.68 | 16.000 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | raw | 125 / 134 | 0.66 | 16.000 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | raw | 224 / 239 | 9.92 | 16.000 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | raw | 204 / 214 | 1.25 | 16.000 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | raw | 218 / 226 | 0.61 | 16.000 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | raw | 370 / 401 | 9.00 | 16.000 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | raw | 357 / 399 | 2.24 | 16.000 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | raw | 413 / 415 | 0.53 | 16.000 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | raw | 668 / 713 | 7.58 | 16.000 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | raw | 650 / 709 | 3.82 | 16.000 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | raw | 874 / 884 | 0.41 | 16.000 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | raw | 1279 / 1341 | 5.70 | 16.000 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | raw | 3814 / 4049 | 0.18 | 16.000 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | raw | 2089 / 2953 | 7.88 | 16.000 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | raw | 2377 / 2777 | 3.83 | 16.000 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | raw | 2926 / 3453 | 4.23 | 16.000 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | raw | 3479 / 6807 | 10.70 | 16.000 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | raw | 4704 / 9207 | 2.18 | 16.000 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | raw | 3937 / 7082 | 1.84 | 16.000 | 0 |
| native | 512 | 1 | 128 | 8 | sz1 | exactkv | 119 / 124 | 0.67 | 12.600 | 0 |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | exactkv | 158 / 161 | 0.64 | 12.593 | 0 |
| native | 128 | 16 | 128 | 32 | sz16 | exactkv | 276 / 439 | 9.13 | 12.621 | 0 |
| native | 1024 | 2 | 128 | 8 | sz1024 | exactkv | 223 / 234 | 1.23 | 12.593 | 0 |
| native | 2048 | 1 | 128 | 8 | sz1 | exactkv | 251 / 256 | 0.60 | 12.592 | 0 |
| native | 256 | 16 | 128 | 32 | sz16 | exactkv | 440 / 598 | 8.31 | 12.606 | 0 |
| native | 1024 | 4 | 128 | 8 | sz1024 | exactkv | 397 / 440 | 2.19 | 12.593 | 0 |
| native | 4096 | 1 | 128 | 8 | sz1 | exactkv | 454 / 462 | 0.52 | 12.590 | 0 |
| native | 512 | 16 | 128 | 32 | sz16 | exactkv | 701 / 913 | 7.15 | 12.597 | 0 |
| native | 1024 | 8 | 128 | 16 | sz1024 | exactkv | 690 / 810 | 3.67 | 12.593 | 0 |
| native | 8192 | 1 | 128 | 8 | sz1 | exactkv | 916 / 922 | 0.41 | 12.588 | 0 |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | exactkv | 1284 / 1538 | 5.52 | 12.593 | 0 |
| native | 16384 | 1 | 128 | 4 | sz1 | exactkv | 2825 / 2893 | 0.22 | 12.587 | 0 |
| native | 1024 | 32 | 128 | 64 | sz1024 | exactkv | 2481 / 2979 | 7.46 | 12.593 | 0 |
| native | 2048 | 16 | 128 | 32 | sz16 | exactkv | 2428 / 2888 | 3.74 | 12.591 | 0 |
| native | 1920 | 20 | 128 | 40 | ibp | exactkv | 2855 / 3359 | 4.24 | 12.591 | 0 |
| native | 1024 | 64 | 128 | 128 | sz1024 | exactkv | 4133 / 8368 | 9.58 | 12.593 | 0 |
| native | 4096 | 16 | 128 | 32 | sz16 | exactkv | 4109 / 7686 | 2.44 | 12.589 | 0 |
| native | 4096 | 16 | 256 | 32 | exactkv | exactkv | 3593 / 5781 | 1.95 | 12.589 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

13 in the engine logs, 0 inside a measured point.

- 21:36:47 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _encode_pages_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:36:52 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:36:52 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:36:52 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:36:52 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:36:52 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

