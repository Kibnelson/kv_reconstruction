# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| nixl | 0/4 | 2/4 | - |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nixl | 0.5 | 0.48 | 0.48 | 624 / 1546 | 11.2 | - | - / - | - | - |
| nixl | 1.0 | 0.94 | 0.94 | 614 / 1561 | 11.2 | - | - / - | - | - |
| nixl | 2.0 | 1.69 | 1.48 | 1040 / 2955 | 11.3 | - | - / - | - | - |
| nixl | inf | 1.76 | 0.16 | 15065 / 16655 | 13.4 | - | - / - | - | - |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 512 | 1 | 128 | 8 | sz1 | nixl | 183 / 185 | 0.64 | - | - |
| native | 1024 | 1 | 128 | 8 | sz1/sz1024 | nixl | 326 / 335 | 0.58 | - | - |
| native | 128 | 16 | 128 | 32 | sz16 | nixl | 409 / 625 | 8.73 | - | - |
| native | 1024 | 2 | 128 | 8 | sz1024 | nixl | 334 / 524 | 1.13 | - | - |
| native | 2048 | 1 | 128 | 8 | sz1 | nixl | 615 / 619 | 0.49 | - | - |
| native | 256 | 16 | 128 | 32 | sz16 | nixl | 625 / 1123 | 7.07 | - | - |
| native | 1024 | 4 | 128 | 8 | sz1024 | nixl | 368 / 973 | 1.92 | - | - |
| native | 4096 | 1 | 128 | 8 | sz1 | nixl | 1219 / 1263 | 0.37 | - | - |
| native | 512 | 16 | 128 | 32 | sz16 | nixl | 1382 / 2318 | 5.01 | - | - |
| native | 1024 | 8 | 128 | 16 | sz1024 | nixl | 942 / 2066 | 2.80 | - | - |
| native | 8192 | 1 | 128 | 8 | sz1 | nixl | 2412 / 2476 | 0.25 | - | - |
| native | 1024 | 16 | 128 | 32 | sz16/sz1024 | nixl | 1978 / 4865 | 3.49 | - | - |
| native | 16384 | 1 | 128 | 4 | sz1 | nixl | 5391 / 5430 | 0.14 | - | - |
| native | 1024 | 32 | 128 | 64 | sz1024 | nixl | 6529 / 8688 | 3.61 | - | - |
| native | 2048 | 16 | 128 | 32 | sz16 | nixl | 6526 / 9154 | 1.71 | - | - |
| native | 1920 | 20 | 128 | 40 | ibp | nixl | 7144 / 11581 | 2.04 | - | - |
| native | 1024 | 64 | 128 | 128 | sz1024 | nixl | 12903 / 25592 | 3.97 | - | - |
| native | 4096 | 16 | 128 | 32 | sz16 | nixl | 12690 / 26375 | 0.87 | - | - |
| native | 4096 | 16 | 256 | 32 | exactkv | nixl | 12415 / 21765 | 0.90 | - | - |

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 19:23:00 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:23:00 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:23:00 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:23:00 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:23:00 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:23:00 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:23:01 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:23:01 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:23:01 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:23:01 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:23:01 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:23:01 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

