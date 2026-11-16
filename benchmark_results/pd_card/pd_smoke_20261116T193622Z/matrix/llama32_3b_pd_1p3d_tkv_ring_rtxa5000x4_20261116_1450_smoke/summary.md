# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| nvcomp-ans | 4/4 | 4/4 | 13.369 |
| nvcomp-zstd | 4/4 | 4/4 | 12.896 |
| nvcomp-bitcomp | 4/4 | 4/4 | 13.790 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.88 | 0.70 | 2393 / 2806 | 11.5 | 16.000 | 1.1 / 2.7 | 469.8 | 3.7 |
| nvcomp-ans | inf | 1.98 | 0.74 | 2224 / 2577 | 11.5 | 13.365 | 17.3 / 11.1 | 387.2 | 3.8 |
| nvcomp-zstd | inf | 1.77 | 0.00 | 2705 / 3049 | 11.6 | 12.892 | 142.8 / 27.5 | 429.4 | 3.3 |
| nvcomp-bitcomp | inf | 1.95 | 0.73 | 2301 / 2638 | 11.6 | 13.786 | 15.0 / 12.1 | 414.5 | 3.8 |

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 14:52:10 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:52:10 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:52:10 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:52:10 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:52:10 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:52:10 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:52:10 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:52:10 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:52:10 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:52:10 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:52:10 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:52:10 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

