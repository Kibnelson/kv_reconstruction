# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 2 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| ours-hbh | 4/4 | 4/4 | 9.053 |
| t2-hbh | 4/4 | 4/4 | 8.489 |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 3.50 | 3.50 | 737 / 763 | 12.0 | 16.000 | 1.4 / 2.7 | 12.3 | 124.9 |
| ours-hbh | inf | 2.81 | 2.81 | 1291 / 1318 | 12.0 | 8.684 | 32.4 / 145.6 | 5.6 | 135.8 |
| t2-hbh | inf | 2.78 | 2.78 | 1331 / 1356 | 12.0 | 8.146 | 44.8 / 153.7 | 5.5 | 137.1 |

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 14:54:28 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:54:28 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:54:28 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:54:28 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:54:28 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:54:28 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:54:28 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:54:28 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:54:28 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:54:28 engine_prefill2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:54:28 engine_prefill2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:54:28 engine_prefill2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

