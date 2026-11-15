# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| splitzip | 4/4 | 4/4 | 12.044 |
| exans | 4/4 | 4/4 | 10.923 |
| adacode | 4/4 | 4/4 | 11.371 |
| exactkv | 4/4 | 4/4 | 12.629 |
| ours-hbh | 4/4 | 4/4 | 9.053 |
| t2-hbh | 4/4 | 4/4 | 8.489 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.87 | 0.70 | 2405 / 2819 | 11.5 | 16.000 | 1.1 / 2.7 | 469.8 | 3.7 |
| splitzip | inf | 1.94 | 0.73 | 2362 / 2654 | 11.6 | 12.042 | 32.0 / 5.1 | 359.1 | 3.8 |
| exans | inf | 2.06 | 1.03 | 2069 / 2393 | 11.7 | 10.884 | 50.7 / 11.9 | 293.9 | 4.1 |
| adacode | inf | 2.07 | 0.77 | 2074 / 2396 | 11.7 | 11.372 | 13.8 / 8.0 | 320.7 | 3.8 |
| exactkv | inf | 2.02 | 0.76 | 2165 / 2493 | 11.6 | 12.591 | 16.2 / 8.7 | 367.9 | 3.8 |
| ours-hbh | inf | 2.11 | 1.05 | 2034 / 2295 | 12.8 | 8.684 | 32.2 / 155.3 | 228.4 | 4.0 |
| t2-hbh | inf | 2.12 | 1.06 | 2019 / 2265 | 12.8 | 8.146 | 46.9 / 156.9 | 211.5 | 3.9 |

## Kernels compiled while serving (vLLM's JIT warnings)

15 in the engine logs, 0 inside a measured point.

- 17:30:40 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 17:30:40 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 17:30:53 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _encode_pages_kernel. This causes a latency spike; consider ex
- 17:31:08 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:31:08 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:31:09 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:31:09 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:31:09 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:31:09 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:31:09 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:31:09 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:31:09 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:31:09 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:31:09 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:31:09 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

