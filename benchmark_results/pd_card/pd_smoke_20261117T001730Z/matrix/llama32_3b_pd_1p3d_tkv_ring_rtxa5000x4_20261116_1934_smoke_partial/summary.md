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
| t2-hbh | 3/4 (FAILED) | 4/4 | 8.627 |
| aquakv-2 | 0/4 | 2/4 | 2.126 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.87 | 0.70 | 2392 / 2808 | 11.5 | 16.000 | 1.1 / 2.9 | 469.8 | 3.8 |
| splitzip | inf | 1.94 | 0.73 | 2351 / 2644 | 11.6 | 12.042 | 31.5 / 5.3 | 359.1 | 3.8 |
| exans | inf | 2.06 | 0.77 | 2083 / 2406 | 11.6 | 10.884 | 50.1 / 11.9 | 298.7 | 3.9 |
| adacode | inf | 2.07 | 0.77 | 2079 / 2398 | 11.6 | 11.372 | 14.7 / 8.6 | 320.7 | 3.8 |
| exactkv | inf | 2.02 | 0.76 | 2161 / 2487 | 11.6 | 12.590 | 14.5 / 8.8 | 367.9 | 3.8 |
| ours-hbh | inf | 2.12 | 1.06 | 2021 / 2273 | 12.6 | 8.684 | 31.8 / 151.6 | 228.8 | 3.9 |
| t2-hbh | inf | 2.14 | 1.07 | 1977 / 2234 | 12.7 | 8.146 | 50.2 / 155.5 | 207.6 | 4.2 |
| aquakv-2 | inf | 1.78 | 0.67 | 2312 / 3023 | 11.9 | 2.130 | 201.2 / 72.8 | 25.0 | 9.7 |

## Kernels compiled while serving (vLLM's JIT warnings)

15 in the engine logs, 0 inside a measured point.

- 19:36:22 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 19:36:22 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 19:36:36 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _encode_pages_kernel. This causes a latency spike; consider ex
- 19:36:57 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:57 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:58 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:58 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:58 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:36:58 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:58 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:58 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:36:58 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:36:58 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:36:58 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:36:58 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

