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
| aquakv-2 | 0/4 | 2/4 | 2.126 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.87 | 0.70 | 2404 / 2815 | 11.5 | 16.000 | 1.1 / 2.8 | 469.8 | 3.7 |
| splitzip | inf | 1.93 | 0.72 | 2375 / 2670 | 11.6 | 12.042 | 30.6 / 5.1 | 359.1 | 3.8 |
| exans | inf | 2.06 | 1.03 | 2090 / 2408 | 11.6 | 10.884 | 49.8 / 11.9 | 293.6 | 4.1 |
| adacode | inf | 2.08 | 0.78 | 2053 / 2372 | 11.6 | 11.372 | 14.4 / 8.4 | 320.7 | 3.8 |
| exactkv | inf | 2.01 | 0.75 | 2189 / 2514 | 11.6 | 12.590 | 14.1 / 8.7 | 367.9 | 3.7 |
| ours-hbh | inf | 2.12 | 1.06 | 2021 / 2276 | 12.6 | 8.684 | 31.9 / 156.8 | 228.3 | 4.0 |
| t2-hbh | inf | 2.14 | 1.07 | 2001 / 2233 | 12.6 | 8.146 | 45.9 / 148.5 | 210.9 | 4.0 |
| aquakv-2 | inf | 1.99 | 1.00 | 2033 / 2540 | 11.7 | 2.130 | 144.3 / 43.5 | 25.0 | 9.8 |

## Kernels compiled while serving (vLLM's JIT warnings)

15 in the engine logs, 0 inside a measured point.

- 21:40:46 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 21:40:46 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 21:40:59 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _encode_pages_kernel. This causes a latency spike; consider ex
- 21:41:20 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:41:20 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:41:21 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:41:21 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:41:21 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 21:41:21 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:41:21 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:41:21 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 21:41:21 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 21:41:21 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 21:41:21 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 21:41:21 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

