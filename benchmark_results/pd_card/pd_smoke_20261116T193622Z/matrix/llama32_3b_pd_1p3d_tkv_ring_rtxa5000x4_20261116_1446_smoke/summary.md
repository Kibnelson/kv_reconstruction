# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| splitzip | 4/4 | 3/4 | 12.044 |
| exans | 4/4 | 4/4 | 10.923 |
| adacode | 4/4 | 4/4 | 11.371 |
| exactkv | 4/4 | 3/4 | 12.629 |
| ours-hbh | 4/4 | 4/4 | 9.053 |
| t2-hbh | 4/4 | 4/4 | 8.489 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.89 | 0.71 | 2380 / 2788 | 11.4 | 16.000 | 1.1 / 2.7 | 469.8 | 3.7 |
| splitzip | inf | 1.95 | 0.73 | 2341 / 2634 | 11.6 | 12.042 | 31.1 / 5.3 | 359.1 | 3.8 |
| exans | inf | 2.08 | 1.04 | 2060 / 2376 | 11.6 | 10.884 | 50.3 / 11.9 | 293.6 | 4.1 |
| adacode | inf | 2.08 | 0.78 | 2061 / 2379 | 11.6 | 11.372 | 14.2 / 8.6 | 320.7 | 3.8 |
| exactkv | inf | 2.01 | 0.75 | 2176 / 2511 | 11.6 | 12.590 | 14.8 / 8.8 | 367.9 | 3.8 |
| ours-hbh | inf | 2.11 | 1.05 | 2050 / 2305 | 12.7 | 8.684 | 31.9 / 154.7 | 228.2 | 4.0 |
| t2-hbh | inf | 2.14 | 1.07 | 1998 / 2242 | 12.7 | 8.146 | 45.9 / 157.5 | 210.9 | 4.0 |

## Kernels compiled while serving (vLLM's JIT warnings)

15 in the engine logs, 0 inside a measured point.

- 14:48:28 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 14:48:29 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 14:48:42 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _encode_pages_kernel. This causes a latency spike; consider ex
- 14:48:56 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:48:57 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:48:57 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:48:57 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:48:57 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:48:57 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:48:58 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:48:58 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:48:58 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:48:58 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:48:58 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:48:58 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

