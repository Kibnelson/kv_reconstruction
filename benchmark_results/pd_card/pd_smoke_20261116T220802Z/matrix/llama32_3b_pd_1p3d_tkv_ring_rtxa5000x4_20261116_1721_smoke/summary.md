# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |
| splitzip | 4/4 | 3/4 | 12.044 |
| exans | 4/4 | 4/4 | 10.923 |
| adacode | 4/4 | 3/4 | 11.371 |
| ours-hbh | 4/4 | 3/4 | 9.053 |
| t2-hbh | 4/4 | 4/4 | 8.489 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.88 | 0.71 | 2387 / 2798 | 11.5 | 16.000 | 1.1 / 2.7 | 469.8 | 3.7 |
| splitzip | inf | 1.94 | 0.73 | 2350 / 2642 | 11.6 | 12.042 | 30.7 / 5.2 | 359.1 | 3.8 |
| exans | inf | 2.03 | 0.76 | 2146 / 2465 | 11.6 | 10.884 | 61.4 / 13.0 | 298.6 | 3.9 |
| adacode | inf | 2.07 | 0.78 | 2066 / 2393 | 11.6 | 11.372 | 13.9 / 8.3 | 320.7 | 3.8 |
| ours-hbh | inf | 2.13 | 1.06 | 2015 / 2268 | 12.6 | 8.684 | 32.1 / 155.4 | 228.3 | 4.0 |
| t2-hbh | inf | 2.14 | 1.07 | 2001 / 2243 | 12.7 | 8.146 | 48.2 / 151.3 | 210.9 | 4.0 |

Codecs not run (did not load on every engine): exactkv (CalledProcessError(128, ['git', '-C', '/tmp/nbore/kvpd/30293/home/sources/exactkv', 'checkout', '-q', 'd90376b84e8c11642); aquakv-2 (ValueError('round trip not exact'))

## Kernels compiled while serving (vLLM's JIT warnings)

14 in the engine logs, 0 inside a measured point.

- 17:23:40 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 17:23:40 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 17:24:04 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:24:04 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:24:05 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:24:05 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:24:05 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:24:05 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:24:05 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:24:05 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:24:05 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:24:05 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:24:05 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:24:05 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

