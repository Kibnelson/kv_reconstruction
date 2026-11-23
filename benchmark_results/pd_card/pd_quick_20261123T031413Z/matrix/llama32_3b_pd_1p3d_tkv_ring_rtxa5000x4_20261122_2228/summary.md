# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| ours-hbh | 4/4 | 3/4 | 9.110 |
| t2-hbh | 4/4 | 3/4 | 8.562 |
| t2-rank256 | 4/4 | 4/4 | 9.619 |
| splitzip | 4/4 | 3/4 | 12.043 |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2048 | 1 | 128 | 4 | sz1 | raw | 394 / 396 | 0.55 | 16.000 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | raw | 3536 / 3539 | 0.19 | 16.000 | 0 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | raw | 4279 / 5535 | 2.28 | 16.000 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | ours-hbh | 356 / 359 | 0.56 | 8.784 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | ours-hbh | 2846 / 2863 | 0.22 | 8.701 | 0 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | ours-hbh | 3525 / 4237 | 2.77 | 8.700 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | t2-hbh | 446 / 514 | 0.53 | 8.256 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | t2-hbh | 2837 / 2838 | 0.22 | 8.338 | 0 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | t2-hbh | 3827 / 4520 | 2.65 | 8.128 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | t2-rank256 | 349 / 353 | 0.56 | 9.317 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | t2-rank256 | 2894 / 2901 | 0.22 | 9.371 | 0 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | t2-rank256 | 3597 / 4313 | 2.74 | 9.215 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | splitzip | 359 / 361 | 0.56 | 12.042 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | splitzip | 3101 / 3119 | 0.21 | 12.049 | 0 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | splitzip | 3826 / 4708 | 2.59 | 12.042 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | raw | 225 / 227 | 0.61 | 16.000 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | raw | 2326 / 2333 | 0.25 | 16.000 | 0 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | raw | 2745 / 2987 | 3.48 | 16.000 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | ours-hbh | 262 / 264 | 0.59 | 8.784 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | ours-hbh | 2255 / 2339 | 0.25 | 8.701 | 0 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | ours-hbh | 2808 / 3006 | 3.46 | 8.700 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | t2-hbh | 272 / 397 | 0.58 | 8.256 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | t2-hbh | 2349 / 2358 | 0.25 | 8.338 | 0 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | t2-hbh | 2978 / 3421 | 3.19 | 8.128 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | t2-rank256 | 249 / 251 | 0.60 | 9.317 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | t2-rank256 | 2138 / 2149 | 0.26 | 9.371 | 0 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | t2-rank256 | 2807 / 3016 | 3.46 | 9.215 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | splitzip | 233 / 238 | 0.60 | 12.042 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | splitzip | 2097 / 2100 | 0.26 | 12.049 | 0 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | splitzip | 2688 / 2822 | 3.60 | 12.042 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

14 in the engine logs, 0 inside a measured point.

- 22:30:30 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 22:30:30 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 22:30:35 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 22:30:35 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 22:30:36 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 22:30:36 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 22:30:36 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 22:30:36 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 22:30:36 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 22:30:36 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 22:30:36 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 22:30:36 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 22:30:36 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 22:30:36 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

