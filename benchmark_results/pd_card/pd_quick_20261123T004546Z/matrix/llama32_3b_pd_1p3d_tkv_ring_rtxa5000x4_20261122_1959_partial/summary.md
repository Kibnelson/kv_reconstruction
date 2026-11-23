# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |
| ours-hbh | 2/4 (FAILED) | 3/4 | 9.110 |
| t2-hbh | 2/4 (FAILED) | 3/4 | 8.562 |
| t2-rank256 | 2/4 (FAILED) | 3/4 | 9.619 |
| splitzip | 4/4 | 3/4 | 12.043 |

## The published workloads (TTFT p50 / p99 in ms; requests per second)

| Link | Prompt tokens | At once | Out | Requests | Families | Codec | TTFT p50 / p99 | Req/s | Bits | Failed transfers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2048 | 1 | 128 | 4 | sz1 | raw | 394 / 396 | 0.55 | 16.000 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | raw | 3531 / 3541 | 0.19 | 16.000 | 0 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | raw | 4268 / 5441 | 2.32 | 16.000 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | ours-hbh | 352 / 356 | 0.56 | 8.784 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | ours-hbh | 4632 / 4674 | 0.16 | 8.701 | 2 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | ours-hbh | 3489 / 4151 | 2.82 | 8.700 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | t2-hbh | 501 / 505 | 0.52 | 8.256 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | t2-hbh | 4630 / 4645 | 0.16 | 8.338 | 2 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | t2-hbh | 3822 / 4558 | 2.64 | 8.128 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | t2-rank256 | 346 / 352 | 0.57 | 9.317 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | t2-rank256 | 4729 / 4755 | 0.15 | 9.371 | 2 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | t2-rank256 | 3505 / 4220 | 2.79 | 9.215 | 0 |
| 10 | 2048 | 1 | 128 | 4 | sz1 | splitzip | 361 / 362 | 0.56 | 12.042 | 0 |
| 10 | 16384 | 1 | 128 | 2 | sz1 | splitzip | 3115 / 3117 | 0.21 | 12.049 | 0 |
| 10 | 2048 | 16 | 128 | 16 | sz16 | splitzip | 3851 / 4739 | 2.58 | 12.041 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | raw | 224 / 227 | 0.61 | 16.000 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | raw | 2330 / 2336 | 0.25 | 16.000 | 0 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | raw | 2735 / 2976 | 3.49 | 16.000 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | ours-hbh | 259 / 262 | 0.59 | 8.784 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | ours-hbh | 3954 / 3955 | 0.18 | 8.701 | 2 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | ours-hbh | 2794 / 2999 | 3.47 | 8.700 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | t2-hbh | 271 / 274 | 0.59 | 8.256 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | t2-hbh | 4040 / 4060 | 0.17 | 8.338 | 2 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | t2-hbh | 3005 / 3457 | 3.19 | 8.128 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | t2-rank256 | 251 / 253 | 0.60 | 9.317 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | t2-rank256 | 3972 / 3983 | 0.18 | 9.371 | 2 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | t2-rank256 | 2829 / 3059 | 3.42 | 9.215 | 0 |
| 100 | 2048 | 1 | 128 | 4 | sz1 | splitzip | 234 / 234 | 0.61 | 12.042 | 0 |
| 100 | 16384 | 1 | 128 | 2 | sz1 | splitzip | 2078 / 2089 | 0.26 | 12.049 | 0 |
| 100 | 2048 | 16 | 128 | 16 | sz16 | splitzip | 2696 / 2787 | 3.63 | 12.041 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

14 in the engine logs, 0 inside a measured point.

- 20:01:58 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _enc_4bit_quad64_count2_chunks. This causes a latency spike; c
- 20:01:58 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _write_escapes_chunk. This causes a latency spike; consider ex
- 20:02:03 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:02:04 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:02:05 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:02:05 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:02:05 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 20:02:05 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:02:05 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:02:05 engine_decode2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 20:02:05 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 20:02:05 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 20:02:05 engine_decode3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 20:02:05 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

