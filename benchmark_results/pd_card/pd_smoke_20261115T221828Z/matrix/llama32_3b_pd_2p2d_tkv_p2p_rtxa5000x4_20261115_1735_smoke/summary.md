# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 2 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |
| ours-hbh | 4/4 | 3/4 | 9.053 |
| t2-hbh | 4/4 | 3/4 | 8.489 |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 3.47 | 3.47 | 749 / 762 | 12.1 | 16.000 | 1.3 / 2.7 | 13.4 | 117.4 |
| ours-hbh | inf | 2.80 | 2.80 | 1313 / 1318 | 12.1 | 8.684 | 32.5 / 150.4 | 5.8 | 128.7 |
| t2-hbh | inf | 2.76 | 2.76 | 1354 / 1356 | 12.1 | 8.146 | 45.1 / 158.3 | 5.8 | 123.8 |

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 6 inside a measured point (those points' latencies include the compile):

- 17:36:49 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:36:49 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:36:49 engine_decode1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:36:49 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:36:49 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:36:49 engine_prefill1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:36:50 engine_decode2.log: sweep_raw_nativegbps_rinf - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:36:50 engine_decode2.log: sweep_raw_nativegbps_rinf - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:36:50 engine_decode2.log: sweep_raw_nativegbps_rinf - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:36:50 engine_prefill2.log: sweep_raw_nativegbps_rinf - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:36:50 engine_prefill2.log: sweep_raw_nativegbps_rinf - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:36:50 engine_prefill2.log: sweep_raw_nativegbps_rinf - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex

