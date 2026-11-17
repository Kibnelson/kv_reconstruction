# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

Prefix hits: vLLM's own counters (GPU cache / external, i.e. the tier: ours or LMCache's), as % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed.

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 8 | 5.78 | 73 / 133 / 226 | 12.0 | 67.1 / 12.7 | 10 | 3376 | 36.8 (2.2 / 12.3) | 16.000 | 69 | 89 | None |
| exans | 8 | 5.97 | 70 / 137 / 345 | 11.6 | 63.9 / 16.7 | 10 | 4704 | 41.0 (1.9 / 18.8) | 11.022 | 77 | 89 | 16 |
| ibp | 8 | 5.02 | 99 / 190 / 287 | 12.8 | 63.2 / 11.0 | 6 | 0 | - (- / -) | - | - | 92 | None |
| ours-hbh | 8 | 5.46 | 76 / 248 / 543 | 11.5 | 65.9 / 8.3 | 7 | 1968 | 280.7 (1.2 / 259.1) | 8.372 | 64 | 85 | 0 |
| t2-hbh | 8 | 5.69 | 65 / 172 / 1268 | 11.7 | 67.7 / 5.2 | 3 | 1120 | 361.7 (1.8 / 340.5) | 8.375 | 62 | 79 | None |
| t2-rank256 | 8 | 5.76 | 71 / 150 / 228 | 11.6 | 64.1 / 14.2 | 8 | 3856 | 68.9 (1.9 / 55.5) | 9.396 | 79 | 85 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 07:09:50 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 07:09:50 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 07:09:50 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 07:09:50 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 07:09:50 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 07:09:50 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 07:09:50 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 07:09:50 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 07:09:51 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 07:09:51 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 07:09:51 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 07:09:51 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

