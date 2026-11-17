# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

Prefix hits: vLLM's own counters (GPU cache / external, i.e. the tier: ours or LMCache's), as % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed.

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 8 | 6.09 | 82 / 140 / 272 | 11.7 | 64.7 / 15.9 | 9 | 4176 | 54.1 (3.0 / 17.6) | 16.000 | 71 | 87 | None |
| exans | 8 | 6.02 | 69 / 163 / 197 | 11.7 | 67.2 / 10.7 | 11 | 2800 | 34.5 (1.3 / 14.5) | 11.024 | 61 | 86 | 14 |
| ibp | 8 | 5.21 | 104 / 173 / 214 | 12.9 | 65.4 / 17.7 | 10 | 4960 | 65.2 (0.0 / 48.2) | 16.125 | 78 | 89 | None |
| ours-hbh | 8 | 4.87 | 71 / 523 / 816 | 11.8 | 63.0 / 17.6 | 10 | 4960 | 389.8 (1.5 / 371.1) | 8.350 | 78 | 87 | 0 |
| t2-hbh | 8 | 5.06 | 68 / 429 / 850 | 11.7 | 63.6 / 17.1 | 9 | 4640 | 401.6 (1.7 / 382.2) | 8.398 | 79 | 86 | 23 |
| t2-rank256 | 8 | 6.13 | 66 / 162 / 243 | 11.6 | 63.8 / 15.0 | 8 | 3936 | 88.2 (2.1 / 70.8) | 9.393 | 73 | 84 | 0 |

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 08:30:29 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 08:30:29 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 08:30:29 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 08:30:29 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 08:30:30 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 08:30:30 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 08:30:30 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 08:30:30 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 08:30:30 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 08:30:30 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 08:30:30 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 08:30:30 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

