# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

Prefix hits: vLLM's own counters (GPU cache / external, i.e. the tier: ours or LMCache's), as % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed.

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 8 | 5.85 | 81 / 145 / 181 | 11.9 | 66.6 / 12.6 | 9 | 3280 | 48.2 (2.4 / 14.5) | 16.000 | 71 | 87 | 19 |
| exans | 8 | 5.81 | 68 / 150 / 209 | 11.6 | 65.5 / 12.5 | 8 | 3408 | 39.6 (1.7 / 21.0) | 11.022 | 78 | 86 | None |
| ours-hbh | 8 | 5.35 | 67 / 170 / 1784 | 11.6 | 65.6 / 8.7 | 4 | 1968 | 378.5 (1.6 / 366.6) | 8.338 | 85 | 80 | 0 |
| t2-hbh | 8 | 5.13 | 74 / 268 / 555 | 11.7 | 66.2 / 11.5 | 7 | 2944 | 350.1 (1.7 / 331.4) | 8.398 | 72 | 85 | None |
| t2-rank256 | 8 | 6.04 | 65 / 150 / 292 | 11.8 | 66.5 / 12.9 | 10 | 3472 | 54.7 (1.3 / 36.9) | 9.448 | 71 | 87 | 0 |

Codecs not run (did not load on every engine): ibp (ImportError('IBP left out: the CUDA 13 toolkit did not install (setup: cuda13_installer.log)'))

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 19:40:08 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:40:08 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:40:08 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:40:08 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:40:08 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:40:08 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:40:08 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:40:08 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:40:08 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 19:40:08 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 19:40:08 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 19:40:08 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

