# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

Prefix hits: vLLM's own counters (GPU cache / external, i.e. the tier: ours or LMCache's), as % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed.

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 8 | 5.69 | 81 / 136 / 278 | 11.8 | 64.0 / 16.3 | 9 | 4400 | 43.5 (2.9 / 17.1) | 16.000 | 75 | 85 | 21 |
| exans | 8 | 5.98 | 68 / 152 / 548 | 11.7 | 68.3 / 6.3 | 5 | 1408 | 43.4 (1.8 / 15.8) | 11.024 | 69 | 83 | None |
| ours-hbh | 8 | 5.19 | 70 / 262 / 687 | 11.6 | 65.2 / 15.0 | 9 | 4080 | 366.1 (1.7 / 345.8) | 8.393 | 76 | 90 | 0 |
| t2-hbh | 8 | 5.42 | 65 / 217 / 914 | 11.7 | 66.7 / 8.1 | 5 | 1872 | 317.5 (1.2 / 297.3) | 8.381 | 73 | 82 | None |
| t2-rank256 | 8 | 5.48 | 68 / 180 / 355 | 11.7 | 66.3 / 14.5 | 10 | 3856 | 63.0 (1.6 / 47.2) | 9.448 | 75 | 89 | 0 |

Codecs not run (did not load on every engine): ibp (ModuleNotFoundError("No module named 'ibp'"))

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 17:40:22 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:40:22 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:40:22 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:40:22 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:40:22 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:40:22 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:40:22 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:40:22 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:40:22 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:40:22 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:40:22 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:40:22 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex

