# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

Prefix hits: vLLM's own counters (GPU cache / external, i.e. the tier: ours or LMCache's), as % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed.

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 8 | 5.62 | 87 / 156 / 205 | 11.9 | 64.1 / 20.5 | 10 | 5936 | 60.0 (3.4 / 21.3) | 16.000 | 80 | 90 | 23 |
| exans | 8 | 5.62 | 68 / 153 / 446 | 11.6 | 67.6 / 5.0 | 5 | 1104 | 36.0 (1.3 / 14.7) | 11.021 | 54 | 81 | None |
| ours-hbh | 8 | 5.31 | 72 / 404 / 718 | 11.7 | 65.4 / 15.2 | 9 | 4128 | 385.0 (1.5 / 364.4) | 8.373 | 73 | 88 | 0 |
| t2-hbh | 8 | 5.16 | 70 / 250 / 679 | 11.7 | 65.2 / 12.9 | 8 | 3456 | 333.1 (1.3 / 315.1) | 8.386 | 79 | 86 | None |
| t2-rank256 | 8 | 5.88 | 72 / 153 / 340 | 11.8 | 64.8 / 13.7 | 7 | 3568 | 65.5 (1.7 / 47.8) | 9.401 | 77 | 86 | 0 |

Codecs not run (did not load on every engine): ibp (ModuleNotFoundError("No module named 'ibp'"))

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 17:32:38 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:32:38 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:32:38 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:32:38 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:32:38 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:32:38 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:32:38 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:32:38 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:32:38 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 17:32:38 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 17:32:38 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 17:32:38 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

