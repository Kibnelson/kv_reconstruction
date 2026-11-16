# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

Prefix hits: vLLM's own counters (GPU cache / external, i.e. the tier: ours or LMCache's), as % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed.

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 8 | 5.73 | 82 / 215 / 559 | 11.9 | 63.5 / 15.1 | 9 | 3920 | 52.1 (3.1 / 15.6) | 16.000 | 77 | 83 | 10 |
| exans | 8 | 5.97 | 77 / 141 / 188 | 11.6 | 63.1 / 19.3 | 11 | 5632 | 43.4 (2.0 / 22.0) | 11.021 | 79 | 87 | None |
| ours-hbh | 8 | 5.75 | 73 / 246 / 583 | 11.7 | 67.7 / 8.6 | 7 | 2112 | 262.0 (1.3 / 245.0) | 8.401 | 75 | 86 | 0 |
| t2-hbh | 8 | 5.60 | 68 / 372 / 517 | 11.7 | 65.4 / 11.3 | 8 | 2880 | 302.1 (1.5 / 283.1) | 8.384 | 70 | 85 | None |
| t2-rank256 | 8 | 5.88 | 72 / 160 / 182 | 11.8 | 64.0 / 19.2 | 11 | 5648 | 58.6 (1.6 / 38.8) | 9.433 | 85 | 88 | 0 |

Codecs not run (did not load on every engine): ibp (ModuleNotFoundError("No module named 'ibp'"))

## Kernels compiled while serving (vLLM's JIT warnings)

12 in the engine logs, 0 inside a measured point.

- 14:58:01 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:58:01 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:58:01 engine_engine1.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:58:01 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:58:01 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:58:01 engine_engine2.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:58:01 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:58:01 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:58:01 engine_engine3.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 14:58:01 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 14:58:01 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 14:58:01 engine_engine4.log: outside the points - Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex

