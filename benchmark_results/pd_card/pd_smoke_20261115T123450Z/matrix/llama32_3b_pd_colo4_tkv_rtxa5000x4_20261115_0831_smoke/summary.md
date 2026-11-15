# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits stored | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 4 | 2.86 | 100 / 265 / 276 | 11.1 | 0 | 0 | - (- / -) | - | 44 | None |
| exans | 4 | 2.91 | 84 / 200 / 217 | 11.3 | 0 | 0 | - (- / -) | - | 44 | 0 |
| ours-hbh | 4 | 3.04 | 87 / 164 / 249 | 11.4 | 0 | 0 | - (- / -) | - | 45 | None |
| t2-hbh | 4 | 3.11 | 112 / 154 / 157 | 11.2 | 0 | 0 | - (- / -) | - | 43 | None |

Codecs not run (did not load on every engine): ibp (ModuleNotFoundError("No module named 'ibp'"))

