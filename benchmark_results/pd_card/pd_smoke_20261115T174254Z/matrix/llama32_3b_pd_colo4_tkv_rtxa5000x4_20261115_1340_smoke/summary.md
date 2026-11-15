# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 4 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## CPU tier (colocated engines, multi-turn conversations)

| Tier codec | Clients | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Tier loads | Tokens loaded | Load ms (copy / decode) | Bits stored | Chunks stored | Chunks in tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 8 | 5.74 | 87 / 167 / 256 | 11.9 | 5 | 1040 | 43.0 (1.7 / 16.0) | 27.569 | 112 | 27 |
| exans | 8 | 6.00 | 70 / 185 / 272 | 12.0 | 2 | 416 | 47.9 (1.2 / 11.2) | 20.351 | 107 | None |
| ours-hbh | 8 | 5.58 | 78 / 203 / 401 | 11.9 | 2 | 608 | 309.3 (1.2 / 289.1) | 13.991 | 104 | 3 |
| t2-hbh | 8 | 5.27 | 67 / 251 / 673 | 11.8 | 4 | 1936 | 355.5 (1.5 / 334.9) | 9.944 | 110 | None |

Codecs not run (did not load on every engine): ibp (ModuleNotFoundError("No module named 'ibp'"))

