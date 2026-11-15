# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| t2-hbh | 4/4 | 4/4 | 8.489 |
| t2-rank256 | 4/4 | 4/4 | 9.578 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.85 | 0.69 | 2395 / 2816 | 11.9 | 16.000 | 1.1 / 2.6 | 469.8 | 3.7 |
| t2-hbh | inf | 3.06 | 3.06 | 877 / 1079 | 12.5 | 8.146 | 40.7 / 147.5 | 207.2 | 4.0 |
| t2-rank256 | inf | 3.11 | 3.11 | 778 / 1051 | 12.1 | 9.230 | 28.3 / 18.9 | 250.9 | 3.8 |

