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
| raw | inf | 1.46 | 0.00 | 3545 / 3968 | 11.9 | 16.000 | 1.1 / 2.6 | 469.8 | 3.7 |
| t2-hbh | inf | 2.12 | 1.06 | 2009 / 2248 | 13.1 | 8.146 | 42.0 / 151.6 | 211.5 | 3.9 |
| t2-rank256 | inf | 2.14 | 1.33 | 1973 / 2224 | 12.1 | 9.230 | 29.6 / 20.2 | 247.5 | 3.9 |

