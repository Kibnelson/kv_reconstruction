# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| t2-hbh | 4/4 | 4/4 | 8.189 |
| t2-rank256 | 4/4 | 4/4 | 9.199 |

## Link 10 Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.87 | 0.70 | 2382 / 2790 | 11.8 | 16.000 | 1.1 / 2.6 | 469.8 | 3.7 |
| t2-hbh | inf | 2.15 | 1.08 | 1963 / 2193 | 12.9 | 7.860 | 46.1 / 145.1 | 202.0 | 4.0 |
| t2-rank256 | inf | 1.91 | 1.19 | 1907 / 2181 | 14.9 | 8.867 | 31.4 / 23.0 | 234.4 | 4.0 |

