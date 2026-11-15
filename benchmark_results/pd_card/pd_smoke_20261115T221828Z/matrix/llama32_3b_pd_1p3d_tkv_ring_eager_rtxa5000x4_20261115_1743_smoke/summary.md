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
| raw | inf | 1.86 | 0.70 | 2382 / 2790 | 11.9 | 16.000 | 1.1 / 2.6 | 463.1 | 3.9 |
| t2-hbh | inf | 2.01 | 1.00 | 1985 / 2222 | 14.2 | 7.860 | 43.9 / 148.4 | 203.0 | 3.9 |
| t2-rank256 | inf | 2.17 | 1.35 | 1896 / 2170 | 12.1 | 8.867 | 31.3 / 22.4 | 233.4 | 4.0 |

