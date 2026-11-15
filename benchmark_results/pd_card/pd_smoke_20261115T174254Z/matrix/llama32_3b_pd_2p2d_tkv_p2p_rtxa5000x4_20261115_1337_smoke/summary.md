# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 2 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| ours-hbh | 4/4 | 4/4 | 9.053 |
| t2-hbh | 4/4 | 4/4 | 8.489 |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.69 | 0.00 | 3110 / 3204 | 12.1 | 16.000 | 1.4 / 2.8 | 12.6 | 125.2 |
| ours-hbh | inf | 2.80 | 2.80 | 1319 / 1332 | 12.1 | 8.684 | 32.2 / 152.2 | 5.5 | 137.1 |
| t2-hbh | inf | 2.82 | 2.82 | 1293 / 1297 | 12.1 | 8.146 | 42.1 / 146.4 | 5.5 | 136.9 |

