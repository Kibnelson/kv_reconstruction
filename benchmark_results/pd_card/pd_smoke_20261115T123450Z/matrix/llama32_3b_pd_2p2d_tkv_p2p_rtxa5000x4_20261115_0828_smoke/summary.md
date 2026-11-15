# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 2 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |
| ours-hbh | 4/4 | 4/4 | 9.053 |
| t2-hbh | 4/4 | 4/4 | 8.490 |

## Link native Gb/s

| Codec | Rate | Req/s | Goodput | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Bits | Encode / install (ms) | Link time (ms) | Achieved Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | inf | 1.15 | 0.14 | 5363 / 5392 | 12.1 | 16.000 | 5.0 / 2.7 | 92.1 | 73.7 |
| ours-hbh | inf | 3.27 | 3.27 | 888 / 912 | 12.1 | 8.684 | 32.0 / 151.1 | 6.5 | 102.1 |
| t2-hbh | inf | 2.70 | 2.70 | 1364 / 1433 | 12.0 | 8.146 | 41.0 / 147.3 | 6.4 | 128.1 |

