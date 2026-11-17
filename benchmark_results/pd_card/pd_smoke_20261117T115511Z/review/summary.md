# PD matrix smoke on gpu-grad-01, job 30322 (completed), reviewed from the kept results

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 1 runs: `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261117_0708_smoke`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| colo4, host memory tier | raw (c8) | 19/19 chunks | - | yes | 16.000 |
| colo4, host memory tier | exans (c8) | 24/24 chunks | - | yes | 11.022 |
| colo4, host memory tier | ibp (c8) | 0/0 chunks | - | NO | - |
| colo4, host memory tier | ours-hbh (c8) | 12/12 chunks | - | yes | 8.372 |
| colo4, host memory tier | t2-hbh (c8) | 7/7 chunks | - | yes | 8.375 |
| colo4, host memory tier | t2-rank256 (c8) | 19/19 chunks | - | yes | 9.396 |

## CPU tier, colo4: 8 clients

Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2-hbh | 5.69 | 65 / 172 / 1268 | 11.7 | 67.7 / 5.2 | 3 | 1120 | 361.7 (1.8 / 340.5) | 8.375 | 62 | 79 |
| exans | 5.97 | 70 / 137 / 345 | 11.6 | 63.9 / 16.7 | 10 | 4704 | 41.0 (1.9 / 18.8) | 11.022 | 77 | 89 |
| t2-rank256 | 5.76 | 71 / 150 / 228 | 11.6 | 64.1 / 14.2 | 8 | 3856 | 68.9 (1.9 / 55.5) | 9.396 | 79 | 85 |
| raw | 5.78 | 73 / 133 / 226 | 12.0 | 67.1 / 12.7 | 10 | 3376 | 36.8 (2.2 / 12.3) | 16.000 | 69 | 89 |
| ours-hbh | 5.46 | 76 / 248 / 543 | 11.5 | 65.9 / 8.3 | 7 | 1968 | 280.7 (1.2 / 259.1) | 8.372 | 64 | 85 |
| ibp | 5.02 | 99 / 190 / 287 | 12.8 | 63.2 / 11.0 | 6 | 0 | - (- / -) | - | - | 92 |

