# PD matrix smoke on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 1 runs: `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261117_0828_smoke`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| colo4, host memory tier | raw (c8) | 23/23 chunks | - | yes | 16.000 |
| colo4, host memory tier | exans (c8) | 18/18 chunks | - | yes | 11.024 |
| colo4, host memory tier | ibp (c8) | 25/25 chunks | - | yes | 16.125 |
| colo4, host memory tier | ours-hbh (c8) | 25/25 chunks | - | yes | 8.350 |
| colo4, host memory tier | t2-hbh (c8) | 23/23 chunks | - | yes | 8.398 |
| colo4, host memory tier | t2-rank256 (c8) | 21/21 chunks | - | yes | 9.393 |

## CPU tier, colo4: 8 clients

Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2-rank256 | 6.13 | 66 / 162 / 243 | 11.6 | 63.8 / 15.0 | 8 | 3936 | 88.2 (2.1 / 70.8) | 9.393 | 73 | 84 |
| t2-hbh | 5.06 | 68 / 429 / 850 | 11.7 | 63.6 / 17.1 | 9 | 4640 | 401.6 (1.7 / 382.2) | 8.398 | 79 | 86 |
| exans | 6.02 | 69 / 163 / 197 | 11.7 | 67.2 / 10.7 | 11 | 2800 | 34.5 (1.3 / 14.5) | 11.024 | 61 | 86 |
| ours-hbh | 4.87 | 71 / 523 / 816 | 11.8 | 63.0 / 17.6 | 10 | 4960 | 389.8 (1.5 / 371.1) | 8.350 | 78 | 87 |
| raw | 6.09 | 82 / 140 / 272 | 11.7 | 64.7 / 15.9 | 9 | 4176 | 54.1 (3.0 / 17.6) | 16.000 | 71 | 87 |
| ibp | 5.21 | 104 / 173 / 214 | 12.9 | 65.4 / 17.7 | 10 | 4960 | 65.2 (0.0 / 48.2) | 16.125 | 78 | 89 |

