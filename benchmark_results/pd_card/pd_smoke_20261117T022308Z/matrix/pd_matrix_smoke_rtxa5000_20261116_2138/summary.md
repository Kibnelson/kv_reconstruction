# PD matrix smoke on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 2 runs: `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261116_2138_smoke`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261116_2143_smoke_partial`.

Runs that did not finish: `SMOKE=1 LAYOUT=colo4 TIER_CODECS=raw,exans,ibp,ours-hbh,t2-hbh,t2-rank256`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory | splitzip | 4/4 | 4/4 | yes | 12.044 |
| 1p3d, host memory | exans | 4/4 | 4/4 | yes | 10.923 |
| 1p3d, host memory | adacode | 4/4 | 4/4 | yes | 11.371 |
| 1p3d, host memory | exactkv | 4/4 | 4/4 | yes | 12.629 |
| 1p3d, host memory | ours-hbh | 4/4 | 4/4 | yes | 9.053 |
| 1p3d, host memory | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| 1p3d, host memory | aquakv-2 (lossy) | lossy (KV error 10.50%) | 2/4 | yes | 2.126 |
| colo4, host memory tier | raw (c8) | 18/18 chunks | - | yes | 16.000 |
| colo4, host memory tier | exans (c8) | 14/14 chunks | - | yes | 11.023 |
| colo4, host memory tier | ibp (c8) | 0/0 chunks | - | NO | - |

## 1p3d, host memory: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.87 (0.70); 2404 / 2815 | 11.5 |
| splitzip | 12.042 | 1.93 (0.72); 2375 / 2670 | 11.6 |
| exans | 10.884 | 2.06 (1.03); 2090 / 2408 | 11.6 |
| adacode | 11.372 | 2.08 (0.78); 2053 / 2372 | 11.6 |
| exactkv | 12.590 | 2.01 (0.75); 2189 / 2514 | 11.6 |
| ours-hbh | 8.684 | 2.12 (1.06); 2021 / 2276 | 12.6 |
| t2-hbh | 8.146 | 2.14 (1.07); 2001 / 2233 | 12.6 |
| aquakv-2 (lossy) | 2.130 | 1.99 (1.00); 2033 / 2540 | 11.7 |

Best by goodput, then TTFT p50: rate inf: t2-hbh (goodput 1.07/s, TTFT p50 2001 ms).

## CPU tier, colo4: 8 clients

Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exans | 5.87 | 67 / 143 / 191 | 11.5 | 67.5 / 9.9 | 8 | 2464 | 35.6 (1.4 / 14.5) | 11.023 | 69 | 85 |
| raw | 5.87 | 85 / 138 / 163 | 11.9 | 66.8 / 13.5 | 9 | 3536 | 51.8 (2.4 / 14.8) | 16.000 | 77 | 89 |
| ibp | 4.71 | 93 / 168 / 204 | 13.0 | - / - | 36679 | 0 | - (- / -) | - | - | 79 |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | t2-hbh | 10 | inf | 0 | 8 | {'news': 1, 'code': 2, 'wikitext': 2, 'exam': 1, 'math': 2} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |

## Note on the lossy reference

aquakv-<b> is AQUA-KV (arXiv 2501.19392, the authors' algorithm on the GPU), lossy: the KV it delivers differs from the original (its error is in the correctness table). It is here for speed and size only, beside the bit-exact codecs.

