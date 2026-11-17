# PD matrix smoke on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 2 runs: `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261116_1934_smoke_partial`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261116_1939_smoke`.

Runs that did not finish: `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,splitzip,exans,adacode,exactkv,ours-hbh,t2-hbh,aquakv-2`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory | splitzip | 4/4 | 4/4 | yes | 12.044 |
| 1p3d, host memory | exans | 4/4 | 4/4 | yes | 10.923 |
| 1p3d, host memory | adacode | 4/4 | 4/4 | yes | 11.371 |
| 1p3d, host memory | exactkv | 4/4 | 4/4 | yes | 12.629 |
| 1p3d, host memory | ours-hbh | 4/4 | 4/4 | yes | 9.053 |
| 1p3d, host memory | t2-hbh | 3/4 | 4/4 | NO | 8.627 |
| 1p3d, host memory | aquakv-2 (lossy) | lossy (KV error 10.50%) | 2/4 | yes | 2.126 |
| colo4, host memory tier | raw (c8) | 18/18 chunks | - | yes | 16.000 |
| colo4, host memory tier | exans (c8) | 17/17 chunks | - | yes | 11.022 |
| colo4, host memory tier | ours-hbh (c8) | 9/9 chunks | - | yes | 8.338 |
| colo4, host memory tier | t2-hbh (c8) | 16/16 chunks | - | yes | 8.398 |
| colo4, host memory tier | t2-rank256 (c8) | 19/19 chunks | - | yes | 9.448 |

## 1p3d, host memory: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.87 (0.70); 2392 / 2808 | 11.5 |
| splitzip | 12.042 | 1.94 (0.73); 2351 / 2644 | 11.6 |
| exans | 10.884 | 2.06 (0.77); 2083 / 2406 | 11.6 |
| adacode | 11.372 | 2.07 (0.77); 2079 / 2398 | 11.6 |
| exactkv | 12.590 | 2.02 (0.76); 2161 / 2487 | 11.6 |
| ours-hbh | 8.684 | 2.12 (1.06); 2021 / 2273 | 12.6 |
| t2-hbh | 8.146 | 2.14 (1.07); 1977 / 2234 | 12.7 |
| aquakv-2 (lossy) | 2.130 | 1.78 (0.67); 2312 / 3023 | 11.9 |

Best by goodput, then TTFT p50: rate inf: t2-hbh (goodput 1.07/s, TTFT p50 1977 ms).

## CPU tier, colo4: 8 clients

Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2-rank256 | 6.04 | 65 / 150 / 292 | 11.8 | 66.5 / 12.9 | 10 | 3472 | 54.7 (1.3 / 36.9) | 9.448 | 71 | 87 |
| ours-hbh | 5.35 | 67 / 170 / 1784 | 11.6 | 65.6 / 8.7 | 4 | 1968 | 378.5 (1.6 / 366.6) | 8.338 | 85 | 80 |
| exans | 5.81 | 68 / 150 / 209 | 11.6 | 65.5 / 12.5 | 8 | 3408 | 39.6 (1.7 / 21.0) | 11.022 | 78 | 86 |
| t2-hbh | 5.13 | 74 / 268 / 555 | 11.7 | 66.2 / 11.5 | 7 | 2944 | 350.1 (1.7 / 331.4) | 8.398 | 72 | 85 |
| raw | 5.85 | 81 / 145 / 181 | 11.9 | 66.6 / 12.6 | 9 | 3280 | 48.2 (2.4 / 14.5) | 16.000 | 71 | 87 |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | t2-hbh | 10 | inf | 0 | 8 | {'news': 1, 'code': 2, 'wikitext': 2, 'math': 2, 'exam': 1} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |

## Codecs that did not load

- ibp: ImportError('IBP left out: the CUDA 13 toolkit did not install (setup: cuda13_installer.log)')

## Note on the lossy reference

aquakv-<b> is AQUA-KV (arXiv 2501.19392, the authors' algorithm on the GPU), lossy: the KV it delivers differs from the original (its error is in the correctness table). It is here for speed and size only, beside the bit-exact codecs.

