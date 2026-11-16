# PD matrix smoke on gpu-grad-01, job 30293 (completed), reviewed from the kept results

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 7 runs: `llama32_3b_pd_1p3d_tkv_ring_eager_rtxa5000x4_20261116_1735_smoke`, `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261116_1721_smoke`, `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261116_1725_smoke`, `llama32_3b_pd_2p2d_nixl_allgpus_rtxa5000x4_20261116_1730_smoke`, `llama32_3b_pd_2p2d_tkv_p2p_rtxa5000x4_20261116_1728_smoke`, `llama32_3b_pd_colo4_lmcache_rtxa5000x4_20261116_1734_smoke`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261116_1731_smoke`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory, eager | t2-hbh | 4/4 | 4/4 | yes | 8.189 |
| 1p3d, host memory, eager | t2-rank256 | 4/4 | 4/4 | yes | 9.199 |
| 1p3d, host memory | raw | 4/4 | 3/4 | yes | 16.000 |
| 1p3d, host memory | splitzip | 4/4 | 3/4 | yes | 12.044 |
| 1p3d, host memory | exans | 4/4 | 4/4 | yes | 10.923 |
| 1p3d, host memory | adacode | 4/4 | 3/4 | yes | 11.371 |
| 1p3d, host memory | ours-hbh | 4/4 | 3/4 | yes | 9.053 |
| 1p3d, host memory | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| 1p3d, host memory | raw | 4/4 | 3/4 | yes | 16.000 |
| 1p3d, host memory | nvcomp-ans | 4/4 | 3/4 | yes | 13.369 |
| 1p3d, host memory | nvcomp-zstd | 4/4 | 3/4 | yes | 12.896 |
| 1p3d, host memory | nvcomp-bitcomp | 4/4 | 3/4 | yes | 13.790 |
| 2p2d, NIXL (vLLM), every GPU visible | nixl | n/a | 4/4 | yes | - |
| 2p2d, GPU to GPU | raw | 4/4 | 3/4 | yes | 16.000 |
| 2p2d, GPU to GPU | ours-hbh | 4/4 | 3/4 | yes | 9.053 |
| 2p2d, GPU to GPU | t2-hbh | 4/4 | 3/4 | yes | 8.489 |
| colo4, LMCache (vLLM) tier | lmcache (c8) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | raw (c8) | 29/29 chunks | - | yes | 16.000 |
| colo4, host memory tier | exans (c8) | 8/8 chunks | - | yes | 11.021 |
| colo4, host memory tier | ours-hbh (c8) | 22/22 chunks | - | yes | 8.373 |
| colo4, host memory tier | t2-hbh (c8) | 17/17 chunks | - | yes | 8.386 |
| colo4, host memory tier | t2-rank256 (c8) | 18/18 chunks | - | yes | 9.401 |

## 1p3d, host memory: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.88 (0.71); 2389 / 2797 | 11.5 |
| splitzip | 12.042 | 1.94 (0.73); 2350 / 2642 | 11.6 |
| exans | 10.884 | 2.03 (0.76); 2146 / 2465 | 11.6 |
| adacode | 11.372 | 2.07 (0.78); 2066 / 2393 | 11.6 |
| ours-hbh | 8.684 | 2.13 (1.06); 2015 / 2268 | 12.6 |
| t2-hbh | 8.146 | 2.14 (1.07); 2001 / 2243 | 12.7 |
| nvcomp-ans | 13.365 | 1.99 (0.74); 2212 / 2565 | 11.6 |
| nvcomp-zstd | 12.892 | 1.84 (0.23); 2521 / 2895 | 11.7 |
| nvcomp-bitcomp | 13.786 | 1.96 (0.74); 2274 / 2614 | 11.6 |

Best by goodput, then TTFT p50: rate inf: t2-hbh (goodput 1.07/s, TTFT p50 2001 ms).

## 1p3d, host memory, eager engines: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.86 (0.70); 2373 / 2793 | 11.8 |
| t2-hbh | 7.860 | 2.14 (1.07); 2031 / 2204 | 12.4 |
| t2-rank256 | 8.867 | 2.17 (1.36); 1909 / 2166 | 12.0 |

Best by goodput, then TTFT p50: rate inf: t2-rank256 (goodput 1.36/s, TTFT p50 1909 ms).

## 2p2d, GPU to GPU: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 3.52 (3.52); 728 / 743 | 12.0 |
| ours-hbh | 8.684 | 2.82 (2.82); 1281 / 1307 | 12.0 |
| t2-hbh | 8.146 | 2.80 (2.80); 1310 / 1326 | 12.0 |

Best by goodput, then TTFT p50: rate inf: raw (goodput 3.52/s, TTFT p50 728 ms).

## 2p2d, NIXL (vLLM), every GPU visible: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| nixl | 16.000 | 2.36 (2.36); 1878 / 1887 | 11.8 |

Best by goodput, then TTFT p50: rate inf: nixl (goodput 2.36/s, TTFT p50 1878 ms).

## CPU tier, colo4: 8 clients

Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exans | 5.62 | 68 / 153 / 446 | 11.6 | 67.6 / 5.0 | 5 | 1104 | 36.0 (1.3 / 14.7) | 11.021 | 54 | 81 |
| t2-hbh | 5.16 | 70 / 250 / 679 | 11.7 | 65.2 / 12.9 | 8 | 3456 | 333.1 (1.3 / 315.1) | 8.386 | 79 | 86 |
| t2-rank256 | 5.88 | 72 / 153 / 340 | 11.8 | 64.8 / 13.7 | 7 | 3568 | 65.5 (1.7 / 47.8) | 9.401 | 77 | 86 |
| ours-hbh | 5.31 | 72 / 404 / 718 | 11.7 | 65.4 / 15.2 | 9 | 4128 | 385.0 (1.5 / 364.4) | 8.373 | 73 | 88 |
| LMCache CPU offload (raw) | 5.56 | 76 / 219 / 340 | 11.5 | 67.9 / 5.0 | - | - | - (- / -) | - | - | - |
| raw | 5.62 | 87 / 156 / 205 | 11.9 | 64.1 / 20.5 | 10 | 5936 | 60.0 (3.4 / 21.3) | 16.000 | 80 | 90 |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | t2-hbh | 10 | inf | 8 | 0 | {'news': 1, 'code': 2, 'wikitext': 2, 'math': 2, 'exam': 1} | equal |
| 1p3d, host memory, eager | t2-rank256 | 10 | inf | 8 | 0 | {'news': 1, 'code': 2, 'wikitext': 2, 'math': 2, 'exam': 1} | equal |
| 1p3d, host memory | t2-hbh | 10 | inf | 0 | 8 | {'code': 2, 'wikitext': 2, 'math': 2, 'news': 1, 'exam': 1} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |
| 2p2d, GPU to GPU | t2-hbh | native | inf | 0 | 8 | {'news': 1, 'wikitext': 2, 'math': 2, 'code': 2, 'exam': 1} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |

## Codecs that did not load

- aquakv-2: ValueError('round trip not exact')
- exactkv: CalledProcessError(128, ['git', '-C', '/tmp/nbore/kvpd/30293/home/sources/exactkv', 'checkout', '-q', 'd90376b84e8c116424298e98027995bdbbf0e8ee'])
- ibp: ModuleNotFoundError("No module named 'ibp'")

