# PD matrix smoke on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 7 runs: `llama32_3b_pd_1p3d_tkv_ring_eager_rtxa5000x4_20261116_1501_smoke`, `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261116_1446_smoke`, `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261116_1450_smoke`, `llama32_3b_pd_2p2d_nixl_allgpus_rtxa5000x4_20261116_1455_smoke`, `llama32_3b_pd_2p2d_tkv_p2p_rtxa5000x4_20261116_1453_smoke`, `llama32_3b_pd_colo4_lmcache_rtxa5000x4_20261116_1459_smoke`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261116_1456_smoke`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory, eager | t2-hbh | 4/4 | 4/4 | yes | 8.189 |
| 1p3d, host memory, eager | t2-rank256 | 4/4 | 4/4 | yes | 9.199 |
| 1p3d, host memory | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory | splitzip | 4/4 | 3/4 | yes | 12.044 |
| 1p3d, host memory | exans | 4/4 | 4/4 | yes | 10.923 |
| 1p3d, host memory | adacode | 4/4 | 4/4 | yes | 11.371 |
| 1p3d, host memory | exactkv | 4/4 | 3/4 | yes | 12.629 |
| 1p3d, host memory | ours-hbh | 4/4 | 4/4 | yes | 9.053 |
| 1p3d, host memory | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| 1p3d, host memory | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory | nvcomp-ans | 4/4 | 4/4 | yes | 13.369 |
| 1p3d, host memory | nvcomp-zstd | 4/4 | 4/4 | yes | 12.896 |
| 1p3d, host memory | nvcomp-bitcomp | 4/4 | 4/4 | yes | 13.790 |
| 2p2d, NIXL (vLLM), every GPU visible | nixl | n/a | 4/4 | yes | - |
| 2p2d, GPU to GPU | raw | 4/4 | 4/4 | yes | 16.000 |
| 2p2d, GPU to GPU | ours-hbh | 4/4 | 4/4 | yes | 9.053 |
| 2p2d, GPU to GPU | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| colo4, LMCache (vLLM) tier | lmcache (c8) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | raw (c8) | 20/20 chunks | - | yes | 16.000 |
| colo4, host memory tier | exans (c8) | 28/28 chunks | - | yes | 11.021 |
| colo4, host memory tier | ours-hbh (c8) | 11/11 chunks | - | yes | 8.401 |
| colo4, host memory tier | t2-hbh (c8) | 16/16 chunks | - | yes | 8.384 |
| colo4, host memory tier | t2-rank256 (c8) | 26/26 chunks | - | yes | 9.433 |

## 1p3d, host memory: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.88 (0.70); 2393 / 2806 | 11.5 |
| splitzip | 12.042 | 1.95 (0.73); 2341 / 2634 | 11.6 |
| exans | 10.884 | 2.08 (1.04); 2060 / 2376 | 11.6 |
| adacode | 11.372 | 2.08 (0.78); 2061 / 2379 | 11.6 |
| exactkv | 12.590 | 2.01 (0.75); 2176 / 2511 | 11.6 |
| ours-hbh | 8.684 | 2.11 (1.05); 2050 / 2305 | 12.7 |
| t2-hbh | 8.146 | 2.14 (1.07); 1998 / 2242 | 12.7 |
| nvcomp-ans | 13.365 | 1.98 (0.74); 2224 / 2577 | 11.5 |
| nvcomp-zstd | 12.892 | 1.77 (0.00); 2705 / 3049 | 11.6 |
| nvcomp-bitcomp | 13.786 | 1.95 (0.73); 2301 / 2638 | 11.6 |

Best by goodput, then TTFT p50: rate inf: t2-hbh (goodput 1.07/s, TTFT p50 1998 ms).

## 1p3d, host memory, eager engines: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.87 (0.70); 2382 / 2790 | 11.8 |
| t2-hbh | 7.860 | 2.15 (1.08); 1963 / 2193 | 12.9 |
| t2-rank256 | 8.867 | 1.91 (1.19); 1907 / 2181 | 14.9 |

Best by goodput, then TTFT p50: rate inf: t2-rank256 (goodput 1.19/s, TTFT p50 1907 ms).

## 2p2d, GPU to GPU: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 3.50 (3.50); 737 / 763 | 12.0 |
| ours-hbh | 8.684 | 2.81 (2.81); 1291 / 1318 | 12.0 |
| t2-hbh | 8.146 | 2.78 (2.78); 1331 / 1356 | 12.0 |

Best by goodput, then TTFT p50: rate inf: raw (goodput 3.50/s, TTFT p50 737 ms).

## 2p2d, NIXL (vLLM), every GPU visible: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| nixl | 16.000 | 2.09 (1.04); 2080 / 2313 | 11.9 |

Best by goodput, then TTFT p50: rate inf: nixl (goodput 1.04/s, TTFT p50 2080 ms).

## CPU tier, colo4: 8 clients

Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2-hbh | 5.60 | 68 / 372 / 517 | 11.7 | 65.4 / 11.3 | 8 | 2880 | 302.1 (1.5 / 283.1) | 8.384 | 70 | 85 |
| t2-rank256 | 5.88 | 72 / 160 / 182 | 11.8 | 64.0 / 19.2 | 11 | 5648 | 58.6 (1.6 / 38.8) | 9.433 | 85 | 88 |
| ours-hbh | 5.75 | 73 / 246 / 583 | 11.7 | 67.7 / 8.6 | 7 | 2112 | 262.0 (1.3 / 245.0) | 8.401 | 75 | 86 |
| exans | 5.97 | 77 / 141 / 188 | 11.6 | 63.1 / 19.3 | 11 | 5632 | 43.4 (2.0 / 22.0) | 11.021 | 79 | 87 |
| raw | 5.73 | 82 / 215 / 559 | 11.9 | 63.5 / 15.1 | 9 | 3920 | 52.1 (3.1 / 15.6) | 16.000 | 77 | 83 |
| LMCache CPU offload (raw) | 5.75 | 88 / 223 / 400 | 11.4 | 62.3 / 22.8 | - | - | - (- / -) | - | - | - |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | t2-hbh | 10 | inf | 8 | 0 | {'news': 1, 'code': 2, 'wikitext': 2, 'math': 2, 'exam': 1} | equal |
| 1p3d, host memory, eager | t2-rank256 | 10 | inf | 8 | 0 | {'code': 2, 'news': 1, 'math': 2, 'wikitext': 2, 'exam': 1} | equal |
| 1p3d, host memory | t2-hbh | 10 | inf | 0 | 8 | {'code': 2, 'math': 2, 'wikitext': 2, 'news': 1, 'exam': 1} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |
| 2p2d, GPU to GPU | t2-hbh | native | inf | 0 | 8 | {'code': 2, 'math': 2, 'news': 1, 'wikitext': 2, 'exam': 1} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |

## Codecs that did not load

- ibp: ModuleNotFoundError("No module named 'ibp'")

