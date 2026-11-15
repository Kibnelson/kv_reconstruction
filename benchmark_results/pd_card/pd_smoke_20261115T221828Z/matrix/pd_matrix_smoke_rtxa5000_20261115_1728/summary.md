# PD matrix smoke on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 7 runs: `llama32_3b_pd_1p3d_tkv_ring_eager_rtxa5000x4_20261115_1743_smoke`, `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261115_1728_smoke`, `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261115_1733_smoke_partial`, `llama32_3b_pd_2p2d_nixl_rtxa5000x4_20261115_1737_smoke`, `llama32_3b_pd_2p2d_tkv_p2p_rtxa5000x4_20261115_1735_smoke`, `llama32_3b_pd_colo4_lmcache_rtxa5000x4_20261115_1741_smoke`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261115_1739_smoke`.

Runs that did not finish: `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,nvcomp-ans,nvcomp-zstd,nvcomp-bitcomp`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory, eager | t2-hbh | 4/4 | 4/4 | yes | 8.189 |
| 1p3d, host memory, eager | t2-rank256 | 4/4 | 4/4 | yes | 9.199 |
| 1p3d, host memory | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory | splitzip | 4/4 | 4/4 | yes | 12.044 |
| 1p3d, host memory | exans | 4/4 | 4/4 | yes | 10.923 |
| 1p3d, host memory | adacode | 4/4 | 4/4 | yes | 11.371 |
| 1p3d, host memory | exactkv | 4/4 | 4/4 | yes | 12.629 |
| 1p3d, host memory | ours-hbh | 4/4 | 4/4 | yes | 9.053 |
| 1p3d, host memory | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| 1p3d, host memory | raw | 4/4 | 3/4 | yes | 16.000 |
| 1p3d, host memory | nvcomp-ans | 1/4 | 4/4 | NO | 13.369 |
| 1p3d, host memory | nvcomp-zstd | 1/4 | 4/4 | NO | 12.896 |
| 1p3d, host memory | nvcomp-bitcomp | 1/4 | 4/4 | NO | 13.790 |
| 2p2d, NIXL (vLLM) | nixl | n/a | 4/4 | yes | - |
| 2p2d, GPU to GPU | raw | 4/4 | 3/4 | yes | 16.000 |
| 2p2d, GPU to GPU | ours-hbh | 4/4 | 3/4 | yes | 9.053 |
| 2p2d, GPU to GPU | t2-hbh | 4/4 | 3/4 | yes | 8.489 |
| colo4, LMCache (vLLM) tier | lmcache (c8) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | raw (c8) | 23/23 chunks | - | yes | 16.000 |
| colo4, host memory tier | exans (c8) | 8/8 chunks | - | yes | 11.024 |
| colo4, host memory tier | ours-hbh (c8) | 21/21 chunks | - | yes | 8.393 |
| colo4, host memory tier | t2-hbh (c8) | 10/10 chunks | - | yes | 8.381 |
| colo4, host memory tier | t2-rank256 (c8) | 20/20 chunks | - | yes | 9.448 |

## 1p3d, host memory: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.86 (0.70); 2406 / 2828 | 11.5 |
| splitzip | 12.042 | 1.94 (0.73); 2362 / 2654 | 11.6 |
| exans | 10.884 | 2.06 (1.03); 2069 / 2393 | 11.7 |
| adacode | 11.372 | 2.07 (0.77); 2074 / 2396 | 11.7 |
| exactkv | 12.591 | 2.02 (0.76); 2165 / 2493 | 11.6 |
| ours-hbh | 8.684 | 2.11 (1.05); 2034 / 2295 | 12.8 |
| t2-hbh | 8.146 | 2.12 (1.06); 2019 / 2265 | 12.8 |
| nvcomp-ans | 13.365 | 1.89 (0.71); 2376 / 2733 | 12.9 |
| nvcomp-zstd | 12.892 | 1.75 (0.22); 2699 / 3082 | 13.0 |
| nvcomp-bitcomp | 13.786 | 1.88 (0.70); 2355 / 2764 | 12.8 |

Best by goodput, then TTFT p50: rate inf: t2-hbh (goodput 1.06/s, TTFT p50 2019 ms).

## 1p3d, host memory, eager engines: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.86 (0.70); 2382 / 2790 | 11.9 |
| t2-hbh | 7.860 | 2.01 (1.00); 1985 / 2222 | 14.2 |
| t2-rank256 | 8.867 | 2.17 (1.35); 1896 / 2170 | 12.1 |

Best by goodput, then TTFT p50: rate inf: t2-rank256 (goodput 1.35/s, TTFT p50 1896 ms).

## 2p2d, GPU to GPU: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 3.47 (3.47); 749 / 762 | 12.1 |
| ours-hbh | 8.684 | 2.80 (2.80); 1313 / 1318 | 12.1 |
| t2-hbh | 8.146 | 2.76 (2.76); 1354 / 1356 | 12.1 |

Best by goodput, then TTFT p50: rate inf: raw (goodput 3.47/s, TTFT p50 749 ms).

## 2p2d, NIXL (vLLM): link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| nixl | 16.000 | 2.09 (0.00); 2271 / 2296 | 12.1 |

Best by goodput, then TTFT p50: rate inf: nixl (goodput 0.00/s, TTFT p50 2271 ms).

## CPU tier, colo4: 8 clients

Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2-hbh | 5.42 | 65 / 217 / 914 | 11.7 | 66.7 / 8.1 | 5 | 1872 | 317.5 (1.2 / 297.3) | 8.381 | 73 | 82 |
| t2-rank256 | 5.48 | 68 / 180 / 355 | 11.7 | 66.3 / 14.5 | 10 | 3856 | 63.0 (1.6 / 47.2) | 9.448 | 75 | 89 |
| exans | 5.98 | 68 / 152 / 548 | 11.7 | 68.3 / 6.3 | 5 | 1408 | 43.4 (1.8 / 15.8) | 11.024 | 69 | 83 |
| ours-hbh | 5.19 | 70 / 262 / 687 | 11.6 | 65.2 / 15.0 | 9 | 4080 | 366.1 (1.7 / 345.8) | 8.393 | 76 | 90 |
| LMCache CPU offload (raw) | 5.79 | 77 / 180 / 252 | 11.5 | 63.4 / 15.7 | - | - | - (- / -) | - | - | - |
| raw | 5.69 | 81 / 136 / 278 | 11.8 | 64.0 / 16.3 | 9 | 4400 | 43.5 (2.9 / 17.1) | 16.000 | 75 | 85 |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | t2-hbh | 10 | inf | 8 | 0 | {'code': 2, 'news': 1, 'wikitext': 2, 'math': 2, 'exam': 1} | equal |
| 1p3d, host memory, eager | t2-rank256 | 10 | inf | 8 | 0 | {'math': 2, 'code': 2, 'wikitext': 2, 'news': 1, 'exam': 1} | equal |
| 1p3d, host memory | t2-hbh | 10 | inf | 0 | 8 | {'wikitext': 2, 'math': 2, 'code': 2, 'news': 1, 'exam': 1} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |
| 2p2d, GPU to GPU | t2-hbh | native | inf | 0 | 8 | {'wikitext': 2, 'math': 2, 'exam': 1, 'code': 2, 'news': 1} | differs in 1979472 of 4194304 words (2048 of 2048 tokens) with the scheduler's row counts; differs in 2027116 of 4194304 words (2048 of 2048 tokens) with the sc |

## Kernels compiled inside measured points

vLLM compiled these while a point was measured, so that point's latencies include the compile:

- 2p2d, GPU to GPU: sweep_raw_nativegbps_rinf (engine_decode2.log): Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 2p2d, GPU to GPU: sweep_raw_nativegbps_rinf (engine_decode2.log): Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 2p2d, GPU to GPU: sweep_raw_nativegbps_rinf (engine_decode2.log): Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex
- 2p2d, GPU to GPU: sweep_raw_nativegbps_rinf (engine_prefill2.log): Triton kernel JIT compilation during inference: _topp_sb_mask_kernel. This causes a latency spike; consider ex
- 2p2d, GPU to GPU: sweep_raw_nativegbps_rinf (engine_prefill2.log): Triton kernel JIT compilation during inference: _topp_sb_stats_kernel. This causes a latency spike; consider e
- 2p2d, GPU to GPU: sweep_raw_nativegbps_rinf (engine_prefill2.log): Triton kernel JIT compilation during inference: _topp_sb_step_kernel. This causes a latency spike; consider ex

## Codecs that did not load

- ibp: ModuleNotFoundError("No module named 'ibp'")

