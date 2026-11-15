# PD matrix smoke on gpu-grad-01, job 30176 (completed with failures), reviewed from the kept results

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 2 runs: `llama32_3b_pd_2p2d_nixl_rtxa5000x4_20261114_2250_smoke`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261114_2251_smoke`.

Runs that did not finish: `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,splitzip,exans,adacode,exactkv,ours-hbh,t2-hbh`; `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,nvcomp-ans,nvcomp-zstd,nvcomp-bitcomp`; `SMOKE=1 LAYOUT=2p2d TRANSPORT=p2p CODECS=raw,ours-hbh,t2-hbh`; `SMOKE=1 LAYOUT=colo4 CONNECTOR=lmcache`; `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring ENGINE_EAGER=1 CODECS=raw,t2-hbh,t2-rank256`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 2p2d, NIXL (vLLM) | nixl | n/a | 4/4 | yes | - |
| colo4, host memory tier | raw (c4) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | exans (c4) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | ours-hbh (c4) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | t2-hbh (c4) | 0/0 chunks | - | yes | - |

## 2p2d, NIXL (vLLM): link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| nixl | 16.000 | 1.27 (0.00); 4734 / 4756 | 12.0 |

Best by goodput, then TTFT p50: rate inf: nixl (goodput 0.00/s, TTFT p50 4734 ms).

## CPU tier, colo4: 4 clients

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Loads | Tokens loaded | Load ms (copy / decode) | Bits stored | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2-hbh | 2.94 | 89 / 180 / 192 | 11.6 | 0 | 0 | - (- / -) | - | 43 |
| ours-hbh | 3.04 | 95 / 153 / 189 | 11.4 | 0 | 0 | - (- / -) | - | 43 |
| exans | 2.98 | 96 / 162 / 226 | 11.3 | 0 | 0 | - (- / -) | - | 44 |
| raw | 2.87 | 117 / 213 / 237 | 11.1 | 0 | 0 | - (- / -) | - | 45 |

## Codecs that did not load

- ibp: ModuleNotFoundError("No module named 'ibp'")

