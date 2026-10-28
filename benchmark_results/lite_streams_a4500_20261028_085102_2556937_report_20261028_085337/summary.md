# Complete lossless Lite stream-length tuning

Status: **completed**. GPU measured: **True**.

Same original BF16 words, fused execution, fitted predictor and entropy kernels. Only stream length and lanes per program change.
Means average per-document synchronized wall medians. Complete packets charge framing, all stream states and INT64 directories.

| Method | Complete bpv | Saved | Encode ms | Decode ms | Exact |
|---|---:|---:|---:|---:|---|
| raw_bf16 | 16.00009 | -0.00% | 0.406 | 0.340 | True |
| splitzip_official_framed | 12.04281 | 24.73% | 0.865 | 0.518 | True |
| exans_official_fixed_framed | 12.37970 | 22.63% | 15.161 | 3.669 | True |
| lite_v128_b8_before | 8.93339 | 44.17% | 156.029 | 158.098 | True |
| lite_v64_b8 | 9.18328 | 42.60% | 155.406 | 159.502 | True |
| lite_v32_b8 | 9.68380 | 39.48% | 155.961 | 159.131 | True |
| lite_v64_b32 | 9.09012 | 43.19% | 152.456 | 156.483 | True |
| lite_v32_b32 | 9.49689 | 40.64% | 157.137 | 160.398 | True |
| lite_v128_b8_after | 8.93339 | 44.17% | 157.234 | 159.553 | True |

Calculated serial network model; these are not measured network or serving latencies.

| Effective Gb/s | Best fresh encode + transfer + decode | Best precompressed fetch + decode |
|---:|---|---|
| 0.1 | lite_v128_b8_before | lite_v128_b8_before |
| 0.25 | lite_v128_b8_before | lite_v128_b8_before |
| 0.5 | splitzip_official_framed | lite_v128_b8_before |
| 1 | splitzip_official_framed | splitzip_official_framed |
| 2.5 | splitzip_official_framed | splitzip_official_framed |
| 5 | splitzip_official_framed | splitzip_official_framed |
| 10 | splitzip_official_framed | splitzip_official_framed |
| 25 | splitzip_official_framed | splitzip_official_framed |
| 50 | splitzip_official_framed | splitzip_official_framed |
| 100 | splitzip_official_framed | splitzip_official_framed |

Candidate decode budgets use each new complete packet size and the retimed official SplitZip result.

| Candidate | Gb/s | Decode budget ms | Current decode ms | Modeled fetch ms | SplitZip fetch ms |
|---|---:|---:|---:|---:|---:|
| lite_v128_b8_before | 1 | 91.811 | 158.098 | 420.383 | 354.096 |
| lite_v128_b8_before | 2.5 | 37.035 | 158.098 | 263.012 | 141.949 |
| lite_v128_b8_before | 5 | 18.776 | 158.098 | 210.555 | 71.233 |
| lite_v128_b8_before | 10 | 9.647 | 158.098 | 184.326 | 35.876 |
| lite_v64_b8 | 1 | 84.474 | 159.502 | 429.124 | 354.096 |
| lite_v64_b8 | 2.5 | 34.100 | 159.502 | 267.351 | 141.949 |
| lite_v64_b8 | 5 | 17.309 | 159.502 | 213.426 | 71.233 |
| lite_v64_b8 | 10 | 8.913 | 159.502 | 186.464 | 35.876 |
| lite_v32_b8 | 1 | 69.779 | 159.131 | 443.448 | 354.096 |
| lite_v32_b8 | 2.5 | 28.222 | 159.131 | 272.858 | 141.949 |
| lite_v32_b8 | 5 | 14.370 | 159.131 | 215.994 | 71.233 |
| lite_v32_b8 | 10 | 7.444 | 159.131 | 187.562 | 35.876 |
| lite_v64_b32 | 1 | 87.209 | 156.483 | 423.371 | 354.096 |
| lite_v64_b32 | 2.5 | 35.194 | 156.483 | 263.238 | 141.949 |
| lite_v64_b32 | 5 | 17.856 | 156.483 | 209.861 | 71.233 |
| lite_v64_b32 | 10 | 9.187 | 156.483 | 183.172 | 35.876 |
| lite_v32_b32 | 1 | 75.266 | 160.398 | 439.228 | 354.096 |
| lite_v32_b32 | 2.5 | 30.417 | 160.398 | 271.930 | 141.949 |
| lite_v32_b32 | 5 | 15.468 | 160.398 | 216.164 | 71.233 |
| lite_v32_b32 | 10 | 7.993 | 160.398 | 188.281 | 35.876 |
| lite_v128_b8_after | 1 | 91.811 | 159.553 | 421.838 | 354.096 |
| lite_v128_b8_after | 2.5 | 37.035 | 159.553 | 264.467 | 141.949 |
| lite_v128_b8_after | 5 | 18.776 | 159.553 | 212.010 | 71.233 |
| lite_v128_b8_after | 10 | 9.647 | 159.553 | 185.781 | 35.876 |

Separate optional diagnostic processes ran after ordinary timings. Full traces stay in runs/.

| Profile | Direction | GPU kernels | Sum kernel ms |
|---|---|---:|---:|
| lite_v128_b8_before | encode | 19341 | 109.658 |
| lite_v128_b8_before | decode | 18877 | 127.169 |
| lite_v64_b32 | encode | 19341 | 81.252 |
| lite_v64_b32 | decode | 18877 | 100.341 |

V128/B8 controls before/after the sweep must match the previous fused packet bytes. Other layouts preserve every original BF16 word but have newly measured packet bytes.
Retimed raw BF16, official SplitZip and official ExANS retain their original environments, fixed state and accounting. No AQUA adapter is substituted; its official run remains blocked.
Every method has a fresh GPU packet-only receiver. Timing workers never profile. Input upload/export, cold initialization and shared predictor state are separate.
No fitting, model download, baseline rebuild, actual network, SSD, pool conversion, attention or serving measurement is performed.
