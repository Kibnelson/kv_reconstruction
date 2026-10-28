# GPU decoder partner-register reuse versus official controls

Status: **completed**. GPU measured: **True**.

Same frozen Llama-3.2-3B BF16 words, all 28 layers, text/code TEST documents, 512 and 8192 tokens. Packets, predictor coefficients, CDFs and official native controls are unchanged.
Decoder kernel warps: **1**. This explicit run setting applies to the four own decoder arms; official controls keep their original launch settings.
Primary timing starts with compressed payload on GPU and covers validation, normal API metadata work, decode and an independently owned output. Host bulk hashing/import/upload, encoding, disk and real networking are excluded.
graph_reference/direct_reference retain the frozen binary-search decoder. graph_register/direct_register reuse partner prediction and decoded-word vectors through a static half swap, avoiding duplicate loads and the feature barrier. Both execution modes return independently owned outputs. Four-arm rotation/reversal is predeclared; controls run before and after.

| Tokens | Method | Arm / round | Complete bpv | Saved | Full decode ms | CUDA interval ms | Exact |
|---:|---|---|---:|---:|---:|---:|---|
| 512 | exans_official_fixed_framed | native / after | 12.37970 | 22.63% | 3.705 | 3.699 | True |
| 512 | exans_official_fixed_framed | native / before | 12.37970 | 22.63% | 3.711 | 3.705 | True |
| 512 | raw_bf16 | native / after | 16.00009 | -0.00% | 0.365 | 0.363 | True |
| 512 | raw_bf16 | native / before | 16.00009 | -0.00% | 0.365 | 0.363 | True |
| 512 | splitzip_official_framed | native / after | 12.04281 | 24.73% | 0.573 | 0.572 | True |
| 512 | splitzip_official_framed | native / before | 12.04281 | 24.73% | 0.575 | 0.574 | True |
| 512 | team_dense_uniform | direct_reference / paired | 9.11376 | 43.04% | 6.417 | 6.415 | True |
| 512 | team_dense_uniform | direct_register / paired | 9.11376 | 43.04% | 6.550 | 6.547 | True |
| 512 | team_dense_uniform | graph_reference / paired | 9.11376 | 43.04% | 4.497 | 4.494 | True |
| 512 | team_dense_uniform | graph_register / paired | 9.11376 | 43.04% | 4.535 | 4.533 | True |
| 512 | team_rank256_uniform | direct_reference / paired | 9.50504 | 40.59% | 7.807 | 7.804 | True |
| 512 | team_rank256_uniform | direct_register / paired | 9.50504 | 40.59% | 7.921 | 7.918 | True |
| 512 | team_rank256_uniform | graph_reference / paired | 9.50504 | 40.59% | 5.092 | 5.089 | True |
| 512 | team_rank256_uniform | graph_register / paired | 9.50504 | 40.59% | 5.141 | 5.139 | True |
| 8192 | exans_official_fixed_framed | native / after | 12.34020 | 22.87% | 23.673 | 23.667 | True |
| 8192 | exans_official_fixed_framed | native / before | 12.34020 | 22.87% | 23.764 | 23.760 | True |
| 8192 | raw_bf16 | native / after | 16.00001 | -0.00% | 3.470 | 3.468 | True |
| 8192 | raw_bf16 | native / before | 16.00001 | -0.00% | 3.468 | 3.466 | True |
| 8192 | splitzip_official_framed | native / after | 12.05013 | 24.69% | 3.444 | 3.442 | True |
| 8192 | splitzip_official_framed | native / before | 12.05013 | 24.69% | 3.439 | 3.437 | True |
| 8192 | team_dense_uniform | direct_reference / paired | 8.92097 | 44.24% | 28.413 | 28.409 | True |
| 8192 | team_dense_uniform | direct_register / paired | 8.92097 | 44.24% | 28.698 | 28.694 | True |
| 8192 | team_dense_uniform | graph_reference / paired | 8.92097 | 44.24% | 33.599 | 33.594 | True |
| 8192 | team_dense_uniform | graph_register / paired | 8.92097 | 44.24% | 33.726 | 33.722 | True |
| 8192 | team_rank256_uniform | direct_reference / paired | 9.30632 | 41.84% | 21.621 | 21.617 | True |
| 8192 | team_rank256_uniform | direct_register / paired | 9.30632 | 41.84% | 21.894 | 21.891 | True |
| 8192 | team_rank256_uniform | graph_reference / paired | 9.30632 | 41.84% | 26.676 | 26.673 | True |
| 8192 | team_rank256_uniform | graph_register / paired | 9.30632 | 41.84% | 26.805 | 26.802 | True |

Calculated serial wire + full GPU decode, milliseconds. All methods/arms are shown; native before/after rounds remain separate. Ideal overlap max(wire,decode) is a separate unmeasured bound in results.json.

| Tokens | Method | Arm / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | exans_official_fixed_framed | native / after | 367.174 | 7.340 | 5.522 | 4.916 | 4.614 |
| 512 | exans_official_fixed_framed | native / before | 367.181 | 7.346 | 5.528 | 4.923 | 4.620 |
| 512 | raw_bf16 | native / after | 470.130 | 5.063 | 2.714 | 1.931 | 1.539 |
| 512 | raw_bf16 | native / before | 470.129 | 5.062 | 2.713 | 1.931 | 1.539 |
| 512 | splitzip_official_framed | native / after | 354.152 | 4.109 | 2.341 | 1.752 | 1.457 |
| 512 | splitzip_official_framed | native / before | 354.154 | 4.111 | 2.343 | 1.754 | 1.459 |
| 512 | team_dense_uniform | direct_reference / paired | 273.998 | 9.093 | 7.755 | 7.309 | 7.086 |
| 512 | team_dense_uniform | direct_register / paired | 274.131 | 9.225 | 7.888 | 7.442 | 7.219 |
| 512 | team_dense_uniform | graph_reference / paired | 272.078 | 7.173 | 5.835 | 5.389 | 5.166 |
| 512 | team_dense_uniform | graph_register / paired | 272.116 | 7.211 | 5.873 | 5.427 | 5.204 |
| 512 | team_rank256_uniform | direct_reference / paired | 286.876 | 10.598 | 9.202 | 8.737 | 8.505 |
| 512 | team_rank256_uniform | direct_register / paired | 286.990 | 10.711 | 9.316 | 8.851 | 8.618 |
| 512 | team_rank256_uniform | graph_reference / paired | 284.161 | 7.882 | 6.487 | 6.022 | 5.789 |
| 512 | team_rank256_uniform | graph_register / paired | 284.211 | 7.932 | 6.537 | 6.072 | 5.839 |
| 8192 | exans_official_fixed_framed | native / after | 5820.630 | 81.643 | 52.658 | 42.996 | 38.166 |
| 8192 | exans_official_fixed_framed | native / before | 5820.721 | 81.734 | 52.749 | 43.088 | 38.257 |
| 8192 | raw_bf16 | native / after | 7519.665 | 78.632 | 41.051 | 28.524 | 22.260 |
| 8192 | raw_bf16 | native / before | 7519.663 | 78.630 | 41.049 | 28.522 | 22.258 |
| 8192 | splitzip_official_framed | native / after | 5664.136 | 60.051 | 31.747 | 22.313 | 17.596 |
| 8192 | splitzip_official_framed | native / before | 5664.131 | 60.046 | 31.742 | 22.308 | 17.591 |
| 8192 | team_dense_uniform | direct_reference / paired | 4219.147 | 70.321 | 49.367 | 42.383 | 38.890 |
| 8192 | team_dense_uniform | direct_register / paired | 4219.431 | 70.605 | 49.651 | 42.667 | 39.175 |
| 8192 | team_dense_uniform | graph_reference / paired | 4224.332 | 75.506 | 54.553 | 47.568 | 44.076 |
| 8192 | team_dense_uniform | graph_register / paired | 4224.459 | 75.633 | 54.680 | 47.695 | 44.203 |
| 8192 | team_rank256_uniform | direct_reference / paired | 4393.378 | 65.339 | 43.480 | 36.194 | 32.550 |
| 8192 | team_rank256_uniform | direct_register / paired | 4393.652 | 65.612 | 43.753 | 36.467 | 32.824 |
| 8192 | team_rank256_uniform | graph_reference / paired | 4398.433 | 70.394 | 48.535 | 41.249 | 37.605 |
| 8192 | team_rank256_uniform | graph_register / paired | 4398.563 | 70.523 | 48.664 | 41.378 | 37.735 |

Per-link budgets and speed ratios versus the faster SplitZip and ExANS control rounds are in results.json. No routing or method selection on TEST is deployed by this sweep.

Separate 8K text profiles record CPU/Torch operators and actual CUDA kernels. Summed kernel work may overlap across streams; it is not wall latency. Stage/eager/clone diagnostics are separate calls and must not be added as a graph decomposition.

| Method | Arm | Full diagnostic ms | Stage ms | Native eager ms | Output clone ms |
|---|---|---:|---:|---:|---:|
| exans_official_fixed_framed | native | 24.320 | — | — | — |
| splitzip_official_framed | native | 3.545 | — | — | — |
| team_dense_uniform | direct_reference | 29.583 | 0.407 | — | — |
| team_dense_uniform | direct_register | 28.514 | 0.391 | — | — |
| team_dense_uniform | graph_reference | 33.729 | 2.377 | 27.886 | 3.383 |
| team_dense_uniform | graph_register | 34.392 | 2.312 | 28.016 | 3.383 |
| team_rank256_uniform | direct_reference | 22.007 | 0.472 | — | — |
| team_rank256_uniform | direct_register | 22.012 | 0.482 | — | — |
| team_rank256_uniform | graph_reference | 26.814 | 2.458 | 21.486 | 3.376 |
| team_rank256_uniform | graph_register | 27.002 | 2.498 | 21.722 | 3.388 |

All arms require original-word equality, unchanged wire hashes, held-output ownership and mode-specific graph counts. Fixed shared state and scratch are additional to packet bytes. ExANS native allocation peaks are unmeasured.
AQUA-KV is lossy and remains a separate official evaluation. No compressed attention, T-to-P conversion, fresh encode, SSD/network or serving performance is measured here.
