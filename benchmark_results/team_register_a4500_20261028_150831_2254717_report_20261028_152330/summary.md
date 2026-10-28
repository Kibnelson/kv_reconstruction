# GPU decoder partner-register reuse versus official controls

Status: **completed**. GPU measured: **True**.

Same frozen Llama-3.2-3B BF16 words, all 28 layers, text/code TEST documents, 512 and 8192 tokens. Packets, predictor coefficients, CDFs and official native controls are unchanged.
Decoder kernel warps: **2**. This explicit run setting applies to the four own decoder arms; official controls keep their original launch settings.
Primary timing starts with compressed payload on GPU and covers validation, normal API metadata work, decode and an independently owned output. Host bulk hashing/import/upload, encoding, disk and real networking are excluded.
graph_reference/direct_reference retain the frozen binary-search decoder. graph_register/direct_register reuse partner prediction and decoded-word vectors through a static half swap, avoiding duplicate loads and the feature barrier. Both execution modes return independently owned outputs. Four-arm rotation/reversal is predeclared; controls run before and after.

| Tokens | Method | Arm / round | Complete bpv | Saved | Full decode ms | CUDA interval ms | Exact |
|---:|---|---|---:|---:|---:|---:|---|
| 512 | exans_official_fixed_framed | native / after | 12.37970 | 22.63% | 3.719 | 3.714 | True |
| 512 | exans_official_fixed_framed | native / before | 12.37970 | 22.63% | 3.728 | 3.723 | True |
| 512 | raw_bf16 | native / after | 16.00009 | -0.00% | 0.362 | 0.360 | True |
| 512 | raw_bf16 | native / before | 16.00009 | -0.00% | 0.364 | 0.363 | True |
| 512 | splitzip_official_framed | native / after | 12.04281 | 24.73% | 0.579 | 0.578 | True |
| 512 | splitzip_official_framed | native / before | 12.04281 | 24.73% | 0.576 | 0.575 | True |
| 512 | team_dense_uniform | direct_reference / paired | 9.11376 | 43.04% | 6.454 | 6.451 | True |
| 512 | team_dense_uniform | direct_register / paired | 9.11376 | 43.04% | 6.996 | 6.993 | True |
| 512 | team_dense_uniform | graph_reference / paired | 9.11376 | 43.04% | 4.785 | 4.783 | True |
| 512 | team_dense_uniform | graph_register / paired | 9.11376 | 43.04% | 5.379 | 5.377 | True |
| 512 | team_rank256_uniform | direct_reference / paired | 9.50504 | 40.59% | 7.806 | 7.802 | True |
| 512 | team_rank256_uniform | direct_register / paired | 9.50504 | 40.59% | 7.947 | 7.944 | True |
| 512 | team_rank256_uniform | graph_reference / paired | 9.50504 | 40.59% | 5.185 | 5.182 | True |
| 512 | team_rank256_uniform | graph_register / paired | 9.50504 | 40.59% | 5.696 | 5.693 | True |
| 8192 | exans_official_fixed_framed | native / after | 12.34020 | 22.87% | 23.690 | 23.683 | True |
| 8192 | exans_official_fixed_framed | native / before | 12.34020 | 22.87% | 23.725 | 23.719 | True |
| 8192 | raw_bf16 | native / after | 16.00001 | -0.00% | 3.473 | 3.471 | True |
| 8192 | raw_bf16 | native / before | 16.00001 | -0.00% | 3.475 | 3.473 | True |
| 8192 | splitzip_official_framed | native / after | 12.05013 | 24.69% | 3.457 | 3.455 | True |
| 8192 | splitzip_official_framed | native / before | 12.05013 | 24.69% | 3.431 | 3.429 | True |
| 8192 | team_dense_uniform | direct_reference / paired | 8.92097 | 44.24% | 30.774 | 30.770 | True |
| 8192 | team_dense_uniform | direct_register / paired | 8.92097 | 44.24% | 33.035 | 33.031 | True |
| 8192 | team_dense_uniform | graph_reference / paired | 8.92097 | 44.24% | 36.075 | 36.069 | True |
| 8192 | team_dense_uniform | graph_register / paired | 8.92097 | 44.24% | 38.696 | 38.690 | True |
| 8192 | team_rank256_uniform | direct_reference / paired | 9.30632 | 41.84% | 23.671 | 23.667 | True |
| 8192 | team_rank256_uniform | direct_register / paired | 9.30632 | 41.84% | 27.131 | 27.128 | True |
| 8192 | team_rank256_uniform | graph_reference / paired | 9.30632 | 41.84% | 28.913 | 28.911 | True |
| 8192 | team_rank256_uniform | graph_register / paired | 9.30632 | 41.84% | 31.850 | 31.846 | True |

Calculated serial wire + full GPU decode, milliseconds. All methods/arms are shown; native before/after rounds remain separate. Ideal overlap max(wire,decode) is a separate unmeasured bound in results.json.

| Tokens | Method | Arm / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | exans_official_fixed_framed | native / after | 367.189 | 7.354 | 5.537 | 4.931 | 4.628 |
| 512 | exans_official_fixed_framed | native / before | 367.197 | 7.362 | 5.545 | 4.939 | 4.636 |
| 512 | raw_bf16 | native / after | 470.127 | 5.060 | 2.711 | 1.928 | 1.537 |
| 512 | raw_bf16 | native / before | 470.129 | 5.062 | 2.713 | 1.930 | 1.539 |
| 512 | splitzip_official_framed | native / after | 354.158 | 4.115 | 2.347 | 1.758 | 1.463 |
| 512 | splitzip_official_framed | native / before | 354.155 | 4.112 | 2.344 | 1.755 | 1.460 |
| 512 | team_dense_uniform | direct_reference / paired | 274.035 | 9.130 | 7.792 | 7.346 | 7.123 |
| 512 | team_dense_uniform | direct_register / paired | 274.577 | 9.672 | 8.334 | 7.888 | 7.665 |
| 512 | team_dense_uniform | graph_reference / paired | 272.366 | 7.461 | 6.123 | 5.677 | 5.454 |
| 512 | team_dense_uniform | graph_register / paired | 272.960 | 8.055 | 6.717 | 6.271 | 6.048 |
| 512 | team_rank256_uniform | direct_reference / paired | 286.875 | 10.596 | 9.201 | 8.736 | 8.503 |
| 512 | team_rank256_uniform | direct_register / paired | 287.017 | 10.738 | 9.343 | 8.877 | 8.645 |
| 512 | team_rank256_uniform | graph_reference / paired | 284.254 | 7.976 | 6.580 | 6.115 | 5.883 |
| 512 | team_rank256_uniform | graph_register / paired | 284.765 | 8.487 | 7.091 | 6.626 | 6.394 |
| 8192 | exans_official_fixed_framed | native / after | 5820.646 | 81.659 | 52.674 | 43.013 | 38.182 |
| 8192 | exans_official_fixed_framed | native / before | 5820.681 | 81.694 | 52.709 | 43.048 | 38.217 |
| 8192 | raw_bf16 | native / after | 7519.668 | 78.635 | 41.054 | 28.527 | 22.264 |
| 8192 | raw_bf16 | native / before | 7519.670 | 78.637 | 41.056 | 28.529 | 22.266 |
| 8192 | splitzip_official_framed | native / after | 5664.149 | 60.064 | 31.760 | 22.326 | 17.608 |
| 8192 | splitzip_official_framed | native / before | 5664.124 | 60.038 | 31.735 | 22.300 | 17.583 |
| 8192 | team_dense_uniform | direct_reference / paired | 4221.508 | 72.682 | 51.728 | 44.743 | 41.251 |
| 8192 | team_dense_uniform | direct_register / paired | 4223.768 | 74.942 | 53.988 | 47.004 | 43.512 |
| 8192 | team_dense_uniform | graph_reference / paired | 4226.809 | 77.983 | 57.029 | 50.044 | 46.552 |
| 8192 | team_dense_uniform | graph_register / paired | 4229.429 | 80.603 | 59.649 | 52.665 | 49.172 |
| 8192 | team_rank256_uniform | direct_reference / paired | 4395.428 | 67.388 | 45.530 | 38.243 | 34.600 |
| 8192 | team_rank256_uniform | direct_register / paired | 4398.889 | 70.849 | 48.990 | 41.704 | 38.061 |
| 8192 | team_rank256_uniform | graph_reference / paired | 4400.670 | 72.631 | 50.772 | 43.486 | 39.842 |
| 8192 | team_rank256_uniform | graph_register / paired | 4403.607 | 75.567 | 53.708 | 46.422 | 42.779 |

Per-link budgets and speed ratios versus the faster SplitZip and ExANS control rounds are in results.json. No routing or method selection on TEST is deployed by this sweep.

Separate 8K text profiles record CPU/Torch operators and actual CUDA kernels. Summed kernel work may overlap across streams; it is not wall latency. Stage/eager/clone diagnostics are separate calls and must not be added as a graph decomposition.

| Method | Arm | Full diagnostic ms | Stage ms | Native eager ms | Output clone ms |
|---|---|---:|---:|---:|---:|
| exans_official_fixed_framed | native | 24.092 | — | — | — |
| splitzip_official_framed | native | 3.547 | — | — | — |
| team_dense_uniform | direct_reference | 30.111 | 0.389 | — | — |
| team_dense_uniform | direct_register | 34.792 | 0.453 | — | — |
| team_dense_uniform | graph_reference | 36.142 | 2.293 | 29.789 | 3.464 |
| team_dense_uniform | graph_register | 38.720 | 2.368 | 33.125 | 3.473 |
| team_rank256_uniform | direct_reference | 23.996 | 0.486 | — | — |
| team_rank256_uniform | direct_register | 26.868 | 0.391 | — | — |
| team_rank256_uniform | graph_reference | 29.270 | 2.358 | 23.652 | 3.372 |
| team_rank256_uniform | graph_register | 32.058 | 2.364 | 26.700 | 3.470 |

All arms require original-word equality, unchanged wire hashes, held-output ownership and mode-specific graph counts. Fixed shared state and scratch are additional to packet bytes. ExANS native allocation peaks are unmeasured.
AQUA-KV is lossy and remains a separate official evaluation. No compressed attention, T-to-P conversion, fresh encode, SSD/network or serving performance is measured here.
