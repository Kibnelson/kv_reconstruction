# GPU lookup and borrowed-input exact KV decode

Status: **completed**. GPU measured: **True**.

Same frozen Llama-3.2-3B BF16 words, all 28 layers, text/code TEST documents, 512 and 8192 tokens. Packets, predictor coefficients, CDFs and official native controls are unchanged.
Primary timing starts with compressed payload on GPU and covers validation, normal API metadata work, decode and an independently owned output. Host bulk hashing/import/upload, encoding, disk and real networking are excluded.
compact_search is the current staged CUDA-graph decoder; compact_lookup selects the existing exact inverse-CDF lookup. direct_search/direct_lookup borrow the read-only GPU payload and allocate their own output directly. Four-arm rotation/reversal is predeclared; controls run before and after.

| Tokens | Method | Arm / round | Complete bpv | Saved | Full decode ms | CUDA interval ms | Exact |
|---:|---|---|---:|---:|---:|---:|---|
| 512 | exans_official_fixed_framed | native / after | 12.37970 | 22.63% | 3.718 | 3.712 | True |
| 512 | exans_official_fixed_framed | native / before | 12.37970 | 22.63% | 3.704 | 3.697 | True |
| 512 | raw_bf16 | native / after | 16.00009 | -0.00% | 0.363 | 0.360 | True |
| 512 | raw_bf16 | native / before | 16.00009 | -0.00% | 0.375 | 0.373 | True |
| 512 | splitzip_official_framed | native / after | 12.04281 | 24.73% | 0.582 | 0.580 | True |
| 512 | splitzip_official_framed | native / before | 12.04281 | 24.73% | 0.582 | 0.581 | True |
| 512 | team_dense_uniform | compact_lookup / paired | 9.11376 | 43.04% | 4.542 | 4.540 | True |
| 512 | team_dense_uniform | compact_search / paired | 9.11376 | 43.04% | 4.499 | 4.497 | True |
| 512 | team_dense_uniform | direct_lookup / paired | 9.11376 | 43.04% | 6.302 | 6.300 | True |
| 512 | team_dense_uniform | direct_search / paired | 9.11376 | 43.04% | 6.315 | 6.312 | True |
| 512 | team_rank256_uniform | compact_lookup / paired | 9.50504 | 40.59% | 5.473 | 5.471 | True |
| 512 | team_rank256_uniform | compact_search / paired | 9.50504 | 40.59% | 5.077 | 5.074 | True |
| 512 | team_rank256_uniform | direct_lookup / paired | 9.50504 | 40.59% | 7.384 | 7.381 | True |
| 512 | team_rank256_uniform | direct_search / paired | 9.50504 | 40.59% | 7.409 | 7.406 | True |
| 8192 | exans_official_fixed_framed | native / after | 12.34020 | 22.87% | 23.717 | 23.711 | True |
| 8192 | exans_official_fixed_framed | native / before | 12.34020 | 22.87% | 23.748 | 23.741 | True |
| 8192 | raw_bf16 | native / after | 16.00001 | -0.00% | 3.467 | 3.465 | True |
| 8192 | raw_bf16 | native / before | 16.00001 | -0.00% | 3.481 | 3.479 | True |
| 8192 | splitzip_official_framed | native / after | 12.05013 | 24.69% | 3.440 | 3.438 | True |
| 8192 | splitzip_official_framed | native / before | 12.05013 | 24.69% | 3.441 | 3.439 | True |
| 8192 | team_dense_uniform | compact_lookup / paired | 8.92097 | 44.24% | 33.694 | 33.691 | True |
| 8192 | team_dense_uniform | compact_search / paired | 8.92097 | 44.24% | 33.408 | 33.404 | True |
| 8192 | team_dense_uniform | direct_lookup / paired | 8.92097 | 44.24% | 29.181 | 29.176 | True |
| 8192 | team_dense_uniform | direct_search / paired | 8.92097 | 44.24% | 28.596 | 28.593 | True |
| 8192 | team_rank256_uniform | compact_lookup / paired | 9.30632 | 41.84% | 26.748 | 26.745 | True |
| 8192 | team_rank256_uniform | compact_search / paired | 9.30632 | 41.84% | 26.693 | 26.689 | True |
| 8192 | team_rank256_uniform | direct_lookup / paired | 9.30632 | 41.84% | 21.834 | 21.830 | True |
| 8192 | team_rank256_uniform | direct_search / paired | 9.30632 | 41.84% | 21.766 | 21.762 | True |

Calculated serial wire + full GPU decode, milliseconds. All methods/arms are shown; native before/after rounds remain separate. Ideal overlap max(wire,decode) is a separate unmeasured bound in results.json.

| Tokens | Method | Arm / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | exans_official_fixed_framed | native / after | 367.188 | 7.353 | 5.535 | 4.930 | 4.627 |
| 512 | exans_official_fixed_framed | native / before | 367.174 | 7.339 | 5.522 | 4.916 | 4.613 |
| 512 | raw_bf16 | native / after | 470.127 | 5.060 | 2.711 | 1.929 | 1.537 |
| 512 | raw_bf16 | native / before | 470.139 | 5.072 | 2.724 | 1.941 | 1.549 |
| 512 | splitzip_official_framed | native / after | 354.160 | 4.118 | 2.350 | 1.760 | 1.466 |
| 512 | splitzip_official_framed | native / before | 354.161 | 4.118 | 2.350 | 1.761 | 1.466 |
| 512 | team_dense_uniform | compact_lookup / paired | 272.123 | 7.218 | 5.880 | 5.434 | 5.211 |
| 512 | team_dense_uniform | compact_search / paired | 272.080 | 7.175 | 5.837 | 5.391 | 5.168 |
| 512 | team_dense_uniform | direct_lookup / paired | 273.883 | 8.978 | 7.640 | 7.194 | 6.971 |
| 512 | team_dense_uniform | direct_search / paired | 273.896 | 8.991 | 7.653 | 7.207 | 6.984 |
| 512 | team_rank256_uniform | compact_lookup / paired | 284.543 | 8.264 | 6.869 | 6.404 | 6.171 |
| 512 | team_rank256_uniform | compact_search / paired | 284.146 | 7.867 | 6.472 | 6.007 | 5.774 |
| 512 | team_rank256_uniform | direct_lookup / paired | 286.453 | 10.174 | 8.779 | 8.314 | 8.081 |
| 512 | team_rank256_uniform | direct_search / paired | 286.478 | 10.200 | 8.804 | 8.339 | 8.107 |
| 8192 | exans_official_fixed_framed | native / after | 5820.673 | 81.686 | 52.701 | 43.040 | 38.209 |
| 8192 | exans_official_fixed_framed | native / before | 5820.704 | 81.717 | 52.732 | 43.071 | 38.240 |
| 8192 | raw_bf16 | native / after | 7519.662 | 78.629 | 41.048 | 28.521 | 22.257 |
| 8192 | raw_bf16 | native / before | 7519.676 | 78.642 | 41.062 | 28.535 | 22.271 |
| 8192 | splitzip_official_framed | native / after | 5664.132 | 60.047 | 31.744 | 22.309 | 17.592 |
| 8192 | splitzip_official_framed | native / before | 5664.133 | 60.048 | 31.745 | 22.310 | 17.593 |
| 8192 | team_dense_uniform | compact_lookup / paired | 4224.427 | 75.601 | 54.647 | 47.663 | 44.170 |
| 8192 | team_dense_uniform | compact_search / paired | 4224.141 | 75.315 | 54.361 | 47.377 | 43.884 |
| 8192 | team_dense_uniform | direct_lookup / paired | 4219.914 | 71.088 | 50.135 | 43.150 | 39.658 |
| 8192 | team_dense_uniform | direct_search / paired | 4219.329 | 70.503 | 49.550 | 42.565 | 39.073 |
| 8192 | team_rank256_uniform | compact_lookup / paired | 4398.505 | 70.465 | 48.607 | 41.320 | 37.677 |
| 8192 | team_rank256_uniform | compact_search / paired | 4398.450 | 70.410 | 48.551 | 41.265 | 37.622 |
| 8192 | team_rank256_uniform | direct_lookup / paired | 4393.592 | 65.552 | 43.693 | 36.407 | 32.763 |
| 8192 | team_rank256_uniform | direct_search / paired | 4393.524 | 65.484 | 43.625 | 36.339 | 32.696 |

Per-link budgets and speed ratios versus the faster SplitZip and ExANS control rounds are in results.json. No routing or method selection on TEST is deployed by this sweep.

Separate 8K text profiles record CPU/Torch operators and actual CUDA kernels. Summed kernel work may overlap across streams; it is not wall latency. Stage/eager/clone diagnostics are separate calls and must not be added as a graph decomposition.

| Method | Arm | Full diagnostic ms | Stage ms | Native eager ms | Output clone ms |
|---|---|---:|---:|---:|---:|
| exans_official_fixed_framed | native | 24.204 | — | — | — |
| splitzip_official_framed | native | 3.540 | — | — | — |
| team_dense_uniform | compact_lookup | 34.189 | 2.284 | 27.920 | 3.385 |
| team_dense_uniform | compact_search | 33.848 | 2.356 | 27.846 | 3.378 |
| team_dense_uniform | direct_lookup | 29.161 | 0.386 | — | — |
| team_dense_uniform | direct_search | 29.308 | 0.382 | — | — |
| team_rank256_uniform | compact_lookup | 26.978 | 2.358 | 21.428 | 3.382 |
| team_rank256_uniform | compact_search | 26.898 | 2.446 | 21.496 | 3.392 |
| team_rank256_uniform | direct_lookup | 22.121 | 0.480 | — | — |
| team_rank256_uniform | direct_search | 22.014 | 0.381 | — | — |

All arms require original-word equality, unchanged wire hashes, held-output ownership and mode-specific graph counts. Fixed shared state and scratch are additional to packet bytes. ExANS native allocation peaks are unmeasured.
AQUA-KV is lossy and remains a separate official evaluation. No compressed attention, T-to-P conversion, fresh encode, SSD/network or serving performance is measured here.
