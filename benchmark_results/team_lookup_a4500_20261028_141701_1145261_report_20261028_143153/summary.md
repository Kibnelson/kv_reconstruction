# GPU lookup and borrowed-input exact KV decode

Status: **completed**. GPU measured: **True**.

Same frozen Llama-3.2-3B BF16 words, all 28 layers, text/code TEST documents, 512 and 8192 tokens. Packets, predictor coefficients, CDFs and official native controls are unchanged.
Primary timing starts with compressed payload on GPU and covers validation, normal API metadata work, decode and an independently owned output. Host bulk hashing/import/upload, encoding, disk and real networking are excluded.
compact_search is the current staged CUDA-graph decoder; compact_lookup selects the existing exact inverse-CDF lookup. direct_search/direct_lookup borrow the read-only GPU payload and allocate their own output directly. Four-arm rotation/reversal is predeclared; controls run before and after.

| Tokens | Method | Arm / round | Complete bpv | Saved | Full decode ms | CUDA interval ms | Exact |
|---:|---|---|---:|---:|---:|---:|---|
| 512 | exans_official_fixed_framed | native / after | 12.37970 | 22.63% | 3.702 | 3.697 | True |
| 512 | exans_official_fixed_framed | native / before | 12.37970 | 22.63% | 3.716 | 3.710 | True |
| 512 | raw_bf16 | native / after | 16.00009 | -0.00% | 0.368 | 0.365 | True |
| 512 | raw_bf16 | native / before | 16.00009 | -0.00% | 0.362 | 0.361 | True |
| 512 | splitzip_official_framed | native / after | 12.04281 | 24.73% | 0.573 | 0.572 | True |
| 512 | splitzip_official_framed | native / before | 12.04281 | 24.73% | 0.577 | 0.576 | True |
| 512 | team_dense_uniform | compact_lookup / paired | 9.11376 | 43.04% | 4.528 | 4.526 | True |
| 512 | team_dense_uniform | compact_search / paired | 9.11376 | 43.04% | 4.494 | 4.491 | True |
| 512 | team_dense_uniform | direct_lookup / paired | 9.11376 | 43.04% | 6.439 | 6.437 | True |
| 512 | team_dense_uniform | direct_search / paired | 9.11376 | 43.04% | 6.369 | 6.366 | True |
| 512 | team_rank256_uniform | compact_lookup / paired | 9.50504 | 40.59% | 5.451 | 5.448 | True |
| 512 | team_rank256_uniform | compact_search / paired | 9.50504 | 40.59% | 5.028 | 5.025 | True |
| 512 | team_rank256_uniform | direct_lookup / paired | 9.50504 | 40.59% | 7.651 | 7.648 | True |
| 512 | team_rank256_uniform | direct_search / paired | 9.50504 | 40.59% | 7.625 | 7.621 | True |
| 8192 | exans_official_fixed_framed | native / after | 12.34020 | 22.87% | 23.706 | 23.700 | True |
| 8192 | exans_official_fixed_framed | native / before | 12.34020 | 22.87% | 23.771 | 23.765 | True |
| 8192 | raw_bf16 | native / after | 16.00001 | -0.00% | 3.475 | 3.473 | True |
| 8192 | raw_bf16 | native / before | 16.00001 | -0.00% | 3.473 | 3.471 | True |
| 8192 | splitzip_official_framed | native / after | 12.05013 | 24.69% | 3.430 | 3.428 | True |
| 8192 | splitzip_official_framed | native / before | 12.05013 | 24.69% | 3.429 | 3.427 | True |
| 8192 | team_dense_uniform | compact_lookup / paired | 8.92097 | 44.24% | 33.462 | 33.458 | True |
| 8192 | team_dense_uniform | compact_search / paired | 8.92097 | 44.24% | 33.616 | 33.613 | True |
| 8192 | team_dense_uniform | direct_lookup / paired | 8.92097 | 44.24% | 28.900 | 28.896 | True |
| 8192 | team_dense_uniform | direct_search / paired | 8.92097 | 44.24% | 28.151 | 28.147 | True |
| 8192 | team_rank256_uniform | compact_lookup / paired | 9.30632 | 41.84% | 26.812 | 26.809 | True |
| 8192 | team_rank256_uniform | compact_search / paired | 9.30632 | 41.84% | 26.703 | 26.699 | True |
| 8192 | team_rank256_uniform | direct_lookup / paired | 9.30632 | 41.84% | 21.803 | 21.799 | True |
| 8192 | team_rank256_uniform | direct_search / paired | 9.30632 | 41.84% | 21.709 | 21.704 | True |

Calculated serial wire + full GPU decode, milliseconds. All methods/arms are shown; native before/after rounds remain separate. Ideal overlap max(wire,decode) is a separate unmeasured bound in results.json.

| Tokens | Method | Arm / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | exans_official_fixed_framed | native / after | 367.172 | 7.337 | 5.520 | 4.914 | 4.611 |
| 512 | exans_official_fixed_framed | native / before | 367.186 | 7.351 | 5.533 | 4.928 | 4.625 |
| 512 | raw_bf16 | native / after | 470.132 | 5.066 | 2.717 | 1.934 | 1.542 |
| 512 | raw_bf16 | native / before | 470.127 | 5.060 | 2.711 | 1.928 | 1.536 |
| 512 | splitzip_official_framed | native / after | 354.152 | 4.109 | 2.341 | 1.752 | 1.457 |
| 512 | splitzip_official_framed | native / before | 354.156 | 4.113 | 2.345 | 1.756 | 1.461 |
| 512 | team_dense_uniform | compact_lookup / paired | 272.109 | 7.204 | 5.866 | 5.420 | 5.197 |
| 512 | team_dense_uniform | compact_search / paired | 272.075 | 7.170 | 5.832 | 5.386 | 5.163 |
| 512 | team_dense_uniform | direct_lookup / paired | 274.021 | 9.115 | 7.777 | 7.331 | 7.108 |
| 512 | team_dense_uniform | direct_search / paired | 273.950 | 9.045 | 7.707 | 7.261 | 7.038 |
| 512 | team_rank256_uniform | compact_lookup / paired | 284.521 | 8.242 | 6.847 | 6.381 | 6.149 |
| 512 | team_rank256_uniform | compact_search / paired | 284.097 | 7.819 | 6.423 | 5.958 | 5.726 |
| 512 | team_rank256_uniform | direct_lookup / paired | 286.720 | 10.441 | 9.046 | 8.581 | 8.348 |
| 512 | team_rank256_uniform | direct_search / paired | 286.694 | 10.415 | 9.020 | 8.555 | 8.322 |
| 8192 | exans_official_fixed_framed | native / after | 5820.663 | 81.676 | 52.691 | 43.029 | 38.199 |
| 8192 | exans_official_fixed_framed | native / before | 5820.727 | 81.740 | 52.756 | 43.094 | 38.263 |
| 8192 | raw_bf16 | native / after | 7519.670 | 78.637 | 41.056 | 28.529 | 22.265 |
| 8192 | raw_bf16 | native / before | 7519.668 | 78.635 | 41.054 | 28.527 | 22.263 |
| 8192 | splitzip_official_framed | native / after | 5664.122 | 60.037 | 31.733 | 22.299 | 17.582 |
| 8192 | splitzip_official_framed | native / before | 5664.121 | 60.036 | 31.733 | 22.298 | 17.581 |
| 8192 | team_dense_uniform | compact_lookup / paired | 4224.195 | 75.369 | 54.415 | 47.431 | 43.938 |
| 8192 | team_dense_uniform | compact_search / paired | 4224.350 | 75.524 | 54.570 | 47.586 | 44.093 |
| 8192 | team_dense_uniform | direct_lookup / paired | 4219.633 | 70.807 | 49.854 | 42.869 | 39.377 |
| 8192 | team_dense_uniform | direct_search / paired | 4218.884 | 70.058 | 49.104 | 42.120 | 38.627 |
| 8192 | team_rank256_uniform | compact_lookup / paired | 4398.569 | 70.530 | 48.671 | 41.384 | 37.741 |
| 8192 | team_rank256_uniform | compact_search / paired | 4398.460 | 70.420 | 48.561 | 41.275 | 37.632 |
| 8192 | team_rank256_uniform | direct_lookup / paired | 4393.561 | 65.521 | 43.662 | 36.376 | 32.733 |
| 8192 | team_rank256_uniform | direct_search / paired | 4393.466 | 65.426 | 43.567 | 36.281 | 32.638 |

Per-link budgets and speed ratios versus the faster SplitZip and ExANS control rounds are in results.json. No routing or method selection on TEST is deployed by this sweep.

Separate 8K text profiles record CPU/Torch operators and actual CUDA kernels. Summed kernel work may overlap across streams; it is not wall latency. Stage/eager/clone diagnostics are separate calls and must not be added as a graph decomposition.

| Method | Arm | Full diagnostic ms | Stage ms | Native eager ms | Output clone ms |
|---|---|---:|---:|---:|---:|
| exans_official_fixed_framed | native | 24.201 | — | — | — |
| splitzip_official_framed | native | 3.543 | — | — | — |
| team_dense_uniform | compact_lookup | 33.351 | 2.279 | 27.768 | 3.390 |
| team_dense_uniform | compact_search | 33.232 | 2.361 | 27.926 | 3.384 |
| team_dense_uniform | direct_lookup | 28.624 | 0.482 | — | — |
| team_dense_uniform | direct_search | 28.294 | 0.425 | — | — |
| team_rank256_uniform | compact_lookup | 26.895 | 2.382 | 21.450 | 3.473 |
| team_rank256_uniform | compact_search | 27.008 | 2.365 | 21.514 | 3.480 |
| team_rank256_uniform | direct_lookup | 21.800 | 0.382 | — | — |
| team_rank256_uniform | direct_search | 22.051 | 0.474 | — | — |

All arms require original-word equality, unchanged wire hashes, held-output ownership and mode-specific graph counts. Fixed shared state and scratch are additional to packet bytes. ExANS native allocation peaks are unmeasured.
AQUA-KV is lossy and remains a separate official evaluation. No compressed attention, T-to-P conversion, fresh encode, SSD/network or serving performance is measured here.
