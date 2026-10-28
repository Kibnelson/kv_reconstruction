# GPU-resident exact KV decode versus official controls

Status: **completed**. GPU measured: **True**.

Frozen original BF16 words, all28 layers, two TEST documents, 512/8192 tokens. Packets, fitted predictors and official native method bodies are unchanged.
Timing starts with compressed payload on GPU and includes the full decode API, its metadata validation/staging and an independently owned BF16 output. Host packet export/checksum/import/upload, encode, SSD and real network are excluded.
Means of per-document synchronized wall medians. Reference and direct-compact arms are paired; official controls are retimed before and after. Native graph replay is diagnostic only.

| Tokens | Method | Arm / round | Complete bpv | Saved | Full decode ms | CUDA interval ms | Native graph wall ms | Exact |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 512 | exans_official_fixed_framed | native / after | 12.37970 | 22.63% | 3.708 | 3.702 | — | True |
| 512 | exans_official_fixed_framed | native / before | 12.37970 | 22.63% | 3.708 | 3.703 | — | True |
| 512 | raw_bf16 | native / after | 16.00009 | -0.00% | 0.366 | 0.364 | — | True |
| 512 | raw_bf16 | native / before | 16.00009 | -0.00% | 0.374 | 0.374 | — | True |
| 512 | splitzip_official_framed | native / after | 12.04281 | 24.73% | 0.573 | 0.574 | — | True |
| 512 | splitzip_official_framed | native / before | 12.04281 | 24.73% | 0.576 | 0.574 | — | True |
| 512 | team_dense_uniform | compact / paired | 9.11376 | 43.04% | 4.486 | 4.484 | 3.869 | True |
| 512 | team_dense_uniform | reference / paired | 9.11376 | 43.04% | 4.698 | 4.696 | 3.872 | True |
| 512 | team_rank256_uniform | compact / paired | 9.50504 | 40.59% | 5.088 | 5.086 | 4.455 | True |
| 512 | team_rank256_uniform | reference / paired | 9.50504 | 40.59% | 5.303 | 5.301 | 4.473 | True |
| 8192 | exans_official_fixed_framed | native / after | 12.34020 | 22.87% | 23.743 | 23.737 | — | True |
| 8192 | exans_official_fixed_framed | native / before | 12.34020 | 22.87% | 23.728 | 23.721 | — | True |
| 8192 | raw_bf16 | native / after | 16.00001 | -0.00% | 3.475 | 3.473 | — | True |
| 8192 | raw_bf16 | native / before | 16.00001 | -0.00% | 3.471 | 3.469 | — | True |
| 8192 | splitzip_official_framed | native / after | 12.05013 | 24.69% | 3.429 | 3.427 | — | True |
| 8192 | splitzip_official_framed | native / before | 12.05013 | 24.69% | 3.434 | 3.432 | — | True |
| 8192 | team_dense_uniform | compact / paired | 8.92097 | 44.24% | 33.841 | 33.838 | 27.888 | True |
| 8192 | team_dense_uniform | reference / paired | 8.92097 | 44.24% | 35.654 | 35.651 | 27.794 | True |
| 8192 | team_rank256_uniform | compact / paired | 9.30632 | 41.84% | 26.654 | 26.649 | 21.107 | True |
| 8192 | team_rank256_uniform | reference / paired | 9.30632 | 41.84% | 28.764 | 28.761 | 21.096 | True |

Calculated serial wire+decode models below use the faster SplitZip before/after control as an explicit conservative target. They do not measure a network or host ingress. Ideal overlap bounds remain separate in results.json.

| Tokens | Own compact method | Gb/s | Wire + decode ms | SplitZip wire + decode ms | Decode budget to tie ms |
|---:|---|---:|---:|---:|---:|
| 512 | team_dense_uniform | 1 | 272.067 | 354.152 | 86.571 |
| 512 | team_dense_uniform | 100 | 7.162 | 4.109 | 1.433 |
| 512 | team_dense_uniform | 200 | 5.824 | 2.341 | 1.003 |
| 512 | team_dense_uniform | 300 | 5.378 | 1.752 | 0.860 |
| 512 | team_dense_uniform | 400 | 5.155 | 1.457 | 0.788 |
| 512 | team_rank256_uniform | 1 | 284.157 | 354.152 | 75.083 |
| 512 | team_rank256_uniform | 100 | 7.879 | 4.109 | 1.319 |
| 512 | team_rank256_uniform | 200 | 6.483 | 2.341 | 0.946 |
| 512 | team_rank256_uniform | 300 | 6.018 | 1.752 | 0.822 |
| 512 | team_rank256_uniform | 400 | 5.785 | 1.457 | 0.760 |
| 8192 | team_dense_uniform | 1 | 4224.574 | 5664.121 | 1473.388 |
| 8192 | team_dense_uniform | 100 | 75.748 | 60.036 | 18.128 |
| 8192 | team_dense_uniform | 200 | 54.795 | 31.732 | 10.779 |
| 8192 | team_dense_uniform | 300 | 47.810 | 22.298 | 8.329 |
| 8192 | team_dense_uniform | 400 | 44.318 | 17.581 | 7.104 |
| 8192 | team_rank256_uniform | 1 | 4398.412 | 5664.121 | 1292.364 |
| 8192 | team_rank256_uniform | 100 | 70.372 | 60.036 | 16.318 |
| 8192 | team_rank256_uniform | 200 | 48.513 | 31.732 | 9.874 |
| 8192 | team_rank256_uniform | 300 | 41.227 | 22.298 | 7.725 |
| 8192 | team_rank256_uniform | 400 | 37.583 | 17.581 | 6.651 |

Separate 8K text profiles archive cProfile, Torch operators and actual CUDA kernel-event summaries. Full Chrome traces stay local. The eager/stage/clone diagnostics are measured separately and must not be summed as a decomposition of graph replay.

| Method | Arm | Full diagnostic ms | Stage ms | Native eager ms | Output clone ms |
|---|---|---:|---:|---:|---:|
| exans_official_fixed_framed | native | 24.152 | — | — | — |
| splitzip_official_framed | native | 3.643 | — | — | — |
| team_dense_uniform | compact | 33.861 | 2.415 | 27.944 | 3.380 |
| team_dense_uniform | reference | 35.287 | 4.294 | 27.806 | 3.392 |
| team_rank256_uniform | compact | 26.762 | 2.375 | 21.456 | 3.478 |
| team_rank256_uniform | reference | 28.911 | 4.378 | 21.541 | 3.471 |

Direct compact decoding changes buffer layout/start addresses while preserving predictor, probability and entropy-kernel arithmetic and every packet byte. Explicit per-codec accounting is separate from paired-process global peaks. ExANS native allocation peaks are unmeasured.
AQUA-KV is lossy and its official evaluation remains blocked; no adapted AQUA result is substituted in this exact comparison. This test does not measure compressed attention, pool conversion, fresh encoding or serving.
