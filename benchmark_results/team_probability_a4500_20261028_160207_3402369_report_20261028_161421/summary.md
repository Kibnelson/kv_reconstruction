# Fixed-prior head-spread calibration versus GPU baselines

Status: **completed**. GPU measured: **True**.

Llama-3.2-3B, all 28 original BF16 layers. Only TABLES selects per-head spread factors from 0.5/0.75/1/1.25/1.5 using actual compact GPU packet words. Predictor tensors, inverse surprise scales and decoder kernel structure/launch configuration are unchanged. Data-dependent coding work may change. TEST never selects parameters.
Both variants use the one-warp reference binary-search decoder: graph for 512 tokens, direct for 8192. Native controls retain their original implementations and settings.

| Tokens | Method | Variant / round | Complete bpv | Packet MiB | Decode ms | Exact |
|---:|---|---|---:|---:|---:|---|
| 512 | exans_official_fixed_framed | after | 12.379698 | 43.329 | 3.704 | True |
| 512 | exans_official_fixed_framed | before | 12.379698 | 43.329 | 3.713 | True |
| 512 | raw_bf16 | after | 16.000086 | 56.000 | 0.364 | True |
| 512 | raw_bf16 | before | 16.000086 | 56.000 | 0.359 | True |
| 512 | splitzip_official_framed | after | 12.042813 | 42.150 | 0.577 | True |
| 512 | splitzip_official_framed | before | 12.042813 | 42.150 | 0.568 | True |
| 512 | team_dense_uniform | calibrated | 9.113758 | 31.898 | 4.490 | True |
| 512 | team_dense_uniform | control | 9.113758 | 31.898 | 4.491 | True |
| 512 | team_rank256_uniform | calibrated | 9.505044 | 33.268 | 5.141 | True |
| 512 | team_rank256_uniform | control | 9.505044 | 33.268 | 5.079 | True |
| 8192 | exans_official_fixed_framed | after | 12.340198 | 691.051 | 23.731 | True |
| 8192 | exans_official_fixed_framed | before | 12.340198 | 691.051 | 23.798 | True |
| 8192 | raw_bf16 | after | 16.000005 | 896.000 | 3.470 | True |
| 8192 | raw_bf16 | before | 16.000005 | 896.000 | 3.475 | True |
| 8192 | splitzip_official_framed | after | 12.050127 | 674.807 | 3.441 | True |
| 8192 | splitzip_official_framed | before | 12.050127 | 674.807 | 3.441 | True |
| 8192 | team_dense_uniform | calibrated | 8.920970 | 499.574 | 28.349 | True |
| 8192 | team_dense_uniform | control | 8.920970 | 499.574 | 28.261 | True |
| 8192 | team_rank256_uniform | calibrated | 9.306323 | 521.154 | 21.662 | True |
| 8192 | team_rank256_uniform | control | 9.306323 | 521.154 | 21.683 | True |

TABLES-selected fixed factors (K heads then V heads; shared state, no per-request hints):
- team_dense_uniform: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- team_rank256_uniform: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

Cold fixed-state archive and manifest: control 146.250 MiB; calibrated 146.251 MiB. Warm link models below assume this state is already shared; a cold state transfer must additionally charge these bytes.

Calculated serial wire + measured full GPU decode, milliseconds. Networking is not measured. Ideal overlap remains a separate bound in results.json.

| Tokens | Method | Variant / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | exans_official_fixed_framed | after | 367.174 | 7.339 | 5.522 | 4.916 | 4.613 |
| 512 | exans_official_fixed_framed | before | 367.183 | 7.348 | 5.531 | 4.925 | 4.622 |
| 512 | raw_bf16 | after | 470.129 | 5.062 | 2.713 | 1.930 | 1.539 |
| 512 | raw_bf16 | before | 470.124 | 5.057 | 2.708 | 1.925 | 1.533 |
| 512 | splitzip_official_framed | after | 354.156 | 4.113 | 2.345 | 1.756 | 1.461 |
| 512 | splitzip_official_framed | before | 354.147 | 4.104 | 2.336 | 1.747 | 1.452 |
| 512 | team_dense_uniform | calibrated | 272.072 | 7.166 | 5.828 | 5.382 | 5.159 |
| 512 | team_dense_uniform | control | 272.072 | 7.167 | 5.829 | 5.383 | 5.160 |
| 512 | team_rank256_uniform | calibrated | 284.210 | 7.932 | 6.537 | 6.071 | 5.839 |
| 512 | team_rank256_uniform | control | 284.149 | 7.870 | 6.475 | 6.010 | 5.777 |
| 8192 | exans_official_fixed_framed | after | 5820.688 | 81.701 | 52.716 | 43.054 | 38.223 |
| 8192 | exans_official_fixed_framed | before | 5820.754 | 81.767 | 52.782 | 43.121 | 38.290 |
| 8192 | raw_bf16 | after | 7519.665 | 78.632 | 41.051 | 28.524 | 22.260 |
| 8192 | raw_bf16 | before | 7519.671 | 78.637 | 41.056 | 28.529 | 22.266 |
| 8192 | splitzip_official_framed | after | 5664.133 | 60.048 | 31.744 | 22.310 | 17.593 |
| 8192 | splitzip_official_framed | before | 5664.133 | 60.048 | 31.745 | 22.310 | 17.593 |
| 8192 | team_dense_uniform | calibrated | 4219.082 | 70.256 | 49.303 | 42.318 | 38.826 |
| 8192 | team_dense_uniform | control | 4218.994 | 70.168 | 49.214 | 42.230 | 38.737 |
| 8192 | team_rank256_uniform | calibrated | 4393.419 | 65.379 | 43.520 | 36.234 | 32.591 |
| 8192 | team_rank256_uniform | control | 4393.441 | 65.401 | 43.542 | 36.256 | 32.612 |

Own full encode means of two document medians (export separately recorded; native encode not remeasured):

| Tokens | Method | Variant | Encode ms |
|---:|---|---|---:|
| 512 | team_dense_uniform | control | 23.036 |
| 512 | team_dense_uniform | calibrated | 23.490 |
| 512 | team_rank256_uniform | control | 7.425 |
| 512 | team_rank256_uniform | calibrated | 7.446 |
| 8192 | team_dense_uniform | control | 81.067 |
| 8192 | team_dense_uniform | calibrated | 82.438 |
| 8192 | team_rank256_uniform | control | 41.994 |
| 8192 | team_rank256_uniform | calibrated | 41.855 |

Primary decode starts with compressed payload on GPU and includes normal API validation/metadata work and an independently owned exact output. Bulk host hashing/import/upload, disk, real networking and calibration are excluded. Complete packet bytes include framing, directories, states, alignment and checksum. Fixed shared state and scratch are additional.
No fresh end-to-end native encode comparison, NIC/SSD, compressed attention, T-to-P conversion or serving result is claimed. Smaller TEST packets are a hypothesis, not guaranteed by TABLES selection.
