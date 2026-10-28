# Paired Lite CUDA graph decoder experiment

Status: **completed**. GPU measured: **True**.

Same original packet bytes and fixed fitted state. Eager and graph calls alternate AB/BA and alternate two packets with different payload lengths.
Full graph decode includes received-directory validation, staging copies, replay and an independently owned output clone.

| Method | Saved | Complete bpv | Decode ms | Exact |
|---|---:|---:|---:|---|
| raw_bf16 | -0.00% | 16.00009 | 0.349 | True |
| splitzip_official_framed | 24.73% | 12.04281 | 0.555 | True |
| exans_official_fixed_framed | 22.63% | 12.37970 | 3.661 | True |
| lite_v128_b8_eager | 44.17% | 8.93339 | 158.642 | True |
| lite_v128_b8_graph | 44.17% | 8.93339 | 122.853 | True |
| lite_v64_b32_eager | 43.19% | 9.09012 | 156.818 | True |
| lite_v64_b32_graph | 43.19% | 9.09012 | 96.236 | True |

Cold setup and resident memory. Allocated/reserved columns include the resident engine and packets; before/after refer to graph preparation.

| Layout | Cold setup ms | Allocated before → after MiB | Reserved before → after MiB | Peak allocated / reserved MiB | Staging payload MiB | Directory KiB | Captured output MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| lite_v128_b8 | 1258.017 | 388.799 → 590.457 | 454.000 → 704.000 | 677.841 / 758.000 | 112.938 | 224.125 | 56.000 |
| lite_v64_b32 | 1239.731 | 388.580 → 591.191 | 454.000 → 704.000 | 678.575 / 758.000 | 113.877 | 112.125 | 56.000 |

Calculated serial precompressed fetch, not measured transport. No fresh encode projection is produced.

| Gb/s | Method | Transfer + decode ms |
|---:|---|---:|
| 1 | raw_bf16 | 470.114 |
| 1 | splitzip_official_framed | 354.133 |
| 1 | exans_official_fixed_framed | 367.130 |
| 1 | lite_v128_b8_eager | 420.927 |
| 1 | lite_v128_b8_graph | 385.138 |
| 1 | lite_v64_b32_eager | 423.706 |
| 1 | lite_v64_b32_graph | 363.123 |
| 2.5 | raw_bf16 | 188.255 |
| 2.5 | splitzip_official_framed | 141.986 |
| 2.5 | exans_official_fixed_framed | 149.048 |
| 2.5 | lite_v128_b8_eager | 263.556 |
| 2.5 | lite_v128_b8_graph | 227.767 |
| 2.5 | lite_v64_b32_eager | 263.573 |
| 2.5 | lite_v64_b32_graph | 202.991 |
| 5 | raw_bf16 | 94.302 |
| 5 | splitzip_official_framed | 71.270 |
| 5 | exans_official_fixed_framed | 76.354 |
| 5 | lite_v128_b8_eager | 211.099 |
| 5 | lite_v128_b8_graph | 175.310 |
| 5 | lite_v64_b32_eager | 210.196 |
| 5 | lite_v64_b32_graph | 149.613 |
| 10 | raw_bf16 | 47.326 |
| 10 | splitzip_official_framed | 35.913 |
| 10 | exans_official_fixed_framed | 40.007 |
| 10 | lite_v128_b8_eager | 184.870 |
| 10 | lite_v128_b8_graph | 149.081 |
| 10 | lite_v64_b32_eager | 183.507 |
| 10 | lite_v64_b32_graph | 122.925 |

Decode budgets against retimed SplitZip, using each complete packet size.

| Candidate | Gb/s | Decode budget ms | Current decode ms |
|---|---:|---:|---:|
| lite_v128_b8_eager | 1 | 91.848 | 158.642 |
| lite_v128_b8_eager | 2.5 | 37.072 | 158.642 |
| lite_v128_b8_eager | 5 | 18.813 | 158.642 |
| lite_v128_b8_eager | 10 | 9.684 | 158.642 |
| lite_v128_b8_graph | 1 | 91.848 | 122.853 |
| lite_v128_b8_graph | 2.5 | 37.072 | 122.853 |
| lite_v128_b8_graph | 5 | 18.813 | 122.853 |
| lite_v128_b8_graph | 10 | 9.684 | 122.853 |
| lite_v64_b32_eager | 1 | 87.246 | 156.818 |
| lite_v64_b32_eager | 2.5 | 35.231 | 156.818 |
| lite_v64_b32_eager | 5 | 17.893 | 156.818 |
| lite_v64_b32_eager | 10 | 9.224 | 156.818 |
| lite_v64_b32_graph | 1 | 87.246 | 96.236 |
| lite_v64_b32_graph | 2.5 | 35.231 | 96.236 |
| lite_v64_b32_graph | 5 | 17.893 | 96.236 |
| lite_v64_b32_graph | 10 | 9.224 | 96.236 |

Separate synchronized phase diagnostics; do not add these to ordinary paired timings. Raw wall/CUDA samples remain in JSON.

| Layout / document | Validation wall ms | Input-copy CUDA ms | Replay CUDA ms | Replay CPU submit ms | Output-clone CUDA ms |
|---|---:|---:|---:|---:|---:|
| lite_v128_b8 / test_code | 0.603 | 0.152 | 119.652 | 0.062 | 0.328 |
| lite_v128_b8 / test_text | 0.578 | 0.142 | 119.599 | 0.105 | 0.325 |
| lite_v64_b32 / test_code | 0.564 | 0.161 | 93.001 | 0.053 | 0.330 |
| lite_v64_b32 / test_text | 0.548 | 0.167 | 93.315 | 0.076 | 0.328 |

Cold graph warmup/capture, persistent staging, graph-private Torch allocations, sample distributions and allocation peaks are in worker_results.json. Driver graph metadata is not fully covered by Torch accounting.
Both arms share a resident fitted engine and graph buffers during paired timing; allocation baselines include that setup. Uploading serialized packets is measured separately.
Fresh GPU receivers get only fixed state and packet bytes, and verify alternating packets and output ownership. No source KV or reference predictions are used.
Original official baseline workers/environments are unchanged. No fitting, model download, native rebuild, actual network, SSD, pool attention or serving-concurrency measurement is performed.
