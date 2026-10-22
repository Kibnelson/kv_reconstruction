# Matched Qwen GPU transfer comparison

Status: **failed**. GPU measured: **True**.

Original pinned Qwen BF16 cache; every method receives identical HF-layout words.
Pilot uses four of 36 layers; full uses all layers. This is a new calibration/workload, so the earlier 44.67% is an anchor, not this table’s expected result.

| Track | Method | Request tokens | Complete bpv | Saved | Encode ms | Decode ms | Exact |
|---|---|---:|---:|---:|---:|---:|---|
| wikitext2 | raw_bf16 | 32 | 16.01347 | -0.08% | 0.151 | 0.184 | True |
| wikitext2 | raw_bf16 | 128 | 16.00339 | -0.02% | 0.173 | 0.185 | True |
| wikitext2 | block32_A | 32 | 9.53426 | 40.41% | 118.338 | 173.140 | True |
| wikitext2 | block32_A | 128 | 9.42146 | 41.12% | 466.150 | 647.323 | True |
| wikitext2 | block32_B | 32 | 9.68953 | 39.44% | 189.005 | 208.817 | True |
| wikitext2 | block32_B | 128 | 9.54620 | 40.34% | 750.685 | 770.794 | True |
| wikitext2 | previous32_B | 32 | 10.04948 | 37.19% | 177.384 | 196.144 | True |
| wikitext2 | previous32_B | 128 | 9.95897 | 37.76% | 702.696 | 728.157 | True |
| wikitext2 | aqua_style32_B | 32 | 9.93895 | 37.88% | 175.187 | 194.798 | True |
| wikitext2 | aqua_style32_B | 128 | 9.80540 | 38.72% | 695.420 | 732.411 | True |
| wikitext2 | strong_layers32_B | 32 | 9.88098 | 38.24% | 202.897 | 222.551 | True |
| wikitext2 | strong_layers32_B | 128 | 9.70033 | 39.37% | 804.903 | 825.187 | True |
| wikitext2 | strong_layers64_B | 32 | 9.93684 | 37.89% | 116.880 | 182.913 | True |
| wikitext2 | strong_layers64_B | 128 | 9.75394 | 39.04% | 460.815 | 687.279 | True |
| wikitext2 | splitzip_native | 32 | 12.32648 | 22.96% | 0.422 | 0.491 | True |
| wikitext2 | splitzip_native | 128 | 12.31104 | 23.06% | 0.430 | 0.493 | True |
| c4 | raw_bf16 | 32 | 16.01347 | -0.08% | 0.152 | 0.185 | True |
| c4 | raw_bf16 | 128 | 16.00339 | -0.02% | 0.149 | 0.180 | True |
| c4 | block32_A | 32 | 9.62308 | 39.86% | 117.934 | 172.779 | True |
| c4 | block32_A | 128 | 9.58343 | 40.10% | 466.341 | 646.801 | True |
| c4 | block32_B | 32 | 10.00737 | 37.45% | 188.705 | 209.022 | True |
| c4 | block32_B | 128 | 9.89429 | 38.16% | 749.663 | 770.771 | True |
| c4 | previous32_B | 32 | 10.02644 | 37.33% | 177.001 | 195.909 | True |
| c4 | previous32_B | 128 | 9.96008 | 37.75% | 702.988 | 725.799 | True |
| c4 | aqua_style32_B | 32 | 10.17809 | 36.39% | 174.962 | 194.808 | True |
| c4 | aqua_style32_B | 128 | 10.08652 | 36.96% | 692.736 | 724.314 | True |
| c4 | strong_layers32_B | 32 | 10.30101 | 35.62% | 202.898 | 222.428 | True |
| c4 | strong_layers32_B | 128 | 10.16435 | 36.47% | 803.382 | 826.680 | True |
| c4 | strong_layers64_B | 32 | 10.36603 | 35.21% | 116.544 | 182.992 | True |
| c4 | strong_layers64_B | 128 | 10.22751 | 36.08% | 459.258 | 691.491 | True |
| c4 | splitzip_native | 32 | 12.33192 | 22.93% | 0.426 | 0.515 | True |
| c4 | splitzip_native | 128 | 12.31767 | 23.01% | 0.441 | 0.489 | True |

| Track | Added context | Tokens | bpv saved vs B | Encode speedup | Decode speedup |
|---|---|---:|---:|---:|---:|
| wikitext2 | strong_layers32_B | 32 | -0.19145 | 0.932× | 0.938× |
| wikitext2 | strong_layers32_B | 128 | -0.15413 | 0.933× | 0.934× |
| wikitext2 | strong_layers64_B | 32 | -0.24731 | 1.617× | 1.142× |
| wikitext2 | strong_layers64_B | 128 | -0.20774 | 1.629× | 1.122× |
| c4 | strong_layers32_B | 32 | -0.29364 | 0.930× | 0.940× |
| c4 | strong_layers32_B | 128 | -0.27006 | 0.933× | 0.932× |
| c4 | strong_layers64_B | 32 | -0.35866 | 1.619× | 1.142× |
| c4 | strong_layers64_B | 128 | -0.33321 | 1.632× | 1.115× |

Positive bpv differences mean smaller complete packets. Speedup below 1 means slower. Inspect both tracks and shared state before promoting a candidate.

| Track | Method | Shared GPU state MiB |
|---|---|---:|
| wikitext2 | raw_bf16 | 0.000 |
| wikitext2 | block32_A | 87.219 |
| wikitext2 | block32_B | 87.860 |
| wikitext2 | previous32_B | 54.391 |
| wikitext2 | aqua_style32_B | 45.891 |
| wikitext2 | strong_layers32_B | 138.860 |
| wikitext2 | strong_layers64_B | 134.352 |
| wikitext2 | splitzip_native | 0.001 |
| c4 | raw_bf16 | 0.000 |
| c4 | block32_A | 87.219 |
| c4 | block32_B | 87.860 |
| c4 | previous32_B | 54.391 |
| c4 | aqua_style32_B | 45.891 |
| c4 | strong_layers32_B | 138.860 |
| c4 | strong_layers64_B | 134.352 |
| c4 | splitzip_native | 0.001 |

nvCOMP native persistent state and allocation peaks are unmeasured.

Sizes include a common checked model/state/layout envelope and all required packet parts. Shared calibration is additional.
Times are means of per-request synchronized wall medians; CUDA intervals, arithmetic means, samples and Torch peaks are in results.json.
Input uploads, offline fitting and compressed-byte serialization are separate. Per-method file cycles include compressed export, fsync, read, upload and decode; filesystem cache remains warm.
Native SplitZip charges its actual 7-byte escape entries. nvCOMP consumes the same exact words through an int16 view in lossless mode.

**Scope:** strong_layers adds same-token layer 0 and l−2 to previous-layer plus earlier-group context. It does not implement the team’s attention predictor, pivot order, repeat-token context or value-slice coder; 51.4% is not an achieved GPU result.
AQUA-style is a predictor ablation inside our exact coder, not the published lossy AQUA-KV system. FP8 speculative-model coding and ExANS remain pending adapters.
This benchmark measures GPU transfer codecs. It does not measure P1 attention, T-to-P conversion, cold SSD, network or serving throughput.

| Track | Unavailable/failed method | Reason |
|---|---|---|
| wikitext2 | nvcomp_bitcomp | ValueError('Unknown output data type') |
| wikitext2 | nvcomp_cascaded | ValueError('Unknown output data type') |
| c4 | nvcomp_bitcomp | ValueError('Unknown output data type') |
| c4 | nvcomp_cascaded | ValueError('Unknown output data type') |
