# Matched Qwen GPU transfer comparison

Status: **completed**. GPU measured: **True**.

Original pinned Qwen BF16 cache; every method receives identical HF-layout words.
Pilot uses four of 36 layers; full uses all layers. This is a new calibration/workload, so the earlier 44.67% is an anchor, not this table’s expected result.

| Track | Method | Request tokens | Complete bpv | Saved | Encode ms | Decode ms | Exact |
|---|---|---:|---:|---:|---:|---:|---|
| wikitext2 | raw_bf16 | 32 | 16.01347 | -0.08% | 0.137 | 0.150 | True |
| wikitext2 | raw_bf16 | 128 | 16.00339 | -0.02% | 0.145 | 0.153 | True |
| wikitext2 | raw_bf16 | 488 | 16.00089 | -0.01% | 0.136 | 0.156 | True |
| wikitext2 | block32_A | 32 | 9.53426 | 40.41% | 121.560 | 171.601 | True |
| wikitext2 | block32_A | 128 | 9.42146 | 41.12% | 481.157 | 623.775 | True |
| wikitext2 | block32_A | 488 | 9.42648 | 41.08% | 1927.636 | 2400.806 | True |
| wikitext2 | block32_B | 32 | 9.68953 | 39.44% | 193.191 | 205.931 | True |
| wikitext2 | block32_B | 128 | 9.54620 | 40.34% | 772.672 | 749.437 | True |
| wikitext2 | block32_B | 488 | 9.55101 | 40.31% | 3087.394 | 2931.331 | True |
| wikitext2 | aqua_style32_A | 32 | 9.72029 | 39.25% | 107.053 | 158.143 | True |
| wikitext2 | aqua_style32_A | 128 | 9.62856 | 39.82% | 426.213 | 582.476 | True |
| wikitext2 | aqua_style32_A | 488 | 9.62593 | 39.84% | 1702.843 | 2217.357 | True |
| wikitext2 | splitzip_native | 32 | 12.32648 | 22.96% | 0.357 | 0.407 | True |
| wikitext2 | splitzip_native | 128 | 12.31104 | 23.06% | 0.380 | 0.409 | True |
| wikitext2 | splitzip_native | 488 | 12.30390 | 23.10% | 0.375 | 0.407 | True |
| wikitext2 | exans_native | 32 | 20.76579 | -29.79% | 1.853 | 1.582 | True |
| wikitext2 | exans_native | 128 | 15.17791 | 5.14% | 3.246 | 2.284 | True |
| wikitext2 | exans_native | 488 | 11.45155 | 28.43% | 4.584 | 2.957 | True |
| c4 | raw_bf16 | 32 | 16.01347 | -0.08% | 0.151 | 0.179 | True |
| c4 | raw_bf16 | 128 | 16.00339 | -0.02% | 0.149 | 0.178 | True |
| c4 | raw_bf16 | 488 | 16.00089 | -0.01% | 0.153 | 0.176 | True |
| c4 | block32_A | 32 | 9.62308 | 39.86% | 121.840 | 172.733 | True |
| c4 | block32_A | 128 | 9.58343 | 40.10% | 482.029 | 648.569 | True |
| c4 | block32_A | 488 | 9.53663 | 40.40% | 1927.590 | 2451.996 | True |
| c4 | block32_B | 32 | 10.00737 | 37.45% | 195.298 | 209.101 | True |
| c4 | block32_B | 128 | 9.89429 | 38.16% | 776.309 | 774.503 | True |
| c4 | block32_B | 488 | 9.84759 | 38.45% | 3104.692 | 2963.478 | True |
| c4 | aqua_style32_A | 32 | 9.76618 | 38.96% | 108.544 | 158.332 | True |
| c4 | aqua_style32_A | 128 | 9.72647 | 39.21% | 429.121 | 595.680 | True |
| c4 | aqua_style32_A | 488 | 9.68501 | 39.47% | 1719.919 | 2244.188 | True |
| c4 | splitzip_native | 32 | 12.33192 | 22.93% | 0.446 | 0.487 | True |
| c4 | splitzip_native | 128 | 12.31767 | 23.01% | 0.443 | 0.492 | True |
| c4 | splitzip_native | 488 | 12.31021 | 23.06% | 0.443 | 0.517 | True |
| c4 | exans_native | 32 | 20.78056 | -29.88% | 1.926 | 1.646 | True |
| c4 | exans_native | 128 | 15.18443 | 5.10% | 3.319 | 2.405 | True |
| c4 | exans_native | 488 | 11.45302 | 28.42% | 4.655 | 3.071 | True |

AQUA-KV with released Quanto residual quantization (lossy single-page adaptation):

| Track | Method | Tokens | Complete bpv | Saved | Encode ms | Decode ms | KV relative MSE | Word match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| wikitext2 | aquakv_quanto2 | 32 | 6.50835 | 59.32% | 4.275 | 2.353 | 0.00407037 | 23.38% |
| wikitext2 | aquakv_quanto2 | 128 | 6.40863 | 59.95% | 4.242 | 2.310 | 0.00363678 | 23.16% |
| wikitext2 | aquakv_quanto2 | 488 | 6.38387 | 60.10% | 4.351 | 2.354 | 0.00359332 | 22.94% |
| wikitext2 | aquakv_quanto4 | 32 | 8.00853 | 49.95% | 4.071 | 2.307 | 9.82319e-05 | 26.08% |
| wikitext2 | aquakv_quanto4 | 128 | 7.90868 | 50.57% | 4.071 | 2.326 | 9.04549e-05 | 25.92% |
| wikitext2 | aquakv_quanto4 | 488 | 7.88387 | 50.73% | 4.272 | 2.350 | 9.14697e-05 | 25.74% |
| c4 | aquakv_quanto2 | 32 | 6.50835 | 59.32% | 4.315 | 2.380 | 0.00417888 | 23.32% |
| c4 | aquakv_quanto2 | 128 | 6.40863 | 59.95% | 4.500 | 2.457 | 0.00362068 | 23.11% |
| c4 | aquakv_quanto2 | 488 | 6.38387 | 60.10% | 4.425 | 2.406 | 0.00345148 | 22.92% |
| c4 | aquakv_quanto4 | 32 | 8.00853 | 49.95% | 4.110 | 2.428 | 0.000102569 | 25.74% |
| c4 | aquakv_quanto4 | 128 | 7.90868 | 50.57% | 4.104 | 2.336 | 9.37234e-05 | 25.66% |
| c4 | aquakv_quanto4 | 488 | 7.88387 | 50.73% | 4.263 | 2.390 | 9.11995e-05 | 25.49% |

First-layer raw anchor, quantizer scales/shifts, packed codes and cos/sin are included. Predictors are fitted separately on reconstructed TRAIN KV. No perplexity/LongBench evaluation, HIGGS, recent buffer or attention sinks are measured.

Positive bpv differences mean smaller complete packets. Speedup below 1 means slower. Compare against fixed-width A as well as B before promoting a candidate.

| Track | Method | Shared GPU state MiB |
|---|---|---:|
| wikitext2 | raw_bf16 | 0.000 |
| wikitext2 | block32_A | 87.219 |
| wikitext2 | block32_B | 87.860 |
| wikitext2 | aqua_style32_A | 45.250 |
| wikitext2 | splitzip_native | 0.001 |
| wikitext2 | exans_native | unmeasured |
| wikitext2 | aquakv_quanto2 | 18.012 |
| wikitext2 | aquakv_quanto4 | 18.012 |
| c4 | raw_bf16 | 0.000 |
| c4 | block32_A | 87.219 |
| c4 | block32_B | 87.860 |
| c4 | aqua_style32_A | 45.250 |
| c4 | splitzip_native | 0.001 |
| c4 | exans_native | unmeasured |
| c4 | aquakv_quanto2 | 18.012 |
| c4 | aquakv_quanto4 | 18.012 |

nvCOMP native persistent state and allocation peaks are unmeasured.

Sizes include a common checked model/state/layout envelope and all required packet parts. Shared calibration is additional.
Times are means of per-request synchronized wall medians; CUDA intervals, arithmetic means, samples and Torch peaks are in results.json.
Input uploads, offline fitting and compressed-byte serialization are separate. Per-method file cycles include compressed export, fsync, read, upload and decode; filesystem cache remains warm.
Native SplitZip charges its actual 7-byte escape entries. nvCOMP consumes the same exact words through an int16 view in lossless mode.

**Scope:** strong_layers adds same-token layer 0 and l−2 to previous-layer plus earlier-group context. It does not implement the team’s attention predictor, pivot order, repeat-token context or value-slice coder; 51.4% is not an achieved GPU result.
AQUA-style is a predictor ablation inside our exact coder. aquakv_quanto2/4 instead run released AQUA cache/Quanto method bodies with local HF compatibility and packet framing; they are lossy page adapters, not paper serving reproduction.
exans_native is pinned OpenLake version 2 phase-bitmap plus nvCOMP ANS. Physical input padding, native slot reservation and charged raw overflow blocks are recorded per request; complete packet bytes include all transmitted parts. Native allocator peaks are unmeasured. FP8 speculative-model coding remains pending.
This benchmark measures GPU transfer codecs. It does not measure P1 attention, T-to-P conversion, cold SSD, network or serving throughput.
