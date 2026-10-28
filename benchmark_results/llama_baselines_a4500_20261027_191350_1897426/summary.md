# Matched causal run with baseline controls

Status: **completed**. New GPU measured: **True**.

Completion covers the raw/FP8/SplitZip controls; official AQUA-KV and ExANS evaluations remain incomplete.
Pinned Llama-3.2-3B, all 28 BF16 layers, identical frozen TEST words. Block32/Lite measurements are reused unchanged.
Complete framed bytes; warm synchronized per-request wall medians. Shared state is additional.

| Document | Method / step | Mode | Complete bpv | Saved | Encode ms | Decode ms | Exact | PPL change | Shared GPU MiB |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| test_code | block32 | lossless | 8.96788 | 43.95% | 1262.205 | 1626.508 | True | +0.000% | 329.367 |
| test_code | block32 / 0.5 | lossy | 4.04082 | 74.74% | 1777.615 | 1614.387 | False | -1.456% | 329.367 |
| test_code | block32 / 1 | lossy | 3.11306 | 80.54% | 1778.122 | 1601.771 | False | +2.590% | 329.367 |
| test_code | fp8_e4m3fn_saturating | lossy | 8.00009 | 50.00% | 1.234 | 0.410 | False | +0.395% | 0.000 |
| test_code | lite | lossless | 8.89163 | 44.43% | 331.488 | 419.408 | True | +0.000% | 324.110 |
| test_code | lite / 0.5 | lossy | 3.99045 | 75.06% | 462.488 | 423.896 | False | +0.665% | 324.110 |
| test_code | lite / 1 | lossy | 3.09592 | 80.65% | 463.890 | 421.461 | False | -0.056% | 324.110 |
| test_code | raw_bf16 | lossless | 16.00009 | -0.00% | 0.432 | 0.345 | True | +0.000% | 0.000 |
| test_code | splitzip_official_framed | lossless | 12.04244 | 24.73% | 0.884 | 0.531 | True | +0.000% | 0.001 |
| test_text | block32 | lossless | 8.99569 | 43.78% | 1279.271 | 1629.557 | True | +0.000% | 329.367 |
| test_text | block32 / 0.5 | lossy | 4.08985 | 74.44% | 1782.039 | 1610.896 | False | +2.723% | 329.367 |
| test_text | block32 / 1 | lossy | 3.16122 | 80.24% | 1779.598 | 1602.629 | False | +6.691% | 329.367 |
| test_text | fp8_e4m3fn_saturating | lossy | 8.00009 | 50.00% | 1.243 | 0.411 | False | +0.678% | 0.000 |
| test_text | lite | lossless | 8.86772 | 44.58% | 330.014 | 417.829 | True | +0.000% | 324.110 |
| test_text | lite / 0.5 | lossy | 3.99420 | 75.04% | 464.315 | 426.269 | False | +3.061% | 324.110 |
| test_text | lite / 1 | lossy | 3.11033 | 80.56% | 464.005 | 422.771 | False | +1.784% | 324.110 |
| test_text | raw_bf16 | lossless | 16.00009 | -0.00% | 0.414 | 0.333 | True | +0.000% | 0.000 |
| test_text | splitzip_official_framed | lossless | 12.04319 | 24.73% | 0.864 | 0.512 | True | +0.000% | 0.001 |

Official unchanged SplitZip author benchmark, saved-tensor mode (separate native timing/byte scope):

| Document | Native bpv | Encode GB/s | Decode GB/s |
|---|---:|---:|---:|
| test_code | 12.04225 | 130.150 | 309.424 |
| test_text | 12.04300 | 130.248 | 309.341 |

| Baseline | Status | Evidence / next requirement |
|---|---|---|
| AQUA-KV official | pending official evaluation | Prior official training stopped before a successful HIGGS/PPL result; adapted Quanto rows are excluded. |
| ExANS official | prior build blocked | Pinned official C++ build previously failed in expans_codec.cu; no matched measurement. See the earlier official report. |
| raw_bf16 | measured | identical TEST words, fresh-process packet receiver verified |
| fp8_e4m3fn_saturating | measured | identical TEST words, fresh-process packet receiver verified |
| splitzip_official_framed | measured | identical TEST words, fresh-process packet receiver verified |

SplitZip's algorithm and official throughput script are unmodified. The framed row adds local transport glue and validation; it is not the paper's full evaluation.
SplitZip runs in the author's pinned environment; package versions and hashes are recorded. Its calibration uses the author's default 4,194,304-value TRAIN prefix cap, not TEST.
FP8 is a native unscaled E4M3FN cast with saturation at ±448, not a named paper baseline. Model weights stay BF16.
New packets are replayed by a separate receiver with wire and fixed LUT state only. Export/upload, codec setup and calibration/state export are recorded separately from warm codec timings.
Quality uses the same short continuation and exact reference as the base run, not full WikiText PPL or LongBench.
Historical AQUA adaptations and ExANS results on other inputs are not substituted. No compressed attention, T-to-P, SSD/network or serving throughput is measured.

Historical official evaluation: https://github.com/Kibnelson/kv_reconstruction/blob/cd5df510d42739d11470d50d99f6c1ccf4caec14/benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517/summary.md
