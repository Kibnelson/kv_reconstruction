# Complete Lite packets versus retimed baselines

Status: **completed**. GPU measured: **True**.

All methods use identical original BF16 words, all 28 layers. Complete packet bytes include framing and directories.
Each method runs in a fresh process, with independent packet-only replay. Baseline implementations and fitted state are unchanged.
Means below average per-document synchronized wall medians. Shared fixed state is additional.

| Method | Kappa | Saved | Encode ms | Decode ms | Exact original |
|---|---:|---:|---:|---:|---|
| raw_bf16 | 0 | -0.00% | 0.424 | 0.351 | True |
| splitzip_official_framed | 0 | 24.73% | 0.869 | 0.523 | True |
| fp8_e4m3fn_saturating | 0 | 50.00% | 1.239 | 0.411 | False |
| exans_official_fixed_framed | 0 | 22.63% | 15.148 | 3.658 | True |
| lite_b64_before | 0 | 44.50% | 223.181 | 305.217 | True |
| lite_b64_before | 0.5 | 75.05% | 365.635 | 305.586 | False |
| lite_b64_before | 1 | 80.61% | 365.059 | 303.081 | False |
| lite_b8 | 0 | 44.17% | 227.939 | 254.770 | True |
| lite_b8 | 0.5 | 74.71% | 387.615 | 311.627 | False |
| lite_b8 | 1 | 80.27% | 387.978 | 312.449 | False |
| lite_b16 | 0 | 44.36% | 227.671 | 268.999 | True |
| lite_b16 | 0.5 | 74.91% | 389.764 | 314.963 | False |
| lite_b16 | 1 | 80.46% | 387.924 | 313.656 | False |
| lite_b32 | 0 | 44.46% | 216.844 | 266.488 | True |
| lite_b32 | 0.5 | 75.00% | 370.180 | 297.227 | False |
| lite_b32 | 1 | 80.56% | 370.542 | 296.823 | False |
| lite_b64_after | 0 | 44.50% | 224.006 | 305.378 | True |
| lite_b64_after | 0.5 | 75.05% | 375.187 | 307.417 | False |
| lite_b64_after | 1 | 80.61% | 373.264 | 305.754 | False |

Previously measured short continuation quality, reused only after identical reconstructed words.

| Lossy method | Kappa | Document | PPL change |
|---|---:|---|---:|
| fp8_e4m3fn_saturating | 0 | test_code | +0.395% |
| fp8_e4m3fn_saturating | 0 | test_text | +0.678% |
| lite_b64_before | 0.5 | test_code | +0.665% |
| lite_b64_before | 0.5 | test_text | +3.061% |
| lite_b64_before | 1 | test_code | -0.056% |
| lite_b64_before | 1 | test_text | +1.784% |
| lite_b8 | 0.5 | test_code | +0.665% |
| lite_b8 | 0.5 | test_text | +3.061% |
| lite_b8 | 1 | test_code | -0.056% |
| lite_b8 | 1 | test_text | +1.784% |
| lite_b16 | 0.5 | test_code | +0.665% |
| lite_b16 | 0.5 | test_text | +3.061% |
| lite_b16 | 1 | test_code | -0.056% |
| lite_b16 | 1 | test_text | +1.784% |
| lite_b32 | 0.5 | test_code | +0.665% |
| lite_b32 | 0.5 | test_text | +3.061% |
| lite_b32 | 1 | test_code | -0.056% |
| lite_b32 | 1 | test_text | +1.784% |
| lite_b64_after | 0.5 | test_code | +0.665% |
| lite_b64_after | 0.5 | test_text | +3.061% |
| lite_b64_after | 1 | test_code | -0.056% |
| lite_b64_after | 1 | test_text | +1.784% |

Time ratios >1 mean slower than the baseline; packet ratios <1 mean fewer bytes.

| Lossless method | Versus | Packet ratio | Encode time ratio | Decode time ratio |
|---|---|---:|---:|---:|
| lite_b64_before | raw_bf16 | 0.555 | 526.27 | 870.75 |
| lite_b64_before | splitzip_official_framed | 0.737 | 256.68 | 583.82 |
| lite_b64_before | exans_official_fixed_framed | 0.717 | 14.73 | 83.44 |
| lite_b8 | raw_bf16 | 0.558 | 537.49 | 726.83 |
| lite_b8 | splitzip_official_framed | 0.742 | 262.15 | 487.32 |
| lite_b8 | exans_official_fixed_framed | 0.722 | 15.05 | 69.65 |
| lite_b16 | raw_bf16 | 0.556 | 536.86 | 767.42 |
| lite_b16 | splitzip_official_framed | 0.739 | 261.84 | 514.54 |
| lite_b16 | exans_official_fixed_framed | 0.719 | 15.03 | 73.54 |
| lite_b32 | raw_bf16 | 0.555 | 511.33 | 760.26 |
| lite_b32 | splitzip_official_framed | 0.738 | 249.39 | 509.74 |
| lite_b32 | exans_official_fixed_framed | 0.718 | 14.32 | 72.85 |
| lite_b64_after | raw_bf16 | 0.555 | 528.22 | 871.21 |
| lite_b64_after | splitzip_official_framed | 0.737 | 257.63 | 584.12 |
| lite_b64_after | exans_official_fixed_framed | 0.717 | 14.79 | 83.49 |

Calculated serial network model — not measured transfer or serving latency.

| Effective Gb/s | Best fresh encode + transfer + decode | Best precompressed transfer + decode |
|---:|---|---|
| 0.1 | lite_b32 | lite_b32 |
| 0.25 | splitzip_official_framed | lite_b8 |
| 0.5 | splitzip_official_framed | splitzip_official_framed |
| 1 | splitzip_official_framed | splitzip_official_framed |
| 2.5 | splitzip_official_framed | splitzip_official_framed |
| 5 | splitzip_official_framed | splitzip_official_framed |
| 10 | splitzip_official_framed | splitzip_official_framed |
| 25 | splitzip_official_framed | splitzip_official_framed |
| 50 | splitzip_official_framed | splitzip_official_framed |
| 100 | splitzip_official_framed | splitzip_official_framed |

B64 controls must match original packets byte for byte. B8/B16/B32 change stream grouping and charge their actual bytes; all decoded words must match the original reconstruction.
B64 is measured before and after the sweep to expose drift. Timing workers never run the profiler.
Lossy quality is reused only after matching the prior reconstructed-word hash; it is a short continuation smoke check, not full task validation.
Official AQUA remains blocked; no AQUA-style/page adapter substitutes for it. ExANS native allocation peaks are unmeasured.
No SSD, actual network, overlap, pool attention or serving result is measured by this benchmark.
