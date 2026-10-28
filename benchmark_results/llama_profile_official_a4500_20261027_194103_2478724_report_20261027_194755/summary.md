# Lite profiling and official baseline follow-up

Status: **partial**. Actual GPU checked: **True**.

Existing fits and measured results are preserved. Profile diagnostics are separate from unprofiled timing.

| Stage | Status | Detail |
|---|---|---|
| profile | completed |  |
| toolchain | completed |  |
| exans_build | blocked | exit 127; see /home/nelson/kv_reconstruction/runs/llama_profile_official_a4500_20261027_194103_2478724/logs/exans_author_smoke.log |
| aqua_setup | completed |  |
| aqua_raw | completed |  |
| aqua_train | blocked | exit 1; see /home/nelson/kv_reconstruction/runs/llama_profile_official_a4500_20261027_194103_2478724/logs/aqua_identify_train.log |

Same frozen TEST inputs; existing measurements reused, ExANS added. Means of per-request warm wall medians; complete framed bytes.

| Method / step | Complete bpv | Saved | Encode ms | Decode ms | Exact |
|---|---:|---:|---:|---:|---|
| raw_bf16 | 16.00009 | -0.00% | 0.423 | 0.339 | True |
| fp8_e4m3fn_saturating | 8.00009 | 50.00% | 1.238 | 0.410 | False |
| splitzip_official_framed | 12.04281 | 24.73% | 0.874 | 0.521 | True |
| block32 | 8.98178 | 43.86% | 1270.738 | 1628.033 | True |
| block32 / 0.5 | 4.06533 | 74.59% | 1779.827 | 1612.641 | False |
| block32 / 1 | 3.13714 | 80.39% | 1778.860 | 1602.200 | False |
| lite | 8.87967 | 44.50% | 330.751 | 418.618 | True |
| lite / 0.5 | 3.99232 | 75.05% | 463.402 | 425.083 | False |
| lite / 1 | 3.10313 | 80.61% | 463.947 | 422.116 | False |

| Lite step | Direction | Ordinary ms | Profiled ms | CUDA events |
|---:|---|---:|---:|---:|
| 0 | encode | 328.732 | 570.485 | 40905 |
| 0 | decode | 416.329 | 674.450 | 48053 |
| 1 | encode | 559.471 | 854.470 | 76295 |
| 1 | decode | 502.426 | 683.464 | 61041 |

Top self-device operators, Lite 0 encode:

- kv/request: 569.581 ms across 1 calls
- kv/entropy_encode: 97.514 ms across 448 calls
- kv/rope_inverse: 77.638 ms across 448 calls
- kv/rope_forward: 77.419 ms across 448 calls
- _encode_kernel: 74.887 ms across 448 calls

Top self-device operators, Lite 0 decode:

- kv/request: 673.311 ms across 1 calls
- kv/entropy_decode: 156.292 ms across 448 calls
- _decode_kernel: 134.693 ms across 448 calls
- kv/rope_forward: 76.917 ms across 448 calls
- kv/rope_inverse: 76.087 ms across 448 calls

Top self-device operators, Lite 1 encode:

- kv/request: 855.023 ms across 1 calls
- kv/rope_forward: 78.545 ms across 448 calls
- kv/rope_inverse: 76.289 ms across 448 calls
- _res_encode: 67.232 ms across 448 calls
- kv/entropy_encode: 67.232 ms across 448 calls

Top self-device operators, Lite 1 decode:

- kv/request: 681.449 ms across 1 calls
- kv/entropy_decode: 123.605 ms across 448 calls
- _res_decode: 103.027 ms across 448 calls
- kv/rope_forward: 75.279 ms across 448 calls
- kv/rope_inverse: 72.827 ms across 448 calls

ExANS: unchanged pinned author kernels and native C ABI; documented compiler-injected allocation-error overload. Physical fixed slots, input padding and local raw overflow fallback are charged. Native allocator peaks are unmeasured.
AQUA: unchanged author train/evaluation, official calibration/context/buffers. Its command wall time includes model inference and is not compared numerically with transfer codec latency. Physical AQUA wire bytes are unmeasured.
Profiler scopes are nested/inclusive, not additive; GPU operator self-times and CUDA events are diagnostics. Ordinary wall samples are the performance control.
No CPU-estimated GPU performance, pool attention, network/SSD, T-to-P or serving speedup is claimed.
