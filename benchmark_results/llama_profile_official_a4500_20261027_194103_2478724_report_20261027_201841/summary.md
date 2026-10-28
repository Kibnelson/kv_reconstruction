# Lite profiling and official baseline follow-up

Status: **partial**. Actual GPU checked: **True**.

Existing fits and measured results are preserved. Profile diagnostics are separate from unprofiled timing.

| Stage | Status | Detail |
|---|---|---|
| profile | completed |  |
| toolchain | completed |  |
| exans_build | completed |  |
| aqua_setup | completed |  |
| aqua_raw | completed |  |
| aqua_train | blocked | exit 1; see /home/nelson/kv_reconstruction/runs/llama_profile_official_a4500_20261027_194103_2478724/logs/aqua_prepare_backup.log |
| exans | blocked | 'str' object has no attribute 'open' |

AQUA calibration explicitly uses a pinned third-party RedPajama backup. The author code and sampling settings are preserved; equivalence to the removed original dataset is unverified.

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

- _encode_kernel: 74.887 ms across 448 calls
- aten::_index_put_impl_: 30.351 ms across 476 calls
- void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool): 25.682 ms across 476 calls
- aten::index: 20.418 ms across 4032 calls
- aten::fill_: 15.874 ms across 12440 calls

Top self-device operators, Lite 0 decode:

- _decode_kernel: 134.693 ms across 448 calls
- aten::_index_put_impl_: 48.424 ms across 924 calls
- void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool): 25.640 ms across 476 calls
- aten::fill_: 18.506 ms across 15573 calls
- aten::index: 17.533 ms across 3584 calls

Top self-device operators, Lite 1 encode:

- _res_encode: 67.232 ms across 448 calls
- aten::_index_put_impl_: 48.666 ms across 924 calls
- aten::fill_: 32.829 ms across 28569 calls
- void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool): 25.816 ms across 476 calls
- aten::copy_: 25.228 ms across 8063 calls

Top self-device operators, Lite 1 decode:

- _res_decode: 103.027 ms across 448 calls
- aten::_index_put_impl_: 48.344 ms across 924 calls
- void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool): 25.617 ms across 476 calls
- aten::fill_: 23.606 ms across 20053 calls
- aten::copy_: 19.946 ms across 6271 calls

ExANS: unchanged pinned author kernels and native C ABI; documented compiler-injected allocation-error overload. Physical fixed slots, input padding and local raw overflow fallback are charged. Native allocator peaks are unmeasured.
AQUA: unchanged author train/evaluation; calibration source is recorded explicitly and context/buffers are preserved. Its command wall time includes model inference and is not compared numerically with transfer codec latency. Physical AQUA wire bytes are unmeasured.
Profiler scopes are nested/inclusive, not additive; GPU operator self-times and CUDA events are diagnostics. Ordinary wall samples are the performance control.
No CPU-estimated GPU performance, pool attention, network/SSD, T-to-P or serving speedup is claimed.
