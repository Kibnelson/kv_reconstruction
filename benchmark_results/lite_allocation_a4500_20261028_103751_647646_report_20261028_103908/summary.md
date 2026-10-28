# Controlled uninitialized-allocation fill policy

Status: **completed**. GPU measured: **True**.

The same FinishEngine and V64/B32 packets run under fill-on and fill-off graphs. Deterministic algorithms stay ON in both Lite arms; only empty-allocation initialization changes.
Both engines and both graph buffers are resident during paired timing. Full decode includes validation, staging copies, replay and an independently owned output clone.

| Method | Saved | Complete bpv | Decode ms | Exact |
|---|---:|---:|---:|---|
| raw_bf16 | -0.00% | 16.00009 | 0.332 | True |
| splitzip_official_framed | 24.73% | 12.04281 | 0.532 | True |
| exans_official_fixed_framed | 22.63% | 12.37970 | 3.691 | True |
| lite_allocation_fill_on_graph | 43.19% | 9.09012 | 88.162 | True |
| lite_allocation_fill_off_graph | 43.19% | 9.09012 | 83.243 | True |

Effective measured runtime flags. Native baselines keep their ORIGINAL device setup; allocation_policy is never applied to them. If deterministic mode is OFF, its empty-allocation fill is inactive regardless of the fill flag.

| Method / arm | Deterministic algorithms | Fill flag | Effective uninitialized fill | Runtime origin |
|---|---|---|---|---|
| raw_bf16 | False | True | False | original native runtime |
| splitzip_official_framed | False | True | False | original native runtime |
| exans_official_fixed_framed | False | True | False | original native runtime |
| fill_on_graph | True | True | True | controlled Lite allocation context |
| fill_off_graph | True | False | False | controlled Lite allocation context |

Interface metadata only; current generation can be lower at idle. This is not a transfer-bandwidth measurement.

`name,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max`

```text
NVIDIA RTX A4500, 580.126.20, 4, 4, 16, 16
```

Cold graph setup and cumulative paired-setup memory; Torch accounting excludes some driver metadata.

| Arm | Cold ms | Allocated before → after MiB | Reserved before → after MiB | Peak allocated / reserved MiB | Staging payload MiB | Directory KiB | Output MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fill_on_graph | 637.530 | 746.380 → 948.491 | 1254.000 → 1104.000 | 1035.625 / 1442.000 | 113.877 | 112.125 | 56.000 |
| fill_off_graph | 561.089 | 948.491 → 1150.601 | 1104.000 → 1350.000 | 1237.736 / 1406.000 | 113.877 | 112.125 | 56.000 |

Separate synchronized phase diagnostics; raw samples remain JSON and are not ordinary paired latencies.

| Arm / document | Validation wall ms | Input-copy CUDA ms | Replay CUDA ms | Replay CPU submit ms | Output-clone CUDA ms |
|---|---:|---:|---:|---:|---:|
| fill_on_graph / test_code | 0.556 | 0.150 | 87.154 | 0.050 | 0.328 |
| fill_on_graph / test_text | 0.516 | 0.145 | 85.798 | 0.078 | 0.324 |
| fill_off_graph / test_code | 0.534 | 0.156 | 81.082 | 0.090 | 0.223 |
| fill_off_graph / test_text | 0.507 | 0.143 | 78.024 | 0.065 | 0.221 |

Separate instrumented eager profiles; these identify kernel work, not captured-graph latency. Fill totals include required explicit zeros/ones.

| Arm | Kernel calls | Sum kernel ms | Fill-kernel calls | Fill-kernel ms | Entropy calls | Entropy ms |
|---|---:|---:|---:|---:|---:|---:|
| fill_on_graph | 14397 | 90.917 | 3727 | 5.926 | 448 | 49.472 |
| fill_off_graph | 10727 | 85.285 | 57 | 0.281 | 448 | 49.467 |

Top device kernels for fill_on_graph:

- `_decode_kernel`: 49.472 ms, 448 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 4.322 ms, 2829 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.905 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`: 3.047 ms, 1764 calls.
- `_finish_kernel`: 3.001 ms, 448 calls.

Top device kernels for fill_off_graph:

- `_decode_kernel`: 49.467 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.833 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`: 3.099 ms, 1764 calls.
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1})`: 3.026 ms, 420 calls.
- `_finish_kernel`: 2.996 ms, 448 calls.

Calculated serial precompressed fetch, not measured network/serving latency. No fresh encoder projection.

| Gb/s | Method | Transfer + decode ms |
|---:|---|---:|
| 1 | raw_bf16 | 470.096 |
| 1 | splitzip_official_framed | 354.110 |
| 1 | exans_official_fixed_framed | 367.161 |
| 1 | lite_allocation_fill_on_graph | 355.049 |
| 1 | lite_allocation_fill_off_graph | 350.131 |
| 2.5 | raw_bf16 | 188.238 |
| 2.5 | splitzip_official_framed | 141.963 |
| 2.5 | exans_official_fixed_framed | 149.079 |
| 2.5 | lite_allocation_fill_on_graph | 194.917 |
| 2.5 | lite_allocation_fill_off_graph | 189.998 |
| 5 | raw_bf16 | 94.285 |
| 5 | splitzip_official_framed | 71.247 |
| 5 | exans_official_fixed_framed | 76.385 |
| 5 | lite_allocation_fill_on_graph | 141.540 |
| 5 | lite_allocation_fill_off_graph | 136.621 |
| 10 | raw_bf16 | 47.308 |
| 10 | splitzip_official_framed | 35.890 |
| 10 | exans_official_fixed_framed | 40.038 |
| 10 | lite_allocation_fill_on_graph | 114.851 |
| 10 | lite_allocation_fill_off_graph | 109.932 |
| 25 | raw_bf16 | 19.122 |
| 25 | splitzip_official_framed | 14.675 |
| 25 | exans_official_fixed_framed | 18.230 |
| 25 | lite_allocation_fill_on_graph | 98.838 |
| 25 | lite_allocation_fill_off_graph | 93.919 |
| 50 | raw_bf16 | 9.727 |
| 50 | splitzip_official_framed | 7.603 |
| 50 | exans_official_fixed_framed | 10.961 |
| 50 | lite_allocation_fill_on_graph | 93.500 |
| 50 | lite_allocation_fill_off_graph | 88.581 |
| 100 | raw_bf16 | 5.029 |
| 100 | splitzip_official_framed | 4.067 |
| 100 | exans_official_fixed_framed | 7.326 |
| 100 | lite_allocation_fill_on_graph | 90.831 |
| 100 | lite_allocation_fill_off_graph | 85.912 |
| 200 | raw_bf16 | 2.680 |
| 200 | splitzip_official_framed | 2.300 |
| 200 | exans_official_fixed_framed | 5.509 |
| 200 | lite_allocation_fill_on_graph | 89.497 |
| 200 | lite_allocation_fill_off_graph | 84.578 |
| 400 | raw_bf16 | 1.506 |
| 400 | splitzip_official_framed | 1.416 |
| 400 | exans_official_fixed_framed | 4.600 |
| 400 | lite_allocation_fill_on_graph | 88.829 |
| 400 | lite_allocation_fill_off_graph | 83.911 |

Required decoder budgets against raw BF16 and official SplitZip, using complete packet sizes. The 400 Gb/s row is a compute-gap target, not an achieved link rate.

| Candidate | Baseline | Gb/s | Decode budget ms | Current decode ms | Further reduction |
|---|---|---:|---:|---:|---:|
| lite_allocation_fill_on_graph | raw_bf16 | 1 | 203.209 | 88.162 | 0.43x |
| lite_allocation_fill_on_graph | raw_bf16 | 2.5 | 81.483 | 88.162 | 1.08x |
| lite_allocation_fill_on_graph | raw_bf16 | 5 | 40.907 | 88.162 | 2.16x |
| lite_allocation_fill_on_graph | raw_bf16 | 10 | 20.619 | 88.162 | 4.28x |
| lite_allocation_fill_on_graph | raw_bf16 | 25 | 8.447 | 88.162 | 10.44x |
| lite_allocation_fill_on_graph | raw_bf16 | 50 | 4.389 | 88.162 | 20.09x |
| lite_allocation_fill_on_graph | raw_bf16 | 100 | 2.360 | 88.162 | 37.35x |
| lite_allocation_fill_on_graph | raw_bf16 | 200 | 1.346 | 88.162 | 65.50x |
| lite_allocation_fill_on_graph | raw_bf16 | 400 | 0.839 | 88.162 | 105.10x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 1 | 87.223 | 88.162 | 1.01x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 2.5 | 35.208 | 88.162 | 2.50x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 5 | 17.870 | 88.162 | 4.93x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 10 | 9.201 | 88.162 | 9.58x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 25 | 3.999 | 88.162 | 22.04x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 50 | 2.266 | 88.162 | 38.92x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 100 | 1.399 | 88.162 | 63.04x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 200 | 0.965 | 88.162 | 91.35x |
| lite_allocation_fill_on_graph | splitzip_official_framed | 400 | 0.748 | 88.162 | 117.80x |
| lite_allocation_fill_off_graph | raw_bf16 | 1 | 203.209 | 83.243 | 0.41x |
| lite_allocation_fill_off_graph | raw_bf16 | 2.5 | 81.483 | 83.243 | 1.02x |
| lite_allocation_fill_off_graph | raw_bf16 | 5 | 40.907 | 83.243 | 2.03x |
| lite_allocation_fill_off_graph | raw_bf16 | 10 | 20.619 | 83.243 | 4.04x |
| lite_allocation_fill_off_graph | raw_bf16 | 25 | 8.447 | 83.243 | 9.86x |
| lite_allocation_fill_off_graph | raw_bf16 | 50 | 4.389 | 83.243 | 18.97x |
| lite_allocation_fill_off_graph | raw_bf16 | 100 | 2.360 | 83.243 | 35.27x |
| lite_allocation_fill_off_graph | raw_bf16 | 200 | 1.346 | 83.243 | 61.84x |
| lite_allocation_fill_off_graph | raw_bf16 | 400 | 0.839 | 83.243 | 99.23x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 1 | 87.223 | 83.243 | 0.95x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 2.5 | 35.208 | 83.243 | 2.36x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 5 | 17.870 | 83.243 | 4.66x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 10 | 9.201 | 83.243 | 9.05x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 25 | 3.999 | 83.243 | 20.81x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 50 | 2.266 | 83.243 | 36.74x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 100 | 1.399 | 83.243 | 59.52x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 200 | 0.965 | 83.243 | 86.25x |
| lite_allocation_fill_off_graph | splitzip_official_framed | 400 | 0.748 | 83.243 | 111.23x |

Both allocation policies must reproduce the complete original packet bytes outside timing. Fresh GPU receivers decode fixed state plus unchanged packets and verify policy, new data and output ownership.
The allocation flag is process-global; these are exclusive benchmark-worker contexts with restoration, not a concurrent serving API. Required explicit zeros/ones, deterministic algorithms and codec mathematics are unchanged.
Diagnostics/profiles run in separate processes after all ordinary measurements. Full traces remain local. Packet upload, cold capture, shared state and graph memory are additional.
Pinned official baseline methods retain their original worker/runtime policies. No fitting, model download, native build, actual network, SSD, pool attention or serving-concurrency measurement is performed.
