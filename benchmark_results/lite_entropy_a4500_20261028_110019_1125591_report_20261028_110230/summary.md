# Exact entropy lookup and initial-refinement ablation

Status: **completed**. GPU measured: **True**.

V64/B32 packet bytes, predictors and probabilities are unchanged. Four graph arms compare the original coarse stage-1 lookup against a full exact lookup with 2, 1 or 0 initial word refinements.
All four fitted engines, decoder tables and graph buffers are resident during balanced rotation/reversal timing. All Lite arms keep deterministic algorithms ON and uninitialized-allocation fill OFF. Full decode includes validation, staging copies, replay and an independently owned output clone.

| Method | Saved | Complete bpv | Decode ms | Exact |
|---|---:|---:|---:|---|
| raw_bf16 | -0.00% | 16.00009 | 0.335 | True |
| splitzip_official_framed | 24.73% | 12.04281 | 0.517 | True |
| exans_official_fixed_framed | 22.63% | 12.37970 | 3.665 | True |
| lite_entropy_reference_graph | 43.19% | 9.09012 | 83.316 | True |
| lite_entropy_full_r2_graph | 43.19% | 9.09012 | 84.860 | True |
| lite_entropy_full_r1_graph | 43.19% | 9.09012 | 84.973 | True |
| lite_entropy_full_r0_graph | 43.19% | 9.09012 | 94.439 | True |

Decoder table accounting is additional to packet bytes. Predictor memory is separate from coder tables. Full-LUT construction includes host generation, upload and CUDA synchronization; total engine initialization is recorded separately.

| Arm | Lookup | Initial refine | Predictor MiB | Retained coder KiB | Full LUT MiB | Total coder MiB | Added shared MiB | LUT construction ms | Full LUT SHA256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| reference_graph | coarse | 2 | 324.110 | 43.195 | 0.000 | 0.042 | 0.000 | 0.000 | none |
| full_r2_graph | full | 2 | 324.110 | 43.195 | 1.250 | 1.292 | 1.250 | 3.072 | 57fe89079d5c16db0db1435a9f11ee0176de20e0b6a5983c901c3a56548f2862 |
| full_r1_graph | full | 1 | 324.110 | 43.195 | 1.250 | 1.292 | 1.250 | 2.903 | 57fe89079d5c16db0db1435a9f11ee0176de20e0b6a5983c901c3a56548f2862 |
| full_r0_graph | full | 0 | 324.110 | 43.195 | 1.250 | 1.292 | 1.250 | 3.018 | 57fe89079d5c16db0db1435a9f11ee0176de20e0b6a5983c901c3a56548f2862 |

Effective measured runtime flags. Native baselines keep their ORIGINAL device setup; allocation_policy is never applied to them. If deterministic mode is OFF, its empty-allocation fill is inactive regardless of the fill flag.

| Method / arm | Deterministic algorithms | Fill flag | Effective uninitialized fill | Runtime origin |
|---|---|---|---|---|
| raw_bf16 | False | True | False | original native runtime |
| splitzip_official_framed | False | True | False | original native runtime |
| exans_official_fixed_framed | False | True | False | original native runtime |
| reference_graph | True | False | False | controlled Lite allocation context |
| full_r2_graph | True | False | False | controlled Lite allocation context |
| full_r1_graph | True | False | False | controlled Lite allocation context |
| full_r0_graph | True | False | False | controlled Lite allocation context |

Interface metadata only; current generation can be lower at idle. This is not a transfer-bandwidth measurement.

`name,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max`

```text
NVIDIA RTX A4500, 580.126.20, 4, 4, 16, 16
```

Cold graph setup and cumulative paired-setup memory; Torch accounting excludes some driver metadata.

| Arm | Cold ms | Allocated before → after MiB | Reserved before → after MiB | Peak allocated / reserved MiB | Staging payload MiB | Directory KiB | Output MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_graph | 590.552 | 1400.540 → 1602.651 | 2012.000 → 1864.000 | 1689.785 / 2200.000 | 113.877 | 112.125 | 56.000 |
| full_r2_graph | 1686.780 | 1602.651 → 1804.761 | 1864.000 → 2110.000 | 1891.896 / 2166.000 | 113.877 | 112.125 | 56.000 |
| full_r1_graph | 1627.597 | 1804.761 → 2006.871 | 2110.000 → 2356.000 | 2094.006 / 2412.000 | 113.877 | 112.125 | 56.000 |
| full_r0_graph | 1657.709 | 2006.871 → 2208.980 | 2356.000 → 2602.000 | 2296.116 / 2658.000 | 113.877 | 112.125 | 56.000 |

Separate synchronized phase diagnostics; raw samples remain JSON and are not ordinary paired latencies.

| Arm / document | Validation wall ms | Input-copy CUDA ms | Replay CUDA ms | Replay CPU submit ms | Output-clone CUDA ms |
|---|---:|---:|---:|---:|---:|
| reference_graph / test_code | 0.526 | 0.151 | 81.544 | 0.057 | 0.222 |
| reference_graph / test_text | 0.533 | 0.153 | 78.501 | 0.069 | 0.233 |
| full_r2_graph / test_code | 0.572 | 0.154 | 83.778 | 0.039 | 0.243 |
| full_r2_graph / test_text | 0.511 | 0.149 | 83.317 | 0.059 | 0.225 |
| full_r1_graph / test_code | 0.528 | 0.151 | 83.970 | 0.050 | 0.223 |
| full_r1_graph / test_text | 0.491 | 0.143 | 83.662 | 0.062 | 0.223 |
| full_r0_graph / test_code | 0.528 | 0.154 | 92.924 | 0.033 | 0.223 |
| full_r0_graph / test_text | 0.514 | 0.146 | 88.929 | 0.066 | 0.220 |

Separate instrumented eager profiles; these identify kernel work, not captured-graph latency. Fill totals include required explicit zeros/ones.

| Arm | Kernel calls | Sum kernel ms | Fill-kernel calls | Fill-kernel ms | Entropy calls | Entropy ms |
|---|---:|---:|---:|---:|---:|---:|
| reference_graph | 10727 | 85.346 | 57 | 0.281 | 448 | 49.532 |
| full_r2_graph | 10727 | 86.746 | 57 | 0.280 | 448 | 50.949 |
| full_r1_graph | 10727 | 86.965 | 57 | 0.281 | 448 | 51.181 |
| full_r0_graph | 10727 | 96.241 | 57 | 0.279 | 448 | 60.472 |

Top device kernels for reference_graph:

- `_decode_kernel`: 49.532 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.838 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`: 3.095 ms, 1764 calls.
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1})`: 3.025 ms, 420 calls.
- `_finish_kernel`: 2.999 ms, 448 calls.

Top device kernels for full_r2_graph:

- `_decode_full_kernel`: 50.949 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.816 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`: 3.101 ms, 1764 calls.
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1})`: 3.039 ms, 420 calls.
- `_finish_kernel`: 2.996 ms, 448 calls.

Top device kernels for full_r1_graph:

- `_decode_full_kernel`: 51.181 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.804 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`: 3.102 ms, 1764 calls.
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1})`: 3.040 ms, 420 calls.
- `_finish_kernel`: 3.002 ms, 448 calls.

Top device kernels for full_r0_graph:

- `_decode_full_kernel`: 60.472 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.792 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`: 3.103 ms, 1764 calls.
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} const&)::{lambda(int)#1})`: 3.033 ms, 420 calls.
- `_finish_kernel`: 3.002 ms, 448 calls.

Calculated serial precompressed fetch, not measured network/serving latency. No fresh encoder projection.

| Gb/s | Method | Transfer + decode ms |
|---:|---|---:|
| 1 | raw_bf16 | 470.100 |
| 1 | splitzip_official_framed | 354.095 |
| 1 | exans_official_fixed_framed | 367.134 |
| 1 | lite_entropy_reference_graph | 350.203 |
| 1 | lite_entropy_full_r2_graph | 351.747 |
| 1 | lite_entropy_full_r1_graph | 351.860 |
| 1 | lite_entropy_full_r0_graph | 361.326 |
| 2.5 | raw_bf16 | 188.241 |
| 2.5 | splitzip_official_framed | 141.948 |
| 2.5 | exans_official_fixed_framed | 149.053 |
| 2.5 | lite_entropy_reference_graph | 190.071 |
| 2.5 | lite_entropy_full_r2_graph | 191.615 |
| 2.5 | lite_entropy_full_r1_graph | 191.727 |
| 2.5 | lite_entropy_full_r0_graph | 201.194 |
| 5 | raw_bf16 | 94.288 |
| 5 | splitzip_official_framed | 71.233 |
| 5 | exans_official_fixed_framed | 76.359 |
| 5 | lite_entropy_reference_graph | 136.694 |
| 5 | lite_entropy_full_r2_graph | 138.237 |
| 5 | lite_entropy_full_r1_graph | 138.350 |
| 5 | lite_entropy_full_r0_graph | 147.817 |
| 10 | raw_bf16 | 47.312 |
| 10 | splitzip_official_framed | 35.875 |
| 10 | exans_official_fixed_framed | 40.012 |
| 10 | lite_entropy_reference_graph | 110.005 |
| 10 | lite_entropy_full_r2_graph | 111.548 |
| 10 | lite_entropy_full_r1_graph | 111.661 |
| 10 | lite_entropy_full_r0_graph | 121.128 |
| 25 | raw_bf16 | 19.126 |
| 25 | splitzip_official_framed | 14.660 |
| 25 | exans_official_fixed_framed | 18.204 |
| 25 | lite_entropy_reference_graph | 93.992 |
| 25 | lite_entropy_full_r2_graph | 95.535 |
| 25 | lite_entropy_full_r1_graph | 95.648 |
| 25 | lite_entropy_full_r0_graph | 105.115 |
| 50 | raw_bf16 | 9.731 |
| 50 | splitzip_official_framed | 7.588 |
| 50 | exans_official_fixed_framed | 10.934 |
| 50 | lite_entropy_reference_graph | 88.654 |
| 50 | lite_entropy_full_r2_graph | 90.197 |
| 50 | lite_entropy_full_r1_graph | 90.310 |
| 50 | lite_entropy_full_r0_graph | 99.777 |
| 100 | raw_bf16 | 5.033 |
| 100 | splitzip_official_framed | 4.053 |
| 100 | exans_official_fixed_framed | 7.299 |
| 100 | lite_entropy_reference_graph | 85.985 |
| 100 | lite_entropy_full_r2_graph | 87.529 |
| 100 | lite_entropy_full_r1_graph | 87.641 |
| 100 | lite_entropy_full_r0_graph | 97.108 |
| 200 | raw_bf16 | 2.684 |
| 200 | splitzip_official_framed | 2.285 |
| 200 | exans_official_fixed_framed | 5.482 |
| 200 | lite_entropy_reference_graph | 84.651 |
| 200 | lite_entropy_full_r2_graph | 86.194 |
| 200 | lite_entropy_full_r1_graph | 86.307 |
| 200 | lite_entropy_full_r0_graph | 95.774 |
| 400 | raw_bf16 | 1.510 |
| 400 | splitzip_official_framed | 1.401 |
| 400 | exans_official_fixed_framed | 4.573 |
| 400 | lite_entropy_reference_graph | 83.984 |
| 400 | lite_entropy_full_r2_graph | 85.527 |
| 400 | lite_entropy_full_r1_graph | 85.640 |
| 400 | lite_entropy_full_r0_graph | 95.107 |

Required decoder budgets against raw BF16 and official SplitZip, using complete packet sizes. The 400 Gb/s row is a compute-gap target, not an achieved link rate.

| Candidate | Baseline | Gb/s | Decode budget ms | Current decode ms | Further reduction |
|---|---|---:|---:|---:|---:|
| lite_entropy_reference_graph | raw_bf16 | 1 | 203.213 | 83.316 | 0.41x |
| lite_entropy_reference_graph | raw_bf16 | 2.5 | 81.486 | 83.316 | 1.02x |
| lite_entropy_reference_graph | raw_bf16 | 5 | 40.911 | 83.316 | 2.04x |
| lite_entropy_reference_graph | raw_bf16 | 10 | 20.623 | 83.316 | 4.04x |
| lite_entropy_reference_graph | raw_bf16 | 25 | 8.450 | 83.316 | 9.86x |
| lite_entropy_reference_graph | raw_bf16 | 50 | 4.393 | 83.316 | 18.97x |
| lite_entropy_reference_graph | raw_bf16 | 100 | 2.364 | 83.316 | 35.24x |
| lite_entropy_reference_graph | raw_bf16 | 200 | 1.350 | 83.316 | 61.73x |
| lite_entropy_reference_graph | raw_bf16 | 400 | 0.843 | 83.316 | 98.88x |
| lite_entropy_reference_graph | splitzip_official_framed | 1 | 87.208 | 83.316 | 0.96x |
| lite_entropy_reference_graph | splitzip_official_framed | 2.5 | 35.193 | 83.316 | 2.37x |
| lite_entropy_reference_graph | splitzip_official_framed | 5 | 17.855 | 83.316 | 4.67x |
| lite_entropy_reference_graph | splitzip_official_framed | 10 | 9.186 | 83.316 | 9.07x |
| lite_entropy_reference_graph | splitzip_official_framed | 25 | 3.985 | 83.316 | 20.91x |
| lite_entropy_reference_graph | splitzip_official_framed | 50 | 2.251 | 83.316 | 37.02x |
| lite_entropy_reference_graph | splitzip_official_framed | 100 | 1.384 | 83.316 | 60.21x |
| lite_entropy_reference_graph | splitzip_official_framed | 200 | 0.950 | 83.316 | 87.67x |
| lite_entropy_reference_graph | splitzip_official_framed | 400 | 0.734 | 83.316 | 113.57x |
| lite_entropy_full_r2_graph | raw_bf16 | 1 | 203.213 | 84.860 | 0.42x |
| lite_entropy_full_r2_graph | raw_bf16 | 2.5 | 81.486 | 84.860 | 1.04x |
| lite_entropy_full_r2_graph | raw_bf16 | 5 | 40.911 | 84.860 | 2.07x |
| lite_entropy_full_r2_graph | raw_bf16 | 10 | 20.623 | 84.860 | 4.11x |
| lite_entropy_full_r2_graph | raw_bf16 | 25 | 8.450 | 84.860 | 10.04x |
| lite_entropy_full_r2_graph | raw_bf16 | 50 | 4.393 | 84.860 | 19.32x |
| lite_entropy_full_r2_graph | raw_bf16 | 100 | 2.364 | 84.860 | 35.89x |
| lite_entropy_full_r2_graph | raw_bf16 | 200 | 1.350 | 84.860 | 62.87x |
| lite_entropy_full_r2_graph | raw_bf16 | 400 | 0.843 | 84.860 | 100.72x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 1 | 87.208 | 84.860 | 0.97x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 2.5 | 35.193 | 84.860 | 2.41x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 5 | 17.855 | 84.860 | 4.75x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 10 | 9.186 | 84.860 | 9.24x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 25 | 3.985 | 84.860 | 21.30x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 50 | 2.251 | 84.860 | 37.70x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 100 | 1.384 | 84.860 | 61.32x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 200 | 0.950 | 84.860 | 89.29x |
| lite_entropy_full_r2_graph | splitzip_official_framed | 400 | 0.734 | 84.860 | 115.67x |
| lite_entropy_full_r1_graph | raw_bf16 | 1 | 203.213 | 84.973 | 0.42x |
| lite_entropy_full_r1_graph | raw_bf16 | 2.5 | 81.486 | 84.973 | 1.04x |
| lite_entropy_full_r1_graph | raw_bf16 | 5 | 40.911 | 84.973 | 2.08x |
| lite_entropy_full_r1_graph | raw_bf16 | 10 | 20.623 | 84.973 | 4.12x |
| lite_entropy_full_r1_graph | raw_bf16 | 25 | 8.450 | 84.973 | 10.06x |
| lite_entropy_full_r1_graph | raw_bf16 | 50 | 4.393 | 84.973 | 19.34x |
| lite_entropy_full_r1_graph | raw_bf16 | 100 | 2.364 | 84.973 | 35.94x |
| lite_entropy_full_r1_graph | raw_bf16 | 200 | 1.350 | 84.973 | 62.95x |
| lite_entropy_full_r1_graph | raw_bf16 | 400 | 0.843 | 84.973 | 100.85x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 1 | 87.208 | 84.973 | 0.97x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 2.5 | 35.193 | 84.973 | 2.41x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 5 | 17.855 | 84.973 | 4.76x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 10 | 9.186 | 84.973 | 9.25x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 25 | 3.985 | 84.973 | 21.33x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 50 | 2.251 | 84.973 | 37.75x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 100 | 1.384 | 84.973 | 61.40x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 200 | 0.950 | 84.973 | 89.41x |
| lite_entropy_full_r1_graph | splitzip_official_framed | 400 | 0.734 | 84.973 | 115.82x |
| lite_entropy_full_r0_graph | raw_bf16 | 1 | 203.213 | 94.439 | 0.46x |
| lite_entropy_full_r0_graph | raw_bf16 | 2.5 | 81.486 | 94.439 | 1.16x |
| lite_entropy_full_r0_graph | raw_bf16 | 5 | 40.911 | 94.439 | 2.31x |
| lite_entropy_full_r0_graph | raw_bf16 | 10 | 20.623 | 94.439 | 4.58x |
| lite_entropy_full_r0_graph | raw_bf16 | 25 | 8.450 | 94.439 | 11.18x |
| lite_entropy_full_r0_graph | raw_bf16 | 50 | 4.393 | 94.439 | 21.50x |
| lite_entropy_full_r0_graph | raw_bf16 | 100 | 2.364 | 94.439 | 39.95x |
| lite_entropy_full_r0_graph | raw_bf16 | 200 | 1.350 | 94.439 | 69.97x |
| lite_entropy_full_r0_graph | raw_bf16 | 400 | 0.843 | 94.439 | 112.09x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 1 | 87.208 | 94.439 | 1.08x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 2.5 | 35.193 | 94.439 | 2.68x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 5 | 17.855 | 94.439 | 5.29x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 10 | 9.186 | 94.439 | 10.28x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 25 | 3.985 | 94.439 | 23.70x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 50 | 2.251 | 94.439 | 41.96x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 100 | 1.384 | 94.439 | 68.25x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 200 | 0.950 | 94.439 | 99.37x |
| lite_entropy_full_r0_graph | splitzip_official_framed | 400 | 0.734 | 94.439 | 128.73x |

Every entropy arm must reproduce the complete original packet bytes outside timing. Fresh GPU receivers decode fixed state plus unchanged packets and verify policy, decoder table identity, new data and output ownership.
The allocation flag is process-global; these are exclusive benchmark-worker contexts with restoration, not a concurrent serving API. Required explicit zeros/ones, deterministic algorithms and codec mathematics are unchanged.
Diagnostics/profiles run in separate processes after all ordinary measurements. Full traces remain local. Packet upload, cold capture, shared state and graph memory are additional.
Pinned official baseline methods retain their original worker/runtime policies. No fitting, model download, native build, actual network, SSD, pool attention or serving-concurrency measurement is performed.
