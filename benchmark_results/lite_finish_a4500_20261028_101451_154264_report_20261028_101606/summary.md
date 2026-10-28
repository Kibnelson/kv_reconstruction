# Exact post-decode finish kernel under CUDA graph replay

Status: **completed**. GPU measured: **True**.

V64/B32 packets and fitted predictors are unchanged. Reference and finish graphs alternate AB/BA on two different-sized packets.
Both engines and both graph buffers are resident during paired timing. Full decode includes validation, staging copies, replay and an independently owned output clone.

| Method | Saved | Complete bpv | Decode ms | Exact |
|---|---:|---:|---:|---|
| raw_bf16 | -0.00% | 16.00009 | 0.350 | True |
| splitzip_official_framed | 24.73% | 12.04281 | 0.512 | True |
| exans_official_fixed_framed | 22.63% | 12.37970 | 3.660 | True |
| lite_finish_reference_graph | 43.19% | 9.09012 | 96.206 | True |
| lite_finish_finish_graph | 43.19% | 9.09012 | 88.224 | True |

Interface metadata only; current generation can be lower at idle. This is not a transfer-bandwidth measurement.

`name,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max`

```text
NVIDIA RTX A4500, 580.126.20, 4, 4, 16, 16
```

Cold graph setup and cumulative paired-setup memory; Torch accounting excludes some driver metadata.

| Arm | Cold ms | Allocated before → after MiB | Reserved before → after MiB | Peak allocated / reserved MiB | Staging payload MiB | Directory KiB | Output MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference_graph | 784.196 | 746.160 → 948.271 | 1254.000 → 1106.000 | 1035.655 / 1444.000 | 113.877 | 112.125 | 56.000 |
| finish_graph | 815.114 | 948.271 → 1150.381 | 1106.000 → 1352.000 | 1237.516 / 1408.000 | 113.877 | 112.125 | 56.000 |

Separate synchronized phase diagnostics; raw samples remain JSON and are not ordinary paired latencies.

| Arm / document | Validation wall ms | Input-copy CUDA ms | Replay CUDA ms | Replay CPU submit ms | Output-clone CUDA ms |
|---|---:|---:|---:|---:|---:|
| reference_graph / test_code | 0.557 | 0.148 | 92.636 | 0.084 | 0.329 |
| reference_graph / test_text | 0.516 | 0.142 | 94.507 | 0.091 | 0.335 |
| finish_graph / test_code | 0.578 | 0.149 | 86.532 | 0.041 | 0.327 |
| finish_graph / test_text | 0.576 | 0.149 | 86.710 | 0.082 | 0.328 |

Separate instrumented eager profiles; these identify kernel work, not captured-graph latency.

| Arm | Kernel calls | Sum kernel ms | Finish-kernel calls | Finish-kernel ms |
|---|---:|---:|---:|---:|
| reference_graph | 18877 | 100.174 | 0 | 0.000 |
| finish_graph | 14397 | 90.930 | 448 | 3.002 |

Top device kernels for reference_graph:

- `_decode_kernel`: 49.490 ms, 448 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 4.916 ms, 3277 calls.
- `_rotate_group`: 4.736 ms, 896 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.950 ms, 896 calls.
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)`: 3.240 ms, 895 calls.

Top device kernels for finish_graph:

- `_decode_kernel`: 49.472 ms, 448 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 4.325 ms, 2829 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.910 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`: 3.048 ms, 1764 calls.
- `_finish_kernel`: 3.002 ms, 448 calls.

Calculated serial precompressed fetch, not measured network/serving latency. No fresh encoder projection.

| Gb/s | Method | Transfer + decode ms |
|---:|---|---:|
| 1 | raw_bf16 | 470.114 |
| 1 | splitzip_official_framed | 354.091 |
| 1 | exans_official_fixed_framed | 367.130 |
| 1 | lite_finish_reference_graph | 363.093 |
| 1 | lite_finish_finish_graph | 355.111 |
| 2.5 | raw_bf16 | 188.256 |
| 2.5 | splitzip_official_framed | 141.944 |
| 2.5 | exans_official_fixed_framed | 149.048 |
| 2.5 | lite_finish_reference_graph | 202.960 |
| 2.5 | lite_finish_finish_graph | 194.979 |
| 5 | raw_bf16 | 94.303 |
| 5 | splitzip_official_framed | 71.228 |
| 5 | exans_official_fixed_framed | 76.354 |
| 5 | lite_finish_reference_graph | 149.583 |
| 5 | lite_finish_finish_graph | 141.601 |
| 10 | raw_bf16 | 47.326 |
| 10 | splitzip_official_framed | 35.870 |
| 10 | exans_official_fixed_framed | 40.007 |
| 10 | lite_finish_reference_graph | 122.894 |
| 10 | lite_finish_finish_graph | 114.913 |
| 25 | raw_bf16 | 19.140 |
| 25 | splitzip_official_framed | 14.656 |
| 25 | exans_official_fixed_framed | 18.199 |
| 25 | lite_finish_reference_graph | 106.881 |
| 25 | lite_finish_finish_graph | 98.899 |
| 50 | raw_bf16 | 9.745 |
| 50 | splitzip_official_framed | 7.584 |
| 50 | exans_official_fixed_framed | 10.930 |
| 50 | lite_finish_reference_graph | 101.543 |
| 50 | lite_finish_finish_graph | 93.562 |
| 100 | raw_bf16 | 5.047 |
| 100 | splitzip_official_framed | 4.048 |
| 100 | exans_official_fixed_framed | 7.295 |
| 100 | lite_finish_reference_graph | 98.874 |
| 100 | lite_finish_finish_graph | 90.893 |
| 200 | raw_bf16 | 2.699 |
| 200 | splitzip_official_framed | 2.280 |
| 200 | exans_official_fixed_framed | 5.478 |
| 200 | lite_finish_reference_graph | 97.540 |
| 200 | lite_finish_finish_graph | 89.558 |
| 400 | raw_bf16 | 1.524 |
| 400 | splitzip_official_framed | 1.396 |
| 400 | exans_official_fixed_framed | 4.569 |
| 400 | lite_finish_reference_graph | 96.873 |
| 400 | lite_finish_finish_graph | 88.891 |

Required decoder budgets against raw BF16 and official SplitZip, using complete packet sizes. The 400 Gb/s row is a compute-gap target, not an achieved link rate.

| Candidate | Baseline | Gb/s | Decode budget ms | Current decode ms | Further reduction |
|---|---|---:|---:|---:|---:|
| lite_finish_reference_graph | raw_bf16 | 1 | 203.227 | 96.206 | 0.47x |
| lite_finish_reference_graph | raw_bf16 | 2.5 | 81.501 | 96.206 | 1.18x |
| lite_finish_reference_graph | raw_bf16 | 5 | 40.925 | 96.206 | 2.35x |
| lite_finish_reference_graph | raw_bf16 | 10 | 20.637 | 96.206 | 4.66x |
| lite_finish_reference_graph | raw_bf16 | 25 | 8.465 | 96.206 | 11.37x |
| lite_finish_reference_graph | raw_bf16 | 50 | 4.407 | 96.206 | 21.83x |
| lite_finish_reference_graph | raw_bf16 | 100 | 2.378 | 96.206 | 40.45x |
| lite_finish_reference_graph | raw_bf16 | 200 | 1.364 | 96.206 | 70.53x |
| lite_finish_reference_graph | raw_bf16 | 400 | 0.857 | 96.206 | 112.27x |
| lite_finish_reference_graph | splitzip_official_framed | 1 | 87.204 | 96.206 | 1.10x |
| lite_finish_reference_graph | splitzip_official_framed | 2.5 | 35.189 | 96.206 | 2.73x |
| lite_finish_reference_graph | splitzip_official_framed | 5 | 17.851 | 96.206 | 5.39x |
| lite_finish_reference_graph | splitzip_official_framed | 10 | 9.182 | 96.206 | 10.48x |
| lite_finish_reference_graph | splitzip_official_framed | 25 | 3.980 | 96.206 | 24.17x |
| lite_finish_reference_graph | splitzip_official_framed | 50 | 2.246 | 96.206 | 42.83x |
| lite_finish_reference_graph | splitzip_official_framed | 100 | 1.379 | 96.206 | 69.75x |
| lite_finish_reference_graph | splitzip_official_framed | 200 | 0.946 | 96.206 | 101.71x |
| lite_finish_reference_graph | splitzip_official_framed | 400 | 0.729 | 96.206 | 131.95x |
| lite_finish_finish_graph | raw_bf16 | 1 | 203.227 | 88.224 | 0.43x |
| lite_finish_finish_graph | raw_bf16 | 2.5 | 81.501 | 88.224 | 1.08x |
| lite_finish_finish_graph | raw_bf16 | 5 | 40.925 | 88.224 | 2.16x |
| lite_finish_finish_graph | raw_bf16 | 10 | 20.637 | 88.224 | 4.27x |
| lite_finish_finish_graph | raw_bf16 | 25 | 8.465 | 88.224 | 10.42x |
| lite_finish_finish_graph | raw_bf16 | 50 | 4.407 | 88.224 | 20.02x |
| lite_finish_finish_graph | raw_bf16 | 100 | 2.378 | 88.224 | 37.09x |
| lite_finish_finish_graph | raw_bf16 | 200 | 1.364 | 88.224 | 64.68x |
| lite_finish_finish_graph | raw_bf16 | 400 | 0.857 | 88.224 | 102.96x |
| lite_finish_finish_graph | splitzip_official_framed | 1 | 87.204 | 88.224 | 1.01x |
| lite_finish_finish_graph | splitzip_official_framed | 2.5 | 35.189 | 88.224 | 2.51x |
| lite_finish_finish_graph | splitzip_official_framed | 5 | 17.851 | 88.224 | 4.94x |
| lite_finish_finish_graph | splitzip_official_framed | 10 | 9.182 | 88.224 | 9.61x |
| lite_finish_finish_graph | splitzip_official_framed | 25 | 3.980 | 88.224 | 22.17x |
| lite_finish_finish_graph | splitzip_official_framed | 50 | 2.246 | 88.224 | 39.28x |
| lite_finish_finish_graph | splitzip_official_framed | 100 | 1.379 | 88.224 | 63.96x |
| lite_finish_finish_graph | splitzip_official_framed | 200 | 0.946 | 88.224 | 93.27x |
| lite_finish_finish_graph | splitzip_official_framed | 400 | 0.729 | 88.224 | 121.00x |

Every finish encoder check must reproduce the complete original packet bytes outside timing. Fresh GPU receivers only decode fixed state plus unchanged packets and verify new data and output ownership.
Diagnostics/profiles run in separate processes after all ordinary measurements. Full traces remain local. Packet upload, cold capture, shared state and graph memory are additional.
Pinned official baseline methods retain their original worker/runtime policies. No fitting, model download, native build, actual network, SSD, pool attention or serving-concurrency measurement is performed.
