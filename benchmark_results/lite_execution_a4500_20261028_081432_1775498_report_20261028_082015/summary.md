# Lite execution study

Status: **completed**. GPU measured: **True**.

Complete original packets, unchanged fit and entropy kernels. Timings come from fresh unprofiled processes.
Reference B8 is repeated before/after the lossless ablations. Reference B32 brackets the lossy comparison.

| Method | Kappa | Saved | Encode ms | Decode ms | Exact original |
|---|---:|---:|---:|---:|---|
| raw_bf16 | 0 | -0.00% | 0.414 | 0.344 | True |
| splitzip_official_framed | 0 | 24.73% | 0.871 | 0.535 | True |
| fp8_e4m3fn_saturating | 0 | 50.00% | 1.238 | 0.413 | False |
| exans_official_fixed_framed | 0 | 22.63% | 15.149 | 3.664 | True |
| lite_b8_ref_before | 0 | 44.17% | 212.698 | 249.423 | True |
| lite_b8_scatter | 0 | 44.17% | 199.777 | 204.451 | True |
| lite_b8_rope | 0 | 44.17% | 186.538 | 216.061 | True |
| lite_b8_fused | 0 | 44.17% | 157.745 | 158.274 | True |
| lite_b8_ref_after | 0 | 44.17% | 225.623 | 253.756 | True |
| lite_b32_ref_before | 0.5 | 75.00% | 372.365 | 299.215 | False |
| lite_b32_ref_before | 1 | 80.56% | 374.710 | 298.433 | False |
| lite_b32_fused | 0.5 | 75.00% | 291.321 | 214.833 | False |
| lite_b32_fused | 1 | 80.56% | 290.914 | 213.527 | False |
| lite_b32_ref_after | 0.5 | 75.00% | 370.277 | 297.782 | False |
| lite_b32_ref_after | 1 | 80.56% | 371.578 | 295.526 | False |

Separate instrumented diagnostics. Kernel times exclude inclusive aten/user ranges; do not add host and device totals.

| Profile | Direction | GPU kernel calls | Sum kernel ms | GPU busy union ms | First-to-last kernel span ms |
|---|---|---:|---:|---:|---:|
| lite_b8_ref_before | encode | 37737 | 180.950 | 180.950 | 481.963 |
| lite_b8_ref_before | decode | 44889 | 230.094 | 230.094 | 507.755 |
| lite_b8_fused | encode | 19341 | 109.720 | 109.720 | 325.103 |
| lite_b8_fused | decode | 18877 | 127.186 | 127.186 | 312.694 |
| lite_b32_ref_before | encode | 72682 | 247.296 | 247.296 | 788.457 |
| lite_b32_ref_before | decode | 57432 | 266.263 | 266.263 | 610.438 |
| lite_b32_fused | encode | 46670 | 145.233 | 145.233 | 594.287 |
| lite_b32_fused | decode | 31421 | 163.201 | 163.201 | 418.565 |

Top kernels for lite_b8_ref_before / encode:

- `_encode_kernel`: 53.994 ms, 448 calls.
- `void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool)`: 25.602 ms, 476 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 13.411 ms, 2688 calls.
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)`: 5.985 ms, 1791 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 4.856 ms, 3277 calls.

Top kernels for lite_b8_ref_before / decode:

- `_decode_kernel`: 76.550 ms, 448 calls.
- `void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool)`: 25.828 ms, 476 calls.
- `void (anonymous namespace)::indexing_backward_kernel_stride_1<short>(long const*, long const*, short const*, short*, long, long, long, long, bool)`: 13.811 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 13.447 ms, 2688 calls.
- `void at_cuda_detail::cub::DeviceRadixSortSingleTileKernel<at_cuda_detail::cub::DeviceRadixSortPolicy<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy900, false, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, at_cuda_detail::cub::detail::identity_decomposer_t>(long const*, long*, at::cuda::cub::detail::OpaqueType<8> const*, at::cuda::cub::detail::OpaqueType<8>*, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t)`: 9.017 ms, 924 calls.

Top kernels for lite_b8_fused / encode:

- `_encode_kernel`: 54.485 ms, 448 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 4.952 ms, 3277 calls.
- `_rotate_group`: 4.733 ms, 896 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.968 ms, 896 calls.
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)`: 3.264 ms, 895 calls.

Top kernels for lite_b8_fused / decode:

- `_decode_kernel`: 76.431 ms, 448 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 4.940 ms, 3277 calls.
- `_rotate_group`: 4.730 ms, 896 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 3.958 ms, 896 calls.
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)`: 3.273 ms, 895 calls.

Top kernels for lite_b32_ref_before / encode:

- `_res_encode`: 42.398 ms, 448 calls.
- `void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool)`: 25.600 ms, 476 calls.
- `void (anonymous namespace)::indexing_backward_kernel_stride_1<short>(long const*, long const*, short const*, short*, long, long, long, long, bool)`: 13.806 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 13.656 ms, 2688 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 10.496 ms, 7757 calls.

Top kernels for lite_b32_ref_before / decode:

- `_res_decode`: 91.354 ms, 448 calls.
- `void (anonymous namespace)::indexing_backward_kernel_stride_1<float>(long const*, long const*, float const*, float*, long, long, long, long, bool)`: 25.678 ms, 476 calls.
- `void (anonymous namespace)::indexing_backward_kernel_stride_1<short>(long const*, long const*, short const*, short*, long, long, long, long, bool)`: 13.785 ms, 448 calls.
- `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})`: 13.523 ms, 2688 calls.
- `void at_cuda_detail::cub::DeviceRadixSortSingleTileKernel<at_cuda_detail::cub::DeviceRadixSortPolicy<long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long>::Policy900, false, long, at::cuda::cub::detail::OpaqueType<8>, unsigned long long, at_cuda_detail::cub::detail::identity_decomposer_t>(long const*, long*, at::cuda::cub::detail::OpaqueType<8> const*, at::cuda::cub::detail::OpaqueType<8>*, unsigned long long, int, int, at_cuda_detail::cub::detail::identity_decomposer_t)`: 9.017 ms, 924 calls.

Top kernels for lite_b32_fused / encode:

- `_res_encode`: 42.533 ms, 448 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 10.540 ms, 7757 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)`: 8.437 ms, 6274 calls.
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)`: 6.901 ms, 2689 calls.
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)`: 6.008 ms, 1791 calls.

Top kernels for lite_b32_fused / decode:

- `_res_decode`: 91.110 ms, 448 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)`: 5.523 ms, 3725 calls.
- `_rotate_group`: 4.748 ms, 896 calls.
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)`: 4.667 ms, 3584 calls.
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)`: 4.583 ms, 1343 calls.

Lossless fetch targets against retimed official SplitZip. The decode budget includes the candidate packet-size advantage.

| Candidate | Decode ms | Required decode ms at 5 Gb/s | Current / required |
|---|---:|---:|---:|
| lite_b8_ref_before | 249.423 | 18.794 | 13.27x |
| lite_b8_scatter | 204.451 | 18.794 | 10.88x |
| lite_b8_rope | 216.061 | 18.794 | 11.50x |
| lite_b8_fused | 158.274 | 18.794 | 8.42x |
| lite_b8_ref_after | 253.756 | 18.794 | 13.50x |

Calculated serial link model, not measured network latency.

| Gb/s | Best fresh encode + transfer + decode | Best precompressed transfer + decode |
|---:|---|---|
| 0.1 | lite_b8_fused | lite_b8_fused |
| 0.25 | lite_b8_fused | lite_b8_fused |
| 0.5 | splitzip_official_framed | lite_b8_fused |
| 1 | splitzip_official_framed | splitzip_official_framed |
| 2.5 | splitzip_official_framed | splitzip_official_framed |
| 5 | splitzip_official_framed | splitzip_official_framed |
| 10 | splitzip_official_framed | splitzip_official_framed |
| 25 | splitzip_official_framed | splitzip_official_framed |
| 50 | splitzip_official_framed | splitzip_official_framed |
| 100 | splitzip_official_framed | splitzip_official_framed |

Every Lite row must match its previous packet bytes and intended BF16 reconstruction. Nonzero kappa remains lossy.
Raw BF16, FP8, official SplitZip and official ExANS are retimed using the previous checked worker requests and unchanged baseline harness.
Shared state, cold initialization and per-call allocation peaks are additional. Kernel fusion currently uses the reference path for attention scaling other than 1.
Lossy quality is the prior short continuation check, not full task validation. Official AQUA remains blocked.
Profiles run in separate processes after ordinary measurements. Full traces stay in runs/. No SSD, actual network, pool attention or serving result is measured.
