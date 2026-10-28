# First A4500 Lite profile findings

Evidence: [published run 04ffa73](https://github.com/Kibnelson/kv_reconstruction/tree/04ffa7385ccbb810b125a623388a78d9e6cb4cd2/benchmark_results/llama_profile_official_a4500_20261027_194103_2478724_report_20261027_194755).

Profiling completed with identical packet and decoded-word digests. The
lossless 512-token/all-28-layer text request measured 328.732 ms encode and
416.329 ms decode in the ordinary control. Instrumented times were 570.485 and
674.450 ms, so instrumented wall time is not an optimization result.

| Profiled lossless request | Encode | Decode |
|---|---:|---:|
| CUDA launch API calls | 37,103 | 44,256 |
| cudaStreamSynchronize calls | 1,794 | 1,793 |
| Forward RoPE scope calls | 448 | 448 |
| Inverse RoPE scope calls | 448 | 448 |

`Rope.tables()` forms its cache key with `int(positions[0])` and
`int(positions[-1])`. Positions are on CUDA, so these Python integer conversions
read GPU scalars back to the host. Every group invokes table lookup twice:
forward prediction and inverse reconstruction. Thus 448 × 2 × 2 gives 1,792
scalar readbacks, accounting for almost all observed synchronizations. This
explains why caching the trig values alone did not remove the overhead: even
a cache hit first constructs the key using device reads.

The next controlled optimization should form the key from the CPU-known request
start/length/device and reuse the same float64 tables. Preserve every arithmetic
operation and require the existing packet SHA256 and receiver word digests to
match. Measure the resulting unprofiled encode/decode time and launch/sync counts
before considering wider changes. No speedup from that proposed change has been
measured yet; the current baseline repair does not modify the measured core.

Many additional launches remain for indexing, conversions, scatter and
elementwise work. Removing scalar readbacks will not establish that the whole
latency gap has been solved. After that isolated experiment, profile again to
choose which operations to fuse or capture.

CUDA user-annotation ranges named `kv/...` are inclusive timeline spans and may
contain idle gaps. They are not standalone kernels. The updated report excludes
them from the top operator list; existing full trace and profile evidence are
preserved. CPU/self-device/inclusive times should not be summed indiscriminately.

Official AQUA BF16 WikiText2 PPL completed at 6.986454579. Quantized AQUA did not
run because calibration data was inaccessible. ExANS built, but its standalone
test could not locate libcudart.so.12. Neither failure is a codec performance
result. Recovery addresses runtime-library discovery and offers an explicit,
fully identified backup calibration route without replacing original-data
provenance silently.
