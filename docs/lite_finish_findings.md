# Finish fusion: measured result and next experiment

The A4500 finish-fusion run completed at commit `66d82eb`. It used the
unchanged original BF16 Llama-3.2-3B cache, all 28 layers, on the same two
512-token documents. Results below average per-document synchronized wall
medians from paired AB/BA calls.

Source: [published report](../benchmark_results/lite_finish_a4500_20261028_101451_154264_report_20261028_101606/summary.md).

| Method | Complete packet bytes, mean | Saved versus BF16 | Decode ms |
|---|---:|---:|---:|
| Raw BF16 | 58,720,573 | -0.00054% | 0.350 |
| Official SplitZip, framed | 44,197,315 | 24.73% | 0.512 |
| Official ExANS, fixed slots and framed | 45,433,689 | 22.63% | 3.660 |
| Lite, previous graph | 33,360,893 | 43.19% | 96.206 |
| Lite, finish-fused graph | 33,360,893 | 43.19% | 88.224 |

The new path removes **7.982 ms, or 8.30%**, without changing the wire format,
fitted predictors, probability model or adaptive reduction. Every packet SHA
matches, the new encoder reproduces the complete original packet, and a fresh
packet-only receiver verifies every original word. Alternating packet lengths
reuse one graph per arm; returned outputs retain independent ownership.

## What remains expensive

Separate eager profiles identify work; their times are not a breakdown of
ordinary graph latency and must not be added to phase medians.

| Profile quantity | Previous graph engine | Finish engine |
|---|---:|---:|
| Device kernel calls | 18,877 | 14,397 |
| Sum of device-kernel time, ms | 100.174 | 90.930 |
| Entropy decode kernels, ms | 49.490 | 49.472 |
| Allocator/explicit fill kernels, ms | 8.470 | 5.927 |

The unchanged entropy kernel now accounts for 54.4% of the instrumented kernel
sum. There are also 3,727 fill kernels in the finish profile. This last count
includes necessary initialization, so 5.927 ms is an opportunity bound rather
than a predicted speed gain.

## The next isolated test: allocation fills

Lite's harness enables deterministic algorithms. PyTorch also fills otherwise
uninitialized allocations by default under that policy. Many decoder
temporaries are subsequently overwritten completely. PyTorch documents that
the fill can be disabled while retaining determinism when no operation reads
uninitialized input:
[PyTorch 2.11 allocation-fill documentation](https://docs.pytorch.org/docs/2.11/deterministic.html).

Compare the same finish engine and graph with the fill setting on and off.
Retain deterministic algorithms, matmul settings, all arithmetic, explicit
zeros/ones, packet validation, staging and owned output copies. Audit custom
buffer writes and test multiple initial poison patterns, packet lengths and
nonzero positions before using the off arm. Restore the process-global setting
after each scope, including failures; this is an exclusively owned benchmark
process experiment, not a concurrent-serving policy API.

Official controls retain their original workers, source pins, environments and
runtime settings. Their original worker does not enable Torch deterministic
algorithms, so this extra fill policy is already inactive there. Enabling it
only to create a slower baseline would change the comparison. Record effective
flags and retime the unchanged controls instead.

This is execution overhead removal. The proposed research contribution remains
conditional exact BF16 coding from recovered KV; allocator settings are not a
novel compression method.

Run on the existing A4500 checkout after pulling the source branch:

```bash
bash scripts/run_lite_allocation_a4500.sh
```

The wrapper runs actual-CUDA correctness gates, pairs both policies, retimes
official controls, gathers separate profiles, packages the small report and
publishes it. It reuses the completed finish run and existing fixed state;
there is no refitting or model download. Resume an interrupted benchmark with
`bash scripts/run_lite_allocation_a4500.sh --resume RUN_DIR`.
The new allocation experiment has software evidence only until that GPU run.

## Latency targets after this run

For precompressed fetch, calculate `decode_ms + 8*packet_bytes/(Gbps*1e6)`.
No network is measured by this formula. Fresh encode, serialization, upload,
cold capture, SSD and queueing are separate.

| Effective link | Lite fetch ms | SplitZip fetch ms | Lite decode budget to match SplitZip, ms |
|---:|---:|---:|---:|
| 1 Gb/s | 355.111 | 354.091 | 87.204 |
| 5 Gb/s | 141.601 | 71.228 | 17.851 |
| 10 Gb/s | 114.913 | 35.870 | 9.182 |
| 100 Gb/s | 90.893 | 4.048 | 1.379 |
| 400 Gb/s | 88.891 | 1.396 | 0.729 |

Lite sends 24.52% fewer complete packet bytes than SplitZip here. Another
1.020 ms reduction reaches modeled 1 Gb/s parity, but 400 Gb/s requires about
121 times faster decoding. A small allocation or fusion improvement cannot
close that gap. The A4500's PCIe 4 x16 ingress also cannot physically sustain
50 GB/s; the 400 Gb/s row remains a compute-gap target.

After this allocation test, the next same-format candidates are cached static
group spreads, pre-coder preparation fusion, and a full stage-1 inverse CDF
lookup using the existing frequencies. The full lookup replaces search without
changing the entropy mapping; its cache footprint and actual speed need a
separate measurement. The larger high-link work still needs a cheaper exact
entropy coder and cheaper predictors or fewer dependency waves, as described in
[the high-link roadmap](lite_high_link_roadmap.md).
