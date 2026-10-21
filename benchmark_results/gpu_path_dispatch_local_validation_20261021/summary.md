# Exact prediction input chunks and isolated profiling: local validation

Reference commit `9df187e` contains the completed A4500 compact-packet run:
eight CUDA tests, 24 attention cases and 20 exact transfers. Qwen B measures
8.853500 complete bpv, a 44.67% BF16 reduction. This directory validates the
next host implementation controls; it does not report new GPU measurements.

Local regression: **69 tests, 58 passed, 11 skipped**, 106.468 seconds with
the explicit Triton CPU interpreter. Hardware and optional capture integration
were unavailable. New checks cover independent NumPy integer products with
769 columns, keys/RoPE and values, 64/512/all-column grouping, identical
complete packet bytes and decoding across implementations.

`frozen_parameters.json` additionally checks all 30 layers and 360 groups of
one original frozen SmolLM page. Every mean/inverse-width/offset tensor and
state identity is identical across the three implementations. This uses known
KV to validate encoder parameters on CPU; it is not entropy decoding.

The profiler regression excludes both CPU parents and CUDA user-annotation
ranges. Diagnostic orchestration delegates profiling to a fresh packet-only
child; the parent never enables it. Failed byte-parity reports remain failed,
and A/B comparisons match the prediction implementation.

Standalone rebuild/check, Python/Bash syntax, diff whitespace, CPU-only frozen
input validation and historical-report compatibility passed. All three GPU
kernel source files are byte-identical to the successful A4500 reference.
The host helper regenerates/publishes the large standalone artifact.

Run all available cases with:

```bash
bash scripts/run_gpu_path_a4500.sh --publish-results \
  --predictor-input-chunks 64,512,0 --diagnostics
```

Actual variant CUDA correctness, latency, transient memory and corrected
profiles remain pending. Read `docs/gpu_path_dispatch_a4500.md` for the full
fetch/run command and the old profiler-total correction.
