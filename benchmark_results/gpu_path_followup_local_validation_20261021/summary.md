# Compact T1 identity framing and GPU bottleneck controls: local validation

The reference A4500 batch in commit `1e8d5bd` completed all eight CUDA unit
tests, 24 attention cases and 20 exact transfer cases. This directory records
CPU development validation of its follow-up, not another GPU result.

Local regression: **63 tests, 53 passed, 10 skipped** in 96.996 seconds with
the explicit Triton CPU interpreter. CUDA and optional capture integration
were unavailable. Compact and legacy envelopes round-trip identical entropy
bytes, reject mismatched/ambiguous identities, and reconstruct original words.
Prepared-parameter entropy encoding matches complete encoding. Diagnostic
orchestration, profile aggregation, restoration after errors and unavailable
CUDA profiling are checked. Packaging excludes captures and encoded KV.

The generated standalone runner was rebuilt and checked locally. Python and
Bash syntax, diff whitespace, frozen inputs and historical report compatibility
passed. The three GPU kernel source files are byte-identical to the measured
A4500 reference; no new kernel compilation result is claimed. The host helper
will regenerate and publish the large standalone artifact before GPU tests.

`identity_projection.json` reconstructs all 20 old envelope lengths exactly.
Replacing the full repeated model identity with its verified SHA256 removes
5,564 bytes per Qwen packet and 283 per SmolLM packet. Holding the measured
payload/directory counts fixed projects Qwen B at 8.853500 complete bpv,
44.67% below original BF16 bytes. This is an accounting projection.

Actual compact-packet CUDA restore, new timings and CUDA profiling require
the next A4500 run:

```bash
bash scripts/run_gpu_path_a4500.sh --publish-results --diagnostics
```

Read `docs/gpu_path_a4500_20261021_review.md` for the measured reference
results, limitations and full fetch/run command.
