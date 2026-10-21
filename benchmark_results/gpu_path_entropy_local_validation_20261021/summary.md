# Exact entropy scheduling: local validation

Status: **passed local checks; new GPU execution and timings pending**.
Base: `e3e296d82f2f71fb1f4aabe6bf8c0690e733bc10`.

The completed A4500 input-chunk run passed nine CUDA tests, 24 attention
cases and 60 transfer implementation cases. All 40 larger-chunk alternatives
preserved complete T1 packet bytes. `reference_metrics.json` aggregates that
existing report; its timings do not describe the new cached backend.

The next experiment fixes the input reduction chunk at 512 and compares
`scalar,cached`. Cached encoding prepares the same 16 binary frequencies in
parallel, using one reusable UINT16 `[tokens,group,16]` array. At 32 tokens
and 32 channels this is 32 KiB of additional temporary workspace, not packet
or shared-state data. Cached decoding reuses interval endpoint masses while
decoding causally. It needs 18 boundary evaluations per word instead of 48.
The format, state identity, probability model, rank order and rANS order stay
the same. No latency gain follows from operation counts alone.

Local verification:

- 72 tests discovered: 60 passed, 12 skipped, in 115.863 seconds. Eleven
  require actual CUDA; one requires fresh real-model captures. See `tests.log`.
- The CPU interpreter executes actual Triton kernels and checks all 16
  cached probability planes against an independent oracle, including varied
  means/scales and a masked tail. Complete packet bytes, cross-backend exact
  decoding, corrupt-lane rejection and the pool/file/attention bridge pass.
- All 26 offline SM86 kernel variants compile with Triton 3.6.0. This is
  compilation, not CUDA execution; see `sm86.json`.
- CPU fixture/input validation, historical 60-case report compatibility,
  Python compilation, Bash syntax, whitespace and standalone build checks pass.

The generated standalone is checked locally. Source publication excludes
that generated file; `scripts/run_gpu_path_a4500.sh` rebuilds and publishes
it before actual CUDA tests. `validation.json` records its expected digest.

Actual cached-backend CUDA probability parity, all-65,536-pattern execution,
40 matched transfer cases, timings and allocation peaks remain pending on
the A4500. Native generation/concurrency, cold SSD and real network remain
separate work. See `docs/gpu_path_entropy_a4500.md` for the command and limits.
