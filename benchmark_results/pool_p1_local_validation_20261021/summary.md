# P1 local validation: CUDA measurements pending

Task 1 implements GPU packing into a bounded P1 head-sector arena and selected
head-tile unpacking. The development host has no CUDA device. These results
validate representation and kernel logic on CPU and compile kernels offline;
they do not establish A4500 correctness or performance.

- 34 tests: 30 passed, four skipped. Two P1 CUDA tests, one P0 CUDA test and
  one optional fresh-capture integration test require unavailable resources.
- Six tests execute the actual Triton kernel source through its CPU interpreter.
  They cover mixed widths, ordered escapes, every BF16 bit pattern through
  packed and raw paths, partial tails, invalid reads/capacity rejection,
  Qwen head dimension 128, and one full frozen model page.
- All 12 kernel/shape variants compile to SM86 cubins with Triton 3.6.0.
  No cubin was executed on CUDA.
- The independent NumPy oracle exactly round-trips 1,474,560 original words
  over four frozen real-model pages. All four fit the declared 80% arena.
- A request for CUDA fails with exit code 1 and all GPU coverage flags false.
  The generated standalone source also matches the reviewable source.

| Frozen page | Required payload bpv | Arena plus directory bpv | CPU exact |
|---|---:|---:|---|
| page_000 | 11.716667 | 13.050000 | True |
| page_001 | 11.397569 | 13.050000 | True |
| page_002 | 11.665278 | 13.050000 | True |
| page_003 | 11.360764 | 13.050000 | True |

The 80% arena is 589,824 bytes per page, with an 11,520-byte directory.
Calibration is 23,040 shared bytes and the encoder scan workspace is 2,880
bytes. The projected packet plus calibration is 13.55 bits/value on these
isolated pages. Original input, output tiles and encoder workspace are
additional live allocations. A 100% arena control budgets 16.25 bits/value
before calibration. Payload bpv alone is not resident GPU memory savings.

Actual A4500 pack/unpack timing, byte parity, peak allocation/reservation and
paired P0 controls remain pending. Use the command in
[the P1 guide](../../docs/pool_p1_gpu.md), review that report, and obtain approval
before starting the next task. T entropy coding, T-to-P, compressed attention
and serving capacity are outside this implementation.

`validation.json` binds these checks to source hashes. The full CPU oracle
report, test log and offline compilation metadata are included alongside it.
