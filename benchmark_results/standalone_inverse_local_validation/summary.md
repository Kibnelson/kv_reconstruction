# Standalone inverse local validation

Status: **completed**. GPU execution/performance measured: **False**.

20 CPU tests passed: 14 protocol tests, four core tests and two controller
integration tests. Both new algorithms reproduce all 655,360 fixed CDF
class/slot inverses exactly. Tests also cover mixed resolved/tail lanes,
bucket boundaries, safe sparse addresses, table ownership and preservation
of native word arithmetic and the layer decoder. Both controller modes
produce original complete packets and reconstruct exact BF16 words.

All eight actual decoder variants compile offline for CUDA SM86: packed and
masked algorithms, each with uniform first/middle/last and exact middle.
Production pointer types are used, including the masked INT32 placeholder.
Interpreter mode is disabled for compilation so GPU assembly is checked.
Compiler wall times and PTX hashes are provenance, not GPU performance.
The standalone reproduction script and compile receipts are included.

An initial copied-helper namespace mismatch was caught by packet integration
before publication. It was corrected, an explicit namespace guard was added,
and the full codec cohort passed on rerun. The initial failure is retained
separately. Expected nonfinite-operation warnings remain in the passing log.

Six new Python source files and the offline helper parse; Bash syntax and
staged whitespace checks pass. Independent core/controller and runner reviews
found no blocker. Only new files are added; existing tracked sources and
native baseline implementations remain unchanged.

The NUC comparison requires three actual CUDA gates before timing, 26 workers,
52 rows and 26 fresh receivers. All 24 own rows preserve frozen packet SHA;
16 candidate/document groups compare against current single-feature/two-chunk
references. Packed mode adds 45,568 persistent tensor bytes; masked adds zero.
No GPU speedup is claimed by this local validation.
