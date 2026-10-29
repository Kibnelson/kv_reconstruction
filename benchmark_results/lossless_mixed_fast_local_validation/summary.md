# Lossless mixed execution local validation

**Software checks passed. No GPU performance was measured locally.**

The recorded root protocol/regression run completed 37 tests: 33 passed and four actual-CUDA tests were skipped. Separate local interpreter suites passed the selective and bundled packet-parity, exact-decode, malformed-input and output-ownership cases. These cohorts overlap; their counts must not be added. Full cohort details are in `validation.json`.

Selective preparation preserves the complete reference packet. The combined change preserves the bundled packet, and all four arms preserve the rANS payload and directories. CPU fixtures and the Triton interpreter are semantic checks, not GPU measurements.

All 84 frozen source files and all 18 measurement files from the prior completed run match their original receipts. The prior 177-file A4500 package and policy digests verify. Compilation, shell syntax, parent imports and an independent code review passed. Only new files are added on the isolated branch.

The NUC runner must pass four actual-CUDA tests without skips before benchmarking. See `docs/lossless_mixed_fast_v1.md` for the experiment and command.
