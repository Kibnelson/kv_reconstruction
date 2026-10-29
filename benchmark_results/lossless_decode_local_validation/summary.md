# Lossless decoder v2 local validation

**Software checks passed. No GPU performance was measured locally.**

The root protocol/regression run completed 37 tests: 34 passed and three CUDA cases were skipped. The separate mapped, context and integrated interpreter suites passed exact field/packet/word checks. The cohorts overlap; full details are in `validation.json`.

Checked all 65,536 BF16 patterns, original bundled wire parity, sparse/adjacent layer schedules, state carry across gaps, poisoned workspaces, malformed inputs and independent output ownership. Special-value fixtures produce expected interpreter NaN/Inf arithmetic warnings; bit equality assertions passed.

All 84 frozen source files and prior measurement/state receipts match. The previous 213-file completed A4500 package and its calibration chain verify. New Python compilation, Bash syntax and an independent code review passed. Existing tracked files are unchanged; derived SplitZip kernels retain the original MIT license and attribution.

Three actual-CUDA tests must execute without skips on the NUC before the benchmark. See `docs/lossless_decode_v2.md` for scope and the run command.
