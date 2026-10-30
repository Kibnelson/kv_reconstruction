# Standalone warp-prefix local validation

**Software checks completed. No GPU execution or GPU performance measurement.**

Twenty CPU tests passed; three actual-CUDA tests were skipped. Python
compilation and shell syntax checks passed. The previous completed A4500
report and unchanged source receipts were verified.

Ten offline SM86 production/probe variants compiled and passed exact
identity-lane layout checks: uniform/exact first, middle, last and single-layer
decoders, changing-mask prefix probe and original-state refill probe.
These are generic compiler specializations. The exact runtime-specialized
production variants must also pass cold inspection on the A4500 before use.

The CPU algebra checks cover 131,072 half-warp pattern combinations and
1,568 structured/random full 128-channel masks. This is not exhaustive over
all 128-bit masks and is not a CUDA execution proof. AST checks constrain the
new state/word/layer kernels to the intended prefix-helper substitutions.

The compiled uniform-middle reference has 56 static warp-shuffle sites; the
candidate has two, plus eight ballots and 16 population counts. These are
instruction counts, not GPU latency, throughput or a promised speedup.

The runner requires three real CUDA gates, then 22 workers, 44 document rows,
22 fresh receivers and eight candidate/reference document pairs. All 16 own
rows must retain the original frozen complete packet SHA. Reference and
candidate retain the same 45,568 packed-inverse shared bytes; the new prefix
adds no GPU tables. Baseline sources are unchanged.

The Rank256 100 Gb/s precompressed target is approximately 16.3 ms versus the
previous measured 21.184 ms. The new run recomputes targets from its own
measured baseline and packet sizes. Target attainment is unmeasured locally.
Links are calculated scenarios; this is T with full BF16 restoration, not
measured NIC/SSD performance or resident compressed attention P.

Run on the NUC branch `experiment/standalone_prefix_v1` with
`bash scripts/run_standalone_prefix_a4500.sh`. Raw test/compile logs,
reproducible compiler script, hashes and runtime details are included here.
