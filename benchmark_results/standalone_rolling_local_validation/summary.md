# Standalone rolling encoder local validation

Status: **completed**. GPU performance measured: **False**.

18 CPU checks passed: 14 protocol tests, one controller contract test and
three explicit Triton interpreter tests. Dense four-layer and Rank256
three-layer packets match the original complete serialized bytes and restore
all BF16 words exactly. Native output/layer-count interfaces and the
single-layer anchor also pass. The interpreter cohort took 295.647 seconds;
this is test duration, not codec performance. Expected nonfinite-operation
warnings are retained in the log.

Four Python files parse, Bash syntax passes, and the staged patch is checked
for whitespace. Independent source and runner reviews found no blocker.
Only new files are added; existing tracked sources and baseline code remain
unchanged. Two actual CUDA test methods must pass on the NUC before timing.

The NUC experiment includes 24 workers, 48 rows, 24 fresh receivers and
12 candidate/document packet comparisons. It includes a current cached-Dense
control, both original standalone references and unchanged SplitZip/ExANS
controls. No GPU speedup is claimed by this local validation.
