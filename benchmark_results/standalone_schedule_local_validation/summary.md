# Standalone schedule local validation

Status: **completed**. GPU performance measured: **False**.

19 CPU checks passed: 13 runner protocol tests, four controller contract
tests and two explicit Triton interpreter tests. The feature-reuse encoder
preserves complete Dense packet bytes and exact decoded words with one
feature pointer. Alternative Dense chunk partitions restore the existing
packet exactly. Header identity, workspace partitioning, Rank geometry
rejection, caller-stream ownership and arithmetic-source parity also pass.

The interpreter cohort took 94.212 seconds; this is test duration, not codec
performance. Expected nonfinite-operation warnings are retained in its log.
Two actual CUDA methods remain required before NUC timing: exhaustive BF16
pattern feature replay and production Dense/Rank packet/lifetime coverage.

Four new Python files parse, Bash syntax passes and staged whitespace is
checked. Independent source and runner reviews found no blocker. Only new
files are added; existing tracked sources and baseline code remain unchanged.

The NUC experiment has 28 workers, 56 rows and 28 fresh receivers, including
current rolling references and unchanged baseline controls. All 28 own rows
must preserve original packet SHA; 20 candidate/document groups are paired.
Eight candidate/reference profiles check actual dispatch after warm timing.

Feature scratch is 432 to 16 MiB at 8K. Full API peaks may fall by much less
because native arena plus compact output can set the next peak. All memory
and speed hypotheses require the new GPU measurement.
