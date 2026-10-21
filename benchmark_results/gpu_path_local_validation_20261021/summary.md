# GPU path local validation; A4500 batch pending

The final local suite ran 58 tests: 48 passed, 10 skipped in 89.897 seconds.
Eight path-specific CUDA tests, the existing P0 CUDA test and one optional
fresh-capture integration test were skipped. No CUDA execution
or GPU timing occurred on this development host.

Twenty new attention/T1 kernel variants and twelve P1 regression variants compile
for SM86 with Triton 3.6.0. Cubin hashes are included separately. CPU interpreter
tests execute the actual Triton source and compare independent byte, probability,
word and attention oracles. They exercise per-layer attention, GQA, raw tails,
nonfinite/signed-zero word preservation, lane framing, causal parameter updates,
external byte round trips, rejection, and the batch controller's allocation retry.

Source and frozen-state hashes are verified. Reusing saved tiny A/B calibration
was checked without a new model forward. The generated standalone artifact
matches the current source locally; the A4500 helper regenerates and publishes
it without sending its large contents through chat.

Actual CUDA correctness, T1 rate, attention/codec costs, memory peaks and
fresh-process GPU decode remain unmeasured until the hardware batch runs.
Native cache-writer/generation, serving concurrency/capacity, cold SSD/network
and new hint/coset research arms remain separate integration/experiment work.

See [the batch guide](../../docs/gpu_path_batch.md) for the one-command A4500 run.
