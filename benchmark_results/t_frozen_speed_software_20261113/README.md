# Frozen-byte T speed comparison: software validation

Based on experiment commit `2b46bf92130f4eb725e99b8c069fe194f0e469e9`.
No GPU was attached. CUDA graph support and all new A4500 speed results remain
**pending NUC validation**. The earlier GPU result remains preserved.

Seven unittest methods passed. An explicit Triton interpreter subprocess
compared the new dense forward features and global priors to original T,
verified identical complete stream arenas/starts, and decoded with the
original decoder. It also included BF16 special patterns. The entropy encoder
and decoder methods are inherited unchanged.

The capture-style software benchmark completed for `dense_original`,
`dense_fast` and `hbh_learned` on a small two-layer synthetic fixture. All three
packets matched their earlier frozen-reference SHA-256 exactly. Three fresh
state-plus-packet decoder processes reconstructed every word. CPU results
could not pass either GPU review gate. Interpreter timing is not performance
evidence and was omitted from this validation summary.

Additional checks covered graph CPU rejection, bandwidth units, round-size
accounting, changed-packet rejection and incomplete graph result handling.
The CUDA capture wrapper itself can only be validated on an actual GPU; the
NUC smoke checks changed-input replay, restoration, exact words and exact
reference packet bytes before full measurement.

See [the comparison runbook](../../docs/t_frozen_speed_comparison.md).
