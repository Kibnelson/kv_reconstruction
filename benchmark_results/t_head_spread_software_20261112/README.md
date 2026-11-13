# T head-aware spread: local validation

These are **software and compilation checks only**. No GPU was attached and no
pretrained Llama KV improvement was measured. NUC CUDA validation is **pending**.
The experiment is based on T commit `d95fd10a1fae447d47c4ac6abaf77b90af1b7952`.

- Eight unittest methods passed (two invocations, five plus three methods).
- The interpreter checked six policies across dense and head-by-head prediction,
  including exact BF16 special patterns, a separate NumPy prior oracle, original
  stream-byte parity for global/zero-blend controls, compact packets, frozen
  state integrity and damaged-packet rejection.
- Twelve native CUDA specializations compiled to `sm_86` cubins without launching
  a kernel. This does not establish device execution or speed.
- Two synthetic capture-style driver runs passed: two layers with both prediction
  schedules (12 cases), then a one-layer reporting/gating smoke (6 cases). Every
  case passed a fresh decoder process whose data inputs were only frozen state
  and its compact packet. All 18 decoded hashes matched their original words.
- DEV choices stayed frozen, original/matched-global packet bytes matched,
  shared-state amortization fields were exercised, and CPU runs could not pass
  the GPU experiment gate. Six original T source blobs passed their pinned hashes.

`validation.json` contains the functional proofs and source identities.
`compile_sm86.json` records native compile results; `tests.txt` retains the test
logs. Synthetic rates and interpreter timings are deliberately not used as
compression or performance evidence.

See [the experiment runbook](../../docs/t_head_spread_experiment.md) for the
isolated NUC worktree, smoke test and full held-out test. No adoption or merge
is authorized by these local checks.
