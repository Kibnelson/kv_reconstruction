# Review of cancelled PD job 30196

Evidence commit: `90012974da8fe60ae67f20d4cb4b1f47760cb835`.
Source pin: `90112640bc59bac6ea1a82f79b8c53f2b22aa996`.
Branch: `evidence/experiment/t-pd-matched-campaign-v1/90112640bc59/smoke/job30196`.

## What the evidence establishes

- The run was cancelled. Cleanup job 30204 completed and verification job 30205
  recorded the successful cleanup accounting receipt. Existing review/publication
  stages are closed out; that is not a successful serving benchmark.
- Four RTX A5000s, 24,564 MiB each, driver 595.71.05 were recorded. GPU 0 reaches
  the other GPUs through NODE topology; GPUs 1, 2 and 3 are PIX-connected.
- The saved `pd_selftest.json` reports exact round trips for all four added
  methods at both 8,192 and 3,000 tokens, under Torch 2.13.0+cu130. These are
  readiness inputs from the setup capture, not held-out serving results.
- The codec environment's node bandwidth child aborted with exit -6. The parent
  recorded `bandwidth_error` but the old survey returned success. The specific
  earlier CUDA operation that caused the fault is not established by its
  asynchronous synchronization traceback.
- Compiled/default vLLM startup failed when FlashInfer attempted to run Ninja.
  LMCache had an additional missing vLLM adapter module. Eager engine logs show
  successful server startup, but no completed sweep/results were published.
- `run.json` remained `started`: the controller did not finalize it before
  cancellation. The comparison review correctly rejects coverage and hardware
  readiness rather than treating that label as successful completion.

## Missing evidence and next step

The published tree contains no `setup/` records, including the dedicated
compiled-BF16/independent-receiver report and package freezes. Cleanup salvaged
engine logs but did not call the normal setup collector before deleting the
runtime. The new collection fix preserves these records on future cancellations;
it cannot reconstruct the missing job-30196 files from already-deleted state.

The first new test isolates small GPU copy routes in fresh processes with the
codec environment's Torch 2.11.0+cu128. It uses our own minimal environment and
cache. No serving speed claim, recalibration, codec numerical change or automatic
merge is made. Larger buffers, full topology, engine JIT startup and LMCache
adapter compatibility still require subsequent GPU checks.

Sources, relative to the evidence commit:

- `benchmark_results/pd_card/pd_smoke_tmatched_20261115T112147Z_594875/run.json`
- `.../verify_cleanup.json`
- `.../node_specs/specs.json` and `.../node_specs/summary.md`
- `.../review/matched_comparison.json`
- `.../matrix/llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261115_114347_4106208_smoke_partial/pd_selftest.json`
- The first matrix prefill log and LMCache engine log, plus the salvaged eager
  prefill log and `env.json`.
