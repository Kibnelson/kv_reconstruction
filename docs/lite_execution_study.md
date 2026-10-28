# Where Lite spends time, and the next controlled reductions

The current lossless B8 result is **254.770 ms decode**, **227.939 ms encode**,
and **44.166% saved** for the two 56 MiB Llama-3.2-3B requests. Official SplitZip
decodes in **0.523 ms**, saving **24.732%**, on the same original BF16 words.
The extra size reduction has value, but current decode cost exceeds the saved
transmission time at our primary effective link targets of 1–10 Gb/s.

At 5 Gb/s, B8 saves **18.259 ms** of transmission compared with SplitZip. It
therefore needs decode below **18.781 ms** to win a serial precompressed fetch.
That is another **13.57x** reduction from the current B8 decoder. This budget
excludes serialization, staging, SSD, queueing and overlap; it is a calculated
target, not a measured network result. Fresh compression must also pay encode.

## Evidence and its limits

The latest B8 comparison has ordinary synchronized wall samples, but no
instrumented B8 trace. The existing detailed profile is the earlier host-RoPE
**B64 lossless text** request. Its instrumented decode took 531.543 ms; its
operator/kernel durations cannot be subtracted from the later 254.770 ms B8
wall time to predict an optimization's speedup.

| Existing B64 diagnostic | Evidence | Implication |
|---|---|---|
| Entropy decoder | 448 `_decode_kernel` calls, 134.653 ms device duration | Symbol decoding is a substantial remaining target. |
| Matrix multiplication | 447 `aten::mm` calls, 7.112 ms device time | Faster GEMMs alone cannot close the overall gap. |
| Generic indexed writes | 924 `aten::_index_put_impl_` calls; 48.414 ms self device time | Examine work used to handle arbitrary index collisions. |
| Sorting for indexed writes | 924 single-tile radix-sort kernel calls | Our validated permutations already guarantee unique destinations. |
| CPU launch API | 44,256 `cudaLaunchKernel` calls; 106.440 ms self CPU time under profiling | Reduce small operations and dispatch, as well as GPU arithmetic. |
| Host synchronization | One `cudaStreamSynchronize` call in decode | The earlier scalar-readback problem is already removed. |

These rows overlap. The indexed-write operator includes kernels also named in
the trace, and CPU/device work can overlap. Do not add their durations into a
purported end-to-end breakdown. The new profiler reports raw CUDA kernel
activities separately from inclusive operators and user ranges. It also records
the union of kernel intervals and their first-to-last span, exposing idle gaps
without counting overlapping kernels twice. GPU copies/memsets are separate.

The request contains 28 layers x 16 causal groups = **448 sequential groups**.
Each typical group has 512 tokens x 128 columns. B8 increased independent
entropy programs per group from 8 to 64 and cut decode by 16.5%; it did not
remove the surrounding per-group work or shorten each 128-value rANS chain.

## Execution changes to measure now

1. **Unique-column stores.** Replace four assignment sites in the execution
   loop: the current group's features, recovered output, lossy encoder output,
   and the layer's feature permutation. Each uses columns from a permutation
   checked when fitted state is loaded. One Triton kernel performs exact copies
   without sorting, atomics or accumulation. The other statements in the loop
   remain identical; an AST comparison test enforces that narrow difference.
2. **Fused group RoPE.** Gather frequencies and partner channels, perform the
   original float64 multiply/multiply/multiply/add sequence, select keys versus
   values, and cast to float32 in one kernel. Reuse the same Torch-generated
   cosine/sine tables and disable FMA. Preserve signed zero when reversing sine.
   Attention scales other than 1 use the reference implementation for now.
3. **Combine the changes.** Measure whether benefits add or expose another
   bottleneck. Neither predictor coefficients, probabilities, entropy kernels,
   group dependencies, validation, packet representation nor fitted state change.

The experiment retimes raw BF16, official SplitZip, official ExANS and FP8 using
their previous worker requests and runtime policies. Lossless Lite uses B8,
with reference / stores / RoPE / both / reference in fresh processes. Lossy
Lite uses B32 for kappa 0.5 and 1, with reference / both / reference. This avoids
promoting the lossless B8 tuning to modes where it did not help.

Every Lite encode must match the previous **serialized packet bytes**, and
every decode must match the previous reconstruction. Independent receiver
processes load only fixed state and original packet bytes. Lossless words must
also match the original cache; lossy quality is reused only after identical
reconstructed words, and remains the earlier short continuation smoke check.

After all ordinary timings, four separate diagnostic processes profile encode
and decode for text: B8 reference/both at kappa 0, and B32 reference/both at
kappa 0.5. No timing worker runs a profiler. Results include the remaining
kernel costs, launch counts, before/after reference drift, complete sizes,
retimed baseline gaps, and decode budgets at 5 Gb/s.

## What follows from the new profile

| If the new trace shows… | Next experiment |
|---|---|
| Many small launches or large gaps between kernels | Fuse prediction preparation and post-decode statistics; consider CUDA graphs after charging capture, buffers and required input copies. Keep directory validation. |
| Entropy dominates after fusion | Tune stream length, mapping and decoder searches using complete packets. Charge extra rANS state and directories, and verify decoder-only causal replay. |
| Feature conversion/copies dominate | Keep a suitable contiguous predictor representation and fuse gathers/conversions, preserving BF16 product rounding. |
| The 448-group dependency chain dominates | Evaluate fewer/larger prediction groups as a separate compression-versus-latency design change with new fits and measured bytes. |

The team's bulk component throughput demonstrates useful potential, but uses
known predictions and much larger batches. It is not the throughput of this
causal receiver. Optimizing only wrapper work may be insufficient: the next
B8 trace determines how much entropy work must also be reduced.

## GPU command

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_lite_execution_a4500.sh
```

The default parent is the completed run
`runs/lite_baselines_a4500_20261028_070735_342774`. No fitting, model downloads,
baseline rebuilds or dependency installs are performed. The wrapper first runs
actual-GPU word/packet checks, then measurements and separate profiles, and
publishes the small report. Full traces remain local under the new run.

Resume an interruption with
`bash scripts/run_lite_execution_a4500.sh --resume RUN_DIR`. Completed jobs
are retained when the source, inputs, packets and recipe hashes still match.
No new speedup is claimed until this GPU experiment completes. CPU interpreter
checks and offline sm86 compilation provide correctness/build evidence only.

Evidence: commit `1843ee85f63cb5309e949c9d22cf39e88ec0e96f`,
`benchmark_results/lite_baselines_a4500_20261028_070735_342774_report_20261028_071217/results.json`,
and the earlier
`benchmark_results/lite_fast_a4500_20261028_060544_3215314_report_20261028_060828/results.json`.
The numerical study and source digests are in
`benchmark_results/lite_execution_study_20261028/study.json`.
