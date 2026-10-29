# Standalone lossless compute experiment

All 28 layers use our predictive/rANS codec. SplitZip and ExANS remain
separate unchanged controls, with no baseline fallback inside a candidate.
The current binary CDF search is retained. This iteration preserves original
coefficients, probability arithmetic and complete Dense/Rank256 packets.

## Result motivating this experiment

The full inverse-LUT experiment completed at
`e087a4c9383f97b8f9ed26444c3cdbdc87c5d709`, report
`benchmark_results/standalone_lut_a4500_20261029_143932_3085067`.
All 44 original-word checks, 22 fresh receivers, two actual CUDA gates and
eight exact-packet groups passed. Its 142 packaged files verify against their
receipts; frozen sources, predictors and native libraries remain unchanged.

The complete lookup table regressed 8K isolated decode: Dense 28.807 to
32.132 ms, Rank256 21.539 to 23.894 ms. It also added 1.25 MiB of shared
state without changing packet bytes. It is not promoted into this experiment.

Paired 8K text profiles place the slowdown inside the replacement GPU decoder
kernel: Dense accumulated kernel work increased from 22.330 to 26.383 ms per
request, Rank from 21.265 to 23.574 ms. CPU dispatch and directory staging
remain roughly unchanged. These are nonadditive diagnostics; hardware cache
or occupancy counters were not collected, so cache stalls are not established.

The original standalone packets remain about 499.574 MiB Dense and
521.154 MiB Rank256, compared with 674.807 MiB SplitZip and 609.129 MiB
compact ExANS on the two 8K development documents. Original Rank decode is
21.539 ms versus 23.554 ms for compact ExANS. SplitZip still decodes faster.
At 50 Gb/s, calculated precompressed Rank cost is 108.974 ms versus
116.589 ms for SplitZip; fresh encoding costs must be accounted separately.

## Separate compute arms

### Decoder matrix products

The original layer decoder uses cuBLAS for INT8-feature times INT8-coefficient
products, yielding INT32 predictions before unchanged FP32 scaling and RoPE.
The frozen source already provides an exact Triton integer matrix kernel.
Test two fixed tiles: `(128,128,64,4)` and `(64,128,64,4)`, interpreted as
M, N, reduction tile sizes and warp count.

Only the decoder controller changes backend. An owned shallow controller
shares the original immutable coefficients and CDF tensors, receives the same
workspace, and uses the checked Triton product. The original encoder retains
cuBLAS, so its packet `product='cublas'` identity remains accurate. Candidate
execution receipts explicitly identify the decoder override and tile sizes.

All timed geometries divide both tiles: decode rows are 256 at 512 tokens and
4096 at 8192 tokens; output width is 2048; reduction width is 2048 for Dense
and 256 for Rank256's second product. The first Rank product and requantization
remain unchanged. Unsupported geometry is rejected before measurement, never
silently repaired with cuBLAS. Integer products cannot overflow: the worst
Dense bound is `2048 * 128 * 128 = 33,554,432`, below the INT32 limit.

Smaller matrix tiles may coexist better with another token chunk's entropy
kernel, but may reload coefficients more often. This is a measured scheduling
hypothesis, not a predicted speedup. The existing integer kernel is reused
without editing its source or the entropy decoder.

### Dense encoder model preparation

The original Dense encoder constructs a `FusedOneStep` helper on every call,
including rebuilding CDF tables that its `model()` operation never reads.
The latest reference profile attributes 59.673 ms across three profiled
encode/decode cycles to `slice_tables`. That is diagnostic CPU attribution,
not a promised wall-time saving.

The separate cache arm constructs one owned helper at cold initialization and
reuses its exact `model()` method. It shares the existing native CDF tensors,
so no duplicate persistent GPU tables are retained. Features, priors and
temporary outputs are still freshly generated from each actual input.
This arm keeps the reference cuBLAS decoder. It is Dense-only and is not
combined with either decoder-product arm in this experiment.

All candidates add zero persistent GPU tensor bytes, retain the original
57,688-byte coder tables, and install no full inverse LUT.

## Fixed matched evaluation

There are ten new jobs: two decoder tiles for Dense and Rank256 at both
512/8192 tokens, plus the cached Dense model at both lengths. Sixteen frozen
reference/control jobs and two unchanged compact-ExANS jobs bring the total
to 28 workers, 56 rows and 28 independent receivers. Each job processes the
same original text/code documents. All candidates require exact original
serialized packet SHA, bytes, header, state and original-word hashes.

Reference and candidate keep graph execution at 512 tokens and direct
borrowed-input decoding with fresh output at 8192. Dispatch is installed
before graph capture. Fresh receivers receive only fixed state and compressed
packets, alternate 0/1/0, retain returned outputs and check packet immutability.
No source KV, predictor outputs, masks or baseline codebooks are supplied.

Two actual-CUDA test methods gate benchmarking: integer product parity for
both tiles, K256/K2048, signed extremes and offset views; then production
Dense/Rank graph/direct packet replay and Dense warm-cache checks. Local
interpreter checks must execute the Triton product itself, since the original
CPU `product='triton'` flag otherwise falls back to a software product.

Full encode/decode API calls retain the unchanged author timing body:
10 warmups, 50 calls, five repeats. Separate isolated measurements use two
warmups and five samples with synchronized wall time, CUDA intervals and
Torch allocation peaks. Cold initialization/capture, file export, fitting
and upload are separate. Candidate and reference profiles are warmed and run
after ordinary timing; summed kernel work and stage diagnostics are nonadditive.

Profile the selected backend explicitly: decoder-only product arms should
execute 54 Triton second-product kernels per 8K decode, with no corresponding
sender backend change. Cache-only Dense should remove repeated helper table
construction while preserving the original predictor GPU kernels.

## Objective and run

Prioritize lower and middle link rates while preserving the smaller packets.
Report calculated 1, 5, 10, 25, 50, 75, 100, 200, 300 and 400 Gb/s costs,
separating precompressed `wire + decode` from fresh
`encode + wire + decode`. These are not measured NIC/SSD or serving results.
Do not select a deployment policy from the reused TEST documents.

On the A4500 NUC, switch to `experiment/standalone_compute_v1` and run
`bash scripts/run_standalone_compute_a4500.sh` in `~/kv_reconstruction`.
Reports publish automatically; failures retain their original exit code.
Packets and complete profiler traces remain in the run directory.

The model remains pinned Llama-3.2-3B, original BF16, all 28 layers. This is
lossless transfer/storage T with full BF16 reconstruction, not compressed
resident attention P. Lossy methods remain separate. Two repeatedly reused
development documents do not establish independent generalization.
