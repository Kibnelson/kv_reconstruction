# Lossless decoder and mapped I/O experiment v2

The goal is to improve complete lossless transfer latency against the measured
lossless baselines, preserving every original BF16 word and the existing packet
bytes. This experiment is on `experiment/lossless_decode_v2`, based on
`ad305c56bf6446b39cc5698f1b7f94d3d58ebbf0`. Existing source tracks, baseline
implementations, fitted state, TRAIN LUTs, packets and reports remain unchanged.
Only new files are added.

## Evidence from the previous A4500 run

The completed previous run passed four actual-CUDA tests, 68 exact result rows,
34 independent receivers and eight packet-equivalence groups. Its 213 packaged
files, policy hashes, source receipts and shared-state receipts have been checked.
The four execution arms established that selective preparation helps encoding,
while bundle copies limit the larger-request gains:

| 8K method | Complete MiB | Isolated encode ms | Isolated decode ms |
|---|---:|---:|---:|
| SplitZip, after control | 674.807 | 10.339 | 3.383 |
| Existing Rank256 | 521.154 | 41.945 | 21.551 |
| k2 reference | 664.004 | 30.527 | 10.446 |
| k2 selective + bundle | 663.993 | 15.547 | 8.871 |
| k14 reference | 595.842 | 37.202 | 16.270 |
| k14 selective + bundle | 595.837 | 28.871 | 15.228 |

For k14, separate profiles recorded about 2.36 ms of GPU gather work and 2.01 ms
of scatter work per full request. These are nonadditive accumulated GPU times,
not a wall-latency decomposition. The bundle materializes 448 MiB of fallback
words; k2 materializes 832 MiB. Measured incremental Torch decode peaks rose
from about 928 MiB to 1,344/1,728 MiB for k14/k2. These include output and scratch.

The original decoder also always produces features and surprise statistics.
Those outputs are unnecessary if the immediately following layer uses SplitZip,
or if the current layer is last. For k2 layers `[1,27]`, both outputs are unused.
The rANS state and read position still need to carry between those layers.

## Four execution arms, one packet format

| Arm | Fallback I/O | Decoder context |
|---|---|---|
| Reference | Previous bundle gather, decode temporary and scatter | Original outputs |
| Mapped | Direct source reads and final-output writes | Original outputs |
| Context | Previous bundle path | Only outputs consumed by the next layer |
| Both | Direct mapped I/O | Only consumed context |

All arms retain the previously measured selective encoder preparation. Every
complete packet must match the prior `fast_both` packet exactly for the same
document, mask and length. All rANS words, offsets, relative positions, SplitZip
codes, escapes, framing and state identities are preserved.

Mapped I/O uses new local kernels derived from the pinned author SplitZip
implementation at `42c743118a29db3270ae6a921a4a4eecb20cad04`. Encoding reads each
logical fallback position from its original layer. Decoding stores its exact
sign/exponent/mantissa reconstruction directly into the final layer. Escape
patches use the same mapping and finish before any rANS feature read. Counts,
prefix sums, field order and bit arithmetic retain the author definitions.
The author's MIT copyright and license are retained alongside the derived code.
These are labelled local integration kernels, not unchanged official baselines.

The context-aware kernel uses the original prediction, CDF and rANS arithmetic.
It produces a decoded layer's feature and surprise outputs only when the next
adjacent layer also uses rANS. It still stores coder state/read positions until
the last coded layer, even across SplitZip gaps. The original decoder, author
sources and prior adapters remain untouched.

## Frozen protocol and checks

Use all 28 original BF16 layers of `unsloth/Llama-3.2-3B`, revision
`d4446454d87d51aa42e1fb174f25acc5f8762331`, with the same fitted predictor and
TRAIN SplitZip LUTs. Reuse the prior TABLES k2/k14 masks at 512 and 8,192 tokens
on both original documents. No refit, model download, environment installation,
new mask search or TEST winner selection is performed. These repeatedly used
documents are development benchmarks, not independent generalization evidence.

The run begins with three actual-CUDA tests covering mapped native field parity,
selective-context decoding of production packets, and integrated byte/word/output
ownership checks. A skipped CUDA test does not pass. CPU interpreter tests are
semantic software checks and provide no GPU performance result.

Fresh receivers are given only compressed packets, shared state and expected
identity assertions. They restore real serialized packets, alternate documents
0/1/0, retain earlier returned outputs and verify original-word hashes and
compressed-input immutability. Source, state, native runtime DSO and policy
receipts are checked before and after execution. Old measurement files and the
84 frozen core files must retain their recorded hashes.

## Measurements and promotion criteria

Raw, official SplitZip and fixed ExANS retain their original workers and pinned
environments and bracket the new arms. Dense, Rank256 and the complete compact
ExANS carrier remain paired controls. The SplitZip control keeps its original
Torch 2.9/Triton 3.5 environment; new integration uses the existing main
Torch 2.11/Triton 3.6 environment. This distinction remains in the receipts.

Measure full encode and full decode using the unchanged author timing body:
10 warmups, 50 calls per sample and five repeats. Separate isolated timings use
two warmups and five samples. Capture cold initialization, actual complete
serialized bytes, shared state, Torch output/scratch peaks and warmed diagnostic
profiles. Export, upload, disk and offline fitting are outside GPU codec timing.
Native allocations outside Torch remain unmeasured.

Profiles should confirm the gather/scatter kernels and large fallback temporary
are removed in mapped arms, and that dead feature/surprise output work disappears
in context arms. These are hypotheses for measured validation; semantic parity
alone does not establish speed or memory improvement.

Calculated serial link tables retain all fixed arms at 1, 5, 10, 25, 50, 75,
100, 200, 300 and 400 Gb/s, for precompressed wire-plus-decode and fresh
encode-plus-wire-plus-decode. The summary shows 8K isolated comparisons; JSON
retains both lengths and timing bases. No overlap or measured NIC/SSD claim is
made. With the previous k14 packet, matching SplitZip requires decode near
10.01 ms at 100 Gb/s and 5.04 ms at 400 Gb/s. Copy removal alone is not assumed
to meet those targets. Promote only from complete measured costs, rather than
combining encoder and decoder times from incompatible representations.

## Run on the NUC

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/lossless_decode_v2
git pull --ff-only origin experiment/lossless_decode_v2
bash scripts/run_lossless_decode_a4500.sh
```

Small reports and failure logs publish to this isolated branch. Full packets and
profiler traces stay in the NUC run directory. Local preparation does not measure
GPU performance. This is lossless transfer T; resident P attention, SSD/network
and serving throughput require separate experiments.

## Later scheduling candidate

If the mapped/context paths remain limited by fallback readiness, test a
separate schedule that restores only critical predecessor layers first: `[0,26]`
for k2 or `[0,3,15]` for k14. Other fallback layers could decode concurrently
while the ordered rANS chain proceeds. Critical escape patches must complete
before rANS begins, bulk writes must join before returning, and all stream
lifetimes must be recorded. This schedule is not implemented or measured here;
GPU contention may erase overlap gains.
