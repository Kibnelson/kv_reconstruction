# A4500 prediction results and exact entropy scheduling comparison

The completed input-chunk run is
`benchmark_results/gpu_path_a4500_20261021_113837_508750`, commit `e3e296d`.
All nine actual CUDA tests, 24 attention cases and 60 transfer implementation
cases passed. The latter cover 20 source/arm cases at three reduction chunks.
All 40 alternatives produced exactly the same complete T1 bytes as chunk64;
every case passed causal reconstruction, external restore and fresh-process
GPU decoding. There were no raw fallback cases.

## Measured result

Qwen B remains at 8.853500 complete bpv, a **44.67% BF16 reduction**. Its
entropy payload is 8.828415 bpv. Qwen A remains at 9.579681 complete bpv;
SmolLM A remains at 9.441417. Shared state is additional. These are the same
exploratory sampled pages, with four Qwen layers and 30 SmolLM layers.

The following are means of per-page warm synchronized wall medians for Qwen
B. Peak columns are the maximum measured incremental Torch allocation for
one call across those pages, not resident-pool or whole-GPU memory.

| Prediction input chunk | KV to T1 ms | T1 to KV ms | Encode peak MiB | Decode peak MiB |
|---|---:|---:|---:|---:|
| 64 columns | 465.97 | 332.74 | 12.43 | 4.54 |
| 512 columns | 314.83 | 209.02 | 12.43 | 7.53 |
| All columns | 294.16 | 206.38 | 39.13 | 35.53 |

Median paired speedups versus chunk64 are 1.481x/1.591x for 512-column
encode/decode, and 1.583x/1.612x for all-column encode/decode. These numbers
are medians of ratios; they need not equal ratios of the means above.

For Qwen A, mean encode/decode times are 393.57/259.40 ms at chunk64,
241.05/181.55 ms at chunk512 and 220.62/178.47 ms at all columns. Its paired
all-column speedups are 1.784x/1.450x. SmolLM gains are smaller: approximately
1.15–1.17x encode and 1.09x decode. A universal default is not established.

512 columns is the next controlled baseline: compared with all columns it
spends about 20.7 ms more encoding and 2.65 ms more decoding for Qwen B,
while reducing the maximum incremental decode allocation by 28 MiB. This
temporary-memory choice does not change the 32-channel prediction groups,
T1 bytes or the P1 resident layout. The API still defaults to the original
scalar/64 implementation; the experiment selects 512 explicitly.

Qwen B's directly measured file-cycle mean fell from 821.48 ms at chunk64
to 572.34 ms at chunk512 and 539.58 ms at all columns. It remains expensive
for a 524,288-byte original chunk. The serial no-overlap link diagnostic is
approximately 0.45 MB/s break-even at chunk512 and is not a network result.
No serving or movement-speedup claim follows from this run.

## What the corrected profiles identify

The profiles now exclude CUDA annotation ranges and execute in separate
packet-only child processes. On a selected Qwen B chunk512 profile, the
encoder's entropy kernel accounts for about 147.1 ms of device activity and
the decoder's entropy kernel about 141.6 ms. At all-column reduction those
entropy costs remain approximately 147.1/141.8 ms. Prediction chunking cannot
remove that work. Profiled durations include instrumentation and are separate
from normal wall medians; activity durations can overlap.

The current scalar encoder computes three CDF boundary masses per bit while
advancing rANS sequentially. Sixteen bits per BF16 word therefore invoke 48
boundary evaluations. The scalar decoder repeats the same three-boundary
calculation at each bit, although two endpoints are already known from the
previous bit's interval.

## Implemented next control: `scalar` versus `cached`

Both backends retain the exact same fixed-point predictor, width context,
logistic table, BF16 rounding intervals, 12-bit counts, rank mapping, rANS
lane order, restarts and envelope. They have the same state identity and
must produce identical complete packet bytes. This is an implementation
optimization, not a new probability model or format.

**Encoding:** because the encoder has the original symbols, it can calculate
the 16 conditional frequencies of each word in parallel. A new Triton kernel
writes the same counts to a reusable `[tokens,group,16]` UINT16 cache. The
rANS kernel reads those counts in the original reverse encoding order.
At 32 tokens and 32 channels the cache is 32,768 bytes and is reused for each
group; it is not a full-chunk array. It is temporary workspace and is charged
in measured peaks. One additional preparation launch per group is also paid.
The number of boundary evaluations is unchanged; they are scheduled in
parallel rather than computed along the rANS state dependency chain.

**Decoding:** the decoder cannot use original symbols. It starts with the
same low/high masses and computes a midpoint for each decoded bit. The bit
selects which endpoint becomes the midpoint for the next step. Reusing those
integer masses reduces evaluations from 48 to 18 per word: two initial
endpoints and 16 midpoints. The arithmetic, count rounding/clamping, bit
order, renormalization and bounds checks are unchanged. No known-source KV
or transmitted probability cache is available to the decoder.

The extra encoder cache does not enter the stored packet or shared state.
Raw fallback retains the existing container policy. Peak accounting records
the probability-cache capacity separately from the lane byte workspace.
The reference scalar path remains available, and the receiver can decode a
packet with either implementation.

CPU interpreter checks compare all cached counts with an independent CPU
table on varied means/scales and all 16 planes, including a masked tail.
They check byte-identical streams, special BF16 words, cross-backend decoding,
corrupt-lane rejection and the complete pool/file/attention bridge. The CUDA
suite adds the same checks and exercises both backends over all 65,536 BF16
patterns. Offline SM86 compilation checks the new production kernels. Those
local checks are not new GPU execution or timing results.

## One command for the next matched batch

```bash
(
set -euo pipefail
cd ~/kv_reconstruction
git fetch origin
git switch experiment/gpu_kv_path
git pull --ff-only origin experiment/gpu_kv_path
bash scripts/run_gpu_path_a4500.sh --publish-results \
  --predictor-input-chunks 512 --entropy-backends scalar,cached --diagnostics
)
```

With the Qwen captures present this repeats 24 attention cases, ten CUDA
tests and 40 transfer implementation cases over the same 20 source/arm
cases. It holds the prediction input chunk at 512 for both backends. Every
candidate must match scalar parameters and complete packet hashes, reconstruct
words exactly, restore attention-readable P1 and pass its independent child
decode. CPU T0 runs once per source/arm. Paired speedups match dataset, page,
arm and prediction input chunk. The first requested input chunk is now the
reference, permitting this focused test without repeating the earlier
64/512/all sweep; that original command remains supported.

The helper rebuilds/publishes the standalone runner and allowlisted reports.
Raw captures, encoded KV and child inputs remain local. New cached-backend
CUDA timings and whole-workload parity remain pending. No speedup is inferred
from operation counts or compilation. Native generation/concurrency, real
network, cold SSD, paid hints and cosets remain separate experiments.
