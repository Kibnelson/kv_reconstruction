# T speed comparison while retaining the measured compressed bytes

This experiment stays on `experiment/t-head-aware-spread-v1`. The original T
branch and main remain untouched. It adds a dense encoder forward pass and a
benchmark; no original codec or previous run file is edited.

## What is being compared

The completed A4500 run at `20261113T151348Z` measured two useful points:

- Dense learned: 8.947437 bpv, 44.0785% saved, approximately 61.8 ms encode and
  35.7 ms decode per 8,192-token, all-layer window.
- Head-by-head learned: 8.229093 bpv, **48.5682% saved**, approximately 110.6 ms
  encode and 194.1 ms decode.

The unchanged-probability dense control already encoded in approximately
61.8 ms, versus 84.0 ms for original T. This motivates isolating that encoder
improvement while retaining the original entropy encoder and decoder methods.
These earlier times are warm CUDA event measurements, not network timings.

| Benchmark name | Probability policy | Encoder | Decoder |
|---|---|---|---|
| `dense_original` | Frozen dense global | Original fused forward pass | Original |
| `dense_fast` | Same frozen dense global | cuBLAS forward statistics | Original, unchanged method |
| `dense_learned` | Frozen dense learned, alpha 0.75 | Existing experimental forward pass | Existing learned-width decoder |
| `hbh_original` | Frozen head-by-head global | Original grouped forward pass | Original grouped decoder |
| `hbh_learned` | Frozen head-by-head learned, alpha 0.75 | Existing experimental forward pass | Existing learned-width grouped decoder |

Both ordinary execution (`eager`) and CUDA graph replay (`graph`) are timed.
Graphs test whether reducing Python/driver launch overhead makes the grouped
version faster, while preserving its 48.5682% saving. A graph result is accepted
only if it reconstructs deliberately changed input and then the restored input.
Capture/setup time and graph memory are recorded separately from warm replay.

`FastDenseEncode` changes only `model()`. It uses the original feature,
surprise and global-prior kernels with exact int8-to-int32 cuBLAS products.
It inherits `LayerwiseOneStep.encode` and `decode` without overriding either.
No new rANS kernel, numerical approximation, predictor or quantizer is added.

## Exact matching and measurement

The completed run's **state files remain on the NUC**. This comparison loads
them, including the existing learned maps. It does not fit on TRAIN or DEV,
recapture KV, retune alpha, or change tile sizes. It verifies the prior run's
source hashes, frozen selection, input hashes and state identities before use.

Every new packet must have the **same complete SHA-256 and byte count** as its
reference packet. `dense_fast` must match `dense_original`; `hbh_learned` must
match the completed learned head-by-head packet. Thus a full four-window run
retains the same 48.5681708% saving by construction. A one-window smoke may
have a different aggregate rate because its data subset is different.

CUDA preflight checks the new dense forward features, global priors, original
stream bytes and original decoder over all 65,536 BF16 patterns. During the
model comparison, every execution is checked word for word. Each method/window
also gets a fresh-process decoder whose only data inputs are the existing
frozen state and the newly produced compact packet.

The full test uses the same 28-layer Llama-3.2-3B, the same four WikiText-2
8,192-token windows and the same complete wire accounting. It measures:

- Encode: full forward statistics and entropy encoding, with resident exact KV.
- Decode: complete reconstruction, with resident streams and preallocated output
  and scratch.
- Pair: a directly timed sequential encode plus decode on the same GPU. Graph
  pair uses two replay launches with the encoder's captured starts as input to
  the decoder graph.
- CUDA event time, synchronized wall time, raw GB/s, warm allocation peaks,
  static arena/output/scratch bytes, state bytes and capture memory/setup.
- Packet packing/D2H and compact-packet expansion/H2D as separate measurements;
  fresh-worker cold state/setup and decode time are retained separately.

Three rounds rotate implementation and operation order. Each timed operation
has three warmups and ten samples. The aggregate takes the median round value
for each window, then the mean across windows. Per-sample timing and the range
of round means remain in `results.json`; the packet size is counted once per
window, rather than multiplied by rounds or timing repetitions.

The report also shows **modeled** serial latency for 3, 10, 25, 50, 100, 200
and 400 Gbps: measured codec pair time plus complete packet bytes divided by
link bandwidth. That model assumes warm resident state and no overlap. It does
not measure network, SSD, admission queues, transfer contention or protocol
overhead. The separately measured host packet path is not included in that
ideal link table. This is a codec comparison, not a serving-throughput claim.

CUDA graphs use static tensor addresses, explicit side-stream warmup, retained
input/output references and separate private pools, following the
[PyTorch 2.11 graph constraints](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).
One side stream is reused per device to avoid accumulating cuBLAS workspace
caches across repeated captures; graph-private memory pools remain separate.
Each input shape requires its own captured graph. Unsupported capture/replay is
reported as an error, never relabeled as a graph speed measurement.

## Run on the NUC

Use the existing **test worktree**, without changing the original checkout:

```bash
cd ~/kv_t_head_spread_v1
git pull --ff-only origin experiment/t-head-aware-spread-v1
unset TRITON_INTERPRET
KV_HOME=~/kv_a5000 SMOKE=1 bash scripts/run_t_speed_nuc.sh
```

The smoke keeps all 28 layers and 8,192 tokens, but uses only the first held-out
window and fewer timing rounds. It validates the new forward pass and graph
capture before the longer comparison. After it passes:

```bash
KV_HOME=~/kv_a5000 bash scripts/run_t_speed_nuc.sh
```

The existing venv is activated; no packages are installed or upgraded. The
defaults read `~/kv_a5000/runs/t_head_spread_v1_20261113T151348Z` and write a
fresh `~/kv_a5000/runs/t_frozen_speed_v1_*` directory. Existing captures and state
files are only read. No model download or training is needed.

Optional `REFERENCE_RUN`, `CAPTURES`, `T_SPEED_VENV`, `T_SPEED_OUTPUT`, `VARIANTS`
and `EXECUTIONS` environment variables override the corresponding paths or
methods. A moved captures folder must still match all reference input hashes.
`EXECUTIONS=eager` deliberately requests an eager-only run; it does not certify
graph performance. Do not substitute a smoke reference for the completed run.

Send the new `share.zip` back for review. It includes the small reports and
preflight checks; no original KV, weights, frozen state or expanded arenas are
included. Temporary compact packets are removed only after independent decode
proof passes. No run script commits, pushes or merges anything.

Candidate flags require an actual GPU, all reference windows, at least three
rounds, unchanged packets and at least a 5% relevant speed improvement. The
dense candidate must also reduce pair time; the grouped graph candidate must
pass the changed-input test. These are experiment review gates, not adoption.
The full original compression/speed results remain visible even when a gate
fails. An absolute serving SLO still needs a serving benchmark.

## Software checks

```bash
python -m unittest -v kv_compare.test_t_speed
```

The suite runs a small explicit CPU Triton interpreter subprocess to verify
forward-prior parity, exact streams and decoding with the original decoder.
It also checks unchanged coder methods, CPU rejection for graph replay,
bandwidth units, round accounting and incomplete-result/GPU review gates.
Local software validation cannot establish A4500 speed or CUDA graph support.
