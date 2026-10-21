# Task 1: GPU pool pages and head-tile measurements

This task implements GPU KV-to-P packing and selected head-tile unpacking.
It does not implement T entropy coding, T-to-P conversion, compressed attention
or a serving allocator. The canonical lifecycle remains T in DRAM/SSD and in
transit, P for active GPU attention. The frozen fixture and P0 comparison here
are benchmark inputs/controls, not a different external-storage policy.

## P1 representation

Input is contiguous original BF16 bit patterns as UINT16
`[layers, tokens, combined K-then-V channels]`. The declared head dimension and
KV head count describe the source layout; the pool does not apply RoPE or
numerical conversion. Existing replay/capture inputs use their recorded
adjacent-pair key order. Native attention integration must preserve or explicitly
map that order before comparing outputs.

Each sector holds one token tile and one K/V head. Sector order is layer,
token page, then all K heads followed by all V heads. At 16 tokens and head
dimension 128, a reconstructed sector is 4 KiB. It has:

- A paid 32-byte directory entry: INT64 byte offset, aligned allocation size,
  mode and stored escape count.
- For packed mode, raw sign/mantissa bytes, LSB-first exponent offset codes
  in UINT32 words, then original escape exponents in token/channel order.
- For raw mode, the original UINT16 words. Raw is selected when aligned
  packed storage would not be smaller.
- Zero-padded incomplete token tiles, with the original valid token count
  supplied by the page layout. Padding is charged and never returned as a
  real token.

Widths and reference exponents remain shared per-layer/channel calibration.
P1 changes the original P0 layout and bit order; it is not P0-byte-compatible.
An independent byte-oriented NumPy oracle defines the P1 representation.
This prototype supports power-of-two head dimensions from 8 to 256 and
power-of-two token tiles up to 32, including the measured SmolLM/Qwen shapes.

## GPU operations and capacity

The encoder launches a sector count/mode kernel, a GPU prefix scan and an
offset-placement kernel, then a pack kernel. It writes into an explicitly
preallocated bounded arena. No KV values or allocation-size scalars are read
back to the CPU during `encode_into`. Shared state initialization and later
diagnostics may synchronize and are outside warm codec intervals.

One Triton program owns each sector. It computes ordered escape ranks with a
cooperative scan. Packed exponent words are initialized, synchronized within
the program, then filled using disjoint bit-field atomic OR operations.
Unpacking uses the sector's descriptor and local scans; it needs neither
another layer's KV nor an original-input tensor.

The default benchmark tests arena budgets of 80% and 100% of padded raw bytes.
Those budgets are declared controls, not a TEST-selected optimum. The entire
reservation and directory count as resident allocation. Payload bytes alone
are reported separately. Calibration and the GPU scan buffer also count, with
the latter identified as encoder workspace rather than packet payload.

If a sector does not fit, no payload is written for it and its mode is -1.
Tile reads return invalid status and zero output; the caller retains the
source. The benchmark checks this behavior. This is safe rejection, not a
successful compressed page or a CPU fallback. A production controller must
honor that status and handle allocation/backpressure before publication.

The current arena is one append-batch prototype. It does not reuse freed
sectors, implement concurrent descriptor publication or provide a general
serving allocator. Tile reads return an explicit GPU tile buffer. The next
task would fuse that load into attention so a full-cache buffer is unnecessary.

## Measurements

The runner records cold calibration/source upload, arena allocation and first
pack including JIT separately from warm measurements. Warm KV-to-P includes
counting, scan, offsets and packing. It measures:

- KV-to-P synchronized wall time and CUDA stream intervals.
- Full-sector unpack as a correctness/throughput control.
- Independent selected tile reads for 1, 8 and 32 sectors, with output/status
  allocation included. The maximum selected output size is bounded by that
  count, not the whole cache.
- Required payload bytes, actual arena reservation, directory, calibration,
  encoder scan workspace, partial-tile padding and raw fallback counts.
- Per-call Torch peak allocated/reserved memory and incremental peaks, which
  include actual resident original inputs and output tiles. On-chip resources
  are outside Torch's allocation counters.
- Optional P0 pack/full-unpack controls on the same original words and
  calibration. A one-tile P1 time is never compared with a whole-cache P0
  decode as if they performed equal work.

Each actual CUDA case compares P1 descriptors and every published payload
byte against the independent oracle, then checks decoded original words.
Software cases cover every BF16 bit pattern through packed escapes and raw
fallback, zero escapes, partial tails and capacity rejection. These software
rates are not model compression measurements.

Full-model serving capacity, latency under concurrent attention, SSD time and
T-to-P are explicitly marked unimplemented. CPU-oracle mode projects the
declared byte ledger but does not measure GPU allocations or timing. Offline
SM86 compilation and Triton CPU interpretation are also separate from CUDA
execution.

## A4500 commands

Use the working CUDA PyTorch environment. Triton should match that PyTorch
installation; the local offline/compiler checks used Triton 3.6.0. This task
does not require reinstalling Torch or downloading model weights.

First use the frozen real-word fixture for a bounded full-model SmolLM check:

```bash
(
set -euo pipefail
cd ~/kv_reconstruction
git pull --ff-only origin main
source .venv/bin/activate
python -c 'import torch, triton; assert torch.cuda.is_available(); print(torch.__version__, triton.__version__, torch.cuda.get_device_name())'
python -m unittest tests.test_pool_p1 tests.test_pool_p1_runner -v

run_dir="runs/pool_p1_a4500_$(date +%Y%m%d_%H%M%S)"
python scripts/bench_pool_p1.py \
  --fixture benchmark_fixtures/team_transfer_smol_replay_v1 \
  --device cuda:0 --compare-p0 \
  --arena-ratios 0.8,1.0 --tile-counts 1,8,32 \
  --warmup 3 --repeats 20 --output "$run_dir"
cat "$run_dir/summary.md"
)
```

Only words and pool calibration from the fixture are needed. The unused
split T predictor NPZ does not need reassembly for this P1 task. All required
files are verified against the fixture's hash ledger before use.

After reviewing that result, reuse the original Qwen probability run for a
paired four-layer pool measurement on exactly its TEST pages:

```bash
run_dir="runs/pool_p1_qwen4_a4500_$(date +%Y%m%d_%H%M%S)"
python scripts/bench_pool_p1.py \
  --captures-from runs/block32_probability_qwen4_20261021_065346 \
  --device cuda:0 --compare-p0 \
  --page-tokens 32 --page-starts=0,512,1024,-1 \
  --arena-ratios 0.8,1.0 --tile-counts 1,8,32 \
  --warmup 3 --repeats 20 --output "$run_dir"
cat "$run_dir/summary.md"
```

This import verifies the original self-contained capture manifest and hashes.
It fits pool calibration on TRAIN only, without loading model weights, H or
the dense transfer predictor. It reuses exploratory TEST fixtures and is not
a fresh holdout or full-model Qwen measurement.

Reports are small: `results.json`, `summary.md` and `source_manifest.json`.
No original captures or pool payloads are exported by this runner.

## Local validation

The development workspace has no CUDA device. Explicit CPU validation is:

```bash
python scripts/bench_pool_p1.py --cpu-oracle \
  --output runs/pool_p1_cpu_oracle
python -m unittest tests.test_pool_p1 tests.test_pool_p1_runner -v
TRITON_INTERPRET=1 python -m unittest tests.test_pool_p1 tests.test_pool_p1_runner -v
python scripts/check_pool_p1_kernels.py \
  --output runs/pool_p1_compile_sm86.json
```

The interpreter command requires CPU-compatible Torch and Triton but executes
on CPU and is labeled accordingly. The compiler check emits SM86 cubins
without executing them. Actual CUDA tests remain pending until run on the
A4500; a CUDA request here fails rather than running a CPU codec silently.
