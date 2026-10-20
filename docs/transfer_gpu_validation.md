# Separate A4500 validation of the pool format and transfer predictor

The first run replays the exact CPU test words, rather than generating different
KV with another model. The frozen fixture contains four 32-token pages across
all 30 layers, fitted pool calibration, the block32 transfer state and its
original packets. It contains no language-model weights. Files and source
identities are checked before benchmarking.
Keys are already in the capture's adjacent RoPE-pair order. Model prefill,
cache-layout conversion and installation into a serving cache are outside these
component timings.

| Stage | Execution | Check |
|---|---|---|
| Pool encode and decode | Requested CUDA device | Identical CPU record bytes and original BF16 words |
| Raw versus complete packed-buffer H2D | Pinned CPU memory to CUDA | Actual copied bytes, CUDA event intervals and synchronized wall time |
| Transfer fixed-point predictor | Requested CUDA device | Every replayed mean equals the NumPy integer reference |
| Transfer packet decode | CPU constriction | Original packet reproduces its source words; reported separately |

**This is not a full GPU transfer entropy decoder.** The original range coder
and probability construction remain on CPU. Predictor replay uses known KV to
advance causal context and cannot be counted as decompressing a packet. The pool
codec does perform device-side packing and unpacking. It is a tensor reference
with Python orchestration and synchronization, rather than fused attention.

## Run the fixed-input experiment

Use the existing working CUDA virtual environment on the NUC:

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -r requirements-transfer.txt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/bench_transfer_gpu.py \
  --replay-fixture benchmark_fixtures/team_transfer_smol_replay_v1 \
  --device cuda --coefficient-dtype int32 \
  --warmup 1 --repeats 5 \
  --output runs/transfer_gpu_replay_a4500
```

The default CUDA request fails if CUDA is unavailable; it never turns a CPU run
into a GPU pass. Explicit `--device cpu` is available for implementation tests
and marks GPU measurements as unmeasured. The fixture's generation and local
validation do not constitute a CUDA result.

The [local CPU replay](../benchmark_results/team_transfer_gpu_cpu_validation_20261020/summary.md)
passes all four pages: 1,474,560 source words, original pool bytes/offsets,
transfer packet decodes and predictor means. The focused suite runs 37 tests:
35 pass and two actual-CUDA checks are skipped because no CUDA device exists in
this workspace. A separate real `--device cuda` request exits with a failure
report and no CPU fallback.

For the full unchanged fixture, expected original payload totals are:

| Representation | Bytes | Bits/value | Reduction from BF16 |
|---|---:|---:|---:|
| Raw BF16 | 2,949,120 | 16 | 0% |
| Original pool records | 2,114,562 | 11.472233 | 28.2985% |
| Block32 transfer packets, CPU entropy codec | 1,738,248 | 9.430599 | 41.0588% |

The pool total excludes the newly charged offset directory and serialized
geometry wrapper. The report separately measures those complete bytes, shared
device state and peak temporary allocations. INT32 predictor storage is used
only after every coefficient passes an exact range check; arithmetic still uses
INT64 products and accumulation. This fixture tests the block32 baseline, not
the hidden-metadata ablations or the team's historical 8.36-bits/value inputs.

Inspect `summary.md` and `results.json` under the output directory. Success
requires all byte/word comparisons and predictor comparisons to pass on CUDA.
The summary includes synchronized wall medians for pool encode/decode, raw H2D,
complete packed H2D plus unpack, predictor replay, and persistent device state.
The JSON retains the individual samples, CUDA events, allocation peaks and
cold-state costs needed for a detailed review.
Smaller payloads alone do not establish faster restoration. Event intervals
can include idle gaps from host orchestration; wall time and allocator peaks
must be considered as well. H2D tests use matched pinned sources and preallocated
destinations; they do not measure disk I/O, network transfer or concurrent
serving. Dense outputs are materialized, and no compressed-resident attention or
admission-capacity claim follows from this test.

## Test fresh Qwen captures separately

For Qwen, follow the [CUDA capture and CPU reference commands](transfer_metadata_coset_validation.md#run-on-the-a4500).
Then test that run's actual words and fitted state:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/bench_transfer_gpu.py \
  --reference-run runs/qwen_transfer_metadata_coset \
  --variant block32 --device cuda --coefficient-dtype int32 \
  --warmup 1 --repeats 5 \
  --output runs/qwen_transfer_gpu
```

The referenced Qwen smoke command captures a contiguous four-layer prefix; its
forward pass still executes the whole model. Those results must remain labeled
partial-layer. Changing the model, capture device or inputs can change KV and
compression rates. CPU and GPU codecs must agree on the *same captured words*;
Qwen is not expected to reproduce SmolLM's rates. Existing paid-metadata variants
can be selected by their exact variant name from the reference run; this tests
their predictor with packet-provided codes, not GPU metadata extraction.

To sweep all ten DEV-selected Qwen variants after that reference run finishes,
use the recorded names rather than hard-coding selected rank/residue suffixes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

reference = Path('runs/qwen_transfer_metadata_coset')
report = json.loads((reference / 'results.json').read_text())
variants = ['block32'] + [name for name in report['aggregate'] if name != 'block32']
for name in variants:
    subprocess.run([
        sys.executable, 'scripts/bench_transfer_gpu.py',
        '--reference-run', str(reference), '--variant', name,
        '--device', 'cuda', '--coefficient-dtype', 'int32',
        '--components', 'pool,predictor' if name == 'block32' else 'predictor',
        '--warmup', '1', '--repeats', '3',
        '--output', str(Path('runs/qwen_transfer_all10_gpu_components') / name),
    ], check=True)
PY
```

Keep the full Qwen reference run on the NUC: its captures, fitted states and
packets are required. The portable fixture contains only Block32. Coset rows
reuse the relevant predictor; this sweep does not add a GPU entropy/coset
decoder. Use fresh output directories for subsequent repetitions.

## Share the small reports

Keep large run artifacts on the NUC. Publish only the report files:

```bash
set -e
report_dir="benchmark_results/transfer_gpu_replay_a4500_$(date +%Y%m%d_%H%M%S)"
mkdir "$report_dir"
cp runs/transfer_gpu_replay_a4500/{results.json,summary.md,source_manifest.json,rows.csv,run.log} "$report_dir/"
git add -f "$report_dir"
git commit -m "Add A4500 pool and transfer predictor GPU validation"
git pull --rebase origin main
git push origin main
```

If the runner fails, inspect its `summary.md` and `run.log` before proceeding;
`results.json` records the exception and no GPU pass is claimed. A failed run
has no `rows.csv`. Share the failure reports for diagnosis. Do not commit
captures, predictor states or full reference-run directories with the reports.

The full GPU transfer decoder remains a separate implementation task. It must
match the probability/CDF semantics. A parallel design also needs independently
decodable stream lanes, with their state, offsets and framing charged before
GPU throughput can be claimed for that design.
