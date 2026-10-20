# Testing the supplied transfer format and paid prefill information

The supplied `kv_combined_test.zip` is preserved as readable, unmodified source
under [experiments/team_transfer_original](../experiments/team_transfer_original).
Its source manifest records the archive and file hashes. The archive contains
scripts and historical logs, but no original ACT captures or fitted constants.
Consequently the historical 8.360-bits/value result can be checked for internal
consistency; reproducing that exact rate requires the original captures.

The [source audit](../benchmark_results/team_transfer_source_audit_20261020/summary.md)
distinguishes historical evidence from newly executed probability and accounting
checks. The independent [pool reference test](../benchmark_results/team_pool_reference_cpu_20261020/summary.md)
uses the original pool functions on fresh real-model captures. It measures packed
bytes and exact reconstruction, not an attention kernel or GPU memory allocator.

## Independent CPU validation identity

The available checkpoint is the **base** `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, loaded through HuggingFace in BF16.
It is a different checkpoint and execution path from the team's
`SmolLM2-135M-Instruct` Q4_1 GGUF with NumPy forward computation. The local
checkpoint download manifest names that base revision; all eight listed files
were independently checked against its SHA256 values. In particular:

- `model.safetensors`: 269,060,552 bytes;
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
- `config.json`:
  `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`.

The capture manifest also records exact input-ID, KV and hidden-array hashes.
The independent run uses three 512-token TRAIN captures, a 512-token DEV
capture and two 512-token TEST captures, rather than the team's six 8K TRAIN
documents and 8K test documents. These differences prohibit interpreting a
similar or different coding rate as reproduction or refutation of 8.360 bpv.
The completed transfer run and its artifact review are linked below. The compact
bundle also retains the independently verified `model_download_manifest.json`.

The observed CPU environment is NumPy 2.3.5, PyTorch 2.14.1+cpu,
Transformers 4.57.6, safetensors 0.8.0, tokenizers 0.22.2 and constriction 0.5.0.
`requirements-transfer.txt` pins the direct NumPy/Transformers/constriction
versions through `requirements.txt`; it does not constitute a full environment
lock. safetensors is a version range and transitive dependencies are not all
pinned. The A4500's CUDA environment has a separate Torch build, so CPU timings
must retain their actual runtime identity.

## Controlled transfer comparison

`scripts/bench_transfer_extensions.py` captures original BF16 K/V and actual
attention-input hidden states from a model forward. Hooks return `None` and do
not replace model tensors. Standard HuggingFace split-half key coordinates are
permuted into the original codec's adjacent RoPE-pair order; values are unchanged.

All variants use the same source words, positions and probability construction:

| Variant | Prediction or transmitted information |
|---|---|
| Mean | Per-channel numerical mean |
| Previous | The same token's previous-layer K/V |
| AQUA-style | Previous K to K; previous V and decoded current K to V |
| Block | Previous-layer K/V and earlier decoded 32-channel groups |
| Block + KV metadata | TRAIN-fitted PCA of produced KV, quantized to paid int8 codes |
| Block + hidden metadata | TRAIN-fitted PCA of actual upstream hidden states, paid int8 codes |
| Shuffled hidden control | Same code budget with token alignment deliberately broken |
| Block + uniform coset | Transmitted low-bit residue followed by conditional quotient coding |
| Hidden + uniform coset | The paid hidden codes and residue factorization together |
| Model-coset control | Factorization of the same joint probability as direct coding |

This AQUA-style control tests predictor dependencies with the common exact coder;
it is not an implementation or performance comparison of AQUA-KV's quantizer.
The sign/exponent group is coded first; the seven-bit ordered mantissa symbol
is then factored into its low-bit residue and conditional quotient. These
modulo cosets are a scalar binary factorization, not an E8 or LDPC codec.
Under the same ideal joint distribution the model-coset control has identical
information cost. Actual coder precision can change its physical stream size.

TRAIN fits predictors and metadata projectors/quantizers. The first half of DEV
calibrates probability scales; a disjoint DEV page selects metadata rank and
residue width. In addition to each family's own DEV winner, KV-derived and
shuffled-hidden controls are retained at the DEV-selected hidden rank. This
provides an equal transmitted-code budget even when family winners differ.
Shuffled features stay within the corresponding DEV partition. TEST reports the
frozen choices; its per-layer diagnostics do not define a deployed policy.

Each TEST packet is decoded in a new process with only its frozen state and
packet as data inputs. Packet sizes include int8 metadata, stream bytes, shape,
the contiguous token start, state identity and checksums. Unsupported fixed-point contexts use an
explicit raw fallback. The original arithmetic and block-predictor functions are
checked for parameter and stream equivalence on a controlled fixture.

Report separately:

- Complete packet bytes and exact word comparisons.
- Decoder-resident arrays, encoder-only projector arrays and serialized state.
- Projection/quantization time, codec encode time and codec decode time.
- Per-layer ideal coding costs, metadata costs, clipping and fallback counts.
- Capture/device/runtime versions, token IDs and content/source hashes.

Capture can use CUDA. **The probability coder and all entropy-codec timings are
CPU reference measurements.** Codec encode times include metadata projection
and quantization but exclude the original model forward and activation capture.
CUDA capture copies full hidden/KV arrays to the CPU; measuring and avoiding
those observation costs is part of a future serving implementation. The current
format has one entropy stream per
packet; it does not establish token-parallel GPU decoding, cross-machine parity,
bounded scratch or compressed-resident attention. State-amortized coding ledgers
are arithmetic scenarios, not measured GPU admission capacity.

## Measured CPU results

The [compact result bundle](../benchmark_results/team_transfer_extensions_smol_cpu_20261020/summary.md)
contains the full tables, exact token IDs, model/source provenance and the
[422/422 passing artifact checks](../benchmark_results/team_transfer_extensions_smol_cpu_20261020/review/summary.md).
The [73-test focused suite](../benchmark_results/team_transfer_extension_tests_20261020.txt)
also passes, including the ACT import guard and regenerated standalone checks.
All 40 TEST packets decoded in separate processes. Each variant recovered the
same 1,474,560 source BF16 words exactly: four 32-token pages, at positions 0 and
480 in the two 512-token TEST captures, across all 30 layers. These are short
context correctness and rate measurements, not a concurrency or long-context
serving benchmark.

| Representation | Complete packet bits/value | Reduction from BF16 |
|---|---:|---:|
| Per-channel mean | 10.227322 | 36.08% |
| Previous-layer KV | 9.893273 | 38.17% |
| AQUA-style dependency control | 9.840777 | 38.50% |
| **Previous layer + 32-channel blocks** | **9.430599** | **41.06%** |
| Block + paid KV PCA metadata | 9.502062 | 40.61% |
| Block + paid actual-hidden PCA metadata | 9.507791 | 40.58% |
| Block + paid shuffled-hidden metadata | 9.514171 | 40.54% |
| Block + uniform two-bit mantissa coset | 9.430621 | 41.06% |
| Block + same-joint modelled coset control | 9.430859 | 41.06% |
| Hidden metadata + uniform coset | 9.507791 | 40.58% |

The block predictor wins on both text pages combined (41.40% reduction) and both
code pages combined (40.71%). It therefore meets the 40% **warm transfer-packet**
target in this experiment. The independently packed pool format uses
11.472233 bits/value, or 28.30% reduction, on the same pages.

DEV selected four int8 metadata coordinates per layer/token and two-bit cosets.
The hidden hint costs 0.083333 bits/value before framing. It makes the complete
packets 14,228 bytes larger than the block baseline over the four pages; KV hints
cost 13,172 extra bytes. Neither metadata variant wins on any of the four pages.
The uniform coset costs four extra bytes overall, and the same-joint control
costs 48 extra bytes. Those are effectively the same rate, with finite-coder and
framing differences. Combining the current hidden hint and coset adds no saving.
This result concerns raw-hidden PCA hints and scalar mantissa cosets; it does not
test the earlier Hamming-syndrome or E8 constructions.

The per-layer ideal-bit screen finds a positive net hidden-hint benefit only in
layer zero, where no previous-layer KV exists. Even there, the KV-derived hint
helps more. This is a diagnostic from TEST, not a measured gated representation;
any layer/rank selection must be fitted on TRAIN/DEV before a new TEST run.

Shared state is additional. The measured block decoder's arrays occupy
51,795,464 bytes (49.40 MiB). If that entire state is charged once over 8,192
aggregate token positions, and the measured packet rate is assumed constant,
the arithmetic total is 13.821 bits/value (13.62% reduction). Over 131,072 token
positions it is 9.705 bits/value (39.34%). These scenarios omit model weights,
pool layout, raw outputs and scratch; they are not measured GPU capacity gains.
The separately validated INT32 storage improvement below is not substituted
into the main results.

CPU timings are retained in the results, including intermittent executor stalls.
They do not support ranking the variants by speed or predicting GPU latency.
The executed runner and codec sources are preserved in
[commit 8b08725](https://github.com/Kibnelson/kv_reconstruction/commit/8b08725eb9333eab10d6ebd83a598df9496b8392).
A subsequent ACT import guard fixes sparse hidden-layer indexing; it does not
change these fresh-capture measurements. For a later full-artifact review, use
`--source-root` pointing to those executed sources. The compact bundle omits
large states, packets and activations, so it records the executed review rather
than providing those full artifacts.

## Separate fixed-state storage check

The [state probe](../benchmark_results/team_transfer_state_probe_cpu_20261020/summary.md)
stores the channel-block predictor's coefficients as INT32 instead of INT64.
Total frozen arrays fall from 51,795,464 to 26,580,488 bytes. All 184,320 checked
predictions are identical. More strongly, an unchanged 444,216-byte held-out
entropy payload decodes to the same 368,640 BF16 words with the compact state.
Only the packet header's state identity is explicitly rebound for this test.
This reduces fixed state; it adds no KV packet savings and measures no GPU speed.

A per-block INT16/INT32 range calculation suggests 16,068,104 total array bytes,
but that mixed-dtype candidate has not passed the packet test. The main transfer
comparison continues to charge its actual INT64 state, independently of these
storage experiments.

## Run on the A4500

In the existing working CUDA virtual environment:

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main
python -m pip install -r requirements-transfer.txt

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/bench_transfer_extensions.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --device cuda --max-layers 4 \
  --train-tokens 256 --capture-tokens 2048 \
  --page-tokens 32 --page-starts=0,-1 \
  --metadata-ranks 4,8 --coset-bits 2,4 --threads 2 \
  --output runs/qwen_transfer_metadata_coset

python scripts/review_transfer_extensions.py \
  runs/qwen_transfer_metadata_coset \
  --output runs/qwen_transfer_metadata_coset_review

python scripts/package_transfer_results.py \
  runs/qwen_transfer_metadata_coset \
  --review runs/qwen_transfer_metadata_coset_review \
  --destination benchmark_results/qwen_transfer_metadata_coset_a4500
```

The contiguous four-layer limit bounds CPU calibration and gives a partial-layer
smoke study. The model forward still runs normally. Remove `--max-layers 4` for a
full-model study; dense calibration and the CPU reference coder are expensive.
Use `--reuse-captures` with the same output and capture configuration to avoid
another forward pass. Reuse checks the capture identity and content hashes.
The compact packager retains results, review evidence and exact token IDs. It
omits large fitted states, hidden/KV captures and decoded arrays; rerunning the
artifact reviewer requires the full original run directory. Publish only the
packaged directory:

```bash
git add -f benchmark_results/qwen_transfer_metadata_coset_a4500
git commit -m "Add A4500 transfer metadata and coset comparison"
git push origin main
```

For the team's original ACT directory, use `--act-dir PATH --train-cases 6`.
`--no-metadata` permits its original partial-hidden captures and still tests the
baseline and coset variants. Supplying files with those names alone does not
establish that they are the historical experiment's exact inputs.

## Decision rule

Prefer a metadata variant only when its complete frozen-configuration packet
beats the matched baseline after paying metadata and framing. Compare its added
state and execution cost as well. The reviewer computes per-layer ideal gains
minus paid metadata bits to identify a later TRAIN/DEV-fitted gating experiment;
this screening is not an additional encoded saving. Useful next steps depend on
these measured gains, not on multiplying separate representation percentages.

## Untested next hypothesis: predict the useful innovation

Raw-hidden PCA retains directions with high hidden-state variance. Those need
not be the directions that explain KV information left uncertain by the
previous-layer/earlier-channel predictor. A next candidate is a **TRAIN-only,
residual-directed reduced-rank sketch**: fit the baseline KV predictor, measure
its unexplained KV on TRAIN, and fit a small hidden projection specifically to
predict that residual. Conditioning the hidden sketch on decoder-known
previous-layer KV could further remove already-predictable information.
Any earlier-channel conditioning must use only the prefix available at that
decoding step. This is a hypothesis, not a tested codec or a new mathematical
primitive.

The comparison must charge the complete code length:

\[
L(z) + L(\text{original BF16 words}\mid\text{known KV},z)
+ L(\text{headers}),
\]

plus separate projector/predictor state and execution costs. A lower residual
MSE or a lower conditional ideal bitrate alone is insufficient. Metadata must
save more than its transmitted code bytes and added framing, and state must
amortize at the intended workload. There is no inherent Shannon gain from
sending deterministic metadata: a practical improvement would come from a
better tractable conditional model or execution format. Freeze rank, quantizer
and any layer gate on TRAIN/DEV before evaluating fresh TEST data.
