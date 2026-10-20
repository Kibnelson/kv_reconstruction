# Prefill side codes for independent exact KV pages

This experiment tests whether a small code created from actual prefill hidden
states can improve exact KV coding while removing the previous-layer decoding
dependency. The [first A4500 smoke review](qwen_prefill_side_gpu_review.md)
validates independent exact restoration, but finds no savings advantage for
the hidden code over no-code. Longer training and broader validation remain
necessary; a serving advantage is unmeasured.

## Flow

1. Observe the actual normalized hidden tensor feeding Qwen's K/V projections
   with read-only forward hooks. TRAIN, DEV and TEST use separate fixtures.
2. Fit a small shared encoder and probability decoder on TRAIN only. The encoder
   outputs `rank` values per token, rounded to BF16. A straight-through gradient
   trains the encoder through that rounding.
3. During DEV/TEST prefill, create the retained BF16 codes inside the attention
   callback. The callback runs before model forward returns. Actual hidden/KV
   arrays are diagnostic captures, rather than decoder inputs.
4. Predict a logistic distribution for each original BF16 KV symbol from the
   retained code, layer ID and position. Encode the original symbols with the
   existing exact CPU arithmetic coder. All symbols have nonzero probability.
5. Store each page's code, exact entropy stream, geometry, identity and integrity
   metadata. Store an offset index for the collection of pages.
6. Reopen the serialized container and restore pages in shuffled order using
   only their own bytes and frozen shared model state. Compare every restored
   word with its original. Restore the complete DEV prefix and compare finite
   logits, generated tokens and appended KV for continued generation.

The neural network predicts probabilities. The entropy stream carries everything
the prediction misses, so a poor predictor increases bytes rather than changing
the reconstructed KV. The same-runtime arithmetic/predictor contract is checked;
cross-platform floating-point CDF reproducibility is not promised.

## Matched candidates

| Candidate | Encoder input | Retained code | Other-page prerequisites |
|---|---|---|---|
| `none` | Layer ID only | None | None |
| `kv` | Same-page actual final K and V | Learned BF16 code | None |
| `kv_canonical` | Final K inverse-rotated using known positions/fixed RoPE, plus V | Learned BF16 code | None |
| `hidden` | Actual normalized pre-projection hidden states | Learned BF16 code | None |
| Frozen exponent control | Produced KV | Exact packed fields | None |
| Shared previous-layer neural control | Previously restored layer's exact KV | Existing exact probability stream | Previous-layer chain |

`kv_canonical` is computed from the final produced K, rather than copying the
hook's original pre-RoPE K. This tests whether any gain over `kv` comes merely
from removing the positional rotation at the encoder. Its BF16 canonical input
is an encoder feature; the reconstruction target remains the original final KV.

The four local models share the same decoder width, optimizer-update count and
seeded record/window schedule. `kv`, `kv_canonical` and `hidden` share the same
code rank. Encoder input dimensions and normalization state differ, so parameter
counts, MACs and state costs must still be compared. The no-code control has a
smaller first decoder projection. The previous-layer control is a different
architecture; set its width and total updates explicitly for comparisons.

## Objective and accounting

The training objective is the existing logistic density times BF16-bin-width
rate proxy. It is neither reconstruction MSE nor the exact integer-CDF packet
cost. Actual entropy streams and their complete serialized bytes decide results.

For a page of `T` tokens with rank `r`, the retained-code cost is exactly
`16 * r * T` bits. This cost is constant during a fixed-rank fit and is fully
charged in the packet. The complete storage ledger is:

`exact KV stream + BF16 code + page framing + container/index + shared state`.

Shared state conservatively includes encoder and decoder weights, all
normalizers and RoPE parameters. Headline accounting charges the larger of
serialized and resident state sizes. State amortization across compatible
contexts is an estimate, rather than a measured concurrency result.

Only the selected TEST pages are encoded and timed. Their ratio extrapolated to
the complete context is explicitly an estimate. The same prechosen token starts
are observed at every layer and used by every candidate. Complete DEV restores
are correctness tests on that prefix, rather than full long-context TEST proofs.

## Run on the A4500

Local validation passed 293 source tests under both Transformers 4.57.6 and
5.20.1 with Torch 2.11.0 CPU, plus 12 packaging/build tests. Eleven
accelerator/optional tests were skipped in each source suite. The portable
end-to-end tiny-Qwen run covers two TEST fixtures at 16 and 24 tokens, all six
methods, and five complete 16-token DEV learned-cache continuation checks.
Every checked KV word and every continued finite logit/token/appended-KV result
matches exactly. These random tiny weights establish functionality, rather than
pretrained compression rates or GPU performance. The
[packaged run](../benchmark_results/prefill_side_tiny_cpu_20261020/summary.md)
retains the evidence and source identities.

Use a fresh output directory for every run. Existing packages and the working
Qwen checkpoint can be reused.

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main

python qwen_kv_gpu_bench.py --prefill-side-study \
  --device cuda --rank 8 --width 128 --steps 200 --batch-tokens 64 \
  --shared-width 128 --shared-steps 200 \
  --train-tokens 512 --dev-tokens 128 \
  --lengths 2048 --cases 1 --chunk-tokens 512 \
  --tile-tokens 64 --max-pairs 72 --generation-steps 2 \
  --output runs/qwen_prefill_side_smoke
```

This first run checks the pipeline and produces preliminary held-out rates;
200 updates across 36 layers is a small training budget. The smoke now shows
only 2–10 updates per layer and a no-code lead. First isolate training duration:
keep rank 8, width 128, batch 64 and the original data/geometry, while raising
the aggregate update count to 7,000:

```bash
python qwen_kv_gpu_bench.py --prefill-side-study \
  --device cuda --rank 8 --width 128 --steps 7000 --batch-tokens 64 \
  --shared-width 128 --shared-steps 7000 \
  --train-tokens 512 --dev-tokens 128 \
  --lengths 2048 --cases 1 --chunk-tokens 512 \
  --tile-tokens 64 --max-pairs 72 --generation-steps 2 \
  --output runs/qwen_prefill_side_train7000
```

Then broaden to `--lengths 2048 8192 --cases 3 --max-pairs 144` with a fresh
output directory. Width 512 and rank 32 are separate later ablations, after
testing whether training duration helps at the current capacity and side cost.
Reusing the smoke fixture is diagnostic; choose changes using DEV and use new
held-out data before presenting a selected configuration's final result. The
Python CPU arithmetic reference can make these runs take substantial time.

Create a size-safe results package and push it:

```bash
python scripts/package_results.py runs/qwen_prefill_side_smoke \
  --destination benchmark_results/qwen_prefill_side_smoke_a4500

git add -f benchmark_results/qwen_prefill_side_smoke_a4500
git commit -m "Add A4500 prefill side-code smoke results"
git push origin main
```

## Interpretation

A useful representation result requires `hidden` to improve actual complete
packet/state costs against the same-page KV controls, or to trade a small number
of bits for demonstrably cheaper independent restoration than the chain model.
An improvement over `none` alone does not establish an upstream-information
advantage. The earlier Huffman/sign codec is another practical comparator before
claiming the strongest overall compression result.

The experiment logs TRAIN hashes, frozen artifact identities, per-layer and
position costs, retained-code bits, predictor versus CPU entropy times,
dependency depth and output-buffer requirements. The purpose is to determine
whether the next effort belongs in the side representation, probability model,
training data, framing or entropy implementation.

The hooks include diagnostic CPU copies and the learned encoder may transfer
those arrays back to the chosen device. Encoding the exact stream happens after
capture. This measures a real producer-created side code, not a fused production
encoder or its marginal prefill overhead. Full raw model KV remains resident
during capture. CPU arithmetic, output buffers, neural workspace and weights
remain necessary during restoration. Encoded-resident attention, SSD/network
pipelining, capacity-bounded scheduling and concurrent-serving gains are outside
this implementation.

Learned side information, low-rank KV representations and independent exact
blocks have precedents. This experiment tests a particular combination and
tradeoff; successful execution alone does not establish research novelty.
