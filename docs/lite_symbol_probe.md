# Exact integer symbols: rate probe and link targets

The next iteration tests whether an exact small-integer alphabet is worth a
new GPU decoder. It reuses the current fixed Lite predictor and original BF16
captures. It does not implement a new entropy packet or measure new decoder
latency. The completed coarse-lookup graph remains the current implementation.

## Starting point and targets

The latest paired reference decodes in 83.316 ms with 43.19% byte savings.
The full slice-LUT variants were slower and are not selected. Source:
[entropy GPU report](../benchmark_results/lite_entropy_a4500_20261028_110019_1125591_report_20261028_110230/summary.md).

At the existing packet size, the decoder budgets to match the latest retimed
SplitZip control are:

| Effective link | Required decode at most, ms |
|---:|---:|
| 1 Gb/s | 87.208 |
| 100 Gb/s | 1.384 |
| 200 Gb/s | 0.950 |
| 300 Gb/s | 0.806 |
| 400 Gb/s | 0.734 |

These are calculated serial precompressed-fetch targets, not measured network
rates. The new candidate's targets must be recomputed from its own estimated
byte count. A changed rate changes the permissible decoder cost.

## The exact symbol representation

Round the predictor to a BF16 word using the original round-to-nearest-even
bit rule. Map the predicted and original words to monotonically ordered
integer ranks. Code their exact signed rank difference within a radius `R`.
An out-of-radius value or nonfinite pair uses ESC followed by the original
16-bit word. Reversing the rank difference or taking the raw escape recovers
every original bit; prediction rounding does not quantize the original input.

For example, prediction 1.0 has BF16 word `0x3f80`. Original word `0x3f82`
represents 1.015625 and has rank difference +2. With radius 127, encode symbol
129 and reconstruct exactly. Original 2.0 has word `0x4000`, difference +128;
it takes an escape carrying `0x4000`. No original word is approximated.

The hypotheses are that a small discrete table can replace floating-point BF16
interval inversion, and that decoder-available scale information can sharpen
the discrete probabilities. Neither rate improvement nor speed is assumed.

## Predeclared calibration and evaluation

Use the same original BF16 Llama-3.2-3B revision and all 28 layers. The original
TRAIN documents are train0 through train3, using their first 1,024 tokens. The
validation document is tables, using 1,024 tokens. Held-out TEST consists of
test_text and test_code, using the same 512-token prefixes as the GPU controls.
Original predictors are not refitted. Existing spread calibration is retained.

The candidate grid contains radii 127, 255, 511, 1023 and 2047, each with one
unconditional table or 32 fixed scale/ULP contexts. Contexts use the exponent
difference between the predicted BF16 word and the float32 scale. The rule is
fixed before evaluation; the decoder can recompute it from its own prediction
and scale without transmitted per-value labels.

Fit strictly positive frequencies summing to 65,536 from TRAIN counts only.
Integer largest-remainder normalization resolves equal remainders by ascending
symbol index. Empty contexts use a deterministic uniform fallback. Select by
validation cost, with the declared radius/context tie order, and write the
selection receipt before collecting TEST costs. TEST evaluates that fixed
selection and a predeclared unconditional control at the same selected radius.
It is not used to revise tables, bins or radii.

Known-KV replay must supply exactly the original encoder's predictions and
adaptive scales. A spy test compares their bit patterns. This replay has the
original values available and therefore is a diagnostic, not an independent
receiver or a decoder throughput benchmark. It covers all layers, including
the predicted first layer; there is no newly introduced raw anchor.

## Costs and patterns to record

The empirical normalized-frequency cost is `-log2(f/65536)` per symbol, plus
16 raw bits for every escape. A planning packet estimate adds all 32-bit rANS
states for every padded block lane, INT64 program directories and explicit
hypothetical framing that identifies the new representation and tables.
These estimates are not actual rANS byte counts or proven size bounds.

Report physically fitted frequency/CDF arrays separately from a projected
future GPU inverse lookup and total table layout. Predictor state is additional.
No candidate fetch latency is calculated by borrowing the old decoder time.
Instead, report the required candidate decoder budget against inherited
measured controls at 1, 100, 200, 300 and 400 Gb/s. Those controls retain their
original protocol and are identified as reused measurements, not retimed ones.

Record per-layer cost, context occupancy, escape fraction, central rank mass
and conditional-versus-unconditional cost. Predeclared escape diagnostics
include finite sign crossings and predicted zero/subnormal cases; these
categories can overlap. Rank distance is not uniform numerical distance, so
these diagnostics can explain a failure of the representation.

Source verification, uploads, cold setup and table fitting are separate.
Known-KV predictor-only and instrumented replay/histogram intervals are
diagnostics; the instrumented pass can include proof checks and host gaps.
Document histogram checkpoints and the selection receipt support safe resume.

## Run and decision

```bash
bash scripts/run_lite_symbol_probe_a4500.sh
```

The wrapper verifies actual CUDA, runs correctness gates, performs the probe
and publishes a small report. It reuses local captures and fitted state, with
no model download, predictor refit or native baseline build. The default
parent is `runs/lite_entropy_a4500_20261028_110019_1125591` and source is
`runs/llama32_3b_compare`. Local software evidence cannot substitute for the
GPU run.

Use the held-out rates and decoder budgets to decide whether to implement a
versioned GPU codec. Exact integer mapping alone is not a measured lossless
codec result. The larger high-link objective also needs cheaper predictors
or fewer dependency waves; this probe isolates the entropy representation.
