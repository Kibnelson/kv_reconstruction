# From allocation cleanup to exact entropy lookup

The completed allocation experiment reduced paired decode latency from
88.162 to 83.243 ms: **4.919 ms, or 5.58%**. Complete packets remain
33,360,893 bytes on average, saving **43.19%** of the original BF16 tensor.
All packet-byte and original-word checks passed. Native baseline runtime
policies were preserved.

Source: [allocation GPU report](../benchmark_results/lite_allocation_a4500_20261028_103751_647646_report_20261028_103908/summary.md),
commit `46810dc`.

## What the current result means

| Method | Complete packet bytes, mean | Decode ms | Modeled precompressed fetch at 1 Gb/s, ms |
|---|---:|---:|---:|
| Raw BF16 | 58,720,573 | 0.332 | 470.096 |
| Official SplitZip, framed | 44,197,315 | 0.532 | 354.110 |
| Official ExANS, framed fixed slots | 45,433,689 | 3.691 | 367.161 |
| Lite, fill disabled | 33,360,893 | 83.243 | 350.131 |

Lite now has a **calculated 3.980 ms / 1.12% advantage at 1 Gb/s**, with
24.52% fewer packet bytes than SplitZip. The calculated crossover against
SplitZip is approximately 1.048 Gb/s. This is a small model margin, not a
measured network or serving win. Larger workloads and actual transport need
separate validation. The experiment still uses two 512-token documents from
the pinned original BF16 Llama-3.2-3B cache, all 28 layers.

The eager profile contains only 57 fill kernels after cleanup, matching the
28 feature-zero initializations, 28 surprise-zero initializations and one
prior-one initialization. Keep these required fills. Entropy decoding remains
49.467 ms, about 58% of the instrumented device-kernel sum. Profile durations
are diagnostic; they are not an additive breakdown of ordinary graph latency.

## The next test keeps the wire format unchanged

The current decoder first recovers a slice symbol, then the exact original BF16
word within that slice. Stage 1 starts from a coarse slot lookup and performs
a binary search. The next implementation replaces that search with a complete
slot-to-symbol table derived from the **same integer cumulative frequencies**.

For each of ten classes and each slot from 0 through 65,535:

`inverse[class, slot] = searchsorted(cumulative[class], slot, side="right") - 1`

There are 655,360 entries. A uint16 table occupies **1.25 MiB**. It changes
decoder execution and shared memory use, not probabilities or packet bytes.
Its construction, digest, retained original tables and total coder-state bytes
must be reported. A larger lookup may also cost cache traffic; speed is an
experimental question.

Stage 2 estimates the word before the final exact cumulative search. Two
refinement steps improve that estimate in the reference. Test zero, one and
two refinements with the full Stage-1 lookup. Every variant retains the
original cumulative mapping, rounding rules and final exact search.

| Arm | Stage-1 lookup | Stage-2 estimate refinements | Allocation fills |
|---|---|---:|---|
| Reference | Original coarse table and search | 2 | Disabled |
| Full R2 | Full inverse CDF | 2 | Disabled |
| Full R1 | Full inverse CDF | 1 | Disabled |
| Full R0 | Full inverse CDF | 0 | Disabled |

Removing a refinement can increase final-search iterations. These arms isolate
that tradeoff; they do not assume fewer instructions always mean lower latency.
Deterministic arithmetic remains enabled, with explicit zeros/ones retained.

## Why fewer estimate refinements can remain exact

For an active finite slice, let `C` be its unchanged strictly increasing
cumulative map and `oe` any estimated word clamped to the valid range.
If `C(oe) > slot`, the correct word precedes `oe`. If `C(oe+1) <= slot`, it
follows `oe`. Otherwise `oe` already identifies the symbol. The unchanged
endpoint values and binary search recover the same exact word from the
appropriate bracket. Refinement changes the starting estimate, not this map.

This argument applies to valid encoded streams. Edge tests must also cover
one-word slices, empty inactive lanes, escapes, signs, signed zero, subnormals
and interval boundaries. The inherited raw fallback still handles exceptional
whole-engine inputs. Exhaustively check every Stage-1 table slot against the
original cumulative intervals.

## Measurement and acceptance

Reuse the existing packets and fitted state. Balance four-arm order and
alternate the two different compressed payload lengths at fixed geometry.
Capture one graph per arm under the fill-disabled policy. Full decode pays
validation, staging, replay and an independently owned output clone. All four
fitted engines and graph buffers are resident during paired timing; report
that memory scope explicitly.

Each arm must reproduce original encoder packet bytes outside timing. A fresh
receiver gets only fixed state, decoder configuration and serialized packets.
It must reconstruct the original words, alternate A/B/A packets, preserve
previous outputs and capture once per arm. Retimed official controls keep
their original workers, environments and runtime policies. Separate eager
profiles should compare entropy-kernel time and total kernel work. Cold setup,
table construction, uploads and full traces remain separate.

Run on the existing A4500 checkout after pulling the source branch:

```bash
bash scripts/run_lite_entropy_a4500.sh
```

The wrapper performs actual-CUDA correctness gates and publishes the resulting
small report. It does not download a model, refit predictors or rebuild native
baselines. GPU performance for these lookup arms remains unmeasured until run.

## The high-link target still requires larger changes

With the latest retimed SplitZip control, current Lite decoding must fall below
approximately 17.870 ms at 5 Gb/s, 9.201 ms at 10 Gb/s, and 0.748 ms at
400 Gb/s to match calculated serial precompressed-fetch latency. The final
target is about 111 times faster than 83.243 ms. A lookup experiment diagnoses
search cost; it is not a claim that this format can meet that target.

Further work should evaluate an exact small-alphabet conditional coder, cheaper
predictors and fewer dependency waves, followed by independent-request batching
and direct endpoint buffers. Keep transfer savings, single-request latency,
throughput and queueing separate. The [high-link roadmap](lite_high_link_roadmap.md)
describes these structural paths. The proposed contribution remains exact
conditional BF16 coding from reconstructed KV; inverse-CDF tables and allocation
settings are established execution techniques, not standalone novelty claims.
