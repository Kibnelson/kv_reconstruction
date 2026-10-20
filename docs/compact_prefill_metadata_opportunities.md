# Reducing the cost of prefill metadata and exact KV storage

Study date: 2026-10-20. This is an implementation audit and experiment proposal,
not a new codec result. Evidence comes from the delivered A4500 2×2 training
study at commit `58c4fe2ac4b6ae4f3d84f1ce2ae58fe5995a912f`, its
[corrected review](qwen_coset_training_gpu_review.md), and frozen calibration
arrays. No Qwen forward pass, compact-code training, GPU timing or new encoded
TEST packet was run for this document.

## Two different meanings of 16

Each original KV value is a **16-bit BF16 word**. Separately, our current hidden
metadata contains **eight BF16 values: 16 bytes per token per layer**. Reducing
the latter helps only if it lowers the complete exact-KV representation.

For this Qwen3-4B model, each token/layer has 1,024 K and 1,024 V values. Its
raw KV costs 4,096 bytes. The current side code adds 128 bits, or 0.0625 bits
per original KV value. Eliminating that payload alone recovers only 0.390625
raw percentage points before any change to predictions, corrections or state.

The measured field baseline reduces storage by about 30.5%, corresponding to
about 11.12 bits/value including its charged state estimate. A 40% reduction
requires 9.6 bits/value: roughly **1.52 additional bits/value** must disappear.
The present metadata-specific probability advantage, at best about 0.003011
bits/value in the DEV screen, is far too small. We need stronger conditional
structure as well as cheaper metadata.

## What the completed experiments tell us

| Finding | Evidence | Design implication |
|---|---|---|
| Longer, balanced training supplies a weak probability signal | Balanced-long bit-6 BCE is 0.952712 with hidden metadata versus 0.954210 without it | There is a signal to investigate, but it is not yet an encoded saving |
| Current hard cosets cannot exploit that signal | All four training arms retained raw planes and zero short syndrome groups | Changing metadata size alone will not create correction savings in this codec |
| Much of the useful bit-6 difference concerns V | V contributes 93.5% of balanced-long hidden-versus-no-code BCE improvement; 89.5% in random-long | Test V-specific conditioning; this is an aggregate association, not proof that a smaller V code works |
| Benefits vary across layers | Layers 18, 17, 0 and 16 carry 31.7% of positive balanced-long bit-6 gain | Selective emission deserves a control, but these layers do not contain all the gain |
| Old latent dimensions are not obviously redundant | Top four singular components retain 58.4% of balanced-long encoder projection energy | Simple rank truncation is not a demonstrated near-exact repack; activation and prediction ablations are still needed |
| Encoder state is dominated by normalization | 737,280 of 819,232 encoder-only numeric bytes are source means/stds, about 90% | Redesign the encoder's calibration along with its transmitted code |

The best raw-or-probability DEV screen permits only about 6.17 side bits per
token/layer before extra state and framing, **if its current predictions were
preserved**. Eight INT4 features still cost 32 bits. We should therefore test
a few bits **total**, rather than a few bits for each of eight features.
This screen uses aggregate losses and free fallback decisions. It is neither
an entropy bound on KV nor a measured result for a new entropy coder.

## 1. Derive more context from bits the decoder already has

In `syndrome_models._forward`, each target word sees its own exact sign,
exponent and retained low mantissa bits, plus channel/layer/position information
and the optional code. It does not see a pooled view of all 2,048 words'
known fields in that token, or exact same-page K while predicting V.

First add a **zero-payload control**: compute a small summary from all known
fields in the same token/page. Candidates include per-head exponent/sign
statistics or projections of a coarse value reconstructed from those fields.
Both ends compute precisely the same summary. No unknown mantissa bit may
enter it. Computation and new parameters still count.

Then ask the actual prefill hidden state to supply only information missing
from that summary. One concrete family is

`a = projection(actual_hidden); s = Q(a - predictor(known_field_summary))`.

The decoder computes the same predictor and adds the decoded small innovation
to form its conditioning signal. Alternatively, `s` directly selects a small
probability-model adjustment. Train for the cost of the exact correction
stream, not accurate reproduction of the old eight floating-point features.

This can remove redundant side information and expose cross-channel
dependencies. It might also make upstream metadata unnecessary; the matched
zero-payload control is essential for deciding whether prefill adds anything.

Let `Y` be the transmitted known fields, `M` the remaining exact KV bits and
`S` the prefill hint. Optimize the complete length
`L(Y) + L(S | Y) + L(M | Y, S) + framing + amortized state`.
There is no information-theoretic free lunch: adding a paid hint cannot beat
an ideal joint code for KV merely by adding information. It can help a cheap,
restricted decoder model describe KV dependencies that it otherwise misses.
That operational benefit, and its compute cost, are what we need to measure.

## 2. Replace eight floats with a tiny prefill decision

Try 1, 2 and 4 **total bits** per token/layer, with a learned scalar threshold
or small finite grid. A two-bit code selects one of four prediction adjustments.
It need not reconstruct the previous eight-dimensional code.

A concrete cheap encoder projects the 2,560-dimensional actual hidden input
onto one learned scalar, applies per-layer scalar calibration, and emits one
of four IDs. A small lookup supplies the decoder's conditioning vector or
first-layer bias. This tests whether the relevant information is a compact
decision about the KV distribution rather than a high-precision latent.

| Side representation | Payload bytes/token/layer | Bits/original KV value |
|---|---:|---:|
| Current eight BF16 values | 16 | 0.062500 |
| Eight INT4 values | 4 | 0.015625 |
| Four bits total | 0.5 | 0.001953 |
| Two bits total | 0.25 | 0.000977 |
| One bit total | 0.125 | 0.000488 |
| Sixteen bits shared by a 64-token page | 0.03125 | 0.000122 |

These are **payload arithmetic, not predicted final savings**. Tables, flags,
padding, parameters and changed correction lengths remain payable. The current
packet format requires `tokens × rank × 2` feature bytes: quantizing internally
and then writing BF16 would save no wire bytes. A versioned compact payload is
required.

Finite scalar quantization supplies an established way to create small discrete
representations without a large learned codebook [1]. The proposed tiny KV
configuration and its ability to preserve exact-correction benefit are untested.

## 3. Share a small description within one page

Let actual prefill activations choose a page-level scale, distribution mode or
predictor adjustment. For example, sixteen **bits** for a 64-token page average
0.25 bits per token. This is different from sixteen **bytes** per page, which
average two bits per token. Add token-specific refinements only when they pay
for their values, presence flags and locations.

Also compare using this descriptor for joint sign/exponent distributions,
rather than only omitted mantissa bits. The current lower planes are close
to uniform, while the field baseline already demonstrates that exponent
structure exists. A page descriptor still needs to outperform statistics
computed directly from produced KV under the same coder.

Compare a single pooled descriptor with two anchors inside the page, or a small
piecewise-constant sequence. Pooling may erase the useful token-specific
signal; current moments and hashes cannot establish page correlation. Keep
each page independently restorable, with no dependency on the previous page.

This borrows the hyperprior principle: pay a small description of the coding
distribution only when it saves more main-stream bits [2]. The cited image
codec is lossy; our KV remains exact through a separately charged correction
stream. These are different reconstruction requirements.

## 4. Restore K first, and spend the extra signal on V

Restore K exactly using the independent field path, then predict V from the
same page's exact K, V's known fields and an optional tiny prefill code.
Compare all four combinations: neither addition, K only, prefill code only,
and K plus prefill code. A joint/dropout-trained decoder can share parameters
across those modes instead of retaining two independent model banks.

The V-dominated DEV signal motivates this test. It does not establish that
exact K exposes the missing information. The K-to-V stage adds serial work;
measure that cost, page scratch memory and performance under concurrent
restoration. It avoids an inter-layer dependency, but is still a dependency
inside the page.

KV dependency prediction is established prior work: AQUA-KV exploits key/value
and cross-layer dependencies [3]. Its near-lossless quantization differs from
original-BF16 exactness. Our candidate contribution would have to be the
specific actual-prefill innovation, independent-page exact representation and
measured net benefit, rather than the general K/V-prediction ingredient.

## 5. Change the reconstruction order to expose numerical structure

The current residue keeps the lowest `b` mantissa bits. Those identify a word
modulo a power of two; they do not tightly locate its numerical value. Test a
different exact layout: transmit sign/exponent and the highest mantissa bits
first, giving the decoder a coarse numerical approximation of every KV value.
Use the whole token/page's coarse approximation as context for the next group
of bits, then refine using bits already restored.

One concrete schedule has three stages: the base with two high mantissa bits,
the next two bits coded as a four-symbol alphabet, and the last three as an
eight-symbol alphabet. Symbol probabilities can be computed in parallel
across channels within a stage. Entropy decoding retains stream dependencies
unless separate streams/lanes are implemented and their overhead is charged.
This tests within-word multi-bit structure and cross-channel coarse context
that seven independent binary outputs cannot fully represent. It is a new
layout/model/coder experiment, not a modification with established savings.
Ordering alone cannot reduce the true joint entropy. Low rounding bits may
be less predictable than high mantissa bits, so retain the existing low-bit
residue control and compare all plane/tuple rates before selecting a layout.

At prefill, encode a tiny innovation from actual hidden states **conditioned
on this same coarse base**. Compare the identical refinement decoder with no
hint, a produced-KV hint, and an actual-hidden hint. Only their matched
differences identify the value of actual prefill information. The coarse base
itself and every refinement/correction must count in the final bytes.

Use original integer BF16 fields for exact restoration; floating approximate
values are predictor inputs only. Specify rounding, zeros, subnormals and
nonfinite escape handling. A finite-data smoke result cannot establish all
bit-pattern behavior. Mask and assemble unsigned integer fields, preserving
negative values and signed zeros without exponent carries. Require identical
integer CDFs at both ends; the same FP32 formula alone does not guarantee
cross-device entropy-decoder synchronization. More refinement passes add latency, so compare two and
three stages with the existing page path. Any cross-channel autoregressive
groups would add more serial stages and require a separate throughput case.

This is the larger-storage-impact hypothesis: to approach 40% we must remove
roughly 1.52 bits/value, rather than merely preserve today's approximately
0.003 bits/value metadata advantage. Progressive refinement and conditional
entropy coding are established concepts; their usefulness and any specific
prefill contribution in this exact, page-local construction remain untested.

## Shrink the fixed state at the same time

Current hidden state exceeds the independently fitted no-code model by
819,573 complete artifact bytes. Even with a zero-byte side payload, that
difference costs 0.04342 bits/value at 2K or 0.01086 at 8K for a single cache,
already larger than the present best screened probability benefit.

These numeric encoder payloads identify concrete alternatives:

| Encoder design | Numeric parameter payload | Status |
|---|---:|---|
| Current per-input-coordinate means/stds, rank 8 | 819,232 B | Measured arrays |
| Fold means into a per-layer rank-8 bias, retain input stds | 451,712 B | Same real-arithmetic affine transform; FP32 order changes, so regenerate codes and validate |
| Normalize eight projected outputs per layer | 84,256 B | Different encoder; requires fitting |
| One scalar projection, two scalar calibration parameters/layer, four 8D lookup entries | 10,660 B | Proposed design arithmetic; excludes decoder, manifest and packet costs |

The last two rows are redesigns, not equivalent repackings. Decoder-only GPU
residency can also omit the current 819,232 encoder-only bytes, but they still
exist at the producer and remain part of total system accounting. Existing
`Prepared` loads them all. State shared across compatible concurrent caches
can be amortized once per worker; token metadata is paid for every cache.

A small policy ID also allows a precomputed first-layer bias instead of
repeating the code projection for each KV channel. Reassociation changes FP32
rounding, so encode and decode must use the new path consistently and generate
new corrections. This is a candidate optimization, not a measured speedup.

## What to test first, and what to retain

1. **Replay saved packets before loading Qwen again.** Original NUC run packets
   contain the eight BF16 code words and sufficient fields/corrections to
   reconstruct exact KV. Export code arrays and test coordinate removal,
   rounding, dictionaries and page pooling with the small frozen predictor.
   No 4B-model forward is needed for these sensitivity diagnostics. Uploaded
   results omit packets; this step uses the original NUC run directory.
2. **Fit small alternatives using TRAIN only.** Existing TRAIN code arrays were
   not retained, so learned quantizers/encoders need a TRAIN recapture or new
   TRAIN export. Old-code replay measures sensitivity; it does not train or
   validate a new actual-hidden encoder. Freeze widths/modes on DEV and use
   fresh TEST text/code at 2K and 8K.
3. **Run matched zero-payload, two-bit, four-bit and page-shared variants.**
   Keep predictor capacity comparable. Separately vary pooled known context
   and K-to-V conditioning so the source of any improvement is identifiable.
   Treat high-bit-first grouped refinement as a separate layout experiment,
   with the same three hint-source controls and a charged coarse base.
4. **Measure operational correction bytes.** Retain the current hard-coset
   control, but add a soft-probability entropy reference with actual coded
   packets. The hard single-error scheme currently falls back everywhere;
   probability improvements alone cannot establish it will save bytes.

For each variant retain code histograms and escapes, within/between-page
covariance, temporal change rates, per-coordinate ablations, per-layer/K/V/
bit-plane losses, probability calibration, coset error histograms, and complete
packet component sizes. Export small raw code arrays rather than only hashes
and moments. Keep TRAIN/DEV/TEST provenance and frozen parameters.

Validate original BF16 equality from independently restored packets, shuffled
page order and continuation checks. Record capture, encoding, transfer and
restoration time separately; count producer/decoder resident state and peak
page buffers. Use the same concurrency and transport assumptions for controls.
If a variant does not beat its matched no-code and field baselines after all
costs, record the failure before optimizing its GPU kernel.

The most informative next combination is **decoder-known pooled context + a
two-bit actual-prefill innovation**, followed by the K-to-V variant. More
ambitious high-bit-first grouped refinement tests whether there is enough
joint structure to change the final KV rate materially. More elaborate
lattices, parity constraints or binning should follow evidence of
stronger conditional structure. The current known fields are available to
both ends, so ordinary conditional coding is a necessary control before
invoking decoder-only side-information theory.

## Sources and retained analysis

1. Mentzer et al., [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/html/2309.15505v2), §§3.1–3.3. Borrowed ingredient: a small discrete representation; no KV savings claim is transferred.
2. Ballé et al., [Variational image compression with a scale hyperprior](https://arxiv.org/html/1802.01436v2), introduction and §3. Borrowed ingredient: jointly account for side and primary code rates.
3. Shutova et al., [Cache Me If You Must: Adaptive Key-Value Quantization for Large Language Models](https://proceedings.mlr.press/v267/shutova25a.html), ICML 2025. Prior art for exploiting KV dependencies, with a different exactness target.

The [analysis bundle](../benchmark_results/compact_metadata_study_20261020.zip)
contains the encoder audit, selective-layer scenarios and research notes.
Its [manifest](../benchmark_results/compact_metadata_study_20261020/manifest.json)
records individual hashes and the distinction between measurements, diagnostic
projections and proposed designs. These notes do not establish research novelty
or a 40% reduction.
