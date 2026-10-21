# Exact GPU KV: research arms and conversion-cost contract

Date: 2026-10-21. Status: experimental design, not new measurements or a
novelty conclusion. The [lifecycle roadmap](gpu_kv_transfer_pool_roadmap.md)
remains the architecture: T is canonical in DRAM/SSD and in transit; P is the
active GPU representation. Complete and report one implementation task before
proceeding with the user's approval.

## Research question

Can information available while KV is generated be distilled into a small,
paid code that reduces exact storage cost, GPU reconstruction work or decoder
state beyond what produced KV and decoder-known context can achieve?

A prefill hint is not automatically a contribution. Predictor-based exact KV
coding already exists, and paid side information for entropy models is a
known coding mechanism. A positive result needs a specific new dependency,
extractor, access schedule or cost tradeoff and a comparison with close prior
methods. Full-method novelty assessment follows a positive candidate; adding
an ingredient does not establish priority.

The concrete candidate is a producer-generated compact hint that helps a
cheap exact GPU decoder, including conversion from external T into a resident
pool. Its distinguishing claim must be measured, not assumed. A hint can be
valuable through either fewer stored bytes or less GPU work/state at a useful
byte rate. Those are separate outcomes.

## Existing evidence constrains the next experiments

- B currently wins at 8.828674 complete packet bits/value in the four-layer
  Qwen pilot; its original mean predictor is unchanged. That entropy codec is
  CPU-only, with GPU parameter computation.
- Current C/D hints select zero strength on DEV. Forced hints worsen the
  main stream even before paying hint bytes. A new run must change a declared
  limitation rather than repeat that fitted scalar surrogate unchanged.
- The group-local hard-bit screen is promising near the start of a context,
  but constructs no syndrome packet and is not equivalent to the older coset
  layout. It cannot be called a new compression saving.
- T-to-P GPU installation and compressed-resident attention are not yet
  implemented. Current predictor timings cannot substitute for those costs.

## Candidate arms, tested separately

| Arm | Hypothesis and implementation change | Required matched controls | Success evidence |
|---|---|---|---|
| R0: stronger free context | Separate refitted static widths, previous-layer context, current-prefix error context and both; later test width range and exact interval-cost fitting separately | Same words, means, framing, fitting budget and GPU coder | Frozen TEST packet improvement with state/latency; establish which context contributes |
| R1: rate-directed producer hint | Start with two total bits per token/layer. Fit the paid code to exact conditional coding benefit after R0, with broader TRAIN coverage rather than the old global log-residual surrogate | No hint, produced-KV hint and actual-H hint with equal transmitted budget and decoder capacity; alignment sensitivity | Complete bytes saved after hints/framing/state; actual GPU hook and conversion cost |
| R2: hint buys a simpler decoder | Use a paid code to choose a cheaper probability/mean model or restart configuration, reducing dense predictor input/state or sequential dependencies | The same cheaper decoder without hints and with equally budgeted KV/H hints; full R0 reference | Measured rate-versus-T-to-P-time/state frontier, exact words and packet-only decode; no assertion that two bits can replace a full predictor |
| R3: selective syndrome/coset | Test a TRAIN/DEV-defined gate where a joint correction code can beat the actual conditional entropy stream; use prefix/K patterns only as motivation | R0 exact entropy coder on identical fields; raw fallback and all mode/position/syndrome costs paid | Actual encoded packets, positive complete net gain and GPU correction cost; an eligibility histogram is insufficient |
| R4: joint model or restart tradeoff | Model residual dependencies missed by scalar widths, or pay small restart anchors for more GPU coder lanes | Same marginal baseline, plus same-joint factorization as a negative control when relevant | New modeled dependency or useful measured parallelism; include lane directories, anchors, padding, state and launches |

Do not combine hint, coset, width and predictor changes in the first comparison.
TRAIN fits candidates, DEV selects settings and gates, and fresh TEST data
evaluates the frozen configuration. The current reused fixture suite is useful
for diagnostics but cannot remain a final holdout after iterative tuning.

R2 is a potentially useful addition to the earlier rate-only research:
producer work might replace expensive decoder work rather than only shrinking
the stream. For example, a slightly larger T could still be useful if it uses
much less decoder state and restores faster within the declared serving
budget. Report that tradeoff; do not label it a rate improvement.

## Information accounting

For any candidate, account for complete T length:

`L(T) = L(paid hints) + L(original-word/correction payload) + L(framing, indexes and restarts)`.

The payload fields are disjoint: a syndrome that replaces a field is not added
to the original coding cost of that same field. Shared encoder and decoder
state are charged separately, including cold setup and their amortization.

Decoder-known context must actually be available at the coding step. Unknown
channels or hidden tensors cannot be used as free inputs. The hint is a paid
input, and the decoder must reconstruct the original BF16 words using only
the packet, declared frozen state and earlier recovered words/fields.

Transmitting side information cannot be assumed to beat an ideal joint model.
The practical opportunity is a better tractable distribution or cheaper
execution. Likewise, merely factoring an unchanged joint distribution into
residue and quotient does not create ideal entropy savings. A proposed coset
must introduce useful joint/side-information structure or a measured execution
benefit beyond that bookkeeping control.

## Producer-hint lifetime

The lifecycle has to support encoding T long after prefill. At that time P
contains exact KV but the original hidden input may no longer exist.

The selected initial policy for an H-hint arm is:

1. Generate its small packed ID during the actual KV-producing forward and
   record the encoder identity. Do this for both prefill and later generated
   tokens whose KV may eventually be offloaded.
2. Retain those IDs alongside the active P pages until they can be written
   into T. Count the ID sidecar and any retained encoder state in GPU memory.
3. Store the IDs inside the T object in DRAM/SSD. T-to-P decoding reads them
   from T; it receives neither H nor a predictor-model rerun on the prompt.
4. If existing external T backing is reused, avoid re-encoding unchanged
   pages. If a hint-bearing page must be re-encoded after restore, preserve
   its original small IDs as necessary. Validate code/encoder identity.

Do not retain full hidden tensors to support later offload. If no hint was
captured, the implementation must use an explicitly supported no-hint mode
or report that the requested variant is unavailable. It must not silently
reconstruct H from KV or treat missing H as a zero-cost encoder input.

A KV-hint control can derive its code from produced or unpacked KV. Measure
the actual path it uses: generating a code during prefill and generating it
while evicting P have different costs. Equal bits do not imply equal encoder
cost or state.

## Mandatory timing boundaries

All codec/conversion rows below are GPU operations in the target. CPU storage
I/O and transport coordination have their own wall-time fields. Until a GPU
stage exists, mark it **not implemented**, rather than filling it with CPU
codec time or GPU predictor replay time.

| Measurement | Start and completion boundary | Included costs |
|---|---|---|
| Prefill/step hint overhead | Paired actual model forwards with and without the hint path; all output work complete | Hook, feature extraction, projection/quantization, ID write and extra synchronization. Report prefill and generation steps separately |
| KV to T | Original GPU cache words ready to complete T chunk in GPU memory | Layout conversion, predictor/context work, hint handling, probability construction, entropy/correction coding, framing, indexes and integrity work |
| KV to P | Original GPU words ready to published P page | Escape count/rank, allocation, packing, raw fallback, descriptor publication and scratch |
| P to T | P pages ready to complete T chunk | Pool reads/unpack plus the full encoder path; clean-backing reuse is a separate zero-reencode case |
| T to KV | T ready on GPU to exact diagnostic BF16 output | GPU parsing/probabilities/entropy, causal contexts, corrections, output buffers and integrity checks |
| T to P | T ready on GPU to P pages safe for attention | Full T decode, bounded word buffers, pool packing/allocation and readiness synchronization. Measure a fused bridge directly |
| Pool attention | Q and valid pool/tail descriptors ready to attention output complete | Tile reconstruction, escape handling, masks, attention math and per-call workspace |
| External I/O | DRAM copy or SSD read/write request to specified completion | Complete T bytes and indexes, staging/pinned buffers, transport waits; record SSD page-cache conditions and write-persistence policy |
| Full offload/restore | Offload request to safe external copy; restore request to first usable attention/token | All codec, I/O, queues, allocation, state setup and waits on the real path |

For fresh KV, measure both direct KV-to-T and KV-to-P where implemented. For
eviction, measure P-to-T. Do not substitute direct KV-to-T timing for eviction
when the actual encoder must first read the pool.

Measure T-to-KV and KV-to-P components for diagnosis, but do not claim their
isolated medians equal fused T-to-P. Likewise, a sum of isolated kernel times
does not measure an overlapped pipeline. Define the completion event across
all participating streams and report synchronized wall time as well as GPU
events. Report cold setup separately from warm state; do not silently remove
codec-specific setup or allocator costs from an end-to-end measurement.

Use enough repeated samples to report a distribution and repeatability.
Three warm samples are a pilot, not a reliable tail-latency characterization.
Label warm SSD page-cache hits separately from physical SSD reads. Include
both latency and throughput, using original BF16 bytes for comparable codec
throughput and actual compressed bytes for transport bandwidth.

## Decision rule and evidence bundle

For each frozen candidate, report a row containing complete T bits/value,
hint/correction bytes, P allocated bits/value, encoder/decoder/sidecar state,
peak VRAM, KV-to-T, P-to-T, T-to-P, pool-attention time and end-to-end cold
restore time. Record device, format/state/source identities, model/layer scope,
chunk geometry, workload/positions, and every applicable exactness check.

Use two distinct acceptance questions:

- Does the candidate save complete stored/wire bytes after its paid code,
  framing and state at the declared amortization scenario?
- Does it improve a measured latency/capacity/communication tradeoff within
  the declared serving budget, including hook work and hint lifetime?

An apparent hint benefit at selected strength zero is not a positive hint
result. Forced-strength diagnostics remain diagnostics. A decoder that uses
original KV/H or an undeclared previous chunk does not establish independent
restoration. Compare loaded BF16 words exactly; validate attention outputs and
continuation under separately declared arithmetic criteria.

The immediate roadmap task remains GPU P pages and tile unpacking. It must
include KV-to-P, tile-unpack and allocation/scratch measurements from the
start. This research matrix governs later optimizations after the GPU path is
available; it does not authorize silently replacing an unimplemented stage
with the current CPU reference.

## Precedent and claim boundaries

- [Speculative KV coding](https://fergusfinn.com/blog/kv-entropy-coder/)
  already describes predictor-based exact entropy coding, with a predictor
  forward at both ends. Our proposed producer-only packed code avoids that
  particular rerun but is not thereby established as a first.
- [AQUA-KV](https://arxiv.org/html/2501.19392v4)
  already uses linear dependencies across layers and K/V, with quantized
  residual reconstruction. Its predictor precedent must remain in matched
  method comparisons.
- [A scale hyperprior](https://arxiv.org/abs/1802.01436)
  is precedent for paid distribution side information in learned image
  compression. That image codec is lossy; it does not establish or refute our
  precise exact-BF16 GPU pipeline. The generic side-information ingredient
  cannot carry our novelty claim.

These targeted references are not an exhaustive novelty search. Keep claims
as hypotheses until a positive measured mechanism and full-method comparison
support them.
