# A4500 review: Block32 probability widths and paid hints

Arm B is the result to build on. Decoder-known context reduces complete warm
packets from **9.555740 to 8.828674 bits/value**, with the original Block32 mean
predictor unchanged. This is **7.6087% fewer compressed bytes**, or an increase
from 40.2766% to 44.8208% reduction relative to original BF16. The current KV
and actual-prefill hidden hints add no selected main-stream saving.

Results: [commit 69c5f6a](https://github.com/Kibnelson/kv_reconstruction/commit/69c5f6a699a6800e11dcc39614af1852cf91d167),
[evidence directory](https://github.com/Kibnelson/kv_reconstruction/tree/69c5f6a699a6800e11dcc39614af1852cf91d167/benchmark_results/block32_probability_qwen4_20261021_065346).
This review checks existing evidence and computes descriptive aggregates. It
does not run another model forward, GPU benchmark or entropy packet decode.

## Measured comparison and validation

| Arm | Width model | Complete packet bits/value | Bytes versus A | Shared CPU decoder arrays | Decode median seconds/page |
|---|---|---:|---:|---:|---:|
| A | Fixed per-channel width | 9.555740 | 0 | 163.735 MiB | 2.660680 |
| B | Width from causal decoded context | **8.828674** | **-190,596** | 164.375 MiB | 2.743100 |
| C | B plus 2-bit produced-KV hint | 8.829651 | -190,340 | 164.377 MiB | 2.758985 |
| D | B plus 2-bit actual-prefill-H hint | 8.829651 | -190,340 | 164.377 MiB | 2.758792 |

All arms use identical 132-byte page framing. B adds 671,744 CPU numeric state
bytes over A, with no extra packet payload. C and D each pay 256 hint bytes
over the eight TEST pages. State is excluded from the warm packet rates above.

The run records 32 successful arm/page checks over eight unique pages and
2,097,152 distinct original BF16 words. CPU and CUDA parameter digests, encoded
packet bytes, hint IDs and restored words match. Each measured packet also
passes a fresh-process decoder supplied only with baseline state, adapter state
and packet. There are zero raw fallback pages. Every A entropy stream matches
the legacy Block32 stream; the frozen baseline state remains unchanged.

This review independently verifies all 30 delivered file hashes/sizes, all nine
executed source hashes against this checkout, rate/component arithmetic and
recorded exactness/parity flags. The package's own 212 checks are recorded
evidence, not new GPU executions. Raw captures, baseline state and complete
packets are omitted, so those original decode claims cannot be independently
rerun from this evidence bundle alone.

The model is Qwen3-4B-Instruct-2507 at the pinned revision. Only the first four
of 36 layers are measured, using two existing synthetic TEST fixtures and four
32-token page positions in each 2K context. This is an exploratory comparison,
not a full-model or broad-workload rate. Earlier 9.430599-bits/value SmolLM
results have a different model and fitting protocol and are not its baseline.

## What the successful change means

The decoder can estimate how predictable a token is while reconstructing it.
Previously decoded errors describe whether later groups should use a narrow or
wide probability distribution. Exact previous-layer words provide additional
context. This changes uncertainty around the same predicted mean, rather than
improving the mean prediction or quantizing original KV values.

At fixed selected B, replacing real context by its TRAIN mean raises TEST ideal
cost by 0.9244 bits/value; rolling context across tokens raises it by 1.4321.
This supports a useful token-aligned signal. Neither is a separately refitted
best static-width baseline. Previous-layer-only and current-prefix-only width
models have not been isolated, so their separate contributions remain unknown.

B improves actual packet bytes on all eight pages. Text averages 9.730713 to
9.084808 bits/value; code averages 9.380768 to 8.572540. Layer-zero ideal gain
is about 1.1051 bits/value, compared with 0.6955, 0.5849 and 0.5230 in layers
one through three. Layer zero lacks previous-layer features; its benefit is
compatible with prefix context and static adjustment, which still need a
matched ablation.

## Opportunities for further reduction

### 1. Test the width limit and a stronger no-hint model first

Among B's token/group decisions, **29.91%** use the minimum width, one quarter
of the original calibrated width; only 0.14% use the maximum. In the first
32-token pages, 70.94% hit the minimum. This makes a lower limit of 1/8 or 1/16
a concrete hypothesis. Hitting a bound does not establish that removing it
improves actual code length: outliers can make narrower widths expensive.

Fit candidates on TRAIN and select the range using complete DEV packets. Also
compare width fitting against exact BF16 interval coding cost. Current fitting
regresses a log mean-absolute-residual surrogate, which need not minimize the
actual finite-precision probability code. These are proposed changes, not
measured new rates.

The first controlled comparison should hold means, framing, data and fitting
objective fixed and compare: refitted static widths; previous-layer context
only; already-decoded current-layer error context only; and both. Then test
the selected no-hint model's width range as a separate step. This establishes
which dependency contributes before attributing gains to finer conditioning.

### 2. Preserve position effects in evaluation

| Sampled page start | A packet bits/value | B packet bits/value | Saved bits/value |
|---|---:|---:|---:|
| 0 | 9.128235 | 7.690674 | 1.437561 |
| 512 | 9.801697 | 9.370056 | 0.431641 |
| 1024 | 9.695740 | 9.185303 | 0.510437 |
| 2016 | 9.597290 | 9.068665 | 0.528625 |

The first pages contribute 49.43% of measured byte savings and 25% of sampled
values. Those first 32 tokens are only 1.5625% of each complete 2K context.
The six noninitial pages still improve by 0.490234 bits/value on average.
Report position strata and use complete contexts or representative page
sampling before extrapolating a whole-cache reduction. New experiments should
freeze choices before fresh TEST prompts; repeated tuning on these fixtures
would no longer be a final holdout evaluation.

### 3. Revisit cosets only as a targeted feasibility test

The group-local 31-bit single-error screen is highly position-dependent:

| Position stratum | Eligible groups |
|---|---:|
| First 32 tokens | **13.7488%** |
| Start 512 | 0.3433% |
| Start 1024 | 0.4794% |
| Start 2016 | 0.4794% |
| All sampled groups | 3.7627% |

The basic five-syndrome-bit plus one-mode-bit comparison against 31 raw bits
requires more than 3.8462% eligibility, before extra framing. Thus an
everywhere-on scheme is unsupported, while a position-specific feasibility
screen merits investigation. K aggregates also have higher eligibility than
V: K layers range from 4.11% to 7.10% in this sample, versus 0.01% to 3.79%
for V. These descriptive TEST patterns cannot select a deployed gate.

The screen uses a different group-local layout from the earlier syndrome
experiments. It constructs no correction packet. A new scheme must beat B's
already entropy-coded mantissa costs, not merely a raw-bit reference, and must
pay all flags, syndrome bits and state. B does not change the mean: bit-6 hard
prediction error remains 38.4461% in every arm. Better uncertainty modeling
therefore does not itself make the old hard coset decoder succeed.

### 4. Diagnose hints before investing in larger extractors

DEV selects hint strength zero for C and D. Their selected main streams are
identical to B; the 256-byte TEST loss is entirely the hint payload. All four
hint IDs are occupied, so this is not simply a collapsed single-ID encoder.

With full strength forced on DEV, C adds 6,464 main-stream bytes versus B and
D adds 5,832, before paying hints. Actual IDs beat rolled IDs by 1,908 and
2,160 bytes within those same forced models. This indicates some alignment
sensitivity, but insufficiently useful calibration; it is not an encoded
advantage over B. Forced TEST ideal costs also worsen relative to B by 0.0786
and 0.0985 bits/value respectively. Those are counterfactual ideal costs, not
new measured TEST packet rates.

Both extractors fit on only one structured 512-token adapter TRAIN capture.
A single scalar readout of a 16-dimensional random sketch targets average
scale innovation across all 64 groups. Possible limitations include workload
shift, the surrogate objective and averaging away group-specific information.
The evidence does not distinguish these explanations. If hints are revisited,
use broader TRAIN data and target exact coding benefit remaining after the
stronger free-context model. A hypothesis focused on early K groups is more
specific than another global hidden code; it still needs TRAIN/DEV evidence.
There is no demonstrated special advantage for actual H over produced KV.

## GPU and memory implications

The A4500 measured genuine causal packet decoding with GPU integer parameter
work. It is still a hybrid: interval probabilities and entropy coding run on
CPU, and contexts/parameters cross devices for each group.

B's median decode time is 2.7431 seconds per 0.5-MiB raw page, 3.10% above A.
Averaging all warm B samples gives 2.7428 seconds: 2.2008 in CPU probability
and entropy work (80.24%), 0.4339 in parameter work/staging (15.82%), and
0.1000 in CPU context work (3.65%). The remaining fraction is other overhead.
Moving only the predictor to GPU cannot remove the dominant CPU cost.

The base predictor performs about 2,544 MACs per original value, or 667 million
per page. B's persistent GPU parameter tensors occupy 86.61 MiB, of which
the new width coefficients occupy only 16 KiB. Its CPU decoder numeric arrays
occupy 164.38 MiB. CPU and GPU state are distinct physical allocations; do not
add them to a packet-rate percentage and call that measured GPU capacity.

The large dense mean predictor is the main state/scaling concern. The GPU
already stores its coefficients as INT32; merely repeating that conversion is
not another GPU gain. CPU INT32 storage and smaller structured predictors
remain separate engineering comparisons, with re-encoded packet rates and
exact reconstruction checked. A changed mean can preserve exactness through
entropy coding while improving or worsening rate.

No GPU entropy decoder, compressed-resident attention, migration speedup or
serving-capacity gain is demonstrated. Those require separate implementation
and measurements after a useful rate/state/compute tradeoff is established.

## Recommended next task

Implement only the matched static/previous-layer/prefix/combined width
comparison first. Keep the current means and packet accounting, use broader
TRAIN/DEV coverage, and retain a fresh TEST set. Report complete packet rates,
state, source identities and exact decoder checks. Stop and review those
results before changing the width range, implementing targeted cosets or
redesigning prefill hints.
