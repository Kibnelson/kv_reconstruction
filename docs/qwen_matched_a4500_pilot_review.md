# Matched Qwen A4500 pilot: findings and repair

The pilot provides valid GPU results for the predictive codecs and native
SplitZip. It does not demonstrate an improvement to 51% compression savings.
The overall exit code is 1 because Bitcomp and Cascaded failed before producing
measurements, with `ValueError: Unknown output data type`.

[Original immutable report](https://github.com/Kibnelson/kv_reconstruction/blob/16f4db7/benchmark_results/qwen_matched_a4500_20261021_184512_1225465_report_20261021_195217/summary.md).
The original report and its failed baseline records remain unchanged.

## What completed

- Nine actual CUDA correctness tests passed in 1.785 seconds.
- 128 completed method/request cases recovered every original BF16 word exactly.
- All six predictive arms, raw BF16 and pinned native SplitZip completed on both
  WikiText-2 and C4. The four nvCOMP method/track attempts failed at encoding.
- The target was pinned BF16 Qwen3-4B-Instruct-2507, with the first four of its
  36 KV layers retained. Each track used two held-out 2,048-token contexts,
  sampled at the beginning and end with 32- and 128-token requests.
- Device: RTX A4500 SM86, Torch 2.11.0+cu128, Triton 3.6.0,
  Transformers 4.57.6, NumPy 2.3.5 and nvCOMP 4.2.0.14.

## Measured compression

Complete packet rates for 128-token requests, including common framing:

| Method | WikiText-2 bpv | Saved | C4 bpv | Saved |
|---|---:|---:|---:|---:|
| Block32 A, fixed widths | 9.42146 | 41.12% | 9.58343 | 40.10% |
| Block32 B, adapted widths | 9.54620 | 40.34% | 9.89429 | 38.16% |
| Previous-layer mean, B widths | 9.95897 | 37.76% | 9.96008 | 37.75% |
| AQUA-style mean, B widths | 9.80540 | 38.72% | 10.08652 | 36.96% |
| Strong layer inputs, group 32, B | 9.70033 | 39.37% | 10.16435 | 36.47% |
| Strong layer inputs, group 64, B | 9.75394 | 39.04% | 10.22751 | 36.08% |
| Native SplitZip | 12.31104 | 23.06% | 12.31767 | 23.01% |

Fixed-width A is the best measured rate in this run, and is also faster than B.
The earlier 44.67% used different calibration/workload data; it is not an
equivalent control for these held-out corpus tracks. The relevant feature
comparison is within this matched run. AQUA-style is an ablation in our exact
coder, not the published lossy AQUA-KV system.

## What the diagnostics suggest

**Width adaptation regressed independently of the mean.** Block32 A and B share
the identical frozen mean predictor. Their rate difference therefore comes
from the width rule. On the first 32-token WikiText-2 test page, B improves the
layer-0 ideal coding cost but worsens the other three layers. C4 shows the same
pattern. These are CPU diagnostic ideal costs, distinct from GPU packet rates.

**The calibration policy changed.** The earlier four-arm runner fitted means
on `base_train`, fitted B on disjoint `adapter_train`, then selected strength
on DEV. This matched pilot fitted B on the same selected TRAIN rows as its
mean and fixed strength to 4. Using in-sample residuals can underestimate
held-out prediction error and make the fitted widths too narrow. That is a
plausible explanation, not a verified complete cause. It is not TEST leakage.

**Added inputs also worsened some held-out means.** On the first 32-token test
page, the group-32 strong predictor's mean absolute errors increase in both K
and V for layers 2 and 3 on both tracks. For example, WikiText-2 layer-2 K MAE
goes from 0.36621 to 0.40066; C4 goes from 0.40814 to 0.44615. Layers 0 and 1
are identical between Block32 B and the strong group-32 arm, as their input
sets are identical there. These observations cover one diagnostic page per
track, not a full-corpus prediction-error estimate.

The mean fit used 2,048 selected rows out of 4,096 candidates. In Qwen's fourth
layer, the last strong group has 3 × 2,048 + 2,016 = **8,160 input features**.
This is a demanding fit for that sample size. More known context lowers the
ideal conditional entropy in principle; a finite-data linear fit can still
generalize worse. More independent calibration data, stronger regularization
or fewer predictor inputs need evaluation before promoting this arm.

## GPU latency is the larger systems gap

WikiText-2, 128-token requests, synchronized wall means of request medians:

| Method | Encode ms | Decode ms | Shared GPU state MiB |
|---|---:|---:|---:|
| Block32 A | 466.150 | 647.323 | 87.219 |
| Block32 B | 750.685 | 770.794 | 87.860 |
| Strong group 32 B | 804.903 | 825.187 | 138.860 |
| Strong group 64 B | 460.815 | 687.279 | 134.352 |
| Native SplitZip | 0.430 | 0.493 | 0.001 |

Group 64 is about 1.63× faster to encode than B, but offers essentially the
same encode latency as the better fixed-width A control, with a slower decode
and larger packets. It is not an overall improvement over A here. A's encode
and decode are about 1,083× and 1,314× slower than native SplitZip at this small
request size. Larger-buffer throughput and serving impact are not measured.

This prototype establishes a compact, exact representation. Its conversion
cost remains unsuitable for claiming a transfer or serving speedup. Kernel
fusion, launch reduction and the faster entropy format need work before a
full-layer repeated GPU run becomes the priority.

## Repair without another mean fit

The old nvCOMP pin's native dtype parser rejects valid `<i2`/`<f2` type strings
with NumPy 2.3.5. A CPU-only metadata probe reproduces that failure and accepts
the same strings with NumPy 1.26.4. No GPU data is processed in that probe.

The repair pins both `nvidia-nvcomp-cu12==5.3.0.16` and its separate
`nvidia-libnvcomp-cu12==5.3.0.16` runtime, preserving NumPy/Torch/Triton. Native
CUDA preflight tests exercise all 65,536 BF16 patterns before benchmarking.
The new SDK's actual A4500 execution still requires that preflight run.

The second comparison adds fixed-width A variants of the previous-layer,
AQUA-style and strong group-32/64 predictors. They use the existing base files
and stored per-channel widths; no mean fitting, prefill or B retraining occurs.
Base-file hashes are checked against the original fit manifests. This isolates
mean quality from B's width model. Reports compare against both A and B when
available. New output is written separately, preserving the original evidence.

```bash
git fetch origin
git pull --ff-only origin experiment/qwen_matched_gpu
bash scripts/repair_qwen_matched_a4500.sh \
  runs/qwen_matched_a4500_20261021_184512_1225465 --publish-results
```

After the fixed-width comparison, revisit B with out-of-sample residuals and
DEV-only strength selection, with zero adaptation as an allowed control.
Retain TEST for evaluation. More layers, attention features, prefill hints or
cosets should follow calibration and latency evidence rather than an assumed
monotonic improvement from adding information.
