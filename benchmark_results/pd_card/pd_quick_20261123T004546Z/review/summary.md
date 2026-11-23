# PD matrix quick on gpu-grad-01, job 30563 (completed with failures), reviewed from the kept results

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 1 runs: `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261122_1959_partial`.

Runs that did not finish: `TRANSPORT=ring LINKS=10,100 CODECS=raw,ours-hbh,t2-hbh,t2-rank256,splitzip LAYOUT=1p3d STAGES=check,shapes,report SHAPES=2048:1:128:4:sz1,16384:1:128:2:sz1,2048:16:128:1:sz16 RUN_TIMEOUT_S=5400`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | raw | 4/4 | 3/4 | yes | 16.000 |
| 1p3d, host memory | ours-hbh | 2/4 | 3/4 | NO | 9.110 |
| 1p3d, host memory | t2-hbh | 2/4 | 3/4 | NO | 8.562 |
| 1p3d, host memory | t2-rank256 | 2/4 | 3/4 | NO | 9.619 |
| 1p3d, host memory | splitzip | 4/4 | 3/4 | yes | 12.043 |

## Where compression helps and where it does not

Per layout and link speed (and its share per prefill GPU when the engines share one link), per codec: the workloads (stage shapes) where its TTFT p50 beats raw's by more than 3% (helps), is within 3% (same) or is slower (hurts), as helps / same / hurts, then its speedup over raw (raw's TTFT / the codec's) from its worst workload to its best.

| Run | Link | Per prefill GPU | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | 10 Gb/s | 10 Gb/s | 2 / 0 / 1, 0.76-1.22x | 1 / 0 / 2, 0.76-1.12x | 2 / 0 / 1, 0.75-1.22x | 3 / 0 / 0, 1.09-1.13x |
| 1p3d, host memory | 100 Gb/s | 100 Gb/s | 0 / 1 / 2, 0.59-0.98x | 0 / 0 / 3, 0.58-0.91x | 0 / 0 / 3, 0.59-0.97x | 1 / 1 / 1, 0.96-1.12x |

## Published workloads: 1p3d, host memory, link 10 Gb/s

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer over the GPU budget while others were held), which costs copies the others did not pay.

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | 394 | 352 (1.12x) | 501 (0.79x) | **346** (1.14x) | 361 (1.09x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 3531 (host) | 4632 (0.76x) (failed) | 4630 (0.76x) (failed) | 4729 (0.75x) (failed) (host) | **3115** (1.13x) (host) |
| SplitZip, 16 at once | 2048 x 16, 128 | 4268 (host) | **3489** (1.22x) (host) | 3822 (1.12x) (host) | 3505 (1.22x) (host) | 3851 (1.11x) (host) |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | 0.55 | 0.56 (1.02x) | 0.52 (0.95x) | **0.57** (1.03x) | 0.56 (1.02x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.19 (host) | 0.16 (0.83x) (failed) | 0.16 (0.83x) (failed) | 0.15 (0.81x) (failed) (host) | **0.21** (1.09x) (host) |
| SplitZip, 16 at once | 2048 x 16, 128 | 2.32 (host) | **2.82** (1.22x) (host) | 2.64 (1.14x) (host) | 2.79 (1.21x) (host) | 2.58 (1.11x) (host) |

## Published workloads: 1p3d, host memory, link 100 Gb/s

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer over the GPU budget while others were held), which costs copies the others did not pay.

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | **224** | 259 (0.87x) | 271 (0.83x) | 251 (0.90x) | 234 (0.96x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 2330 (host) | 3954 (0.59x) (failed) | 4040 (0.58x) (failed) | 3972 (0.59x) (failed) | **2078** (1.12x) |
| SplitZip, 16 at once | 2048 x 16, 128 | 2735 (host) | 2794 (0.98x) | 3005 (0.91x) | 2829 (0.97x) | **2696** (1.01x) |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | **0.61** | 0.59 (0.98x) | 0.59 (0.97x) | 0.60 (0.98x) | 0.61 (1.00x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.25 (host) | 0.18 (0.71x) (failed) | 0.17 (0.70x) (failed) | 0.18 (0.71x) (failed) | **0.26** (1.07x) |
| SplitZip, 16 at once | 2048 x 16, 128 | 3.49 (host) | 3.47 (0.99x) | 3.19 (0.92x) | 3.42 (0.98x) | **3.63** (1.04x) |

