# PD matrix quick on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 1 runs: `llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261122_2228`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory | ours-hbh | 4/4 | 3/4 | yes | 9.110 |
| 1p3d, host memory | t2-hbh | 4/4 | 3/4 | yes | 8.562 |
| 1p3d, host memory | t2-rank256 | 4/4 | 4/4 | yes | 9.619 |
| 1p3d, host memory | splitzip | 4/4 | 3/4 | yes | 12.043 |

## Where compression helps and where it does not

Per layout and link speed (and its share per prefill GPU when the engines share one link), per codec: the workloads (stage shapes) where its TTFT p50 beats raw's by more than 3% (helps), is within 3% (same) or is slower (hurts), as helps / same / hurts, then its speedup over raw (raw's TTFT / the codec's) from its worst workload to its best.

| Run | Link | Per prefill GPU | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory | 10 Gb/s | 10 Gb/s | 3 / 0 / 0, 1.11-1.24x | 2 / 0 / 1, 0.88-1.25x | 3 / 0 / 0, 1.13-1.22x | 3 / 0 / 0, 1.10-1.14x |
| 1p3d, host memory | 100 Gb/s | 100 Gb/s | 1 / 1 / 1, 0.86-1.03x | 0 / 1 / 2, 0.83-0.99x | 1 / 1 / 1, 0.91-1.09x | 1 / 1 / 1, 0.97-1.11x |

## Published workloads: 1p3d, host memory, link 10 Gb/s

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer over the GPU budget while others were held), which costs copies the others did not pay.

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | 394 | 356 (1.11x) | 446 (0.88x) | **349** (1.13x) | 359 (1.10x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 3536 (host) | 2846 (1.24x) | **2837** (1.25x) | 2894 (1.22x) (host) | 3101 (1.14x) (host) |
| SplitZip, 16 at once | 2048 x 16, 128 | 4279 (host) | **3525** (1.21x) (host) | 3827 (1.12x) (host) | 3597 (1.19x) (host) | 3826 (1.12x) (host) |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | 0.55 | 0.56 (1.02x) | 0.53 (0.96x) | **0.56** (1.03x) | 0.56 (1.02x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.19 (host) | 0.22 (1.15x) | **0.22** (1.15x) | 0.22 (1.14x) (host) | 0.21 (1.09x) (host) |
| SplitZip, 16 at once | 2048 x 16, 128 | 2.28 (host) | **2.77** (1.21x) (host) | 2.65 (1.16x) (host) | 2.74 (1.20x) (host) | 2.59 (1.13x) (host) |

## Published workloads: 1p3d, host memory, link 100 Gb/s

Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, "Workloads"), for every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer over the GPU budget while others were held), which costs copies the others did not pay.

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | **225** | 262 (0.86x) | 272 (0.83x) | 249 (0.91x) | 233 (0.97x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 2326 (host) | 2255 (1.03x) | 2349 (0.99x) | 2138 (1.09x) | **2097** (1.11x) |
| SplitZip, 16 at once | 2048 x 16, 128 | 2745 (host) | 2808 (0.98x) | 2978 (0.92x) | 2807 (0.98x) | **2688** (1.02x) (host) |

Requests per second, and in brackets the gain over raw:

| Workload | Prompt tokens x at once, out | raw | ours-hbh | t2-hbh | t2-rank256 | splitzip |
| --- | --- | --- | --- | --- | --- | --- |
| SplitZip, 1 at a time | 2048 x 1, 128 | **0.61** | 0.59 (0.98x) | 0.58 (0.96x) | 0.60 (0.99x) | 0.60 (1.00x) |
| SplitZip, 1 at a time | 16384 x 1, 128 | 0.25 (host) | 0.25 (1.02x) | 0.25 (0.99x) | 0.26 (1.05x) | **0.26** (1.06x) |
| SplitZip, 16 at once | 2048 x 16, 128 | 3.48 (host) | 3.46 (0.99x) | 3.19 (0.92x) | 3.46 (0.99x) | **3.60** (1.03x) (host) |

