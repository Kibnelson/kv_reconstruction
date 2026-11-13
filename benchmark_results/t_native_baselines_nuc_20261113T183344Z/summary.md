# Frozen T versus native exact GPU baselines

Status: **completed**. Full reference coverage: **True**. No automatic merge.

| Method | BF16 saved | bpv | Encode CUDA ms | Decode CUDA ms | Pair CUDA ms | Pair wall ms | Complete |
|---|---:|---:|---:|---:|---:|---:|---|
| raw_bf16 | -0.0000% | 16.000006 | 0.006 | 3.306 | 3.305 | 3.317 | True |
| t_fast | 44.0484% | 8.952261 | 61.903 | 35.915 | 98.337 | 98.360 | True |
| t_hbh | 48.5682% | 8.229093 | 110.559 | 194.164 | 312.090 | 312.118 | True |
| splitzip | 24.6863% | 12.050196 | 6.695 | 3.046 | 9.672 | 9.686 | True |
| exans | 32.0454% | 10.872737 | 172.832 | 17.828 | 190.693 | 190.715 | True |

CUDA and wall times use warm native API calls with resident inputs; medians across samples and rounds per window, then mean across windows. Pair is a direct sequential encode+decode measurement. Authors' host synchronizations remain included. Each method retains its native allocation behavior, recorded in results.json.

Complete packet bytes include headers, indexes, every raw fallback block and the partial-block tail. Original T packets must match the earlier experiment byte for byte. Decode is checked against every BF16 word, including in a fresh process with only packet and shared state. Shared state, native workspace, input staging, packet packing/D2H and expansion/H2D are recorded separately. Setup and host paths are not included in warm codec times.

Raw BF16 is a resident reference copy; network transport is not measured. Ideal link times in results.json use wall codec pair plus packet bytes / bandwidth, with shared state resident and no overlap. SplitZip and ExANS are authors' pinned GPU implementations. ExANS adds paid raw handling for rejected and partial blocks; its existing one-line build fix is recorded.

AQUA-KV is a separate lossy quality stage requiring its authors' environment and actual packed buffers; this exact-BF16 table makes no AQUA-KV performance or quality claim.
