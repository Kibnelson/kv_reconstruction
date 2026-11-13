# Checked A4500 speed results and the original-dense capture failure

Uploaded result commit: `b2dae7bc2446ad40b3333b322fc37e7bb7daeb39`.
Tested code commit: `cb72808b53af3faafcecf7f19cedc7611dc27b6b`.
Model: frozen Llama-3.2-3B, 28 layers, four WikiText-2 windows of 8,192 tokens.
GPU: RTX A4500; Torch 2.11.0+cu128; Triton 3.6.0.

## Validation of the uploaded archive

The fetched ZIP and all four members match their GitHub blob hashes. The ZIP
SHA-256 is `7b2391338388f1ec737dc57eaeb46666e5126470bc148ad684409dbf1a5d3faa`.

- All 108 successful GPU timing rows have the exact earlier reference packet
  SHA-256 and byte count, and every direct word comparison passed.
- All 20 fresh decoder processes reconstructed the original words using only
  the frozen state and packet. No source KV was given to those decoders.
- All 48 graph rows passed the changed-input and restoration test.
- The new dense preflight passed over all 65,536 BF16 word patterns.
- All 12 capture errors belong to `dense_original`. No other method's graph
  failed. The original T source and measured experimental source hashes match
  the versions reported by the NUC.

## Measured comparison

| Implementation | Saved from BF16 | Encode ms | Decode ms | Sequential pair ms |
|---|---:|---:|---:|---:|
| Original dense, eager | 44.0484% | 87.475 | 36.827 | 124.493 |
| cuBLAS dense, eager | 44.0484% | 63.657 | 36.758 | 100.686 |
| cuBLAS dense, graph | 44.0484% | 63.423 | 36.633 | 100.433 |
| Learned head-by-head, eager | 48.5682% | 112.581 | 195.575 | 315.898 |
| Learned head-by-head, graph | 48.5682% | 112.426 | 193.348 | 313.333 |

The dense implementation took 27.228% less encode time and 19.123% less pair
time than the unchanged original implementation. The head-by-head graph pair
improved by only 0.812% and remained approximately 3.12 times the dense graph
pair time. Graph replay alone has not made the 48.5682% point competitive with
the faster dense codec on this workload.

Times cover warm codec work with resident inputs. They exclude the report
packet packing/D2H path, whose median was approximately 1.0-1.1 seconds per
window. The ideal serial link table in `summary.md` is a model, not measured
network or serving latency. With the measured pair costs and payload sizes,
the dense/head-by-head crossover is approximately 1.6 Gbps; dense is faster
throughout the modeled 3-400 Gbps range. Overlap, queueing, storage and the host
packet path can change a deployed system's result.

## Capture cause and required control

`LayerwiseOneStep.model` constructs a fresh `FusedOneStep` each time it encodes
dense KV. `FusedOneStep.__init__` calls `gpu_coder.slice_tables(device)`, which
builds static probability tables on the CPU and uses `torch.as_tensor(...,
device=device)` to transfer them. This setup occurs inside the attempted graph
capture. The reported unpinned CPU-to-CUDA copy error matches that path.

This also means some of the original dense per-call cost is setup work. The
27% measured implementation improvement cannot yet be attributed entirely to
the cuBLAS forward computation. A cached-original control is needed before
making that claim.

The follow-up adds `CachedFusedDense`, which constructs the same fused forward
object once and reuses its unmodified `model` method. It keeps the original
entropy encoder, decoder and compressed bytes. The unchanged original remains
eager-only, and its known capture omission is explicit. Both caching-only and
cuBLAS improvements get separate comparisons. New-control GPU timings remain
pending; no old measurements are replaced and no merge is automatic.

See [the focused test command](../../docs/t_frozen_speed_comparison.md).
