# Llama-3.2-3B official GPU evaluation

Status: **failed**. GPU measured: **True**.

Model: `unsloth/Llama-3.2-3B` at `d4446454d87d51aa42e1fb174f25acc5f8762331`; original BF16, all 28 layers.

| Stage | Status | Detail |
|---|---|---|
| hardware | completed | actual CUDA required before downloading model weights or installing baseline environments |
| model | completed |  |
| toolchain | completed |  |
| splitzip_env | completed |  |
| splitzip_preflight | completed | actual CUDA kernel and full WikiText2 text/token identity; no codec performance claim |
| capture | completed | all 28 original BF16 layers; TRAIN/validation/TEST separated |
| splitzip | completed | unchanged author benchmark; two identical-to-ours TEST word slices; native codec byte/timing scope |
| ours | completed | fresh Block32 A fit and exact packet restore, all 28 layers; no AQUA adapter |
| exans | blocked | exans_build_official exited 2; see logs/exans_build_official.log |
| aqua_env | completed |  |
| aqua_preflight | completed | actual CUDA kernel and full WikiText2 text/token identity; no codec performance claim |
| aqua | failed | aqua_train_official exited 1; see logs/aqua_train_official.log |

Official SplitZip native codec results:

| Case | Native bpv | Saved | Encode GB/s | Decode GB/s |
|---|---:|---:|---:|---:|
| splitzip_case_0 | 12.04466 | 24.72% | 136.808 | 311.363 |
| splitzip_case_1 | 12.05049 | 24.68% | 136.815 | 309.569 |

Our transfer codec results (separate packet/timing scope):

| Method | Start | Payload bpv | Complete bpv | Encode wall ms | Decode wall ms | Exact |
|---|---:|---:|---:|---:|---:|---|
| raw_bf16 | 0 | 16.00000 | 16.00006 | 0.497 | 0.513 | True |
| raw_bf16 | 7168 | 16.00000 | 16.00006 | 0.494 | 0.499 | True |
| block32_A | 0 | 8.90887 | 8.90973 | 26746.051 | 37227.728 | True |
| block32_A | 7168 | 8.80361 | 8.80448 | 26801.538 | 37454.222 | True |

Official SplitZip results are native codec metrics. Our results include separate payload and framed bytes, synchronized wall samples and CUDA intervals.
AQUA PPL, native codec throughput and our packet latency have different scopes; no cross-scope speed ranking is produced.
Blocked stages have no substituted, synthetic, CPU-estimated or adapted performance results.
This base-model test does not measure LongBench, compressed attention, T-to-P conversion, SSD/network or serving throughput.
