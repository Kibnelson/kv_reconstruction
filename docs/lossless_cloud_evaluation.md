# Lossless baselines and cloud evaluation plan

Checked 13 November 2026. This extends the earlier `lossless_related_work.md` audit. It is a researched specification and readiness inventory, not a built cloud image or a completed cloud experiment. The new pinned transfer-path NUC test remains pending; its code and all existing measurements are preserved.

## Baselines to cover

The latest full A4500 run contains raw BF16, Fast T, Compact T, SplitZip and ExANS. It is a validated comparison of those implementations and inputs, not the entire lossless literature. More adapters already exist in `kv_compare/more_baselines.py`; having an adapter does not mean it was measured in the latest run.

| Method | Relevant experiment and hardware | Role in our evaluation | Current status |
|---|---|---|---|
| SplitZip | H200; dual Xeon Platinum 8468; real BF16 Qwen3-32B KV; separate codec, Mooncake transfer and SGLang serving tests | Direct fresh P/D transfer baseline | Measured on A4500; public codec available |
| ExANS | H100, external KV snapshot with 256K context; independent-block and aligned-batch modes; overlapped reconstruction described | Transfer/retrieval baseline | Measured pinned API on A4500; exact blog reproduction unavailable |
| AdaCode | A800 80GB; Triton/Hugging Face; batch 1 by default; cache capacity, reconstruction and inference evaluation | Direct BF16 KV codec | Existing adapter; absent from latest five-method run |
| IBP | A100; 300GB host RAM; PCIe Gen4 x16; CUDA 11.7, Python 3.8, Torch 1.13.1; compressed host fetch | PCIe offload/retrieval baseline | Existing optional adapter; absent from latest run |
| nvCOMP LZ4/Cascaded/Bitcomp/ANS/Zstd | General GPU codecs; SplitZip includes several in its H200 KV comparison | Generic GPU controls | Existing adapters; actual byte/element modes and version must be audited |
| LCT | General bit-exact BF16 tensor compression for repeated training updates; released code requires Python 3.14 | Additional GPU BF16 codec candidate | Public author code found; full paper hardware/timing audit and adapter pending |
| DFloat11, ZipServ, Falcon, ZipNN | SplitZip's paper includes these related codecs; several originated as weight/float compression systems | Secondary codec comparisons on the same KV, where their APIs support it | Need validated actual-buffer adapters; CPU encode or offline work must be identified |
| exactkv | RTX 4090; lossless page format and reconstruction inside attention | Resident compressed KV / P track; page codec can also be evaluated separately | Existing page codec adapter; not the latest transfer run |
| ZipFusion | H100 80GB; CUDA 13.0, Torch 2.11.0, FlashInfer 0.6.12; SGLang with 16-token pages | Resident attention/serving comparison | Paper verified; public implementation and framework revision not verified |

DietGPU and NeuZip belong in the secondary artifact/dtype audit as well. This is a scoped baseline set, not a claim that no other lossless implementation exists. A candidate must preserve the original BF16 words and supply an actual runnable codec before it enters a measured table. A computed entropy/Huffman floor is a bound, not an implementation result. Quantization, pruning, or lossless packing after a lossy conversion belong in the separate quality track.

The role distinction matters: ZipFusion measures attention that reads compressed KV directly, while a bulk transfer decoder reconstructs a tensor first. Their latencies are different operations. T's transfer result cannot establish superiority over a fused attention kernel. KV bit identity also does not by itself guarantee identical floating-point attention output across kernels that use different arithmetic order.

## What to reproduce from each publication

**SplitZip.** Its codec workload assembles 256M BF16 values, checks bit identity, warms up, then averages ten runs. Throughput uses uncompressed byte count. Its reported transfer configuration is RoCE 4x200G with about 700Gbps effective payload bandwidth. The original requirements pin Torch 2.9.0/Triton 3.5.0. The public README says the full out-of-tree SGLang/Mooncake integration is not released. Reproduce the public codec script separately from a reconstructed serving integration. Do not present repeated/tiled codec inputs as an independent natural-text workload.

**ExANS.** Its blog does not name the snapshot's model or dataset, or provide a complete timing harness. Our public-data run is a pinned native API measurement, not a reproduction of the blog's 600+GB/s figure. An H100 alone does not establish equivalent batching, streams or timing boundaries. Retain full raw fallback and partial-block tails in every comparison.

**AdaCode.** Follow the paper's calibration and adaptive parameter rule, record short-code length and exponent range, and distinguish its cache/inference evaluation from a new transfer benchmark. Its reported ratio is remaining compressed size, so normalize ratio conventions before comparison. An online adjuster may use current input only when its computation and receiver metadata are charged; do not insert a free TEST-selected oracle policy.

**IBP.** Time its pinned-host fetch path. Its microbenchmark uses 100,000 tensors and 100 iterations; model experiments use FlexGen/InfiniGen. Charge preparation for fresh handoffs. Its full artifact needs about 1TB disk; smaller KV tests are separate experiments.

**LCT.** Verify its authored benchmark on synthetic distributions separately from our real KV. The README identifies shared codebooks, overflow allocation and excluded unused shared-arena capacity; all receiver-required state and actual transmitted overflow must be accounted for. Its code lists H100/H200 autotuning as unchecked. The primary abstract/code were verified here, but the full primary PDF could not be retrieved, so this audit deliberately quotes no LCT hardware throughput result from secondary summaries.

**ZipFusion/exactkv.** Put these in a separate compressed-attention experiment. ZipFusion excludes prefill and initial compression from its model decode timing; newly generated KV remains raw in that evaluation. Use both fixed batch and maximum batch under the same total memory budget. In its kernel experiment, L2 is cleared and reconstruction is part of attention time. These choices cannot be compared directly with a fresh P/D handoff total.

## Shared evaluation protocol we propose

Keep two result sets: an authors' protocol reproduction when the public artifact permits it, and a matched comparison using identical captured BF16 bytes on a common GPU. Neither replaces the other.

1. Start with the frozen Llama-3.2-3B windows/state to connect to the NUC results. Add Llama-3.1-8B and Qwen3-8B, public natural-text and code inputs, and a declared TRAIN/DEV/TEST split. Resolve model/tokenizer and dataset revisions and record capture hashes. Calibration is separate from held-out evaluation. Cross-domain tests include C4 and the relevant public task datasets.
2. Measure page/request sizes as well as the current 8K window: 16/32/128/512 tokens, 2K/8K/32K, and longer contexts only when the complete model, KV, predictor and workspace fit. Compare batch 1 and a declared batch sweep. Reserve H200 or validated multi-GPU configurations for large-model/long-context reproduction rather than assuming an 80GB GPU fits every case.
3. Test all 65,536 BF16 patterns, special values, outliers, rejected blocks, partial pages/tails and independent packet-only reconstruction. Use integer-word comparisons, including for NaNs. Check T's frozen packet identity on the new architecture before timing; investigate differences rather than silently relaxing the earlier reference. Charge every header, directory, raw escape/fallback, table and padding byte.
4. Publish codec-only resident encode/decode, complete fresh handoff, prepared-packet retrieval, and compressed attention/serving as separate measurements. Include cold shared-state bytes/setup separately from warm-state results. The current local GPU-host-GPU runner covers local staging, not network I/O.
5. Use wall-clock complete-path timing and CUDA stage timing, warmups, randomized/rotated order, at least three rounds, declared samples, medians and variability. Separate cold compilation/autotuning. Disclose L2/cache policy, graphs versus eager, native allocator behavior and whether CPU work is included. Report GB/s versus GiB/s explicitly using a declared byte denominator.
6. For real transfers use two nodes and the same transport/backend/buffer ownership for every codec. Measure raw GPU-buffer payload bandwidth first, then compressed traffic with packet sizes verified. Host TCP bandwidth or a VM's advertised aggregate bandwidth is not GPU-to-GPU application throughput. Report per-flow/per-GPU and aggregate numbers separately. Emulated bandwidth settings are only valid where the selected backend's rate control is actually effective.
7. Measure serving TTFT, decode latency/throughput, concurrency and total GPU memory under a fixed budget only after the codec/local/real-transfer checks pass. Preserve the T branch and compare against raw BF16 and uncompressed serving, with the same model/weights.

## AWS and GCP options

These are advertised ceilings checked against provider documentation, not reserved capacity, prices or measured payload rates. No VM has been launched.

| Phase | AWS candidate | GCP candidate | Qualification |
|---|---|---|---|
| One-GPU codec/local path | `p5.4xlarge`: one H100 80GB, 100Gbps | `a3-highgpu-1g`: one H100 SXM 80GB, 25Gbps | Good first common GPU; AWS's small P5 lacks GPUDirect RDMA; GCP's small A3 requires Spot/Flex-start |
| Two-node H100 transfer | Two `p5.48xlarge`: eight H100 each, up to 3.2Tbps per VM | Two `a3-highgpu-8g`: eight H100 each, up to 1Tbps per VM | AWS EFA/libfabric versus GCP TCPX require different verified integrations |
| H200 / closer SplitZip platform family | Two `p5en.48xlarge`: eight H200 each, EFAv3 | Two `a3-ultragpu-8g`: eight H200 each, RDMA/RoCE | H200/RoCE is closer to SplitZip's platform; still validate topology and effective bandwidth |

GCP A3 Mega is an additional H100/TCPXO option; its stack is not interchangeable with ordinary RoCE. The A3 Ultra maximum is 3.6Tbps per VM. Multi-NIC ceilings do not mean one KV transfer gets that rate. Mooncake currently documents an EFA backend (`USE_EFA`, libfabric); pin a compatible Mooncake/SGLang pair and test the actual selected protocol rather than assuming `rdma` and `efa` are interchangeable.

Recommendation: validate the NUC path first, then use one H100 allocation for the expanded codec suite. A second small H100 VM can test our host-staged path within its measured network limits; large GPU-direct nodes are needed only for the corresponding high-bandwidth track. Choose AWS/GCP by real quota/capacity and an explicit cost estimate; no cheapest-region claim is made. On GCP Spot/Flex-start, checkpoint artifacts to durable storage. An intra-node NVLink test is a useful additional result but does not replace inter-node network transfer.

## Environment design and readiness tool

`env/lossless_cloud_plan.json` records profiles, source pins, runtime constraints, unmeasured candidates and unresolved dependencies. Treat its null image/framework fields as unresolved work, not a lockfile. Build images in a new cloud-specific directory, record their immutable digests and complete package/build manifests, and keep the NUC venv and frozen sources intact.

Use separate original-artifact runtimes and a common matched T runtime (the existing Python 3.12/Torch 2.11+cu128/Triton 3.6 stack). IBP's original CUDA 11.7/A100 environment needs a validated newer-toolkit port for native H100 compilation; LCT's Python 3.14 requirements need an independent version lock. ZipFusion reports a CUDA 13 environment. Validate driver/runtime compatibility and use native SM90 builds on H100/H200. The current ExANS helper hardcodes SM86 and must be rebuilt for the actual cloud architecture in a new run. Its nvCOMP 5.3 ABI must remain isolated from a separate nvCOMP 6 comparison.

`scripts/cloud_lossless_preflight.py` collects CPU/RAM limits, disk space, GPU/driver/topology, CUDA compiler, package versions and EFA/verbs diagnostics. It checks the matched T runtime and identifies the SM86 rebuild blocker. It does not provision, install, download, benchmark or certify a real network. Run on a candidate VM after preparing its environment:

```bash
python scripts/cloud_lossless_preflight.py \
  --profile aws_h100_codec \
  --output /path/to/new/cloud_preflight.json
```

Use `gcp_h100_codec` or another JSON profile as appropriate. `full_environment_validated` and `network_performance_validated` remain false until the separate kernel/packet and two-node experiments are implemented and executed. Regional capacity, quota, reservation model and current price still need checking before allocation.

## Primary sources

- [SplitZip paper, setup and evaluation](https://arxiv.org/pdf/2605.01708), [author artifact](https://github.com/Intelligent-Microsystems-Lab/SplitZip).
- [ExANS author report](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache), [native implementation](https://github.com/openlake-project/openlake).
- [AdaCode paper, including Appendix A.3](https://aclanthology.org/2026.emnlp-main.269.pdf), [author code](https://github.com/lichangshiwhu/Adacode).
- [IBP paper and artifact requirements](https://arxiv.org/html/2605.30728v1), [author code](https://github.com/AKKamath/InvariantBitPacking).
- [LCT primary abstract](https://arxiv.org/abs/2610.22179), [author code and requirements](https://github.com/Maxzhu123/Losslessly_Compressed_Tensors).
- [ZipFusion paper, evaluation section](https://arxiv.org/pdf/2610.33719), [author publication page](https://henryhxu.github.io/papers.html), [exactkv author artifact](https://github.com/zyfyz666/exactkv).
- [NVIDIA nvCOMP](https://developer.nvidia.com/nvcomp), [version archive](https://developer.nvidia.com/nvcomp-archive).
- [AWS accelerated instance tables](https://aws.amazon.com/ec2/instance-types/accelerated-computing/), [AWS EC2 specification tables](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html), [P5en platform](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p5en-instances-with-nvidia-h200-tensor-core-gpus-and-efav3-networking/).
- [GCP accelerator machine types](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines), [network capabilities](https://docs.cloud.google.com/compute/docs/gpus/gpu-network-bandwidth), [H100 GPUDirect setup](https://docs.cloud.google.com/compute/docs/gpus/gpudirect).
- [Mooncake transport/build documentation](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/getting_started/build.md).
