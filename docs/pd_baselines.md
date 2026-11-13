# Every baseline in real vLLM PD on 4 GPUs (branch `pd-baselines`)

Branch `pd-baselines`, from `pd-vllm-protocol`. Goal: the same vLLM prefill/decode system, the same prompts and
the same 4 RTX A5000s on `gpu-grad-01`, with the KV moved by each bit-exact codec in turn (the published baselines,
then T and T2). Measured end to end: requests per second, TTFT, TPOT and goodput, plus each transfer's bytes and
times.

## Stage 0: what the node is (`scripts/node_specs.sbatch`)

One 4-GPU job of about 10 minutes, before any of the runs. `scripts/node_specs.py` records:
- the GPUs (memory, PCIe generation and width, bus, NUMA node, power, ECC, driver), topology and peer-to-peer
  matrices;
- the CPU, NUMA, memory, `/dev/shm`, disks and quota, network interfaces, RDMA devices;
- the job's and the account's Slurm limits;
- the software on the node and in `KV_HOME`'s environments (torch, vLLM, NIXL, LMCache, nvCOMP, IBP).

It measures, with one 8K document of Llama-3.2-3B KV (896 MiB) per transfer:
- every GPU's host legs and every ordered GPU pair: direct (peer to peer), through host, through host pipelined;
- the planned layouts with every leg at once, direct and through pinned host memory:
  - 1 prefill + 3 decode, with each GPU as the prefill one;
  - 2 prefill + 2 decode, every pairing;
  - a CPU cache tier refilling 1-4 GPUs at once, and 4 GPUs storing to it.

The results go to `benchmark_results/<host>_specs_<gpu>x4_<date>` (`summary.md`, `specs.json`). They decide which
GPU prefills, which pairs to use for 2 + 2, and which link speeds the machine can actually reach.

```bash
module load slurm
cd ~/kv_llama32      # or your clone
git fetch -q origin pd-baselines && git checkout -q pd-baselines && git pull -q --ff-only origin pd-baselines
export KV_HOME=/a/place/with/70GB/kv_a5000
sbatch scripts/node_specs.sbatch
squeue --me; tail -f kv-specs.<job id>.out      # success: PUSHED benchmark_results/..._specs_...
```

## The matrix (planned; built after stage 0)

- **Codecs:** raw BF16, SplitZip, ExANS, nvCOMP (its GPU algorithms), AdaCode, exactkv, then T (five modes) and
  T2. IBP joins the CPU-cache path, which is what it was designed for (decompressing on the way from host memory).
- **Layouts:** 1 prefill + 3 decode, and 2 prefill + 2 decode.
- **Paths:**
  - GPU to GPU directly (peer to peer over PCIe), and through pinned host memory at set link speeds (the transport
    of `pd-vllm-protocol`);
  - a CPU cache tier: the KV of earlier requests kept in host memory (LMCache's role) and loaded to the decode
    GPU on a hit. vLLM with LMCache's CPU offload is the production reference (it keeps the KV uncompressed).
- **References:** vLLM's own connectors without a codec (NIXL for GPU to GPU, LMCache for the CPU tier), to show
  what vLLM does today on the same machine.
