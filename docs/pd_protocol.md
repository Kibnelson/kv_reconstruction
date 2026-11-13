# PD transfer protocol on vLLM (1 prefill + 3 decode)

Branch `pd-baselines` extends this with every published baseline, T2, 2 prefill + 2 decode, the direct GPU-to-GPU path, a CPU tier, and vLLM's NIXL and LMCache as references: `docs/pd_baselines.md`.

Branch `pd-vllm-protocol`. Goal: in prefill/decode disaggregation on ordinary GPUs, show how many requests a
system serves, and how fast, when the KV moves in our format (T) against the published bit-exact codecs. Then find
out, before building any of it, whether a smarter protocol could avoid sending KV the decode side already holds or
held: on its GPU, in its host memory (DRAM), or on another decode engine.

## What runs

- **vLLM 0.32.0, unmodified.** One prefill engine on GPU 0 and three decode engines on GPUs 1-3, all serving
  Llama-3.2-3B (the model our codec constants are calibrated on).
- **TKVConnector** (`kv_compare/pd/connector.py`): our protocol as a vLLM KV connector, loaded through vLLM's
  external-connector option (`kv_connector_module_path`). Push mode:
  1. The proxy sends each request to the prefill engine (one token) and, at the same time, to a decode engine.
  2. When the prefill finishes, the prefill engine's worker gathers the prompt's KV from vLLM's paged cache, frees
     the blocks, encodes the KV with the codec the request names, and queues the coded transfer for the link.
  3. The decode engine holds the request (vLLM's `WAITING_FOR_REMOTE_KVS`) with its blocks allocated.
  4. As each transfer completes, the decode engine decodes it layer by layer straight into those blocks.
  5. vLLM recomputes only the last prompt token and starts generating.
- **Codecs** (`kv_compare/pd/codecs.py`), all bit-exact, the same on both ends from one exported constants file:
  - raw BF16;
  - ours (T) in five modes: dense, keys then values, 4 head groups, head by head, rank 256;
  - SplitZip (the authors' codec and code book);
  - AdaCode (the authors' codec and parameter rule).

  ExANS and nvCOMP are measured in the codec benchmark; they are not yet wired into vLLM. They are native
  libraries, and nvCOMP crashed processes on the NUC.
- **Link** (`kv_compare/pd/transport.py`): one virtual link per prefill engine, shared by all its destinations,
  first come first served, set to 3-400 Gb/s.
  - On one machine the bytes move through a shared-memory ring registered as pinned memory: one copy from the
    sender's GPU, one to the receiver's GPU, chunk by chunk. That is about what GPUDirect over a PCIe network card
    does, and it is bounded by the same PCIe slots (about 190 Gb/s measured on the NUC).
  - A chunk is released no earlier than a real link of the set speed would deliver it. Where the machine is slower
    than the set speed, the transfer takes the machine's time; every record keeps the speed actually achieved.
  - TCP works across machines.
- **Proxy** (`scripts/pd_proxy.py`) routes to the decode engines round-robin, by fewest requests in flight, or by
  conversation. Its `local` mode is the baseline without disaggregation: each decode engine prefills its own
  requests.
- **Correctness.** In the check pass, every transfer carries a checksum of the sender's words. The receiver compares
  it with what it installed in its paged cache. A mismatch fails the transfer, and vLLM is told which blocks are bad.

## Phases

1. **Everything sent (now).** For every link speed (3, 5, 10, 25, 40, 100, 200, 400 Gb/s) × codec × request rate,
   run `vllm bench serve` through the proxy on real text prompts (WikiText-2, HumanEval, GSM8K, MMLU, PTB; 2K and
   8K tokens, 128 output tokens). Measured:
   - requests/s, goodput at a TTFT target, TTFT p50/p99, TPOT;
   - per transfer: bits, gather/encode/install times, link time, achieved speed.

   The same prompts without disaggregation are the baseline.
2. **Opportunities first (now; nothing is sent differently yet).** vLLM's multi-turn benchmark runs through the proxy
   with conversation routing. The connector logs, per transferred request:
   - what the decode engine already had in its own GPU prefix cache, as vLLM measured it;
   - the chained block hashes of the prompt.

   `scripts/pd_opportunity.py` splits the KV sent into four parts: already on the decode GPU; held before by that
   engine (a DRAM tier would have kept it, unbounded or within N GB); held only by another decode engine; new.
   Bytes come out both raw and as coded. Only if those numbers are large is phase 3 worth building.
3. **Smart transfer (later, if phase 2 says so).** Track where every block lives (decode GPU, decode DRAM, another
   engine, only the prefill side). Send only the missing blocks, and pull bases evicted to DRAM over the local PCIe
   at the same time as the network transfer, all in T format. Compute layer 0 on the decode side from the token IDs.

## Runbook

**Prerequisites.**
- `scripts/a5000_setup.sh` has run: the codec environment `$KV_HOME/venv` and the authors' sources.
- The branch is checked out in `~/kv_llama32` (or your clone), on `pd-vllm-protocol`.
- On the NUC the scripts take `ALLOW_NO_SLURM=1`.

**NUC smoke (one GPU shared by one prefill and one decode engine):**

```bash
bash <<'NUC'
set -Eeuo pipefail
cd ~/kv_llama32
git fetch -q origin pd-vllm-protocol && git checkout -q pd-vllm-protocol && git pull -q --ff-only origin pd-vllm-protocol
export ALLOW_NO_SLURM=1 KV_HOME=$HOME/kv_a5000
bash scripts/pd_setup.sh
SHARE_GPU=1 NUM_DECODE=1 SMOKE=1 bash scripts/run_pd_vllm.sh
NUC
```

Success markers:
- `PD SETUP DONE` after `SELFTEST OK`: every codec bit-exact through the paced transport in vLLM's environment;
- `check <codec>: OK` for each codec;
- `PUSHED benchmark_results/llama32_3b_pdvllm_..._smoke`.

**A5000, four GPUs (`gpu-grad-01`, account `oana2026`, QoS `research12h`).** The QoS allows 4 GPUs at a time,
so 1 prefill + 3 decode is the largest layout it gives. Run it as one batch job (`scripts/pd_a5000.sbatch`): the
codec setup, the vLLM setup and the run, with no SSH session or login-host tmux to lose. From the login host, in
the repository's root:

```bash
module load slurm
cd ~/kv_llama32
git fetch -q origin pd-vllm-protocol && git checkout -q pd-vllm-protocol && git pull -q --ff-only origin pd-vllm-protocol
export KV_HOME=/a/place/with/70GB/kv_a5000                 # not the home directory: see below
SMOKE=1 sbatch --time=3:00:00 scripts/pd_a5000.sbatch       # first: the whole path with few prompts
squeue --me; tail -f kv-pd.<job id>.out                     # success: PUSHED benchmark_results/..._smoke
sbatch scripts/pd_a5000.sbatch                              # then: the full sweep
```

Before the first job:
- **Disk.** `KV_HOME` holds both Python environments (about 8 and 12 GB), the model (6 GB), the calibration
  captures (about 7 GB) and every cache (pip, Triton, vLLM's compile cache). The home directory's quota ran out
  installing vLLM alone in September ("Disk quota exceeded"), so put `KV_HOME` on larger storage (`quota -s` and
  `df -h` show what you have). `/tmp/$USER` on gpu-grad-01 had 84 GB free, but it is the node's own disk, shared and
  not backed up: it works only while every job runs on that node. Results do not depend on it: they are pushed.
- **Push access** from the compute node without a prompt (an SSH key without a passphrase, or a stored token), as
  in `docs/a5000_runbook.md`. Without it the results stay committed in the job's checkout.
- **The node is shared.** The engines listen on 127.0.0.1 only, their ports come from the job id (`PORT_BASE`), and
  the run stops at once if one is taken. Engine i gets the i-th GPU of Slurm's `CUDA_VISIBLE_DEVICES`.
- **Another node.** If gpu-grad-01 has no 4 free cards, `sbatch --nodelist=gpu-grad-02 scripts/pd_a5000.sbatch`
  (8 A5000s, the same setup; `KV_HOME` must then be shared storage, not `/tmp`).

The interactive form still works, from tmux or screen on the login host, if you want to watch the engines:

```bash
srun --partition=all --account=oana2026 --qos=research12h --nodelist=gpu-grad-01 \
     --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=12:00:00 --job-name=kv-pd --pty bash -l
```

then

```bash
bash <<'A5000'
set -Eeuo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "STOP: run this inside the allocation"; exit 1; }
cd ~/kv_llama32
git fetch -q origin pd-vllm-protocol && git checkout -q pd-vllm-protocol && git pull -q --ff-only origin pd-vllm-protocol
export KV_HOME=/a/place/with/70GB/kv_a5000
SKIP_EXANS=1 SKIP_IBP=1 bash scripts/a5000_setup.sh
bash scripts/pd_setup.sh
MODEL=llama32_3b bash scripts/run_pd_vllm.sh
A5000
```

Defaults run every stage: check, sweep, colocated, multiturn, report.
- The default sweep is 5 links (3, 10, 25, 100, 400 Gb/s) × 7 codecs × 4 rates (0.5, 1, 2 req/s and all at once)
  of 48 prompts: about 3.5 hours.
- Each run folder is new (dated). To add links (`LINKS=5,40,200`) or rates later, start another run with only those
  settings.
- Results are pushed to `benchmark_results/llama32_3b_pdvllm_rtxa5000x4_<date>`:
  - `summary.md` and `results.json`;
  - every `vllm bench` result;
  - the connector logs (gzip);
  - the reuse report `opportunity.json`.

## Honest limits

- On gpu-grad-01 (from `nvidia-smi topo -m` in a 4-GPU job in September), GPU 0 reaches the others through the host
  bridges, while GPUs 1-3 sit behind one PCIe switch. The transport stages every transfer through pinned host
  memory, so the three decode engines probably share that switch's one link to the host. That link and PCIe bound
  the achieved speed, which every transfer records.
- The links are emulated on one machine. The transfer bytes and every codec's GPU work are real; the network is a
  pacer on top of a PCIe-bound copy path. Above about 190 Gb/s the machine, not the setting, sets the speed, and the
  achieved speed is reported per run.
- One prefill engine on an A5000 produces far less KV per second than a 25-400 Gb/s link carries (about 7.5 Gb/s raw
  for 8K prompts, by estimate; the runs measure it). At those speeds the sweep therefore shows TTFT and GPU-time
  effects rather than a link bottleneck. The link-capacity comparison up to 400 Gb/s (requests per second a link
  carries) stands separately in the codec benchmark.
- Decoding takes decode-GPU time inside vLLM's engine step: 23-83 ms per 8K request depending on the mode.
  Reviewers ask about this interference, so it is measured, not hidden: TPOT in the sweep, and the install times.
- The DRAM figures of phase 2 are conservative. They count prompt blocks only, not blocks of the answers a decode
  engine generated itself; those are in the measured GPU hits.
