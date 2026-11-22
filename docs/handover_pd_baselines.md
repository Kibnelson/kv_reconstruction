# Handover: every baseline in real vLLM PD, and making T faster (status 22 November 2026)

This note records the PD-baselines work of 13-22 November 2026 so it can be resumed: what ran, what it showed,
what was changed after it and is not yet measured, and the commands to go on. It follows
[docs/resume_t_and_diff_protocol.md](resume_t_and_diff_protocol.md), which covers T itself and the diff protocol up
to 30 October. The full design and the history of each campaign are in `docs/pd_baselines.md` on the branch
`pd-baselines`.

## Start here

1. **Free space in the home directory on the login host** (teach-node-06). The last start failed at `git fetch`
   with `Disk quota exceeded`. Home held about 10 GB: `context_pilot_paper/traces` 2.2 GB and
   `context_pilot_paper/.git` 1.7 GB were the largest. A campaign needs only a few MB in home (section 6.1).
2. **Run the `quick` campaign** (under an hour; section 6.3). It is the first measurement of the speed work in
   section 4.
3. **Read the `quick` results against the criteria in section 6.7.** Then run `xfer` (about 6-8 hours), then
   `capacity`.

Nothing has run on the cluster since job 30436 (18-19 November). No campaign is open: the failed start stopped before
`create`.

## 1. Where things stand

| What | Where | State |
| --- | --- | --- |
| Campaign code: connector, codecs, bench, card | branch `pd-baselines`, `b388494` | All the speed work and the `quick` and `xfer` matrices; tested on the CPU, not yet run on a GPU |
| Same code plus `scripts/pd_breakdown.py` | branch `pd-decode`, `53a4708` | `pd-baselines` can be moved here (section 6.2); the campaign does not need it |
| Results of each campaign | `evidence/pd-baselines/<pin12>/<matrix>/job<id>` | 17 branches: smoke 30163-30328, workloads 30334 (partial) and 30371, wlours 30436 |
| T itself (codec, calibration, lossless comparison) | `llama32-3b-lossy-lossless-compare` | See the resume note |
| This note | `main` | |

The machine is gpu-grad-01: 4x RTX A5000 24 GB, Slurm account `oana2026`, QoS `research12h`. It is reached from the
login host teach-node-06. The model is Llama-3.2-3B: 28 layers, 8 KV heads x 128, so 114,688 bytes of KV per token.

## 2. What is compared, and how

- **Layout.** vLLM 0.32 with our KV connector (`kv_compare/pd/connector.py`). 1 prefill engine and 3 decode engines
  (`1p3d`); also `2p2d`, and `3p1d` with one shared link.
- **Paths.**
  - `ring`: host memory, paced to a set link speed (10, 25, 50, 100 Gb/s). This is an emulated network link, one
    per prefill engine.
  - `p2p`: GPU to GPU, unpaced.
  - vLLM's NIXL connector, as a reference.
- **Codecs.**
  - raw BF16.
  - **T**, our predictive lossless codec, in its modes. The runs use `ours-hbh` (head by head, the fewest bits) and
    `ours-rank256` (the fastest).
  - **T2**: T with per-set calibration and an option to recompute layer 0; modes `t2-hbh` and `t2-rank256`.
  - The published bit-exact codecs from their authors' code: SplitZip, ExANS, AdaCode, exactkv, IBP, nvCOMP.
  - AQUA-KV (lossy), for speed only.
- **Prompts.** Real text (WikiText-2, HumanEval, GSM8K, MMLU, PTB) cut to exact token counts; greedy decoding;
  caches reset before each point.
- **Measures.** TTFT p50 and its speedup over raw (SplitZip's measure); requests per second; goodput at a TTFT
  target. Every transfer is checksummed in the check stage, so every codec is bit-exact through vLLM.
- **How the baselines compared themselves with raw.** Only SplitZip ran a PD transfer test: SGLang + Mooncake on
  H200, RoCE 4x200G (about 700 Gb/s), up to 1.30x TTFT; at 2K x 16, 56.5 -> 53.1 ms. ExANS, AdaCode, exactkv, IBP
  and AQUA-KV report codec speed, accuracy or memory, with no PD transfer.

## 3. What the campaigns showed

**Job 30334 (workloads, partial).** exactkv hit an illegal memory access on all three decode GPUs at once: 128-token
prompts, 16 at once, 10 Gb/s, after 5-7 good installs of that shape. A CUDA error of that kind kills every engine.
- exactkv now runs in runs of its own, last.
- The receiver checks exactkv's pages before decoding them.
- The bench stops at a dead engine: it asks every engine's `/health`, gives a point 10 minutes idle and 1 hour in
  all, and SIGTERM writes the report.
- The cause inside exactkv is still unknown.

**Job 30371 (workloads, complete).**
- T's and T2's head-by-head decoder took a fixed 135-190 ms per request from 128 to 8K tokens. It issued about 7,000
  small launches per request.
- Their link time was the lowest of all, half of SplitZip's.
- 16K requests ran out of memory on the prefill GPU (the encoder's scratch is about 4x the request's KV).
- Raw's 1.88 GB 16K transfer went through host memory while the others stayed on the GPU.
- ExANS rejected one of its own record headers twice.
- T2's layer-0 recompute never matched the compiled engine's cache: 47 % of the words differ. So T2 codes layer 0,
  which is exact but larger; still open.

**Job 30436 (wlours, complete).**
- The decoder in one chunk made T's installs 2.4-2.6x faster (40-68 ms).
- 16K went through in four 4K pieces, and raw's 16K transfer stayed on the GPU.
- With 32 2K prompts all at once at 10 Gb/s, in requests per second: T2 rank 256 3.68, T 3.60, SplitZip 3.43,
  raw 3.00.

Where a 2K request's time went, one request at a time (job 30436; ExANS from job 30371), in ms:

| Codec | Bits | TTFT 100 Gb/s | Encode | Link 100 | Install | TTFT 10 Gb/s | Link 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 16.0 | 223 | 1 | 20 | 3 | 394 | 190 |
| SplitZip | 12.0 | 230 | 8 | 15 | 5 | 356 | 143 |
| ExANS | 10.9 | 281 | 54 | 14 | 12 | 393 | 129 |
| T2 rank 256 | 9.3 | 265 | 34 | 12 | 17 | 361 | 110 |
| **T (hbh)** | **8.5** | **311** | **33** | **11** | **66** | **400** | **101** |

Prefill (about 169 ms) and the rest (about 29 ms) are the same for every codec. These come from the runs' own logs:
`scripts/pd_breakdown.py` joins the proxy's, prefill's and decode's records per request. Its parts add up to the
bench's TTFT within about 2 ms.

What that says about T:
- **Install is a fixed cost, not data.** 62, 62, 66 and 76 ms at 512, 1K, 2K and 4K tokens, once per 4K piece.
  The head-by-head decoder decodes 16 head groups one after another in each of 28 layers, with 4 launches per group:
  1,736 per piece.
- **The self-test (no vLLM) shows the same.** At 3,000 tokens: 24 ms with 1 group, 28 with 2, 35 with 4 and 53 with
  16, for bits from 9.27 to 8.53.
- **Encode grows with length** (about 15 ms per 1K tokens). In the head-group modes it made every layer's products
  twice: 46 ms at 3K, against 36 for the one-group mode.
- **Encode, link and install ran one after another.** The encode also ran inside the prefill engine's step.

Codec speeds at 2K from the self-test, in Gb/s of raw KV (encode / decode): T 57 / 28, T2 rank 256 57 / 110,
SplitZip 235 / 376, AdaCode 67 / 235, exactkv 65 / 188, ExANS 31 / 157, AQUA-KV 13 / 43. A codec pays off only when
its speeds are above the link's. One A5000 prefill engine produces KV at about 8-11 Gb/s, so a 100 Gb/s link per GPU
is mostly idle, and compression cannot gain much there.

## 4. What changed after job 30436 (not yet measured on a GPU)

Commits `0a311a3` and `b388494` on `pd-baselines`. Each change has a switch back, and the setup's profile times the
old calls beside the new.

| Change | Switch (default) | What it fixes | Expected, from the logs |
| --- | --- | --- | --- |
| The sender makes each layer's sums once (`LayerwiseOneStep.encode_once`); the same stream bit for bit | `TKV_ENCODE_ONCE=1` | Encoder products made twice | encode 33 -> ~26 ms at 2K |
| A head group's product of the groups before it in one Triton launch (`_gacc_kernel`) | `TKV_GROUP_PRODUCT=fused` (`torch`: as before) | 1,736 launches per piece -> 896 | install 66 -> ~40-50 ms |
| Each layer's decode replayed as a CUDA graph, on static buffers (~460 MB per decode GPU); kept only if its first replay matches the eager decode's checksums; sizes up to 4K recorded at warm-up | `TKV_DECODE_GRAPHS=1` | Python issuing each launch | install -> ~15-30 ms |
| Requests over 4K tokens go in pieces for every codec, raw included; coded, sent and installed piece by piece, so they overlap; the sender holds one piece at a time (64K prompts fit) | `PIPELINE_TOKENS=4096` (0: one transfer) | Serial encode, link and install; memory | 16K at 10 Gb/s: 1.34 s after prefill -> ~0.9 s |
| The sender codes on a thread and stream of its own; requests up to 2 GiB of KV are copied out in the engine's step | `ASYNC_ENCODE=true` | The prefill engine waiting on our encode | throughput under load |
| Several prefill engines share one paced link (layout `3p1d`) | `SHARED_LINK=0` | (test setup, not speed) | the transfer-bound regime |

Expected time to first token (estimates):
- **2K at 10 Gb/s:** raw 394, T 400 -> ~345 ms (about 12 % faster than raw; SplitZip 356).
- **2K at 100 Gb/s:** raw 223, T 311 -> ~255 ms (still about 15 % slower).
- **16K at 10 Gb/s:** 1.06x -> ~1.2x faster than raw.

Two things the logs cannot settle; the profile will:
- How much of the 17 us per launch is CPU, which graphs remove, and how much is small GPU kernels, which they don't.
- Whether the fused Triton product is as fast as cuBLAS on the A5000. If not, set `TKV_GROUP_PRODUCT=torch`.

Not done:
- Choosing our mode per request from the link speed. With the old decoder, head by head's smaller transfer beat
  rank 256's faster install only below about 2 Gb/s at 2K.
- Overlapping the encode with the prefill itself, layer by layer.
- Speeding up the encode kernels themselves: after the changes, about 12 ms per 1K tokens remains, which is the gap
  to raw on fast idle links.

## 5. The campaigns (`MATRIX` of `scripts/pd_matrix.sh`)

| Matrix | What it runs | Time |
| --- | --- | --- |
| `quick` | raw, T (hbh), T2 (hbh, rank 256), SplitZip; 2K and 16K one at a time, 16 x 2K at once; host memory at 10 and 100 Gb/s. Setup's profile times T and T2 against their calls as of job 30436 | < 1 h |
| `xfer` | Where the transfer dominates and where it does not: 8K, 16K, 32K, 64K prompts one at a time at 10/25/50/100 Gb/s and GPU to GPU (vLLM's GPU share 0.68 prefill, 0.66 decode); 3 prefill GPUs behind one link of 10-100 Gb/s (3.3-33 per GPU) with 2K/8K/16K many at once and 2K at 2-8 per second; exactkv apart, last. The report adds "where compression helps and where it does not" (per link and codec: wins / ties within 3 % / losses against raw) | ~6-8 h |
| `capacity` | 96 2K prompts at 1-8 per second: the highest rate where 90 % meet a 1 s TTFT, host memory at 10/25/50/100 Gb/s and GPU to GPU | ~4-5 h |
| `wlong` | 32K prompts, 16 at once of 8K/16K, 128 and 256 at once of 1K; exactkv apart | ~4 h |
| `workloads`, `wlours` | the baselines' own workloads (job 30371) and our re-run (job 30436) | 9 h, 3.5 h |

Codecs in `xfer`: raw, T (hbh, rank 256), T2 (hbh, rank 256), SplitZip, AdaCode, ExANS; exactkv with raw in a run of
its own.

## 6. Runbook

### 6.1 Before a start (login host teach-node-06)

```bash
quota -s 2>/dev/null || true; df -h ~                          # the home quota: git fetch needs a few MB
du -xsh ~/* ~/.[!.]* 2>/dev/null | sort -h | tail -25           # what uses it
module load slurm                                              # every new shell, before squeue/scancel/srun
tmux new -s kvpd                                               # the run stage holds the terminal for hours
```

A reboot of the login host ends the tmux session and its `srun`, and the GPU run with it. Root broadcasts the time;
one was announced for Friday 20 November at 04:00. Start a long matrix only if it ends before such a time.

### 6.2 Which code the campaign runs

A campaign runs GitHub's `pd-baselines` head, recorded as the pin. To move it to newer work, only fast-forwards,
and **never while a campaign is between `create` and the start of `run`**:

```bash
git fetch origin pd-baselines pd-decode
git merge-base --is-ancestor origin/pd-baselines origin/pd-decode && git push origin origin/pd-decode:refs/heads/pd-baselines
```

### 6.3 Start a campaign

In tmux on the login host. Set `KVPD_MATRIX` to `quick`, `xfer`, `capacity`, ...; the rest is the same.

```bash
module load slurm
bash <<'LOGIN'
set -Eeuo pipefail
umask 077
trap 'printf "PD CAMPAIGN STOPPED: line=%s exit=%s; the campaign and its output are kept\n" "$LINENO" "$?" >&2' ERR
test -z "${SLURM_JOB_ID:-}"
cd ~/kv_reconstruction
test -z "$(git status --porcelain --untracked-files=no)"
git fetch -q --no-tags origin pd-baselines
git switch -q pd-baselines 2>/dev/null || git switch -q -c pd-baselines --track origin/pd-baselines
git merge -q --ff-only origin/pd-baselines
export KVPD_PIN=$(git rev-parse HEAD)
test "$(git ls-remote origin refs/heads/pd-baselines | awk '{print $1}')" = "$KVPD_PIN"
export KVPD_MATRIX=quick
export KVPD_TAG="pd_${KVPD_MATRIX}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p ~/.kvpd_campaigns
STATE=~/.kvpd_campaigns/$KVPD_TAG.sh
printf 'export KVPD_PIN=%q\nexport KVPD_TAG=%q\n' "$KVPD_PIN" "$KVPD_TAG" > "$STATE"
printf 'PIN=%s\nTAG=%s\nSTATE=%s\n' "$KVPD_PIN" "$KVPD_TAG" "$STATE"
for STAGE in create run cleanup verify-cleanup review publish; do
    bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG"
done
echo "Campaign $KVPD_TAG published (the PUBLISHED line above names its branch)."
LOGIN
```

`PIN=` must print the commit you meant (`b388494...` today, or `53a4708...` if `pd-baselines` was moved).

Every stage runs in a Slurm allocation of its own on gpu-grad-01:
- `run` gets 4 GPUs, 16 CPUs, 192 GB and 12 h; the other stages get 2 CPUs and 1 h.
- Everything is written under the node-local `/tmp/nbore`:

  | Folder | What |
  | --- | --- |
  | `kvpd/<run job>` | the runtime; `cleanup` deletes it |
  | `kvpd-evidence/<tag>/repo` | the results |
  | `kvpd-cache` | uv, Python and the model, reused |

- Home gets only the checkout's git fetch and `~/.kvpd_campaigns/<tag>.sh`.
- `publish` pushes only `benchmark_results/pd_card/<tag>/` to its evidence branch and prints it as `PUBLISHED`.

### 6.4 Watch

```bash
module load slurm
squeue -u $USER -o '%.10i %.9P %.30j %.8T %.10M %.6D %R'
sacct -u $USER -S today --format=JobID,JobName%30,State,Elapsed,ExitCode
```

The block's terminal shows each stage's output: the run's setup, then each PD run's points.

### 6.5 Continue or close a campaign after a stop

For a lost terminal, a reboot, or a stage that stopped: the state file names the pin and tag. `run` happens once per
campaign, so a new measurement is a new tag.

```bash
module load slurm
cd ~/kv_reconstruction
ls -t ~/.kvpd_campaigns/ | head -3                 # the latest tags
source ~/.kvpd_campaigns/<tag>.sh                  # KVPD_PIN, KVPD_TAG (write the tag out: no <...> left in)
git switch -q --detach "$KVPD_PIN"                 # the card runs only from a checkout at its pin
for STAGE in cleanup verify-cleanup review publish; do bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG" || break; done
git switch -q pd-baselines
```

To stop a run early, use the job ID from `squeue`, then the block above (cleanup only deletes
`/tmp/nbore/kvpd/<run job>`):

```bash
module load slurm
scancel <run job id>
```

### 6.6 Read the results

A campaign's results are 25-55 MB. This puts them in the login host's `/tmp`, not in home. Set `B` to the branch
the `PUBLISHED` line names, written out:

```bash
B=evidence/pd-baselines/<pin12>/quick/job<id>
D=/tmp/$USER/kvpd_results/${B//\//_}
git clone -q --depth 1 --single-branch -b "$B" git@github.com:Kibnelson/kv_reconstruction.git "$D"
R=$(ls -d "$D"/benchmark_results/pd_card/pd_*)
less $R/matrix/pd_matrix_*/summary.md              # the combined report: per link and workload, TTFT and x over raw
less $R/matrix/*_ring_*/summary.md                 # one run's report
less $R/matrix/*_ring_*/pd_profile.md              # setup's profile: T and T2 against their calls as of job 30436
python3 -I ~/kv_reconstruction/scripts/pd_breakdown.py $R/matrix/*_ring_* --markdown "$D/breakdown.md"
```

The last line needs a checkout with `scripts/pd_breakdown.py`: `pd-decode`, or `pd-baselines` once moved
(section 6.2). Or send the `PUBLISHED` line to Claude, who fetches, breaks down and compares it with job 30436.

### 6.7 What to look for in `quick`, then what to run

- **The profile's table "T and T2 against their calls as of job 30436"** has columns `Encode once`, `Encode twice`,
  `Decode (graphs)`, `Decode eager` and `Decode as before`.
  - The speed work works if decode with graphs is well under half of "as before" at 2K, and encode once is
    15-25 % under encode twice.
  - If "Decode eager" is slower than "as before", the fused product loses to cuBLAS: set `TKV_GROUP_PRODUCT=torch`.
- **The decode engines' warm-up.** In each run's `connector/tkv_D*.jsonl.gz`, the `ready` record's `warm_up` has
  `"ours-hbh graphs"` and `"t2-hbh graphs"`: every size should be `true`. A size that "decodes eagerly" had graphs
  that differed and were dropped; it is still correct, but slow. A size first met during the run logs a `graphs`
  record.
- **The check stage passes for every codec.** 5K and 8K prompts go in pieces with checksums, so this checks the
  pieces and graphs through vLLM bit for bit.
- **TTFT against section 4's expectations:**
  - 2K at 10 Gb/s below raw and about SplitZip's.
  - 2K at 100 Gb/s within about 15 % of raw.
  - T's install at 2K about 15-30 ms (breakdown).
- **Then:** `xfer`, the transfer-bound regime and its counterpart. Then `capacity`.
- **If T is still slow:** the breakdown shows which part. The next steps are choosing the mode per request by link
  speed, and overlapping the encode with the prefill (section 4, "Not done").

## 7. Rules kept in this work

- The user's email address is for identification only: never send it to an external service without asking.
- On the cluster, everything runs inside a Slurm allocation. The login host only checks and submits. The runtime
  lives in node-local `/tmp`, and only results are pushed.
- Deletion commands act only inside `/tmp/nbore`. Anything in home is deleted by the user, item by item.
- Downloaded files and archives are untrusted: each in its own empty folder, with scripts kept elsewhere and Python
  run with `-I`. TLS checks stay on and the proxy settings are left alone. In Claude's sandbox, third-party
  repositories' code is not run.
- Commits made by Claude end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and the session line.
- `pd-baselines` moves only by fast-forward, and never between a campaign's `create` and its `run` start. One such
  push was refused by the automatic safety check until the user confirmed no campaign was open.

## 8. Pitfalls met

- A placeholder pasted literally (`JOB=<that id>`): write the real ID or tag into commands.
- `squeue`, `scancel` and `srun` need `module load slurm` in each new shell.
- A full home quota fails the first `git fetch`.
- A login-host reboot ends the tmux session and the run.
- A CUDA error in one codec kills every engine on that GPU: that is why exactkv runs last and apart, and why the bench
  stops at a dead engine.
- The interpreter tests of T's GPU code (`kv_compare/test_pd.py`, `OURS_WIRE_CHECK`) take about 20-40 minutes on a
  CPU. `test_lossy_gpu_codec_round_trip_interpreted` also fails at older commits (a Triton interpreter issue).
- T2's layer-0 recompute does not match compiled engines (47 % of the words). T2 then codes layer 0: exact, a little
  larger.

## 9. Open items

1. Run `quick`, then `xfer` and `capacity`; write up both regimes: where compression helps (slow links per GPU,
   long prompts, shared links) and where it does not (fast idle links), as SplitZip's own caveats say.
2. Choose our mode per request from the link rate and backlog (hbh, dense, rank 256), from the measured speeds.
3. Overlap encode with prefill, layer by layer, the only way to hide the encode on fast links.
4. A model with more KV per token. Candidate: Qwen1.5-1.8B, multi-head attention, 192 KB per token, about 2.9x
   Llama-3.2-3B's KV per second. It needs hbh generalised to as many groups as heads, and every codec exported and
   calibrated. Qwen1.5-4B breaks AQUA-KV's 1,024-channel groups.
5. exactkv's illegal memory access (job 30334): cause unknown.
6. T2's layer-0 recompute against compiled engines.
