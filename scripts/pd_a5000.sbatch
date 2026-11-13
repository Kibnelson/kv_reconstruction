#!/bin/bash
# The 4-GPU PD run (docs/pd_protocol.md) on gpu-grad-01 as ONE batch job: the codec setup, the vLLM setup, then
# 1 prefill + 3 decode engines and the sweep. Nothing depends on an SSH session or a tmux on a login host.
# From the login host, in the repository's root on branch pd-vllm-protocol:
#
#   module load slurm
#   git pull --ff-only origin pd-vllm-protocol
#   export KV_HOME=/a/place/with/70GB/kv_a5000           (the home quota is too small: see docs/pd_protocol.md)
#   SMOKE=1 sbatch --time=3:00:00 scripts/pd_a5000.sbatch  # first: the whole path, few prompts (about 1-2 hours)
#   sbatch scripts/pd_a5000.sbatch                         # then: the full sweep (about 6-8 hours with the setup)
#   squeue --me; tail -f kv-pd.<job id>.out
#
# sbatch passes the submitting shell's environment (KV_HOME, KV_RUNS, SMOKE, LINKS, CODECS, ...) to the job.
# Options on the command line override the lines below (another node: --nodelist=gpu-grad-02).
# The QoS research12h allows 4 GPUs at a time: 1 prefill + 3 decode is the largest layout it gives.
#SBATCH --job-name=kv-pd
#SBATCH --partition=all
#SBATCH --account=oana2026
#SBATCH --qos=research12h
#SBATCH --nodelist=gpu-grad-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --propagate=NONE
#SBATCH --output=kv-pd.%j.out
set -Eeuo pipefail
trap 'echo "PD JOB STOPPED at line $LINENO (exit $?)" >&2' ERR
cd "${SLURM_SUBMIT_DIR:?not a Slurm job}"
[ -f scripts/run_pd_vllm.sh ] || { echo "STOP: submit from the repository's root (here: $PWD)"; exit 1; }
export KV_HOME=${KV_HOME:-$HOME/kv_a5000}
echo "== job $SLURM_JOB_ID on $(hostname): GPUs ${CUDA_VISIBLE_DEVICES:-?}, KV_HOME $KV_HOME, commit $(git log --oneline -1)"
nvidia-smi -L
nvidia-smi topo -m || true
# the codec environment and the authors' SplitZip and AdaCode; the PD runs need neither ExANS's CUDA toolkit (about
# 8 GB) nor IBP: SKIP_EXANS=0 builds them as for the lossless benchmark
SKIP_EXANS=${SKIP_EXANS:-1} SKIP_IBP=${SKIP_IBP:-1} bash scripts/a5000_setup.sh
bash scripts/pd_setup.sh
MODEL=${MODEL:-llama32_3b} bash scripts/run_pd_vllm.sh
