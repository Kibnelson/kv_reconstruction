#!/bin/bash
# Every baseline in real vLLM PD on gpu-grad-01's 4 A5000s (docs/pd_baselines.md) as ONE batch job: the setup (codec
# environment with ExANS's toolkit, vLLM, nvCOMP, NIXL, LMCache, T2's constant sets, the self-test), then
# scripts/pd_matrix.sh with MATRIX. From the login host, in the repository's root on branch pd-baselines:
#
#   module load slurm
#   git pull --ff-only origin pd-baselines
#   export KV_HOME=/a/place/with/90GB/kv_a5000
#   MATRIX=smoke sbatch --time=4:00:00 scripts/pd_matrix.sbatch     # first (also does the one-time setup)
#   MATRIX=ring1p3d sbatch scripts/pd_matrix.sbatch                 # then each of ring1p3d ring2p2d direct tier
#   squeue --me; tail -f kv-pd-matrix.<job id>.out                  #   t2eager, one job each
#
# sbatch passes the submitting shell's environment (KV_HOME, MATRIX, RUNS, SMOKE, ...) to the job; options on the
# command line override the lines below (another node: --nodelist=gpu-grad-02, with KV_HOME on shared storage).
# The QoS research12h allows 4 GPUs at a time, so the jobs run one after another.
#SBATCH --job-name=kv-pd-matrix
#SBATCH --partition=all
#SBATCH --account=oana2026
#SBATCH --qos=research12h
#SBATCH --nodelist=gpu-grad-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --propagate=NONE
#SBATCH --output=kv-pd-matrix.%j.out
set -Eeuo pipefail
trap 'echo "PD MATRIX JOB STOPPED at line $LINENO (exit $?)" >&2' ERR
cd "${SLURM_SUBMIT_DIR:?not a Slurm job}"
[ -f scripts/pd_matrix.sh ] || { echo "STOP: submit from the repository's root (here: $PWD)"; exit 1; }
export KV_HOME=${KV_HOME:-$HOME/kv_a5000}
echo "== job $SLURM_JOB_ID on $(hostname): GPUs ${CUDA_VISIBLE_DEVICES:-?}, KV_HOME $KV_HOME, MATRIX ${MATRIX:-smoke}, commit $(git log --oneline -1)"
nvidia-smi -L
SKIP_EXANS=${SKIP_EXANS:-0} SKIP_IBP=${SKIP_IBP:-1} bash scripts/a5000_setup.sh   # IBP: see docs/pd_baselines.md
bash scripts/pd_setup.sh
MATRIX=${MATRIX:-smoke} bash scripts/pd_matrix.sh
