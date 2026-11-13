#!/bin/bash
# What gpu-grad-01 is, before the PD baseline runs (docs/pd_baselines.md): scripts/node_specs.py on the same 4 GPUs
# the runs use, packaged into benchmark_results/<host>_specs_<gpu>x<n>_<date> and pushed to this branch.
# From the login host, in the repository's root on branch pd-baselines:
#
#   module load slurm
#   git pull --ff-only origin pd-baselines
#   export KV_HOME=/a/place/with/70GB/kv_a5000     (where the PD runs will keep their environments)
#   sbatch scripts/node_specs.sbatch                 # about 10 minutes; 30-45 more when it must set up torch first
#   squeue --me; tail -f kv-specs.<job id>.out       # success: PUSHED benchmark_results/..._specs_...
#
# The bandwidth part needs a python with torch: KV_HOME's codec environment, the vLLM one, or ~/venvs/vllm-stock.
# With none, the job runs scripts/a5000_setup.sh first (the codec environment, which the PD runs need anyway);
# SETUP=0 skips that and measures only what needs no torch.
#SBATCH --job-name=kv-specs
#SBATCH --partition=all
#SBATCH --account=oana2026
#SBATCH --qos=research12h
#SBATCH --nodelist=gpu-grad-01
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --propagate=NONE
#SBATCH --output=kv-specs.%j.out
set -Eeuo pipefail
trap 'echo "SPECS JOB STOPPED at line $LINENO (exit $?)" >&2' ERR
cd "${SLURM_SUBMIT_DIR:?not a Slurm job}"
[ -f scripts/node_specs.py ] || { echo "STOP: submit from the repository's root (here: $PWD)"; exit 1; }
BRANCH=pd-baselines
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { echo "STOP: not on $BRANCH"; exit 1; }
export KV_HOME=${KV_HOME:-$HOME/kv_a5000}
export TMPDIR=$KV_HOME/tmp XDG_CACHE_HOME=$KV_HOME/cache
mkdir -p "$TMPDIR"
echo "== job $SLURM_JOB_ID on $(hostname): GPUs ${CUDA_VISIBLE_DEVICES:-?}, KV_HOME $KV_HOME, commit $(git log --oneline -1)"
HAVE=0
for py in "$KV_HOME/venv/bin/python" "$KV_HOME/venv_vllm/bin/python" "$HOME/venvs/vllm-stock/bin/python"; do
    [ -x "$py" ] && "$py" -c "import torch" 2>/dev/null && HAVE=1 && break
done
if [ "$HAVE" = 0 ] && [ "${SETUP:-1}" = 1 ]; then
    echo "== no python with torch yet: the codec environment first (scripts/a5000_setup.sh)"
    SKIP_EXANS=1 SKIP_IBP=1 bash scripts/a5000_setup.sh
fi
HOST=$(hostname -s)
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p | tr 'A-Z' 'a-z' | sed 's/nvidia//' | tr -cd 'a-z0-9')
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
DEST=benchmark_results/${HOST}_specs_${GPU_TAG}x${NGPU}_$(date +%Y%m%d_%H%M)
python3 -u scripts/node_specs.py --output "$DEST" --kv-home "$KV_HOME" 2>&1 | tee "$DEST.log"
mv "$DEST.log" "$DEST/console.log"
git add "$DEST"
git commit -q -m "Node specs: $HOST, $NGPU x $GPU_TAG (job $SLURM_JOB_ID)"
git fetch -q origin "$BRANCH"
git rebase -q "origin/$BRANCH" || { git rebase --abort; echo "STOP: rebase failed; committed locally"; exit 1; }
git push -q origin "HEAD:$BRANCH"
echo "PUSHED $DEST"
