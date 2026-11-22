#!/usr/bin/env bash
# Login-host card for one PD-baselines campaign on gpu-grad-01 (docs/pd_baselines.md, "On gpu-grad-01: the card").
# Only checks run here. Every stage runs in its own Slurm allocation (scripts/pd_campaign.py on the node); the run
# stage holds the 4 GPUs. Nothing is installed or run outside an allocation, and only the results are pushed, to
# their own branch.
#
#   bash scripts/pd_card.sh create|run|cleanup|verify-cleanup|review|publish PIN TAG
#
#   PIN   the 40-hex commit this checkout is at, GitHub's pd-baselines head (for create and run: exactly its head)
#   TAG   pd_<matrix>_<name>: matrix smoke, ring1p3d, ring2p2d, direct, tier, t2eager, links, workloads, wlong, wlours, capacity, quick, trace, capture or xfer
#         (scripts/pd_matrix.sh)
#
# On the node, under /tmp/$USER (the node's own disk; it survives between jobs, not reboots):
#   kvpd-tools/<pin>             the code at the pin (shares the home checkout's git objects)
#   kvpd-evidence/<tag>/repo     the campaign: the pin's tree with only benchmark_results/pd_card/<tag>/ checked out;
#                                the results, logs and receipts go there, and publish pushes them
#   kvpd/<run job>               the run's runtime (environments, sources, captures, caches); cleanup deletes it
#   kvpd-cache                   uv's package cache, the Python build, the model, the CUDA 13 toolkit: kept, reused
#
# Options (environment, checked on the node): KVPD_TIME (the run stage's limit; default 6 h for smoke, else 12 h),
# KVPD_IBP (auto: build IBP when the matrix uses it and the disk allows; 0 or 1), KVPD_BRANCH (pd-baselines), RUNS
# (scripts/pd_matrix.sh's own list instead of the matrix's).
set -Eeuo pipefail
umask 077
trap 'printf "PD CARD STOPPED: stage=%s line=%s exit=%s; the campaign and its output are kept\n" "${KVPD_STAGE:-setup}" "$LINENO" "$?" >&2' ERR
if (( $# != 3 )); then
    printf 'Usage: bash %s create|run|cleanup|verify-cleanup|review|publish PIN TAG\n' "$0" >&2
    exit 2
fi
export KVPD_STAGE=$1 KVPD_PIN=$2 KVPD_TAG=$3
case "$KVPD_STAGE" in create|run|cleanup|verify-cleanup|review|publish) ;; *) echo "STOP: unknown stage $KVPD_STAGE" >&2; exit 2 ;; esac
[[ "$KVPD_PIN" =~ ^[0-9a-f]{40}$ ]] || { echo "STOP: PIN is a full 40-character commit" >&2; exit 2; }
[[ "$KVPD_TAG" =~ ^pd_(smoke|ring1p3d|ring2p2d|direct|tier|t2eager|links|workloads|wlong|wlours|capacity|quick|trace|capture|xfer)_[A-Za-z0-9_-]+$ ]] || {
    echo "STOP: TAG is pd_<matrix>_<name> with matrix smoke, ring1p3d, ring2p2d, direct, tier, t2eager, links," \
         "workloads, wlong, wlours, capacity, quick, trace, capture or xfer" >&2; exit 2; }
export KVPD_MATRIX=${BASH_REMATCH[1]}
[ -z "${SLURM_JOB_ID:-}" ] || { echo "STOP: run the card on the login host, not inside an allocation" >&2; exit 2; }
[[ "${USER:-}" =~ ^[A-Za-z0-9_]+$ ]] || { echo "STOP: USER is not set to a plain user name" >&2; exit 2; }
export KVPD_BRANCH=${KVPD_BRANCH:-pd-baselines} KVPD_NODE=${KVPD_NODE:-gpu-grad-01} KVPD_ROOT=${KVPD_ROOT:-/tmp/$USER}

# the card is this checkout's, at the pin, unmodified; the checkout is GitHub's branch
CARD=$(readlink -f -- "${BASH_SOURCE[0]}")
export KVPD_HOME_REPO
KVPD_HOME_REPO=$(git -C "$(dirname "$CARD")" rev-parse --show-toplevel)
test "$CARD" = "$KVPD_HOME_REPO/scripts/pd_card.sh"
case "$KVPD_HOME_REPO/" in "$KVPD_ROOT"/*) echo "STOP: run the card from the home checkout, not a node-local copy" >&2; exit 2 ;; esac
test "$(git -C "$KVPD_HOME_REPO" rev-parse HEAD)" = "$KVPD_PIN" || {
    echo "STOP: the checkout is at $(git -C "$KVPD_HOME_REPO" rev-parse HEAD), not the pin $KVPD_PIN" >&2; exit 2; }
test -z "$(git -C "$KVPD_HOME_REPO" status --porcelain --untracked-files=no)" || {
    echo "STOP: uncommitted changes in tracked files of $KVPD_HOME_REPO" >&2; exit 2; }
# publish commits as you: checked now, not after the hours of the run
if [ -z "$(git -C "$KVPD_HOME_REPO" config user.name || true)" ] || [ -z "$(git -C "$KVPD_HOME_REPO" config user.email || true)" ]; then
    printf 'STOP: git has no user.name and user.email for %s (the author of the evidence commit). Set them for this
' "$KVPD_HOME_REPO" >&2
    printf '      checkout only:  git -C %s config user.name "Your Name"; git -C %s config user.email "you@example.com"
' \
        "$KVPD_HOME_REPO" "$KVPD_HOME_REPO" >&2
    exit 2
fi
test "$(git -C "$KVPD_HOME_REPO" hash-object "$CARD")" = "$(git -C "$KVPD_HOME_REPO" rev-parse "$KVPD_PIN:scripts/pd_card.sh")" || {
    echo "STOP: this card's bytes are not the pin's scripts/pd_card.sh" >&2; exit 2; }
REMOTE=$(GIT_TERMINAL_PROMPT=0 git -C "$KVPD_HOME_REPO" ls-remote origin "refs/heads/$KVPD_BRANCH" | awk '{print $1}')
case "$KVPD_STAGE" in
    create|run)
        test "$REMOTE" = "$KVPD_PIN" || {
            echo "STOP: GitHub's $KVPD_BRANCH is at ${REMOTE:-nothing}, not the pin: a new campaign starts from its head" >&2; exit 2; } ;;
    *)  # closing a campaign out after the branch moved on is fine: the pin only has to be in its history
        GIT_TERMINAL_PROMPT=0 git -C "$KVPD_HOME_REPO" fetch -q --no-tags origin "refs/heads/$KVPD_BRANCH"
        git -C "$KVPD_HOME_REPO" merge-base --is-ancestor "$KVPD_PIN" FETCH_HEAD || {
            echo "STOP: the pin is not in GitHub's $KVPD_BRANCH" >&2; exit 2; } ;;
esac
if type module >/dev/null 2>&1; then module load slurm; fi
command -v srun >/dev/null || { echo "STOP: no srun (module load slurm)" >&2; exit 2; }
export KVPD_TOOLS=$KVPD_ROOT/kvpd-tools/$KVPD_PIN

SRUN=(srun -p "${KVPD_PARTITION:-all}" -A "${KVPD_ACCOUNT:-oana2026}" -q "${KVPD_QOS:-research12h}" -N 1 -w "$KVPD_NODE"
      --chdir=/tmp --export=ALL --propagate=NONE)
if [ "$KVPD_STAGE" = run ]; then
    TIME=${KVPD_TIME:-$([ "$KVPD_MATRIX" = smoke ] && echo 06:00:00 || echo 12:00:00)}
    SRUN+=(--gpus=4 -c 16 --mem=192G -t "$TIME")
else
    SRUN+=(-c 2 --mem=8G -t 01:00:00)
fi
"${SRUN[@]}" bash -s <<'NODE'
set -Eeuo pipefail
umask 077
test "$(hostname -s)" = "$KVPD_NODE"
if type module >/dev/null 2>&1; then module load slurm; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_TERMINAL_PROMPT=0
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
private_dir() {   # an existing (or new) directory of this user only, not a link
    [[ -e "$1" || -L "$1" ]] || install -d -m 700 -- "$1"
    [[ -d "$1" && ! -L "$1" && -O "$1" ]] && test "$(readlink -f -- "$1")" = "$1" || {
        echo "STOP: $1 is not a directory of $USER only" >&2; exit 2; }
    (( (8#$(stat -c '%a' -- "$1") & 0022) == 0 )) || { echo "STOP: $1 is writable by others" >&2; exit 2; }
}
if [[ "$KVPD_STAGE" = create ]]; then
    private_dir "$KVPD_ROOT"
    private_dir "$KVPD_ROOT/kvpd-tools"
    if [[ ! -e "$KVPD_TOOLS" && ! -L "$KVPD_TOOLS" ]]; then
        # the code at the pin: the home checkout's objects (no copy), GitHub's branch checked to be the pin; every
        # folder but the results (benchmark_results is 600 MB of earlier runs)
        git clone -q --shared --no-checkout "$KVPD_HOME_REPO" "$KVPD_TOOLS"
        chmod 700 "$KVPD_TOOLS"
        git -C "$KVPD_TOOLS" remote set-url origin "$(git -C "$KVPD_HOME_REPO" remote get-url origin)"
        git -C "$KVPD_TOOLS" fetch -q --no-tags origin "refs/heads/$KVPD_BRANCH"
        test "$(git -C "$KVPD_TOOLS" rev-parse FETCH_HEAD)" = "$KVPD_PIN"
        git -C "$KVPD_TOOLS" sparse-checkout init --no-cone
        git -C "$KVPD_TOOLS" sparse-checkout set --no-cone '/*' '!/benchmark_results/' \
            '!/benchmark_fixtures/team_transfer_smol_replay_v1/'
        git -C "$KVPD_TOOLS" checkout -q --detach "$KVPD_PIN"
    fi
fi
private_dir "$KVPD_TOOLS"
test "$(git -C "$KVPD_TOOLS" rev-parse HEAD)" = "$KVPD_PIN"
test -z "$(git -C "$KVPD_TOOLS" status --porcelain --untracked-files=normal)" || {
    echo "STOP: the node-local code $KVPD_TOOLS has changes; it must stay the pin's" >&2; exit 2; }
python3 -I -B "$KVPD_TOOLS/scripts/pd_campaign.py" "$KVPD_STAGE" --pin "$KVPD_PIN" --tag "$KVPD_TAG" \
    --home-repo "$KVPD_HOME_REPO"
NODE
printf 'PD CARD: stage=%s matrix=%s pin=%s campaign=%s\n' "$KVPD_STAGE" "$KVPD_MATRIX" "$KVPD_PIN" \
    "$KVPD_ROOT/kvpd-evidence/$KVPD_TAG/repo"
