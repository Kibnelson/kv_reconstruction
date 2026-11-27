set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/268a9a96a32412d73b41a979e7f139f952549282'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261127T034204Z/repo/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z/node_specs' --kv-home "$KV_HOME"
