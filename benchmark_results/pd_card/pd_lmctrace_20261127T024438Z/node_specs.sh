set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/133bef914eaf8dba9ac9e74389864825c38e0843'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261127T024438Z/repo/benchmark_results/pd_card/pd_lmctrace_20261127T024438Z/node_specs' --kv-home "$KV_HOME"
