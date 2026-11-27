set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/eea6fcf20beb66678c8ef274799b43399ee8a8d8'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261127T011530Z/repo/benchmark_results/pd_card/pd_lmctrace_20261127T011530Z/node_specs' --kv-home "$KV_HOME"
