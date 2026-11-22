set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/a2ce4e903758a76ef2b4a1d15d62974e3f47f8a5'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_trace_20261122T181202Z/repo/benchmark_results/pd_card/pd_trace_20261122T181202Z/node_specs' --kv-home "$KV_HOME"
