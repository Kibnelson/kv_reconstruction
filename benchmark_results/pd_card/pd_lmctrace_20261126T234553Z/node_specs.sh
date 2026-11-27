set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/0447555704eb4e7e34d4d74d66c7b1ba85c8b53a'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261126T234553Z/repo/benchmark_results/pd_card/pd_lmctrace_20261126T234553Z/node_specs' --kv-home "$KV_HOME"
