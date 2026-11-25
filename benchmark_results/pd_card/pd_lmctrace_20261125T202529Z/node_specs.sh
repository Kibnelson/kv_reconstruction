set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/ef21c334544904fe35236718998e5278a3820a1a'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_lmctrace_20261125T202529Z/repo/benchmark_results/pd_card/pd_lmctrace_20261125T202529Z/node_specs' --kv-home "$KV_HOME"
