set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/649dc8556541c8167781a11e9563352c267eed98'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_diffedge_20261125T153406Z/repo/benchmark_results/pd_card/pd_diffedge_20261125T153406Z/node_specs' --kv-home "$KV_HOME"
