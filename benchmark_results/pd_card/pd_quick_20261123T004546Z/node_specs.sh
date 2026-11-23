set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/fba9b8571974949e29a6ce071cd04e52046cae4d'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_quick_20261123T004546Z/repo/benchmark_results/pd_card/pd_quick_20261123T004546Z/node_specs' --kv-home "$KV_HOME"
