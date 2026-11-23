set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/41b13a14d2d158655bddffd00b5f2fc3c42b4843'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_quick_20261123T031413Z/repo/benchmark_results/pd_card/pd_quick_20261123T031413Z/node_specs' --kv-home "$KV_HOME"
