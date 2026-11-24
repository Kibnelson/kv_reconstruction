set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/c56fa39ebaeaffdc8b1e8b52cf0c98ae890be62a'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_capture_20261124T152618Z/repo/benchmark_results/pd_card/pd_capture_20261124T152618Z/node_specs' --kv-home "$KV_HOME"
