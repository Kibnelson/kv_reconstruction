set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/d4987666a39e915bb42160d8f565fd9f2558a944'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_capture_20261122T214636Z/repo/benchmark_results/pd_card/pd_capture_20261122T214636Z/node_specs' --kv-home "$KV_HOME"
