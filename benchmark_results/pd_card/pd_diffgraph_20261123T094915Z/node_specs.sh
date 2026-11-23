set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/b81bbdae35e361a665a29ac557a265f9d00bebb6'
python3 -u scripts/node_specs.py --output '/tmp/nbore/kvpd-evidence/pd_diffgraph_20261123T094915Z/repo/benchmark_results/pd_card/pd_diffgraph_20261123T094915Z/node_specs' --kv-home "$KV_HOME"
