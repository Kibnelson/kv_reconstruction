set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/fba9b8571974949e29a6ce071cd04e52046cae4d'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
