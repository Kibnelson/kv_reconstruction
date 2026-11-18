set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/e7572c1b03faa12fbe225e814ca4fbc1c3786e9c'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
