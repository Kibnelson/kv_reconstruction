set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/590538a967615cbab0c56603e2bb88926bd03da4'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
