set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/649dc8556541c8167781a11e9563352c267eed98'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
