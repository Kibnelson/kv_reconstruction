set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/819bc54572e3825cf53accc60ee3e2fc8907df83'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
