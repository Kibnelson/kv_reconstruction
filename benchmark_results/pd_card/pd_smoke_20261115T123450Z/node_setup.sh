set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/7d42b833515309342861e3f2c7f83473d053d951'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
