set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/b81bbdae35e361a665a29ac557a265f9d00bebb6'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
