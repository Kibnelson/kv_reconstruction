set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/7731196f837621be8abfd6205575429d1265689f'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
