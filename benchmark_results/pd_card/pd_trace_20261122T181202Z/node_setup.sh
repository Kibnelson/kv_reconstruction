set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/a2ce4e903758a76ef2b4a1d15d62974e3f47f8a5'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
