set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-t-matched-nbore/kvpd-tools/90112640bc59bac6ea1a82f79b8c53f2b22aa996'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
