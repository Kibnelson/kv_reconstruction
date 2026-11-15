set -Eeuo pipefail
umask 077
cd '/tmp/nbore/kvpd-tools/badb38842369fb8445d4c86500212169bbee0ed0'
bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh
