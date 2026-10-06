#!/bin/bash
# 3-minute thermal / sanity check with all flags. Usage: LV=256 quick.sh
exec >> /root/res/quick.log 2>&1
source <(sed -n "/^declare -A F/,/^ALL=/p" /root/jobs/day2.sh)
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader -l 15 > /root/res/quick-therm.log 2>&1 & T=$!
TAG=quick-all LEVELS="${LV:-256}" PROCS=4 DUR=180 ENVS="$ALL" timeout 900 /root/jobs/knee.sh
kill $T; echo "therm (temp, sm clock, power, reasons):"; awk 'NR%3==0' /root/res/quick-therm.log
echo "== quick done $(date -u +%T)"
