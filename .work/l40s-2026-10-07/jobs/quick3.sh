#!/bin/bash
# 3-minute thermal/sanity check, new defaults tree + L13b/L13d + A1a, pinned to the GPU node, C768.
exec >> /root/res/quick.log 2>&1
BEST="MYNAH_CUDA_STEP_OVERLAP=1 MYNAH_CUDA_DECODE_OVERLAP=1 MYNAH_CUDA_FAST_FIRST_CHUNK_WAIT_US=3000 MYNAH_CUDA_FIRST_FRAME_FIRST=1 MYNAH_CTX_HOST_POOL=1"
export PIN=32-63,96-127 CLIPIN=0-31,64-95
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader -l 15 > /root/res/quick-therm.log 2>&1 & T=$!
TAG=quick-best LEVELS="768" PROCS=4 DUR=180 ENVS="$BEST" timeout 900 /root/jobs/knee.sh
kill $T; echo "therm (temp, sm clock, power, reasons):"; awk 'NR%3==0' /root/res/quick-therm.log
grep -m1 MHz /proc/cpuinfo >/dev/null; echo "max core MHz under load sample: $(grep MHz /proc/cpuinfo | awk "{print \$4}" | sort -n | tail -1)"
echo "== quick done $(date -u +%T)"
