#!/bin/bash
exec > /root/res/chain3.log 2>&1
echo "== chain3 start $(date -u +%T)"
TREE=/root/m768b TAG=e4-reset LEVELS=640 PROCS=4 timeout 1500 /root/jobs/knee.sh
grep -E "\[SERVE\]   sync" /root/res/e4-reset/server.log
TREE=/root/m768b TAG=e5-two ROWS=320 PROCS=2 timeout 1500 /root/jobs/twoeng.sh
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
if nvidia-cuda-mps-control -d; then echo "MPS on"; TREE=/root/m768b TAG=e6-two-mps ROWS=320 PROCS=2 timeout 1500 /root/jobs/twoeng.sh; echo quit | nvidia-cuda-mps-control; echo "MPS off"; else echo "MPS could not start"; fi
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
TREE=/root/m1024 TAG=e7-1024 LEVELS="896 1024" PROCS=4 DUR=60 timeout 1500 /root/jobs/knee.sh
echo "== chain3 done $(date -u +%T)"
