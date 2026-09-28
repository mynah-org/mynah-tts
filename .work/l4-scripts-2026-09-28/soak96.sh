#!/bin/bash
cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-soak96.log 2>&1; echo BUILD:$?
grep -c "MYNAH_CUDA_TILE_TC\", false" gpu/cuda/backend_cuda.cu
( while sleep 60; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) rss=$(ps -o rss= -p $(pgrep -f "[m]ynah-tts-server.*-p 18080" | head -1) 2>/dev/null)"; done ) > /root/evidence/soak96-vram.log 2>&1 &
MON=$!
MYNAH_L4_BATCH=96 /root/mynah-head/tools/l4/soak.sh soak96 96 1800
kill $MON
echo SOAK96-DONE
