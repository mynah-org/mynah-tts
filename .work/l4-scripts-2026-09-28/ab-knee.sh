#!/bin/bash
while ! grep -q GROW2-DONE /root/evidence/grow2.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
cp /root/evidence/cap256/engine_pocket.c src/; cp /root/evidence/cap256/graph.h src/; cp /root/evidence/cap256/backend_cuda.cu gpu/cuda/
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-knee.log 2>&1; echo BUILD:$?
grep error /root/evidence/build-knee.log | head
MYNAH_L4_BATCH=192 /root/mynah-head/tools/l4/ab.sh knee models/pocket-english-24l 128,144,160,176,192 5 25
echo KNEE-DONE
