#!/bin/bash
while ! grep -q ABWIDE-DONE /root/evidence/ab-wide.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-pool.log 2>&1; echo BUILD:$?
grep -E "error" /root/evidence/build-pool.log | head
echo "== self-check"; MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none flock /root/gpu.lock timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda 2>&1 | tail -1
echo "== leak"; flock /root/gpu.lock /root/evidence/leak.sh
T=/root/mynah-head/tools/l4/ab.sh
$T pool1 models/pocket-english-24l 32,48,64 5 20
$T pool0 models/pocket-english-24l 32,48,64 5 20 MYNAH_CUDA_SLOT_POOL=0
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T cpu4-pool1 models/pocket-english-24l 32,48,64 5 20
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T cpu4-pool0 models/pocket-english-24l 32,48,64 5 20 MYNAH_CUDA_SLOT_POOL=0
echo ABPOOL-DONE
