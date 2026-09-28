#!/bin/bash
while ! grep -q NSYS2-DONE /root/evidence/ab-tilews.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
cp /root/evidence/backend_cuda.tc.cu gpu/cuda/backend_cuda.cu
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-tc.log 2>&1; echo BUILD:$?
grep -E "error|warning" /root/evidence/build-tc.log | grep -v "Wno" | head
echo "== self-check tc"; MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none flock /root/gpu.lock timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda 2>&1 | tail -2
/root/mynah-head/tools/l4/quality.sh q-tc 64
T=/root/mynah-head/tools/l4/ab.sh
export MYNAH_L4_BATCH=96
$T tc models/pocket-english-24l 32,64,96 5 20
$T simt models/pocket-english-24l 32,64,96 5 20 MYNAH_CUDA_TILE_TC=0
$T cublas models/pocket-english-24l 32,64,96 5 20 MYNAH_CUDA_PREFILL_FIXED=0
echo ABTC-DONE
