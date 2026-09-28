#!/bin/bash
while ! grep -q ABF32-DONE /root/evidence/ab-f32pack.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
cp /root/evidence/backend_cuda.tc2.cu gpu/cuda/backend_cuda.cu
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-tc2.log 2>&1; echo BUILD:$?
grep -E "error" /root/evidence/build-tc2.log | head
echo "== self-check tc2"; MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none flock /root/gpu.lock timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda 2>&1 | tail -1
T=/root/mynah-head/tools/l4/ab.sh
export MYNAH_L4_BATCH=96
$T tc2 models/pocket-english-24l 32,64,96 5 20
$T cublas2 models/pocket-english-24l 32,64,96 5 20 MYNAH_CUDA_PREFILL_FIXED=0
$T simt2 models/pocket-english-24l 64 5 20 MYNAH_CUDA_TILE_TC=0
/root/mynah-head/tools/l4/quality.sh q-tc2 64
echo ABTC2-DONE
