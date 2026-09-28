#!/bin/bash
while ! grep -q ABREUSE-DONE /root/evidence/ab-reuse.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
cp /root/evidence/backend_cuda.attn.cu gpu/cuda/backend_cuda.cu
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-attn.log 2>&1; echo BUILD:$?
grep -E " error|error:" /root/evidence/build-attn.log | head
for m in fast legacy; do
  echo "== self-check $m"; MYNAH_CUDA_BACKBONE_ATTN=$m MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none flock /root/gpu.lock timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda 2>&1 | tail -4
done
Q=/root/mynah-head/tools/l4/quality.sh
$Q q-attnfast 64
T=/root/mynah-head/tools/l4/ab.sh
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-attnfast models/pocket-english-24l 32,48,64,80 5 20
MYNAH_L4_CPUS=0-3 MYNAH_L4_WORKERS=4 $T g6-attnlegacy models/pocket-english-24l 32,48,64,80 5 20 MYNAH_CUDA_BACKBONE_ATTN=legacy
echo ABATTN-DONE
