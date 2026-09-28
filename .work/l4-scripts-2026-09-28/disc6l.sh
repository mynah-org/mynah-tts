#!/bin/bash
# 6L discovery after the 24L control: self-check, then a ladder to the knee.
while ! grep -q CTRL-DONE /root/evidence/ctrl.out 2>/dev/null; do sleep 30; done
cd /root/mynah-head
echo "== self-check 6L"; MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none flock /root/gpu.lock timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-6l --device cuda 2>&1 | tail -1
MYNAH_L4_BATCH=256 /root/mynah-head/tools/l4/ab.sh disc6l models/pocket-english-6l 128,160,192,224,256 5 25
echo DISC6L-DONE
