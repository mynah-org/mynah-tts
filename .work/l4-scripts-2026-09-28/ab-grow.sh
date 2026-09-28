#!/bin/bash
while ! grep -q SOAK96-DONE /root/evidence/soak96.out 2>/dev/null; do sleep 10; done
cd /root/mynah-head
cp /root/evidence/engine_pocket.grow.c src/engine_pocket.c
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-grow.log 2>&1; echo BUILD:$?
grep -E "error" /root/evidence/build-grow.log | head
for m in 1 0; do echo "== self-check grow=$m"; MYNAH_CUDA_KV_GROW=$m MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none flock /root/gpu.lock timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda 2>&1 | tail -1; done
echo "== growth md5"; flock /root/gpu.lock /root/evidence/growtest.sh
grep -h -i "grow" /root/evidence/grow/*INITIAL*.log | head -5
/root/mynah-head/tools/l4/quality.sh q-growforced 64 MYNAH_CUDA_KV_GROW_INITIAL_STEPS=16
T=/root/mynah-head/tools/l4/ab.sh
MYNAH_L4_BATCH=128 $T grow128 models/pocket-english-24l 64,96,128 5 20
echo ABGROW-DONE
if grep -A0 "^C128" /root/evidence/grow128/grow128-summary.jsonl >/dev/null 2>&1; then :; fi
fails=$(python3 -c "import json;r=[json.loads(l) for l in open(\"/root/evidence/grow128/grow128-summary.jsonl\")];c=[x for x in r if x[\"offered\"]==128];print(\"ok\" if c and c[0][\"failed\"]==0 and c[0][\"completed\"]>0 and c[0][\"rtf_stream\"][\"p95\"]<1.0 else \"no\")")
echo "C128 short stress: $fails"
if [ "$fails" = ok ]; then
  ( while sleep 60; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) rss=$(ps -o rss= -p $(pgrep -f "[m]ynah-tts-server.*-p 18080" | head -1) 2>/dev/null)"; done ) > /root/evidence/soak128-vram.log 2>&1 &
  MON=$!
  MYNAH_L4_BATCH=128 /root/mynah-head/tools/l4/soak.sh soak128 128 1800
  kill $MON
fi
echo GROWALL-DONE
