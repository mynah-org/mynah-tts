#!/bin/bash
set -u
mkdir -p /root/res
exec > /root/res/prov.log 2>&1
echo "== start $(date -u +%T)"
PY=/venv/main/bin/python; [ -x $PY ] || PY=python3
cd /root/mt
( timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 -j24 > /root/res/build-cuda.log 2>&1 && echo "CUDA BUILD OK" || { echo "CUDA BUILD FAILED"; tail -20 /root/res/build-cuda.log; } )
( timeout 1800 make -j24 > /root/res/build-cpu.log 2>&1 && echo "CPU BUILD OK" || { echo "CPU BUILD FAILED"; tail -20 /root/res/build-cpu.log; } )
ls build
timeout 600 $PY -m pip install -q huggingface_hub numpy safetensors 2>&1 | tail -2
export HF_TOKEN=$(tr -d "[:space:]" < /root/.hf_token)
SNAP=$(timeout 1200 $PY -c "from huggingface_hub import snapshot_download as d; print(d(\"kyutai/pocket-tts\", allow_patterns=[\"languages/english_2026-04*/*\",\"embeddings*/*\",\"*.md\"]))" 2>/root/res/dl.err | tail -1)
rm -f /root/.hf_token; unset HF_TOKEN
echo "snapshot: $SNAP"; ls $SNAP/languages 2>&1
for L in $(ls $SNAP/languages); do
  n=${L#english_2026-04}; n=${n#_}; [ -z "$n" ] && n=base
  timeout 1200 $PY tools/convert_pocket.py --source "$SNAP" --language $L --output /root/mt/models/pocket-english-$n 2>&1 | tail -2
done
ls models
timeout 300 ./build/cuda/mynah-tts --gpu-self-test cuda 2>&1 | tail -1
echo "== PROV-DONE $(date -u +%T)"
