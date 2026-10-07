#!/bin/bash
set -u
exec > /root/res/prov.log 2>&1
echo "== start $(date -u +%T)"
PY=/venv/main/bin/python; [ -x $PY ] || PY=python3
for cap in 384 768; do
  ( cd /root/m$cap && timeout 1800 make cuda cuda-server CUDA_ARCH=sm_89 ROW_CAP=$cap -j24 > /root/res/build-$cap.log 2>&1 && echo "BUILD $cap OK" || { echo "BUILD $cap FAILED"; tail -20 /root/res/build-$cap.log; } )
done
timeout 600 $PY -m pip install -q huggingface_hub numpy safetensors 2>&1 | tail -2
export HF_TOKEN=$(sed -e "s/^HF_TOKEN=//" /root/.hf_token | tr -d "[:space:]")
SNAP=$(timeout 1200 $PY -c "from huggingface_hub import snapshot_download as d; print(d(\"kyutai/pocket-tts\", allow_patterns=[\"languages/english_2026-04_24l/*\",\"embeddings*/*\",\"*.md\"]))" 2>/root/res/dl.err | tail -1)
echo "snapshot: $SNAP"; ls $SNAP/languages 2>&1
cd /root/m384 && timeout 1200 $PY tools/convert_pocket.py --source "$SNAP" --language english_2026-04_24l --output /root/models/pocket-english-24l 2>&1 | tail -3
for cap in 384 768; do
  ( cd /root/m$cap && MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none timeout 900 ./build/cuda/mynah-tts --pocket-self-check models/pocket-english-24l --device cuda 2>&1 | tail -1; timeout 300 ./build/cuda/mynah-tts --gpu-self-test cuda 2>&1 | tail -1 )
done
echo "== PROV-DONE $(date -u +%T)"
