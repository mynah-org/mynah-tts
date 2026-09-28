#!/bin/bash
S=$(ls -d /workspace/.hf_home/hub/models--kyutai--pocket-tts/snapshots/*)
cd /root/mynah-head; mkdir -p models
for p in "english_2026-04_24l pocket-english-24l" "english_2026-04 pocket-english-6l"; do
  set -- $p; echo "== $1 -> $2"
  flock /root/gpu.lock /venv/main/bin/python tools/convert_pocket.py --source $S --language $1 --output models/$2 2>&1 | tail -4
done
ls -la models/*/; echo MODELS2-DONE
