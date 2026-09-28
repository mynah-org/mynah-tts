#!/bin/bash
source /root/.hf_token
cd /root/mynah-head
/venv/main/bin/python -m pip install -q huggingface_hub numpy safetensors 2>&1 | tail -1
/venv/main/bin/hf download kyutai/pocket-tts > /root/evidence/hf-download.log 2>&1 || /venv/main/bin/huggingface-cli download kyutai/pocket-tts >> /root/evidence/hf-download.log 2>&1
SNAP=$(ls -d /root/.cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/* | head -1)
echo "snapshot: $SNAP"; ls $SNAP
mkdir -p models
for d in $(ls $SNAP); do
  [ -d "$SNAP/$d" ] || continue
  echo "== convert $d"; /venv/main/bin/python tools/convert_pocket.py --language $d --output models/pocket-$d 2>&1 | tail -4
done
ls models; for m in models/*/; do echo "$m: $(python3 -c "import json;m=json.load(open(\"$m/model.json\"));print(m.get(\"transformer_layers\"), m.get(\"dtype\"))")"; done
echo MODELS-DONE
