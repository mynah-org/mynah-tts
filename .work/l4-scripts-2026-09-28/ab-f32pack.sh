#!/bin/bash
while ! grep -q ABTC-DONE /root/evidence/ab-tc.out 2>/dev/null; do sleep 10; done
S=$(ls -d /workspace/.hf_home/hub/models--kyutai--pocket-tts/snapshots/*)
cd /root/mynah-head
flock /root/gpu.lock /venv/main/bin/python tools/convert_pocket.py --source $S --language english_2026-04_24l --output models/pocket-english-24l-f32 --dtype source 2>&1 | tail -3
ls -la models/pocket-english-24l-f32/tts.safetensors
Q=/root/mynah-head/tools/l4/quality.sh
sed "s#models/pocket-english-24l #\${MYNAH_L4_MODEL:-models/pocket-english-24l} #" $Q > /tmp/q.sh && cp /tmp/q.sh $Q
MYNAH_L4_MODEL=models/pocket-english-24l-f32 $Q q-f32pack 64
$Q q-bf16pack 64
T=/root/mynah-head/tools/l4/ab.sh
$T f32pack models/pocket-english-24l-f32 32,64 5 20
$T bf16pack models/pocket-english-24l 32,64 5 20
echo ABF32-DONE
