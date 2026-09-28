#!/bin/bash
# Unloaded control for the qualification: the same request ids (same text,
# voice and seed as the first ~4000 of qualA) at C8, captured and transcribed.
while ! grep -q QUAL-DONE /root/evidence/qual.out 2>/dev/null; do sleep 30; done
cd /root/mynah-head
export MYNAH_L4_BATCH=8 MYNAH_L4_CORPUS=tools/corpus/pocket_v2_en.jsonl MYNAH_L4_VOICES=alba,marius,javert,jean MYNAH_L4_SAVE_EVERY=10
MYNAH_L4_SEED=1234 MYNAH_L4_SAVE_AUDIO=/root/evidence/ctrlA/audio tools/l4/soak.sh ctrlA 8 900
flock /root/gpu.lock /venv/main/bin/python tools/pocket_quality.py --from-jsonl /root/evidence/ctrlA/ctrlA-c8.jsonl 2>&1 | head -4
echo CTRL-DONE
