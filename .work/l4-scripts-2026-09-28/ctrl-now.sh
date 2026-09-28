#!/bin/bash
# The C8 control now, beside the CPU-only ASR of soak B: the GPU is idle and
# only the audio of the control matters, not its latency.
cd /root/mynah-head
export MYNAH_L4_LOCK=/root/ctrl.lock MYNAH_L4_BATCH=8 MYNAH_L4_CORPUS=tools/corpus/pocket_v2_en.jsonl MYNAH_L4_VOICES=alba,marius,javert,jean MYNAH_L4_SAVE_EVERY=10
MYNAH_L4_SEED=1234 MYNAH_L4_SAVE_AUDIO=/root/evidence/ctrlA/audio tools/l4/soak.sh ctrlA 8 900
/venv/main/bin/python tools/pocket_quality.py --from-jsonl /root/evidence/ctrlA/ctrlA-c8.jsonl --asr-workers 16 2>&1 | grep -v "FLAG req" | head -8
echo CTRL-DONE
