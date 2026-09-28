#!/bin/bash
# After the 6L discovery: pick the top level that passes the gate, then run the
# same qualification as the 24L (two 30-min soaks, capture, ASR, bundle, C8 control).
while ! grep -q DISC6LB-DONE /root/evidence/disc6lb.out 2>/dev/null; do sleep 30; done
cd /root/mynah-head
C=$(python3 - <<PY
import json
best=None
for l in open("/root/evidence/disc6lb/disc6lb-summary.jsonl"):
    s=json.loads(l)
    if s["failed"]==0 and s["stalls_250"]==0 and s["rtf_stream"]["p95"] is not None and s["rtf_stream"]["p95"]<0.88 and (s["ttfa"]["p95"] or 9)<0.5:
        best=s["offered"]
print(best or 0)
PY
)
echo "6L qualification level: C$C"; [ "$C" = 0 ] && { echo "no level passed"; echo QUAL6L-DONE; exit 0; }
[ "$C" = 352 ] && echo "NOTE: C352 is the top screened level; the 6L knee was not reached"
export MYNAH_L4_MODEL=models/pocket-english-6l MYNAH_L4_BATCH=$C MYNAH_L4_CORPUS=tools/corpus/pocket_v2_en.jsonl MYNAH_L4_VOICES=alba,marius,javert,jean MYNAH_L4_SAVE_EVERY=10
vmon() { ( while sleep 30; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) rss=$(ps -o rss= -p $(pgrep -f "[m]ynah-tts-server.*-p 18080" | head -1) 2>/dev/null)"; done ) > /root/evidence/$1-vram.log 2>&1 & echo $!; }
for s in "qual6A 4321" "qual6B 8765"; do set -- $s
  M=$(vmon $1); MYNAH_L4_SEED=$2 MYNAH_L4_SAVE_AUDIO=/root/evidence/$1/audio tools/l4/soak.sh $1 $C 1800; kill $M
  echo "$1-DONE"
done
for t in qual6A qual6B; do flock /root/gpu.lock /venv/main/bin/python tools/pocket_quality.py --from-jsonl /root/evidence/$t/$t-c$C.jsonl --asr-workers 64 --asr-threads 1 2>&1 | grep -v "FLAG req" | head -20; done
tools/l4/bundle.sh pocket-6l-c$C-final qual6A qual6B 2>&1 | tail -3
du -sh /root/evidence/bundles/pocket-6l-c$C-final/*/audio-*.zip
export MYNAH_L4_BATCH=8
MYNAH_L4_SEED=4321 MYNAH_L4_SAVE_AUDIO=/root/evidence/ctrl6A/audio tools/l4/soak.sh ctrl6A 8 900
flock /root/gpu.lock /venv/main/bin/python tools/pocket_quality.py --from-jsonl /root/evidence/ctrl6A/ctrl6A-c8.jsonl --asr-workers 64 --asr-threads 1 2>&1 | grep -v "FLAG req" | head -6
echo QUAL6L-DONE
