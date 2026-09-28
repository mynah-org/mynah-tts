#!/bin/bash
# Final qualification at C=$1: tooling check, two independent 30-min soaks with
# captured audio and VRAM log, ASR on the captures (server stopped), bundle.
C=$1; cd /root/mynah-head
flock /root/gpu.lock make cuda-server cuda CUDA_ARCH=sm_89 -j16 > /root/evidence/build-qual.log 2>&1; echo BUILD:$?
export MYNAH_L4_BATCH=$C MYNAH_L4_CORPUS=tools/corpus/pocket_v2_en.jsonl MYNAH_L4_VOICES=alba,marius,javert,jean MYNAH_L4_SAVE_EVERY=10
vmon() { ( while sleep 30; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) rss=$(ps -o rss= -p $(pgrep -f "[m]ynah-tts-server.*-p 18080" | head -1) 2>/dev/null)"; done ) > /root/evidence/$1-vram.log 2>&1 & echo $!; }
Q="/venv/main/bin/python tools/pocket_quality.py"
# 1) tooling check: 2 minutes, then ASR + bundle on it
M=$(vmon qcheck); MYNAH_L4_SEED=99 MYNAH_L4_SAVE_AUDIO=/root/evidence/qcheck/audio tools/l4/soak.sh qcheck $C 120; kill $M
flock /root/gpu.lock $Q --from-jsonl /root/evidence/qcheck/qcheck-c$C.jsonl 2>&1 | tail -15
tools/l4/bundle.sh check-c$C qcheck 2>&1 | tail -3
echo QCHECK-DONE
# 2) the two soaks
for s in "qualA 1234" "qualB 5678"; do set -- $s
  M=$(vmon $1); MYNAH_L4_SEED=$2 MYNAH_L4_SAVE_AUDIO=/root/evidence/$1/audio tools/l4/soak.sh $1 $C 1800; kill $M
  echo "$1-DONE"
done
# 3) ASR on the captures, server stopped
for t in qualA qualB; do flock /root/gpu.lock $Q --from-jsonl /root/evidence/$t/$t-c$C.jsonl 2>&1 | tail -20; done
tools/l4/bundle.sh pocket-24l-c$C-final qualA qualB 2>&1 | tail -5
du -sh /root/evidence/bundles/pocket-24l-c$C-final/*/audio-*.zip
echo QUAL-DONE
