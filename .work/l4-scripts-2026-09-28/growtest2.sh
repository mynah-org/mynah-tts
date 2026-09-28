#!/bin/bash
# Warm each voice first (the device voice cache loads on first use), then the
# compared requests; forced growth must log and match full capacity bit for bit.
cd /root/mynah-head; ev=/root/evidence/grow2; mkdir -p $ev
mk() { echo "{\"model\":\"pocket\",\"input\":\"$1\",\"voice\":\"$2\",\"response_format\":\"pcm\",\"stream\":true,\"seed\":$3}"; }
A=$(mk "The quick brown fox jumps over the lazy dog, twice." alba 42)
L=$(mk "When the old lighthouse was finally restored, the whole village came down to the harbour to watch the lamp being lit for the first time in forty years, and children climbed onto the sea wall while fishermen left their nets half mended." marius 7)
W1=$(mk "Warm up." alba 1); W2=$(mk "Warm up." marius 2)
for mode in "MYNAH_CUDA_KV_GROW=0" "MYNAH_CUDA_KV_GROW=1 MYNAH_CUDA_KV_GROW_INITIAL_STEPS=16 MYNAH_CUDA_KV_GROW_LOG=1" "MYNAH_CUDA_KV_GROW=1 MYNAH_CUDA_SLOT_POOL=0 MYNAH_CUDA_KV_GROW_INITIAL_STEPS=16 MYNAH_CUDA_KV_GROW_LOG=1"; do
  tag=$(echo $mode | tr " =" "__")
  pkill -f "[m]ynah-tts-server.*-p 18081"; sleep 2
  tmux new-session -d -s srv-18081 "tools/l4/serve.sh models/pocket-english-24l 64 64 18081 $mode > $ev/$tag.log 2>&1"
  for i in $(seq 90); do curl -sf localhost:18081/health >/dev/null && break; sleep 2; done
  for w in "$W1" "$W2"; do curl -s -X POST localhost:18081/v1/audio/speech -H "Content-Type: application/json" -d "$w" -o /dev/null; done
  curl -s -X POST localhost:18081/v1/audio/speech -H "Content-Type: application/json" -d "$A" -o $ev/$tag-a.pcm
  curl -s -X POST localhost:18081/v1/audio/speech -H "Content-Type: application/json" -d "$L" -o $ev/$tag-l.pcm
  curl -s -X POST localhost:18081/v1/audio/speech -H "Content-Type: application/json" -d "$A" -o $ev/$tag-b.pcm
  pkill -INT -f "[m]ynah-tts-server.*-p 18081"; sleep 3
  echo "$tag grow lines: $(grep -c "KV grew" $ev/$tag.log)"; grep "KV grew" $ev/$tag.log | head -3
done
md5sum $ev/*.pcm
