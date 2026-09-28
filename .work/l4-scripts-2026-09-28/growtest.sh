#!/bin/bash
# Same request twice in a row per pool mode; PCM must be identical everywhere.
cd /root/mynah-head; ev=/root/evidence/grow; mkdir -p $ev
body="{\"model\":\"pocket\",\"input\":\"The quick brown fox jumps over the lazy dog, twice.\",\"voice\":\"alba\",\"response_format\":\"pcm\",\"stream\":true,\"seed\":42}"
other="{\"model\":\"pocket\",\"input\":\"When the old lighthouse was finally restored, the whole village came down to the harbour to watch the lamp being lit for the first time in forty years, and children climbed onto the sea wall while fishermen left their nets half mended.\",\"voice\":\"marius\",\"response_format\":\"pcm\",\"stream\":true,\"seed\":7}"
for mode in "MYNAH_CUDA_KV_GROW=0" "MYNAH_CUDA_KV_GROW=1 MYNAH_CUDA_KV_GROW_INITIAL_STEPS=16 MYNAH_CUDA_KV_GROW_LOG=1" "MYNAH_CUDA_KV_GROW=1"; do
  tag=$(echo $mode | tr " =" "__")
  pkill -f "[m]ynah-tts-server.*-p 18081"; sleep 2
  tmux new-session -d -s srv-18081 "tools/l4/serve.sh models/pocket-english-24l 64 64 18081 $mode > $ev/$tag.log 2>&1"
  for i in $(seq 90); do curl -sf localhost:18081/health >/dev/null && break; sleep 2; done
  curl -s -X POST localhost:18081/v1/audio/speech -H "Content-Type: application/json" -d "$body" -o $ev/$tag-a.pcm
  curl -s -X POST localhost:18081/v1/audio/speech -H "Content-Type: application/json" -d "$other" -o $ev/$tag-o.pcm
  curl -s -X POST localhost:18081/v1/audio/speech -H "Content-Type: application/json" -d "$body" -o $ev/$tag-b.pcm
  pkill -INT -f "[m]ynah-tts-server.*-p 18081"; sleep 3
done
md5sum $ev/*.pcm; ls -la $ev/*.pcm | awk "{print \$5, \$9}"
