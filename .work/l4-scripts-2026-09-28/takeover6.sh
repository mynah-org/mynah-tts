#!/bin/bash
# When soak B of the 6L ends: stop qual6l.sh before its full ASR and C8 control,
# transcribe every 4th captured WAV of each soak (64 workers), then bundle.
while ! grep -q qual6B-DONE /root/evidence/qual6l.out 2>/dev/null; do sleep 3; done
tmux kill-session -t qual6l; sleep 2; pkill -f "[p]ocket_quality.py --from-jsonl /root/evidence/qual6"
cd /root/mynah-head; C=256
for t in qual6A qual6B; do
  j=/root/evidence/$t/$t-c$C.jsonl
  python3 -c "import json,sys;n=0
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get(\"wav\"):
        n+=1
        if n%4: continue
    print(l.rstrip())" $j > /root/evidence/$t/$t-asr-sample.jsonl
  echo "$t sample: $(grep -c wav /root/evidence/$t/$t-asr-sample.jsonl) of $(grep -c wav $j) captured"
  flock /root/gpu.lock /venv/main/bin/python tools/pocket_quality.py --from-jsonl /root/evidence/$t/$t-asr-sample.jsonl --out /root/evidence/$t/quality --asr-workers 64 --asr-threads 1 2>&1 | grep -v "FLAG req" | grep -v "BAD req" | head -6
done
MYNAH_L4_MODEL=models/pocket-english-6l tools/l4/bundle.sh pocket-6l-c$C-final qual6A qual6B 2>&1 | tail -3
ls -la /root/evidence/bundles/pocket-6l-c$C-final/*/audio-*.zip
echo TAKEOVER6-DONE
