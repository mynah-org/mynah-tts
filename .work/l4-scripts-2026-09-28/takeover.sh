#!/bin/bash
# When soak B ends, stop qual.sh before its slow ASR and redo ASR + bundle
# with more workers; then mark QUAL-DONE so the C8 control starts.
while ! grep -q qualB-DONE /root/evidence/qual.out 2>/dev/null; do sleep 3; done
tmux kill-session -t qual; sleep 2; pkill -f "[p]ocket_quality.py --from-jsonl /root/evidence/qual"
cd /root/mynah-head; C=160
for t in qualA qualB; do flock /root/gpu.lock /venv/main/bin/python tools/pocket_quality.py --from-jsonl /root/evidence/$t/$t-c$C.jsonl --asr-workers 24 --asr-threads 4 2>&1 | grep -v "FLAG req" | head -20; done
tools/l4/bundle.sh pocket-24l-c$C-final qualA qualB 2>&1 | tail -5
du -sh /root/evidence/bundles/pocket-24l-c$C-final /root/evidence/bundles/pocket-24l-c$C-final/*/audio-*.zip
echo QUAL-DONE >> /root/evidence/qual.out
echo TAKEOVER-DONE
