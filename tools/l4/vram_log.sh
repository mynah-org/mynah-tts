#!/bin/bash
# Log GPU memory, GPU utilisation and the server's RSS every N seconds until
# killed.  usage: vram_log.sh <out-file> [seconds] [port]
out=$1; every="${2:-30}"; port="${3:-${MYNAH_L4_PORT:-18080}}"
while sleep "$every"; do
  pid=$(pgrep -f "[m]ynah-tts-server.*-p $port" | head -1)
  echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) rss=$( [ -n "$pid" ] && ps -o rss= -p "$pid")"
done >> "$out"
