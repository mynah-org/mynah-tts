#!/bin/bash
. "$(dirname "$0")/compat.sh"
# Start the CUDA server with the qualified Pocket profile.
# usage: serve.sh <model-dir> <max-batch> <max-inflight> <port> [VAR=value ...]
# Extra VAR=value pairs override the profile (rollback switches, MYNAH_CUDA_QUANT, ...).
# MYNAH_GPU_CPUS, when set (e.g. "0-3"), confines the server to those cores.
cd "${MYNAH_GPU_ROOT:-/root/mynah-head}"
export MYNAH_THREADS=1 MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none MYNAH_SERVE_PROFILE=1
m=$1; b=$2; i=$3; p=$4; shift 4
for kv in "$@"; do export "$kv"; done
env | grep -E '^MYNAH_' | sort
pin=""; [ -n "$MYNAH_GPU_CPUS" ] && pin="taskset -c $MYNAH_GPU_CPUS"
exec $pin ./build/cuda/mynah-tts-server --device cuda -w "${MYNAH_GPU_WORKERS:-8}" \
  --max-batch "$b" --max-inflight "$i" -p "$p" -m "$m"
