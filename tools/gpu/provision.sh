#!/bin/bash
# Prepare a fresh GPU box for Pocket CUDA serving measurements.
# usage: provision.sh [CUDA_ARCH]      (run from the checkout, e.g. /root/mynah-head)
#
# - builds the CUDA server and CLI (CUDA_ARCH defaults to native, e.g. sm_89 for L4)
# - installs the Python tooling the harness needs (huggingface_hub, numpy,
#   safetensors, faster-whisper, jiwer) into $MYNAH_L4_PY's environment
# - downloads kyutai/pocket-tts with the read token from $HF_TOKEN or
#   /root/.hf_token (never from the repo) and converts the English 24L and 6L
#   packs to models/pocket-english-24l and models/pocket-english-6l
# Honours HF_HOME (vast.ai images point it at /workspace/.hf_home).
set -u
arch="${1:-${CUDA_ARCH:-native}}"
py="${MYNAH_L4_PY:-/venv/main/bin/python}"; [ -x "$py" ] || py=python3
root="$(pwd)"
[ -f Makefile ] && [ -d src ] || { echo "run from the mynah-tts checkout" >&2; exit 2; }

echo "== build (CUDA_ARCH=$arch)"
make cuda-server cuda CUDA_ARCH="$arch" -j"$(nproc)" > build-provision.log 2>&1 \
  && echo "BUILD:0" || { echo "BUILD failed, see build-provision.log" >&2; tail -20 build-provision.log; exit 1; }

echo "== python tooling"
"$py" -m pip install -q huggingface_hub numpy safetensors faster-whisper jiwer 2>&1 | grep -v "pip as the" | tail -2

echo "== model download"
if [ -z "${HF_TOKEN:-}" ] && [ -f /root/.hf_token ]; then
  # The file holds either the bare token or an `export HF_TOKEN=...` line.
  HF_TOKEN=$(sed -e 's/^export //' -e 's/^HF_TOKEN=//' /root/.hf_token | tr -d '[:space:]"')
  export HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] || echo "WARN: no HF token (gated repo); set HF_TOKEN or write /root/.hf_token" >&2
"$py" -c "from huggingface_hub import snapshot_download; print(snapshot_download('kyutai/pocket-tts'))" \
  > /tmp/pocket-snapshot.txt 2> /tmp/pocket-download.log || { tail -5 /tmp/pocket-download.log; exit 1; }
snap=$(tail -1 /tmp/pocket-snapshot.txt)
echo "snapshot: $snap"

echo "== convert packs"
mkdir -p models
for pair in "english_2026-04_24l pocket-english-24l" "english_2026-04 pocket-english-6l"; do
  set -- $pair
  [ -f "models/$2/model.json" ] && { echo "models/$2 exists"; continue; }
  "$py" tools/convert_pocket.py --source "$snap" --language "$1" --output "models/$2" 2>&1 | tail -3
done

echo "== check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
for m in models/pocket-english-24l models/pocket-english-6l; do
  MYNAH_CUDA_KV_DTYPE=bf16 MYNAH_QUANT_GROUPS=none timeout 900 ./build/cuda/mynah-tts \
    --pocket-self-check "$m" --device cuda 2>&1 | tail -1
done
echo PROVISION-DONE
