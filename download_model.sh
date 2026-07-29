#!/bin/bash
# Download the Magpie TTS checkpoints from HuggingFace.
#
# All three repositories are public and ungated: no HuggingFace account, no
# token, no `huggingface-cli login`. Plain HTTPS is enough.
#
# Usage:
#   ./download_model.sh                  # everything needed to convert a pack
#   ./download_model.sh --what tts       # just the Magpie checkpoint
#   ./download_model.sh --what codec     # just NanoCodec
#   ./download_model.sh --what tokenizer # just the ByT5 tokenizer assets
#   ./download_model.sh --dir some/where # override the output root
#
# The weights are NOT redistributed by this project and are not covered by its
# MIT license: Magpie and NanoCodec are under the NVIDIA Open Model License,
# ByT5 under Apache-2.0. Downloading them means accepting those terms.

set -euo pipefail

ROOT="models"
WHAT="all"

usage() {
    cat <<'EOF'
Usage: ./download_model.sh [--what all|tts|codec|tokenizer] [--dir DIR]

  --what   which asset to fetch (default: all)
  --dir    output root directory (default: models)

After downloading, build a model pack with:
  make convert MODEL=models/magpie-v2607/magpie_tts_multilingual_357m.nemo \
               CODEC=models/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
               OUTPUT=models/magpie-v2607-pack
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --what) WHAT="$2"; shift 2 ;;
        --dir)  ROOT="$2";  shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$WHAT" in
    all|tts|codec|tokenizer) ;;
    *) echo "Unknown --what: $WHAT" >&2; usage >&2; exit 1 ;;
esac

if command -v curl >/dev/null 2>&1; then
    :
else
    echo "error: curl is required" >&2
    exit 1
fi

HF="https://huggingface.co"

# fetch <repo> <remote-path> <local-path>
fetch() {
    local repo="$1" remote="$2" dest="$3"
    if [[ -s "$dest" ]]; then
        echo "  have  $dest"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  get   $dest"
    # --fail so an HTML error page never lands on disk as if it were a weight.
    if ! curl -fL --progress-bar -o "$dest.part" "$HF/$repo/resolve/main/$remote"; then
        rm -f "$dest.part"
        echo "error: failed to download $remote from $repo" >&2
        exit 1
    fi
    mv "$dest.part" "$dest"
}

if [[ "$WHAT" == "all" || "$WHAT" == "tts" ]]; then
    echo "Magpie TTS 357M (NVIDIA Open Model License)"
    fetch "nvidia/magpie_tts_multilingual_357m" \
          "magpie_tts_multilingual_357m.nemo" \
          "$ROOT/magpie-v2607/magpie_tts_multilingual_357m.nemo"
fi

if [[ "$WHAT" == "all" || "$WHAT" == "codec" ]]; then
    echo "NanoCodec 22kHz (NVIDIA Open Model License)"
    fetch "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps" \
          "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo" \
          "$ROOT/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo"
fi

if [[ "$WHAT" == "all" || "$WHAT" == "tokenizer" ]]; then
    echo "ByT5-small tokenizer assets (Apache-2.0)"
    # Only the tokenizer metadata is needed; the ByT5 weights are not.
    for f in config.json generation_config.json special_tokens_map.json tokenizer_config.json; do
        fetch "google/byt5-small" "$f" "$ROOT/byt5-small-tokenizer/$f"
    done
fi

echo
echo "Done. Next:"
echo "  make convert MODEL=$ROOT/magpie-v2607/magpie_tts_multilingual_357m.nemo \\"
echo "               CODEC=$ROOT/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \\"
echo "               OUTPUT=$ROOT/magpie-v2607-pack"
