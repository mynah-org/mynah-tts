# Mynah TTS

CPU-first C11 runtime for MagpieTTS and future cross-model TTS engines. The
first pinned target is NVIDIA MagpieTTS v2607 with NanoCodec. GPU execution is
explicitly opt-in through separate Metal or CUDA builds.

## Current v1 slice

- scalar float32 kernels with a deterministic self-test plus compiler-selected
  NEON/AVX2 matmul;
- WAV/PCM writer and CLI smoke command;
- archive-only NeMo inspection and Magpie/NanoCodec conversion;
- safetensors model pack with verified architecture metadata;
- C model-pack loader and metadata inspection;
- native C Magpie encoder/decoder, local AR refinement, and NanoCodec decoder;
- optional Metal and CUDA matmul backends used by transformer projections and
  kernel-1 FFN blocks, with resident cached weight buffers and GPU self-tests;
- official NeMo CPU oracle helper for offline reference WAV generation.

The runtime path is the compiled C binary only. Python is used offline for
checkpoint conversion, tokenizer export, and oracle/parity checks.

## Build

```bash
make
make self-test
make info
./build/cpu/mynah-tts --inspect models/magpie-v2607-pack
./build/cpu/mynah-tts --synthesize models/magpie-v2607-pack \
  --normalized 'h|ə|ˈ|l|o|ʊ|<space>|f|ɹ|ʌ|m' \
  --output build/native.wav --speaker 4 --max-steps 0 --seed 42

# A test tone is only a WAV-writer smoke test; it is not TTS.
./build/cpu/mynah-tts --write-test-wav build/smoke.wav

# Optional GPU builds; weights remain outside the repository.
make metal
build/metal/mynah-tts --gpu-self-test metal
build/metal/mynah-tts --synthesize models/magpie-v2607-pack \
  --normalized 'h|ə|ˈ|l|o|ʊ|<space>|f|ɹ|ʌ|m' \
  --output build/metal.wav --device metal --max-steps 32

make cuda
build/cuda/mynah-tts --gpu-self-test cuda
build/cuda/mynah-tts --synthesize models/magpie-v2607-pack \
  --normalized 'h|ə|ˈ|l|o|ʊ|<space>|f|ɹ|ʌ|m' \
  --output build/cuda.wav --device cuda --max-steps 32
```

The default `make` target never requires a GPU SDK. `make metal` requires
macOS and the Metal framework; the current implementation compiles the small
matmul shader through the Metal API at backend startup, so a full Xcode shader
toolchain is not required. `make cuda` requires `nvcc` and an NVIDIA runtime.
The GPU path currently offloads transformer matmuls and kernel-1 FFN matmuls;
attention score loops and the causal NanoCodec decoder remain on CPU so the
backend boundary stays numerically explicit.

## Convert the downloaded checkpoints

```bash
make inspect MODEL=models/magpie-v2607/magpie_tts_multilingual_357m.nemo
make convert \
  MODEL=models/magpie-v2607/magpie_tts_multilingual_357m.nemo \
  CODEC=models/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  OUTPUT=models/magpie-v2607-pack
make test MODEL_DIR=models/magpie-v2607-pack
make tokenizer \
  MODEL=models/magpie-v2607/magpie_tts_multilingual_357m.nemo \
  CODEC=models/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  BYT5=models/byt5-small-tokenizer \
  OUTPUT=models/magpie-v2607-pack/tokenizer/english_phoneme.tsv

make synthesize MODEL_DIR=models/magpie-v2607-pack \
  TEXT='h|ə|ˈ|l|o|ʊ|<space>|f|ɹ|ʌ|m' OUTPUT=build/native.wav SPEAKER=4 MAX_STEPS=0
```

All model files, generated WAVs, build output, and the local `.venv` are
ignored. The runtime never downloads weights implicitly.

## Callback streaming

The C API exposes `mynah_tts_stream_open/push/flush/close`. Token chunks can
be pushed for long-form input; `flush` drives the shared autoregressive graph
and emits causal, already-stable PCM prefixes through fixed-size callback
chunks. The callback stream is sample-identical to offline synthesis for the
same request. A future codec-state refactor can remove the current prefix
re-decode cost; see the streaming/design TODO in `PLAN.md`.

## Reference oracle

The oracle uses the official NeMo package only offline. Its local inference
environment is intentionally ignored:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python \
  'nemo_toolkit[tts] @ git+https://github.com/NVIDIA/NeMo.git@main'
make oracle \
  MODEL=models/magpie-v2607/magpie_tts_multilingual_357m.nemo \
  CODEC=models/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  BYT5=models/byt5-small-tokenizer \
  OUTPUT=build/magpie-oracle.wav
```

The downloaded Magpie model is under the NVIDIA Open Model License. Keep its
license and the runtime license separate when distributing model packs.

See [PLAN.md](PLAN.md) for the full v1/v2/v3 architecture and roadmap.
