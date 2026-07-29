# Mynah TTS

[![Build & Test](https://github.com/mynah-org/mynah-tts/actions/workflows/build.yml/badge.svg)](https://github.com/mynah-org/mynah-tts/actions/workflows/build.yml)
[![Code Quality](https://github.com/mynah-org/mynah-tts/actions/workflows/codeql.yml/badge.svg)](https://github.com/mynah-org/mynah-tts/actions/workflows/codeql.yml)
[![Memory Safety](https://github.com/mynah-org/mynah-tts/actions/workflows/safety.yml/badge.svg)](https://github.com/mynah-org/mynah-tts/actions/workflows/safety.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Fast native C11 inference engine for text-to-speech — llama.cpp-style, no Python
at runtime. Today it runs NVIDIA MagpieTTS v2607 with NanoCodec; the engine seam
is built to host more models tomorrow.

**Faster than real time on a 2020 M1, CPU only** — RTF 0.36 at int8, no GPU
needed.

## Features

- **Pure C11, zero runtime dependencies.** One binary. Python is offline tooling
  only — conversion, tokenizer export, oracle parity.
- **CPU-first.** Tuned for Apple Silicon and x86; Metal and CUDA are optional,
  opt-in builds that fall back to CPU safely.
- **Quantized decode.** `f16`, `int8` and `int4` weights, converted at load.
  f32 stays the bit-exact default.
- **Faster than real time.** RTF 0.257 on a mainstream NVIDIA GPU, 0.361 on an
  M1 CPU. See [performance](docs/performance.md).
- **Offline and streaming** from one autoregressive state machine, with a
  callback API that is sample-identical to offline synthesis.
- **Memory-mapped weights**, reused scratch, no allocation in the decode loop.
- **Verified against the official NeMo oracle**, with per-stage parity tests and
  model-free kernel self-tests on every ISA.

## Performance

RTF = synthesis time ÷ audio duration; **below 1.0 is faster than real time**.

| Device | Backend | Precision | RTF |
|---|---|---|---|
| NVIDIA RTX 4060-class (~270 GB/s) | CUDA + cuBLAS | f32 / FP16 weights | **0.257** |
| Apple M1 | CPU (Accelerate) | **int8** | **0.361** |
| Apple M1 | CPU (Accelerate) | f16 | 0.495 |
| Apple M1 | CPU (Accelerate) | f32 | 0.662 |
| Apple M1 | Metal | f32 | 0.723 |

Decode is bound by memory bandwidth, not arithmetic — which is why quantization
is the big lever and why, on Apple Silicon's unified memory, the GPU is *slower*
than four CPU threads. The full analysis, the measured improvements and how to
benchmark your own box are in **[docs/performance.md](docs/performance.md)**.

## Build

```bash
make                 # CPU build, never needs a GPU SDK
make self-test       # kernel correctness on this ISA
make info            # compiler, OS, arch, BLAS, SIMD
```

```bash
./build/cpu/mynah-tts --inspect models/magpie-v2607-pack
./build/cpu/mynah-tts --synthesize models/magpie-v2607-pack \
  --text "hello from mynah" --lang en \
  --output build/native.wav --speaker 4 --seed 42
```

Optional GPU builds use separate object directories, so CPU/Metal/CUDA objects
can never be mixed:

```bash
make metal && build/metal/mynah-tts --gpu-self-test metal   # macOS
make cuda  && build/cuda/mynah-tts  --gpu-self-test cuda    # Linux/NVIDIA
```

Run quantized with `MYNAH_QUANT=int8` (or `f16` / `int4`) — see
**[docs/quantization.md](docs/quantization.md)** for the accuracy trade-offs.

## Model pack

The checkpoints are public and ungated — **no HuggingFace account or token
needed**:

```bash
./download_model.sh                # Magpie + NanoCodec + ByT5 tokenizer assets
./download_model.sh --what tts     # or fetch one at a time
```

The runtime never downloads weights implicitly and never loads a raw `.nemo`.
Convert once into a model pack:

```bash
make convert \
  MODEL=models/magpie-v2607/magpie_tts_multilingual_357m.nemo \
  CODEC=models/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  OUTPUT=models/magpie-v2607-pack

make tokenizer \
  MODEL=models/magpie-v2607/magpie_tts_multilingual_357m.nemo \
  CODEC=models/nano-codec-22khz/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  BYT5=models/byt5-small-tokenizer \
  OUTPUT=models/magpie-v2607-pack/tokenizer/english_phoneme.tsv
```

A pack carries `model.json`, the tts/codec safetensors, tokenizer assets,
speakers and license metadata. Model files, generated WAVs, build output and the
local `.venv` are all gitignored.

## Streaming

`mynah_tts_stream_open/push/flush/close` accept token chunks for long-form
input. `flush` drives the shared autoregressive graph and emits causal,
already-stable PCM prefixes through fixed-size callback chunks. The stream is
sample-identical to offline synthesis for the same request.

## Docs

- **[Performance](docs/performance.md)** — RTF tables, the bandwidth analysis,
  threading, the Metal verdict, benchmarking your own machine
- **[Quantization](docs/quantization.md)** — f16/int8/int4 trade-offs and how to
  judge quantized audio
- **[Oracle parity](docs/oracle-parity.md)** — validation against the official
  NeMo implementation

## License

This runtime is MIT licensed — see [LICENSE](LICENSE).

That covers the C runtime and tooling in this repository only. **Model weights
are licensed separately and are not distributed here**: the Magpie checkpoint is
under the NVIDIA Open Model License and NanoCodec under its own terms. Converting
a checkpoint into a model pack does not relicense it, and redistributing a pack
is a separate question from redistributing this code.
