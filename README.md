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
- Metal resident autoregressive self/cross-attention kernels with device-side
  KV caches and no CPU round-trip inside a decode step;
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
On Apple Silicon, `--device metal` uses the parity-safe hybrid path: transformer
matmuls stay on Accelerate while NanoCodec activations and convolutions remain
resident on Metal. The codec lowers causal convolutions through `im2col` plus
`MPSMatrixMultiplication`; weights and scratch buffers are cached. Set
`MYNAH_METAL_GPU_MATMUL=1` to diagnose direct per-op Metal matmuls, or
`MYNAH_METAL_CPU_CODEC=1` to force the CPU/BNNS codec.

The complete autoregressive Metal decoder/local-transformer graph exists but
remains experimental behind `MYNAH_METAL_GPU_ATTENTION=1`: small floating-point
differences currently change greedy tokens and EOS duration. Its benchmark is
therefore not a valid speed result.

**Metal is not the faster backend on Apple Silicon, and is not expected to
become one for this model.** Decode is bound by DRAM bandwidth, not by compute:
the local transformer alone streams about 1.0 GB of weights per stacked frame
(16 sequential streams over 2 layers, re-read from memory every stream), and the
measured single-core rate is far below the ~59 GB/s the M1 sustains across four
cores. Apple Silicon memory is unified, so that same ceiling is shared with the
GPU — the GPU has no bandwidth to add, only dispatch overhead. Measured on M1,
one 6.0 s utterance, AC power, one warmup and three serial runs, all five
configurations in a single back-to-back batch (absolute RTF drifts several
percent between batches on this machine, so only compare within one):

| path | RTF |
| --- | --- |
| CPU f32, 4 threads (default) | **0.542** |
| CPU int8, 4 threads | 0.554 |
| Metal, default threads | 0.625 |
| CPU f32, 1 thread | 0.658 |

The Metal backend is kept for correctness coverage, for the CUDA-shared backend
seam, and because a GPU-first engine (dots.tts) will need it. Prefer `--device
cpu` for latency on Apple Silicon.

Decode projections are single-row, and Accelerate never threads an `M=1` sgemm,
so each one used to run on one core's share of memory bandwidth. Splitting the
output rows over the thread pool is bit-exact — every output row is an
independent dot product, so no reduction is reordered — and is the default
whenever the pool has more than one worker. `MYNAH_CPU_MATVEC=1` forces the
serial SIMD matvec, `MYNAH_CPU_MATVEC=parallel` forces the split, and
`MYNAH_CPU_MATVEC=0` restores the previous Accelerate-only behaviour for A/B.
The pool defaults to the performance-core count on Apple Silicon (`MYNAH_THREADS`
overrides it); the efficiency cores add no bandwidth but lengthen every barrier.
`qmat.c` splits int8/int4 decode on the same bit-exact row blocks.

Alternating A/B on the 6.0 s utterance — old behaviour (`MYNAH_CPU_MATVEC=0
MYNAH_THREADS=8`) against the new default, same binary, three repetitions each,
interleaved so both arms see the same machine state — gives median RTF 0.691 →
**0.579 (−16%)**, with byte-identical output. Prefer this interleaved form over
comparing absolute numbers taken minutes apart.

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
