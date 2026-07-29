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

## Quantized decode

`MYNAH_QUANT=f16|int8|int4` converts the decode weights once at load and keeps
them in a name-keyed cache. Prefill is untouched: above 16 rows the call falls
back to the exact f32 BLAS path, so only the single-row autoregressive
projections are quantized.

The decoder and the local transformer both route their projections through this
cache. Until they did, only the local output projection was quantized -- roughly
a tenth of the decode weight traffic -- which is why `int8` used to be no faster
than `f32`.

Measured on M1, 6.0 s utterance, default threads, one warmup and three runs:

| mode | RTF | duration | LTAS corr vs f32 |
| --- | --- | --- | --- |
| f32 | 0.532 | 5.99 s | 1.000 |
| f16 | 0.376 | 6.22 s | 0.979 |
| int8 | **0.256** | 6.13 s | 0.973 |

(int8 includes the threaded Snake below; the f32/f16 figures predate it.)

**None of the quantized modes is a transparent replacement for f32.** All of
them perturb the greedy argmax enough to flip a near-tie token, and over ~65
autoregressive frames that shifts EOS: the duration moves by a few percent. The
output stays valid speech in the same voice -- long-term average spectrum
correlates at 0.97-0.98, RMS and peak match -- but it is a different realization
of the utterance, not the same waveform computed faster. Sample-wise correlation
is meaningless once the trajectory diverges and should not be used as the gate;
compare duration, finiteness, peak/RMS and a time-alignment-insensitive spectral
measure instead.

f16 is the most accurate of the three by a wide margin: on Magpie weights its
mean relative error is 1.8e-4 against 1.7e-2 for int8, about 96x tighter, and
the largest weight is ~6 against the 65504 f16 limit, so overflow is not a
concern. It keeps the activation in f32 and needs no scales. Prefer f16 when
duration drift matters and int8 when speed does.

`f32` remains the default and is bit-exact: routing it through the cache is a
no-op, verified by an unchanged WAV md5.

## Codec

Snake runs one channel per pool worker. Each channel is an independent row and
every step is elementwise, so the split is bit-exact; on M1 it takes Snake from
0.358 s to 0.099 s and the codec from 0.939 s to 0.584 s with the same WAV md5.

Note that `half_snake` in `graph.c` has a device branch guarded by `backend !=
NULL`, and the CPU codec always has a backend object — so the CPU path is
`mynah_backend_snake_dev`'s fallback in `backend.c`, and the vDSP code in
`graph.c` is only reachable when no backend is present. `MYNAH_SNAKE_SCALAR=1`
toggles a branch that a normal CPU synthesis never takes.

Causal convolutions use BNNS and are what remains of codec time (~0.45 s).
`MYNAH_CODEC_SGEMM=1` selects the im2col+SGEMM path, which is much slower here
(codec 1.238 s against 0.593 s) and is kept for A/B only.

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
