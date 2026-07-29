# Performance

RTF = synthesis time ÷ audio duration; **below 1.0 is faster than real time**.
All figures are synthesis-only, median of five serial runs after two warmups,
on the standard bench prompt (`make bench`, 20 steps, speaker 4, seed 42).

## Reference numbers

| Device | ISA / backend | Precision | RTF | Measured |
|---|---|---|---|---|
| NVIDIA RTX 4060-class (~270 GB/s) | CUDA + cuBLAS | f32 / FP16 weights | **0.257** | 2026-07-25 |
| Apple M1 | ARM64 + Accelerate | **int8** | **0.361** | 2026-07-29 |
| Apple M1 | ARM64 + Accelerate | f16 | 0.495 | 2026-07-29 |
| Apple M1 | ARM64 + Accelerate | f32 | 0.662 | 2026-07-29 |
| Apple M1 | Metal | f32 | 0.723 | 2026-07-29 |
| x86-64 | AVX2 / FMA | — | not measured | kernels exist, never run |
| ARM64 server (Grace, Graviton) | NEON / SVE | — | not measured | server-class ARM only |

On a longer 6.0 s utterance the M1 numbers improve, because the fixed prep and
codec cost amortizes: int8 **0.243**, f16 0.376, f32 0.532.

Compare only within one benchmark. Absolute RTF drifts a few percent between
batches on the same machine, so a difference under ~3% taken minutes apart is
noise, not a result.

**ARM64 is measured** — the Apple M1 rows above are ARM64/NEON. What is missing
is *server-class* ARM (Grace, Graviton), which has different cache and bandwidth
behaviour and would need its own run.

The x86 row is empty for a specific reason worth recording, because it looks
like it should be fillable. The x86 kernels exist and have been optimized
(`perf: vectorize x86 GELU`, `perf: batch x86 matvec rows`, `perf: optimize x86
codec and matvec paths`) — but **those commits carry no measurements**, and no
x86 RTF for this model exists anywhere: not in the commit bodies, not in the
docs, not in any session log. The optimizations were written and reasoned about,
never benchmarked.

The box to run them on is known: an **AMD Ryzen 7 6800H** (Zen 3+, 8C/16T,
AVX2 + FMA3 + BMI2, **no** AVX-512, no AVX-VNNI) reachable on the LAN, building
under WSL2 for the Linux/OpenBLAS path. `qwen-tts` was benchmarked there; this
runtime never has. Note this also means the AVX2 path's *correctness* is
unverified on real hardware — run `make self-test` and read it before reading
any RTF.

Do not fill these rows from a sibling project: `qwen-tts` figures describe a
different model and say nothing about Magpie.

## Why decode is bandwidth-bound

This is the fact the rest of the document rests on. Decode is limited by how
fast weights can be read from DRAM, not by arithmetic.

The local transformer streams about **1.0 GB of weights per stacked frame**: 16
sequential streams over 2 layers, each re-reading the full weight set (16 ×
28.3 MB), plus one 6.2 MB output projection per stream. On the M1 that measured
8.0 GB in 0.292 s ≈ 27.5 GB/s against a measured ceiling of ~59 GB/s across four
cores.

Two consequences follow, and both were confirmed by measurement:

- **More cores help** until the memory ceiling is reached.
- **Fewer weight bytes help** — which is what the quantized modes buy.

A different compute unit does not help at all. See below.

## Metal is not the faster backend on Apple Silicon

Measured GPU streaming bandwidth on M1, on a 256 MB resident buffer, sweeping
thread counts to avoid under-measuring:

| | peak |
|---|---|
| GPU, f32 loads | 56.1 GB/s |
| GPU, f16 loads | 55.2 GB/s |
| **CPU, 4 threads** | **59.4 GB/s** |

Apple Silicon memory is unified: the GPU draws on the same ceiling as the CPU
and cannot exceed it, so it adds dispatch overhead and nothing else. The
full device-resident autoregressive graph measured RTF 1.214 against a CPU
0.862 — 42% slower, before counting that it also breaks token/EOS parity.

The Metal backend is kept for correctness coverage, for the backend seam CUDA
shares, and because a GPU-first engine (dots.tts) will need it. Prefer
`--device cpu` for latency on Apple Silicon.

The complete autoregressive Metal graph stays behind
`MYNAH_METAL_GPU_ATTENTION=1` as a diagnostic: small floating-point differences
change greedy tokens and EOS, so its benchmark is not a valid speed result.

## Threading

Decode projections are single-row, and Accelerate never threads an `M=1` sgemm,
so each one used to run on one core's share of memory bandwidth — a thread sweep
moved total synthesis from 0.602 s to 0.593 s, i.e. not at all.

Splitting the output rows over the pool **reorders no arithmetic**: every output
row is an independent dot product, so no reduction is split. It is the default
whenever the pool has more than one worker; serial Accelerate still wins at one
thread.

Output was verified **byte-identical** (same WAV md5 at 1/2/4/8 threads) on
macOS/Accelerate. That is a property of the build as much as the algorithm:
`-ffast-math`, and `-mfma` on x86, let a compiler contract or reassociate the
per-row scaling differently depending on how it shapes the loop, so last-bit
identity is not guaranteed across every compiler. The self-tests assert
equivalence to a tight tolerance for that reason.

The pool defaults to the **performance-core count** on Apple Silicon. On M1
(4P+4E) four threads measure 0.482 s against 0.507 s for all eight: the
efficiency cores add no bandwidth but lengthen every barrier.

Snake in the codec is threaded the same way, one channel per worker, with the
same caveat: 0.358 s → 0.099 s, byte-identical on macOS.

| variable | effect |
|---|---|
| `MYNAH_THREADS=N` | pool size; default = performance cores |
| `MYNAH_CPU_MATVEC=parallel` | force the row split |
| `MYNAH_CPU_MATVEC=1` | force the serial SIMD matvec |
| `MYNAH_CPU_MATVEC=0` | restore Accelerate-only behaviour, for A/B |

## Measured improvements

Interleaved A/B on the same binary, three repetitions alternating within one
batch, because absolute RTF drifts between batches:

| change | before | after | delta | output |
|---|---|---|---|---|
| threaded single-row matvec + P-core pool | 0.691 | 0.579 | −16% | byte-identical |
| threaded CPU Snake | 0.285 | 0.254 | −11% | byte-identical |
| route decode through qmat (int8) | 0.517 | 0.279 | −46% | quantized |
| **overall: f32 baseline → int8** | **0.625** | **0.243** | **−61% (2.6×)** | quantized |

The CUDA campaign (2026-07-25) moved the reference GPU from RTF 14.2 to 0.257:
cuBLAS backend, FP16 weight cache, GPU Snake, fused im2col+SGEMM conv1d (codec
−53%), mapped-buffer zero-copy output, and a `k_bias_add` kernel replacing
`cublasSger`. It also fixed a critical out-of-bounds in `k_layer_norm`.

A 0.209 was reached on CUDA with a GPU GELU but **reverted**: CPU `tanhf` is not
reproducible on CUDA and every variant shifted EOS in the autoregressive loop,
so 0.209 is not a valid result. Closing that blocker is worth roughly
0.257 → 0.20.

## Codec

With Snake threaded, what remains is causal convolution (~0.45 s of a 0.55 s
codec on the 6.0 s utterance). Those use BNNS on macOS.

`MYNAH_CODEC_SGEMM=1` selects an im2col+SGEMM alternative, which is much slower
here — codec 1.238 s against 0.593 s — and is kept for A/B only. Beating BNNS
needs a purpose-built threaded causal convolution, not a different library.

One trap worth knowing: `half_snake` in `graph.c` has a device branch guarded by
`backend != NULL`, and the CPU codec always has a backend object. The real CPU
path is `mynah_backend_snake_dev`'s fallback in `backend.c`; the vDSP code in
`graph.c` is only reachable when no backend is present, and
`MYNAH_SNAKE_SCALAR=1` toggles a branch a normal CPU synthesis never takes.

## Benchmark your own box

```bash
make bench MODEL_DIR=models/magpie-v2607-pack            # RTF, TTFA, RSS
make bench-matrix MODEL_DIR=models/magpie-v2607-pack     # f32 / int8 / int4
make info                                                # compiler, BLAS, SIMD
make self-test                                           # kernels correct on this ISA?
```

Report ISA, thread count, backend and model revision with any number.
Single-request latency and batched throughput are different metrics.
