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

## Throughput: batching concurrent requests

Single-request latency and batched throughput are different metrics; this
section is the second one.

Decode reads far more weight bytes than it does arithmetic, so N requests taking
turns pay N trips to DRAM for the same weights. Stepping them together reads
each weight once and serves every request from cache.
`mynah_tts_synthesize_batch` (and the server, automatically) does this.

M1, int8, warm, same text with one seed per request:

| in flight | serial | batched | speedup | per request |
|---|---|---|---|---|
| 2 | 3.26 s | 2.52 s | 1.30x | 1.26 s |
| 4 | 6.48 s | 4.62 s | 1.40x | 1.16 s |
| 8 | 12.94 s | 7.94 s | **1.63x** | **0.99 s** |

All outputs byte-identical to the same request run alone.

**Single-request latency is unchanged** — batch 1 takes the same path and
produces the same bytes. This buys throughput, not latency.

### Why 1.6x and not 4x

A microbenchmark of one batched matvec shows 4.46x at B=8, and quoting that as
the end-to-end number would be wrong. The phase split says where the time
actually is:

| phase | share |
|---|---|
| prep (text encode + context prefill) | ~20% |
| AR decoder step | ~20% |
| AR local transformer | ~31% |
| codec | ~29% |

Batching covers the decoder and the local transformer — about half of wall
time — so Amdahl caps the gain near 1.6x, which is what was measured. Note the
local transformer is the larger of the two: it runs once per stacked stream, so
one decode step walks its weights sixteen times (252 MB against the decoder's
99 MB at int8). Going beyond 1.6x means batching the codec as well.

Beware `MYNAH_TIMING` when sizing any of this: instrumented runs inflate the
total by roughly 48% and attribute far too much to the encode phase. Use it for
proportions, never for a speedup estimate, and A/B at a fixed `--max-steps` so
both arms generate the same amount of audio.

## Streaming

Streaming decodes each chunk with a bounded window of history instead of
re-running the codec over the whole prefix. Measured on M1 (int8, 60 steps, the
`make stream-test` workload, which includes one offline pass):

| | wall time |
|---|---|
| re-decode the whole prefix | 7.91 s |
| bounded suffix, 32-frame window | 4.08 s |

The streaming portion alone goes from ~6.4 s to ~2.6 s. The gain grows with
utterance length because the cost changes from quadratic to linear: at ~160
steps the old path decoded about 26k frames instead of 320.

Output is byte-identical to offline; see `docs/server.md` for why 32 frames is
the right window and how it was verified.

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

### The codec is not the next batching lever

Worth recording because the phase table makes it look like one: the codec is
~29% of wall time, the largest single remaining block. Two things were measured
before building anything, and both say leave it alone.

**Batching it across requests buys ~1%.** The AR step is bandwidth-bound because
one activation row touches the entire weight set. The codec is not: its 405 MB
of weights are read once per call and applied to every frame in the sequence, so
they are already amortized ~320x. Codec time scales linearly with length (0.229 s
at 50 frames, 0.432 s at 100), and 405 MB in 0.43 s is 940 MB/s against the
~59 GB/s the machine sustains — it is compute-bound. Batching B requests would
save (B-1) x 405 MB of reads: about 0.05 s at B=8, against 3.4 s of arithmetic.

**Threading its convolutions ourselves makes it slower.** Codec conv time is
flat against `MYNAH_THREADS` (0.350 s at one thread, 0.353 s at four) while the
snake and transpose around it scale 3.4x, which looks exactly like three idle
cores. It is not: `MYNAH_THREADS` sizes *our* pool, which BNNS ignores — BNNS
already parallelizes internally. Splitting the convolution over output-channel
blocks and dispatching those on our pool was implemented, produced byte-identical
audio, and ran **slower**: codec 0.430 s to 0.536 s, with the middle stages worst
(stage 2 from 0.118 s to 0.194 s) because smaller per-call channel counts cost
BNNS more efficiency than the split returns. Reverted.

A real codec win would have to come from the kernels themselves, not from
scheduling.

## Benchmark your own box

```bash
make bench MODEL_DIR=models/magpie-v2607-pack            # RTF, TTFA, RSS
make bench-matrix MODEL_DIR=models/magpie-v2607-pack     # f32 / int8 / int4
make info                                                # compiler, BLAS, SIMD
make self-test                                           # kernels correct on this ISA?
```

Report ISA, thread count, backend and model revision with any number.
Single-request latency and batched throughput are different metrics.
