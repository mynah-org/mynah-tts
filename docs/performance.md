# Performance

RTF = synthesis time ÷ audio duration; **below 1.0 is faster than real time**.
All figures are synthesis-only, median of five serial runs after two warmups,
on the standard bench prompt (`make bench`, 20 steps, speaker 4, seed 42).

Every number below is for **`nvidia/magpie_tts_multilingual_357m` revision
v2607** paired with `nemo-nano-codec-22khz-1.89kbps-21.5fps` — 357M parameters,
22050 Hz, 21.5 frames/s. That is the only model shipping today, but RTF is a
property of the model as much as of the machine: quote the pair, never the
number alone, and do not carry these figures over to a future engine.

## Reference numbers

| Model | Device | ISA / backend | Precision | RTF | Measured |
|---|---|---|---|---|---|
| Magpie 357M v2607 | NVIDIA RTX 4060-class (~270 GB/s) | CUDA + cuBLAS | f32 / FP16 weights | **0.257** | 2026-07-25 |
| Magpie 357M v2607 | Apple M1 | ARM64 + Accelerate | **int8** | **0.361** | 2026-07-29 |
| Magpie 357M v2607 | Apple M1 | ARM64 + Accelerate | f16 | 0.495 | 2026-07-29 |
| Magpie 357M v2607 | Apple M1 | ARM64 + Accelerate | f32 | 0.662 | 2026-07-29 |
| Magpie 357M v2607 | Apple M1 | Metal | f32 | 0.723 | 2026-07-29 |
| Magpie 357M v2607 | AMD EPYC 9555P (Zen 5), 4 vCPU | x86-64 + OpenBLAS, AVX2 | **int8** | 0.427¹ | 2026-08-04 |
| Magpie 357M v2607 | AMD EPYC 9555P (Zen 5), 4 vCPU | x86-64 + OpenBLAS, AVX2 | f32 | 0.806¹ | 2026-08-04 |
| Magpie 357M v2607 | ARM64 server (Grace, Graviton) | NEON / SVE | — | not measured | server-class ARM only |

¹ Single warm `--synthesize` run of a short utterance, not the `make bench`
protocol above — treat the absolute values as indicative. The robust part is
the ordering: the int8 lane is a **1.9×** speedup over f32, consistent with the
bandwidth-bound analysis below.

On a longer 6.0 s utterance the M1 numbers improve, because the fixed prep and
codec cost amortizes: int8 **0.243**, f16 0.376, f32 0.532.

Compare only within one benchmark. Absolute RTF drifts a few percent between
batches on the same machine, so a difference under ~3% taken minutes apart is
noise, not a result.

**ARM64 is measured** — the Apple M1 rows above are ARM64/NEON. What is missing
is *server-class* ARM (Grace, Graviton), which has different cache and bandwidth
behaviour and would need its own run.

The x86 rows were filled on 2026-08-04, on a rented **AMD EPYC 9555P** (Zen 5)
cloud instance — 4 vCPU, 15 GB RAM, gcc 15.2, Linux/OpenBLAS — until then the
x86 kernels had been written and optimized but never run on x86 hardware. The
same pass verified *correctness*, not just speed: `make self-test` green on that
ISA, the pack converted on the box, and the f32/int8 WAVs validated by ear
against the Apple-Silicon output. Server-class ARM remains the one unmeasured
column.

**Which ISA those numbers actually ran on — corrected 2026-09-12.** An earlier
revision of this table credited them to AVX-512 VNNI. The host has AVX2,
AVX-512 F/DQ/BW/VL and AVX512-VNNI; the binary does not. Linux x86 builds
default to `-mavx2 -mfma` (`Makefile:22-46`), so AVX-512 was not even enabled,
and the int8 kernel is `dot_q8_i32_avx2` (`src/qmat.c:117-134`), which widens
int8 to int16 and accumulates with `_mm256_madd_epi16`. There is no `_mm512_*`
and no `vpdpbusd` anywhere in `src/`, and hand-written AVX2 intrinsics are never
promoted to VNNI by a compiler — `VPDPBUSD` is u8×s8 while the loop is s8×s8,
and the `-128·Σw` correction is not something a compiler invents. `SIMD=avx512`
adds compiler flags and selects no different kernel. The one AVX-512 consumer in
the run is OpenBLAS, which dispatches its own sgemm at runtime, so the f32
prefill uses it and the int8 decode lane does not.

The measurements stand; the attribution did not. **0.427 is what AVX2 alone
buys**, which makes it a floor for x86 rather than a ceiling. A VNNI/AMX lane is
unbuilt work (`PLAN.md` E4-5, E4-7).

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

### Threaded fused greedy argmax (f32) — accepted 2026-07-30

The fused local-transformer output projection (`mynah_qmat_greedy_argmax`) was
the last single-threaded consumer of decode weight bandwidth on the f32 greedy
path: ~6.2 MB of f32 projection weights per stream, sixteen streams per stacked
frame. It now splits the candidate rows into 32-row blocks over the worker
pool; each block keeps a private best and the reduction visits blocks in
ascending row order with a strict `>`, so the serial loop's first-on-ties
winner — and therefore the token trajectory — is preserved exactly.

M1 (T8103), AC power, f32 greedy (`--temperature 0 --topk 1`), 6.4 s utterance,
one process per arm, 2 warmups + 5 measured runs, three interleaved
repetitions, old arm via `MYNAH_ARGMAX_MT=0` from the same binary:

| rep | old median RTF | new median RTF |
| --- | --- | --- |
| 1 | 0.565 | 0.544 |
| 2 | 0.580 | 0.532 |
| 3 | 0.555 | 0.524 |
| median | 0.565 | **0.532** |

−5.8%, WAV byte-identical (same md5) in every pair. The change only affects
f32 greedy synthesis: sampling requests need the full logits and already used
the threaded batched projection, and quantized modes route through the
threaded qmat row split.

## The serving measurement protocol

Serving numbers are not synthesis numbers. A single-request RTF says nothing
about whether a listener's player stops, and this repo does not let the two be
confused.

**The metrics are defined once**, in `tests/playback_sim.py`: TTFB, TTFA,
STREAM_RTF, `required_prebuffer`, `safe_play_start`, `stall_rate` at
100/250/500/1000 ms, `max_gap` and the coalesced-read share. Every harness
imports them; nothing recomputes them. `python3 tests/playback_sim.py` (or
`make playback-sim-test`) runs 122 known-answer checks over synthetic timelines
in under a second and needs no model, no server and no network.

**Which metric gates.** STREAM_RTF is a *capacity* metric: mandatory `< 1`,
preferred `<= 0.90`, and never allowed to promote anything on its own.
`required_prebuffer` p95 and `stall_rate@250ms` are what *qualify*. The
self-test carries the case that forces the distinction — a timeline whose
STREAM_RTF is 0.800, passing the mandatory, preferred *and* strong RTF gates,
on which a player with a 250 ms jitter buffer still runs dry. Its verdict is
NOT STREAMABLE. If STREAM_RTF could promote, that configuration would ship.

**WAVE and SOAK are different measurements and the tool refuses to conflate
them.**

| | WAVE | SOAK |
|---|---|---|
| shape | C requests fired at t=0, repeated | C in flight continuously for minutes |
| warm-up | none | `--warmup-seconds`, discarded |
| drift gate | none | last window vs best, per metric |
| authority | **screen: may disqualify, never promote** | **qualification: the only mode that may promote** |

The gap is not cosmetic. In the qwen-tts reference a configuration passed the
wave screen at STREAM_RTF 0.919 and failed a 30-minute soak at 1.004 with 596
rejects and 111 broken pipes: *"the hard-capacity boundary, not a product
point."* Capacity is the highest GOOD concurrency with no gap below it —
discovered, not prescribed, and a GOOD level sitting above a MARGINAL one is a
measurement to explain rather than a product point.

**The tool declares its refusals and exits non-zero.**

- **Coalesced reads (exit 3).** A client mark is stamped when `read()` returns.
  A late reader finds several chunks already queued and returns them
  microseconds apart, so N server emissions become N marks in one instant.
  Above a 15% coalesced share the cadence percentiles describe the *client*, and
  the harness refuses to print them. The reference declared a run at 33-37% not
  quotable.
- **Dispatch resolution (exit 4).** Two arms may only be differenced when
  engine, ISA, SIMD, backend, quantization, thread count, build flags, route and
  sink all match. A fact that is *unrecorded in every arm* also refuses: equality
  of two unknowns is not sameness. `/health` today reports none of ISA, SIMD,
  backend or build flags, so that caveat is printed by name on every sweep.

### The decoder quantum

The emit quantum is a first-class serving parameter and is chosen on prebuffer
and stall, never on RTF. It lives in the **model pack**, not the CLI:
`model.json: audio_emit_frames`, read at load time
(`src/engine_pocket.c`, default 1; the gate is `src/inference.c:128`). It is not
a flag, not an environment variable and not a request field, so
`--quantum-sweep` materialises one pack variant per value — `model.json`
rewritten, every other file symlinked — and restarts the server per arm. Arms
run **interleaved** (`q1 q4 q16 q16 q4 q1`), because drift between two identical
runs on a shared box can exceed the effect under test.

We pay no correctness penalty for a small quantum: the codec carries state, so
chunked-vs-one-shot error stays at 1e-7 down to a one-frame chunk (`.work`
E2-3). The cadence knob is free for us in a way it was not for the reference,
where the smallest quantum *failed* on RTF and the largest *passed* on RTF while
stalling half the time.

**What the sweep cannot reach.** `server/main.c` `STREAM_CHUNK` (4096 samples)
bounds each enqueue, but `server/stream_out.c:169` drains the *whole* queued
span into one HTTP chunk, so it is not a delivery ceiling — delivery
granularity follows the decode quantum. Delivering increments *smaller* than the
decode quantum, or pacing them, would need a capped or paced writer span in
`server/stream_out.c` and `STREAM_CHUNK` as a runtime parameter. Both are in
`server/`.

### Running the campaign on Linux

No serving numbers are recorded here yet. **Every number this protocol produces
on macOS is a development signal, not a production claim** — Accelerate and the
P-core thread-pool default do not exist on Linux, and `SIMD=auto` on Linux x86
compiles plain AVX2 with no runtime dispatch. Production is Linux x86-64 and
ARM64, and the campaign belongs there. The harness prints the platform caveat on
every report.

Note that the quantization default changed at `d1ffd01`: per-tensor groups,
codec in int8 and backbone/flow in f16. Any earlier serving number in this repo
was taken against a different configuration and is not comparable.

```bash
# 0. build, and prove the metric definitions before trusting any of them
make server
make playback-sim-test                     # 122 known-answer checks, no model needed

# 1. SCREEN the levels (minutes). Drops levels that cannot work.
make serving-wave MODEL_DIR=models/pocket-en LEVELS=1,2,4,8 WAVES=3 \
  PROFILE_ARGS="--json build/wave.json"

# 2. SWEEP the decoder quantum at the best screened level, interleaved.
#    Chosen on prebuffer and stall@250, never on STREAM_RTF.
make serving-quantum-sweep MODEL_DIR=models/pocket-en \
  QUANTA=1,2,4,8,16 SWEEP_LEVEL=4 SWEEP_REPEATS=2

# 3. QUALIFY with a soak at each surviving level. Only this may promote.
#    Bake the chosen quantum into the pack's model.json first.
make serving-soak MODEL_DIR=models/pocket-en LEVELS=1 SOAK_SECONDS=1800
make serving-soak MODEL_DIR=models/pocket-en LEVELS=2 SOAK_SECONDS=1800
make serving-soak MODEL_DIR=models/pocket-en LEVELS=4 SOAK_SECONDS=1800 \
  PROFILE_ARGS="--json build/soak-c4.json"
make serving-soak MODEL_DIR=models/pocket-en LEVELS=8 SOAK_SECONDS=1800

# 4. A/B two configurations. REFUSES (exit 4) if the dispatch differs.
python3 tools/serving_profile.py --compare build/soak-c4.json build/soak-c4-b.json
```

Record with every cell: model revision, thread count (`MYNAH_THREADS`), ISA,
backend, build flags, quantization, CPU mask, machine, and the exit code. A
NOT QUOTABLE level is not a slow level — it is a level that was not measured.

## Benchmark your own box

```bash
make bench MODEL_DIR=models/magpie-v2607-pack            # RTF, TTFA, RSS
make bench-matrix MODEL_DIR=models/magpie-v2607-pack     # f32 / int8 / int4
make info                                                # compiler, BLAS, SIMD
make self-test                                           # kernels correct on this ISA?
```

Report ISA, thread count, backend and model revision with any number.
Single-request latency and batched throughput are different metrics.

## First production-hardware capacity screen — 2026-09-13

**Host** GCP Axion, 32x Neoverse-V2, SMT off, 80 MiB L3 (one instance), one NUMA
node, gcc 15.2, `BLAS=none/mynah-sgemm`, `SIMD=auto`.
**Build** `565e5c9`. **Pack** `models/pocket-en`, PocketTTS, default quantization
(codec int8, backbone and flow f16). **Topology** `--prefork 4 --prefork-threads 8
--max-batch 16`. **Mode** WAVE, three synchronised waves per level.

| C | completed | TTFB p95 | TTFA p95 | STREAM_RTF p95 | prebuffer p95 | max gap p95 | stall@250 |
|---|---|---|---|---|---|---|---|
| 1 | 3/3 | 0.2 ms | 62 ms | 0.124 | 0 | 10 ms | 0% |
| 4 | 12/12 | 0.2 ms | 137 ms | 0.130 | 0 | 11 ms | 0% |
| 8 | 24/24 | 125 ms | 207 ms | 0.240 | 0 | 20 ms | 0% |
| 16 | 48/48 | 200 ms | 331 ms | 0.468 | 0 | 39 ms | 0% |
| 20 | 60/60 | 269 ms | 509 ms | 0.581 | 0 | 48 ms | 0% |
| 24 | 72/72 | 337 ms | 591 ms | 0.691 | 0 | 57 ms | 0% |
| 30 | 90/90 | 500 ms | 683 ms | **0.922** | 0 | 76 ms | 0% |

**Every mandatory gate passes at every level through C30**: completed equals
launched, STREAM_RTF p95 below 1.0, and no stall at any buffer depth. No request
was rejected, timed out or disconnected, and required prebuffer was zero
throughout — the server never made a player wait.

**Nothing here is promoted.** A wave is a screen: it may disqualify a
configuration and may never promote one. The operating point needs a SOAK at the
candidate level with a drift gate across windows, and that has not been run.

**The cadence percentiles are not quotable** and the harness says so rather than
printing them as fact: 27% to 100% of client reads returned already-queued data,
because generation outruns the reader. The share falls monotonically as
concurrency rises (100% at C1, 54% at C8, 27% at C30), which is itself the
evidence that the server is running well above real time. The columns that *are*
quotable — completion, STREAM_RTF, TTFA, TTFB — do not depend on read granularity.

### What stops C20 and C30 from being GOOD, and it is not synthesis

The preferred gates that fail are **TTFB p95** and **TTFA p95**, not RTF and not
stalls. TTFB is the time to the response header, which this server sends at
admission, before any audio is generated — so a TTFB of 500 ms at C30 is
**admission latency**, not synthesis. With four workers at sixteen slots each
there are 64 slots for 30 arrivals, so nothing is queueing for capacity.

The suspect is the serialised admission the engine inherits: one pending
admission per worker, installed inside the frame loop. Thirty simultaneous
arrivals then queue behind one admission per iteration per worker. That is a
known lever with a known shape — the reference implementation reached for sliced
admission and for a prefill helper, and measured the helper making TTFA five times
worse — so it needs measuring here, not copying.

STREAM_RTF p95 0.922 at C30 also sits above the preferred 0.90 while under the
mandatory 1.0, which is the ordinary shape of a level that is at its edge.

### Reading this against the target

C20 is comfortably inside every mandatory gate with STREAM_RTF p95 0.581 — a
stream generated at better than one and a half times real time while twenty run
together. C30 still completes every request with no stall, at 0.922. The honest
statement is that **C20 is reached and C30 is at the edge**, and that the next
work is admission latency rather than kernels.

Untested: any other topology (2x16, 1x32, 8x4 were not swept), any other text
length distribution, x86, and sustained load.

## 2026-09-13 · PocketTTS after the three profile lanes — GCP Axion, 32 cores

Same host and same topology as the topology sweep above (`--prefork 16
--prefork-threads 2 --max-batch 16`, `BLAS=none`, `SIMD=auto`), so the two are
comparable. Build: the merged tree carrying the prefill prepack, the conv-stack
fusion and the pool meter. FAST screen, three waves per level, quiet box.

| C | done/launched | TTFB p95 | TTFA p95 | STREAM_RTF p95 | prebuffer p95 | **safe-to-play p95** | stall@500 |
|---|---|---|---|---|---|---|---|
| 1 | 3/3 | 0.3 ms | 155 ms | 0.229 | 0 | **155 ms** | 0% |
| 48 | 144/144 | 489 ms | 890 ms | 0.680 | 0 | **890 ms** | 0% |
| 64 | 192/192 | 662 ms | 1001 ms | 0.895 | 0 | **1001 ms** | 0% |
| 80 | 240/240 | 805 ms | 1230 ms | **1.120** | 434 ms | **1638 ms** | 8% |
| 100 | 300/300 | 974 ms | 1490 ms | **1.537** | 1539 ms | **2994 ms** | 100% |

Every request completes at every level, including C100. What fails is cadence,
not completion.

### What the three lanes bought, and what they did not

| level | STREAM_RTF p95 before | after | |
|---|---|---|---|
| C48 | 0.736 | **0.680** | −7.6% |
| C64 | 0.954 | **0.895** | −6.2% |

C64 moves from the edge of the mandatory gate to inside it with margin, and C80
becomes the first level that fails. That is one level of capacity, bought
without a kernel rewrite and without touching the arithmetic.

**Latency did not move at all.** Safe-to-play p95 at C48 was 908 ms before and
is 890 ms now. The single-stream wall fell 17% and the served latency fell 2%,
which is the entire story of this screen: the lanes removed *work*, and what
gates a served request at this concurrency is *waiting*.

TTFB p95 — the response header, which this server sends at admission, before a
single sample exists — is 489 ms at C48 and 974 ms at C100. Synthesis has not
started when that clock stops. With sixteen workers at sixteen slots there are
256 slots for 100 arrivals, so nothing is queueing for capacity, and the warm
prefill measured on this build is ~122 ms. The gap between 122 ms of work and
890 ms of safe-to-play is admission and scheduling.

### Reading this against C100

**C100 is not reached and this build does not get there.** The honest statement
is C64 with margin on the mandatory gates, C80 as the first failure, and
STREAM_RTF 1.537 at C100 — a stream generated at two thirds of real time, which
no player can absorb. Nothing here is promoted: a wave screen may disqualify a
configuration and may never promote one, so C64 needs a SOAK before it is an
operating point.

The next lever is not the kernels. At `16x2` each worker has two threads, where
the pool meter puts the barrier at 1.2% — the pool has nothing left to give at
this width. The queue does.

Untested here: x86, other topologies on this build, mixed-language load, and
sustained load at any level.

## 2026-09-18 · The first qualified operating point — GCP Axion, 32 cores

Host: GCP Axion, 32x Neoverse-V2, 62 GB, Linux 7.0, gcc 15.2, `BLAS=none`.
Pack `models/pocket-en`, 24000 Hz. Bank `tests/load_texts_en_v2.txt`
(`bank-sha256 55b286369ef8dd44`, 277 texts, all five classes, audio 1.04-19.04 s
with sd 4.45). Load generator on the same box, 13% coalesced reads.

**C90, thirty minutes, 53265 requests, every gate passed — GOOD.**

    TTFB p95   82.0 ms      TTFA p95  494.8 ms     STREAM_RTF p50/p95  0.659/0.734
    prebuffer p50/p95 0/10 ms (max 267)            safe_play_start p95  527 ms
    max_gap p50/p95 91/129 ms (max 177)            throughput  128.9 audio-s/s
    stall_rate@500ms  0 of 53265                   stall_rate@250ms  0 of 53265
    drift: rtf +0.0040, prebuffer +0.0000, over ten 3-minute windows

### The exact configuration

    server:  --prefork 16 --prefork-threads 2 --max-batch 8

    MYNAH_QUANT_GROUPS=codec_transformer:int8,codec_conv:int8,codec_convtr:int8,\
                       backbone:f16,flow_net:f16,conditioner:f16
    MYNAH_PREFILL_SLICE=32        # tokens of prefill per slice (default)
    MYNAH_PREFILL_STEP_MS=60      # ms of prefill work per step  (default)

`codec_convtr:int8` is NOT the shipped default. Every number above was measured
with it ON; it is worth 1.86x on the per-slot term at serving thread counts and
costs SNR 28.9-32.9 dB against 36.4-37.9, log-mel nearly unchanged. Until that
default is flipped, this table describes a configuration the binary does not pick
on its own.

### What each knob is for

| knob | what it bounds | measured effect |
|---|---|---|
| `MYNAH_PREFILL_SLICE` | tokens per slice of a resumable prefill | 0 restores the one-shot prefill; bit-identical audio at any value |
| `MYNAH_PREFILL_STEP_MS` | total prefill work per AR step | 0 removes the cap; the cap is what makes zero stalls a bound rather than a statistic |
| `--prefork N --prefork-threads T` | worker shape | `16x2` beats `8x4` decisively here; the optimum is a property of the model's a/b ratio, not of the machine |
| `--max-batch` | slots per worker | capped at 16 by `MYNAH_GRAPH_MAX_JOBS`; not binding at C90 (~6 live per worker) |

### Against the same soak before the work

| | before | after |
|---|---|---|
| highest qualified level, mixed bank | **none** (C99/C98/C96 all NOT STREAMABLE) | **C90 GOOD** |
| `stall_rate@500ms` | 5 of 53895 | **0 of 53265** |
| `stall_rate@250ms` | 233 of 53895 | **0 of 53265** |
| worst freeze (`max_gap` max) | 707 ms | **177 ms** |
| worst client prebuffer | 535 ms | **267 ms** |
| throughput at equal concurrency | 130.5 | 129.6 audio-s/s (-0.7%) |

The client contract that was an interim "600 ms prebuffer" is now **250 ms**, and
it is a scheduler bound rather than a sample maximum.

### What binds next

Not the machine. `STREAM_RTF` p95 is **0.734** at C90, so the box delivers a third
faster than realtime with headroom to spare. The gate that stops a higher
concurrency is **TTFA at 494.8 ms against 500**, which is the price of slicing the
prefill. The lever behind it is the ABSOLUTE cost of a prefill -- 190 ms for a
long text at C1 with no contention, against 28 ms of fixed cost -- and nobody has
optimised it. The detail, including the three predictions of mine that the box
refuted, is in `.work/prefill-blocks-decode.md`.

## 2026-09-19 · The ceiling, and the configuration that reaches it — GCP Axion, 32 cores

Same host, same bank, same protocol as the section above. Three questions were
asked and all three were answered by measurement rather than by argument.

### 1. `codec_convtr:int8` is not an optimisation, it is the difference between a product point and none

Two ten-minute soaks at C90, identical in every respect but this one clause:

| C90, ten minutes | `codec_convtr` OFF (the shipped default until today) | ON |
|---|---|---|
| verdict | **MARGINAL** | **GOOD** |
| TTFA p95 | 544.2 ms — FAIL against 500 | 492 ms |
| `STREAM_RTF` p95 | 0.842 | 0.728 |
| `stall_rate@250ms` | 5 of 15380 — FAIL | **0 of 17904** |
| throughput | 106.5 audio-s/s | 124.3 audio-s/s |

With the clause absent there is **no qualified concurrency at all** on this
machine. The default is now ON (`POCKET_QG_DEFAULT_SPEC_PINNED`), and the cost is
unchanged and known: SNR 28.9-32.9 dB against 36.4-37.9, log-mel nearly unchanged.

**Every number in the section above was measured with this clause exported by
hand while the binary shipped it OFF.** That gap is closed, and the machinery
that makes it a validation failure rather than a footnote is in
`configs/perf/` — see `.work/serving-profiles.md`.

### 2. The ceiling is C96, and above it the failure changes character

| C, convtr ON | 10 min | 30 min | TTFA p95 | RTF p95 | stall@250 |
|---|---|---|---|---|---|
| 90 | GOOD | **GOOD** (09-18) | 494.8 | 0.734 | 0 of 53265 |
| 94 | — | MARGINAL | 494.9 | 0.746 | **1** of 53980 |
| 96 | GOOD | MARGINAL | 490.4 | 0.754 | **1** of 54186 |
| 98 | MARGINAL | — | 499.5 | 0.800 | 13 of 18067 |
| 100 | MARGINAL | — | 522.6 | 0.817 | 22 of 18056 |

Two things to read here, and neither is the obvious one.

**A ten-minute screen promoted C96 and a thirty-minute soak refused it**, by one
stall in 54186 requests. The rule this repository already carried — *a screen may
not promote* — met its first real case within hours of being written down. The
same statistic lesson as 2026-09-18, from the other side: a rate that is zero in
18000 samples is not a bound, it is a small number that has not been given enough
chances yet.

**From C96 to C100, TTFA drifts 28 ms while stalls go 0 → 13 → 22.** The
per-step prefill cap is doing exactly what it promises, so what overruns the 80 ms
frame at the top of the range is no longer the prefill: it is the AR step itself,
as slots per worker rise from ~5.6 at C90 to ~6.25 at C100. Any further work on
"zero stalls at a higher C" belongs to the decode step or to the slot count per
worker, not to another prefill knob.

And the third reading, which is the useful one for capacity planning:
`STREAM_RTF` p95 is **0.817 at C100**, a level that cannot be qualified. The box
still has a fifth of a realtime budget in hand where continuity has already run
out. **Capacity is not what limits this engine; continuity is.**

### 3. The shipped binary reproduces the operating point on its own

C94, thirty minutes, **no `MYNAH_QUANT_GROUPS` in the environment at all**:
TTFA p95 494.9 ms, RTF p95 0.746, throughput 130.8 audio-s/s, `stall@500ms` 0 —
statistically the same run as the env-override arm (490.4 / 0.754 / 131.2). What
the documentation describes and what a plain binary does are now the same thing.

### The configuration, and where it lives now

    tools/perf_profile.py best recommended
    python3 tools/serving_profile.py --profile axion-c4a-32c-pocket-en \
            --model models/pocket-en --server-bin build/cpu/mynah-tts-server

`configs/perf/axion-c4a-32c-pocket-en.json` carries the topology
(`--prefork 16 --prefork-threads 2 --max-batch 8`), the gates, the operating point
and the levels above it that were measured and not promoted. It exports **nothing**:
all three environment knobs are declared *must be absent*, because the shipped
default now covers them, and a run that exports one of them refuses to start.
