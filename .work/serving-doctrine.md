# Serving doctrine — direction from the author of qwen-tts

Status: **REFERENCE** · captured 2026-09-12 from the author of the reference
implementation, who reached **C20/C22 real-time concurrent streams on 32 ARM
cores with Qwen 0.6B**.

This note is not our design. It is what someone who already solved this problem
said, recorded before it gets diluted into "we should optimize". The companion
note [serving-design.md](serving-design.md) extracts the mechanisms; this one
carries the *ranking* and the *warnings*, which are the parts that get lost.

## The ranking, in his words

> "le win sul mio motore cpu server qwen-tts linux son stati dataflow, kernel
> ottimizzati ma **soprattutto design v2 del server e split carichi in cores
> pinned mirati + batching**... c'è **moltissima engineer** lì per arrivare a
> c20+."

Read the ordering, because it is the opposite of the instinct:

1. **Server design + pinned core split + batching** — the largest win
2. **Dataflow** — second
3. **Optimized kernels** — third

We have been working bottom-up. The kernel work matters, but on its own it does
not turn cores into streams: **without the topology, more cores are not more
streams.** A 36× on one region bought us RTF 0.245 and the serving profile still
said the machine tops out near two or three real-time streams, because the
limit was never the kernel.

> "il dataflow e i kernels saranno differenti, ma **le idee motore possono esser
> prese come spunto o idealmente a pezzi copiate**."

So: copy the engine ideas, not the kernels. Our model is a 109M continuous-latent
AR with a SEANet decoder; theirs is a 0.6B dual-track LM over discrete codes.
The graphs share nothing. The *serving machinery* shares almost everything.

## The calibration point, and what it says about C100

**32 ARM cores, Qwen 0.6B → C20/C22.** PocketTTS is 109M, about 5.5× smaller.
His own estimate for us is **double or triple, so C40-C66 — not C100** — because
dataflow counts as much as parameter count.

Use that as the anchor for any projection. A number derived from "we are 5.5×
smaller so we get 5.5× the streams" is arithmetic, not an estimate: the AR step
scales with weight traffic, the SEANet conv stack scales with compute, and the
per-request prefill does not batch at all. **If our streams-per-core comes out
worse than his at equal normalized model size, that is a dataflow bug to
investigate, not a result to accept.**

## The two removals he called out by name

> "una cosa importante fu anche fare profiling e **rimuovere tutti i conv
> datatypes inutili**, e **levare scalar blas** ecc, misurando man mano i colli
> di bottiglia."

**Useless dtype conversions.** Weights land as bf16, may become f32, may become
f16 or int8, activations get quantized and dequantized. Every hop that is not
required is bandwidth and latency on a path that is already memory-bound. First
check on our side: `src/qmat.c` holds **15 conversion sites** while every other
hot-path module holds zero, so that is where to look, and the specific suspicion
is `bf16 → f32 → f16` with an intermediate buffer where one step would do.

**Scalar BLAS.** A scalar path chosen silently is worse than a slow one that
announces itself. Ours is not hypothetical: `--dispatch-map` reports
`blas.accelerate ON` here and `blas.scalar_fallback OFF` — but **Accelerate does
not exist on the target**. The GEMM fast path that bought 36× in the decoder
goes through BLAS. On Linux that is OpenBLAS if configured, and the in-tree
scalar loop if not. A fallback that never fires on macOS can be the normal path
in production.

**Measuring bottlenecks as you go**, not once at the end. Each removal is
justified by a number, and the next target is whatever the measurement names
next — which is why the cost map has to have hooks in the engines rather than
just existing.

## int8 and int4 are the bet

> "int8 è win come qualità e cpu simd, quindi io punterei tutto su quello e int4
> anche"

Two facts make this stronger than it sounds:

- **Our f16 path is ARM-only.** `MYNAH_QMAT_F16` is gated on
  `defined(__aarch64__) && defined(__ARM_NEON)` (`src/qmat.c:26-28`), so on Linux
  x86 it silently becomes f32. The f16 default currently set for PocketTTS gives
  ARM 2× and **x86 nothing**.
- **int8 is what the hardware accelerates**: SDOT and SMMLA on ARM, VNNI and AMX
  on x86. Those kernels now exist here and are proven bit-identical. f16 has no
  equivalent integer-dot acceleration.

The obstacle is that our int8 currently breaks parity on PocketTTS (hidden
`rel_l2` 5.9e-02…1.4e-01 against a 1e-4 tolerance, an extra generated frame,
log-mel 0.921 against f32), while it is fine on Magpie. That is a **bug in what
we quantize, not a verdict on int8**: the reference implementation quantizes
attention and FFN only and leaves the flow-matching network and the Mimi decoder
in f32, and we quantize every linear the hook sees. Tracked as E3-5c.

**So the default question is not "what is fastest on this laptop" but "what is
fastest on the target".** On the target, f16 is a no-op.

## What this changes about how we measure

Every number in this repo was taken on an M1. That was fine while we were
chasing correctness; it is not fine now. Three of our current numbers are
specifically suspect on Linux: the 36× conv-stack win (Accelerate vs OpenBLAS),
the thread-pool default (performance cores via sysctl — a concept that does not
exist on a 64-core NUMA box), and every ISA claim (on Linux x86 `SIMD=auto`
compiles plain AVX2, so VNNI and AMX would never be selected however well they
are written).

Until there is a Linux box or a CI that runs, **no production number can be
quoted from here**. That is E4-9.
