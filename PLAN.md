# Mynah TTS — implementation plan

> **Mynah TTS** is a small, native C runtime for codec-based and continuous-AR
> TTS models. It is CPU-first, portable across macOS/Linux, and uses Metal or
> CUDA only when explicitly requested. The first engine is
> `nvidia/magpie_tts_multilingual_357m`; **PocketTTS (Kyutai) is the second**.

Status: Magpie v1 is implemented and shipping — model pack, converter, CPU
SIMD/scalar kernels, NanoCodec synthesis, WAV writer, HTTP server with continuous
batching, and optional Metal/CUDA matmul backends. Oracle parity is closed.
Current work is the **second engine**: PocketTTS, a continuous-latent AR model,
which requires the engine seam that has been outstanding since July.

**Current focus is CPU — ARM and x86 together. GPU work is deferred.**

**Priority order, from the author of the reference implementation** (who reached
**C20/C22 real-time streams on 32 ARM cores with a 0.6B model**) — see
[`.work/serving-doctrine.md`](.work/serving-doctrine.md):

1. **server design + pinned core split + batching** — the largest win
2. **dataflow** — second
3. **optimized kernels** — third

That is the opposite of the order we have been working in. A 36x on one kernel
region took us to RTF 0.245 and the serving profile still capped the machine at
two or three real-time streams, because the limit was never the kernel.

**Calibration, corrected by reading their repo** → [`.work/linux-production.md`](.work/linux-production.md):
their qualified 32-core point is **x86 Zen5 VNNI at C12** (C11 preferred), not
C20/C22; on 32-core ARM the promoted point is **Graviton5 C4**, and one 32-core
Neoverse-V2 host promotes **nothing**. Their measured ceiling is also 2-3x below
their own physics ceiling on every host, always for the same reason: the
per-slot decoder is **glue-bound, not compute-bound** — ~5% of VNNI peak, with
50-120 pool dispatches per decoder call per item. Our own thread work measured
~20 us of wake-up per region independently. **Expect the same wall.**

**Standing decision: OpenBLAS leaves the process for good.** Not for speed — for
ownership. A second thread pool inside our address space is a permanent source of
non-reproducible measurement, and every profile has to carry environment variables
that must be *absent* to be valid. Whatever it computes, we write ourselves for
x86 and ARM. The surface is already down to **two `cblas_sgemm` call sites**, both
switchable today, and `BLAS=none` links neither OpenBLAS nor Accelerate — the
inventory and the order of work are in [`.work/no-blas.md`](.work/no-blas.md) §3b.
Keep `BLAS=openblas` as a comparison build forever, so the A/B never stops being
available and never becomes the default again.

**Production is Linux server, x86-64 and ARM64.** macOS/M1 is the development
machine and every number in this repo so far was taken there, which makes them
development signals rather than product claims. Three concrete reasons they do
not transfer: the SEANet GEMM fast path that bought 36x goes through BLAS, which
is Accelerate here and OpenBLAS there; the thread pool defaults to performance
cores via sysctl, a concept that does not exist on a 64-core NUMA server; and on
Linux x86 `SIMD=auto` compiles `-mavx2 -mfma` with **no runtime dispatch**, so
VNNI and AMX would never be selected in production however well they are
implemented. Closing that gap is E4-9.

**A premise we recorded and then falsified**: "int8 breaks PocketTTS parity" was
a category error. f32 against f32 with a different seed gives hidden rel_l2
1.05-1.07 and log-mel 0.378 — the 5.9e-2 we had flagged is **twenty times
smaller** than the sampler's own variability. What matters is whether a tensor's
error re-enters the AR loop: outside it the codec takes int8 at log-mel 0.9995
with zero frame difference, inside it f16 measures 1.2e-06 and moves nothing.
The default is now mixed per tensor → [`.work/dtype-and-fallbacks.md`](.work/dtype-and-fallbacks.md).

One measured fact shapes several epics: the six PocketTTS language models are
**independently trained and share nothing**, codec included. One pack per
language, no deduplication, voices valid only for the model that produced them,
and continuous batching cannot mix languages.

The graph, however, is **fully shared**: identical across all six languages, and
identical across both checkpoint generations for generation with the shipped
voices. Adding a language costs zero C code; the three tensors that differ
between revisions live only in the clone-from-wav path (E3-10).

## How this plan is organised

`PLAN.md` is a **board**: one line per work item, with a link to the note under
[`.work/`](.work/) that carries the detail. Detail does not belong here — when a
task is picked up, read its note. Session logs, measurements and post-mortems go
in `.work/`; durable performance results go in `docs/performance.md`.
See [`.work/README.md`](.work/README.md) for the convention.

Sections 1-16 below are the durable contract: mission, model contract, repository
shape, kernels, oracle policy, gates. They change only when a decision changes.

## 0. Work board

Legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped

### Reference — read before starting anything

- [x] PocketTTS facts verified from the weights, incl. cross-language divergence
      and the reference-implementation traps → [`.work/pocket-tts-model-facts.md`](.work/pocket-tts-model-facts.md)
- [x] Architecture diff, mynah-tts as-is vs PocketTTS → [`.work/pocket-tts-vs-mynah.md`](.work/pocket-tts-vs-mynah.md)
- [x] Magpie-era checkpoints and rejected ideas → [`.work/archive-2026-07-checkpoints.md`](.work/archive-2026-07-checkpoints.md)
- [x] The reference serving design, and the fourteen approaches already falsified
      there → [`.work/serving-design.md`](.work/serving-design.md)
- [x] Why `STREAM_RTF < 1` does not mean the player never stops, and the quantum
      floor → [`.work/streaming-cadence.md`](.work/streaming-cadence.md)
- [x] The method: cost model before code, every tool declares a refusal, the
      completion rule → [`.work/engineering-method.md`](.work/engineering-method.md)
- [x] The dtype hunt is **closed on ARM**; what it found instead were two
      production-blocking build defects → [`.work/dtype-and-fallbacks.md`](.work/dtype-and-fallbacks.md)
- [x] BLAS removal: one function, three call sites, two of them ours →
      [`.work/no-blas.md`](.work/no-blas.md)

### E1 — Engine seam and the `graph.c` split → [`.work/engine-seam-refactor.md`](.work/engine-seam-refactor.md)

Blocks E3. Do it *with* PocketTTS in hand, not before.

**Progress**: `graph.c` (4283 LOC) is gone. Steps 1-4 done, each verified against
the goldens with byte-identical audio:
`mynah_util` + kernels → `conv1d` → `codec_nanocodec` → `engine_magpie.c` +
`inference.c`. `src/` is nine files, largest 2186 lines.

- [~] E1-1 `src/tts_engine.h` **written and compiling**; the code movement out of
      `inference.c` is the remaining half and is the one step where audio can change —
      the exact table of what moves where is in the note
- [x] E1-2 slot driver lifted into `src/inference.c` (650 LOC)
- [ ] E1-3 lift streaming state into `src/stream.c`; `decode_audio` takes contiguous
      monotonic ranges and the engine owns continuity (E2-3), so left-context stays internal
- [x] E1-4 done differently and deliberately: there is no shared `transformer.c`,
      because those functions format NeMo tensor names. They stayed in
      `engine_magpie.c`; the genuinely shared parts became `conv1d` and `kernels`.
      A shared transformer needs functions taking **resolved weight pointers** —
      that is E3 work, not a move
- [ ] E1-5 remove Magpie fields from `src/mynah_tts.h` (breaking API change)
- [x] E1-6 **done** `a3823ae` — one real parser in `src/json.{c,h}`, used by both readers; surrogate pairs, nesting, arrays, bounded depth, byte-offset errors · real JSON parser with nesting and arrays, replacing `mynah_tts.c:112-163`
- [ ] E1-7 split the converter into a shared pack writer + per-engine metadata
- [ ] E1-8 gate: Magpie stream↔offline output sample-identical to the pre-refactor binary
- [x] E1-0 **prerequisite done**: `tools/make_fake_pack.py` + `tests/refactor_goldens.sh`
      + `make fake-pack | goldens-capture | goldens`. 8 goldens captured;
      `make stream-test` and the full 15-check `make server-test` both pass against it

### E2 — PocketTTS oracle → [`.work/pocket-tts-oracle.md`](.work/pocket-tts-oracle.md)

Independent of E1. **Start here** — it tells E1 which state the seam must model.

- [ ] E2-1 `tools/oracle_pocket.py`: dump all 12 stages, tokenizer → waveform
- [x] E2-1a `tools/oracle_pocket.py` + `make oracle-pocket` written
- [x] E2-1b `tools/oracle_pocket_tokenizer.py` + `tools/corpus_pocket.py`: ~9000 cases
      per language, hex-encoded so invalid UTF-8 survives, deterministic by seed
- [ ] E2-2 `tests/parity_pocket.py` with per-stage tolerances
- [x] E2-3 streaming decode measured: **carry codec state** (error 1e-7 down to a
      1-frame chunk); replaying context needs 64 frames / 5.12 s for exactness, so
      the Magpie replay strategy does not transfer
- [x] E2-4 **done** `891501b` — and the framing was wrong by 10x: the decoder transformer runs at **200 Hz**, so `context: 250` is **1.25 s of audio**. Every utterance longer than that already takes the windowed path, and the oracle only ever saw the first 20 ms → [`.work/transformer-ar-sliding-window.md`](.work/transformer-ar-sliding-window.md) · dump decoder-transformer behaviour past `context: 250` — needed to exercise
      the sliding window in `transformer_ar`, which no reference data reaches today
- [x] E2-5 **done** `a477b19` — measured, not inherited: 173 tokens ran the whole 900-frame budget and never emitted EOS · dump the text-chunk seam at `MAX_TOKEN_PER_CHUNK = 50`, including the known skip bug

### E3 — `engine_pocket.c` → [`.work/pocket-tts-engine.md`](.work/pocket-tts-engine.md)

**Every part exists and is verified against the oracle.** Tokenizer 45k cases,
backbone 3.4e-06, flow head 1.07e-06, SEANet 5.7e-07, converter 214/214. What is
left is composing them (E3-5) and finishing the seam (E1-1).

Needs E1 and E2.

- [x] E3-1 `tools/convert_pocket.py` **done** — 214/214 tensors, 302 MB pack, 59 scalars
      top-level for the flat parser, schema verified identical en/it
- [ ] E3-1a **verify F16 voice KV against the oracle** (measured 3.9e-3 abs) before
      keeping it as the default; F32 costs +83 MB per pack
- [x] E3-2 `src/tokenizer_sentencepiece.{c,h}` **done** → [`.work/tokenizer-sentencepiece.md`](.work/tokenizer-sentencepiece.md)
      1734 LOC, in `CORE_SOURCES` and `--self-test`; `make tokenizer-parity` replays
      45,197 oracle cases / 2.1M ids across 5 languages; UBSan, ASan and leaks clean
- [x] E3-3 `src/flow_head.{c,h}` **done** — oracle parity 1.07e-06 against a 1e-4 tolerance
- [x] E3-4 `src/seanet.{c,h}` **done** — SEANet parity 5.7e-07; chunked-vs-one-shot 2.98e-07,
      so E2-3's carried-state decision holds in C
- [x] E3-5 `src/engine_pocket.c` **done** — CLI and server both generate audio; full
      utterance parity mel corr 1.000000 with the oracle's noise injected
- [x] E3-5a **performance: RTF 2.03 → 0.245** (f16 default, lossless on a bf16 checkpoint);
      conv stack 36× via a GEMM fast path. int8 reaches 0.191 but breaks parity, so it stays opt-in
- [ ] E3-5c **int8 is the bet, and today it breaks parity on this engine** — hidden
      `rel_l2` 5.9e-02…1.4e-01 against 1e-4, an extra frame, log-mel 0.921. The
      reference quantizes attention and FFN only and leaves the flow head and the Mimi
      decoder in f32, while we quantize every linear the hook sees. Fix what we
      quantize, then int4. **f16 is a dead end on the target: it is ARM-only**
- [ ] E3-5b next 2×: `codec.transformer` is 43% of the wall, running 16 positions as 16 steps
      over the same ~29 MB. A batched GEMM prefill estimates f32 ~0.35 / f16 ~0.17
- [x] E3-6 **done** · model-free self tests for the hot kernels, `make kernels-test`
      (13.4k checks) plus `make kernels-negative-control` (9 deliberate defects, 9
      caught). The two LayerNorms **are** the same function — agree to 2.9e-07 of the
      row scale; variance-RMSNorm and mean-square RMSNorm are 167% apart on offset
      data and each is pinned to its own f64 reference; every kernel run at every
      length 0..40; causal unfold against the contract → [`.work/accelerate-only-kernels.md`](.work/accelerate-only-kernels.md)
- [ ] E3-7 offline parity across all 12 oracle stages, both checkpoint generations
- [ ] E3-8 streaming sample-identical to offline, then batched through the shared driver
- [ ] E3-9 WAV smoke for all 6 languages with explicit language/voice/seed
- [ ] E3-10 **voice cloning from a wav — required, not optional** (see E7)
- [ ] E3-11 one pack = one language; compute `time_embed.*.freqs` at load instead of storing
- [ ] E3-12 NaN-as-BOS sentinel: reproduce it or track validity explicitly — never let NaN reach a matmul
- [x] E3-13 **done** `a477b19` · pack refuses a voice KV from a different model/revision (upstream: it then never emits EOS)

### E7 — Voice cloning → [`.work/voice-cloning.md`](.work/voice-cloning.md)

Zero-shot cloning is a product requirement. The weights are already in the pack
(`mimi.encoder*` + `downsample` + `speaker_proj`, 9.78M params / 19.6 MB BF16).

- [x] E7-1 SEANet **encoder** in C (parity 6.4e-06 vs the PyTorch oracle) — mirrors the decoder, reuses its conv kernels
- [x] E7-2 encoder transformer (2L d512) — reuses the shared attention
- [x] E7-3 `ConvDownsample1d`: stride 16, kernel 32, `pad_mode="replicate"`
- [x] E7-4 `speaker_proj` + optional `bos_before_voice`, config-driven — the only
      place the two checkpoint generations differ
- [x] E7-5 prefill the reference latents through the backbone to produce the voice KV
- [x] E7-6 WAV/audio input decode; require 24 kHz mono first
- [x] E7-7 polyphase resampler to 24 kHz matching `scipy.signal.resample_poly`
      within tolerance — needed for arbitrary input files
- [x] E7-8 truncate reference audio to 30 s, as upstream does
- [x] E7-9 `mynah-tts export-voice`: serialize the KV so reload is instant
- [ ] E7-10 consent gate and notice before cloning — see E6

### E4 — CPU kernels, ARM and x86 in one step → [`.work/cpu-kernels-arm-x86.md`](.work/cpu-kernels-arm-x86.md)

**No kernel is done until both ISAs exist and self-test. GPU deferred.**

- [ ] E4-1 **correct the AVX-512 VNNI claim** in `README.md` — the code is AVX2 only
- [x] E4-2 `--dispatch-map` + costmap **done**; 8 rows resolve UNKNOWN and each names the
      predicate to add — that is E4's real to-do list
- [x] E4-2a **done** `6ad9ada` — 0 UNKNOWN rows in every configuration · add the 8 named predicates so no row resolves UNKNOWN
- [ ] E4-2b place the costmap hooks (deferred: the engines were being refactored)
- [ ] E4-3 baseline per-region profile of the PocketTTS path, on M1 and on EPYC
- [ ] E4-4 thread pool upgrade: lane split, deadline priority, `after_fork` (prereq for E5-6)
- [ ] E4-5 the kernel the profile names — scalar reference, then NEON/SDOT/i8mm **and** AVX2/AVX-512/VNNI in one change
- [ ] E4-6 int8 weight prepack with persistent cache, both ISAs
- [ ] E4-7 AMX-INT8 (Linux/x86 only), last
- [ ] E4-10 **remove useless dtype conversions** — called out by name as one of the two
      profiling wins. `src/qmat.c` holds 15 conversion sites and every other hot-path
      module holds zero; the suspicion is `bf16 -> f32 -> f16` where one step would do
- [x] E4-11 **done** `6ad9ada` · **no silently-chosen scalar BLAS** — a scalar path taken without anyone
      knowing is worse than a slow one that announces itself. `blas.accelerate` is ON
      here and absent on the target, and the 36x conv-stack win goes through BLAS
- [x] E4-12 **done** · fatal ISA guard — `mynah_dispatch_isa_guard()`, first statement of
      `main()`, fires only on a DEFINITE absence so an unprobeable CPU still runs → [`.work/linux-build-and-dispatch.md`](.work/linux-build-and-dispatch.md)
- [x] E4-13 **done** · 16-entry `link-only` CI matrix, x86 + ARM. **It found two pre-existing
      defects on its first run, both invisible on macOS**: `SIMD=scalar` did not compile
      (`src/qmat.c:1798`, fixed in `57596b3`, exemptions removed), and every
      non-`-march=native` ARM profile computes wrong f16 weights on Linux/gcc — a
      **strict-aliasing miscompile** in `src/qmat.c`, not a boundary-precision bug:
      recompiling that one file with `-fno-strict-aliasing` fixes it, `-march=armv8.6-a`
      does not. Open, and it is why the job omits `--self-test` → [`.work/linux-build-and-dispatch.md`](.work/linux-build-and-dispatch.md)
- [x] E4-14 **done** · `build/cpu/.build-flags`; every object depends on the effective
      CC/SIMD/BLAS/CFLAGS/CPPFLAGS/LDLIBS text, rewritten only when it changes → [`.work/linux-build-and-dispatch.md`](.work/linux-build-and-dispatch.md)
- [x] E4-15 **done** · `tools/simd-auto.sh` + `make simd-auto`; double test, printed profile,
      fixtures for 6 real parts and 23 checks in `make test` so the table is falsifiable with
      no x86 host. **Correction to the item**: VNNI was already reachable — `src/qmat.c`
      target-attributes it and picks by CPUID — what was broken is that `auto` passed
      `-mavx2 -mfma` on Linux x86 *without asking the CPU*. AVX-512/VNNI/BF16/AMX are
      detected and reported, never turned into flags no kernel is gated on. `SIMD=portable`
      and `EXTRA_CFLAGS` added; the x86 release job moved off `auto` → [`.work/linux-build-and-dispatch.md`](.work/linux-build-and-dispatch.md)
- [~] E4-16 **kernel landed (`fa3df67`), default not flipped** — `mynah_sgemm_f32` and `BLAS=none` → [`.work/no-blas.md`](.work/no-blas.md)
      Decided: we do not want a second thread pool inside our process. The surface is
      one function (`cblas_sgemm`) at three call sites, and **the whole PocketTTS
      production path is two of them, both in `seanet.c`** — the backbone and flow head
      already go through `qmat`. Skinny shapes, `n` large, `k` small. Removal also
      deletes the weak-symbol clamp in `threads.c`, three dispatch rows, and the
      three-way Makefile split. **Flipping the default is blocked on two separate
      things**: a Linux RTF measurement, and qualifying the loss of vForce's `vvtanhf`
      in the GELU, which is not a GEMM decision and must not ride along on one
- [ ] E4-16b **flip the default to `BLAS=none`** — the only thing blocking it is an
      RTF comparison against `BLAS=openblas`, and the Axion box now exists to take it.
      Two decisions, not one: the GEMM flip, and separately qualifying the loss of
      Accelerate's `vvtanhf` in the GELU, which on a continuous-AR model compounds
- [ ] E4-16c **delete the OpenBLAS compensation machinery** once E4-16b lands — the
      weak-symbol clamp and `mynah_blas_owned()` in `threads.c`, three dispatch rows,
      the per-GEMM `mynah_blas_set_threads`, the three-way Makefile split, and
      `OPENBLAS_THREAD_TIMEOUT` along with the "`OPENBLAS_NUM_THREADS` must be absent"
      rule in every profile recipe
- [~] E4-16d **the `vvtanhf` GELU and the vDSP Snake are ours now**, scalar + NEON +
      AVX2, with the numerical qualification E4-16b asked for. Measured: our tanh
      **1.35 ulp vs Accelerate's 2.60**, our sin **6.8e-08 vs vvsinf's 1.31e-07** — we
      were *fixing* macOS, not degrading it. The two platforms were **already
      producing different audio** at the pinned head (corr 0.9535); they now run one
      function. Left: the BNNS conv1d filter cache (8 sites, `src/conv1d.c`, another
      lane) → [`.work/accelerate-only-kernels.md`](.work/accelerate-only-kernels.md)
- [ ] E4-16e **`mynah_exp_f32`, and the inventory question that missed it.** With the
      GELU unified, macOS and Linux still part at sample 1660 of the same utterance.
      Eliminated by measurement: not BLAS (`BLAS=none` identical), not FMA contraction
      (`-ffp-contract=off` identical), not the pool (`MYNAH_THREADS=1` identical).
      **It is libm**: Apple's and glibc's `expf` and `logf` produce different bitstreams,
      and `expf` is on the hot path twice (`flow_silu`, `softmax`). Tier 3 missed it
      because it was built by grepping Accelerate symbols, and the question that
      predicts a divergence is not "what is macOS-only" but "what is not the same
      function on both machines" → [`.work/accelerate-only-kernels.md`](.work/accelerate-only-kernels.md) §7
- [x] E4-16a **done** · `OPENBLAS_THREAD_TIMEOUT=1` set from a constructor when unset, with
      `blas.thread_timeout` reporting **claim vs fact** — a shared libopenblas
      initialises before this executable's constructors, so only the environment
      before exec is guaranteed to be read → [`.work/decoder-lane.md`](.work/decoder-lane.md) §4.
      TTFA 108 ms **bimodal** to 66 ms stable, 42.5k to 12k context switches/s (theirs).
      Deliberately **not** partitioned: lowering BLAS threads improves RTF and costs
      30% of TTFA. Compensation for a dependency E4-16 removes, not a design.
      Measured here while verifying: OpenBLAS builds a team sized to the HOST per
      worker — **32 threads inside a worker pinned to 8 cpus** (§5)
- [ ] E4-17 **plan from `sched_getaffinity`, not `sysconf`** — `sysconf` sees neither an
      inherited taskset nor a cpuset cgroup, so every containerised deployment plans the
      whole host. Also read cgroup v2 `cpu.max` and warn (they never closed that one)
- [x] E4-18 **closed: not applicable, by measurement.** Measured on the Axion box with an
      `LD_PRELOAD` counter: **2,786 allocations, 3 mmap, 2 munmap** per request (theirs:
      11,899 / 78 / 154), and the count is **constant** over `--max-steps` 20→160 — the AR
      loop allocates nothing. `BLAS=none` is lower still (2 mmap). `codec_nanocodec.c`
      already hands one `columns_workspace` to `conv1d`, which is the data-structure fix
      they arrived at. An arena would be an unmeasured change against a problem we do not
      have → [`.work/linux-build-and-dispatch.md`](.work/linux-build-and-dispatch.md)
- [x] E4-19 **two production-blocking defects found and fixed** (`3892ba6`) →
      [`.work/dtype-and-fallbacks.md`](.work/dtype-and-fallbacks.md). No Linux build
      linked at HEAD (`mynah_conv1d_sgemm_enabled` trapped inside the Accelerate block
      by our own E1 split); `MYNAH_QUANT=f16` was a silent no-op on x86, so the 2x we
      measured on ARM did not exist on the target. x86 now has F16C/AVX2 + scalar half
      kernels; ARM output byte-identical
- [x] E4-20 **done** `6ad9ada` · **`--self-test` fails on x86**: `qmat u8 level=1 not bit-identical at row 3
      (k=200)`. Pre-existing, from the VNNI commit — that work was never self-tested on
      x86, which is the whole point of a model-free self-test
- [x] E4-21 **done** `6ad9ada` · **`seanet.c` has 12 fallbacks and zero dispatch rows.** With `BLAS=scalar`
      the entire codec conv stack drops to the hand-scalar loops — the 8570 ms path,
      **36x slower** — and nothing says so. Also: the depthwise upsample always takes
      `convtr_scatter_scalar` silently; `conv1d.c:487,530` ignore the sgemm return
      (NULL backend gives bias-only output, no error); `seanet.c:53-55` narrows
      `size_t`→`int` six times per call with no guard, where `conv1d.c` has one
- [x] E4-20b/E4-21b **int8 is deterministic again and SMMLA is back on; VNNI has now
      executed** → [`.work/int8-int4-determinism.md`](.work/int8-int4-determinism.md).
      The batched linear's answer depended on a row's position because `dot_q8` let
      each caller add the bias (two roundings) while the quad epilogues fused it
      (one) — the *rounding count*, not the grouping the brief named. One pinned
      epilogue for every int8 kernel; goldens and PocketTTS byte-identical. CI's
      Xeon 8573C resolved `avx512vnni` and `avxvnni` and passed — first execution of
      `VPDPBUSD` on silicon. int4 got its first x86 vector path (AVX2, not VNNI: a
      32-element group scale forces a float flush that is 6 of ~18 instructions, so
      `VPDPBUSD` would move it ~10%)
- [ ] E4-22 **438 MB of dead f32.** RSS f32 626 / f16 805 / int8 724 MB: the f32
      conversion cache stays resident in full while every hot projection is served from
      f16. ~82M of 109.5M params could go bf16→f16 direct, ~328 MB per pack. Memory, not
      RTF — but for a prefork server with N resident languages it is the dominant term.
      Measure PSS, not RSS: weights are quantised before the fork and shared CoW
- [ ] E4-23 **the conversion hunt is closed on ARM, do not re-run it** — every hot-path
      conversion measures under 3% against a 20% bar, f16 runs at 1.96x against a
      2.00x byte ratio (so the in-loop convert is free), and every memcpy in the process
      together is 0.4% of samples. The wall is weight bytes. `matvec_f16` alone is 81%
- [ ] E4-9 **Linux is the target, so measure there**: runtime ISA dispatch on x86 (a
      binary that picks VNNI/AMX when the CPU has them and does not SIGILL when it
      does not), OpenBLAS thread-count coordination with our pool, a CI matrix that
      actually runs, and a Linux measurement box. Until this lands, no production
      performance number can be quoted.
- [x] E4-8 **done** `ffc28b4` — `SIMD=` profiles and the `.build-flags` stamp · `Makefile`: `SIMD=` profiles + `ARCH_STAMP` rebuild-on-flag-change

### E8 — The batched vocoder: the structural ceiling

Two lanes reached this independently. `codec.transformer` + `codec.conv_stack` are
~55% of wall and live inside `decode_audio`, which the vtable declares **per
context** — so the driver never sees two streams' codec work together and that 55%
is multiplied by the stream count. The reference implementation had the same shape
(their decoder was 72-80% of the marginal cost of a stream) and batching it was the
one change that moved their capacity.

Arithmetic, not a promise: with int8 on the codec and VNNI but **without** this,
100 streams need ~20 cores of codec alone before anything else is counted — so C100
wants a 64-core box. With it, a 32-core box returns to the conversation.

- [x] E8-1 **done** (`eb8dccc`): `decode_audio_batch` appended to the vtable — appended,
      not inserted, because both engines use positional initializers — with a default
      loop, both engines untouched, and bit-identity per context as the contract
- [x] E8-2 **done** (`eb8dccc`): `stream_gang()` decides the whole batch at once. No-wait
      is structural, not a check — every ready slot is in the gang before the pull-in
      looks at anyone. Plus the per-slot quantum ramp 1,2,2,4,4 then steady state,
      never above what the engine declares
- [~] E8-3 `step_live()` fails **all** live slots when one slot errors. Harmless at
      width 1, sixteen requests wide once batching is on
- [x] E8-3 done in the batching merge (`0d944e9`): a request that exhausts its step
      budget retires as EOS instead of failing every live slot
- [x] E8-4 **done** `a477b19` — `pocket_decode_audio_batch` fired 48 times at widths 3-6 against the real streaming server · `engine_pocket` implements the `decode_audio_batch` override — and needs its
      own parity check against `decode_audio`, since bit-identity per context has so far
      been tested only against the synthetic engine
- [x] E8-6 **done** `a477b19` — pre-flight, ordered mutation, rollback; the gate was blind until a second injection reached it · **`pocket_step_batch` is not atomic**: it advances contexts `0..i-1` before
      refusing `i`. Harmless at its declared `max_batch` of 1, illegal once that widens —
      the driver's failure isolation depends on the atomicity the header now declares
- [ ] E8-5 **`mynah_qmat_linear_batched_qt`** — the batched twin of
      `mynah_qmat_linear_resolved_qt`. `mynah_qmat_linear_batched` takes no qtype: it
      gates on the cache's own and creates a first-touch entry there, so a group
      carrying an explicit encoding is kept off the weight-stationary path to avoid a
      first-touch race deciding its precision. Consequence today: under
      `MYNAH_QUANT=int8` the backbone and flow head (`:f16` in the default spec) fall
      off the batched path. One function removes the restriction

### E9 — What the profiler found, and where to act next

First cost map on production hardware (`8b27fa1`, GCP Axion, one stream, varying
core count, `nest_mismatch=0`). Percentages are of one request's wall; the scaling
column is 1 core to 16.

**Read this table knowing two of its rows were mislabelled.** E9-1 found that
`MYNAH_RGN_MODEL_LOAD` and `MYNAH_RGN_DECODE_GANG` shared id 44, so every report
ever printed summed them under one name: `driver.decode_gang` has never appeared
and `runtime.model_load` was never only a model load. The five level-1 rows below
do not share ids and stand; anything read off `runtime.model_load` before `50092de`
does not.

| region | % of wall | 1→16 cores | verdict |
|---|---|---|---|
| `prep.decoder_prefill` | 28.7% | 1.5x, flat past 8 | **premise wrong (E9-1): a one-time weight pack, not serial work** |
| `codec.conv_stack` | 25.6% | 2.6x | **fixed (E9-2): 5.71x, and the cause was glue, not the kernels** |
| `codec.transformer` | 22.9% | 4.3x | fine |
| `step.backbone` | 16.8% | 4.6x | fine |
| `flow.head` | 5.6% | — | small |

- [x] E9-1 **`prep.decoder_prefill` was a cold start, not a serial phase** →
      [`.work/prefill-ttfa.md`](.work/prefill-ttfa.md). Splitting first call from
      steady state (`--runs 1` vs `--runs 8`) shows the steady-state prefill divides
      8.4x across sixteen cores and the flat part is a constant ~215 ms — the
      quantized weight cache being built on first touch, one thread, under
      `cache->mutex`. It is now built at engine init (`runtime.weight_prepack`), and
      the prefill's own serial half — attention, RoPE, KV writes, norms — measures
      **11.3 ms and does not move**. The causal claim was wrong: the server warms up
      before accepting, so no client request ever paid it, and it cannot be why
      safe-to-play is 908 ms at C48. What is left is ~122 ms of per-request prefill
      at `16x2`, linear in text tokens, already parallel and starved of threads
- [x] E9-2 **the conv stack's limit was glue, not parallelism** →
      [`.work/seanet-conv-scaling.md`](.work/seanet-conv-scaling.md). Split by *who
      runs the loop*: at sixteen threads **66% of the region never reached the pool**
      — `elu` 77.0 ms at 1.00x (scalar `expf`), `conv.gather` 57.8 ms re-gathering
      weight taps **that never change**, 59 times a request, and `conv.bias` 4.5x
      *slower* at sixteen threads than at one. Dispatch count was real but second:
      25 GEMMs/frame at 51 µs mean, and a fit over eleven shapes puts the pool's
      fixed cost at **26-44 µs on this box**, not the header's 20 — about 52 ms of
      235. Fixed with fewer, larger regions and no new arithmetic (`conv_taps` fuses
      K dispatches into one; ELU on the pool; `want == 1` stops dispatching a
      matvec). **230.2 → 104.1 ms at 16 cores, scaling 2.64x → 5.71x, −17% of total
      wall on default *and* int8**, byte-identical on both SIMD profiles. (First
      reported as 2.59x → 6.02x / −18%: the A/B harness wrote `MYNAH_QUANT=` empty,
      both sides got the same wrong workload, and the ratio looked consistent.
      Direction and verdict held; the figures did not.) Left open: the last 23% is calls
      of 14-42 µs, *below* the measured dispatch cost, so folding them further is a
      rounding change that needs its own qualification
- [x] E9-3 **topology is a first-class serving parameter and the reference's rule does
      not transfer** — and E9-4 now gives the mechanism, not just the ranking. Measured at C48 on 32 cores: `16x2` 0.736 · `8x4` 0.931 ·
      `4x8` at C30 already 0.922 · `2x16` 1.574 with 100% stall · `1x32` 48 of 90
      completed. Narrower workers win because a small model's regions are too short to
      amortise a wide barrier — a cost that does not shrink with the model, so a 100M
      engine pays it *more* often per second of audio, not less. **C64 completes
      192/192 at 0.954 on `16x2`.** Sweep it per host; do not inherit `4x8`
- [x] E9-4 **78 dispatches per frame, and the barrier is the tax** →
      [`.work/pool-barrier-meter.md`](.work/pool-barrier-meter.md). Flat from 4 to 32
      threads — the graph sets the region count, so a wider pool only makes each
      piece shorter. Mean region 301 µs at 1 thread → **43-45 µs at 16-32**, barrier
      0% → 22.4% → **35.6%**, and at 32 the pool admits 27.6 threads per region while
      **19.0** do work. That explains E9-3 instead of restating it: at C48 `16x2` is
      266 µs/region and 1.2% barrier (RTF 0.723), `1x32` is 60 µs and **31.3%** (1.616)
      — and the wide topologies dispatch *fewer* regions per frame (41 vs 62) and lose
      anyway, because the tax is per region, not per frame. `MYNAH_POOL_METER=1` ships
      it per call site, +0.3-0.6% when off, byte-identical. Levers: wake pre-check ON
      (−2.1/−2.3%), admission cap **measured and rejected** (+0.4 to +1.0%), `fastexit`
      **dark** (−1.9/−4.9%) until it runs on x86. All three −7.0% at 32 threads — and
      **production is narrow, where all three are noise**
- [ ] E9-5 **the census and the dispatch report disagree about quantization**: a
      default run carries f16 on every projection while `quant.requested` reads `off`,
      because that row describes the cache default and not the per-group spec. Two
      reports, one truth — fix the row, not the census
- [ ] E9-7 **fuse the regions — this is the one that changes the shape** — with the
      count behind it at last: `qmat_rows_block` (`src/qmat.c:1481`) dispatches
      **3240 regions per request** at 42 µs with a **43.1% barrier**, and `sg_task`
      (`src/sgemm.c:670`) **1825** at 41 µs with **36.1%**; both are **100% below the
      pool's break-even**. The same two sites at 2 threads cost 119 µs/1.2% and
      157 µs/3.3% — which is exactly why `16x2` wins and why fusing is the lever that
      would let a wider pool stop losing. Not yet measured: whether fusing actually
      recovers that barrier, only that it is 36-43% of those sites' wall
- [ ] E9-8 **`fastexit` has never run on x86** — the biggest pool lever (−4.9% at 32
      threads) ships OFF for that reason alone. It is the mirror of a documented trap,
      correct by the standard and clean over 120k regions on ARM with `MYNAH_POOL_SPIN=0`,
      but unrun where the trap actually bites. Validate on x86, then flip it
- [ ] E9-9 **the spin budget default is wrong off aarch64** — the knee on the Axion is
      **65536**, and **4096**, which is what this binary uses on every non-aarch64
      target, costs **+16.7% at 32 threads**. Measure the knee on x86 and set it from a
      number rather than from a constant nobody has re-derived
- [ ] E9-6 **`MYNAH_QUANT=` empty is not `MYNAH_QUANT` unset** — the empty value
      produces different audio *and* a ~1.8x different wall on HEAD, which is how a
      lane's A/B harness measured the wrong workload on both sides of a pair and
      read it as contention. Pre-existing. Either treat empty as unset or refuse it
      loudly; silently meaning a third thing is what makes it a trap
- [x] E9-10 **it is work, not waiting** → [`.work/ttfa-under-load.md`](.work/ttfa-under-load.md).
      Measured, not reasoned: under 48 live streams `connect()` takes **0.1 ms** and
      `GET /health` answers in **0.7 ms**, so neither the backlog nor the HTTP side is
      involved; a request arriving into a loaded server gets audio at **149 ms p50**.
      The screen's number is the *burst* a wave creates. Fire 48 at once and every one
      gets first audio at 580-585 ms — **five milliseconds of spread across
      forty-eight**, flat rather than a ramp or a step — linear in slots/worker
      (+192 ms each) and linear in text (**6.05 ms/word/slot ≈ 4.65 ms/token**, E9-1's
      per-token prefill cost). And the arithmetic closes: one prefill is 331 core-ms at
      two threads, ×48 ÷ 32 cores = 497 ms + 88 ms of first frames = **585 predicted vs
      581 measured**. Sixteen workers × two threads *is* the machine — no idle resource
      exists for a better scheduler to find
- [~] E9-11 **the f16 batched path batches now — measured on macOS, unvalidated on the
      box** → `src/qmat.c`. It carried a two-activations-per-SMMLA path for INT8+i8mm
      and dropped every other encoding, f16 included, into `for (b)
      qmat_rows_dispatch(...)`: one activation at a time, so a 16-row tile read each
      weight block **sixteen times**. Now two weight rows are converted once and
      multiplied into two or four activations (both ISAs), and a three-remainder — the
      production width at C48, where a worker holds three live slots — is served by the
      four-lane kernel with the last activation **repeated**, spending a quarter more
      arithmetic to halve the weight traffic, which wins and thereby proves the kernel
      memory-bound. macOS, clean build, paired: `prep.prefill_proj` **1.62-1.71x**,
      `prep.decoder_prefill` 1.32-1.55x; `step.total` at batch 3 **496.7 → 361.8 ms**.
      Byte-identical, 180/180. New `self_test_f16_lane_widths()` gates widths 1-9 by
      `memcmp` and both new paths were mutation-tested. **Re-take on the Axion before
      any of these numbers are quoted as production**, and run `--self-test` on x86 —
      the x86 kernels were written on arm64 and have never executed
- [ ] E9-12 **int8 never reaches the prefill** — `MYNAH_QUANT=int8` leaves
      `prep.prefill_proj` at 149.9 ms, identical to default, while taking
      `request.total` 1848 → 1496. That is `POCKET_QG_DEFAULT_SPEC` working as designed:
      the backbone is inside the AR loop where int8's per-step error compounds over ~50
      steps into a different trajectory. The prefill is **not** inside that loop, so the
      measurement that rules int8 out does not directly apply — but it does not clear it
      either, because the prefill writes the KV every step attends to, a *static* error
      rather than a compounding one, which is a different risk and not a smaller one.
      Needs its own quality gate (frame count, EOS step, log-mel corr) and a second
      quantized copy of the backbone, since the steps must stay f16
- [-] E9-13 **moot after E9-11: the tile is not what sets weight traffic** — the sweep
      that called it a lever was invalid (the knob moved the accessor while
      `tar_rows_reserve` sizes the scratch from the macro and the prefill clamps to
      `rows_cap`), and the question it was asking is now answered elsewhere: the batched
      kernel amortises a weight block over four activations *regardless of tile*, so
      going from 16 rows to 32 changes nothing per row. Only `QMAT_F16_BATCH_LANES`
      would, and that is register-bound rather than scratch-bound. Byte-identity across
      tiles stands as a result
- [x] E9-0 **allocations are constant across `--max-steps`** (3,441 at both 24 and 96
      steps): the autoregressive loop allocates nothing, and that is now a permanent
      check rather than a belief

### E10 — What five parallel audits found, 2026-09-14 → [`.work/reference-arm-soak-2026-09-13.md`](.work/reference-arm-soak-2026-09-13.md)

Five read-only audits against the engine and against the reference OSS engine
(`../qwen-tts` at `97c0fa1`): kernels, serving design, allocations, thread pool,
dtype conversions. Ordered by **(measured cost touched) ÷ (risk)**, and every
item carries the measurement that justifies it. Numbers marked *(mac)* are
development signals taken while the Axion was off and must be re-taken there.

**Everything measured below is a macOS development signal while the Axion is off.**
The box list, to be re-taken in one pass when it is back: E10-1 (conv gather),
E10-3 (pre-fork footprint, and E10-3b's `smaps_rollup` question), E10-4/4c (int8
batch widths), E10-4d (the default spec's 1.40×), E10-7 when it lands, plus the
simultaneous-worker shape screen (E10-10), the closed-loop mini-soak (E10-9) and
the x86 self-test that judges the two kernels written here and never executed
(E10-4b, and the f16 x86 lanes in `8614117`/`a63d349`).

- [x] E10-0 **the allocation gate skipped macOS for a reason that was not true**
      (`8919950`). `dlsym(RTLD_NEXT,"malloc")` returned the shim's own function and
      `shim_malloc` tail-called it: an infinite loop, read as "too slow to finish".
      Fixed; the check now runs here and separates our allocators (FAIL if they grow)
      from the linked BLAS (PASS, named). Verified: `malloc/calloc/realloc` identical
      across 24 and 96 steps, `posix_memalign` +209 ≈ 19/frame from Accelerate, and
      `BLAS=none` is fully constant at 3606
- [x] E10-1 **hoist the conv tap gather out of the `BLAS` guard** — done, byte-identical,
      `codec.conv_stack` **1.210×** and `request.total` **1.044×** *(mac)*, and `conv.gather`
      is gone from the phase table. Production (`BLAS=none`) is untouched by construction:
      the fused path already runs there, so the memo is never consulted. — `src/seanet.c:666-670`
      re-derives **1,964,224 floats per frame, 1.43 GB per request**, of a weight layout
      that never changes. The gather-free path is behind `#if defined(MYNAH_SEANET_OWN_SGEMM)`,
      i.e. `BLAS=none` only, so **macOS and `BLAS=openblas` builds still pay it**:
      `conv.gather` 109 ms, 31.7% of `codec.conv_stack`. The guard's stated reason — our
      GEMM kernels would change macOS numerics — is true of `conv_taps` and **false of the
      gather**, which is `dst[j] = src[j*kernel]`, a copy. Measured *(mac)*: `request.total`
      −78.8 ms, **byte-identical**. It also unblocks an honest `none`-vs-`openblas` A/B,
      which E4-16 needs and which today partly measures the gather
- [-] E10-2 **superseded by E10-3, and its naive form was unsafe anyway** — `use_q` in
      `mynah_qmat_linear_resolved_qt` requires `count <= QMAT_SMALL_COUNT` (16); above that
      `src/qmat.c:2623` hands `weight_data` to `mynah_backend_matmul`, so freeing the F32
      would leave a **dangling** pointer the engine still holds, not an unreachable one. It
      does not fire today (prefill tile is 16) but that is a runtime property, not a
      guarantee. And with E10-3 the F32 is now one shared copy, so the prize fell from
      ~5.6 GB to 349 MiB once. Revisit only with E10-3b. Original text follows.
- [ ] E10-2b **the F32 copy is still dead weight after packing, now costing once** — BF16→F32 materialisation is
      **399,011,464 B** (`src/weights.c:155`) and the f16 pack **183,142,400 B**
      (`src/qmat.c:2127`). After packing, the F32 is dead: the census reports 1,108
      `matvec-q` + 216 `batched-q`, **100% f16, 0 f32, 0 rowloop**, unchanged at a 527-char
      input. **349 MiB per process.** Must be conditional: `mynah_qmat_linear_resolved_qt`
      falls back to `mynah_backend_matmul(weight_data)` when `count > 16` or `k > 8192`,
      and `MYNAH_QUANT=f32` uses the F32 copy as the working weights — so it needs a
      re-materialise-or-refuse path, not a bare free
- [~] E10-3 **conversions built in the parent — done; the pack is not, and cannot be yet**
      *(`mynah_tts_model_warm`)*. Measured: per-worker physical footprint **572.6 → 201.2 MB**,
      parent 1.7 → 558.7 MB, audio byte-identical — **≈5.6 GB at W=16**. Gated on >1 worker
      because at W=1 it is a 186 MB *loss*. The `prefork.h` warning was narrowed rather than
      deleted: BNNS is 68 references across `conv1d.c`/`codec_nanocodec.c` (Magpie) and
      **zero** in the PocketTTS path. **Still open:** the f16 pack (183 MB) is owned by the
      engine *state*, so it is still built per worker; sharing it means moving that cache to
      the model — a different ownership question. Original item text follows.
- [~] E10-3b **it may already be shared — and macOS cannot tell us.** Reading the code
      after E10-3 landed: `state->qcache = model->qcache` (the cache is *already*
      model-owned), `pocket_model_free()` does not free it, and `pocket_prepack_claim()`
      memoises claimed caches in a **process-global** array — so the parent's warm builds
      the pack, marks it, and children inherit both and skip prepack. If that holds, the
      f16 pack is shared too and there is nothing left to move. **Unresolved here:**
      per-worker physical footprint is flat at 216 MB whether W=2 or W=6, but that metric
      does not separate inherited pages from private ones, and system-level deltas came
      back 706 MB at W=2 against 482 MB at W=6 — noise, not signal. **Answer it on Linux
      in one line:** `/proc/<worker>/smaps_rollup`, `Private_Dirty` against `Shared_Clean` —
      both are immutable after build and both are built **per worker, after the fork**.
      Measured per-worker private footprint: 582.1 MB (f16) / 414.1 MB (quant off), so
      **≈8.3 GB at W=16**. The blocker in `server/prefork.h` does not apply to PocketTTS:
      the pthread_t-keyed BNNS cache is in `conv1d.c`/`codec_nanocodec.c`, the Magpie
      path, and `seanet.c` contains **zero** BNNS references. Warm conversions and prepack
      only — never a synthesis. Also corrects the record: the 594/767 MB pair was RSS and
      each contained the **same** 209 MB shared mmap of `tts.safetensors`
- [~] E10-4 **batched int8 kernel landed (`ba2200d`), and the per-row GEMM loop is routed
      into it (`E10-4c`)** — `matvec_q8_neon_x4`, signed SDOT, bit-exact by arithmetic
      (int32 accumulation). `mynah_qmat_linear_resolved_qt` no longer runs a `[count][k]`
      block as `count` GEMVs. Measured under `MYNAH_QUANT=int8` on `codec.transformer`:
      **1.141× / 1.142× / 1.142×** at batch 2 / 3 / 4. Byte-identical 180/180.
      **Two open ends, both load-bearing:** a single stream gains nothing (every AR step
      is `count=1`, which is 72.5% of pool regions and cannot be batched against itself),
      and **the default spec shows no change at all** (374.0 → 375.8 at batch 3) although
      it names int8 for `codec_transformer` — unexplained, and the difference between a
      measured win and a production one
- [ ] E10-4b **the x86 u8/VNNI batched int8 kernel** — still the fall-through-only path
      there, and it cannot be executed or measured on an arm64 machine. Needs an x86 box
- [x] E10-4d **the default spec did not do what its own measured comment says** — a bare
      clause means "whatever `MYNAH_QUANT` says", and once `src/mynah_tts.c:402` began
      requesting f16 when `MYNAH_QUANT` is unset, the two codec clauses inherited f16 and
      the shipped default became **"f16 everywhere"** — the one configuration that comment
      never measured. Fixed with a pinned twin used only when `MYNAH_QUANT` is absent (an
      unconditional pin also changed `MYNAH_QUANT=int4`, 36/180 cases — caught by the
      identity sweep, not by reasoning). **`codec.transformer` 183.5 → 130.9 ms, 1.40×**,
      WAV the same length to the byte so frame count and EOS do not move. Changes default
      audio, deliberately, and nothing else: 144/180 identical, the 36 differing are all
      the default. This also closes the other half of **E9-5**
- [-] E10-4-orig **original text** — the f16 hole fixed in `8614117`
      one floor down. **x86 falls straight through to `for (b) qmat_rows_dispatch(...)`:
      at m=16 that is sixteen passes over the weight.** ARM has only the 2-wide
      `matvec_q8_pair_i8mm` → eight passes. The structure above is already right
      (`pocket_proj_tile` hands 16 rows down as one call). Touches **22.9%** (codec
      transformer). The reference measured 2.04-2.78× on Sapphire Rapids, 2.48-2.79× on
      EPYC, **2.03-2.10× on Graviton3 — and its self-test reports SMMLA int8 matmat
      L2 = 0.00e+00, bit-identical to B× SDOT matvec**, which is the property our per-row
      promise needs. No dependency; ~250 lines
- [ ] E10-5 **the SEANet conv stack is entirely f32** — `codec_conv` in our spec is only
      `mimi.quantizer.output_proj [512][32]`, so **25.6% of the wall has no quantized
      kernel at all**, and the depthwise upsample is *permanently scalar*
      (`groups == out_channels == 512` can never qualify). The reference measured **−18%
      of whole-request wall** from its int8 decoder conv on Neoverse-N1. **Skip the three
      convtranspose stages** — it measured those slower than f32 sgemm and ships them off.
      Needs its own quality gate (frame count, EOS step, log-mel corr)
- [ ] E10-6 **weighted-LSQ int4 block scales** — the reference solves the block scale in
      closed form with `w = v²` instead of `amax/7`; same layout, same bytes, same
      kernels, ~10 lines. Measured there: word accuracy **83.9% → 90.9%**, utterance
      duration **+71% → +22%** against the gold. Ours is naive absmax RTN
      (`src/qmat.c:490`). For the codec only — **not** the AR loop
- [-] E10-7 **falsified at production width: the site named is making the right call.**
      The audit pointed at `src/seanet.c:410` (`sea_elu_task`, `n=2`, **22% barrier**, 86%
      of regions below break-even) as the worst-behaved dispatcher at two threads. Forcing
      that ELU serial and measuring: `codec.conv_stack` **172.0-176.6 ms serial against
      153.9-156.3 parallel**, audio identical either way. The region pays 12% for its
      barrier and wins anyway — the thresholds at `SEA_ELU_MIN_PARALLEL` are measured
      (~35 µs per region, ~2.5 ns per element) and they are right. The 22% is the price of
      a correct decision, not a defect.
      What is left of this item is **dispatch count**, not thresholds: 62 regions per frame,
      and cutting them needs a persistent team with an intra-region barrier, which
      `src/threads.{c,h}` has no primitive for. That is a structural change, and the
      audit's own ceiling for the whole park/wake question at two threads is **0.8-2.7%**.
      Not worth it before the box says the topology survives (E10-10)
- [ ] E10-8 **the spin budget gate keys on the OS, not the ISA** — `#if defined(__linux__)
      && defined(__aarch64__)` → 65536, everything else 4096, and `src/threads.c:112-117`
      already admits the value is *"transferred from the reference's measurement, not
      measured by us"*. Two consequences: macOS arm64 takes 4096 despite being the same
      ISA with the same `yield`; and **Linux x86-64 production runs 4096 `pause`es, never
      measured**. It is an iteration count, not a time budget, and `yield` vs `pause`
      differ 50-100× in duration — **one integer cannot be right on both**
- [ ] E10-9 **our C64 is a wave result and the reference has the wave-vs-soak gap
      measured** — their closed-loop runs push STREAM_RTF past 1.0 and make stalls
      material at the same concurrency, with zero rejects or timeouts. Their regression
      reproduces in **2-5 minutes at C6-C8**. We already have `--mode soak` (proper
      closed loop, warm-up discarded, windowed drift gate, unit-tested) and **have never
      run it**. Run it low and short before trusting C64
- [ ] E10-10 **the topology claim comes from one machine** — `16x2` beats `1x32` by 2.2×
      on RTF p95, measured only on the Axion. The reference ran the same class of question
      across three Arm parts: on Graviton5 a **4×8** shape lost **4.45× per worker** with
      aggregate bandwidth flat, against 1.57-1.81× on the V2 hosts. **Our 16×2 is four
      times more aggressive than the shape that collapsed.** A standalone simultaneous-worker
      screen is written (scratchpad, `shape_screen.c`, 4096×1024 f16 = our `linear1`, sized
      so 16 workers exceed the 80 MiB L3); it needs ≥32 cores and cannot be answered on an
      8-core Mac
- [ ] E10-11 **serving-loop observability** — we can say where wall time went but not how
      many slots were live per frame, nor *why* a free slot stayed free, nor the
      share-of-wall that was "blocked: no work queued" (the row that decides whether C80's
      1.120 is saturation or variance). The reference has all three plus a per-request
      stage decomposition from admission to first audio. Our 585 ms burst model is
      arithmetic that lands within 4% of measurement — good, but a model
- [ ] E10-12 **small, each real** — `getenv` + `strcmp` on **every** conv call
      (`src/conv1d.c:427`, ~97 per decode) where three neighbouring sites memoize;
      `rows->gelu` allocated `count * ffn_dim` floats and **never written**
      (`src/transformer_ar.c:196`, `mynah_gelu_tanh_array` ignores its scratch argument),
      256 KB dead per live request; `serve()` re-running `engine->model_init` per
      `mynah_tts_synthesize` on the one-shot path — **1,933 allocations, 27 opens, 28
      mmaps and ~1.1 ms per request**, re-validating all 26 voice files each time; and
      **277 MB of KV `calloc`'d per request** sized from max steps rather than the
      admitted text
- [-] **rejected, with the reference's own numbers.** Do not build these: prefill helper
      thread (stall@250 20.1%→46.8%); token-range slicing (occupancy floor invariant at
      83-97 ms); per-layer prefill checkpointing (TTFA p95 223→1208 ms); fixed-target
      admission guard (TTFA p95 241→2805 ms, largest defer 19.4 s, −14% throughput);
      dedicating a core to admission (stall@500 8.6%→24.4%); kernel-layout weight prepack
      (+1.4 GB for +1.1/+3.5%, and +4.3 GB/+10.8% TTFA on the AMX side); batched int4 GEMM
      (0.80-0.97× on three x86 boxes); ConvTranspose int8 (slower than f32 sgemm);
      KleidiAI (Apache-2.0, 40 files, hand-written asm, a second packed copy of every
      weight, **and no committed head-to-head against their own SMMLA**); storing f32
      weights instead of f16 (**2.22× slower on the backbone**, measured); a different f16
      packing (all four candidate layouts lose to the current row-4 row-major); `FMLAL`
      (needs the activation narrowed to f16, which is what the spec exists to prevent)

### E5 — Streaming server v2 → [`.work/streaming-server-v2.md`](.work/streaming-server-v2.md)
Design reference: [`.work/serving-design.md`](.work/serving-design.md) ·
doctrine: [`.work/serving-doctrine.md`](.work/serving-doctrine.md)

Supersedes §24 (P0-P3, all landed). The limit now is concurrent streaming.

**Do not build these** — each was built, measured and lost in the reference
implementation; the mechanism of each failure is architectural, not host-specific
([`.work/serving-design.md`](.work/serving-design.md) §9): global cross-worker
batching (useful coincidence 1.6% at ±0.25 ms against a 25% bar), any
fairness/credit gate (parks 95.8% of checks, 0.838 → 0.986), a priority-based
prefill helper (TTFA 435 → 2379 ms), utilization-aware admission (stall@250
0 → 50%), a second submitter on the engine pool (TTFA 167 → 1200 ms), and a wide
single pool (`1x32` at C8: STREAM 1.55, 62% of frames stalling past 500 ms).

- [x] E5-1 **done** `6ed9c63`/earlier — the scheduler owns `ctx`, nothing left to serialize · **remove the global stream mutex** (`server/main.c:451-460`) — scheduler owns `ctx`
- [x] E5-2 **done** verified: streaming and offline enter the same slot driver · streaming requests enter the same slot driver as offline; no second path
- [x] E5-3 async output writer **done** (`server/stream_out.{c,h}`): a slow client blocked
      the whole process for 66.5 s, now 4.81 s; leaks/ASan/UBSan/TSan clean
- [x] E5-4 **done** `6ed9c63` · cancel on disconnect (`POLLRDHUP`), slot freed within one frame
- [ ] E5-5 long-form **BLOCKED on an engine hook**, and the driver half is already done →
      [`.work/decoder-lane.md`](.work/decoder-lane.md) §6. `pocket_decode_audio` already
      carries conv state and a position counter across calls and refuses a
      non-contiguous range, and `serve()` already decodes in contiguous monotone ranges
      from slots that outlive a call. What is missing is appending TEXT to a prepared
      context: `pocket_ctx_new` sizes `text_embed`, the backbone KV, `latents` and the
      codec KV from a text length known at admission, so the hook and its allocations
      belong in `src/engine_pocket.c`. Do not settle for "chunk the text into N
      requests" — that cannot be byte-identical to the one-shot
- [x] E5-6 **prefork done** (`85d6380`): `server/prefork.{c,h}`, parent opens the pack
      then forks W workers and hands each accepted fd to the least-loaded one over
      `SCM_RIGHTS`. Only `sched_setaffinity` and the sysfs read are Linux-only, so the
      topology runs unpinned on macOS and says so. `MYNAH_THREADS` is fixed **before**
      the model opens — the pool caches its width, so a `setenv` in the child would have
      been a silent no-op that reads as "prefork does not help". 15/15 server tests in
      prefork mode. **Not verified: real `sched_setaffinity`, a real `/sys` read, any
      many-core number** — E5-23
- [ ] E5-23 **verify prefork on real Linux hardware**: `sched_setaffinity` against the
      kernel, `/sys/devices/system/cpu` topology, and the four-step W/T procedure that
      `--prefork-plan` already prints. Nothing about capacity is claimed until this runs
- [x] E5-7 serving profile **done and baselined** (C1 GOOD, C2/C4 MARGINAL — TTFA p95
      110/1847/5596 ms while RTF stays 0.30-0.35; that is the mutex): `tools/serving_profile.py` + `tests/playback_sim.py` —
      C1/C2/C4/C8 with **TTFB, TTFA, STREAM_RTF, prebuffer, stall rate, max gap**,
      each p50/p95, and a GOOD/MARGINAL/NOT-STREAMABLE verdict with its thresholds
      printed so the verdict is falsifiable. Must run against the synthetic pack
      *and* a PocketTTS pack without assuming the engine.
- [x] E5-10 **done** `89070f8` — the timing assertion is deleted, not tuned; the identity checks gained a must-differ control · `tests/test_server.sh` `batching` check needs sub-second timing (flaky, pre-existing)
- [x] E5-9 **done** `0fbf374` — one worker process per language → [`.work/multi-language-serving.md`](.work/multi-language-serving.md) · **per-language slot groups** — batching cannot mix languages. Not a
      transport change: `mynah_graph_serve_continuous()` binds one model for the
      scheduler's life and `language` is only a tokenizer argument, never a routing
      key. Needs a model registry in the driver, or one process per language in
      `server/prefork.c` — decide which before building either
- [x] E5-11 **done** `facdc06` · **admission control against a real-time budget**. Measured: the machine
      sustains ~3x real time aggregate, so at C8 each stream gets 0.35x and stalls.
      There is no setting that makes them all GOOD — only the choice between waiting
      and stuttering — so the server has to choose deliberately and say which
- [x] E5-13 **done** `facdc06` · **prefork preconditions** — costmap mutexes must be *reinitialized* not
      zeroed (an inherited locked mutex can never be unlocked), and no GPU backend state
      may exist before the fork: that is their still-open bug, a wrong answer rather
      than a crash
- [x] E5-14 **done** `85d6380` · **core-major slices, not contiguous logical ones** — Linux numbers the first
      thread of each core first, so a contiguous slice gave two workers the same twelve
      physical cores, one per hyperthread. And print the mask actually set
- [ ] E5-15 **dispatch gate + host profile that refuses to run**, listing the env vars
      that must be *absent*. Their proof it is needed: the right config was versioned
      and three campaigns were still run wrong from memory
- [ ] E5-12 **prefork with pinned core slices** (E5-6) is the mechanism that turns
      cores into streams; without the topology, more cores are not more streams
- [x] E5-16 **done** `18c7578` · **the measurement protocol, before any tuning** → [`.work/serving-design.md`](.work/serving-design.md) §10.
      WAVE (3 synchronised waves) is a *screen*; only a 5-30 min SOAK with a drift gate
      promotes. Their C16 passed the screen at 0.919 and failed the soak at 1.004 with
      596 rejects. Gate on **audio too**: their whole "all-on" ARM profile was faster
      and was rejected at mel-correlation 0.886-0.945 against 0.98
- [x] E5-17 **done** `6ed9c63` (verified structurally, no fix needed) · **send the response header at admission, not at the first chunk** — otherwise
      TTFB equals TTFA and the entire prefill cost is invisible to the client metric
      (theirs: TTFB p50/p95 0.4/0.6 ms vs TTFA 82.6/84.2)
- [x] E5-18 **done** `6ed9c63` · **`TCP_NODELAY` + `SO_RCVTIMEO` on every accepted socket**; coalesced reads
      6.7% → 0.0%. And **name every thread** for `/proc`: free, and it makes the
      ownership table readable without a debugger
- [x] E5-19 **done** `facdc06` · **keep polling the listener while full, and refuse** — a full server that
      stops accepting hides the wait in the kernel backlog where no deadline can see
      it. Theirs measured TTFB/TTFA p95 4470/4635 ms of which >97% was before `accept()`
- [x] E5-20 **done** · **warm up through the same reset the request path uses** → [`.work/decoder-lane.md`](.work/decoder-lane.md) §1.
      `--warmup N` (default 1) enqueues an ordinary job on the ordinary queue, so it
      walks `ctx_new`/`prepare`/`ctx_free` exactly as a request does. Gated by
      `tests/test_server.sh warmup`: warm-first == cold-first == cold-second, with a
      must-differ control. Theirs warmed on leftover CLI state and the first real
      request differed from every one after it
- [x] E5-21 **done, OFF by default** (`--decoder-lane N`) → [`.work/decoder-lane.md`](.work/decoder-lane.md) §2.
      Private **pinned** team on the last N cpus, own submit lock, bounded
      one-unit-per-slot mailbox, redirection inside `mynah_parallel_for()` by a
      thread-local tag — one branch, never discipline at call sites. **Refuses**
      rather than narrowing: both sides need 4 cpus (theirs: 6+2 gave 1.364 vs 0.997
      at 4+4) and it will not run unpinned (theirs: 21 threads on 8 cores). Verified
      engaged and pinned at 4+4 on 32-core Neoverse-V2, byte-identical through
      `server-test` and `server-concurrency-test`, ASan and UBSan clean, 0 mailbox
      overruns. **The lane and the decode gang are alternatives, not layers** — which
      is faster here is unmeasured, so the gang stays the default
- [~] E5-22 **spin budget exposed, not tuned; per-CCX bandwidth still unmeasured** → [`.work/decoder-lane.md`](.work/decoder-lane.md) §3.
      The pool now spins before it parks: `MYNAH_POOL_SPIN`, default 65536 on
      Linux/aarch64 and 4096 elsewhere, in the dispatch table as `pool.spin` with its
      source and its park/spin counters. The default is **TRANSFERRED** — their sweep
      moved STREAM p95 0.893 → 0.808 with context switches 38k → 7.6k/s — and the
      curve does not port, so pin the measured value per box. Still to do: sweep it
      here, and the per-CCX bandwidth reading (their 16-thread mask read *slower
      cache-resident than DRAM*). Both are ten-minute measurements
- [x] E5-8 **done** `89070f8` — proven through HTTP at C=2/4/8, both routes, either arrival order, ragged, single-process and across 4 prefork processes, with two injected contamination mutants caught · gate: **N concurrent streams byte-identical to the same request run alone**

### E6 — Licensing and voice policy → [`.work/licensing-and-voice-policy.md`](.work/licensing-and-voice-policy.md)

- [ ] E6-1 `speakers.json`: `source_dataset`, `license`, `commercial_use` per voice
- [ ] E6-2 default pack ships CC0/CC-BY voices only — `jean` and `cosette` are **non-commercial**
- [ ] E6-3 verify the `embeddings_v3` voice→dataset mapping (assumed CC0, unconfirmed)
- [ ] E6-4 `LICENSES/` + `NOTICE`: CC-BY attribution, modification statement, prohibited-use as notice
- [ ] E6-5 `source.json` pins the official Kyutai repo, never an ungated mirror
- [ ] E6-6 CLI and `/v1/voices` expose the license; `--commercial-only` filter

### Deferred

- [~] **CI red on the OSS repo — job fixed, awaiting one run to prove it** →
  [`.work/ci-red-oss.md`](.work/ci-red-oss.md). `main` was always green; both
  failures were `workflow_dispatch` runs on `lane/int8-*` and both trace to **one
  job bug, not a code bug**: `link-only: x86 SIMD=avx512` executed the binary it
  built on a runner reporting `avx2 fma`, so the startup guard correctly refused
  and exited 1. The guard is untouched; the step now accepts *started* or
  *refused with the guard's own message* and still fails on SIGILL (132) or any
  other exit, verified against four fakes. Left open on purpose: the run that
  proves it (needs a push, twice green — the matrix flapped) and `--self-test`
  for the whole matrix, which waits until the Axion box is free. AVX-512 gets no
  execution coverage on any hosted runner; that is now stated in the note rather
  than implied by a green tick.
- [-] GPU (Metal/CUDA) work — existing backends stay as they are. Metal measured
  *slower* than CPU on Apple Silicon (`docs/performance.md:71-80`).
- [-] `*_24l` PocketTTS variants — non-distilled previews, 672 MB-1.3 GB each,
  same schema with `num_layers: 24`. French exists **only** in this form.
- [-] dots.tts (§9) and Chatterbox (§10) — unchanged as later engines.

## 1. Product mission and soul

Mynah TTS is not a NeMo fork or a general tensor framework. It follows the
working style of `../mynah` ASR and `../qwen-tts`: a small C11 runtime that
loads converted weights and runs without Python at runtime.

Non-negotiable principles:

- C11 is the center: cohesive modules, a small stable C API, and no Python or
  C++ in the default CPU path.
- CPU is the product. `make` must build a useful macOS/Linux binary without a
  GPU SDK; Accelerate/OpenBLAS are optional accelerators.
- Offline, streaming, and long-form synthesis share one state machine. Only the
  audio sink changes.
- Model dimensions, tokenizer behavior, codec layout, and sampling defaults
  come from `model.json`, never from Magpie-specific public constants.
- Every new stage is compared with a NeMo/PyTorch oracle before SIMD,
  quantization, or GPU work.
- The hot loop reuses mmap-backed weights and scratch memory; it does not parse,
  allocate, or load files.
- Metal and CUDA are opt-in, resident-aware, and able to fall back to CPU with a
  clear diagnostic.
- Runtime and model weights are separate. Weights, checksums, tokenizer assets,
  and licenses belong to a model pack and are never committed.

| Surface | Name |
|---|---|
| Repository | `mynah-tts` |
| Library | `libmynah_tts` |
| CLI | `mynah-tts` |
| Future server | `mynah-tts-server` |

## 2. v1 target: MagpieTTS

Primary reference: [NVIDIA MagpieTTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m).
The pinned v2607 release is approximately 364M parameters, covers 12 languages
(`ar`, `de`, `en`, `es`, `fr`, `hi`, `it`, `ja`, `ko`, `pt`, `vi`, `zh`), has five
baked-in voices, and produces mono audio at 22.05 kHz. It must be treated as a
different checkpoint from older releases.

The initial conceptual graph is:

```text
UTF-8 text
  -> text normalization / G2P / language tokenizer
  -> causal text encoder
  -> causal decoder with self- and cross-attention
  -> frame stacking
  -> local multi-codebook transformer
  -> NanoCodec decoder
  -> mono float/PCM WAV at 22050 Hz
```

Reference codec: [NVIDIA NanoCodec 22 kHz](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps).
The initial model card describes 8 codebooks, size 2016, embedding dimension
32, 21.5 frames/s, and FSQ levels `[8, 7, 6, 6]`. Mynah v1 needs the decoder;
the codec encoder is offline tooling only.

These values are hypotheses until conversion confirms them. The checkpoint and
the generated `model.json` are authoritative for layer counts, dimensions,
token IDs, local-transformer mode, KV layout, CFG, and codec state.

v1 supports Magpie's baked-in speakers. It does not promise zero-shot cloning.
Text normalization and G2P must either be exported as deterministic assets or
be made explicit with a `--normalized` contract; the runtime must not silently
replace NeMo text behavior.

## 3. Patterns to reuse

From `../mynah` ASR:

- modular C11 layout (`src`, `cli`, `tools`, `tests`, optional `server`);
- converted model packs, manifests, mmap weights, and explicit model skips;
- an engine/vtable boundary for multiple families;
- Python only for conversion, oracle dumps, inspection, and evaluation via `uv`;
- `make` as the main interface and all artifacts below `build/`;
- CPU/Metal/CUDA capability reporting with safe fallback;
- stage-level numerical parity, UBSan, and native macOS leak checks.

From `../qwen-tts`:

- safetensors views, aligned scratch, and separate kernels for matvec, matmat,
  norms, attention, softmax, sampling, and convolution;
- GCD on macOS and persistent pthread workers on Linux, with the main thread
  participating and no hot-loop allocation;
- `--caps`, deterministic self-tests, and reproducible benchmark reports;
- quantization only after a correct floating-point baseline;
- GPU offload of resident subgraphs rather than host/device copies per op;
- a separable causal audio decoder that can be pipelined later.

Do not copy monolithic source files, placeholder translation units, hardcoded
Qwen dimensions, divergent offline/streaming implementations, or unmeasured
performance claims.

## 4. Repository shape

```text
mynah-tts/
├── src/
│   ├── mynah_tts.h/.c       # public API, model/context lifecycle
│   ├── tts_engine.h         # engine vtable and capabilities
│   ├── engine_magpie.c      # Magpie graph composition
│   ├── text.c/.h            # UTF-8, chunking, special tokens
│   ├── tokenizer.c/.h       # exported tokenizer assets and IDs
│   ├── transformer.c/.h     # shared transformer blocks
│   ├── attention.c/.h       # causal/cross attention and KV cache
│   ├── sampler.c/.h         # greedy, temperature, top-k/top-p, RNG
│   ├── codec_decoder.c/.h   # NanoCodec FSQ and causal decoder
│   ├── audio.c/.h           # WAV/PCM and audio sinks
│   ├── weights.c/.h         # mmap model pack and tensor views
│   ├── kernels.c/.h         # scalar reference kernels
│   ├── kernels_neon.c       # optional ARM kernels
│   ├── kernels_avx.c        # optional x86 kernels
│   ├── threads.c/.h         # cross-platform worker API
│   ├── backend.c/.h         # CPU resolver and capability reporting
│   ├── metal_backend.m      # only in `make metal`
│   └── cuda_backend.cu      # only in `make cuda`
├── cli/main.c
├── tools/
│   ├── inspect_nemo.py
│   ├── convert_magpie.py
│   ├── convert_dots.py
│   ├── convert_chatterbox.py
│   └── oracle/               # offline NeMo/PyTorch reference dumps
├── reference/                # small configs, reports, and tokenizer metadata
├── models/                   # ignored local checkpoints and model packs
├── tests/
├── docs/
├── Makefile
├── CLAUDE.md
├── AGENTS.md
└── PLAN.md
```

Do not create an all-purpose `tensor.c`. A small typed view (`data`, `dtype`,
`ndim`, `shape`, `stride`) is enough to describe existing buffers; hot
operations call explicit kernels so CPU, Metal, and CUDA share a clear seam.

The engine boundary should stay small:

```c
typedef struct {
    int streaming;
    int longform;
    int baked_speakers;
    int local_transformer;
    int codec_decoder;
} mynah_tts_caps;

typedef struct mynah_tts_engine {
    const char *name;
    mynah_tts_caps caps;
    int  (*prepare)(mynah_tts_context *, const mynah_tts_request *);
    int  (*step)(mynah_tts_context *);
    int  (*flush)(mynah_tts_context *);
    void (*reset)(mynah_tts_context *);
} mynah_tts_engine;
```

The model manifest selects the engine. Shared primitives and services are
reused; each engine keeps its graph, conditioning, sampler, EOS, and decoder
state explicit.

## 5. Model pack and converter

The runtime must not read a raw `.nemo` tarball. `convert_magpie.py` must:

1. validate the archive, source revision, checksum, license, and codec;
2. inspect configuration, tensor names/shapes/dtypes, tokenizer, speakers,
   special tokens, KV cache, and local-transformer settings;
3. write normalized `model.json` and row-major mmap-friendly tensor files;
4. export tokenizer/G2P/TN assets, speaker map, and special IDs;
5. emit a tensor report with offsets and checksums;
6. save `source.json` with repository, revision, date, and command line.

Expected pack:

```text
models/magpie-v2607/
├── model.json
├── tts.safetensors
├── codec.safetensors
├── tokenizer/
├── speakers.json
├── source.json
└── LICENSES/
```

`model.json` must describe engine, revision, dtypes, hidden sizes, layers,
heads/KV heads, positional encoding, vocab and special IDs, decoder mode, frame
stacking, codebook count/size/dimension, EOS policy, sample/frame rates, codec
configuration, inference defaults, and language routing.

If exact NeMo text normalization cannot be exported safely, the converter must
mark the pack as requiring normalized input. No silent fallback is allowed.

## 6. Shared inference state

`mynah_tts_synthesize()` and streaming push/flush use the same state machine.
The audio callback receives float PCM chunks; WAV or stdout is only a sink.

Request state owns text tokens, encoder state, decoder position/KV cache, local
transformer state, codec convolution state, RNG/sampler, speaker/language,
CFG, frame counters, EOS, abort, and flush status. Context reset must not leak
the previous speaker or text. Allocate capacity before the autoregressive loop;
grow only for an explicitly supported long-form limit.

Magpie stages:

1. UTF-8, language routing, optional TN/G2P, and tokenization.
2. Causal text encoder with learned positions.
3. Causal decoder with audio BOS, speaker conditioning, self-attention,
   cross-attention, logits, and EOS.
4. Optional CFG using the same context and explicit `cfg_scale`.
5. Stacked-code sampling with model-configured ordering and EOS rules.
6. Local transformer in exactly the mode declared by `model.json`.
7. NanoCodec embedding/FSQ, causal convolutions, upsampling, clipping, and
   final float/int16 conversion.
8. EOS and flush that close the causal queue and keep offline/streaming output
   equivalent within the declared tolerance.

Long-form chunking must preserve language-aware text state, global position,
EOS, and codec state. First ship one-utterance streaming; add long-form only
after chunk boundaries are proven click-free and speaker-safe.

## 7. CPU, SIMD, and memory

The scalar reference uses float32 accumulation with pack-declared bf16/fp16/fp32
weights. Required primitives are embedding lookup, dtype conversion, matvec,
matmat, norm, residual, FFN, causal/cross attention, softmax, KV operations,
sampling, convolution, upsampling, FSQ, and PCM conversion.

| Platform | Baseline | Later candidates |
|---|---|---|
| Apple arm64 | scalar + NEON + Accelerate/GCD | SDOT and fused matvec |
| Linux aarch64 | scalar + NEON + pthread/OpenBLAS | SDOT when available |
| Linux x86-64 | scalar + AVX2/FMA | AVX-512/VNNI/BF16 when verified |
| macOS x86-64 | scalar + optional AVX2 | only when measured |

`SIMD=scalar` always remains available. `--caps` reports compiled and runtime
ISA. Distributed binaries must not rely on `-march=native`; ISA functions are
isolated and tested against scalar output.

Use mmap views for safetensors, aligned reusable scratch, load-time quantized
weight caches, checked `size_t` arithmetic, and explicit ownership. Start with
bf16/fp16/fp32. Evaluate INT8 and Q4 only after audio quality and EOS/length
behavior are measured; keep the codec decoder in f32 until proven otherwise.

## 8. Optional GPU backends

CPU builds include no Metal/CUDA headers. A backend resolver reports `active`,
`not compiled`, `no device`, `unsupported model`, or `runtime failure`. A
recoverable request falls back to CPU; a correctness self-test failure does not.

Metal is macOS-only: resident `MTLBuffer` weights/KV/scratch, reusable command
buffers, and fused prefill, complete autoregressive steps, or NanoCodec work.
Do not offload one tiny matvec if dispatch and synchronization cost more than
the computation. Validate with the same CPU golden dumps.

CUDA is Linux/NVIDIA-only: begin with cuBLAS for prefill/matmat, then resident
weights/KV and a complete decoder step per launch/stream, and only later CUDA
Graphs or custom kernels. `CUDA_ARCH` is configurable and CPU startup must not
require a GPU. Single-request latency and batched throughput are separate
measurements.

The common backend seam can share buffer pools, dtype conversion, GEMM,
matvec, norm, attention, convolution, reductions, capability queries, and
self-tests. The resident graph remains engine-specific:

```text
Magpie:      AR step -> local transformer -> NanoCodec
dots.tts:    LLM step -> flow/MeanFlow NFE loop -> AudioVAE
Chatterbox:  T3 decode -> S3Gen/vocoder -> optional watermark
```

## 9. v2: dots.tts

Reference: [dots.tts](https://github.com/studio-dots-ai/dots.tts), its
[technical report](https://arxiv.org/abs/2606.07080), and the
[`dots.tts-mf` checkpoint](https://huggingface.co/rednote-hilab/dots.tts-mf).

dots.tts is a roughly 2B-parameter continuous AR system, not a discrete codec
model. It combines a semantic encoder, Qwen2.5-1.5B LLM, AR flow-matching or
MeanFlow head, CAM++ speaker x-vector, and a frozen 48 kHz AudioVAE/causal
BigVGAN-style decoder. The distilled `dots.tts-mf` NFE=4 path is the first
candidate for “fast”; the reference GPU profile is an optimization clue, not a
CPU target.

v2 must import the semantic encoder, LLM, flow head, AudioVAE, speaker encoder,
and BPE assets separately. Keep the flow sampler (`num_steps`, schedule,
guidance) explicit, use a bounded paged/ring history, and stream causal audio
patches when the decoder permits it. CPU scalar is required for correctness and
portability; the release remains GPU-first until a measured CPU baseline proves
otherwise. Prompt audio and voice cloning require an explicit consent and
duration policy.

Do not make dots a Magpie flag: continuous latent patches, ODE integration,
full-history conditioning, and AudioVAE state are a separate engine graph that
only reuses tensor primitives and backend services.

## 10. v3: Chatterbox

References: the [official repository](https://github.com/resemble-ai/chatterbox)
and [model card](https://huggingface.co/ResembleAI/chatterbox). The family has
different checkpoints and must not be autodetected by a loose name. Pin Turbo
and Multilingual V3 as separate model-pack variants.

The conceptual graph is:

```text
text tokenizer
  -> T3/Llama autoregressive speech-token generator
     + speaker x-vector/reference audio + language/CFG/exaggeration
  -> speech-token or mel generation (checkpoint-specific S3Gen)
  -> HiFi-GAN-like vocoder/waveform
  -> optional Perth watermark
```

Start with Turbo for latency and then add Multilingual V3 for language coverage
and cross-language reference voice. The converter must pin T3, S3Gen, voice
encoder, tokenizer, conditioning cache, alignment/EOS, and watermark metadata.
Keep `cfg_weight`, `exaggeration`, temperature, and watermark policy in request
state. Reference audio needs explicit consent and bounded cache lifetime.

## 11. Reuse matrix

| Component | Magpie v1 | dots v2 | Chatterbox v3 | Policy |
|---|---|---|---|---|
| mmap/model pack/WAV | yes | yes | yes | shared services |
| tokenizer API | NeMo/G2P | Qwen BPE | checkpoint tokenizer | shared interface, separate assets |
| linear/norm/attention | yes | yes | yes | shared scalar/SIMD/GPU primitives |
| KV/history | causal + cross | paged full history | causal T3 | engine-owned layout |
| sampling/EOS | codebooks | AR + flow | speech tokens/alignment | pluggable implementation |
| audio latent | discrete codes | continuous patches | speech tokens/mel | engine-specific |
| audio decoder | NanoCodec | AudioVAE | S3Gen/vocoder | shared sink, separate graph |
| speaker conditioning | baked IDs | CAM++ x-vector | reference voice | engine-specific |
| GPU residency | TTS + codec | LLM + flow + VAE | T3 + S3Gen | common seam, separate graph |

SIMD work reusable across all models includes dtype conversion, dot products,
matvec/matmat, norms, activations, reductions, RoPE, attention, KV copy,
convolution, upsampling, and quantization packing. v2/v3 additionally need
patch batching, variable-length paged history, DiT/flow math, reference-audio
resampling, x-vector pooling, AudioVAE residual convolutions, and latent
interleaving. Reuse algorithms and tests, not necessarily packing or shapes.

## 12. Makefile contract

The Makefile must be explicit and must never download weights implicitly:

```text
make                         # default CPU build and CLI
make cpu | blas | info       # CPU variants and capability report
make self-test | test        # model-free kernels; available model tests
make bench | bench-matrix     # JSON performance reports
make inspect MODEL=...       # checkpoint/model-pack report
make convert MODEL=...       # Magpie converter
make oracle MODEL=... CODEC=... BYT5=... OUTPUT=... # official CPU reference WAV
make convert-dots MODEL=...  # dots.tts converter
make convert-chatterbox MODEL=... VARIANT=turbo|multilingual-v3
make test-model FAMILY=... MODEL_DIR=...
make metal | cuda             # isolated opt-in backend builds
make gpu-selftest | leaks | ubsan | asan
make lib | shared | install PREFIX=... | dist
make clean
```

Supported variables: `CC`, `CFLAGS`, `LDFLAGS`, `SIMD`, `BLAS`, `ENGINE`,
`MODEL_DIR`, `THREADS`, `CUDA_HOME`, `CUDA_ARCH`, and `EXTRA_CFLAGS`. CPU,
Metal, and CUDA objects must live in separate build directories or be rebuilt
cleanly; they must never be silently mixed.

## 13. Oracle, tests, and benchmarks

Offline Python tooling must dump: normalized text/G2P/token IDs, encoder layer
outputs, decoder prefill and one step of logits, stacked codebooks, local
transformer output, EOS, codec output, and end-to-end WAV. Every stage checks
shape, finiteness, range, absolute/relative tolerance, and argmax/token parity
where applicable. Audio checks duration, RMS, peak, and mel correlation; it
does not require byte-identical output across floating-point backends.

Minimum tests cover UTF-8 and special IDs, scalar-vs-SIMD kernels, synthetic
codec codes, corrupted/missing packs, repeated load/unload, deterministic
sampling, speaker/language isolation, CFG, offline-vs-streaming, long-form
flush, thread counts, UBSan, ASan, and macOS `leaks`.

Benchmarks record commit, OS, CPU, ISA, threads, model revision, seed, dtype,
backend, load time, TTFA, RTF, frames/s, peak RSS, audio duration, and quality.
Do not set an RTF promise before measuring a baseline on representative Apple
and Linux hardware.

## 14. Roadmap and gates

| Milestone | Deliverable | Gate |
|---|---|---|
| M0 | pin Magpie/NanoCodec; inspect archive/config/tokenizer/licenses | repeatable audit |
| M1 | model pack and NeMo/PyTorch oracle | reloadable pack; token parity |
| M2 | mmap, scalar kernels, tokenizer, sampler, thread API | self-tests and UBSan |
| M3 | Magpie encoder/decoder, KV, CFG, stacked codes, local transformer | deterministic logits/codes parity |
| M4 | NanoCodec, WAV/PCM, CLI, errors | first verified WAV |
| M5 | shared streaming/long-form state | offline/stream equivalence; no leaks |
| M6 | NEON/AVX2, threads, measured INT8 | reproducible benchmark; quality held |
| M7 | library/server boundary | API and concurrency tests |
| M8 | Metal matmul/FFN backend | real-device shader self-test and benchmark |
| M9 | CUDA matmul/FFN backend | NVIDIA self-test and benchmark |
| M10 | dots.tts-mf, AudioVAE, flow engine | CPU correctness; bounded GPU-first profile |
| M11 | Chatterbox Turbo, then Multilingual V3 | EOS/audio parity; latency and language report |
| M12 | cross-model batching and hardening | no regressions across all packs |

Do not start GPU work before M3/M4 establish a correct graph. A faster wrong
pipeline is not progress.

## 15. Risks and open decisions

| Risk | Mitigation |
|---|---|
| `.nemo` layout changes | pin revision; version converter and tensor report |
| TN/G2P is not reproducible in C | export assets or require `--normalized` |
| local transformer differs from expectation | read config and implement only verified mode |
| CFG doubles cost | validate first; cache conditional state; benchmark on/off |
| causal codec clicks at chunk boundaries | codec golden tests and offline/stream comparison |
| AR decode is memory-bound | fused matvec, reuse, quantization, batch matmat |
| GPU copies erase gains | resident subgraphs and no per-op synchronization |
| cloning is inferred for Magpie | expose only baked speakers in v1 |
| dots is too large for CPU | CPU correctness + GPU-first release; history budget |
| Chatterbox variants diverge | separate pinned packs and converters |
| voice cloning/watermark misuse | consent, duration, metadata, and capability policy |

M0 must close exact tensor names/shapes, tokenizer format, decoder/KV/EOS rules,
local-transformer mode, NanoCodec causal state, baseline dtype, and runtime/model
license separation. Do not guess them in public headers.

## 16. v1 definition of done

v1 is complete when a pinned Magpie model pack loads on macOS arm64 and Linux
x86-64; `make` builds the CLI/library without GPU SDKs; supported-language and
speaker cases produce verified WAV; offline and streaming share state and pass
goldens; scalar/NEON/AVX2 paths self-test; load/error/UBSan/leak checks are
clean; and benchmark results are reproducible. Metal/CUDA are separate optional
targets with resident buffers, correctness tests, and CPU fallback.

The second and third model families remain follow-up engines, not hidden claims
inside the Magpie API. When this plan and implementation disagree, verified code
and the converted model pack are the source of truth.


---

## Archive

Sections 17-24 — the July/August 2026 session checkpoints, accepted and rejected
optimizations, and the Magpie server roadmap — moved verbatim to
[`.work/archive-2026-07-checkpoints.md`](.work/archive-2026-07-checkpoints.md).
Read it before re-trying Metal on Apple Silicon, vectorized Snake, cross-request
codec batching, or pool-threaded codec convolutions: all four were measured and
closed.
