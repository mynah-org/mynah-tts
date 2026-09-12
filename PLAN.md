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

**Production is Linux server, x86-64 and ARM64.** macOS/M1 is the development
machine and every number in this repo so far was taken there, which makes them
development signals rather than product claims. Three concrete reasons they do
not transfer: the SEANet GEMM fast path that bought 36x goes through BLAS, which
is Accelerate here and OpenBLAS there; the thread pool defaults to performance
cores via sysctl, a concept that does not exist on a 64-core NUMA server; and on
Linux x86 `SIMD=auto` compiles `-mavx2 -mfma` with **no runtime dispatch**, so
VNNI and AMX would never be selected in production however well they are
implemented. Closing that gap is E4-9.

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
- [ ] E1-6 real JSON parser with nesting and arrays, replacing `mynah_tts.c:112-163`
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
- [ ] E2-4 dump decoder-transformer behaviour past `context: 250` — needed to exercise
      the sliding window in `transformer_ar`, which no reference data reaches today
- [ ] E2-5 dump the text-chunk seam at `MAX_TOKEN_PER_CHUNK = 50`, including the known skip bug

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
- [ ] E3-6 new kernels self-tested model-free: LayerNorm **with bias** (two different
      epsilons), **variance-based RMSNorm** (`unbiased=True`, *not* `kernels.c:rmsnorm`),
      GELU-tanh, causal conv, adaLN
- [ ] E3-7 offline parity across all 12 oracle stages, both checkpoint generations
- [ ] E3-8 streaming sample-identical to offline, then batched through the shared driver
- [ ] E3-9 WAV smoke for all 6 languages with explicit language/voice/seed
- [ ] E3-10 **voice cloning from a wav — required, not optional** (see E7)
- [ ] E3-11 one pack = one language; compute `time_embed.*.freqs` at load instead of storing
- [ ] E3-12 NaN-as-BOS sentinel: reproduce it or track validity explicitly — never let NaN reach a matmul
- [ ] E3-13 pack refuses a voice KV from a different model/revision (upstream: it then never emits EOS)

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
- [ ] E4-2a add the 8 named predicates so no row resolves UNKNOWN
- [ ] E4-2b place the costmap hooks (deferred: the engines were being refactored)
- [ ] E4-3 baseline per-region profile of the PocketTTS path, on M1 and on EPYC
- [ ] E4-4 thread pool upgrade: lane split, deadline priority, `after_fork` (prereq for E5-6)
- [ ] E4-5 the kernel the profile names — scalar reference, then NEON/SDOT/i8mm **and** AVX2/AVX-512/VNNI in one change
- [ ] E4-6 int8 weight prepack with persistent cache, both ISAs
- [ ] E4-7 AMX-INT8 (Linux/x86 only), last
- [ ] E4-10 **remove useless dtype conversions** — called out by name as one of the two
      profiling wins. `src/qmat.c` holds 15 conversion sites and every other hot-path
      module holds zero; the suspicion is `bf16 -> f32 -> f16` where one step would do
- [ ] E4-11 **no silently-chosen scalar BLAS** — a scalar path taken without anyone
      knowing is worse than a slow one that announces itself. `blas.accelerate` is ON
      here and absent on the target, and the 36x conv-stack win goes through BLAS
- [ ] E4-12 **fatal ISA guard** — a `-mavx2` binary on a CPU without AVX2 gives an
      opaque SIGILL today. ~15 lines, checked before any allocation
- [ ] E4-13 **CI `link-only` job — no longer theoretical.** Our x86 CI builds only the
      default `-mavx2`, which is exactly the configuration in which their tree shipped
      unlinkable for days; ours shipped unlinkable too, and we found it by hand
      (E4-19). Add `SIMD=scalar`, `SIMD=portable`, `ARCH_FLAGS=-march=armv8-a`,
      `BLAS=scalar`, `BLAS=openblas`
- [ ] E4-14 **flag stamp file in the Makefile** — eight lines; without it `make` then
      `make SIMD=...` without `clean` silently yields a mixed binary
- [ ] E4-15 **`SIMD=auto` must read `/proc/cpuinfo` on x86**, with the kernel-flag +
      `cc_ok` double test and a printed resolved profile. Value is in VNNI and
      AVX-512 BF16; AMX costs far more for less (their AMX 8c does C2, VNNI 32c C12)
- [ ] E4-16 **write `mynah_sgemm_f32` and drop the BLAS dependency** → [`.work/no-blas.md`](.work/no-blas.md)
      Decided: we do not want a second thread pool inside our process. The surface is
      one function (`cblas_sgemm`) at three call sites, and **the whole PocketTTS
      production path is two of them, both in `seanet.c`** — the backbone and flow head
      already go through `qmat`. Skinny shapes, `n` large, `k` small. Removal also
      deletes the weak-symbol clamp in `threads.c`, three dispatch rows, and the
      three-way Makefile split
- [ ] E4-16a **interim, while BLAS is still linked**: `OPENBLAS_THREAD_TIMEOUT=1` —
      TTFA 108 ms **bimodal** to 66 ms stable, 42.5k to 12k context switches/s — and
      report claim vs fact in the dispatch table. Do **not** partition rigidly:
      lowering BLAS threads improves RTF and costs 30% of TTFA. This is compensation
      for a dependency we are removing, not a design
- [ ] E4-17 **plan from `sched_getaffinity`, not `sysconf`** — `sysconf` sees neither an
      inherited taskset nor a cpuset cgroup, so every containerised deployment plans the
      whole host. Also read cgroup v2 `cpu.max` and warn (they never closed that one)
- [ ] E4-18 **arena allocator in the codec before measuring on Linux** — glibc's mmap
      threshold cost them 78 mmap + 154 munmap and 11,899 allocs per request; a
      per-stream bump arena took it to 1.3 and 41, bit-identical. Invisible on macOS
- [x] E4-19 **two production-blocking defects found and fixed** (`3892ba6`) →
      [`.work/dtype-and-fallbacks.md`](.work/dtype-and-fallbacks.md). No Linux build
      linked at HEAD (`mynah_conv1d_sgemm_enabled` trapped inside the Accelerate block
      by our own E1 split); `MYNAH_QUANT=f16` was a silent no-op on x86, so the 2x we
      measured on ARM did not exist on the target. x86 now has F16C/AVX2 + scalar half
      kernels; ARM output byte-identical
- [ ] E4-20 **`--self-test` fails on x86**: `qmat u8 level=1 not bit-identical at row 3
      (k=200)`. Pre-existing, from the VNNI commit — that work was never self-tested on
      x86, which is the whole point of a model-free self-test
- [ ] E4-21 **`seanet.c` has 12 fallbacks and zero dispatch rows.** With `BLAS=scalar`
      the entire codec conv stack drops to the hand-scalar loops — the 8570 ms path,
      **36x slower** — and nothing says so. Also: the depthwise upsample always takes
      `convtr_scatter_scalar` silently; `conv1d.c:487,530` ignore the sgemm return
      (NULL backend gives bias-only output, no error); `seanet.c:53-55` narrows
      `size_t`→`int` six times per call with no guard, where `conv1d.c` has one
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
- [ ] E4-8 `Makefile`: `SIMD=` profiles + `ARCH_STAMP` rebuild-on-flag-change

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

- [ ] E5-1 **remove the global stream mutex** (`server/main.c:451-460`) — scheduler owns `ctx`
- [ ] E5-2 streaming requests enter the same slot driver as offline; no second path
- [x] E5-3 async output writer **done** (`server/stream_out.{c,h}`): a slow client blocked
      the whole process for 66.5 s, now 4.81 s; leaks/ASan/UBSan/TSan clean
- [ ] E5-4 cancel on disconnect (`POLLRDHUP`), slot freed within one frame
- [ ] E5-5 long-form: incremental push into a running decode, persistent conv state across flushes
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
- [ ] E5-10 `tests/test_server.sh` `batching` check needs sub-second timing (flaky, pre-existing)
- [ ] E5-9 **per-language slot groups** — batching cannot mix languages; decide before E5-1
- [ ] E5-11 **admission control against a real-time budget**. Measured: the machine
      sustains ~3x real time aggregate, so at C8 each stream gets 0.35x and stalls.
      There is no setting that makes them all GOOD — only the choice between waiting
      and stuttering — so the server has to choose deliberately and say which
- [ ] E5-13 **prefork preconditions** — costmap mutexes must be *reinitialized* not
      zeroed (an inherited locked mutex can never be unlocked), and no GPU backend state
      may exist before the fork: that is their still-open bug, a wrong answer rather
      than a crash
- [ ] E5-14 **core-major slices, not contiguous logical ones** — Linux numbers the first
      thread of each core first, so a contiguous slice gave two workers the same twelve
      physical cores, one per hyperthread. And print the mask actually set
- [ ] E5-15 **dispatch gate + host profile that refuses to run**, listing the env vars
      that must be *absent*. Their proof it is needed: the right config was versioned
      and three campaigns were still run wrong from memory
- [ ] E5-12 **prefork with pinned core slices** (E5-6) is the mechanism that turns
      cores into streams; without the topology, more cores are not more streams
- [ ] E5-16 **the measurement protocol, before any tuning** → [`.work/serving-design.md`](.work/serving-design.md) §10.
      WAVE (3 synchronised waves) is a *screen*; only a 5-30 min SOAK with a drift gate
      promotes. Their C16 passed the screen at 0.919 and failed the soak at 1.004 with
      596 rejects. Gate on **audio too**: their whole "all-on" ARM profile was faster
      and was rejected at mel-correlation 0.886-0.945 against 0.98
- [ ] E5-17 **send the response header at admission, not at the first chunk** — otherwise
      TTFB equals TTFA and the entire prefill cost is invisible to the client metric
      (theirs: TTFB p50/p95 0.4/0.6 ms vs TTFA 82.6/84.2)
- [ ] E5-18 **`TCP_NODELAY` + `SO_RCVTIMEO` on every accepted socket**; coalesced reads
      6.7% → 0.0%. And **name every thread** for `/proc`: free, and it makes the
      ownership table readable without a debugger
- [ ] E5-19 **keep polling the listener while full, and refuse** — a full server that
      stops accepting hides the wait in the kernel backlog where no deadline can see
      it. Theirs measured TTFB/TTFA p95 4470/4635 ms of which >97% was before `accept()`
- [ ] E5-20 **warm up through the same reset the request path uses** — theirs warmed on
      leftover CLI state and the first real request differed from every one after it
- [ ] E5-21 **decoder lane: a private pinned team on the last N cpus**, own submit lock,
      bounded one-unit-per-slot mailbox, redirection done inside `parallel()` by a
      thread-local tag. Only **after** E5-1/E5-6/E5-12: on a narrow lane it is slower
      than inline (theirs: 6+2 gave 1.364 vs 0.997 at 4+4). Our decoder is 72% of the
      frame and per-slot — the same position theirs was in
- [ ] E5-22 **spin budget and per-CCX bandwidth, measured on our host before choosing W** —
      their spin sweep moved STREAM p95 0.893 → 0.808 with context switches 38k → 7.6k/s,
      and their 16-thread mask read *slower cache-resident than DRAM* because the working
      set straddled two CCX. Both are ten-minute measurements
- [ ] E5-8 gate: **N concurrent streams byte-identical to the same request run alone**

### E6 — Licensing and voice policy → [`.work/licensing-and-voice-policy.md`](.work/licensing-and-voice-policy.md)

- [ ] E6-1 `speakers.json`: `source_dataset`, `license`, `commercial_use` per voice
- [ ] E6-2 default pack ships CC0/CC-BY voices only — `jean` and `cosette` are **non-commercial**
- [ ] E6-3 verify the `embeddings_v3` voice→dataset mapping (assumed CC0, unconfirmed)
- [ ] E6-4 `LICENSES/` + `NOTICE`: CC-BY attribution, modification statement, prohibited-use as notice
- [ ] E6-5 `source.json` pins the official Kyutai repo, never an ungated mirror
- [ ] E6-6 CLI and `/v1/voices` expose the license; `--commercial-only` filter

### Deferred

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
