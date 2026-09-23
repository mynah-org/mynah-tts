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

**Current focus remains CPU — ARM and x86 together.** GPU is still opt-in and
deferred for Magpie; the newly opened E15 is the PocketTTS CUDA/server track
and does not change the CPU default or its qualification gates.

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
- [x] E3-5a **performance: RTF 2.03 → 0.245** (f16 default, and "lossless on a bf16
      checkpoint" is **almost** true -- measured 2026-09-19 over all 109,502,146 weights
      of the English pack: 0.354% change through the f16 round trip and 7,138 flush to
      zero, because f16 goes subnormal below 6.1e-5 and 1.74% of the pack is under that
      line. Worst ABSOLUTE error 2.98e-08 against a largest weight of 4.875, so the
      default stands and the word does not);
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
- [x] E3-12 **already done and gated, verified 2026-09-16** — validity is tracked
      explicitly (`ctx->frames > 0` selects `bos_emb`, "BOS is a tracked fact, not a
      NaN"), a pre-flight refuses a non-finite step input before any context is mutated,
      and `POCKET_INJECT_LATENT_NAN` / `POCKET_INJECT_KV_NAN` force both refusal points
      inside `mynah_engine_pocket_self_check`, which `--pocket-self-check` runs
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
- [x] E7-9 serialize the KV so reload is instant — `mynah_voice_export`.
      **This line said `mynah-tts export-voice` until 2026-09-19 and no such command
      existed**: the serialiser was written and self-tested, the CLI never grew the verb,
      and `grep export-voice` over the whole repo returned nothing. Found by being asked
      to actually clone a voice rather than to trust the board
- [x] E7-11 **the glue, which was the only thing missing** — `mynah_engine_pocket_clone_voice`
      builds a `mynah_voice_clone_config` from a real pack and walks encoder → prefill →
      export; `mynah-tts --clone-voice MODEL_DIR --reference R.wav --output V.safetensors
      --consent "..."`. Nothing in `src/voice_clone.c` changed: it had been oracle-checked
      to 6.4e-06 since 09-12 and unreachable ever since, because no caller assembled the
      configuration. Three of the four sub-configs are the decoder's (the encoder mirrors
      it); only `codec_downsample_stride` had to be parsed. **Verified on both platforms**,
      cloning the pack's own `alba` output (CC-BY-4.0) and speaking a DIFFERENT sentence:
      F0 mean 127.4 against the source's 127.0 and LTAS correlation **0.9861** on
      Neoverse/`BLAS=none`, 127.2 and 0.9839 on M1, against **0.9222 and 86.3 Hz** for a
      different voice as the control. 10.32 s of reference costs 6.73 s on the box. The
      voice files are NOT bit-identical across platforms (6 of 12 tensors match; the KV
      caches differ by up to 6.4% of max, f16 storage and different BLAS), which the
      project's numerical policy allows and which is recorded rather than glossed
- [ ] E7-10 consent gate and notice before cloning — see E6

### E4 — CPU kernels, ARM and x86 in one step → [`.work/cpu-kernels-arm-x86.md`](.work/cpu-kernels-arm-x86.md)

**No kernel is done until both ISAs exist and self-test. GPU deferred.**

- [x] E4-1 **the README understated x86, in the opposite direction** — `src/qmat.c` has
      both an EVEX and a VEX VNNI kernel behind target attributes now; the text called the
      lane unbuilt. 0.427 stays a floor because it predates them
- [x] E4-2 `--dispatch-map` + costmap **done**; 8 rows resolve UNKNOWN and each names the
      predicate to add — that is E4's real to-do list
- [x] E4-2a **done** `6ad9ada` — 0 UNKNOWN rows in every configuration · add the 8 named predicates so no row resolves UNKNOWN
- [ ] E4-2b place the costmap hooks (deferred: the engines were being refactored)
- [ ] E4-3 baseline per-region profile of the PocketTTS path, on M1 and on EPYC
- [ ] E4-4 thread pool upgrade: lane split, deadline priority, `after_fork` (prereq for E5-6)
- [ ] E4-5 the kernel the profile names — scalar reference, then NEON/SDOT/i8mm **and** AVX2/AVX-512/VNNI in one change
- [ ] E4-6 int8 weight prepack with persistent cache, both ISAs
- [~] E4-8 **BUILT AND MEASURED; not a default, and the prefill half is not done** →
      [`.work/bf16-native-weights.md`](.work/bf16-native-weights.md). `--dispatch-map` on
      the box has been printing this for days: five units the CPU has and this binary has
      no kernel for, with `isa.arm.bf16` reading *"no bfdot/bfmmla and no bf16 weight type;
      src/weights.c widens bf16 to f32 at load and f32 paths stay f32"*. Today a bf16
      checkpoint goes **bf16 -> f32 (malloc'd copy) -> f16 (packed copy) -> `vcvt_f32_f16`
      + `vfmaq_f32`**, three representations to return to the width it started at, and
      **4 MACs per 128-bit instruction**. `BFMMLA` is **16**, accumulating in f32 -- which
      is why the `FMLAL` rejection does not cover it: that one narrows the accumulator,
      this one does not. **Four of the five idle units are not worth writing and one
      measurement says so**: `svcntb()` on Neoverse-V2 reports a **128-bit** SVE vector,
      the same width as NEON, so sve/sve2/svei8mm would buy predication and VLA
      portability, not throughput. Where it pays is the PREFILL, not the step: the step is
      memory-bound at 1-6 rows (E9-11 measured it and exploits it), while the prefill at
      16 rows is ~8.4 MACs/byte and compute-bound by roughly 6x. Size of the prize, with
      E9-12's discipline applied: that item refused int8-in-the-prefill because
      `prep.prefill_proj` is 4.9% of `request.total`, but what binds the product now is
      TTFA under load, set by a LONG text's **190 ms** prefill of which the projections are
      order 110 ms -- so 2-4x there is **55-80 ms** against a TTFA p95 of 492 ms with a
      500 ms gate. It also pays twice, since less prefill work per request is less prefill
      work per step, which is what `MYNAH_PREFILL_STEP_MS` rations and what stops C98.
      Free on the side: the f32 intermediate and the pack pass both disappear (E9-1 measured
      that conversion at 302 MB -> 151 MB for the backbone). **To close**: a bf16 weight
      type in `src/weights.c`, a scalar reference, the NEON kernel behind a runtime probe
      like i8mm's, the AVX512-BF16 counterpart, the inventory row, and the gate --
      `--self-test` at widths 1-17 against the scalar reference, then
      `prep.prefill_proj` re-taken **on the Axion** (E9-11's 1.62-1.71x is a macOS number
      never re-taken, and this item must not repeat that), then a 30-minute C98 soak,
      which is the level that should move if the prefill gets cheaper. **Not KleidiAI**:
      that stays rejected on its own grounds; what transfers from qwen-tts is the idea,
      not the dependency.
      **WHAT LANDED (same day)**: the `bf16` encoding, a scalar reference, a NEON BFDOT
      matvec, a four-activation batched kernel, self tests that pass on the production CPU
      both with the unit and with `MYNAH_QMAT_BF16=0` forcing the scalar path, and the
      `isa.arm.bf16` row moved from `src/kernels.c` to `src/qmat.c`, because the file that
      owns a kernel owns its row. **THREE PREDICTIONS CORRECTED BY THE BOX**: (1) BFDOT
      **3.45x** over the shipping f16 kernel at the prefill's shape and **BFMMLA only
      2.41x** -- the instruction with twice the MACs came last, almost certainly because
      that variant carries one accumulator and is latency-bound, so it is a verdict on two
      implementations rather than two instructions; (2) bf16 was predicted NOT to help the
      memory-bound AR step and it helps most -- RTF **0.160 -> 0.119** -- because the f16
      path also pays `vcvt_f32_f16` per weight block and BFDOT skips it; (3) the batched
      kernel did NOT fix the prefill: `prep.prefill_proj` is 6.1 ms f16, 9.0 bf16, 15.9
      bf16-scalar, so BFDOT beats its own scalar path 1.76x and still loses, because **the
      activation is narrowed inside the row loop** and gets converted once per weight row
      while f16 converts weights and amortises over four activations. The microbenchmark
      handed both kernels pre-converted operands and could not see it. **To close**: narrow
      the activation ONCE per call into a `batch * k * 2` scratch, exactly as `quantize_act`
      already does for int8 -- a signature change at the batched entry point, not a kernel
      change. **AND THE GATE THAT DECIDES SHIPPING**: `backbone:bf16` **changes the audio**
      -- same length, same EOS step, but SNR 1.7 dB and correlation 0.667 against f16,
      because an autoregressive model feeds the 2^-8 activation rounding back into the
      sampled latent. That is the same category as `backbone:int8`, already refused on
      quality, except bf16 keeps the frame count. Available, NOT default. The promising
      experiment needs no new code: the CODEC groups are not autoregressive, so
      `MYNAH_QUANT_GROUPS=codec_*:bf16` is additive noise measurable against f32 exactly as
      int8's was -- and int8, far coarser, was accepted there.
      **THE BFMMLA VERDICT WAS WRONG AND THE CORRECTION IS THE BIGGEST NUMBER HERE.**
      "BFMMLA loses to BFDOT" judged a kernel with ONE accumulator. Tiled -- four
      row-pairs x two activation-pairs, eight independent chains -- the same instruction
      measures **173.6 GFLOP/s, 14.68x** the shipping f16 kernel (f16 11.8, BFDOT 41.1),
      at ~90% of the core's peak, and is **bit-identical** to the one-accumulator version,
      so the tiling is scheduling and not a shortcut. It also fixes the prefill by
      construction: the activation operand is built inline and SHARED by every row-pair,
      so narrowing costs once per eight rows instead of once per row -- no scratch, no
      signature change. In the engine: `prep.prefill_proj` **6.186 ms f16 -> 9.308 bf16
      BFDOT -> 4.500 bf16 tiled**, RTF 0.162 -> 0.124 -> 0.121. **OFF BY DEFAULT**, and for
      a contract rather than a doubt: wired in unconditionally `self_test_lane_widths`
      fails at 2.6e-7 -- "a lane width changed a row's answer" -- because BFMMLA
      accumulates k in fours and the one-activation path in two chains of eight. Both are
      right; this runtime promises batching cannot change a row's result, and a 14x kernel
      does not get to be the exception. **To close**: ONE SHAPE EVERYWHERE -- the
      single-activation path reaches the same kernel with the activation repeated, which
      needs the packed pointer in `qmat_rows_job`. **DONE, and the trade did not exist**:
      every bf16 multiply now takes the tile -- a batch of four, a short group with its last
      activation repeated, a row-pair remainder inside the same call, and a SINGLE
      activation repeated into all four lanes. Three quarters of the arithmetic is
      discarded at batch one and costs nothing, because the kernel is memory-bound there.
      `self_test_lane_widths` PASSES with the kernel on, two consecutive runs give the same
      sha256, and `prep.prefill_proj` is **8.571 ms f16 -> 6.338 bf16** with RTF 0.151 ->
      0.121. `MYNAH_QMAT_BF16_TILE` is gone. **Still open, and it is a product question not
      an engineering one**: against an f32 backbone, f16 holds SNR 33.6 dB / corr 0.9998
      while bf16 sits at 3.0-5.4 dB / 0.76-0.85, so bf16 DEVIATES from what the model would
      have said. Decided by listening
- [ ] E4-7 AMX-INT8 (Linux/x86 only), last
- [ ] E4-10 **remove useless dtype conversions** — called out by name as one of the two
      profiling wins. `src/qmat.c` holds 15 conversion sites and every other hot-path
      module holds zero; the suspicion is `bf16 -> f32 -> f16` where one step would do.
      **The suspicion is confirmed** (2026-09-19, E4-8): `src/weights.c:122` converts the
      mapped bf16 to a malloc'd f32 copy, `src/qmat.c:354` packs that to f16, and the
      kernel then widens back to f32 to multiply. E4-8 is the version of this item that
      also buys arithmetic, and subsumes it
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
- [~] E4-9 **runtime ISA dispatch on x86: the f32 half landed 2026-09-22** →
      [`.work/mynah-isa-backend-reachability-audit-20260921.md`](.work/mynah-isa-backend-reachability-audit-20260921.md) §6.
      `src/kernels.c`'s seven hot f32 kernels are now `target("avx2,fma")` + a scalar
      form chosen by `mynah_kernels_x86_avx2()`, so **one x86 binary is safe on a
      pre-Haswell CPU and full speed on a new one** — it was not, and `src/qmat.c`
      dispatching while `kernels.c` did not was the portability gap. `axpy` had no
      x86 vector path at all and now does. Proven by executing an x86_64 build on
      this Mac under Rosetta (`make x86-cross`), which is a CPU with no AVX2 — the
      old-x86 tier we were about to rent. Still open here: **`src/sgemm.c`**, where
      `SG_LANES` leaks into the packed-panel layout so target attributes on the
      micro-kernel are not enough; OpenBLAS thread coordination; a Linux measurement
      box for the numbers.
- [ ] E4-9b **borrow the kernels next door before writing them** →
      [`.work/qwen-tts-kernel-reuse.md`](.work/qwen-tts-kernel-reuse.md). `../qwen-tts`
      has a **VDPBF16PS bf16 matvec** we do not, on the dtype our backbone ships, and
      its weights are already in the plain row-major bf16 layout our cache holds — an
      inner-loop port, not a layout change. Then the **AVX-512BW int8 GEMV without
      VNNI**, which is the Skylake-SP tier nothing in this tree reaches. Not AMX (E12
      says the arithmetic is 5% at B=8), not KleidiAI (a dependency decision, not a
      port). Mynah is the one that is ahead on dispatch: qwen has one target attribute
      in 14,372 lines.
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
- [x] E8-5 **shipped** — `mynah_qmat_linear_batched_qt` exists and carries the
      rationale in `src/qmat.h`: `qtype` decides both the gate and the cache entry, so a
      group's precision comes from the group spec and never from whichever caller
      arrived first. Gated by `self_test_lane_widths` over every cache profile crossed
      with every encoding a group spec can name. The board entry was stale

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
- [x] E9-5 **closed by E10-4d** — the row was not the defect; the default spec's bare
      clauses were inheriting f16 from a base that had changed underneath them. Original
      text follows
- [-] E9-5-orig: a
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
- [-] E9-9 **superseded by E10-8** — the item asked for the x86 knee to be measured and
      the constant set from it. E10-8 answered the question one level up: an iteration
      count cannot be right on both ISAs (`yield` and `pause` differ by two orders of
      magnitude), so the budget is a **time** now, calibrated on the host at startup.
      There is no per-target constant left to measure. The board entry was stale
- [x] E9-6 **one reader now, and empty means unset** — there were four readers and the
      fourth was mine (E10-4d chose its spec with a raw `getenv`). Verified: unset, empty
      and a typo give byte-identical audio, and the typo warns. Original text follows
- [-] E9-6-orig — the empty value
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
- [-] E9-12 **not worth it, and the premise needed correcting** — `MYNAH_CENSUS=1`
      says the prefill is **not** on an unquantized path: every backbone projection in it
      runs `batched-q / f16 / neon` at 16 rows, so f16 does reach it and there is no f32
      fallback to recover. What is left of the item is int8 *instead of* f16 there, and
      the arithmetic decides it: `prep.prefill_proj` is **35.7 ms, 4.9%** of
      `request.total` on this pack, so even a 1.5× on it is ~1.6% of a request — against
      a **second quantized copy of the backbone** (~75 MB, since the steps must stay
      f16) and a new quality risk that is not smaller than the AR loop's, only
      differently shaped (a static error written into the KV every step then attends
      to). Refused on the numbers. Original text follows
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
- [x] E10-3b **answered on the box: the sharing is real** — `/proc/PID/smaps_rollup`
      on four prefork workers gives **RSS 567 MB, PSS 124 MB, Private_Dirty 13 MB** each.
      PSS is what a process actually owns, so the weights are shared after the fork and a
      worker dirties 13 MB of its own: four workers cost about 300 + 3x124 MB rather than
      4x567. E10-3's pre-fork warm does what it claimed, and macOS could not have told us
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
- [~] E10-4b **the hole is closed; the number is the box's** — every weight-stationary
      path (SMMLA, SDOT, the f16 lanes) gates on the SIGNED activation encoding, because
      the unsigned x+128 form and its row-sum correction are x86's — so on the half of
      production that runs VPDPBUSD the batch fell through to one weight pass per
      activation, which is the exact defect the SDOT path was written to fix, left
      standing on the other architecture. **No new kernel and no new instruction**: the
      loop order is swapped, row block outer and batch inner, so four weight rows are
      loaded once and stay in L1 for the whole batch. `dot4_u8_i32` is the same function
      the per-activation path calls on the same bytes, so the int32 is identical by
      construction. **Correctness is gated and the speed is not claimed**: two mutations
      of the new block are caught, and *only* under `MYNAH_QMAT_VNNI=scalar` — the level
      that forces this encoding on ARM, which E10-5 also found had no gate using it.
      A/B'd here under that level: **1.050 / 1.022 / 1.004 — noise**, which is the
      expected null result rather than a refutation (the scalar unsigned kernel is
      arithmetic-bound, so reordering traffic cannot move it — the same reason a
      key-stationary variant lost in `src/transformer_ar.c`). `MYNAH_QMAT_U8_BATCH=0`
      restores the old loop so the box settles it in one command
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
- [~] E10-5a **the ELU was the largest phase, and it was a scalar `expf` loop** — 33.6%
      of the region, larger than both GEMMs the item named. Vectorised with a
      bounded-range exp we own (max absolute error **5.96e-08** on [-88,0], gated and
      mutation-tested): **elu 53.4 → 18.1 ms (2.96×)**. The `codec.conv_stack` figure first reported as
      1.28× was an **Accelerate** number; re-measured in the shipping `BLAS=none`
      configuration it is **1.08×** (335.3 → 309.2 ms), because the ELU is 5.6% of that
      region rather than 11.6% — see [`.work/measuring-the-codec.md`](.work/measuring-the-codec.md). Audio: 51-59 samples of ~120,000 differ, every one by exactly **1 LSB of
      int16**, identical length. AVX2 twin still to write.
- [x] E10-5 **the conv1d half is int8 now; the transposed half is still the open
      question** — [`.work/seanet-int8-conv.md`](.work/seanet-int8-conv.md).
      `codec_conv` meant one matvec (`mimi.quantizer.output_proj [512][32]`), so a spec
      that said `codec_conv:int8` was quantizing ~0.03% of what its name covers; it now
      means the decoder's causal convolutions. New `src/convq8.{c,h}`, int8 arithmetic
      **exported from qmat** rather than rewritten (the activation encoding is a property
      of the host, and a second copy would have dropped VNNI on the x86 half of
      production without saying so). **`codec.conv_stack` 204.2 → 170.0 ms (1.20×, 5/5
      paired rounds), whole request 1.06×**; the conv GEMM family itself 86.9 → 48.8 ms.
      Gated by shape, `k*taps >= 128 && m >= 64`, each half a measured sign change.
      **The quality finding decided the gate**: admitting the 32-channel stage costs
      **8× on log-mel** (0.9976 → 0.9785) to save 6 ms of 44, and the waveform
      correlation would not have shown it — the shipped `codec_transformer:int8` moves
      *waveform* correlation to 0.99852 while leaving log-mel at 0.99953, and this does
      the opposite. One number could not have caught it.
      Sample count identical in 9/9 text×seed pairs. Four things found by measuring:
      the activation quantizer was scalar and bigger than the SDOT it feeds (vectorised,
      byte-identical, the decode loop gets it too); tiling the transpose is *slower*;
      the memo keyed on a pointer malloc recycles; my own error path had a
      use-after-free. Nine mutations, nine caught — **two only under
      `MYNAH_QMAT_VNNI=scalar`**, the level that makes the x86 unsigned algebra
      executable on ARM and that had no gate using it. Original text follows
- [-] E10-5-orig **the SEANet conv stack is entirely f32** — `codec_conv` in our spec is only
      `mimi.quantizer.output_proj [512][32]`, so **25.6% of the wall has no quantized
      kernel at all**, and the depthwise upsample is *permanently scalar*
      (`groups == out_channels == 512` can never qualify). The reference measured **−18%
      of whole-request wall** from its int8 decoder conv on Neoverse-N1. **Skip the three
      convtranspose stages** — it measured those slower than f32 sgemm and ships them off.
      Needs its own quality gate (frame count, EOS step, log-mel corr)
- [x] E10-6 **int4 scale seeded from the signed extreme, then solved** — relative weight
      reconstruction error **4.182% → 3.824%** (−8.6%), same bytes, same kernels, no
      runtime cost. **No audio change, and the reason is ours:** the default spec keeps
      int4 out of the AR loop, so it perturbs only the feed-forward codec — six seeds give
      3-3 on waveform correlation and identical durations to the byte. The reference's
      83.9→90.9% word accuracy came from int4 reaching their sampler; not transferable,
      not claimed. Gated by absolute bounds after an ordering-only gate passed **two**
      mutations. `MYNAH_QMAT_Q4_NAIVE=1` restores the old scale. Original text follows — the reference solves the block scale in
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
- [x] E10-8 **the budget is a time now, derived on the host** — `yield` and `pause`
      differ by two orders of magnitude, so one iteration count could never be right on
      both, and the gate asked the OS a question about an instruction. Target 35 µs (a
      cold wake costs 22-29), calibrated at startup; lands on **65536** here, which is the
      Axion's measured knee. Three measurement traps found and fixed in the calibrator
      itself: a `volatile` counter reported 2.25 ns/relax against ~0.5 real, the mean
      folds in machine noise so it takes the **minimum of five**, and the raw quotient
      moved 43750/70000/100000 across starts so it **snaps to a power of two**.
      `--dispatch-map` reports source, measured ns and target. Original text follows
- [-] E10-8-orig — `#if defined(__linux__)
      && defined(__aarch64__)` → 65536, everything else 4096, and `src/threads.c:112-117`
      already admits the value is *"transferred from the reference's measurement, not
      measured by us"*. Two consequences: macOS arm64 takes 4096 despite being the same
      ISA with the same `yield`; and **Linux x86-64 production runs 4096 `pause`es, never
      measured**. It is an iteration count, not a time budget, and `yield` vs `pause`
      differ 50-100× in duration — **one integer cannot be right on both**
- [~] E10-9 **the soak has been run, and it says the screen was not capacity** →
      [`.work/axion-c99-soak.md`](.work/axion-c99-soak.md). On the Axion, `16x2`, the
      codec fully int8: **C99 on English `medium` for 10 minutes is GOOD** — 25014/25014,
      RTF p95 **0.779**, TTFA p95 178 ms, prebuffer **0/0 ms**, stall 0%, and drift
      **exactly +0.0000** over ten windows. Then **C99 on the MIXED bank for 30 minutes
      is NOT STREAMABLE** — 54645/54645 completed, drift still passing, but
      `stall_rate@500ms = 0.1%` against a mandatory zero. The mix is the difference:
      audio 1.04-18.48 s, **sd 4.39** against the medium-only 0.298. The reference's
      wave-vs-soak gap does **not** reproduce as drift here; what bites is the duration
      spread. **A screen is never a qualification**, and we now have our own instance of
      it, the two verdicts an hour apart. **The descending sweep has since been run and
      the answer is that concurrency does not close it**: C98 and C96 over thirty minutes
      each are also NOT STREAMABLE on the same gate, with every other gate and both drift
      gates passing. Three levels bought a factor of ten on the failing quantity --
      `stall@500ms` 0.1% -> <0.05% -> **5 of 53895** -- and did not reach zero. What is
      measured instead is the client-side number: over 53895 requests at C96 no request
      needed more than **535 ms** of lead, so a **600 ms prebuffer** plays all of them.
      A 10-minute screen had put that bound at 495.4 ms and the 30-minute run moved it
      past 500: a screen cannot qualify a tail either. **To close**: E10-10, because the
      tail is the long-request slot and the topology is the untested variable
- [x] E10-16 **CLOSED — the stall had one cause, and the fix qualified C90 as GOOD, measured and read in the code: the prefill
      runs inside the step loop** → [`.work/prefill-blocks-decode.md`](.work/prefill-blocks-decode.md).
      `slot_start()` calls `engine->prepare()` synchronously in the admission block at the
      top of `mynah_graph_serve_continuous`, so **every resident slot freezes for the new
      request's prefill**. At C1 with no contention a long text's prefill costs **190 ms**
      against 28 ms of fixed cost -- 2.4 frame periods -- and under load `max_gap` is
      **97 ms at p50 and 358 ms at p95** with a frame lasting 80 ms. It is an interruption,
      not a slowdown. It explains why the `medium`-only bank was GOOD (every prefill ~53 ms,
      under one frame) and why three concurrency levels could not clear it: the freeze
      duration is a property of the TEXT, not of the load, so lowering C removes freezes at
      3% a level while the cushion does all the work. Three ways out, priced:
      **DONE, and it cleared its gate.** `prepare_slice` is an optional appended hook;
      the driver alternates one slice with one step and carries a third slot state for a
      prefill in flight. Bit-identical across `MYNAH_PREFILL_SLICE` 0/16/32/48/128 on two
      architectures, single and batched. **C94, slice 48, thirty minutes: MARGINAL with
      every mandatory gate passing** -- 53559/53559, `stall@500ms` **0**, RTF p95 0.761,
      TTFA p95 443, `max_gap` p95 158 (was 358), throughput -0.7%. Against the same soak
      before the fix: `stall@500` 5 -> 0, `stall@250` 233 -> 81. **The operating point is
      C94.** **Concurrency is EXCLUDED as the lever**, by four levels that refuse to line
      up: `stall@250ms` is 0.134% at C96, 0.151% at C94, **0.214% at C90** and 0.109% at
      C80 -- C90 is the worst of the four while carrying less load than two of them, over a
      17% span of load with no monotone trend. `max_gap` p95 across the same four is
      **158 / 158 / 157 / 148 ms**, a near-constant: lowering C removes OCCASIONS and
      leaves DURATION untouched, so the count drifts down noisily and never reaches zero.
      A gate that demands zero is a statement about the WORST CASE, and a worst case needs
      a bound. **Built, measuring**: `MYNAH_PREFILL_STEP_MS` (default 40 ms, half a frame
      period) caps prefill work per STEP. The per-slot token budget bounded one slice and
      never one step -- the pass walks every preparing slot, so prefills landing together
      froze a worker for their SUM, which is the 441 ms `max_gap` maximum against a single
      slice of ~60 ms. The pass now stops when the step's budget is spent and resumes next
      step where it stopped, with a rotating cursor so no slot starves, and always runs one
      slice so the cap cannot deadlock. The reversal it predicted was then measured: with
      the cap, 16-token slices reach `stall@250ms` **0** and a worst freeze of 122 ms, and
      the binding gate becomes TTFA instead. **QUALIFIED: C90, slice 32, cap 60 ms, thirty
      minutes — GOOD, every gate passed.** 53265/53265, `stall@500ms` **0**, `stall@250ms`
      **0**, RTF p95 0.734, TTFA p95 494.8, prebuffer p95 10.2 and max 267, `max_gap` max
      **177 ms** (was 707), throughput 128.9 audio-s/s, drift +0.0040 over ten windows.
      **The client contract is now a 250 ms prebuffer, and it is a bound rather than a
      sample maximum.** What binds next is NOT the machine: RTF p95 0.734 leaves headroom
      and the gate that stops a higher C is TTFA at 494.8 against 500, so the next lever is
      the ABSOLUTE cost of the prefill -- still 190 ms for a long text, never optimised
      - **(1) CHOSEN — chunked prefill.** Make `prepare` resumable and interleave the slices
        with decode steps, so the longest freeze is one slice instead of one prefill. It
        removes the cause. Costs an engine-seam API change and TTFA on the admitted request:
        coarse slicing (two or three) puts the freeze **under one frame period** for ~110 ms
        of TTFA. Gate: `stall_rate@500ms == 0` on a 30-minute mixed soak at C96 with
        `RTF p95 <= 0.80` and `TTFA p95 <= 400 ms` -- the fix may not buy continuity with
        throughput or with first audio, which is the whole point of doing it this way
      - **(2) rejected, recorded so it is not re-proposed — lead-aware admission.** Defer
        while the least-advanced resident slot has less cushion than the incoming prefill
        needs. No API change, but the arithmetic refuses: a slot gains only
        `(1 - RTF) x 80 ms` = 25 ms of lead per frame, so ~0.5 s of wall buys 250 ms of
        cushion, and with an admission every ~0.5 s per worker there is nearly always a
        vulnerable slot. A guard big enough to protect defers almost every admission; one
        small enough not to protects almost nothing
      - **(3) the interim contract, true today at zero cost — a 600 ms client prebuffer.**
        No request in 53895 needed more than **535 ms** of lead at C96 over thirty minutes.
        It ships until (1) lands, and it is a statement about a 30-minute sample of a tail
- [x] E10-18 **the operating point is C96, and the cap that got it there was derived rather than tried** —
      [`.work/prefill-blocks-decode.md`](.work/prefill-blocks-decode.md). Timing the serving
      loop per batch width gives `T_frame(B) = 3.0 + 7.8*B` ms, so a typical step at B6 costs
      50 ms of an 80 ms frame and the SLACK a prefill may use is **30**. `MYNAH_PREFILL_STEP_MS`
      now defaults to that instead of the 60 that was chosen by trying values. Four soaks at
      C94 move `max_gap` max 179 -> 148.6 -> 142.9 -> 133.9 for 60/40/30/20 with **TTFA flat**,
      refuting the tension predicted when the cap was proposed: the per-slice budget already
      bounds one slot at 32 tokens, so the cap binds only when several prefills coincide --
      it bounds the tail and leaves the median path alone. **QUALIFIED: C96, thirty minutes,
      shipped default, nothing exported** -- 53886/53886, `stall@250ms` **0**, `stall@500ms`
      **0**, TTFA p95 492.5, RTF p95 0.759, throughput 130.2 audio-s/s, worst client prebuffer
      **228.9 ms against a declared 250 ms contract** (316 at the old cap), `max_gap` max 142.8.
      Two things this also settled: **`--max-batch` is not a lever** here (the loop never
      reaches 7 slots at this concurrency, and its two arms were the same experiment twice),
      and **the instrument's resolution**: TTFA p95 is 492 +/- 5 against a 500 ms gate, so a
      ten-minute verdict at C94-C96 is a coin toss -- those same two arms returned MARGINAL
      and GOOD, separated by 136 microseconds. **Next**: C98/C100 are stale, measured at the
      old cap
- [ ] E10-21 **START HERE** → [`.work/next-2026-09-20.md`](.work/next-2026-09-20.md).
      Seventeen commits sit on `main` unpushed and CI has seen none of them, including an
      x86 bf16 kernel no machine here can execute -- **push first**. Then the two runs the
      box was stopped in the middle of: a C125 screen and a thirty-minute C120, ~45 minutes,
      which close the only open capacity question. After that the lever is no longer the
      prefill at all: at C130 TTFB triples because 16 workers x 8 slots is 128 places, so it
      is `--max-batch`, the worker count, or admission
- [ ] E10-19 **the prefill cap is stale and the prefill ORDER has never been questioned** —
      [`.work/prefill-blocks-decode.md`](.work/prefill-blocks-decode.md). Two levers on the
      one gate that still fails, and the first needs no code. **(a) Re-derive the cap.**
      `MYNAH_PREFILL_STEP_MS = 30` came from `slack = 80 - T_frame(B6) = 80 - 50` with the
      **f16** kernel; bf16 made the step cheaper, so the slack is now larger and the cap is
      rationing work the frame could absorb. `MYNAH_SERVE_PROFILE=1` re-measures
      `T_frame(B)` and the sweep is four ten-minute runs. **(b) FIFO instead of round-robin.**
      `slots_prefill_slice()` is processor sharing -- the policy that maximises jobs in
      flight, so every prefill finishes near the time the LAST one would. FIFO-to-completion
      wins the mean by construction and the tail through Little's law (lower mean -> fewer
      resident prefills -> less competition). The per-step worst case is unchanged because
      the cap already bounds it. **To measure, not assume**: head-of-line blocking, a long
      text delaying a short one. The mixed bank is the instrument and per-class TTFA is
      already reported. **Where this is going**: at C120 only TTFA fails, by **14 ms**, with
      stalls at zero. **THE OPERATING POINT IS NOW C110** -- thirty minutes, 63120/63120,
      `stall@250ms` and `stall@500ms` both **0**, TTFA p95 482.3, RTF p95 0.726, required
      prebuffer **0.000 ms**, throughput 152.7 audio-s/s, drift +0.0018 over ten windows,
      on the shipped default with nothing exported. Ten window percentiles run 472-493, so
      the margin is stable rather than lucky.
      **(a) IS DONE AND IT WORKED**: at C120, cap 30 -> 40 takes TTFA p95 **510 -> 450 ms**
      and turns MARGINAL into GOOD; 50 gives 448, so the curve flattens at 40. It cost
      `max_gap` p95 103 -> 121 ms, still far inside the 250 ms contract. **And it corrected
      a claim made here this morning**: "cap = deadline - T_frame" is NOT a law. `T_frame(B)`
      with bf16 is **2.9 + 6.8*B** (f16 was 3.0 + 7.8*B), so at the modal width B7 the slack
      is 28 ms and the rule would say LOWER it. A frame that overruns is a debt, not a
      stall: at RTF 0.798 a slot earns 16 ms of lead per frame and repays ten in one. The
      slack is a first guess; the cap is a measured quantity and must be re-measured
      whenever a kernel changes the step cost. **Also expired**: `--max-batch` is a live
      knob again -- the width histogram at C120 reaches **B7 (92015 frames) and B8 (62154)**,
      where at C100 the loop never passed 6.
      **QUALIFIED: the default cap is now 40 and the operating point is C120** -- thirty
      minutes, 64205/64205, `stall@250ms` and `stall@500ms` both **0**, TTFA p95 447.4,
      RTF p95 0.794, required prebuffer 2.8 ms, throughput 155.3 audio-s/s, drift +0.0022.
      **C96 -> C120 in one day, +25% on the same hardware.** What is left of this item is
      **(b) MEASURED AND SHIPPED**: FIFO takes TTFA p95 **445 -> 318 ms at C120** (-29%) and
      **521 -> 407 at C130**, where it stops being the failing gate entirely. The check that
      came first: an aggregate p95 cannot tell "everyone starts sooner" from "long texts
      finish by making short ones wait", and `tools/serving_profile.py` reported only the
      class MIX -- so per-class TTFA was ADDED to the instrument before the decision, not
      after. It says the suspicion was wrong: **short -30%, medium -25%, conversational
      -25%, long -22%** -- every class improves and the two carrying 70% of the traffic
      improve most in proportion. Required prebuffer moves 2 -> 28 ms, which is not
      congestion but prefills finishing in groups (RTF p95 0.796 -> 0.807), against a 250 ms
      contract with stalls at zero. `MYNAH_PREFILL_ORDER=rr` restores the old policy.
      **Worth noting where the win came from**: the tiled BFMMLA kernel is 14.68x on the
      isolated GEMM and bought fourteen points of concurrency; FIFO touches no multiply at
      all and took more off TTFA than the cap did
- [ ] E10-20 **above C120 the admission queue becomes the limit, and `--max-batch` wakes up** —
      at C130 TTFB jumps **74.8 -> 200.7 ms**, which is not synthesis: 16 workers x 8 slots
      is 128 places and 130 requests is the first level that fills them. This morning's
      finding that `--max-batch` is inert (the loop never reached 7 live slots) expires
      exactly there. Fail-fast is already correct and verified in the code -- `job_enqueue`
      refuses at `pending >= max_pending` under the mutex and answers **503** with
      `rejected++` before any synthesis, so a rejection costs an accept and a write
      (`server/main.c:376,1208`), `--max-pending` defaults to 256, and `/health` publishes
      `queued`/`rejected`/`queue_capacity` for an autoscaler. ~~**Gap**: no `Retry-After`
      header on the 503~~ — **closed 2026-09-22**: `send_status()` emits `Retry-After: 1`
      for any 503, at the one place every refusal passes through rather than at the four
      call sites, so a new refusal cannot be added without it. Verified against a live
      server driven past `--max-pending`. Clients must still jitter: the header
      synchronises the retry, it does not spread it
- [x] E10-17 **the configuration of a qualifying run no longer lives in shell history** —
      [`.work/serving-profiles.md`](.work/serving-profiles.md). `configs/perf/*.json` carries
      the deployment shape, the gates AND the operating point measured on that hardware;
      `tools/serving_profile.py --profile <id>` applies all of it and **refuses** when the
      world disagrees -- a variable the profile declares absent that is present in the
      environment, or a flag also given on the command line (a conflict, not an override).
      Built because of a specific six-day failure: every capacity number between 09-13 and
      09-19 was measured with `MYNAH_QUANT_GROUPS` exported by hand while the shipped binary
      chose something 1.86x slower, both facts written down, and nothing in the harness could
      see the difference. The validator encodes the mistakes actually made -- `qualified`
      needs a 30-minute soak (a screen may not promote), GOOD with zero stalls,
      `history` ending at the operating point, no GOOD level parked in `ceiling`. Every point
      must also declare `shipped-default` or `environment-override`, because that difference
      is invisible in every other field. `tools/perf_profile.py detect` answers "what is this
      box and where do I start" in one command, with architecture as a wall and an explicit
      refusal to inherit a topology. Gate: `make perf-profile-test`, 29 checks
- [ ] E10-10 **the topology is answered, and the answer is that it belongs to the MODEL** —
      `8x4` with the same 32 threads and the same 128 slots is decisively WORSE at C96:
      RTF p95 **1.052** against 0.784 (a mandatory gate), throughput 93.6 against 124.8
      audio-s/s, `stall@250ms` **734/13511 against 84/17968**, prebuffer p95 235 ms against
      0. qwen-tts prefers `4x8` on this same box and that is not a contradiction:
      `T_frame(B) = a + b*B`, and a wider worker wins only when doubling the threads more
      than halves `b`. For a 1.7B model `a` (the weight stream) dominates; for PocketTTS's
      109.5M it does not, and the codec's `b` scales sublinearly (16.7 ms at 2 threads,
      7.7 ms at 16). **The optimal shape is a property of the model's a/b ratio, not of the
      machine** -- do not inherit it across engines. Still open below: the original claim
- [ ] E10-10b **the low-concurrency half of the topology claim** — `16x2` beats `1x32` by 2.2x on RTF p95, measured only on the Axion
      and only at low concurrency. Every capacity number in
      [`.work/axion-c99-soak.md`](.work/axion-c99-soak.md) is `16x2`, so it is assumed,
      not tested. **The prediction now points the other way at high load**: `step.backbone`
      is bandwidth-bound (E10-15), so sixteen workers are sixteen independent weight
      streams against one memory roof — qwen-tts measured exactly this and concluded
      *"two workers read the same weights twice per frame-time"*, i.e. prefork forecloses
      the only large lever, which is amortising one weight pass over B slots. At C99 a
      WIDER worker (`8x4`, `4x8`) should win. ~20 minutes on the box: the same soak
      command with three shapes. Also still open: the reference's Graviton5 `4x8`
      collapse (4.45x per worker) is a warning our 16x2 is four times more aggressive than
- [x] E10-11 **the loop can say how many slots were live, and why the rest were not**
      (`f3f7ed3`). `MYNAH_SERVE_PROFILE=1`: live-slot histogram as a share of frames,
      mean live width with 1.00 labelled "never batched", free-slot-found-nothing-queued
      count, and the share of wall blocked waiting for an arrival — labelled, because
      high there means **idle, not saturated**, which is what decides whether a level
      failed on capacity or on variance. Verified against known shapes; the server at
      five concurrent showed **B1=55.7% / B4=44.3%**, which is the kind of thing it
      exists to surface. Original text follows
- [-] E10-11-orig — we can say where wall time went but not how
      many slots were live per frame, nor *why* a free slot stayed free, nor the
      share-of-wall that was "blocked: no work queued" (the row that decides whether C80's
      1.120 is saturation or variance). The reference has all three plus a per-request
      stage decomposition from admission to first audio. Our 585 ms burst model is
      arithmetic that lands within 4% of measurement — good, but a model
- [~] E10-12 **three of four done** — [`.work/kv-window-allocation.md`](.work/kv-window-allocation.md).
      The `getenv` per conv call and the never-written `rows->gelu` were already closed.
      The **277 MB of KV** was measured and the item had it wrong in both directions: the
      buffers are `calloc`, so RSS grows only **754 → 834 MB from a 1-word utterance to a
      96-word one**, +80 MB across 63× the audio — *and* the real defect was worse than a
      size, because the codec transformer allocated **24,016 KV positions for an attention
      whose sliding window is 250**, 196.6 MB of which 2 MB was ever readable. Fixed with
      a moving base and a compaction (not a ring: a ring puts a wrap in the hottest loop
      in the codec to save an amortised one position moved per step). `context == 0`
      keeps the old layout, so the backbone and every voice-prefix path are untouched.
      The largest block left was then the **RoPE table**, 6.1 MB rebuilt identically per
      context for a pure function of (position, head_dim, max_period) — now shared.
      **−184.5 MB of address space per context, −6.1 MB per additional concurrent
      context**, audio byte-identical, four mutations caught. What is left: `serve()`
      re-running `model_init` per one-shot `mynah_tts_synthesize` (~1.1 ms/request, 0.2%
      of a 580 ms TTFA; the streaming path serves many requests per `serve()`)
- [x] E10-13 **`sea_taps_all()` keyed a cache on a pointer malloc can recycle** — fixed
      in `9cfa4fc` with the 64-sample content fingerprint `src/convq8.c` already carried,
      and gated by mutating the buffer in place rather than by free-then-malloc, so the
      hazard is reached deterministically on every platform. E10-5's own gate had found
      it: freed one tensor, allocated another of the same shape at the same address,
      relative error 0.0046 → 1.45 and nothing crashed. The test also gives
      `sea_taps_all()` its first caller on a `BLAS=scalar` build, where it was compiled,
      unreferenced and warned about
- [x] E10-14 **decided, flipped and confirmed on the shipped binary** —
      measured on the Axion at the thread counts serving actually uses, `codec.conv_stack`:
      **2 threads/worker 224.8 → 120.8 ms (1.86x)**, 4 threads 129.7 → 83.9 (1.55x). At 16
      threads it is a wash, which is why the laptop's reading was misleading — and why the
      16-thread number must not be the one quoted for a serving decision. Since the codec
      is the PER-SLOT term of `T_frame(B) = a + b*B`, that factor is capacity.
      **Every capacity number in [`.work/axion-c99-soak.md`](.work/axion-c99-soak.md) was
      measured with `codec_convtr:int8` ON, and it is OFF by default** — so those numbers
      do not describe the shipped configuration until the default is flipped. The cost is
      unchanged and known: SNR 28.9-32.9 dB against 36.4-37.9, log-mel nearly unchanged.
      **DECIDED 2026-09-19 and the default is flipped** (`ea7ddd7`). What decided it was a
      serving A/B, not the kernel table: two ten-minute soaks at C90 differing only in this
      clause gave **MARGINAL with it OFF** (TTFA p95 544 against a 500 ms gate, 5 stalls of
      15380, RTF p95 0.842, 106.5 audio-s/s) and **GOOD with it ON** (TTFA p95 492, 0 stalls
      of 17904, RTF p95 0.728, 124.3). So it is not 17% on top of a qualified configuration:
      without it there is NO qualified concurrency on that machine. The quality half was
      judged by ear on clips captured from the streaming server under C90 of real load.
      **CLOSED.** The closing gate ran: C94, thirty minutes, **nothing exported** --
      TTFA p95 494.9, RTF p95 0.746, throughput 130.8 audio-s/s, `stall@500ms` 0, which is
      the env-override arm to within the noise (490.4 / 0.754 / 131.2). What the docs
      describe and what a plain binary does are now the same thing, and E10-17 is what keeps
      them that way
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

### E11 — Ternary (W1.58) feasibility → [`.work/ternary-feasibility.md`](.work/ternary-feasibility.md)

**Analysis only. No C, no SIMD, no kernel** until the checkpoint is shown to
tolerate ternary. The expensive half is a new qtype through `src/qmat.c`; the
cheap half is a fake-quant sweep in Python, and it runs first.

- [x] E11-1 `tools/ternary_feasibility.py` — census / calib / sweep / cumulative / shapes / report; read-only on the pack, never writes a model
- [x] E11-2 **where the bytes are is not where the MACs are** — backbone holds **72.8%** of the 24.06 MB of streaming ternary weight but does **20.1%** of the MACs per frame; `mimi.decoder*` is the mirror image (27.2% / 79.9%), because one runs at 12.5 Hz and the other up to 24 kHz. Ternary would buy the backbone **bandwidth** and the codec **MAC width** — two different arguments, each to be proven separately
- [x] E11-3 **PT²'s AGA and TWLA's E2M-ATQ stage 2 are the same estimator** (per-row 2×2 solve under `S = XᵀX`, T frozen). Implemented as two methods so the gap measures only the warm start; do not budget for both
- [x] E11-4 calibration capture — exact Gram per module, 69 modules, 51 s. Independently confirmed the taxonomy: **18 encoder modules were never called** during synthesis, exactly the set marked `streaming: false`
- [x] E11-5 per-layer sensitivity, 89 layers x 8 methods → **GPTQ-style compensation is the whole game** (1.3-108x over naive, median ~13x); **AGA ≡ E2M-ATQ to within 0.3%** on every group, so do not budget for both; **2-bit+GPTQ beats 4-bit RTN** on the backbone, 0.0065 vs 0.0368
- [x] E11-6 cumulative + group probes at **temperature 0**, where FP-vs-FP is bit identical. At the shipped 0.3 the FP reseed floor is only **0.648** — the sampler is noisier than any quantizer, and the first curve measured dice. Measure at 0, listen at 0.3
- [x] E11-7 protected set is **~100K parameters, under 0.1% of the model** — the three latent-boundary projections (`flow_net.input_proj`, `flow_lm.input_linear`, `mimi.quantizer.output_proj`) plus the decoder's 1x1 residual convs. They gain 1.3-1.7x from any method against 20-108x elsewhere: irreducibly hard, not badly fitted. Best layout `ternary codec + int8 rest` = **7.48 effective bits/weight**, 102.3 MB, and the ternary part is **2.05 bits/weight** once scales and shifts are counted — never 1.58
- [x] E11-8 **NO-GO on a ternary backbone**, four independent grounds → [`docs/ternary-feasibility.md`](docs/ternary-feasibility.md). (a) **no ISA we target has a sub-byte MAC** — `src/qmat.c:1403` already unpacks int4 to int8 before the dot product, so ternary's MACs/instruction equal int8's and its only lever is traffic; (b) **79.4% of that traffic is the backbone** (which does just 22% of the MACs); (c) the backbone **fails the audio gate** — +49%/+51% duration over-run on long sentences at temp 0.3, outright non-termination at temp 0; (d) the region that passes (Mimi decoder, duration ratio **exactly 1.00** on all 7) holds 11% of the bytes, is cache-resident and is **already int8**
- [x] E11-9 **two of our own gates were wrong, and the audio caught both.** The `collapsed` threshold was a guessed [0.6, 1.6]; the FP-vs-FP control says the real duration spread is **0.90-1.12**, and the loose guess was reporting the failing configuration as clean. And a least-squares ternary fit is a **projection**, biased low in norm by the variance it cannot explain — ~2%/layer, `0.977^18 = 0.65` across the decoder cascade, which is the 38% level drop the rendered audio showed. Correcting it costs ~2% relative NMSE
- [x] E11-11 **KOTMS measured, and it loses.** TWLA's tri-modal rotation beats AGA on all 8 probed layers (1.2-3.7x) so the premise is sound, but GPTQ beats it by **2-6x on every one** — and KOTMS is the only method that leaves a permanent runtime cost (`R = R1 (x) R2` applied to the activation on every call) while GPTQ is build-time and free at inference. Answer to "PT2 or TWLA": neither; the shared part is worth 1.1-8.9x, compensation is worth 1.3-108x
- [x] E11-12 `make ternary-test` — the closed forms checked against `lstsq` to 1e-8 and the method orderings the report claims asserted on synthetic data, no model needed. A transcription error in a 2x2 solve would not crash and would not obviously corrupt audio; it would quietly make one method look worse than another, which is how a wrong number becomes a finding
- [x] E11-10 **the item found a better lever than it went looking for** — promoted out of E11 and into **E12**, which is now the active PocketTTS item
- [x] E11-13 **the harness is the durable artifact, not the verdict.** `tools/ternary_feasibility.py` is a general per-layer sensitivity rig: exact Gram capture from the reference implementation, canonical `(rows, contract)` form for linear/conv/convtr, output-error metrics on real activations, cumulative and group probes with end-to-end audio, honest byte accounting. It already measures **int8 and int4 through the identical path**, so it serves E12 and any future Q4 or QAT work without change. Keep it
- [-] not in scope, deliberately: W1.58A4 (low-bit activations), KOTMS through Adam/Cayley, ILA-AMP bit allocation, and any change to production inference behaviour

### E12 — `backbone:bf16` → `backbone:int8` → [`.work/int8-backbone.md`](.work/int8-backbone.md)

**The active PocketTTS item.** E11 measured the quality side in Python; this
measures the runtime side in C, on the box, under load. Needs **no code**:
`MYNAH_QUANT_GROUPS` already carries the A/B.

- [x] E12-1 the A/B is reachable today — `MYNAH_QUANT_GROUPS` replaces the whole spec, so arm B must respell the codec groups; both arms go in as `configs/perf` entries so the configuration cannot live in shell history
- [x] E12-2 **this Mac cannot answer it, demonstrated not assumed** — M1 is ARMv8.4 with **neither FEAT_BF16 nor FEAT_I8MM** (`--dispatch-map` says so in two rows), so the local A/B compares a degraded f16 path against SDOT-only int8 and came out 23% *slower*. On Neoverse-V2 it is BFMMLA against SMMLA. Discard the local number
- [x] E12-3 **prediction confirmed on both coefficients.** Measured on the box: `T_backbone(B)` is **1.691 + 0.835·B** at bf16 and **1.377 + 0.865·B** at int8 — `a` **−18.6%**, `b` **+3.6%**. −11.1% at B=1, **−0.3% at B=8**. int8 attacks the once-per-step weight pass and nothing else, exactly as SMMLA≡BFMMLA at 16 MACs/instruction predicts
- [x] E12-4 **serving-level A/B at C130, same build, back to back**: int8 vs bf16 within 2% on every percentile (TTFB 199.0/203.5, TTFA 409.2/402.8, RTF 0.822/0.820), both MARGINAL, both on the same missed gate. int8 is a **no-op under load**
- [x] E12-6 **NOT PROMOTED.** Gate 2 of `.work/int8-backbone.md` — TTFA p95 no worse than bf16 — is missed (409.2 vs 402.8). `POCKET_QG_DEFAULT_SPEC` unchanged; gates 4 and 5 never reached because a listening pass and a determinism sweep are not worth spending on a 0.3% change
- [x] E12-8 **`a` is not all bytes.** It fell 18.6% while the bytes fell 50%, so ≈1.06 ms of the 1.69 ms is fixed cost (per-step setup, activation quantization, epilogue, pool dispatch). A bandwidth model that treats `a` as pure weight traffic over-predicts every future weight-format change by ~2.7×. That is a correction to the instrument `backbone-bandwidth.md` reasons with, and it outlives the int8 result
- [x] E12-9 **attribution error, caught by the control.** int8's TTFA of 409.2 was first reported as −29.5% against *yesterday's* 580.9. Today's bf16 arm returns 402.8: the gain is the **build**, not the dtype. Only the arm measured beside it, same build same box same hour, is admissible
- [x] E12-10 **two operational traps, both mine.** `tmux kill-session` does not kill what the session started — a C140 screen ran concurrently with the first soak attempt. And **`pkill -x mynah-tts-server` matches nothing and reports success**: Linux truncates `/proc/<pid>/comm` to 15 chars, so the name is `mynah-tts-serve`; `pgrep -c -x` then says 0 while `ps -C` lists three. Kill by PID from `ps -C`, and prove the box is empty with a different tool than the one that killed
- [~] E12-11 **the server can report an RSS now, which it could not** — `/health`'s
      `process` block carries `rss_bytes` and `rss_peak_bytes` (`mynah_rss_bytes()`:
      `/proc/self/statm` on Linux, the Mach task port on macOS, 0 where the platform
      cannot answer — never a guess). Reads 805 MB / 808 MB peak for a single-process
      bf16 `pocket-en` here. The hypothesis is unchanged and now measurable: 151.0 →
      75.7 MB of backbone weights **times 16 prefork workers if the quantized cache is
      per-process**. Poll every worker and sum; do not read one and multiply until they
      are shown to agree

### E14 — The x86 kernel tiers, before the box → [`.work/x86-kernel-tiers.md`](.work/x86-kernel-tiers.md) · [`.work/sgemm-runtime-dispatch.md`](.work/sgemm-runtime-dispatch.md)

**PRIORITY: MAX.** Everything here is code written so that a rented x86 hour
measures this engine instead of its fallback. E4-9 closed the f32 half; these
are the tiers still missing, and the doctrine that lets them land without
putting an unexecuted kernel on the default path: **a new tier resolves only
after it has been EXECUTED in this process and checked against the scalar
reference** (`.work/x86-kernel-tiers.md`, "prove-on-first-use gate"). CPUID
alone does not promote a kernel.

- [x] E14-1 **done** — the ISA guard is compiled for the baseline; 0 AVX registers inside it against 131 elsewhere in the same avx2 object. **the ISA guard was not compiled for the baseline** — a guard built with
      `-march=native` may contain an instruction the host lacks and die before printing
      the message it exists to print. `qwen-tts` carries `target("arch=x86-64")` on its
      guard for exactly this; ours has none. One line
- [x] E14-2 **done** `dot_q8_i32_avx512bw`, `vpmaddwd` on `zmm` x6, gated on a bit-identical run against the scalar reference. **AVX-512 without VNNI had no int8 kernel** — Skylake-SP, Cascade Lake and
      Zen 3 have 512-bit registers and no VPDPBUSD, so they run the 256-bit AVX2 dot and
      half the register file idles. `_mm512_cvtepi8_epi16` + `_mm512_madd_epi16`, exact
      int32, so the gate is **bit-identical to the scalar reference**, not a tolerance.
      This is the brief's X2 question answered in code
- [x] E14-3 **done** `matvec_bf16_dpbf16_x4`, `vdpbf16ps` x4 + `vcvtneps2bf16` x8, gated on agreeing with `matvec_bf16_scalar` inside `C*FLT_EPSILON`. **x86 had no bf16 multiply and bf16 is what the backbone ships** —
      on Arm the bf16 path is +33% at B=8; on x86 we widen with a shift and multiply in
      f32. The interleave problem `src/qmat.c` documents is **avoided, not solved**:
      `_mm512_cvtneps_pbh` converts 16 f32 in order, so two concatenated with
      `_mm512_inserti64x4` give 32 consecutive bf16 — sixteen lanes of two k-adjacent
      values, which is what the instruction pairs. No scratch, no allocation in a kernel
- [x] E14-4 **done** — two translation units of one source, `src/sgemm_rt.c` picks; **0 `ymm` in `sgemm_base.o`, 621 in `sgemm_avx2.o`**, and the variant is asked of the COMPILER not of `uname -m`, because `make x86-cross` builds x86 objects on an arm64 host. **`src/sgemm.c` was 43% of the wall and compile-time on x86** →
      [`.work/sgemm-runtime-dispatch.md`](.work/sgemm-runtime-dispatch.md). Not a
      smaller version of E4-9: `SG_LANES` reaches the packed panel geometry and the
      public `mynah_sgemm_narrow_max`, so multi-versioning makes the LAYOUT a runtime
      choice. Two translation units at different `-m` flags, scalar kept as reference
- [x] E14-5 **measured on an EPYC 9254 (Genoa, Zen 4), 2026-09-22** →
      [`.work/x86-genoa-first-numbers.md`](.work/x86-genoa-first-numbers.md). From ONE
      `SIMD=portable` binary: f32 matvec **2.26x** and **sgemm 4.44x** over the scalar
      forms the same binary would have run last week, and it reaches VNNI and VDPBF16PS
      with no build flag. int8 tiers `avx512vnni 0.049 / avx512bw 0.088 / avx2 0.111 ms`;
      bf16 `vdpbf16ps 0.161 / avx2-widen 0.479`. **Kernel micro-bench, not an RTF** —
      E12 put the weight pass at 5% of the AR step at B=8, so none of this is a synthesis
      speedup and none may be quoted as one. **And the box was not quiet**: a RAID resync
      ran throughout, f32 matvec repeated between 0.243 and 0.413 ms, so read the large
      ratios and not the milliseconds — axpy 1.12x and avx512bw-vs-avx2 1.26x are inside
      that spread and are NOT measured wins
- [x] E14-6 **the x86 bf16 path was a PESSIMISATION and the bench found it** — one
      activation went through the x4 kernel with itself in all four arguments, so four
      `dpbf16` per weight load with three thrown away. Free on Arm (memory-bound at one
      activation), 3x on Zen 4. An x1 form of both x86 bf16 kernels: **0.489 → 0.161 ms**,
      bit-identical to lane 0 of the x4 so width still cannot change a row's answer
- [x] E14-7 **`make kernel-bench` and `make dispatch-gate`** — shapes without weights, so
      a rented box can be measured in its first five minutes; and the first thing that
      ever checked the dispatch report, which is the document this project relies on most
      and verified least → [`.work/qwen-tts-kernel-reuse.md`](.work/qwen-tts-kernel-reuse.md)
- [ ] E14-8 **which env flags can reach which backend — nothing says** → the one idea from
      `../qwen-tts` still untaken (`flag_parity.py`). This week added three x86-only flags
      beside two Arm-only ones; `MYNAH_QMAT_BF16DOT=off` on Graviton is silence. Build it
      as a TEST that walks the read sites, not as a generated header nothing reads
- [x] E14-9 **the x86 box ran the pack** — EPYC 9254, 24 cores. Seven topologies screened,
      four soaks, seven defects found and fixed, and the first PocketTTS numbers this
      project has from x86 → [`.work/x86-genoa-serving-capacity.md`](.work/x86-genoa-serving-capacity.md)
- [ ] E14-10 **MAX — the C80 figure is NOT a serving capacity and must be re-measured.**
      It was taken under `--same-text --max-steps 64`: 1.65 s per request, sd 0.56. A real
      bank is 4.48 s, sd 4.48, and a ragged mix is a different scheduling problem. Redo with
      `--bank tests/load_texts_en_v2.txt`, no cap, nothing else on the box: screen
      16/24/32/48/64 then soak 10 min. The only counter-evidence today was CONFOUNDED (it
      ran alongside the audio capture) and may not be quoted
      → [`.work/x86-genoa-serving-capacity.md`](.work/x86-genoa-serving-capacity.md)
- [ ] E14-11 **the load generator shares the server's cores, so no cadence percentile from
      that box is certifiable.** 19% coalesced reads against a 15% refusal threshold; Axion
      stayed under it on 32 cores. Needs a SECOND host for the generator — splitting this
      one made it worse (46.1%). Blocks any client-facing TTFA number on x86
- [ ] E14-12 **`configs/perf/epyc-9254-24c-pocket-en.json`** — the x86 twin of the Axion
      profile, so the next x86 soak is one command instead of a five-phase apparatus
- [ ] E14-13 **gcc 13.3 emits 23 warnings clang does not**, 7 of them
      `-Waggressive-loop-optimizations` in `kernels.c`/`qmat.c` (`SIZE_MAX/sizeof(float)`).
      No call site can produce such an `n`, but they cover real warnings on the PRODUCTION
      compiler. Candidate: a zero-cost `__builtin_unreachable()` precondition

### E13 — The ceiling is 128 request slots, not a speed limit → [`.work/int8-backbone.md`](.work/int8-backbone.md)

16 workers x `--max-batch 8` = **128 places**. TTFB p95 is flat at 75-77 ms up to
the 128th stream and nearly triples with two more (203.5 ms at C130). Audio is
still produced ~20% faster than it plays at C130: the queue is the limit, not
compute.

- [x] E13-1 **C128, 30 min, 65,295 requests: MARGINAL by 3 stalls@250ms** — every other gate passed, TTFB 77.4, TTFA 336.3, RTF p95 0.818, throughput **158.2 audio-s/s**, ten windows flat, both drift checks PASS. 0.005% of requests
- [ ] E13-2 **the next capacity increase is a config change, not a kernel**: raise worker count or `--max-batch`. Untried, and now the cheapest lever on the board
- [ ] E13-3 find the three stalls at C128. Smaller job than finding throughput, and it would promote the level
- [x] E13-4 C120 remains the qualified operating point; nothing here changes it

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
- [ ] E5-24 **a non-streaming request parks a worker thread, so the HTTP worker count is
      the batch ceiling** — the streaming path already hands its job to the scheduler and
      goes back to `accept()`, and the comment there says why: *"a worker blocked here for
      the length of an utterance is a server whose parallelism is its worker count, not
      its batch width"*. The non-streaming path cannot, because the worker owns the socket
      it must write the WAV to, so it sits in `job_wait()` for the whole utterance.
      **Measured**, eight concurrent non-streaming requests at `--max-batch 8`:
      `-w 4` → mean_live **3.37**, histogram capped at **B4**; `-w 8` → mean_live **5.94**,
      **B8 = 64.7%** of frames. The shipped default was `-w 4 --max-batch 8`, so `/health`
      advertised a width the machine could not reach — and on an engine whose dominant
      region is bandwidth-bound (E10-15), batch width is the *only* throughput lever.
      Mitigated: the default now follows `--max-batch` and an explicit `-w` below it is
      reported, gated in `tests/test_server.sh`. **The fix is structural**: the completion
      path should write the response the way the streaming path does, so no thread is
      parked per in-flight request. qwen-tts's v2 server is the reference for the shape
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

### E15 — PocketTTS CUDA and Linux GPU streaming → [`.work/pocket-tts-cuda-streaming-parity.md`](.work/pocket-tts-cuda-streaming-parity.md)

New, opt-in track. CPU remains the oracle/default; CUDA gets a separate server
artifact and a resident Pocket graph. Do not call a host-round-trip matmul path
“GPU Pocket”. The target is a measured C100 on an L40S, not an extrapolation.

- [x] E15-0 as-is audit: Pocket CPU driver/state, current CUDA backend, `../qwen-tts` resident CUDA/CI patterns, and vLLM-Omni CUDA-graph/async-chunk designs
- [~] E15-1 add `make cuda-server`; compile/link CLI + server in CI for explicit `sm_70`, `sm_89` (L40S) and `sm_90`; model-free check distinguishes compiled CUDA from no device — GitHub runs `35877105405`/`35877105452`/`35877105478` green
- [~] E15-2 explicit backend capability/lifecycle: backend-owned weights, graphs, batch metadata and scratch; host/device conv split and safe CPU retry are implemented, with backend health/capability counters now exposed and runtime validation remaining
- [~] E15-3 scalar-reference CUDA kernels: projection matmul, LayerNorm, GELU, residual, softmax, RoPE, attention and resident conv primitives plus model-free self-tests are implemented; `nvcc`/GPU shape sweeps remain
- [~] E15-4 resident Pocket prefill/AR + flow batch: per-request KV, batched matmul→matmat, device transformer/flow steps, cross-request attention, pinned staging, bounded host K/V/latent handoff and per-scratch `(batch-width)` CUDA-Graph capture/replay buckets are implemented; flow uses the raw FP32 model weights and still needs GPU stage parity
- [~] E15-5 resident Pocket SEANet/decoder streaming: single-context causal conv/residual/transpose-conv state, persistent device scratch, PCM handoff, graph capture and a CPU-reference model-free CUDA self-test are implemented; codec-transformer device residency and cross-request decoder batching remain
- [~] E15-6 CUDA server integration: one process/GPU build boundary, prefork refusal and existing queue/cancel/stream contract are present; `/health` and SIGUSR1 stats expose graph/H2D/D2H/decoder/fallback counters, while real-GPU validation remains
- [ ] E15-7 CPU↔CUDA stage/EOS/audio parity and solo↔batch/stream parity on a real CUDA device; CPU gates must remain green
- [ ] E15-8 L40S qualification campaign: warmups, serial A/B, batch sweep, C ladder, VRAM/RSS/transfer metrics and 30-minute C100 cadence soak

### E6 — Licensing and voice policy → [`.work/licensing-and-voice-policy.md`](.work/licensing-and-voice-policy.md)

- [ ] E6-1 `speakers.json`: `source_dataset`, `license`, `commercial_use` per voice
- [ ] E6-2 default pack ships CC0/CC-BY voices only — `jean` and `cosette` are **non-commercial**
- [ ] E6-3 verify the `embeddings_v3` voice→dataset mapping (assumed CC0, unconfirmed)
- [ ] E6-4 `LICENSES/` + `NOTICE`: CC-BY attribution, modification statement, prohibited-use as notice
- [ ] E6-5 `source.json` pins the official Kyutai repo, never an ungated mirror
- [ ] E6-6 CLI and `/v1/voices` expose the license; `--commercial-only` filter

### Deferred

- [x] **CI on the OSS repo — green again on `a62cfff`, 3/3 workflows, 30/30
  jobs** → [`.work/ci-red-oss.md`](.work/ci-red-oss.md). Seven red jobs, one
  cause: `ternary-test` made **numpy a hard dependency of `make test`** and no
  hosted runner has it. The tool skips loudly now instead of failing the build,
  CI installs numpy so the gate is not vacuous there, and `make ubsan`/`make
  asan` run a C-only `test-c` — a *memory safety* workflow could go red over a
  Python import, and did. Two standing warnings the red run surfaced are fixed
  with it (a 240-byte dispatch row that gcc measures at 321, a false-positive
  `may be used uninitialized` in `json.c`). Still open: the AVX-512 execution
  gap on hosted runners (stated in the note, not implied by a green tick) and
  `--self-test` across the whole link-only matrix.
- [-] Generic Magpie GPU expansion — existing partial backends stay as they are;
  Metal measured *slower* than CPU on Apple Silicon (`docs/performance.md:71-80`).
  Pocket-specific CUDA work is now tracked in E15 and does not reopen this item.
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
