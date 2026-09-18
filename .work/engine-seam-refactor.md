# E1 — Engine seam and the `graph.c` split

Status: **OPEN** · blocks E3 · does not block E2 · audit done 2026-09-12

Context: [pocket-tts-vs-mynah.md](pocket-tts-vs-mynah.md) §4. This is the July
TODO "Architecture TODO — streaming/server split", now with a second model in
hand and a measured plan.

## Problem

There is no seam. `src/tts_engine.h` and `src/engine_magpie.c` are designed in
the plan and named in `AGENTS.md`, but do not exist. Everything lives in
`src/graph.c` (4283 LOC, 68 functions), the public header carries Magpie's
dimensions, and the JSON parser cannot represent a per-engine config.

Do the refactor **with PocketTTS in hand, not before**. Guessing the seam from
one engine is how the current situation happened.

## The gate problem, and how it is solved

`models/` is empty, so no Magpie golden can be captured. Worse, the audit found
that **`make self-test` proves almost nothing about `graph.c`**:
`mynah_graph_self_test` (`graph.c:210-243`) and `mynah_graph_bnns_self_test`
(`graph.c:1953-2008`) are `return 0` without executing anything outside
Accelerate — **on Linux both are no-ops**. And `make server-test` cannot run
without a pack either (`tests/test_server.sh:12` uses `${MODEL_DIR:?}`).

So the first step is not the refactor. It is **a synthetic pack**: a
deterministic, tiny, generated model pack that exercises the whole graph
(encoder → decoder → local transformer → NanoCodec → WAV) in a few MB. It
restores `make stream-test`, `make server-test`, `--synthesize` and the
offline↔streaming comparison, and gives a byte-identical golden.

This is *sufficient*, not a compromise: the refactor must not change the
numbers, and for "did the numbers change" any deterministic pack is exactly as
good as the real one. It does not replace NeMo oracle parity, which is a
different question and already closed.

Constraints the fake pack must respect, because they are literals in the code:

- codec: `codebook_count = 8`, `codebook_size = 2016`, latent 32, pre-conv 864
  channels, 5 stages `rates {8,8,4,2,2}` (`graph.c:2653-2656, 2671, 2686-2687`).
  FSQ `levels {8,7,6,6}` multiplies out to 2016 — any other size decodes
  out-of-range indices silently.
- transformer: `hidden_dim` divisible by **12** (`graph.c:739, 1016, 2970`),
  `ffn = 4×hidden`, `frame_stacking_factor = 2` **mandatory**
  (`graph.c:770` hard-fails otherwise), `local_transformer_layers ≤ 4`
  (`graph.c:834`).

`hidden_dim=24, encoder_layers=1, decoder_layers=1, local_layers=1,
speaker_count=1, max_decoder_steps=8` fits in a few MB.

## Target shape

| Module | From `graph.c` | ~LOC |
|---|---|---|
| `src/mynah_util.{h,c}` | 63-107 | 40 |
| `src/kernels.c` (+=) | 152-243 | +80 |
| `src/conv1d.{h,c}` | 271-286, 1679-2222 | 420 |
| `src/codec_nanocodec.{h,c}` | 2226-2780 | 700 |
| `src/transformer.{h,c}` | 147-150, 245-265, 288-711, 2795-3111, 3127-3284 | ~860 |
| `src/attention.c` | 3292-3654 (`decoder_run` alone) | 363 |
| `src/engine_magpie.c` | 713-1656 + the Magpie half of 2964-3111 | 1050 |
| `src/sampler.c` | the RNG + top-k out of 1374-1656 | ~80 |
| `src/inference.c` | 3679-4248 | 380 |
| `src/stream.c` | 3656-3672, 3910-3969 + `mynah_tts_stream_*` | 250 |

`graph.c` does not survive the split. `decoder_run` must go in its own file or
`transformer.c` breaks the ~1200 LOC ceiling.

## `src/tts_engine.h`

The driver already calls `decoder_step_batch` (`graph.c:4116`) and *then*
`sample_local_frame_batch` (`graph.c:4159`) as two separate batched passes.
**That split is the seam.** It maps 1:1 onto backbone-step plus head-and-EOS,
and PocketTTS fits it without forcing: 6-layer backbone, then a one-step flow
head and a threshold. **Do not collapse them into one `step()`.**

```c
typedef struct {                      /* capabilities: no field names a codebook */
    unsigned sample_rate, frames_per_step;
    double   frame_rate;
    unsigned audio_emit_frames, min_audio_frames;
    unsigned default_max_steps, max_batch, voice_count;
    unsigned needs_cfg;
    unsigned is_discrete_codec;          /* reporting/dumps ONLY, never dispatch */
    unsigned latent_dim;                 /* 0 when discrete */
} mynah_engine_caps;

typedef struct {
    int      eos;
    unsigned eos_frame;       /* first invalid frame inside this step's window */
    unsigned frames_appended;
    int      failed;          /* per-request; siblings keep running */
} mynah_engine_step_result;

typedef struct {
    const char *name;
    int  (*model_init)(...);  void (*model_free)(...);  int (*caps)(...);
    int  (*ctx_new)(...);     int  (*prepare)(...);
    int  (*reset)(...);       void (*ctx_free)(...);
    int  (*step_batch)(mynah_engine_ctx *const *, size_t, mynah_engine_scratch *, ...);
    int  (*emit_batch)(mynah_engine_ctx *const *, size_t,
                       mynah_engine_step_result *, mynah_engine_scratch *, ...);
    size_t (*frame_count)(const mynah_engine_ctx *);
    void   (*truncate)(mynah_engine_ctx *, size_t);
    /* Ranges MUST be contiguous and monotonically increasing. The engine keeps
     * whatever state continuity needs: Magpie replays 32 frames of left
     * context internally, PocketTTS carries conv ring buffers plus a position
     * counter. Measured in E2-3: replay is not viable for PocketTTS. */
    int  (*decode_audio)(mynah_engine_ctx *, size_t first, size_t count,
                         float **out, size_t *n, ...);
    int  (*scratch_new)(...); void (*scratch_free)(...);
    void (*debug_dump)(mynah_engine_ctx *, const char *stage);  /* optional */
} mynah_tts_engine;
```

How the two engines land:

| | Magpie | PocketTTS |
|---|---|---|
| `frames_per_step` | `frame_stacking_factor` (2) | 1 |
| `min_audio_frames` | `min_generated_frames` *and* the bare `4u` at `graph.c:3971` | 1 |
| `max_batch` | 16 | 1 until measured |
| `is_discrete_codec` / `latent_dim` | 1 / 0 | 0 / 32 |
| `needs_cfg` | 1 | 0 (distilled in) |
| `step_batch` | `decoder_step_batch` | 6-layer causal backbone |
| `emit_batch` | `sample_local_frame_batch` | flow head + `out_eos < -4.0` |
| `decode_audio` | `decode_codec` | decoder transformer + SEANet |

`eos_frame` exists only because Magpie stacks frames, but it is expressed in
generic units (index inside the step window) and is 0 or 1 for PocketTTS. It is
not a Magpie field.

## Constants that become capabilities

`STREAM_CONTEXT_FRAMES 32` (`:27`, becomes Magpie-internal rather than a
capability — see E2-3), `STREAM_EMIT_FRAMES 16` (`:33`),
`MYNAH_MAX_BATCH 16` (`:12`, duplicated as `MYNAH_GRAPH_MAX_JOBS` in
`graph.h:15` — **unify, changing one alone overflows a stack array**),
`heads = 12` (`:739, 1016, 2970`), `ffn = width*4`, `kernel = 3` (`:739`),
`stacking == 2` (`:770`), the `4u` EOS floor (`:3971`), `layers <= 4`
(`:834-847`), and the whole FSQ block (`:2653-2687`).

## Shared mutable state — AGENTS.md rule 3 violations found

1. **`model->codec_cache`** (`mynah_tts_internal.h:16`) — mutable BNNS/tap cache
   hung off a `const mynah_tts_model *`, written by `codec_cached_taps`
   (`:1769`) and `conv1d_causal_bnns` (`:1839`) under a mutex keyed by owning
   `pthread_t` (`:1687, 1863`). Correct but process-global, and it serialises
   under contention. Once `inference.c` is its own TU the const-cast stops being
   file-local and becomes a cross-module contract — make it explicit.
2. **`model->qcache`** — same shape, mutex at `qmat.c:779/836`, touched in 18 hot
   sites.
3. **`model->local_projection_cache`** — read-only after creation, so not a
   violation, but Magpie-specific state hanging off the generic model struct.
   Move into the engine's model state.
4. **`src/threads.c:65-73`** — one global pool with a **single job slot**;
   `mynah_parallel_for` (`:128-158`) `trylock`s and runs inline under contention.
   So `inference.c` **cannot** parallelise slots on top of the `parallel_for`
   already used by the codec convolutions. Know this before designing batching.

There are **no file-scope globals in `graph.c` itself** — the shared state is all
hung off the model.

## Execution order — each step leaves the repo buildable

0. **Synthetic pack** (`tools/make_fake_pack.py`) + capture golden offline and
   streaming, f32 and int8. *Prerequisite, not optional.*
1. `mynah_util.{h,c}` + kernel moves. Watch: `gelu_tanh_array` has an
   `MYNAH_USE_ACCELERATE` path — confirm the define is global in the Makefile,
   not per-file.
2. `conv1d.{h,c}`. **Do not touch the BNNS locking while moving it** (it was
   fixed in `b6ce0ef`).
3. `codec_nanocodec.{h,c}`. Watch: `decode_codec_resident` returns `1` meaning
   "not applicable, continue on CPU" — a three-valued return that is easy to
   break.
4. `transformer.{h,c}` + `attention.c`. **Highest risk step**, see below.
5. `engine_magpie.c` with plain `magpie_*` functions, still called directly.
   Verify in **both** `MYNAH_QUANT=f32` and `int8` — the int8 run is what catches
   a mangled qcache key name.
6. `inference.c` + `stream.c`.
7. `tts_engine.h` + `mynah_engine_lookup`, one engine registered. `graph.c`
   disappears. **This is the only step where audio can legitimately change**, so
   keep it alone.
8. `engine_pocket.c` stub that compiles against the header — the mitigation
   against shaping the seam around Magpie again.
9. Public header cleanup + real JSON parser. Blast radius verified and small:
   `cli/main.c:42-58,312`, `server/main.c:424`, `tests/test_stream.c:69`, plus
   `tools/convert_magpie.py:138-149,174` and `tools/inspect_nemo.py:48-65`.

## Risks found by reading the code — not generic ones

1. **`decoder_run` (363 LOC) has 8 environment-dependent branches read inside the
   loop**: `MYNAH_GPU_RESIDENT` (:3366), `MYNAH_METAL_GPU_ATTENTION` (:3359),
   `MYNAH_GELU_SCALAR` (:3330, :3630), `MYNAH_TIMING` (:3304), plus 4 dumps.
   Moving it to a TU that does not get the same `-D` silently drops the prefill
   to scalar GELU: a ~1e-7 change, too small to break a tolerance and big enough
   to break byte-identity. **Extract the dumps before moving the function.**
2. **The only RNG in the runtime** is an inline xorshift64 at `:1601-1607`,
   reseeded with `0x9e3779b97f4a7c15`, drawing `(state >> 40) & 0xffffff`.
   Move it **bit-for-bit into `sampler.c`; do not tidy it.** A different
   intermediate type changes every `temperature > 0` generation.
3. **Two uncoordinated minimum-length thresholds**: `predicted_stacks >= 4u`
   (`:3971`) and `min_raw_length` inside `sample_local_frame_batch`. Collapsing
   one onto the other changes the duration of short utterances.
4. **`local_cache` has fixed `[4]` arrays** (`:834-847`) and `local_cache_init`
   resolves weights only for `l < 4` (`:1034`), while the KV is allocated at full
   size. `local_transformer_layers > 4` gives NULL pointers with no error. A
   latent bug the refactor must not propagate.
5. **`embed_audio_frame` hard-fails on `frame_stacking_factor != 2`** (`:770`)
   and its body physically assumes two frames (`:776-803`). `caps.frames_per_step`
   is not enough — **rewrite it, do not parameterise it.**
6. **The "shared" transformer functions are not shared.**
   `transformer_stack`/`self_attention`/`cross_attention`/`causal_conv_ffn` take a
   `prefix` but hardcode NeMo naming (`.self_attention.qkv_net.weight`,
   `.pos_ff.proj.conv.weight`, `.position_embeddings.weight`). PocketTTS has
   fused QKV `[3072,1024]`, non-gated FFN, LayerNorm with bias. **The correct
   shared form is functions taking already-resolved weight pointers, not
   functions that format names.** Decide this before step 4, not during.
7. **`layer_norm` (`:147-150`) hardcodes bias NULL and eps 1e-5**, while
   `kernels.h:18-20` shows the kernel supports bias. PocketTTS needs bias at
   1e-5 in the backbone and no affine at 1e-6 in the flow head. Parameterise.
8. **39 `getenv()` calls**, 6 inside hot loops (`:3304, 3330, 3359, 3366, 3630,
   2647`) — a standing violation of rule 4, and a silent behaviour-change risk
   when a call moves to a TU with a different default.
9. **Contention is untested.** `server/main.c:110` serialises whole syntheses, so
   no existing test ever runs two `decode_codec` concurrently. The per-thread
   BNNS cache and the qcache are exercised by nothing under contention.

## Acceptance gate

- Synthetic-pack golden **byte-identical** offline and streaming, f32 and int8,
  batch 1 and batch 4 (batch 4 must equal batch 1 — a promise in
  `src/mynah_tts.h`).
- `make stream-test` and `make server-test` green against the synthetic pack.
- `make ubsan` clean; `make leaks` clean on macOS, since ownership moves.
- `grep -c 'codebook\|frame_stacking\|local_transformer' src/mynah_tts.h` → 0.
- No file in `src/` over ~1200 LOC.
- `engine_pocket.c` compiles against the header before E1 is called done.

## E1-1: header landed, code movement still to do — 2026-09-12

`src/tts_engine.h` exists and compiles. It is not invented: it is the twelve
calls `inference.c` already makes on `engine_magpie`, generalized, with the two
changes that measurement forced — `step_batch` and `emit_batch` stay separate
(the driver already runs the backbone for the whole batch and *then* the head,
and PocketTTS fits that split without forcing), and `decode_audio` takes
contiguous monotonic ranges with no left-context capability (E2-3).

**What still has to move**, and why it is its own step: `inference.c` is still
Magpie-aware. It reads `codebook_count`, `frame_stacking_factor`,
`audio_eos_id`, `audio_vocab_size` and `codebook_size` directly, performs the
`final_proj` argmax over every stream itself, builds the per-codebook code
buffers with their `max_raw_length` stride, and calls the codec. `synth_slot`
holds `decoder_cache` and `local_frame_state` **by value**.

The move, concretely:

| From `inference.c` | To |
|---|---|
| `slot_prepare` lines ~137-232 (encode text, baked context, code seeding, decoder cache, prefill, local state) | `magpie_ctx_new` + `prepare` |
| `slot_advance` the `!use_local_transformer` `final_proj` argmax block | `emit_batch` |
| `slot_advance` code-buffer building and the codec call | `decode_audio` |
| `slot_finalize` code copy + final decode | `flush` / `decode_audio` |
| `decoder_cache` + `local_frame_state` fields of `synth_slot` | inside `magpie_ctx` |

What stays in the driver: slot lifetime, the batching loop, the streaming emit
policy, the sinks, per-request failure, and the RNG *seeding* (the RNG itself is
sampling and goes with the engine).

**This is the one step where the audio can legitimately change**, so it must be
isolated from the mechanical moves and verified against the goldens on its own.
Do not combine it with anything else.
