/* The engine seam.
 *
 * One driver (inference.c) steps N requests together; one engine per model
 * family supplies the graph. The driver must never ask which engine it is
 * holding: everything it needs to make a decision is in the capability block.
 *
 * The shape below is not invented. It is the twelve calls inference.c already
 * makes on engine_magpie, generalized, with two changes forced by measurement:
 *
 *  - `step_batch` and `emit_batch` stay separate. The Magpie driver already
 *    runs the backbone for the whole batch and *then* the head and EOS for the
 *    whole batch. PocketTTS fits that split without forcing: six transformer
 *    layers, then a one-step flow head and a threshold. Collapsing them into a
 *    single `step()` would fit neither.
 *  - `decode_audio` takes **contiguous, monotonically increasing** frame ranges
 *    and the engine owns whatever continuity needs. There is no
 *    `audio_left_context_frames` capability, because replaying context is one
 *    engine's strategy, not a property of the seam: measured in
 *    .work/pocket-tts-oracle.md E2-3, Magpie replays 32 frames while PocketTTS
 *    would need 64 (5.12 s) and instead carries conv state plus a position
 *    counter.
 */
#ifndef MYNAH_TTS_ENGINE_H
#define MYNAH_TTS_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#include "mynah_tts.h"

/* Per-model engine state: weights resolved once, read-only afterwards. */
typedef struct mynah_engine_state mynah_engine_state;
/* Per-request state: KV caches, ring buffers, frame history, sampler position.
 * Exactly one per in-flight request (CLAUDE.md rule 3). */
typedef struct mynah_engine_ctx mynah_engine_ctx;
/* Scratch for one batched step, owned by the driver and lent for the call. */
typedef struct mynah_engine_scratch mynah_engine_scratch;

/* Everything the driver needs in order never to branch on the engine.
 * No field here names a codebook, a stacking factor or a local transformer. */
typedef struct {
    unsigned sample_rate;
    double   frame_rate;            /* audio frames per second */
    unsigned frames_per_step;       /* audio frames one AR step yields */
    unsigned audio_emit_frames;     /* streaming emit threshold */
    unsigned min_audio_frames;      /* EOS ignored before this many */
    unsigned default_max_steps;
    unsigned max_batch;             /* 1 disables the batched path */
    unsigned voice_count;           /* 0 = cloning only */
    unsigned needs_cfg;             /* runtime CFG doubles the step cost */
    /* Reporting and debug dumps only. NEVER a dispatch predicate: the driver
     * branching on this is the bug this whole seam exists to prevent. */
    unsigned is_discrete_codec;
    unsigned latent_dim;            /* 0 when is_discrete_codec */
} mynah_engine_caps;

typedef struct {
    int      eos;
    /* First invalid frame inside THIS step's [0, frames_per_step) window;
     * equals frames_per_step when every frame of the step is valid. Exists
     * because Magpie stacks frames, but it is expressed in generic units and
     * is 0 or 1 for a one-frame-per-step engine. */
    unsigned eos_frame;
    unsigned frames_appended;
    int      failed;                /* per request; siblings keep running */
} mynah_engine_step_result;

typedef struct {
    const char *name;               /* matched against model.json "engine" */

    int  (*model_init)(const mynah_tts_model *model, mynah_engine_state **out,
                       char *error, size_t error_capacity);
    void (*model_free)(mynah_engine_state *state);
    int  (*caps)(const mynah_tts_model *model, const mynah_engine_state *state,
                 mynah_engine_caps *out);

    int  (*ctx_new)(const mynah_tts_model *model, mynah_engine_state *state,
                    const mynah_tts_request *request, size_t max_steps,
                    uint64_t seed, mynah_engine_ctx **out_ctx,
                    char *error, size_t error_capacity);
    /* Text encoding, conditioning prefix, KV prefill. After a successful
     * prepare the context is ready for step 1. */
    int  (*prepare)(mynah_engine_ctx *ctx, char *error, size_t error_capacity);
    /* Rewind to the post-prepare state without re-encoding. */
    int  (*reset)(mynah_engine_ctx *ctx, char *error, size_t error_capacity);
    void (*ctx_free)(mynah_engine_ctx *ctx);

    /* Advance `count` independent contexts by one AR step. Failure here is
     * all-or-none: it is a failure of shared code, not of one request's data. */
    int  (*step_batch)(mynah_engine_ctx *const *ctxs, size_t count,
                       mynah_engine_scratch *scratch,
                       char *error, size_t error_capacity);
    /* Turn each context's step output into appended audio frames and an EOS
     * verdict. Per-request failures go in results[i].failed. */
    int  (*emit_batch)(mynah_engine_ctx *const *ctxs, size_t count,
                       mynah_engine_step_result *results,
                       mynah_engine_scratch *scratch,
                       char *error, size_t error_capacity);

    /* The frame history, as the driver sees it: an opaque, monotonically
     * growing sequence. The driver never inspects a frame's contents. */
    size_t (*frame_count)(const mynah_engine_ctx *ctx);
    void   (*truncate)(mynah_engine_ctx *ctx, size_t frame_count);

    /* Decode [first_frame, first_frame + frame_count) to PCM. Ranges MUST be
     * contiguous and monotonically increasing; the engine guarantees the result
     * is what a one-shot decode of the same frames would produce.
     * *out_samples is malloc'd and owned by the caller. */
    int  (*decode_audio)(mynah_engine_ctx *ctx, size_t first_frame,
                         size_t frame_count, float **out_samples,
                         size_t *out_count, char *error, size_t error_capacity);

    int  (*scratch_new)(const mynah_tts_model *model, mynah_engine_state *state,
                        size_t batch, mynah_engine_scratch **out,
                        char *error, size_t error_capacity);
    void (*scratch_free)(mynah_engine_scratch *scratch);

    /* Optional; NULL is legal. Called only when a MYNAH_DUMP_* env is set. */
    void (*debug_dump)(mynah_engine_ctx *ctx, const char *stage);
} mynah_tts_engine;

/* Resolve by the `engine` field of model.json. NULL when unknown. */
const mynah_tts_engine *mynah_engine_lookup(const char *name);

#endif
