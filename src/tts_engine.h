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

#include "graph.h"       /* mynah_graph_sink, for the explicit-engine entry */
#include "mynah_tts.h"

/* Per-model engine state: weights resolved once, read-only afterwards. */
typedef struct mynah_engine_state mynah_engine_state;
/* Per-request state: KV caches, ring buffers, frame history, sampler position.
 * Exactly one per in-flight request (AGENTS.md rule 3). */
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

/* The vtable below is APPENDABLE, and that is a load-bearing property rather
 * than a convenience: an optional hook has to be able to land before the
 * engines that will implement it, or the seam can only ever be widened by
 * changing every engine in the same commit. C already gives us the semantics --
 * a member left out of an initializer is zero, which is exactly "this engine
 * does not have that hook" -- but -Wextra reads a short positional initializer
 * as a mistake, and both engines' vtables are positional.
 *
 * So the diagnostic is turned off for the translation units that build against
 * this seam, and only for them. It is not a blanket relaxation of -Wextra: it
 * is the one warning whose advice ("list every member") is the opposite of the
 * contract this header is asserting. The real protection against a mis-assigned
 * pointer is that every new member is APPENDED and never inserted, which is
 * type-checked -- inserting one puts a function of the wrong type in the slot
 * and the build fails loudly. */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

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

    /* Advance `count` independent contexts by one AR step.
     *
     * ATOMIC OVER THE BATCH. When this returns non-zero, NO context in the
     * batch has advanced. The driver depends on it: a batched step has no
     * per-request result channel, so the only way it can tell whose data was
     * at fault is to re-step the contexts one at a time and see which one
     * refuses again. That re-step is legal exactly because of this sentence.
     *
     * An engine that advances contexts 0..i-1 and then refuses context i turns
     * one bad request into a batch-wide corruption -- the survivors would be
     * double-stepped by the isolation pass, and no driver can detect that from
     * the outside. Validate every context first, then commit, or keep the
     * per-request checks in `emit_batch` where there is a place to report
     * them. At `caps.max_batch == 1` the question does not arise, which is why
     * an engine can leave it until it widens. */
    int  (*step_batch)(mynah_engine_ctx *const *ctxs, size_t count,
                       mynah_engine_scratch *scratch,
                       char *error, size_t error_capacity);
    /* Turn each context's step output into appended audio frames and an EOS
     * verdict. Per-request failures go in results[i].failed and the return
     * value is still non-zero; a non-zero return that marks nobody means the
     * call failed for reasons that belong to no single request, and the driver
     * has to retire the whole batch -- there is nothing left it could attribute
     * and a step that fails while retiring nobody would spin forever. Marking
     * the guilty request is therefore how an engine keeps its blast radius at
     * one. */
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

    /* ---- decode a gang -------------------------------------------------
     *
     * WHY IT IS IN THE SEAM AT ALL. `decode_audio` is declared per context,
     * so the driver can never show an engine two requests' codec work at the
     * same time, and the codec is not a rounding error: the codec transformer
     * plus the convolution stack are about 55% of wall time here, and in the
     * reference implementation the decoder was 72-80% of the marginal cost of
     * each additional stream (.work/serving-design.md §4). A per-context hook
     * multiplies that by the number of concurrent streams and batches none of
     * it. This entry is the one place where the shape of the seam, rather than
     * any engine's code, decides whether that is possible.
     *
     * Each element i names a context and the half-open frame range
     * [first_frame[i], first_frame[i] + frame_count[i]).
     *
     * WHAT THE ENGINE MAY ASSUME
     *  - Each range obeys exactly the rule `decode_audio` already imposes:
     *    contiguous with, and monotonically after, everything decoded so far
     *    FOR THAT CONTEXT. The seam has no other rule, and being in a gang
     *    adds none: ranges of different contexts are unrelated and routinely
     *    differ in length, because the driver ramps each slot's quantum
     *    separately.
     *  - `count >= 1`, and no context appears twice in one call.
     *  - Every context comes from the same model and the same engine state,
     *    and none of them is inside another engine call.
     *
     * WHAT THE ENGINE MUST GUARANTEE
     *  - BIT-IDENTITY PER CONTEXT. out_samples[i] must be exactly what
     *    `decode_audio(ctxs[i], first_frame[i], frame_count[i], ...)` would
     *    have produced on its own -- the same promise the batched linear rows
     *    make, for the same reason. Which contexts share the call, how many
     *    there are and in what order they appear are a scheduling decision the
     *    driver makes on timing and remakes every step, so anything that leaked
     *    across rows would make a request's audio depend on the server's load.
     *  - PER-CONTEXT FAILURE. A range the engine cannot serve sets failed[i]
     *    and leaves out_samples[i] NULL without disturbing its neighbours. The
     *    return value is non-zero only for a failure that belongs to no single
     *    context, and then every failed[i] is set and `error` explains it.
     *
     * WHO OWNS WHAT
     *  - The driver owns the six parallel arrays, guarantees `count` entries in
     *    each, and pre-clears out_samples[] to NULL and out_count[]/failed[] to
     *    zero. The engine writes them and keeps no pointer to them.
     *  - Each non-NULL out_samples[i] is one malloc'd block per context, freed
     *    by the driver, exactly as `decode_audio` hands one back. The engine
     *    never frees a buffer it has already returned, including when a later
     *    context in the same call fails.
     *  - `scratch` is the driver's, sized for its widest batch, the same handle
     *    `step_batch` receives, and lent only for this call. Nothing in it
     *    survives to the next one.
     *
     * OPTIONAL, AND MEANT TO BE. NULL is the correct value for an engine whose
     * batched codec has not been measured yet, and it is what every engine
     * starts as. The driver never calls this pointer directly -- it goes
     * through `mynah_engine_decode_gang` below, which falls back to `count`
     * calls to `decode_audio`. An engine gaining a batched codec is therefore
     * one line in its vtable and no change anywhere above it.
     *
     * It lives at the END of this struct on purpose: the engine vtables are
     * positional initializers, so a new member anywhere else would silently
     * shift every function pointer after it. Append; never insert. */
    int  (*decode_audio_batch)(mynah_engine_ctx *const *ctxs, size_t count,
                               const size_t *first_frame, const size_t *frame_count,
                               float **out_samples, size_t *out_count, int *failed,
                               mynah_engine_scratch *scratch,
                               char *error, size_t error_capacity);

    /* ---- resumable prefill (APPENDED; see the note above about order) ----
     *
     * OPTIONAL; NULL means "my prefill is not resumable" and the driver calls
     * `prepare` instead. It exists because a prefill is not a small cost paid
     * once by the request that pays it: the driver admits at the top of a step,
     * and for as long as `prepare` runs, every request ALREADY in that batch is
     * frozen. Measured on PocketTTS on the Axion, a long text's prefill is
     * 190 ms against a frame period of 80 ms, and the resident slots show it as
     * a `max_gap` p95 of 358 ms against a p50 of 97 ms -- an interruption, not
     * a slowdown (.work/prefill-blocks-decode.md).
     *
     * Each call does at most `budget` units -- units are the engine's own, and
     * for a text prefill they are tokens -- and sets `*done` to 1 when the
     * context is ready for step 1. `budget == 0` means no limit, which makes a
     * single call equivalent to `prepare`. The driver keeps calling until
     * `*done`, steps the rest of the batch in between, and never steps a
     * context whose prefill has not finished.
     *
     * THE CONTRACT THAT MAKES IT SAFE: the audio produced must be exactly the
     * audio one `prepare` would have produced. An engine whose slicing is only
     * approximately equivalent MUST leave this NULL -- the driver cannot detect
     * the difference, and a cadence fix that quietly changes the output is not
     * a fix. PocketTTS can honour it because its prefill is already tiled at a
     * fixed width and a tile-aligned split is bit-identical to one call; the
     * alignment is asserted inside `pocket_text_flush`, not assumed here. */
    int  (*prepare_slice)(mynah_engine_ctx *ctx, size_t budget, int *done,
                          char *error, size_t error_capacity);
} mynah_tts_engine;

/* The default implementation of `decode_audio_batch`, and the driver's only
 * door to it: dispatches to the engine's hook when it has one and otherwise
 * decodes the gang one context at a time. Same arguments, same ownership, same
 * guarantees -- the fallback satisfies them trivially, which is the point.
 *
 * Returns 0 when the call itself ran; inspect failed[i] for per-context
 * outcomes. Returns non-zero only for a failure that belongs to no context. */
int mynah_engine_decode_gang(const mynah_tts_engine *engine,
                             mynah_engine_ctx *const *ctxs, size_t count,
                             const size_t *first_frame, const size_t *frame_count,
                             float **out_samples, size_t *out_count, int *failed,
                             mynah_engine_scratch *scratch,
                             char *error, size_t error_capacity);

/* Resolve by the `engine` field of model.json. NULL when unknown. */
const mynah_tts_engine *mynah_engine_lookup(const char *name);

/* Serve a sink with the engine handed in directly instead of resolved from
 * model.json. `model` is passed through to the engine untouched and may be NULL
 * for an engine that does not need one. It is the same driver the public entry
 * points run; they differ only in where the engine came from.
 *
 * It exists because of what the driver now decides on its own. The decode gang,
 * the per-request failure blast radius and the quantum ramp are policies that
 * hold for every engine by construction, and a test that has to load a model
 * pack to reach them can only ever check them for the engines that happen to be
 * installed -- and cannot make one request of sixteen fail on purpose, which is
 * the case that matters. A synthetic engine can, deterministically, in
 * milliseconds, with no weights.
 *
 * `strict_batch` refuses a batch wider than the engine's `caps.max_batch`
 * instead of quietly narrowing to it, which is what a fixed array of N jobs
 * means and what a service does not. */
int mynah_graph_serve_engine(const mynah_tts_engine *engine,
                             const mynah_tts_model *model, mynah_graph_sink *sink,
                             size_t max_batch, int strict_batch);

#endif
