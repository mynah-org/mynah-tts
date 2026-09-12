/* Magpie behind the engine seam.
 *
 * `engine_magpie.c` is the graph: text encoder, incremental decoder, local
 * transformer, sampler. This file is the request-shaped wrapper around it --
 * the state one in-flight request owns, and the twelve calls `tts_engine.h`
 * defines on top of it. The split is deliberate: the graph file is already
 * large, and nothing here is arithmetic.
 *
 * Everything Magpie-specific that used to sit in the driver lives here now:
 * the stacked-frame layout of the code buffer, the argmax over the output
 * projection, the EOS floor, the audio-decoder call and the left context a
 * streamed chunk is decoded with.
 */
#include "engine_magpie.h"
#include "codec_nanocodec.h"
#include "mynah_util.h"
#include "tts_engine.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Left context, in codec frames, that a streamed chunk is decoded with.
 *
 * The audio decoder is causal end to end, so an output only ever depends on
 * inputs at or before it, and a chunk decoded with at least its receptive field
 * of history is bit-identical to the same frames decoded from the start.
 *
 * The receptive field, walked backwards through the stack with upsampling rates
 * {8, 8, 4, 2, 2}: the final sample-rate conv contributes 6/1024 frames, each
 * stage's residual layer at most 120 positions at its own rate (kernel 11 over
 * dilations 1, 3, 5, each block being an input conv plus a skip conv), every
 * transposed conv one input position, and the frame-rate pre_conv 6 frames.
 * That sums to under 25 frames; 32 leaves margin without costing anything
 * measurable.
 *
 * This is not a capability. Replaying context is this engine's strategy for
 * keeping decode continuous, not a property of the seam -- another engine
 * carries convolution state instead. The driver only ever asks for a
 * contiguous frame range. */
#define MAGPIE_STREAM_CONTEXT_FRAMES 32u

/* Frames to accumulate before emitting. The decode cost per chunk is dominated
 * by the fixed context above, so emitting every single step would pay it 16
 * times over for the same audio. At ~21.5 frames/s this is well under 1 s of
 * added latency to the first chunk, and none to the total. */
#define MAGPIE_STREAM_EMIT_FRAMES 16u

/* Stacked positions that must exist before an EOS is allowed to stop
 * generation.
 *
 * This is NOT the same threshold as `min_generated_frames`, which forbids the
 * sampler from *drawing* EOS at all. That one changes which tokens come out;
 * this one only decides whether a drawn EOS ends the request. Collapsing them
 * changes the duration of short utterances, so they stay apart. */
#define MAGPIE_EOS_STACK_FLOOR 4u

/* Magpie resolves every weight straight out of the model pack, so there is no
 * per-model engine state yet. The handle exists so the seam is honest about
 * ownership rather than passing NULL around. */
struct mynah_engine_state {
    const mynah_tts_model *model;
};

/* One batched step's quantized-activation scratch, lent by the driver. */
struct mynah_engine_scratch {
    batch_scratch batch;
};

/* Everything one in-flight request owns. Nothing here is shared between
 * contexts except the read-only model, which is what makes the batched
 * projections safe (CLAUDE.md rule 3). */
struct mynah_engine_ctx {
    const mynah_tts_model *model;
    const mynah_tts_request *request;
    /* owned buffers */
    float *memory;
    unsigned *codes;
    float *out_last;
    float *audio_row;
    float *prefill_last;          /* the prefill's hidden row, kept for reset */
    decoder_cache cache;
    local_frame_state local_state;
    float *decoder_dev_last;      /* device-resident hidden row, not owned */
    /* derived constants */
    const float *context;
    size_t context_length;
    size_t prefill_length;
    size_t max_steps;
    size_t max_raw_length;
    size_t min_raw_length;
    size_t stacking;
    size_t codebooks;
    float temperature;
    unsigned topk;
    /* progress */
    uint64_t seed;
    uint64_t rng_state;
    size_t step;                  /* 1-based stacked position of the next step */
    size_t predicted_stacks;
    size_t eos_frame;             /* sticky: the earliest EOS still in force */
    size_t step_eos_frame;        /* the EOS of the step just emitted */
    size_t frame_limit;           /* SIZE_MAX until truncate() fixes the end */
    int prepared;
    int finished;
};

/* ---- frame accounting ---------------------------------------------------- */

/* Frames the request has produced, before any truncation.
 *
 * A step that saw EOS contributes only the frames of its window that precede
 * the EOS, which is why this is not simply stacks * stacking. */
static size_t magpie_frames_raw(const mynah_engine_ctx *ctx) {
    if (ctx->step_eos_frame != SIZE_MAX && ctx->predicted_stacks > 0u) {
        return (ctx->predicted_stacks - 1u) * ctx->stacking + ctx->step_eos_frame;
    }
    return ctx->predicted_stacks * ctx->stacking;
}

static size_t magpie_frame_count(const mynah_engine_ctx *ctx) {
    if (ctx == NULL) return 0u;
    const size_t raw = magpie_frames_raw(ctx);
    return ctx->frame_limit < raw ? ctx->frame_limit : raw;
}

/* Fix the end of the sequence.
 *
 * `frames` is the driver's view of how far the stream got; the engine also
 * applies its own EOS, which may sit earlier than that when an EOS fired below
 * the stack floor and generation carried on past it. The shorter of the two
 * wins, which is exactly what the offline sink used to compute for itself. */
static void magpie_truncate(mynah_engine_ctx *ctx, size_t frames) {
    if (ctx == NULL) return;
    size_t limit = ctx->predicted_stacks * ctx->stacking;
    if (ctx->eos_frame != SIZE_MAX && ctx->predicted_stacks > 0u) {
        limit = (ctx->predicted_stacks - 1u) * ctx->stacking + ctx->eos_frame;
    }
    if (frames < limit) limit = frames;
    ctx->frame_limit = limit;
}

/* ---- lifecycle ----------------------------------------------------------- */

static int magpie_model_init(const mynah_tts_model *model, mynah_engine_state **out,
                             char *error, size_t error_capacity) {
    if (out == NULL) return -1;
    *out = NULL;
    if (model == NULL) {
        mynah_graph_error(error, error_capacity, "engine init needs a model");
        return -1;
    }
    mynah_engine_state *state = (mynah_engine_state *)calloc(1, sizeof(*state));
    if (state == NULL) {
        mynah_graph_error(error, error_capacity, "out of memory creating engine state");
        return -1;
    }
    state->model = model;
    *out = state;
    return 0;
}

static void magpie_model_free(mynah_engine_state *state) {
    free(state);
}

static int magpie_caps(const mynah_tts_model *model, const mynah_engine_state *state,
                       mynah_engine_caps *out) {
    (void)state;
    if (model == NULL || out == NULL) return -1;
    const unsigned stacking = model->info.frame_stacking_factor;
    if (stacking == 0u) return -1;
    memset(out, 0, sizeof(*out));
    out->sample_rate = model->info.sample_rate;
    out->frame_rate = model->info.frame_rate;
    out->frames_per_step = stacking;
    out->audio_emit_frames = MAGPIE_STREAM_EMIT_FRAMES;
    out->min_audio_frames = model->info.min_generated_frames;
    out->default_max_steps = (model->info.max_decoder_steps + stacking - 1u) / stacking;
    out->max_batch = MYNAH_MAX_BATCH;
    out->voice_count = model->info.speaker_count;
    /* The released checkpoint bakes its guidance in; there is no second
     * conditional pass at runtime, so a step costs what it costs. */
    out->needs_cfg = 0u;
    out->is_discrete_codec = 1u;
    out->latent_dim = 0u;
    return 0;
}

static int magpie_ctx_new(const mynah_tts_model *model, mynah_engine_state *state,
                          const mynah_tts_request *request, size_t max_steps,
                          uint64_t seed, mynah_engine_ctx **out_ctx,
                          char *error, size_t error_capacity) {
    (void)state;
    if (out_ctx == NULL) return -1;
    *out_ctx = NULL;
    if (model == NULL || request == NULL) {
        mynah_graph_error(error, error_capacity, "invalid synthesis request");
        return -1;
    }
    if (request->speaker >= model->info.speaker_count) {
        mynah_graph_error(error, error_capacity, "speaker index is outside the model");
        return -1;
    }
    const size_t stacking = model->info.frame_stacking_factor;
    if (stacking == 0u || max_steps == 0u ||
        max_steps + 1u > SIZE_MAX / stacking) {
        mynah_graph_error(error, error_capacity, "request step budget is out of range");
        return -1;
    }
    mynah_engine_ctx *ctx = (mynah_engine_ctx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        mynah_graph_error(error, error_capacity, "out of memory preparing the request");
        return -1;
    }
    ctx->model = model;
    ctx->request = request;
    ctx->stacking = stacking;
    ctx->codebooks = model->info.codebook_count;
    ctx->max_steps = max_steps;
    ctx->max_raw_length = (max_steps + 1u) * stacking;
    ctx->min_raw_length = model->info.min_generated_frames;
    ctx->temperature = request->temperature;
    if (!(ctx->temperature >= 0.0f)) ctx->temperature = model->info.default_temperature;
    ctx->topk = request->topk == 0u ? model->info.default_topk : request->topk;
    ctx->seed = seed == 0u ? UINT64_C(0x9e3779b97f4a7c15) : seed;
    ctx->rng_state = ctx->seed;
    ctx->eos_frame = SIZE_MAX;
    ctx->step_eos_frame = SIZE_MAX;
    ctx->frame_limit = SIZE_MAX;
    *out_ctx = ctx;
    return 0;
}

static void magpie_ctx_free(mynah_engine_ctx *ctx) {
    if (ctx == NULL) return;
    magpie_local_frame_state_free(&ctx->local_state);
    magpie_decoder_cache_free(&ctx->cache);
    free(ctx->memory);
    free(ctx->codes);
    free(ctx->out_last);
    free(ctx->audio_row);
    free(ctx->prefill_last);
    free(ctx);
}

/* Seed the history with the start-of-audio token in every stream. */
static void magpie_seed_codes(mynah_engine_ctx *ctx) {
    memset(ctx->codes, 0,
           ctx->codebooks * ctx->max_raw_length * sizeof(*ctx->codes));
    for (size_t c = 0; c < ctx->codebooks; ++c) {
        for (size_t t = 0; t < ctx->stacking; ++t) {
            ctx->codes[c * ctx->max_raw_length + t] = ctx->model->info.codebook_size;
        }
    }
}

/* Everything before the first autoregressive step: text encoding, the constant
 * cross-attention cache, and the baked speaker context prefill. These are
 * multi-row matmuls that BLAS already runs efficiently and that share nothing
 * between requests, so they stay per context. */
static int magpie_prepare(mynah_engine_ctx *ctx, char *error, size_t error_capacity) {
    if (ctx == NULL) return -1;
    const mynah_tts_model *model = ctx->model;
    const mynah_tts_request *request = ctx->request;
    const size_t width = model->info.hidden_dim;

    if (magpie_encode_text(model, request->text_ids, request->text_length,
                           &ctx->memory, error, error_capacity) != 0) {
        return -1;
    }
    mynah_tensor context_tensor;
    if (mynah_tensor_get(model->tts, "baked_context_embedding.weight", &context_tensor,
                         error, error_capacity) != 0 || context_tensor.rank != 2 ||
        context_tensor.shape[1] % width != 0 ||
        request->speaker >= context_tensor.shape[0]) {
        mynah_graph_error(error, error_capacity, "baked speaker context is invalid");
        return -1;
    }
    ctx->context_length = context_tensor.shape[1] / width;
    ctx->context = context_tensor.data +
                   (size_t)request->speaker * context_tensor.shape[1];

    ctx->codes = (unsigned *)calloc(ctx->codebooks * ctx->max_raw_length,
                                    sizeof(*ctx->codes));
    ctx->out_last = mynah_alloc_floats(width, error, error_capacity);
    ctx->audio_row = mynah_alloc_floats(width, error, error_capacity);
    ctx->prefill_last = mynah_alloc_floats(width, error, error_capacity);
    if (ctx->codes == NULL || ctx->out_last == NULL || ctx->audio_row == NULL ||
        ctx->prefill_last == NULL) {
        mynah_graph_error(error, error_capacity, "out of memory preparing the request");
        return -1;
    }
    magpie_seed_codes(ctx);

    if (magpie_decoder_cache_init(model, &ctx->cache, ctx->memory, request->text_length,
                                  ctx->context_length + ctx->max_steps + 2u,
                                  error, error_capacity) != 0) {
        return -1;
    }
    ctx->decoder_dev_last = NULL;
    if (magpie_decoder_run(model, &ctx->cache, ctx->context, ctx->context_length,
                           ctx->out_last,
                           request->use_local_transformer ? &ctx->decoder_dev_last : NULL,
                           error, error_capacity) != 0) {
        return -1;
    }
    ctx->prefill_length = ctx->cache.length;
    memcpy(ctx->prefill_last, ctx->out_last, width * sizeof(float));
    if (request->use_local_transformer) {
        const size_t vocab = model->info.audio_vocab_size;
        const size_t top_count = ctx->temperature > 0.0f && ctx->topk > 1u
            ? (ctx->topk < vocab ? ctx->topk : vocab) : 0u;
        if (magpie_local_frame_state_init(model, &ctx->local_state, top_count,
                                          error, error_capacity) != 0) {
            return -1;
        }
    }
    ctx->step = 1u;
    ctx->prepared = 1;
    return 0;
}

/* Rewind to the post-prepare state without re-encoding.
 *
 * The prefix of the KV cache written by the prefill is still valid, so the
 * rewind is a length reset plus the sampler state; only the hidden row has to
 * be restored, because the first step reads it back. */
static int magpie_reset(mynah_engine_ctx *ctx, char *error, size_t error_capacity) {
    if (ctx == NULL || !ctx->prepared) {
        mynah_graph_error(error, error_capacity, "engine context is not prepared");
        return -1;
    }
    ctx->cache.length = ctx->prefill_length;
    memcpy(ctx->out_last, ctx->prefill_last,
           ctx->model->info.hidden_dim * sizeof(float));
    ctx->decoder_dev_last = NULL;
    ctx->local_state.cache.length = 0u;
    magpie_seed_codes(ctx);
    ctx->rng_state = ctx->seed;
    ctx->step = 1u;
    ctx->predicted_stacks = 0u;
    ctx->eos_frame = SIZE_MAX;
    ctx->step_eos_frame = SIZE_MAX;
    ctx->frame_limit = SIZE_MAX;
    ctx->finished = 0;
    return 0;
}

/* ---- the autoregressive step --------------------------------------------- */

static int magpie_step_batch(mynah_engine_ctx *const *ctxs, size_t count,
                             mynah_engine_scratch *scratch,
                             char *error, size_t error_capacity) {
    if (count == 0u) return 0;
    if (ctxs == NULL) return -1;
    if (count > MYNAH_MAX_BATCH) {
        mynah_graph_error(error, error_capacity, "step batch exceeds MYNAH_MAX_BATCH");
        return -1;
    }
    const mynah_tts_model *model = ctxs[0]->model;
    decoder_cache *caches[MYNAH_MAX_BATCH];
    const float *inputs[MYNAH_MAX_BATCH];
    float *outs[MYNAH_MAX_BATCH];
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (magpie_embed_audio_frame(model, ctx->codes, ctx->max_raw_length,
                                     ctx->step - 1u, ctx->audio_row,
                                     error, error_capacity) != 0) {
            return -1;
        }
        ctx->decoder_dev_last = NULL;
        caches[i] = &ctx->cache;
        inputs[i] = ctx->audio_row;
        outs[i] = ctx->out_last;
    }
    if (count == 1u) {
        /* Alone: go through decoder_run so a resident GPU step is still
         * available. It reduces to the same batched step on CPU. */
        mynah_engine_ctx *ctx = ctxs[0];
        return magpie_decoder_run(model, &ctx->cache, ctx->audio_row, 1u, ctx->out_last,
                                  ctx->request->use_local_transformer
                                      ? &ctx->decoder_dev_last : NULL,
                                  error, error_capacity);
    }
    return magpie_decoder_step_batch(model, caches, inputs, outs, count,
                                     scratch == NULL ? NULL : &scratch->batch,
                                     error, error_capacity);
}

/* Greedy head for the configuration without the autoregressive local stack:
 * one argmax per stream over the output projection, with the same forbidden
 * token rule the sampled path uses. */
static int magpie_emit_projection(mynah_engine_ctx *ctx, int *saw_eos,
                                  size_t *step_eos_frame,
                                  char *error, size_t error_capacity) {
    const mynah_tts_model *model = ctx->model;
    const size_t width = model->info.hidden_dim;
    const size_t raw_length = ctx->step * ctx->stacking;
    mynah_tensor projection;
    mynah_tensor bias;
    if (mynah_tensor_get(model->tts, "final_proj.weight", &projection,
                         error, error_capacity) != 0 ||
        mynah_tensor_get(model->tts, "final_proj.bias", &bias,
                         error, error_capacity) != 0) {
        return -1;
    }
    const size_t streams = magpie_stacked_stream_count(model);
    const int forbid_eos = ctx->predicted_stacks * ctx->stacking < ctx->min_raw_length;
    const unsigned eos_id = model->info.audio_eos_id;
    const size_t vocab = model->info.audio_vocab_size;
    const float *last = ctx->out_last;
    for (size_t stream = 0; stream < streams; ++stream) {
        unsigned value = 0;
        float best = -FLT_MAX;
        for (size_t candidate = 0; candidate < vocab; ++candidate) {
            /* Real codes always, AUDIO_EOS unless too early, nothing else. */
            const int is_code = candidate < model->info.codebook_size;
            const int is_eos = candidate == eos_id;
            if (!is_code && !(is_eos && !forbid_eos)) continue;
            float score = bias.data[stream * model->info.audio_vocab_size + candidate];
            const float *row = projection.data +
                (stream * model->info.audio_vocab_size + candidate) * width;
            for (size_t d = 0; d < width; ++d) score += row[d] * last[d];
            if (score > best) {
                best = score;
                value = (unsigned)candidate;
            }
        }
        if (value == eos_id) {
            *saw_eos = 1;
            const size_t frame = stream / ctx->codebooks;
            if (frame < *step_eos_frame) *step_eos_frame = frame;
        }
        const unsigned store = value < model->info.codebook_size ? value : 0u;
        const size_t fs = stream / ctx->codebooks;
        const size_t codebook = stream % ctx->codebooks;
        ctx->codes[codebook * ctx->max_raw_length + raw_length + fs] = store;
    }
    return 0;
}

static int magpie_emit_batch(mynah_engine_ctx *const *ctxs, size_t count,
                             mynah_engine_step_result *results,
                             mynah_engine_scratch *scratch,
                             char *error, size_t error_capacity) {
    if (count == 0u) return 0;
    if (ctxs == NULL || results == NULL) return -1;
    if (count > MYNAH_MAX_BATCH) {
        mynah_graph_error(error, error_capacity, "emit batch exceeds MYNAH_MAX_BATCH");
        return -1;
    }
    const mynah_tts_model *model = ctxs[0]->model;
    size_t before[MYNAH_MAX_BATCH];
    local_batch_item items[MYNAH_MAX_BATCH];
    size_t item_of[MYNAH_MAX_BATCH];
    size_t local_count = 0;

    for (size_t i = 0; i < count; ++i) {
        memset(&results[i], 0, sizeof(results[i]));
        results[i].eos_frame = (unsigned)ctxs[i]->stacking;
        before[i] = magpie_frames_raw(ctxs[i]);
        item_of[i] = SIZE_MAX;
    }
    /* The local stack is read once per stacked stream, so a decode step walks
     * its weights sixteen times: batching it across requests matters more than
     * batching the backbone itself. */
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        if (!ctx->request->use_local_transformer) continue;
        local_batch_item *item = &items[local_count];
        memset(item, 0, sizeof(*item));
        item->state = &ctx->local_state;
        item->decoder_last = ctx->out_last;
        item->decoder_dev_last = ctx->decoder_dev_last;
        item->codes = ctx->codes;
        item->raw_offset = ctx->step * ctx->stacking;
        item->code_stride = ctx->max_raw_length;
        item->generated_raw_length = ctx->predicted_stacks * ctx->stacking;
        item->min_raw_length = ctx->min_raw_length;
        item->temperature = ctx->temperature;
        item->topk = ctx->topk;
        item->rng_state = &ctx->rng_state;
        item_of[i] = local_count;
        ++local_count;
    }
    if (local_count > 0u &&
        magpie_sample_local_frame_batch(model, items, local_count,
                                        scratch == NULL ? NULL : &scratch->batch,
                                        error, error_capacity) != 0) {
        /* A batched head fails for all of its requests or none: the failure is
         * in shared code, not in one request's data. */
        for (size_t i = 0; i < count; ++i) {
            if (item_of[i] != SIZE_MAX) results[i].failed = 1;
        }
        return -1;
    }

    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        int saw_eos = 0;
        size_t step_eos_frame = SIZE_MAX;
        if (item_of[i] != SIZE_MAX) {
            saw_eos = items[item_of[i]].saw_eos;
            step_eos_frame = items[item_of[i]].eos_frame;
        } else if (magpie_emit_projection(ctx, &saw_eos, &step_eos_frame,
                                          error, error_capacity) != 0) {
            results[i].failed = 1;
            continue;
        }
        ++ctx->predicted_stacks;
        ctx->step_eos_frame = step_eos_frame;
        const size_t after = magpie_frames_raw(ctx);
        results[i].frames_appended =
            (unsigned)(after > before[i] ? after - before[i] : 0u);
        results[i].eos_frame =
            (unsigned)(step_eos_frame == SIZE_MAX ? ctx->stacking : step_eos_frame);
        if (step_eos_frame != SIZE_MAX) ctx->eos_frame = step_eos_frame;
        if (saw_eos && ctx->predicted_stacks >= MAGPIE_EOS_STACK_FLOOR) {
            ctx->finished = 1;
        } else {
            ++ctx->step;
            if (ctx->step > ctx->max_steps) ctx->finished = 1;
        }
        results[i].eos = ctx->finished;
    }
    return 0;
}

/* ---- audio --------------------------------------------------------------- */

static int magpie_decode_audio(mynah_engine_ctx *ctx, size_t first_frame,
                               size_t frame_count, float **out_samples,
                               size_t *out_count, char *error, size_t error_capacity) {
    if (out_samples != NULL) *out_samples = NULL;
    if (out_count != NULL) *out_count = 0;
    if (ctx == NULL || out_samples == NULL || out_count == NULL) return -1;
    if (frame_count == 0u) return 0;
    /* Decode a bounded suffix, not the whole prefix.
     *
     * The audio decoder is causal throughout, so an output depends only on
     * inputs at or before it. Restarting the sequence a receptive field before
     * the first requested frame therefore reproduces those frames exactly, and
     * the extra samples in front are dropped here rather than by the caller.
     *
     * Re-decoding the whole prefix on every step would be quadratic: a 15 s
     * utterance decoded 26k frames instead of 320. */
    const size_t start = first_frame > MAGPIE_STREAM_CONTEXT_FRAMES
        ? first_frame - MAGPIE_STREAM_CONTEXT_FRAMES : 0u;
    const size_t span = first_frame + frame_count - start;
    if (span == 0u || ctx->codebooks == 0u ||
        span > SIZE_MAX / ctx->codebooks) {
        mynah_graph_error(error, error_capacity, "decode range is out of bounds");
        return -1;
    }
    if (ctx->codes == NULL ||
        first_frame + frame_count + ctx->stacking > ctx->max_raw_length) {
        mynah_graph_error(error, error_capacity, "decode range is out of bounds");
        return -1;
    }
    unsigned *frames = (unsigned *)calloc(ctx->codebooks * span, sizeof(*frames));
    if (frames == NULL) {
        mynah_graph_error(error, error_capacity, "out of memory preparing codec input");
        return -1;
    }
    for (size_t c = 0; c < ctx->codebooks; ++c) {
        memcpy(frames + c * span,
               ctx->codes + c * ctx->max_raw_length + ctx->stacking + start,
               span * sizeof(*frames));
    }
    float *audio = NULL;
    size_t produced = 0;
    const int decoded = mynah_nanocodec_decode(ctx->model, frames, span, &audio,
                                               &produced, error, error_capacity);
    free(frames);
    if (decoded != 0) return -1;
    /* The samples-per-frame ratio is a property of the pack, so take it from
     * what the decoder produced rather than assuming a rate. */
    const size_t skip_frames = first_frame - start;
    const size_t per_frame = produced / span;
    const size_t skip = skip_frames * per_frame;
    if (produced < skip) {
        free(audio);
        mynah_graph_error(error, error_capacity, "streamed codec output regressed");
        return -1;
    }
    if (skip > 0u) {
        memmove(audio, audio + skip, (produced - skip) * sizeof(float));
    }
    *out_samples = audio;
    *out_count = produced - skip;
    return 0;
}

/* ---- scratch ------------------------------------------------------------- */

static int magpie_scratch_new(const mynah_tts_model *model, mynah_engine_state *state,
                              size_t batch, mynah_engine_scratch **out,
                              char *error, size_t error_capacity) {
    (void)state;
    if (out == NULL) return -1;
    *out = NULL;
    if (model == NULL) return -1;
    mynah_engine_scratch *scratch = (mynah_engine_scratch *)calloc(1, sizeof(*scratch));
    if (scratch == NULL) {
        mynah_graph_error(error, error_capacity, "out of memory allocating batch scratch");
        return -1;
    }
    /* One quantized activation buffer for the widest projection in the graph.
     * Every request in a batch shares the model, so the widest k is a property
     * of the pack, not of the slots that happen to be live. */
    size_t k_max = (size_t)model->info.hidden_dim * 4u;
    mynah_tensor cross_q;
    char probe[256];
    probe[0] = '\0';
    if (mynah_tensor_get(model->tts, "decoder.layers.0.cross_attention.q_net.weight",
                         &cross_q, probe, sizeof(probe)) == 0 && cross_q.rank >= 1 &&
        cross_q.shape[0] > k_max) {
        k_max = cross_q.shape[0];
    }
    if (magpie_batch_scratch_init(&scratch->batch, batch, k_max,
                                  error, error_capacity) != 0) {
        free(scratch);
        return -1;
    }
    *out = scratch;
    return 0;
}

static void magpie_scratch_free(mynah_engine_scratch *scratch) {
    if (scratch == NULL) return;
    magpie_batch_scratch_free(&scratch->batch);
    free(scratch);
}

/* ---- debug dumps --------------------------------------------------------- */

static void magpie_dump_rows(const char *path, const float *values, size_t count) {
    FILE *file = fopen(path, "w");
    if (file == NULL) return;
    for (size_t i = 0; i < count; ++i) fprintf(file, "%.9g\n", (double)values[i]);
    fclose(file);
}

static void magpie_debug_dump(mynah_engine_ctx *ctx, const char *stage) {
    if (ctx == NULL || stage == NULL) return;
    const mynah_tts_model *model = ctx->model;
    const size_t width = model->info.hidden_dim;
    char scratch_error[256];
    scratch_error[0] = '\0';
    if (strcmp(stage, "encoder") == 0) {
        const char *path = getenv("MYNAH_DUMP_ENCODER");
        if (path == NULL || ctx->memory == NULL) return;
        magpie_dump_rows(path, ctx->memory, ctx->request->text_length * width);
        return;
    }
    if (strcmp(stage, "prefill") == 0 || strcmp(stage, "hidden") == 0) {
        const char *path = getenv(strcmp(stage, "prefill") == 0 ? "MYNAH_DUMP_PREFILL"
                                                                : "MYNAH_DUMP_HIDDEN");
        if (path == NULL || ctx->out_last == NULL) return;
        if (strcmp(stage, "hidden") == 0 && ctx->step != 1u) return;
        if (ctx->decoder_dev_last != NULL) {
            mynah_backend_d2h(model->backend, ctx->decoder_dev_last, ctx->out_last,
                              width, scratch_error, sizeof(scratch_error));
        }
        magpie_dump_rows(path, ctx->out_last, width);
        return;
    }
    if (strcmp(stage, "codes") == 0) {
        const char *path = getenv("MYNAH_DUMP_CODES");
        if (path == NULL || ctx->codes == NULL || ctx->predicted_stacks == 0u) return;
        FILE *file = fopen(path, "w");
        if (file == NULL) return;
        const size_t cb = ctx->codebooks;
        const size_t fs = ctx->stacking;
        fprintf(file, "[");
        for (size_t step = 0; step < ctx->predicted_stacks; ++step) {
            if (step > 0) fprintf(file, ",");
            fprintf(file, "[[");
            for (size_t c = 0; c < cb; ++c) {
                if (c > 0) fprintf(file, "],[");
                for (size_t f = 0; f < fs; ++f) {
                    if (f > 0) fprintf(file, ",");
                    fprintf(file, "%u",
                            ctx->codes[c * ctx->max_raw_length + (step + 1u) * fs + f]);
                }
            }
            fprintf(file, "]]");
        }
        fprintf(file, "]\n");
        fclose(file);
    }
}

/* ---- the vtable ---------------------------------------------------------- */

static const mynah_tts_engine magpie_engine = {
    "magpie",
    magpie_model_init,
    magpie_model_free,
    magpie_caps,
    magpie_ctx_new,
    magpie_prepare,
    magpie_reset,
    magpie_ctx_free,
    magpie_step_batch,
    magpie_emit_batch,
    magpie_frame_count,
    magpie_truncate,
    magpie_decode_audio,
    magpie_scratch_new,
    magpie_scratch_free,
    magpie_debug_dump,
};

const mynah_tts_engine *mynah_engine_magpie(void) {
    return &magpie_engine;
}

const mynah_tts_engine *mynah_engine_lookup(const char *name) {
    if (name == NULL) return NULL;
    if (strcmp(name, magpie_engine.name) == 0) return &magpie_engine;
    return NULL;
}
