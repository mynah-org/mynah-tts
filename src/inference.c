/* The batched inference driver: slots, admission-free batching, streaming
 * emission and the public synthesize entry points.
 *
 * Split out of graph.c by E1. This file owns request lifetime and knows nothing
 * about Magpie beyond the engine functions it calls; those calls are the list
 * that the engine vtable will replace (PLAN.md E1-1, and the mapping in
 * .work/engine-seam-refactor.md).
 */
#include "engine_magpie.h"
#include "graph.h"
#include "kernels.h"
#include "mynah_util.h"
#include "threads.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Streaming policy. These are Magpie's numbers and become engine capabilities
 * when the vtable lands: PocketTTS carries codec state instead of replaying
 * context, and at 12.5 Hz an emit threshold of 16 frames would be 1.28 s.
 * Measured in .work/pocket-tts-oracle.md, E2-3. */
/* Left context, in codec frames, that a streamed chunk must be decoded with.
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
 * measurable. */
#define STREAM_CONTEXT_FRAMES 32u

/* Frames to accumulate before emitting. The decode cost per chunk is dominated
 * by the fixed context above, so emitting every single step would pay it 16
 * times over for the same audio. At ~21.5 frames/s this is well under 1 s of
 * added latency to the first chunk, and none to the total. */
#define STREAM_EMIT_FRAMES 16u

static int emit_stream_samples(mynah_tts_audio_callback callback, void *user_data,
                               const float *samples, size_t count,
                               size_t chunk_samples, char *error,
                               size_t error_capacity) {
    if (callback == NULL || count == 0) return 0;
    size_t offset = 0;
    while (offset < count) {
        const size_t remaining = count - offset;
        const size_t chunk = remaining < chunk_samples ? remaining : chunk_samples;
        if (callback(samples + offset, chunk, user_data) != 0) {
            mynah_graph_error(error, error_capacity, "audio callback aborted streaming");
            return -1;
        }
        offset += chunk;
    }
    return 0;
}

/* One request in flight.
 *
 * Everything a request needs to advance one step lives here, so the driver can
 * hold several and step them together.  Nothing is shared between slots except
 * the read-only model, which is what makes the batched projections safe. */
typedef struct {
    /* request and sinks */
    const mynah_tts_request *request;
    float **samples;
    size_t *sample_count;
    mynah_tts_audio_callback callback;
    void *user_data;
    size_t chunk_samples;
    char *error;
    size_t error_capacity;
    /* owned state */
    float *memory;
    unsigned *codes;
    float *out_last;
    float *audio_row;
    decoder_cache cache;
    local_frame_state local_state;
    float *decoder_dev_last;
    /* derived constants */
    const float *context;
    size_t context_length;
    size_t max_steps;
    size_t max_raw_length;
    size_t min_raw_length;
    float temperature;
    unsigned topk;
    /* progress */
    uint64_t rng_state;
    size_t step;              /* 1-based stacked position of the next step */
    size_t predicted_stacks;
    size_t eos_frame;
    size_t streamed_samples;
    size_t streamed_frames;
    int active;
    int failed;
} synth_slot;

static void slot_release(synth_slot *slot) {
    if (slot == NULL) return;
    magpie_local_frame_state_free(&slot->local_state);
    magpie_decoder_cache_free(&slot->cache);
    free(slot->memory);
    free(slot->codes);
    free(slot->out_last);
    free(slot->audio_row);
    slot->memory = NULL;
    slot->codes = NULL;
    slot->out_last = NULL;
    slot->audio_row = NULL;
}

static int slot_fail(synth_slot *slot, const char *message) {
    if (message != NULL) mynah_graph_error(slot->error, slot->error_capacity, message);
    slot->failed = 1;
    slot->active = 0;
    return -1;
}

/* Everything before the first autoregressive step: text encoding, the constant
 * cross-attention cache, and the baked speaker context prefill.  These are
 * multi-row matmuls that BLAS already runs efficiently and that share nothing
 * between requests, so they stay per slot. */
static int slot_prepare(const mynah_tts_model *model, synth_slot *slot, int dump) {
    const mynah_tts_request *request = slot->request;
    char *error = slot->error;
    const size_t error_capacity = slot->error_capacity;
    const size_t width = model->info.hidden_dim;

    if (slot->samples != NULL) *slot->samples = NULL;
    if (slot->sample_count != NULL) *slot->sample_count = 0;
    if (request == NULL ||
        ((slot->samples == NULL || slot->sample_count == NULL) && slot->callback == NULL) ||
        error == NULL || error_capacity == 0 || request->text_ids == NULL ||
        request->text_length == 0 || (slot->callback != NULL && slot->chunk_samples == 0)) {
        return slot_fail(slot, "invalid synthesis request");
    }
    if (request->speaker >= model->info.speaker_count) {
        return slot_fail(slot, "speaker index is outside the model");
    }
    if (magpie_encode_text(model, request->text_ids, request->text_length, &slot->memory,
                    error, error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    if (dump && getenv("MYNAH_DUMP_ENCODER") != NULL) {
        FILE *ef = fopen(getenv("MYNAH_DUMP_ENCODER"), "w");
        if (ef != NULL) {
            for (size_t t = 0; t < request->text_length; ++t)
                for (size_t d = 0; d < width; ++d)
                    fprintf(ef, "%.9g\n", (double)slot->memory[t * width + d]);
            fclose(ef);
        }
    }
    mynah_tensor context_tensor;
    if (mynah_tensor_get(model->tts, "baked_context_embedding.weight", &context_tensor,
               error, error_capacity) != 0 || context_tensor.rank != 2 ||
        context_tensor.shape[1] % width != 0 ||
        request->speaker >= context_tensor.shape[0]) {
        return slot_fail(slot, "baked speaker context is invalid");
    }
    slot->context_length = context_tensor.shape[1] / width;
    slot->context = context_tensor.data +
                    (size_t)request->speaker * context_tensor.shape[1];
    slot->max_steps = request->max_steps == 0
        ? (model->info.max_decoder_steps + model->info.frame_stacking_factor - 1u) /
          model->info.frame_stacking_factor
        : request->max_steps;
    slot->max_raw_length = (slot->max_steps + 1u) * model->info.frame_stacking_factor;
    slot->min_raw_length = model->info.min_generated_frames;
    slot->codes = (unsigned *)calloc(model->info.codebook_count * slot->max_raw_length,
                                     sizeof(*slot->codes));
    slot->out_last = mynah_alloc_floats(width, error, error_capacity);
    slot->audio_row = mynah_alloc_floats(width, error, error_capacity);
    if (slot->codes == NULL || slot->out_last == NULL || slot->audio_row == NULL) {
        return slot_fail(slot, "out of memory preparing the request");
    }
    for (size_t c = 0; c < model->info.codebook_count; ++c) {
        for (size_t t = 0; t < model->info.frame_stacking_factor; ++t) {
            slot->codes[c * slot->max_raw_length + t] = model->info.codebook_size;
        }
    }
    slot->temperature = request->temperature;
    if (!(slot->temperature >= 0.0f)) slot->temperature = model->info.default_temperature;
    slot->topk = request->topk == 0 ? model->info.default_topk : request->topk;
    slot->rng_state = request->seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : request->seed;
    slot->eos_frame = SIZE_MAX;
    if (magpie_decoder_cache_init(model, &slot->cache, slot->memory, request->text_length,
                           slot->context_length + slot->max_steps + 2u,
                           error, error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    slot->decoder_dev_last = NULL;
    if (magpie_decoder_run(model, &slot->cache, slot->context, slot->context_length,
                    slot->out_last,
                    request->use_local_transformer ? &slot->decoder_dev_last : NULL,
                    error, error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    if (dump && getenv("MYNAH_DUMP_PREFILL") != NULL) {
        FILE *pf = fopen(getenv("MYNAH_DUMP_PREFILL"), "w");
        if (pf != NULL) {
            if (slot->decoder_dev_last != NULL)
                mynah_backend_d2h(model->backend, slot->decoder_dev_last, slot->out_last,
                                  width, error, error_capacity);
            for (size_t d = 0; d < width; ++d)
                fprintf(pf, "%.9g\n", (double)slot->out_last[d]);
            fclose(pf);
        }
    }
    if (request->use_local_transformer) {
        const size_t vocab = model->info.audio_vocab_size;
        const size_t top_count = slot->temperature > 0.0f && slot->topk > 1u
            ? (slot->topk < vocab ? slot->topk : vocab) : 0u;
        if (magpie_local_frame_state_init(model, &slot->local_state, top_count,
                                   error, error_capacity) != 0) {
            return slot_fail(slot, NULL);
        }
    }
    slot->step = 1u;
    slot->active = 1;
    return 0;
}

/* The decoder row is produced; dump it if asked. */
static void slot_dump_hidden(const mynah_tts_model *model, synth_slot *slot, int dump) {
    const size_t width = model->info.hidden_dim;
    if (!dump || slot->step != 1u || getenv("MYNAH_DUMP_HIDDEN") == NULL) return;
    FILE *hf = fopen(getenv("MYNAH_DUMP_HIDDEN"), "w");
    if (hf == NULL) return;
    if (slot->decoder_dev_last != NULL)
        mynah_backend_d2h(model->backend, slot->decoder_dev_last, slot->out_last,
                          width, slot->error, slot->error_capacity);
    for (size_t d = 0; d < width; ++d)
        fprintf(hf, "%.9g\n", (double)slot->out_last[d]);
    fclose(hf);
}

/* Streaming, the EOS decision and the step counter for one slot, once its
 * frame has been sampled.  Clears `active` when the request is finished. */
static int slot_advance(const mynah_tts_model *model, synth_slot *slot,
                        int saw_eos, size_t step_eos_frame) {
    const mynah_tts_request *request = slot->request;
    char *error = slot->error;
    const size_t error_capacity = slot->error_capacity;
    const size_t width = model->info.hidden_dim;
    const size_t raw_length = slot->step * model->info.frame_stacking_factor;
    if (!request->use_local_transformer) {
        mynah_tensor projection;
        mynah_tensor bias;
        if (mynah_tensor_get(model->tts, "final_proj.weight", &projection, error, error_capacity) != 0 ||
            mynah_tensor_get(model->tts, "final_proj.bias", &bias, error, error_capacity) != 0) {
            return slot_fail(slot, NULL);
        }
        const size_t streams = magpie_stacked_stream_count(model);
        const int forbid_eos =
            slot->predicted_stacks * model->info.frame_stacking_factor < slot->min_raw_length;
        const unsigned eos_id = model->info.audio_eos_id;
        const size_t vocab = model->info.audio_vocab_size;
        const float *last = slot->out_last;
        for (size_t stream = 0; stream < streams; ++stream) {
            unsigned value = 0;
            float best = -FLT_MAX;
            for (size_t candidate = 0; candidate < vocab; ++candidate) {
                /* Same forbidden-token rule as the local path: real codes
                 * always, AUDIO_EOS unless too early, nothing else. */
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
                saw_eos = 1;
                const size_t frame = stream / model->info.codebook_count;
                if (frame < step_eos_frame) step_eos_frame = frame;
            }
            const unsigned store = value < model->info.codebook_size ? value : 0u;
            const size_t fs = stream / model->info.codebook_count;
            const size_t codebook = stream % model->info.codebook_count;
            slot->codes[codebook * slot->max_raw_length + raw_length + fs] = store;
        }
    }
    ++slot->predicted_stacks;
    if (slot->callback != NULL) {
        size_t stream_raw = slot->predicted_stacks * model->info.frame_stacking_factor;
        if (step_eos_frame != SIZE_MAX) {
            stream_raw = (slot->predicted_stacks - 1u) * model->info.frame_stacking_factor +
                         step_eos_frame;
        }
        const int finishing = saw_eos || slot->step >= slot->max_steps;
        const size_t fresh = stream_raw > slot->streamed_frames
            ? stream_raw - slot->streamed_frames : 0u;
        if (fresh > 0u && (finishing || fresh >= STREAM_EMIT_FRAMES)) {
            /* Decode a bounded suffix, not the whole prefix.
             *
             * The codec decoder is causal throughout, so an output depends only
             * on inputs at or before it.  Restarting the sequence
             * STREAM_CONTEXT_FRAMES before the first unsent frame therefore
             * reproduces those frames exactly -- only the outputs inside the
             * receptive field are affected by the missing history, and those
             * have already been sent.
             *
             * Re-decoding the whole prefix every step, as this did, is
             * quadratic: a 15 s utterance decoded 26k frames instead of 320 and
             * streaming ran 30x slower than the same request offline. */
            const size_t start = slot->streamed_frames > STREAM_CONTEXT_FRAMES
                ? slot->streamed_frames - STREAM_CONTEXT_FRAMES : 0u;
            const size_t span = stream_raw - start;
            unsigned *stream_codes = (unsigned *)calloc(
                model->info.codebook_count * span, sizeof(*stream_codes));
            if (stream_codes == NULL) {
                return slot_fail(slot, "out of memory preparing streamed codes");
            }
            for (size_t c = 0; c < model->info.codebook_count; ++c) {
                memcpy(stream_codes + c * span,
                       slot->codes + c * slot->max_raw_length +
                           model->info.frame_stacking_factor + start,
                       span * sizeof(*stream_codes));
            }
            float *stream_audio = NULL;
            size_t stream_count = 0;
            if (mynah_nanocodec_decode(model, stream_codes, span, &stream_audio,
                             &stream_count, error, error_capacity) != 0) {
                free(stream_codes);
                return slot_fail(slot, NULL);
            }
            /* The codec's samples-per-frame is a property of the pack, so take
             * it from what it produced rather than assuming a rate. */
            const size_t skip_frames = slot->streamed_frames - start;
            const size_t per_frame = span > 0u ? stream_count / span : 0u;
            const size_t skip = skip_frames * per_frame;
            if (stream_count < skip ||
                emit_stream_samples(slot->callback, slot->user_data,
                                    stream_audio + skip, stream_count - skip,
                                    slot->chunk_samples, error, error_capacity) != 0) {
                const int regressed = stream_count < skip;
                free(stream_audio);
                free(stream_codes);
                return slot_fail(slot, regressed ? "streamed codec output regressed" : NULL);
            }
            slot->streamed_samples += stream_count - skip;
            slot->streamed_frames = stream_raw;
            free(stream_audio);
            free(stream_codes);
        }
    }
    if (step_eos_frame != SIZE_MAX) slot->eos_frame = step_eos_frame;
    if (saw_eos && slot->predicted_stacks >= 4u) {
        slot->active = 0;
        return 0;
    }
    ++slot->step;
    if (slot->step > slot->max_steps) slot->active = 0;
    return 0;
}

/* Turn the accumulated codes into audio for the offline sink. */
static int slot_finalize(const mynah_tts_model *model, synth_slot *slot, int dump) {
    char *error = slot->error;
    const size_t error_capacity = slot->error_capacity;
    const size_t generated_stacks = slot->predicted_stacks;
    size_t generated_raw = generated_stacks * model->info.frame_stacking_factor;
    if (slot->eos_frame != SIZE_MAX && generated_stacks > 0) {
        generated_raw = (generated_stacks - 1u) * model->info.frame_stacking_factor +
                        slot->eos_frame;
    }
    if (dump && getenv("MYNAH_DUMP_CODES") != NULL && generated_stacks > 0) {
        FILE *dumpf = fopen(getenv("MYNAH_DUMP_CODES"), "w");
        if (dumpf != NULL) {
            const size_t cb = model->info.codebook_count;
            const size_t fs = model->info.frame_stacking_factor;
            fprintf(dumpf, "[");
            for (size_t step = 0; step < generated_stacks; ++step) {
                if (step > 0) fprintf(dumpf, ",");
                fprintf(dumpf, "[[");
                for (size_t c = 0; c < cb; ++c) {
                    if (c > 0) fprintf(dumpf, "],[");
                    for (size_t f = 0; f < fs; ++f) {
                        if (f > 0) fprintf(dumpf, ",");
                        fprintf(dumpf, "%u",
                                slot->codes[c * slot->max_raw_length + (step + 1u) * fs + f]);
                    }
                }
                fprintf(dumpf, "]]");
            }
            fprintf(dumpf, "]\n");
            fclose(dumpf);
        }
    }
    if (slot->failed) return -1;
    if (generated_raw == 0) {
        return slot_fail(slot, "decoder generated no audio frames");
    }
    if (slot->callback != NULL) {
        if (slot->streamed_samples == 0) {
            return slot_fail(slot, "stream produced no audio frames");
        }
        return 0;
    }
    unsigned *predicted_codes = (unsigned *)calloc(
        model->info.codebook_count * generated_raw, sizeof(*predicted_codes));
    if (predicted_codes == NULL) {
        return slot_fail(slot, "out of memory copying generated codes");
    }
    for (size_t c = 0; c < model->info.codebook_count; ++c) {
        memcpy(predicted_codes + c * generated_raw,
               slot->codes + c * slot->max_raw_length + model->info.frame_stacking_factor,
               generated_raw * sizeof(*predicted_codes));
    }
    const int result = mynah_nanocodec_decode(model, predicted_codes, generated_raw,
                                    slot->samples, slot->sample_count,
                                    error, error_capacity);
    free(predicted_codes);
    if (result != 0) return slot_fail(slot, NULL);
    return 0;
}

/* Step every live slot together.
 *
 * This is continuous in the sense that matters here: a slot that reaches EOS
 * drops out of the batch immediately and the rest keep going at the smaller
 * width, rather than the whole group waiting for the longest request. */
static int synthesize_slots(const mynah_tts_model *model, synth_slot *slots,
                            size_t count) {
    if (count == 0u) return 0;
    if (count > MYNAH_MAX_BATCH) return -1;
    const int dump_all = count == 1u;
    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double t_start = timing ? mynah_phase_seconds() : 0.0;
    double t_prep = t_start, t_ar = t_start;

    for (size_t i = 0; i < count; ++i) {
        slot_prepare(model, &slots[i], dump_all);
    }
    if (timing) t_prep = mynah_phase_seconds();

    /* One quantized activation buffer for the widest projection in the graph. */
    batch_scratch scratch;
    size_t k_max = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!slots[i].active) continue;
        const decoder_cache *c = &slots[i].cache;
        if (c->width > k_max) k_max = c->width;
        if (c->ffn_width > k_max) k_max = c->ffn_width;
        if (c->xattn_width > k_max) k_max = c->xattn_width;
    }
    char scratch_error[256];
    scratch_error[0] = '\0';
    if (magpie_batch_scratch_init(&scratch, count, k_max, scratch_error,
                           sizeof(scratch_error)) != 0) {
        for (size_t i = 0; i < count; ++i) {
            if (slots[i].active) slot_fail(&slots[i], scratch_error);
        }
    }

    decoder_cache *step_caches[MYNAH_MAX_BATCH];
    const float *step_inputs[MYNAH_MAX_BATCH];
    float *step_outs[MYNAH_MAX_BATCH];
    size_t step_slot[MYNAH_MAX_BATCH];
    local_batch_item local_items[MYNAH_MAX_BATCH];
    size_t local_slot[MYNAH_MAX_BATCH];
    for (;;) {
        size_t live = 0;
        for (size_t i = 0; i < count; ++i) {
            synth_slot *slot = &slots[i];
            if (!slot->active) continue;
            if (magpie_embed_audio_frame(model, slot->codes, slot->max_raw_length,
                                  slot->step - 1u, slot->audio_row,
                                  slot->error, slot->error_capacity) != 0) {
                slot_fail(slot, NULL);
                continue;
            }
            slot->decoder_dev_last = NULL;
            step_slot[live] = i;
            step_caches[live] = &slot->cache;
            step_inputs[live] = slot->audio_row;
            step_outs[live] = slot->out_last;
            ++live;
        }
        if (live == 0u) break;
        if (live == 1u) {
            /* Alone: go through decoder_run so a resident GPU step is still
             * available.  It reduces to the same batched step on CPU. */
            synth_slot *slot = &slots[step_slot[0]];
            if (magpie_decoder_run(model, &slot->cache, slot->audio_row, 1u, slot->out_last,
                            slot->request->use_local_transformer ? &slot->decoder_dev_last
                                                                 : NULL,
                            slot->error, slot->error_capacity) != 0) {
                slot_fail(slot, NULL);
                continue;
            }
        } else if (magpie_decoder_step_batch(model, step_caches, step_inputs, step_outs,
                                      live, &scratch, slots[step_slot[0]].error,
                                      slots[step_slot[0]].error_capacity) != 0) {
            /* A batched step fails for all its slots or none: the failure is in
             * shared code, not in one request's data. */
            for (size_t j = 0; j < live; ++j) {
                synth_slot *slot = &slots[step_slot[j]];
                if (j > 0) {
                    mynah_graph_error(slot->error, slot->error_capacity,
                                "batched decoder step failed");
                }
                slot_fail(slot, NULL);
            }
            continue;
        }
        /* Sample the frame.  The local transformer is read once per stacked
         * stream, so a decode step walks its weights sixteen times: batching it
         * across slots matters more than batching the decoder itself. */
        size_t local_count = 0;
        for (size_t j = 0; j < live; ++j) {
            synth_slot *slot = &slots[step_slot[j]];
            if (!slot->active) continue;
            slot_dump_hidden(model, slot, dump_all);
            if (!slot->request->use_local_transformer) continue;
            local_batch_item *item = &local_items[local_count];
            memset(item, 0, sizeof(*item));
            item->state = &slot->local_state;
            item->decoder_last = slot->out_last;
            item->decoder_dev_last = slot->decoder_dev_last;
            item->codes = slot->codes;
            item->raw_offset = slot->step * model->info.frame_stacking_factor;
            item->code_stride = slot->max_raw_length;
            item->generated_raw_length =
                slot->predicted_stacks * model->info.frame_stacking_factor;
            item->min_raw_length = slot->min_raw_length;
            item->temperature = slot->temperature;
            item->topk = slot->topk;
            item->rng_state = &slot->rng_state;
            local_slot[local_count] = step_slot[j];
            ++local_count;
        }
        if (local_count > 0u) {
            char local_error[256];
            local_error[0] = '\0';
            if (magpie_sample_local_frame_batch(model, local_items, local_count, &scratch,
                                         local_error, sizeof(local_error)) != 0) {
                for (size_t j = 0; j < local_count; ++j) {
                    slot_fail(&slots[local_slot[j]], local_error);
                }
                continue;
            }
        }
        for (size_t j = 0; j < live; ++j) {
            synth_slot *slot = &slots[step_slot[j]];
            if (!slot->active) continue;
            int saw_eos = 0;
            size_t step_eos_frame = SIZE_MAX;
            if (slot->request->use_local_transformer) {
                for (size_t q = 0; q < local_count; ++q) {
                    if (local_slot[q] != step_slot[j]) continue;
                    saw_eos = local_items[q].saw_eos;
                    step_eos_frame = local_items[q].eos_frame;
                    break;
                }
            }
            slot_advance(model, slot, saw_eos, step_eos_frame);
        }
    }
    if (timing) t_ar = mynah_phase_seconds();

    int result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (slot_finalize(model, &slots[i], dump_all) != 0) result = -1;
    }
    if (timing) {
        fprintf(stderr, "phase: prep=%.3fs ar=%.3fs codec=%.3fs (requests=%zu)\n",
                t_prep - t_start, t_ar - t_prep, mynah_phase_seconds() - t_ar, count);
    }
    magpie_batch_scratch_free(&scratch);
    for (size_t i = 0; i < count; ++i) {
        slot_release(&slots[i]);
        if (!slots[i].failed && slots[i].error != NULL && slots[i].error_capacity > 0)
            slots[i].error[0] = '\0';
    }
    return result;
}

int mynah_graph_synthesize_jobs(const mynah_tts_model *model,
                                mynah_graph_job *jobs, size_t count) {
    if (model == NULL || jobs == NULL) return -1;
    if (count == 0u) return 0;
    if (count > MYNAH_MAX_BATCH) return -1;
    synth_slot slots[MYNAH_MAX_BATCH];
    memset(slots, 0, sizeof(slots));
    for (size_t i = 0; i < count; ++i) {
        slots[i].request = jobs[i].request;
        slots[i].samples = jobs[i].samples;
        slots[i].sample_count = jobs[i].sample_count;
        slots[i].callback = jobs[i].callback;
        slots[i].user_data = jobs[i].user_data;
        slots[i].chunk_samples = jobs[i].chunk_samples;
        slots[i].error = jobs[i].error;
        slots[i].error_capacity = jobs[i].error_capacity;
        slots[i].eos_frame = SIZE_MAX;
    }
    const int result = synthesize_slots(model, slots, count);
    for (size_t i = 0; i < count; ++i) jobs[i].result = slots[i].failed ? -1 : 0;
    return result;
}

int mynah_graph_synthesize_stream(const mynah_tts_model *model,
                                  const mynah_tts_request *request,
                                  float **samples, size_t *sample_count,
                                  mynah_tts_audio_callback callback,
                                  void *user_data, size_t chunk_samples,
                                  char *error, size_t error_capacity) {
    if (samples != NULL) *samples = NULL;
    if (sample_count != NULL) *sample_count = 0;
    if (model == NULL || error == NULL || error_capacity == 0) {
        mynah_graph_error(error, error_capacity, "invalid synthesis request");
        return -1;
    }
    mynah_graph_job job;
    memset(&job, 0, sizeof(job));
    job.request = request;
    job.samples = samples;
    job.sample_count = sample_count;
    job.callback = callback;
    job.user_data = user_data;
    job.chunk_samples = chunk_samples;
    job.error = error;
    job.error_capacity = error_capacity;
    return mynah_graph_synthesize_jobs(model, &job, 1u);
}

size_t mynah_tts_max_batch(void) {
    return MYNAH_GRAPH_MAX_JOBS;
}

int mynah_tts_synthesize_batch(const mynah_tts_model *model,
                               mynah_tts_batch_job *jobs, size_t count) {
    if (model == NULL || jobs == NULL) return -1;
    if (count == 0u) return 0;
    if (count > MYNAH_GRAPH_MAX_JOBS) return -1;
    mynah_graph_job graph_jobs[MYNAH_GRAPH_MAX_JOBS];
    memset(graph_jobs, 0, sizeof(graph_jobs));
    for (size_t i = 0; i < count; ++i) {
        graph_jobs[i].request = jobs[i].request;
        graph_jobs[i].samples = jobs[i].samples;
        graph_jobs[i].sample_count = jobs[i].sample_count;
        graph_jobs[i].error = jobs[i].error;
        graph_jobs[i].error_capacity = jobs[i].error_capacity;
    }
    const int result = mynah_graph_synthesize_jobs(model, graph_jobs, count);
    for (size_t i = 0; i < count; ++i) jobs[i].result = graph_jobs[i].result;
    return result;
}

int mynah_tts_synthesize(const mynah_tts_model *model,
                         const mynah_tts_request *request,
                         float **samples, size_t *sample_count,
                         char *error, size_t error_capacity) {
    return mynah_graph_synthesize_stream(model, request, samples, sample_count,
                                          NULL, NULL, 0, error, error_capacity);
}

void mynah_tts_free_samples(float *samples) {
    free(samples);
}
