/* The batched inference driver: slots, admission-free batching, streaming
 * emission and the public synthesize entry points.
 *
 * This file is engine-agnostic. Every model-shaped decision -- how many audio
 * frames a step yields, when an end-of-speech token is allowed to stop
 * generation, how a frame range turns into PCM -- reaches the driver either as
 * a capability (`mynah_engine_caps`) or as a step result, and never as a
 * question about which engine is loaded. What is left here is request
 * lifetime, the batching loop, the sinks and the streaming emit policy.
 */
#include "graph.h"
#include "mynah_tts_internal.h"
#include "mynah_util.h"
#include "tts_engine.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
 * The request's generation state lives in the engine context; what the driver
 * keeps is the sink it writes to and how much of the engine's frame history it
 * has already turned into audio. */
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
    /* generation state, owned by the engine */
    mynah_engine_ctx *ctx;
    /* progress */
    size_t streamed_samples;
    size_t streamed_frames;
    int active;
    int failed;
} synth_slot;

static int slot_fail(synth_slot *slot, const char *message) {
    if (message != NULL) mynah_graph_error(slot->error, slot->error_capacity, message);
    slot->failed = 1;
    slot->active = 0;
    return -1;
}

/* Validate the request and the sink, then hand everything else to the engine.
 * After this the context is ready for its first step. */
static int slot_start(const mynah_tts_engine *engine, const mynah_tts_model *model,
                      mynah_engine_state *state, const mynah_engine_caps *caps,
                      synth_slot *slot, int dump) {
    const mynah_tts_request *request = slot->request;

    if (slot->samples != NULL) *slot->samples = NULL;
    if (slot->sample_count != NULL) *slot->sample_count = 0;
    if (request == NULL ||
        ((slot->samples == NULL || slot->sample_count == NULL) && slot->callback == NULL) ||
        slot->error == NULL || slot->error_capacity == 0 || request->text_ids == NULL ||
        request->text_length == 0 || (slot->callback != NULL && slot->chunk_samples == 0)) {
        return slot_fail(slot, "invalid synthesis request");
    }
    const size_t max_steps = request->max_steps == 0u
        ? caps->default_max_steps : request->max_steps;
    if (engine->ctx_new(model, state, request, max_steps, request->seed, &slot->ctx,
                        slot->error, slot->error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    if (engine->prepare(slot->ctx, slot->error, slot->error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    if (dump && engine->debug_dump != NULL) {
        engine->debug_dump(slot->ctx, "encoder");
        engine->debug_dump(slot->ctx, "prefill");
    }
    slot->active = 1;
    return 0;
}

/* Turn whatever the engine has appended into streamed PCM.
 *
 * The frame history is opaque and monotonic, so the driver only tracks how far
 * it has got and asks for the rest. A step that appended less than a full
 * step's worth of frames is a boundary -- the engine cut the window short --
 * and is flushed immediately rather than waiting for the emit threshold. */
static int slot_stream(const mynah_tts_engine *engine, const mynah_engine_caps *caps,
                       synth_slot *slot, const mynah_engine_step_result *result) {
    if (slot->callback == NULL) return 0;
    const size_t frames = engine->frame_count(slot->ctx);
    const int finishing = result->eos || result->frames_appended < caps->frames_per_step;
    const size_t fresh = frames > slot->streamed_frames
        ? frames - slot->streamed_frames : 0u;
    if (fresh == 0u || !(finishing || fresh >= caps->audio_emit_frames)) return 0;

    float *audio = NULL;
    size_t produced = 0;
    if (engine->decode_audio(slot->ctx, slot->streamed_frames, fresh, &audio, &produced,
                             slot->error, slot->error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    if (emit_stream_samples(slot->callback, slot->user_data, audio, produced,
                            slot->chunk_samples, slot->error, slot->error_capacity) != 0) {
        free(audio);
        return slot_fail(slot, NULL);
    }
    slot->streamed_samples += produced;
    slot->streamed_frames = frames;
    free(audio);
    return 0;
}

/* Close the sequence and, for the offline sink, decode all of it. */
static int slot_finalize(const mynah_tts_engine *engine, synth_slot *slot, int dump) {
    if (slot->ctx != NULL) {
        engine->truncate(slot->ctx, engine->frame_count(slot->ctx));
        if (dump && engine->debug_dump != NULL) engine->debug_dump(slot->ctx, "codes");
    }
    if (slot->failed) return -1;
    const size_t frames = slot->ctx != NULL ? engine->frame_count(slot->ctx) : 0u;
    if (frames == 0u) {
        return slot_fail(slot, "decoder generated no audio frames");
    }
    if (slot->callback != NULL) {
        if (slot->streamed_samples == 0) {
            return slot_fail(slot, "stream produced no audio frames");
        }
        return 0;
    }
    float *audio = NULL;
    size_t produced = 0;
    if (engine->decode_audio(slot->ctx, 0u, frames, &audio, &produced,
                             slot->error, slot->error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    *slot->samples = audio;
    *slot->sample_count = produced;
    return 0;
}

/* Step every live slot together.
 *
 * This is continuous in the sense that matters here: a slot that reaches the
 * end drops out of the batch immediately and the rest keep going at the smaller
 * width, rather than the whole group waiting for the longest request. */
static int synthesize_slots(const mynah_tts_model *model, synth_slot *slots,
                            size_t count) {
    if (count == 0u) return 0;
    if (count > MYNAH_GRAPH_MAX_JOBS) return -1;

    const mynah_tts_engine *engine = mynah_engine_lookup(model->info.engine);
    if (engine == NULL) {
        for (size_t i = 0; i < count; ++i) {
            slot_fail(&slots[i], "model.json names an engine this build does not have");
        }
        return -1;
    }
    char shared_error[256];
    shared_error[0] = '\0';
    mynah_engine_state *state = NULL;
    if (engine->model_init(model, &state, shared_error, sizeof(shared_error)) != 0) {
        for (size_t i = 0; i < count; ++i) slot_fail(&slots[i], shared_error);
        return -1;
    }
    mynah_engine_caps caps;
    memset(&caps, 0, sizeof(caps));
    if (engine->caps(model, state, &caps) != 0 || caps.frames_per_step == 0u ||
        count > caps.max_batch) {
        for (size_t i = 0; i < count; ++i) {
            slot_fail(&slots[i], "the engine cannot serve this batch");
        }
        engine->model_free(state);
        return -1;
    }

    const int dump_all = count == 1u;
    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double t_start = timing ? mynah_phase_seconds() : 0.0;
    double t_prep = t_start, t_ar = t_start;

    for (size_t i = 0; i < count; ++i) {
        slot_start(engine, model, state, &caps, &slots[i], dump_all);
    }
    if (timing) t_prep = mynah_phase_seconds();

    mynah_engine_scratch *scratch = NULL;
    if (engine->scratch_new(model, state, count, &scratch, shared_error,
                            sizeof(shared_error)) != 0) {
        for (size_t i = 0; i < count; ++i) {
            if (slots[i].active) slot_fail(&slots[i], shared_error);
        }
    }

    mynah_engine_ctx *step_ctxs[MYNAH_GRAPH_MAX_JOBS];
    size_t step_slot[MYNAH_GRAPH_MAX_JOBS];
    mynah_engine_step_result results[MYNAH_GRAPH_MAX_JOBS];
    for (;;) {
        size_t live = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!slots[i].active) continue;
            step_slot[live] = i;
            step_ctxs[live] = slots[i].ctx;
            ++live;
        }
        if (live == 0u) break;

        shared_error[0] = '\0';
        if (engine->step_batch(step_ctxs, live, scratch, shared_error,
                               sizeof(shared_error)) != 0) {
            /* A batched step fails for all its slots or none: the failure is in
             * shared code, not in one request's data. */
            for (size_t j = 0; j < live; ++j) {
                slot_fail(&slots[step_slot[j]], shared_error);
            }
            continue;
        }
        if (dump_all && engine->debug_dump != NULL) {
            for (size_t j = 0; j < live; ++j) engine->debug_dump(step_ctxs[j], "hidden");
        }

        memset(results, 0, sizeof(results));
        shared_error[0] = '\0';
        if (engine->emit_batch(step_ctxs, live, results, scratch, shared_error,
                               sizeof(shared_error)) != 0) {
            for (size_t j = 0; j < live; ++j) {
                if (results[j].failed) slot_fail(&slots[step_slot[j]], shared_error);
            }
            continue;
        }
        for (size_t j = 0; j < live; ++j) {
            synth_slot *slot = &slots[step_slot[j]];
            if (results[j].failed) {
                slot_fail(slot, shared_error);
                continue;
            }
            if (slot_stream(engine, &caps, slot, &results[j]) != 0) continue;
            if (results[j].eos) slot->active = 0;
        }
    }
    if (timing) t_ar = mynah_phase_seconds();

    int result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (slot_finalize(engine, &slots[i], dump_all) != 0) result = -1;
    }
    if (timing) {
        fprintf(stderr, "phase: prep=%.3fs ar=%.3fs codec=%.3fs (requests=%zu)\n",
                t_prep - t_start, t_ar - t_prep, mynah_phase_seconds() - t_ar, count);
    }
    engine->scratch_free(scratch);
    for (size_t i = 0; i < count; ++i) {
        engine->ctx_free(slots[i].ctx);
        slots[i].ctx = NULL;
        if (!slots[i].failed && slots[i].error != NULL && slots[i].error_capacity > 0)
            slots[i].error[0] = '\0';
    }
    engine->model_free(state);
    return result;
}

int mynah_graph_synthesize_jobs(const mynah_tts_model *model,
                                mynah_graph_job *jobs, size_t count) {
    if (model == NULL || jobs == NULL) return -1;
    if (count == 0u) return 0;
    if (count > MYNAH_GRAPH_MAX_JOBS) return -1;
    synth_slot slots[MYNAH_GRAPH_MAX_JOBS];
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
