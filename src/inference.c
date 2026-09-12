/* The batched inference driver: slots, continuous admission, streaming
 * emission and the public synthesize entry points.
 *
 * This file is engine-agnostic. Every model-shaped decision -- how many audio
 * frames a step yields, when an end-of-speech token is allowed to stop
 * generation, how a frame range turns into PCM -- reaches the driver either as
 * a capability (`mynah_engine_caps`) or as a step result, and never as a
 * question about which engine is loaded. What is left here is request
 * lifetime, the batching loop, the sinks and the streaming emit policy.
 *
 * There is one loop, and it is a service loop. A fixed array of jobs is served
 * by an array-shaped sink that hands out its N requests and then says "no
 * more"; a server is served by a sink that keeps saying "here is another one".
 * Both walk the same admission block at the top of the same `for(;;)`, which is
 * why an offline batch and a live stream cannot drift apart (CLAUDE.md rule 7).
 */
#include "graph.h"
#include "mynah_tts_internal.h"
#include "mynah_util.h"
#include "tts_engine.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The default `decode_audio_batch`: one `decode_audio` per context.
 *
 * It is deliberately the only door the driver uses. The gang is formed above
 * this function whether or not the engine can exploit it, so an engine landing
 * a batched codec later changes exactly one pointer in its vtable and nothing
 * in the policy that decides who is in the gang -- which is the landing order
 * this seam was shaped for.
 *
 * Per-context failure is kept per context in both paths: the loop below marks
 * failed[i] and keeps going, so a request asking for a range its engine cannot
 * serve does not cost its neighbours their audio. The first failure's message
 * is the one kept, because it is the one that actually describes a failure;
 * the driver reports it to every context it marked. */
int mynah_engine_decode_gang(const mynah_tts_engine *engine,
                             mynah_engine_ctx *const *ctxs, size_t count,
                             const size_t *first_frame, const size_t *frame_count,
                             float **out_samples, size_t *out_count, int *failed,
                             mynah_engine_scratch *scratch,
                             char *error, size_t error_capacity) {
    if (engine == NULL || ctxs == NULL || first_frame == NULL ||
        frame_count == NULL || out_samples == NULL || out_count == NULL ||
        failed == NULL) {
        mynah_graph_error(error, error_capacity, "invalid decode gang");
        return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        out_samples[i] = NULL;
        out_count[i] = 0u;
        failed[i] = 0;
    }
    if (count == 0u) return 0;
    if (engine->decode_audio_batch != NULL) {
        return engine->decode_audio_batch(ctxs, count, first_frame, frame_count,
                                          out_samples, out_count, failed, scratch,
                                          error, error_capacity);
    }
    char one_error[256];
    int reported = 0;
    for (size_t i = 0; i < count; ++i) {
        one_error[0] = '\0';
        if (engine->decode_audio(ctxs[i], first_frame[i], frame_count[i],
                                 &out_samples[i], &out_count[i], one_error,
                                 sizeof(one_error)) != 0) {
            out_samples[i] = NULL;
            out_count[i] = 0u;
            failed[i] = 1;
            if (!reported) {
                mynah_graph_error(error, error_capacity,
                                  one_error[0] != '\0' ? one_error
                                                       : "decoding audio failed");
                reported = 1;
            }
        }
    }
    return 0;
}

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
    /* admission bookkeeping: whose request this is, and whether the slot is
     * occupied at all.  A slot is free the moment it is retired, which is what
     * lets a late arrival take the place of a finished request instead of
     * waiting for the whole group. */
    void *tag;
    int in_use;
    int cancelled;
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

/* How many frames this slot is waiting to accumulate before it delivers.
 *
 * The ramp is .work/streaming-cadence.md §3, and the reason it is a ramp and
 * not a constant is the cadence law: a chunk of C frames can only be delivered
 * after C steps plus its decode, so the player needs a lead of about C frames
 * of wall time to absorb it, and that lead does not exist yet at the start of
 * a stream. So the first chunk is ONE frame -- the first chunk IS the time to
 * first audio -- and the quantum grows only as the lead that pays for it does.
 *
 * We can afford the smallest possible first chunk where the reference could
 * not: our codec carries state instead of replaying context, and the
 * chunked-versus-one-shot error stays at 1e-7 down to a one-frame chunk
 * (E2-3). The knob is free for us in a way it was not for them.
 *
 * The steady state is the engine's own `audio_emit_frames`, and the ramp is
 * clamped to it, never above: this only ever makes early chunks smaller than
 * the engine asked for, so an engine that declares a one-frame threshold
 * because its decode is cheap keeps exactly the cadence it declared. */
static size_t slot_quantum(const mynah_engine_caps *caps, const synth_slot *slot) {
    const size_t steady = caps->audio_emit_frames > 0u
        ? (size_t)caps->audio_emit_frames : 1u;
    const size_t delivered = slot->streamed_frames;
    size_t quantum;
    if (delivered == 0u)     quantum = 1u;   /* this one is the TTFA */
    else if (delivered < 4u) quantum = 2u;
    else if (delivered < 12u) quantum = 4u;
    else                      quantum = steady;
    return quantum < steady ? quantum : steady;
}

/* Hand one gang member its PCM. */
static int slot_deliver(synth_slot *slot, float *audio, size_t produced,
                        size_t frames) {
    if (emit_stream_samples(slot->callback, slot->user_data, audio, produced,
                            slot->chunk_samples, slot->error,
                            slot->error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    slot->streamed_samples += produced;
    slot->streamed_frames += frames;
    return 0;
}

/* Form the decode gang for this step and run it.
 *
 * This is the reference's decoder gang (.work/serving-design.md §5) and the
 * reason it exists is arithmetic, not taste: the codec transformer plus the
 * convolution stack are about 55% of wall time, and while `decode_audio` is
 * declared per context that 55% is multiplied by the number of concurrent
 * streams with nothing shared. Batching the decoder was the single change that
 * moved the reference's capacity, where their decoder was 72-80% of the
 * marginal cost of each extra stream.
 *
 * The policy, in the order the decisions are made:
 *
 *  1. A slot that has reached its own target MUST decode. `finishing` -- the
 *     engine ended the sequence or cut the step window short -- is a target of
 *     its own, reached at whatever is pending.
 *
 *  2. If at least one of those slots is in its steady-state regime, it is a
 *     leader: the fixed per-call cost of a decode is already being paid, so
 *     every other slot holding at least MYNAH_GANG_MIN_PENDING frames joins the
 *     same call. Riding along is close to free and it spends frames that would
 *     otherwise have needed a call of their own a step or two later.
 *
 *  3. NO SLOT IS EVER DELAYED TO MAKE A BIGGER GANG. Step 1 puts every ready
 *     slot in the gang before step 2 looks at anybody, so the gang can only
 *     ever grow past what the no-wait policy already requires. This is not a
 *     tuning choice: the cadence law forbids withholding ready work, and every
 *     scheduling policy the reference measured that parked ready work lost --
 *     their lead/credit gate went 0.838 to 0.986 while parking 95.8% of the
 *     checks it made.
 *
 * Delivery granularity stays decoupled from compute granularity: a follower
 * pulled in with two pending frames is delivered two frames, not the leader's
 * sixteen. The gang is about how the work is executed, never about how much
 * audio a client is made to wait for.
 *
 * Offline slots have no callback and never appear here; they decode once, in
 * full, when they retire. */
#define MYNAH_GANG_MIN_PENDING 1u

static void stream_gang(const mynah_tts_engine *engine, const mynah_engine_caps *caps,
                        mynah_engine_scratch *scratch, synth_slot *slots,
                        const size_t *step_slot,
                        const mynah_engine_step_result *results, size_t live) {
    size_t pending[MYNAH_GRAPH_MAX_JOBS];
    int ready[MYNAH_GRAPH_MAX_JOBS];
    int leading = 0;

    const size_t steady = caps->audio_emit_frames > 0u
        ? (size_t)caps->audio_emit_frames : 1u;

    for (size_t j = 0; j < live; ++j) {
        const synth_slot *slot = &slots[step_slot[j]];
        pending[j] = 0u;
        ready[j] = 0;
        if (slot->callback == NULL || slot->failed || slot->ctx == NULL) continue;
        const size_t frames = engine->frame_count(slot->ctx);
        if (frames <= slot->streamed_frames) continue;
        pending[j] = frames - slot->streamed_frames;
        const int finishing = results[j].eos ||
            results[j].frames_appended < caps->frames_per_step;
        const size_t quantum = slot_quantum(caps, slot);
        if (finishing || pending[j] >= quantum) {
            ready[j] = 1;
            /* A slot still climbing the ramp is not a leader: its decode is
             * small, the fixed cost it would amortise has not been paid yet,
             * and pulling neighbours into it buys nothing. A one-frame steady
             * state means the engine has declared there is no fixed cost to
             * amortise at all, so nobody leads. */
            if (steady >= 2u && quantum >= steady) leading = 1;
        }
    }

    mynah_engine_ctx *gang[MYNAH_GRAPH_MAX_JOBS];
    size_t member[MYNAH_GRAPH_MAX_JOBS];
    size_t first[MYNAH_GRAPH_MAX_JOBS];
    size_t want[MYNAH_GRAPH_MAX_JOBS];
    float *pcm[MYNAH_GRAPH_MAX_JOBS];
    size_t produced[MYNAH_GRAPH_MAX_JOBS];
    int decode_failed[MYNAH_GRAPH_MAX_JOBS];
    size_t count = 0;

    for (size_t j = 0; j < live; ++j) {
        if (pending[j] == 0u) continue;
        if (!ready[j] && !(leading && pending[j] >= MYNAH_GANG_MIN_PENDING)) continue;
        const synth_slot *slot = &slots[step_slot[j]];
        gang[count] = slot->ctx;
        member[count] = j;
        first[count] = slot->streamed_frames;
        want[count] = pending[j];
        ++count;
    }
    if (count == 0u) return;

    char shared_error[256];
    shared_error[0] = '\0';
    const int gang_failed =
        mynah_engine_decode_gang(engine, gang, count, first, want, pcm, produced,
                                 decode_failed, scratch, shared_error,
                                 sizeof(shared_error)) != 0;
    for (size_t g = 0; g < count; ++g) {
        synth_slot *slot = &slots[step_slot[member[g]]];
        if (gang_failed || decode_failed[g]) {
            free(pcm[g]);
            slot_fail(slot, shared_error[0] != '\0' ? shared_error
                                                    : "decoding audio failed");
            continue;
        }
        slot_deliver(slot, pcm[g], produced[g], want[g]);
        free(pcm[g]);
    }
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

/* Finish a request, report it once, and give the slot back to admission.
 *
 * A cancelled request is not a failed one. It never reaches slot_finalize,
 * because finalize's "decoder generated no audio frames" would turn a client
 * that hung up after two frames into a synthesis error in the log. */
static int slot_retire(const mynah_tts_engine *engine, mynah_graph_sink *sink,
                       synth_slot *slot, int dump) {
    int outcome;
    if (slot->cancelled) {
        outcome = MYNAH_GRAPH_CANCELLED;
        mynah_graph_error(slot->error, slot->error_capacity,
                          "request cancelled before it finished");
    } else {
        outcome = slot_finalize(engine, slot, dump) != 0
            ? MYNAH_GRAPH_FAILED : MYNAH_GRAPH_OK;
    }
    engine->ctx_free(slot->ctx);
    slot->ctx = NULL;
    if (outcome == MYNAH_GRAPH_OK && slot->error != NULL && slot->error_capacity > 0) {
        slot->error[0] = '\0';
    }
    void *const tag = slot->tag;
    memset(slot, 0, sizeof(*slot));
    if (sink->on_done != NULL) sink->on_done(sink->ud, tag, outcome);
    return outcome == MYNAH_GRAPH_OK ? 0 : -1;
}

/* Every request the sink can produce without blocking, refused with one
 * message. Used when the driver cannot start at all -- a missing engine, an
 * impossible batch width -- so that no caller is left waiting on a reply that
 * will never come. */
static int refuse_all(mynah_graph_sink *sink, const char *message) {
    mynah_graph_job job;
    void *tag = NULL;
    for (;;) {
        memset(&job, 0, sizeof(job));
        tag = NULL;
        if (sink->next_job(sink->ud, &job, &tag, 0) != 1) break;
        if (job.samples != NULL) *job.samples = NULL;
        if (job.sample_count != NULL) *job.sample_count = 0;
        mynah_graph_error(job.error, job.error_capacity, message);
        if (sink->on_done != NULL) sink->on_done(sink->ud, tag, MYNAH_GRAPH_FAILED);
    }
    return -1;
}

/* Find out whose data the batched step refused, and retire only that request.
 *
 * A batched step has no per-request result channel, so when it fails the driver
 * has one question it cannot answer from the return value: whose fault was it?
 * It answers it by asking again, one context at a time. `step_batch` is atomic
 * over the batch (see tts_engine.h), so nothing advanced on the failed call and
 * re-stepping a context alone is the same step it would have taken had it been
 * alone all along -- which is also why the survivors' audio is unchanged.
 *
 * Without this a single request that hits its own step budget, or arrives with
 * data its engine refuses, retires every request sharing the batch with it.
 * That is invisible at width one and it is the whole server at width sixteen.
 * One request's failure retires one request.
 *
 * Compacts the step arrays in place and returns how many contexts survived. */
static size_t step_isolate(const mynah_tts_engine *engine,
                           mynah_engine_scratch *scratch, synth_slot *slots,
                           mynah_engine_ctx **step_ctxs, size_t *step_slot,
                           size_t live, const char *shared_error) {
    size_t kept = 0;
    for (size_t j = 0; j < live; ++j) {
        char one_error[256];
        one_error[0] = '\0';
        mynah_engine_ctx *one = step_ctxs[j];
        if (engine->step_batch(&one, 1u, scratch, one_error, sizeof(one_error)) != 0) {
            slot_fail(&slots[step_slot[j]],
                      one_error[0] != '\0' ? one_error : shared_error);
            continue;
        }
        step_ctxs[kept] = one;
        step_slot[kept] = step_slot[j];
        ++kept;
    }
    return kept;
}

/* One AR step for every live slot, plus whatever audio that made final. */
static void step_live(const mynah_tts_engine *engine, const mynah_engine_caps *caps,
                      mynah_engine_scratch *scratch, synth_slot *slots,
                      mynah_engine_ctx **step_ctxs, size_t *step_slot,
                      mynah_engine_step_result *results, size_t live, int dump) {
    char shared_error[256];
    shared_error[0] = '\0';
    if (engine->step_batch(step_ctxs, live, scratch, shared_error,
                           sizeof(shared_error)) != 0) {
        if (live <= 1u) {
            /* Alone in the batch, the attribution is not in doubt and there is
             * nothing to isolate it from. */
            if (live == 1u) slot_fail(&slots[step_slot[0]], shared_error);
            return;
        }
        live = step_isolate(engine, scratch, slots, step_ctxs, step_slot, live,
                            shared_error);
        if (live == 0u) return;
    }
    if (dump && engine->debug_dump != NULL) {
        for (size_t j = 0; j < live; ++j) engine->debug_dump(step_ctxs[j], "hidden");
    }

    memset(results, 0, live * sizeof(*results));
    shared_error[0] = '\0';
    if (engine->emit_batch(step_ctxs, live, results, scratch, shared_error,
                           sizeof(shared_error)) != 0) {
        int attributed = 0;
        for (size_t j = 0; j < live; ++j) {
            if (results[j].failed) {
                slot_fail(&slots[step_slot[j]], shared_error);
                attributed = 1;
            }
        }
        /* A failure the engine attributed to nobody is a failure of shared
         * code, and it has to retire the batch: returning here having failed
         * no one would leave every slot active, and the service loop would
         * take the same step again forever. */
        if (!attributed) {
            for (size_t j = 0; j < live; ++j) slot_fail(&slots[step_slot[j]], shared_error);
        }
        return;
    }
    for (size_t j = 0; j < live; ++j) {
        if (results[j].failed) slot_fail(&slots[step_slot[j]], shared_error);
    }
    /* Delivery is decided for the whole batch at once, not slot by slot: that
     * is the only place the driver can see two requests' codec work together. */
    stream_gang(engine, caps, scratch, slots, step_slot, results, live);
    for (size_t j = 0; j < live; ++j) {
        synth_slot *slot = &slots[step_slot[j]];
        if (slot->failed) continue;
        if (results[j].eos) slot->active = 0;
    }
}

/* The one driver.
 *
 * `want_batch` is how wide the caller would like to run; `strict_batch` says
 * that anything narrower is an error rather than a slower path, which is what
 * a fixed array of N jobs means and what a service does not. Continuous in the
 * sense that matters: a finished slot is retired and refilled from the sink
 * between steps, so a request arriving mid-flight joins the batch that is
 * already running rather than waiting for it to drain. */
static int serve(const mynah_tts_engine *engine, const mynah_tts_model *model,
                 mynah_graph_sink *sink, size_t want_batch, int strict_batch,
                 int dump_all) {
    if (sink == NULL || sink->next_job == NULL) return -1;
    if (want_batch == 0u) return 0;
    if (want_batch > MYNAH_GRAPH_MAX_JOBS) return -1;
    if (engine == NULL) {
        return refuse_all(sink, "model.json names an engine this build does not have");
    }
    char shared_error[256];
    shared_error[0] = '\0';
    mynah_engine_state *state = NULL;
    if (engine->model_init(model, &state, shared_error, sizeof(shared_error)) != 0) {
        return refuse_all(sink, shared_error);
    }
    mynah_engine_caps caps;
    memset(&caps, 0, sizeof(caps));
    if (engine->caps(model, state, &caps) != 0 || caps.frames_per_step == 0u ||
        caps.max_batch == 0u || (strict_batch && want_batch > caps.max_batch)) {
        engine->model_free(state);
        return refuse_all(sink, "the engine cannot serve this batch");
    }
    /* The engine's ceiling wins over the caller's wish: a continuous-latent
     * engine declares 1 and stepping two of its contexts together is not a
     * slower path, it is an out-of-bounds write. */
    const size_t max_batch = want_batch < caps.max_batch ? want_batch : caps.max_batch;

    /* Sized once, for the widest batch this driver will ever step, and never
     * resized. Sizing it on the slots that happen to be present is the bug
     * continuous admission would otherwise introduce twice over: a later,
     * wider batch writing past the end, and -- because a scratch built for one
     * slot is not allocated at all -- a service that starts at one request and
     * grows silently falling back to the per-row path, losing all batching
     * with no error anywhere. */
    mynah_engine_scratch *scratch = NULL;
    if (engine->scratch_new(model, state, max_batch, &scratch, shared_error,
                            sizeof(shared_error)) != 0) {
        engine->model_free(state);
        return refuse_all(sink, shared_error);
    }

    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double t_start = timing ? mynah_phase_seconds() : 0.0;
    double t_prep = t_start, t_ar = t_start;
    size_t admitted = 0;

    synth_slot slots[MYNAH_GRAPH_MAX_JOBS];
    mynah_engine_ctx *step_ctxs[MYNAH_GRAPH_MAX_JOBS];
    size_t step_slot[MYNAH_GRAPH_MAX_JOBS];
    mynah_engine_step_result results[MYNAH_GRAPH_MAX_JOBS];
    memset(slots, 0, sizeof(slots));

    int result = 0;
    size_t used = 0;      /* slots holding a request, live or just finished */
    int drained = 0;      /* the sink said there will be no more work */

    for (;;) {
        /* ---- admission ------------------------------------------------
         * At the top of the step, not before the loop. `block` is set only
         * when there is nothing else to do, so a running batch is never held
         * up waiting for an arrival that may not come. */
        while (!drained && used < max_batch &&
               (sink->running == NULL || sink->running(sink->ud) != 0)) {
            size_t index = max_batch;
            for (size_t i = 0; i < max_batch; ++i) {
                if (!slots[i].in_use) { index = i; break; }
            }
            if (index == max_batch) break;

            mynah_graph_job job;
            memset(&job, 0, sizeof(job));
            void *tag = NULL;
            const int block = (used == 0u);
            if (sink->next_job(sink->ud, &job, &tag, block) != 1) {
                /* Nothing available. If we asked it to block and it still had
                 * nothing, the service is over. */
                if (block) drained = 1;
                break;
            }
            synth_slot *slot = &slots[index];
            memset(slot, 0, sizeof(*slot));
            slot->in_use = 1;
            slot->tag = tag;
            slot->request = job.request;
            slot->samples = job.samples;
            slot->sample_count = job.sample_count;
            slot->callback = job.callback;
            slot->user_data = job.user_data;
            slot->chunk_samples = job.chunk_samples;
            slot->error = job.error;
            slot->error_capacity = job.error_capacity;
            ++used;
            ++admitted;
            if (slot_start(engine, model, state, &caps, slot, dump_all) != 0) {
                /* A request that cannot start never occupies the batch. */
                if (slot_retire(engine, sink, slot, dump_all) != 0) result = -1;
                --used;
            }
        }
        if (timing && t_prep == t_start) t_prep = mynah_phase_seconds();
        if (used == 0u) break;

        /* ---- cancellation --------------------------------------------- */
        if (sink->cancelled != NULL) {
            for (size_t i = 0; i < max_batch; ++i) {
                if (!slots[i].in_use || !slots[i].active) continue;
                if (sink->cancelled(sink->ud, slots[i].tag) != 0) {
                    slots[i].cancelled = 1;
                    slots[i].active = 0;
                }
            }
        }

        /* ---- one step over everything still live ---------------------- */
        size_t live = 0;
        for (size_t i = 0; i < max_batch; ++i) {
            if (!slots[i].in_use || !slots[i].active) continue;
            step_slot[live] = i;
            step_ctxs[live] = slots[i].ctx;
            ++live;
        }
        if (live > 0u) {
            step_live(engine, &caps, scratch, slots, step_ctxs, step_slot, results,
                      live, dump_all);
        }

        /* ---- retire, per slot, as soon as it stops ---------------------
         * Not after the whole group: the slot is the unit of capacity, and
         * holding a finished one until its neighbours catch up is exactly the
         * wait continuous admission exists to remove. */
        for (size_t i = 0; i < max_batch; ++i) {
            if (!slots[i].in_use || slots[i].active) continue;
            if (slot_retire(engine, sink, &slots[i], dump_all) != 0) result = -1;
            --used;
        }
    }
    if (timing) {
        t_ar = mynah_phase_seconds();
        fprintf(stderr, "phase: prep=%.3fs ar=%.3fs (requests=%zu)\n",
                t_prep - t_start, t_ar - t_prep, admitted);
    }
    engine->scratch_free(scratch);
    engine->model_free(state);
    return result;
}

/* ---- the array sink: N jobs, then nothing ------------------------------- */

typedef struct {
    mynah_graph_job *jobs;
    size_t count;
    size_t next;
} array_sink;

static int array_next_job(void *ud, mynah_graph_job *job, void **tag, int block) {
    (void)block;
    array_sink *state = (array_sink *)ud;
    if (state->next >= state->count) return 0;
    mynah_graph_job *source = &state->jobs[state->next++];
    *job = *source;
    *tag = source;
    return 1;
}

static void array_on_done(void *ud, void *tag, int result) {
    (void)ud;
    mynah_graph_job *job = (mynah_graph_job *)tag;
    job->result = result == MYNAH_GRAPH_OK ? 0 : -1;
}

int mynah_graph_serve_engine(const mynah_tts_engine *engine,
                             const mynah_tts_model *model, mynah_graph_sink *sink,
                             size_t max_batch, int strict_batch) {
    if (max_batch == 0u) max_batch = 1u;
    if (max_batch > MYNAH_GRAPH_MAX_JOBS) max_batch = MYNAH_GRAPH_MAX_JOBS;
    return serve(engine, model, sink, max_batch, strict_batch, 0);
}

int mynah_graph_serve_continuous(const mynah_tts_model *model, size_t max_batch,
                                 mynah_graph_sink *sink) {
    if (model == NULL) return -1;
    if (max_batch == 0u) max_batch = 1u;
    if (max_batch > MYNAH_GRAPH_MAX_JOBS) max_batch = MYNAH_GRAPH_MAX_JOBS;
    return serve(mynah_engine_lookup(model->info.engine), model, sink, max_batch, 0, 0);
}

int mynah_graph_synthesize_jobs(const mynah_tts_model *model,
                                mynah_graph_job *jobs, size_t count) {
    if (model == NULL || jobs == NULL) return -1;
    if (count == 0u) return 0;
    if (count > MYNAH_GRAPH_MAX_JOBS) return -1;
    for (size_t i = 0; i < count; ++i) jobs[i].result = 0;
    array_sink state;
    state.jobs = jobs;
    state.count = count;
    state.next = 0;
    mynah_graph_sink sink;
    memset(&sink, 0, sizeof(sink));
    sink.ud = &state;
    sink.next_job = array_next_job;
    sink.on_done = array_on_done;
    return serve(mynah_engine_lookup(model->info.engine), model, &sink, count, 1,
                 count == 1u);
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

size_t mynah_tts_model_max_batch(const mynah_tts_model *model) {
    if (model == NULL) return 1u;
    const mynah_tts_engine *engine = mynah_engine_lookup(model->info.engine);
    if (engine == NULL || engine->caps == NULL) return 1u;
    mynah_engine_caps caps;
    memset(&caps, 0, sizeof(caps));
    if (engine->caps(model, NULL, &caps) != 0 || caps.max_batch == 0u) return 1u;
    return caps.max_batch < MYNAH_GRAPH_MAX_JOBS ? caps.max_batch : MYNAH_GRAPH_MAX_JOBS;
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
