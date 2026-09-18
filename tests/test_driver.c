/* Driver policy tests against a synthetic engine.
 *
 * What is under test is src/inference.c, and only src/inference.c: the decode
 * gang, the blast radius of a per-request failure, and the quantum ramp. All
 * three are engine-independent by construction, and none of them can be
 * exercised properly through a real model pack -- there is no way to make
 * exactly one Magpie request of sixteen refuse a step on purpose, and that is
 * precisely the case that matters. A synthetic engine can, deterministically,
 * with no weights and in milliseconds.
 *
 * The engine below is a real engine as far as the seam is concerned. Its audio
 * is a pure function of (seed, absolute frame index, sample index), so "the
 * same request produces the same samples whoever it shared a batch or a decode
 * gang with" is not an approximation here, it is an identity -- which is
 * exactly the guarantee the seam asks an engine to make, and it lets the test
 * compare batched runs against solo runs byte for byte.
 */
#include "graph.h"
#include "mynah_tts.h"
#include "tts_engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAKE_SAMPLES_PER_FRAME 16u
#define FAKE_EMIT_FRAMES 8u
#define FAKE_MAX_BATCH 8u
#define LOG_MAX 1024u

/* ---- what the test watches the driver do -------------------------------- */

typedef struct {
    size_t decode_calls;          /* decode_audio invocations */
    size_t gang_calls;            /* decode_audio_batch invocations */
    size_t max_gang;              /* widest gang the driver formed */
    size_t multi_member_gangs;    /* gangs that batched more than one context */
    size_t rode_along;            /* members decoded alongside a bigger member */
    size_t withheld;              /* times ready work was NOT decoded: must be 0 */
    size_t quantum[LOG_MAX];      /* frames asked for, in call order */
    uint64_t quantum_seed[LOG_MAX];
    size_t quanta;
} observation;

static observation g_obs;

/* The driver's ramp, restated here on purpose. The engine can compute it
 * because the driver's `streamed_frames` for a slot is exactly this context's
 * `decoded`: the two counters are the same number seen from opposite sides of
 * the seam. That is what makes the no-wait law checkable from inside an
 * engine -- see fake_step_batch. */
static size_t expected_quantum(size_t delivered, size_t steady) {
    size_t quantum;
    if (delivered == 0u)      quantum = 1u;
    else if (delivered < 4u)  quantum = 2u;
    else if (delivered < 12u) quantum = 4u;
    else                      quantum = steady;
    return quantum < steady ? quantum : steady;
}

static void observe_decode(uint64_t seed, size_t frames) {
    ++g_obs.decode_calls;
    if (g_obs.quanta < LOG_MAX) {
        g_obs.quantum[g_obs.quanta] = frames;
        g_obs.quantum_seed[g_obs.quanta] = seed;
        ++g_obs.quanta;
    }
}

/* ---- the synthetic engine ----------------------------------------------- */

struct mynah_engine_state { int live; };
struct mynah_engine_scratch { size_t batch; };

struct mynah_engine_ctx {
    uint64_t seed;
    size_t total_frames;   /* how much audio this request generates */
    size_t step;
    size_t frames;         /* appended so far */
    size_t decoded;        /* handed to the driver so far */
    long   refuse_step;    /* < 0 never; otherwise refuse this step forever */
};

/* Pure in (seed, absolute frame, sample): independent of how the frames were
 * chunked and of who else was in the call. */
static float fake_sample(uint64_t seed, size_t frame, size_t index) {
    uint64_t h = seed * 0x9e3779b97f4a7c15ull +
                 (uint64_t)frame * 0x100000001b3ull + (uint64_t)index;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return (float)(int32_t)(uint32_t)h / 2147483648.0f;
}

static void fake_err(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

static int fake_model_init(const mynah_tts_model *model, mynah_engine_state **out,
                           char *error, size_t capacity) {
    (void)model;
    *out = (mynah_engine_state *)calloc(1, sizeof(**out));
    if (*out == NULL) {
        fake_err(error, capacity, "fake: out of memory");
        return -1;
    }
    (*out)->live = 1;
    return 0;
}

static void fake_model_free(mynah_engine_state *state) { free(state); }

static int fake_caps(const mynah_tts_model *model, const mynah_engine_state *state,
                     mynah_engine_caps *out) {
    (void)model;
    (void)state;
    if (out == NULL) return -1;
    memset(out, 0, sizeof(*out));
    out->sample_rate = 24000u;
    out->frame_rate = 12.5;
    out->frames_per_step = 1u;
    out->audio_emit_frames = FAKE_EMIT_FRAMES;
    out->min_audio_frames = 1u;
    out->default_max_steps = 256u;
    out->max_batch = FAKE_MAX_BATCH;
    out->voice_count = 1u;
    out->latent_dim = 8u;
    return 0;
}

/* The request carries the scenario: `topk` is how many frames this request
 * generates, and `speaker` is one more than the step at which it refuses to
 * advance (0 meaning it never does). */
static int fake_ctx_new(const mynah_tts_model *model, mynah_engine_state *state,
                        const mynah_tts_request *request, size_t max_steps,
                        uint64_t seed, mynah_engine_ctx **out_ctx,
                        char *error, size_t capacity) {
    (void)model;
    (void)state;
    (void)max_steps;
    mynah_engine_ctx *ctx = (mynah_engine_ctx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        fake_err(error, capacity, "fake: out of memory");
        return -1;
    }
    ctx->seed = seed;
    ctx->total_frames = request->topk;
    ctx->refuse_step = request->speaker == 0u ? -1 : (long)request->speaker - 1;
    *out_ctx = ctx;
    return 0;
}

static int fake_prepare(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    (void)error;
    (void)capacity;
    return ctx == NULL ? -1 : 0;
}

static int fake_reset(mynah_engine_ctx *ctx, char *error, size_t capacity) {
    (void)error;
    (void)capacity;
    if (ctx == NULL) return -1;
    ctx->step = ctx->frames = ctx->decoded = 0u;
    return 0;
}

static void fake_ctx_free(mynah_engine_ctx *ctx) { free(ctx); }

/* Atomic over the batch, as tts_engine.h requires: every context is checked
 * before any of them is advanced, so a refusal leaves the batch exactly as it
 * found it and the driver's isolation pass is re-stepping, not double-stepping. */
static int fake_step_batch(mynah_engine_ctx *const *ctxs, size_t count,
                           mynah_engine_scratch *scratch, char *error,
                           size_t capacity) {
    (void)scratch;
    /* THE NO-WAIT LAW, checked from inside the engine at the only moment it can
     * be: the top of a step, when the previous step's delivery is complete. A
     * context still holding at least a full quantum of undelivered frames means
     * the driver decided to sit on ready work, which the cadence law forbids and
     * which every parking policy the reference measured lost on. */
    for (size_t i = 0; i < count; ++i) {
        const mynah_engine_ctx *ctx = ctxs[i];
        const size_t pending = ctx->frames - ctx->decoded;
        if (pending > 0u &&
            pending >= expected_quantum(ctx->decoded, FAKE_EMIT_FRAMES)) {
            ++g_obs.withheld;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (ctxs[i]->refuse_step >= 0 &&
            ctxs[i]->step == (size_t)ctxs[i]->refuse_step) {
            if (error != NULL && capacity > 0) {
                snprintf(error, capacity, "fake: request %llu refuses step %zu",
                         (unsigned long long)ctxs[i]->seed, ctxs[i]->step);
            }
            return -1;
        }
    }
    for (size_t i = 0; i < count; ++i) ctxs[i]->step += 1u;
    return 0;
}

static int fake_emit_batch(mynah_engine_ctx *const *ctxs, size_t count,
                           mynah_engine_step_result *results,
                           mynah_engine_scratch *scratch, char *error,
                           size_t capacity) {
    (void)scratch;
    (void)error;
    (void)capacity;
    for (size_t i = 0; i < count; ++i) {
        mynah_engine_ctx *ctx = ctxs[i];
        ctx->frames += 1u;
        memset(&results[i], 0, sizeof(results[i]));
        results[i].frames_appended = 1u;
        results[i].eos_frame = 1u;
        results[i].eos = ctx->frames >= ctx->total_frames;
    }
    return 0;
}

static size_t fake_frame_count(const mynah_engine_ctx *ctx) {
    return ctx == NULL ? 0u : ctx->frames;
}

static void fake_truncate(mynah_engine_ctx *ctx, size_t frames) {
    if (ctx != NULL && frames < ctx->frames) ctx->frames = frames;
}

static int fake_decode_audio(mynah_engine_ctx *ctx, size_t first, size_t frames,
                             float **out_samples, size_t *out_count, char *error,
                             size_t capacity) {
    *out_samples = NULL;
    *out_count = 0u;
    if (first != ctx->decoded) {
        fake_err(error, capacity, "fake: the driver asked for a non-contiguous range");
        return -1;
    }
    if (first > ctx->frames || frames > ctx->frames - first) {
        fake_err(error, capacity, "fake: the driver asked past the frame history");
        return -1;
    }
    observe_decode(ctx->seed, frames);
    if (frames == 0u) return 0;
    const size_t count = frames * FAKE_SAMPLES_PER_FRAME;
    float *pcm = (float *)malloc(count * sizeof(*pcm));
    if (pcm == NULL) {
        fake_err(error, capacity, "fake: out of memory decoding");
        return -1;
    }
    for (size_t f = 0; f < frames; ++f) {
        for (size_t k = 0; k < FAKE_SAMPLES_PER_FRAME; ++k) {
            pcm[f * FAKE_SAMPLES_PER_FRAME + k] = fake_sample(ctx->seed, first + f, k);
        }
    }
    ctx->decoded += frames;
    *out_samples = pcm;
    *out_count = count;
    return 0;
}

/* The batched variant. It decodes each context exactly as the per-context hook
 * would -- that is the point: an engine may only exploit the gang in ways that
 * leave every row's samples untouched. What it adds is the record of how wide
 * the driver's gangs actually were. */
static int fake_decode_audio_batch(mynah_engine_ctx *const *ctxs, size_t count,
                                   const size_t *first, const size_t *frames,
                                   float **out_samples, size_t *out_count,
                                   int *failed, mynah_engine_scratch *scratch,
                                   char *error, size_t capacity) {
    (void)scratch;
    ++g_obs.gang_calls;
    if (count > g_obs.max_gang) g_obs.max_gang = count;
    if (count > 1u) ++g_obs.multi_member_gangs;
    /* A member that rode along, defined so that nothing else can be mistaken
     * for one: it was handed LESS than its own quantum while it still had more
     * audio to generate. A slot that reached its target cannot look like this,
     * and neither can a slot flushing its last frames at the end of a sequence
     * -- those two are the whole of "everyone ready decodes". What is left is
     * exactly the leader pull-in. */
    for (size_t i = 0; i < count; ++i) {
        const mynah_engine_ctx *ctx = ctxs[i];
        const int finishing = ctx->frames >= ctx->total_frames;
        if (!finishing &&
            frames[i] < expected_quantum(ctx->decoded, FAKE_EMIT_FRAMES)) {
            ++g_obs.rode_along;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        char one[256];
        one[0] = '\0';
        if (fake_decode_audio(ctxs[i], first[i], frames[i], &out_samples[i],
                              &out_count[i], one, sizeof(one)) != 0) {
            failed[i] = 1;
            fake_err(error, capacity, one);
        }
    }
    return 0;
}

static int fake_scratch_new(const mynah_tts_model *model, mynah_engine_state *state,
                            size_t batch, mynah_engine_scratch **out, char *error,
                            size_t capacity) {
    (void)model;
    (void)state;
    *out = (mynah_engine_scratch *)calloc(1, sizeof(**out));
    if (*out == NULL) {
        fake_err(error, capacity, "fake: out of memory");
        return -1;
    }
    (*out)->batch = batch;
    return 0;
}

static void fake_scratch_free(mynah_engine_scratch *scratch) { free(scratch); }

static const mynah_tts_engine fake_engine_loop = {
    "driver-test",
    fake_model_init, fake_model_free, fake_caps,
    fake_ctx_new, fake_prepare, fake_reset, fake_ctx_free,
    fake_step_batch, fake_emit_batch,
    fake_frame_count, fake_truncate, fake_decode_audio,
    fake_scratch_new, fake_scratch_free,
    NULL,
    /* decode_audio_batch deliberately absent: this is the landing state of
     * every engine, and it must behave exactly like the one below. */
};

static const mynah_tts_engine fake_engine_gang = {
    "driver-test-gang",
    fake_model_init, fake_model_free, fake_caps,
    fake_ctx_new, fake_prepare, fake_reset, fake_ctx_free,
    fake_step_batch, fake_emit_batch,
    fake_frame_count, fake_truncate, fake_decode_audio,
    fake_scratch_new, fake_scratch_free,
    NULL,
    fake_decode_audio_batch,
};

/* ---- sinks -------------------------------------------------------------- */

typedef struct {
    float *samples;
    size_t count;
    size_t capacity;
    size_t chunks;
    size_t first_chunk_samples;
} capture;

static int capture_audio(const float *samples, size_t count, void *ud) {
    capture *out = (capture *)ud;
    const size_t required = out->count + count;
    if (required > out->capacity) {
        size_t capacity = out->capacity == 0 ? 256u : out->capacity;
        while (capacity < required) capacity *= 2u;
        float *grown = (float *)realloc(out->samples, capacity * sizeof(*grown));
        if (grown == NULL) return 1;
        out->samples = grown;
        out->capacity = capacity;
    }
    memcpy(out->samples + out->count, samples, count * sizeof(*samples));
    out->count = required;
    if (out->chunks == 0u) out->first_chunk_samples = count;
    ++out->chunks;
    return 0;
}

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
    ((mynah_graph_job *)tag)->result = result;
}

/* ---- the scenario ------------------------------------------------------- */

/* More requests than slots, so the batch is never in lockstep: a short request
 * retires, a fresh one is admitted mid-flight and starts its ramp at step k
 * while its neighbours are deep in their steady state. A test where every slot
 * reaches its target on the same step would have every gang fall out of "all
 * ready slots decode" and would prove nothing about the leader pull-in. */
#define REQUESTS 8u
#define BATCH_WIDTH 4u

static const int g_tokens[] = {1, 2, 3, 4};

typedef struct {
    unsigned frames;    /* -> request.topk */
    unsigned refuse;    /* -> request.speaker: 0 never, else step+1 */
} scenario;

static void build_requests(mynah_tts_request *requests, const scenario *plan,
                           size_t count) {
    for (size_t i = 0; i < count; ++i) {
        memset(&requests[i], 0, sizeof(requests[i]));
        requests[i].text_ids = g_tokens;
        requests[i].text_length = sizeof(g_tokens) / sizeof(g_tokens[0]);
        requests[i].topk = plan[i].frames;
        requests[i].speaker = plan[i].refuse;
        requests[i].seed = 1000u + (uint64_t)i;
        requests[i].max_steps = 64u;
    }
}

/* Run `count` requests through the driver at the given batch width, streaming,
 * and report each one's captured audio and outcome. */
static int run(const mynah_tts_engine *engine, const mynah_tts_request *requests,
               size_t count, size_t width, capture *out, int *results,
               char errors[][256]) {
    mynah_graph_job jobs[REQUESTS];
    memset(jobs, 0, sizeof(jobs));
    for (size_t i = 0; i < count; ++i) {
        memset(&out[i], 0, sizeof(out[i]));
        errors[i][0] = '\0';
        jobs[i].request = &requests[i];
        jobs[i].callback = capture_audio;
        jobs[i].user_data = &out[i];
        jobs[i].chunk_samples = 4096u;   /* one callback per decode */
        jobs[i].error = errors[i];
        jobs[i].error_capacity = 256u;
    }
    array_sink state = {jobs, count, 0u};
    mynah_graph_sink sink;
    memset(&sink, 0, sizeof(sink));
    sink.ud = &state;
    sink.next_job = array_next_job;
    sink.on_done = array_on_done;
    const int rc = mynah_graph_serve_engine(engine, NULL, &sink, width, 0);
    for (size_t i = 0; i < count; ++i) results[i] = jobs[i].result;
    return rc;
}

static int fail(const char *what) {
    fprintf(stderr, "driver test FAILED: %s\n", what);
    return 1;
}

static int same_audio(const capture *a, const capture *b) {
    return a->count == b->count &&
           (a->count == 0u ||
            memcmp(a->samples, b->samples, a->count * sizeof(float)) == 0);
}

static void release(capture *c, size_t count) {
    for (size_t i = 0; i < count; ++i) free(c[i].samples);
}

int main(void) {
    /* Lengths chosen so the slots reach their targets on different steps: a
     * gang that only ever forms when everybody is ready proves nothing. */
    const scenario healthy[REQUESTS] = {
        {20u, 0u}, {13u, 0u}, {31u, 0u}, {7u, 0u},
        {17u, 0u}, {9u, 0u},  {23u, 0u}, {11u, 0u},
    };
    mynah_tts_request requests[REQUESTS];
    capture solo[REQUESTS];
    capture batched[REQUESTS];
    int results[REQUESTS];
    char errors[REQUESTS][256];
    int rc = 0;

    /* ---- 1. the reference: every request on its own ---------------------- */
    build_requests(requests, healthy, REQUESTS);
    memset(&g_obs, 0, sizeof(g_obs));
    if (run(&fake_engine_loop, requests, REQUESTS, 1u, solo, results, errors) != 0) {
        return fail("solo run reported a failure");
    }
    for (size_t i = 0; i < REQUESTS; ++i) {
        if (results[i] != MYNAH_GRAPH_OK) return fail("a solo request failed");
        if (solo[i].count != healthy[i].frames * FAKE_SAMPLES_PER_FRAME) {
            return fail("a solo request delivered the wrong amount of audio");
        }
    }

    /* ---- 2. the quantum ramp -------------------------------------------- *
     * One slot alone, so the log of decode sizes IS its quantum sequence.
     * .work/streaming-cadence.md §3: 1, 2, 2, 4, 4, then the steady state.
     * The first chunk is one frame and that is the time to first audio. */
    {
        scenario one[REQUESTS];
        memset(one, 0, sizeof(one));
        one[0].frames = 20u;
        mynah_tts_request single[REQUESTS];
        capture got[REQUESTS];
        build_requests(single, one, REQUESTS);
        memset(&g_obs, 0, sizeof(g_obs));
        if (run(&fake_engine_loop, single, 1u, 1u, got, results, errors) != 0) {
            release(got, 1u);
            return fail("the ramp run reported a failure");
        }
        const size_t expected[] = {1u, 2u, 2u, 4u, 4u, 7u};  /* then the tail */
        const size_t expected_count = sizeof(expected) / sizeof(expected[0]);
        if (g_obs.quanta < expected_count) {
            release(got, 1u);
            return fail("the ramp produced too few chunks");
        }
        for (size_t i = 0; i < expected_count; ++i) {
            if (g_obs.quantum[i] != expected[i]) {
                fprintf(stderr, "  quantum[%zu] = %zu, expected %zu\n", i,
                        g_obs.quantum[i], expected[i]);
                release(got, 1u);
                return fail("the quantum ramp is not 1, 2, 2, 4, 4, then steady");
            }
        }
        if (got[0].first_chunk_samples != FAKE_SAMPLES_PER_FRAME) {
            release(got, 1u);
            return fail("the first chunk was not exactly one frame");
        }
        if (!same_audio(&got[0], &solo[0])) {
            release(got, 1u);
            return fail("the ramp changed the audio");
        }
        release(got, 1u);
    }

    /* ---- 3. batching changes nobody's audio, and forms real gangs -------- */
    memset(&g_obs, 0, sizeof(g_obs));
    if (run(&fake_engine_gang, requests, REQUESTS, BATCH_WIDTH, batched, results,
            errors) != 0) {
        release(batched, REQUESTS);
        return fail("the batched run reported a failure");
    }
    for (size_t i = 0; i < REQUESTS; ++i) {
        if (results[i] != MYNAH_GRAPH_OK) {
            release(batched, REQUESTS);
            return fail("a batched request failed");
        }
        if (!same_audio(&batched[i], &solo[i])) {
            release(batched, REQUESTS);
            return fail("batching changed a request's audio");
        }
    }
    if (g_obs.gang_calls == 0u) {
        release(batched, REQUESTS);
        return fail("the engine's decode_audio_batch was never called");
    }
    if (g_obs.multi_member_gangs == 0u || g_obs.max_gang < 2u) {
        release(batched, REQUESTS);
        return fail("no gang ever batched more than one context");
    }
    /* The constraint first, because it is the one that is not negotiable:
     * nobody paid for the gang by waiting. */
    if (g_obs.withheld != 0u) {
        fprintf(stderr, "  ready work withheld %zu times\n", g_obs.withheld);
        release(batched, REQUESTS);
        return fail("the driver delayed ready work to form a bigger gang");
    }
    /* Then the optimisation: a member that had not reached its own target was
     * pulled into a leader's decode anyway. */
    if (g_obs.rode_along == 0u) {
        release(batched, REQUESTS);
        return fail("no slot was ever pulled into a leader's decode");
    }
    printf("  gangs: %zu calls, widest %zu, %zu batched >1, %zu rode along, "
           "%zu withheld\n", g_obs.gang_calls, g_obs.max_gang,
           g_obs.multi_member_gangs, g_obs.rode_along, g_obs.withheld);
    release(batched, REQUESTS);

    /* ---- 4. the default loop and the batched hook are interchangeable ---- */
    memset(&g_obs, 0, sizeof(g_obs));
    if (run(&fake_engine_loop, requests, REQUESTS, BATCH_WIDTH, batched, results,
            errors) != 0) {
        release(batched, REQUESTS);
        return fail("the default-loop batched run reported a failure");
    }
    for (size_t i = 0; i < REQUESTS; ++i) {
        if (results[i] != MYNAH_GRAPH_OK || !same_audio(&batched[i], &solo[i])) {
            release(batched, REQUESTS);
            return fail("the default decode_audio_batch loop differs from the hook");
        }
    }
    if (g_obs.gang_calls != 0u) {
        release(batched, REQUESTS);
        return fail("an engine without the hook still had it called");
    }
    release(batched, REQUESTS);

    /* ---- 5. the blast radius -------------------------------------------- *
     * One request of four refuses a step in the middle of the batch. The
     * other three must finish, and finish byte-identically to running alone.
     * Before the isolation pass this retired all four. */
    for (size_t victim = 0; victim < REQUESTS; ++victim) {
        scenario poisoned[REQUESTS];
        capture got[REQUESTS];
        for (size_t i = 0; i < REQUESTS; ++i) poisoned[i] = healthy[i];
        poisoned[victim].refuse = 4u;   /* refuses step 3, well into the run */
        build_requests(requests, poisoned, REQUESTS);
        memset(&g_obs, 0, sizeof(g_obs));
        const int rc_all = run(&fake_engine_gang, requests, REQUESTS, BATCH_WIDTH,
                               got, results, errors);
        if (rc_all == 0) {
            release(got, REQUESTS);
            return fail("a batch with a failing request reported success");
        }
        for (size_t i = 0; i < REQUESTS; ++i) {
            if (i == victim) {
                if (results[i] != MYNAH_GRAPH_FAILED) {
                    release(got, REQUESTS);
                    return fail("the failing request did not fail");
                }
                if (strstr(errors[i], "refuses step") == NULL) {
                    fprintf(stderr, "  victim error: '%s'\n", errors[i]);
                    release(got, REQUESTS);
                    return fail("the failing request got somebody else's message");
                }
                continue;
            }
            if (results[i] != MYNAH_GRAPH_OK) {
                fprintf(stderr, "  request %zu died with victim %zu: '%s'\n", i,
                        victim, errors[i]);
                release(got, REQUESTS);
                return fail("one request's failure retired a sibling");
            }
            if (!same_audio(&got[i], &solo[i])) {
                release(got, REQUESTS);
                return fail("a survivor's audio changed because a sibling failed");
            }
        }
        release(got, REQUESTS);
    }

    /* Restore the healthy requests for anything added after this point. */
    build_requests(requests, healthy, REQUESTS);
    release(solo, REQUESTS);
    if (rc != 0) return rc;
    puts("driver policy: PASS");
    return 0;
}
