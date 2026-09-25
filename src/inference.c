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
 * why an offline batch and a live stream cannot drift apart (AGENTS.md rule 7).
 */
#include "costmap.h"
#include "graph.h"
#include "mynah_tts_internal.h"
#include "mynah_util.h"
#include "threads.h"
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
    /* The sink's own time, inside the synthesis loop. It is measured here and
     * not folded into the decode precisely so that a slow consumer cannot be
     * read as a slow codec (costmap.h, MYNAH_RGN_STREAM_EMIT). */
    mynah_region_begin(MYNAH_RGN_STREAM_EMIT);
    size_t offset = 0;
    while (offset < count) {
        const size_t remaining = count - offset;
        const size_t chunk = remaining < chunk_samples ? remaining : chunk_samples;
        if (callback(samples + offset, chunk, user_data) != 0) {
            mynah_region_end(MYNAH_RGN_STREAM_EMIT);
            mynah_graph_error(error, error_capacity, "audio callback aborted streaming");
            return -1;
        }
        offset += chunk;
    }
    mynah_region_end(MYNAH_RGN_STREAM_EMIT);
    return 0;
}

/* One decode handed to the decoder lane -- E5-21.
 *
 * It lives inside the slot rather than being allocated per decode: nothing in
 * the AR loop may allocate (AGENTS.md rule 4), and a unit the lane is reading
 * must outlive the call that submitted it. One per slot is also the bounded
 * contract made structural -- there is nowhere to put a second. */
typedef struct {
    const mynah_tts_engine *engine;
    mynah_engine_ctx *ctx;
    size_t first;
    size_t frames;
    float *pcm;
    size_t produced;
    int    failed;
    char   error[256];
} lane_unit;

/* Runs ON A LANE THREAD. Everything it dispatches -- the conv stack, the codec
 * transformer, every SGEMM slice inside them -- lands on the lane team,
 * because mynah_parallel_for() reads the thread-local tag that lane thread
 * carries. Nothing in this function or below it knows that, and that is the
 * point of putting the redirection in the dispatch primitive. */
static void lane_decode(void *ud) {
    lane_unit *u = (lane_unit *)ud;
    /* A lane thread owns its own region stack, so this region is a root there
     * and its time shares no clock with the loop thread's rows. The report
     * marks it threads=N for exactly that reason. */
    mynah_region_thread_role("lane");
    mynah_region_begin(MYNAH_RGN_LANE_DECODE);
    u->error[0] = '\0';
    u->pcm = NULL;
    u->produced = 0;
    u->failed = 0;
    const int depth = mynah_region_depth();
    if (u->engine->decode_audio(u->ctx, u->first, u->frames, &u->pcm,
                                &u->produced, u->error, sizeof(u->error)) != 0) {
        u->pcm = NULL;
        u->produced = 0;
        u->failed = 1;
    }
    mynah_region_unwind(depth);
    mynah_region_end(MYNAH_RGN_LANE_DECODE);
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
    /* The prefill is resumable and unfinished: the slot holds a context that is
     * NOT ready for step 1 and must not be stepped, and is not finished either
     * so it must not be retired. Both loops below key on it. */
    int preparing;
    /* Admission order, for the FIFO prefill policy. Monotone per serve loop;
     * only compared, never used as an index. */
    unsigned long long prep_seq;
    /* decoder lane (E5-21). `lane_busy` means this slot owns mailbox entry
     * `index` -- a unit is running, or has finished and not been reaped.
     * `streamed_frames` is advanced at SUBMIT, not at delivery, so the range
     * the lane is decoding is never handed out twice and the next range stays
     * contiguous with it; `streamed_samples` still advances only when the PCM
     * actually reaches the sink. */
    lane_unit unit;
    int lane_busy;
} synth_slot;

static int slot_fail(synth_slot *slot, const char *message) {
    if (message != NULL) mynah_graph_error(slot->error, slot->error_capacity, message);
    slot->failed = 1;
    slot->active = 0;
    return -1;
}

/* How many tokens of prefill one slice may do, 0 = one shot (tts_engine.h,
 * `prepare_slice`).
 *
 * The default is two 16-token tiles, and it is measured rather than reasoned.
 * WITHOUT the per-step cap below, 48 beat it on both axes, because the slice pass
 * walks every preparing slot and smaller slices keep more prefills in flight so
 * one step's freeze becomes their SUM:
 *
 *     slice   stall@500  stall@250  max_gap p95  TTFA p95     (no cap, C96)
 *     0             3/18007     81        358 ms    308 ms
 *     32                  0         21        170       497
 *     48                  0         24        158       444
 *     64                  0         34        186       401
 *
 * WITH the cap that sum cannot happen, and the ranking reverses -- which is what
 * the cap predicted before it was measured (C90, ten minutes per arm):
 *
 *     slice/cap   stall@250   max_gap p95   max_gap MAX   TTFA p95
 *     48 / none        38          157 ms        336 ms     435 ms
 *     48 / 40          16          152           176        434
 *     16 / 40           0          104           122        642   <- TTFA fails
 *     16 / 80           2          107           143        637
 *     32 / 60           0          129           173        496   <- GOOD
 *
 * 32 with a 60 ms cap is the qualified point: C90 for thirty minutes, 53265
 * requests, every gate passed. `MYNAH_PREFILL_SLICE=0` restores the one-shot
 * prefill exactly, which is how the first table was measured. */
static size_t prefill_slice_budget(void) {
    static size_t cached = SIZE_MAX;
    if (cached != SIZE_MAX) return cached;
    const char *env = getenv("MYNAH_PREFILL_SLICE");
    long v = 32;
    if (env != NULL && *env != '\0') {
        char *end = NULL;
        const long parsed = strtol(env, &end, 10);
        if (end != env && parsed >= 0 && parsed < 1000000L) v = parsed;
    }
    cached = (size_t)v;
    return cached;
}

/* Validate the request and the sink, then hand everything else to the engine.
 * After this the context is ready for its first step, UNLESS the engine has a
 * resumable prefill and it was asked for one, in which case the slot comes back
 * `preparing` and the driver finishes the prefill a slice at a time. */
/* `prep_seq_next` hands each admitted request its place in the prefill queue.
 * Passed rather than static because a static counter would be shared by every
 * serve loop in the process, and prefork or not, two loops must not interleave
 * their ordering. */
static int slot_start(const mynah_tts_engine *engine, const mynah_tts_model *model,
                      mynah_engine_state *state, const mynah_engine_caps *caps,
                      synth_slot *slot, int dump,
                      unsigned long long *prep_seq_next) {
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
    if (engine->prepare_slice != NULL && prefill_slice_budget() != 0u) {
        /* Not one byte of prefill here: the whole point is that admission stops
         * being a place where the batch can lose several frame periods. */
        slot->preparing = 1;
        slot->prep_seq = (*prep_seq_next)++;
        return 0;
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

/* How much wall time one step may spend on prefill, in milliseconds; 0 removes
 * the cap.
 *
 * A PER-SLOT token budget bounds one slice, not one step. The pass below walks
 * every slot still preparing, so seven prefills landing in one worker froze it
 * for seven slices -- measured as a `max_gap` maximum of 441 ms against a single
 * slice of about 60 ms. That is why `stall_rate@250ms` did not fall with
 * concurrency between C96 and C94: lowering the load removes freezes, it does
 * not shorten them.
 *
 * A gate that demands ZERO of 53559 is not satisfied by a better distribution,
 * it is satisfied by an upper bound. This is the bound: prefill work per step is
 * capped, so the freeze cannot exceed the cap plus the slice that was already
 * running. One slice always runs even when the budget is already spent, because
 * a cap that can starve a prefill forever is a deadlock, not a bound.
 *
 * The default is 40 ms, and it is MEASURED rather than derived. That distinction
 * cost a retraction, so it is written here rather than in a note.
 *
 * This comment used to say the cap "is derived rather than tried": time the loop
 * per batch width, subtract a typical step from the frame period, and the
 * remainder is the slack a prefill may use. That produced 30 ms and 30 ms was a
 * good value -- for the kernel it was measured on. When bf16 made the AR step
 * cheaper the rule pointed the WRONG WAY: T_frame(B) became 2.9 + 6.8*B, so at
 * the modal width B7 the slack reads 28 ms and the rule says lower the cap,
 * while the box says 40 is better than 30 by sixty milliseconds of TTFA.
 *
 * The rule is wrong because a frame that overruns is a DEBT, not a stall. At
 * RTF 0.79 a slot earns (1 - 0.79) * 80 = 16 ms of lead per frame, so ten
 * milliseconds of overrun are repaid inside one frame. Slack is a fine first
 * guess; the cap is a measured quantity, and it must be re-measured whenever a
 * kernel changes what a step costs.
 *
 * The sweep that set 40, at C120 on the shipped default:
 *
 *   cap ms   TTFA p95   max_gap p95   verdict
 *      30      510.3       103 ms     MARGINAL -- TTFA
 *      40      450         121        GOOD
 *      50      448         124        GOOD  (the curve has flattened)
 *
 * Qualified at 40: C120, thirty minutes, 64205 requests, stall@250ms and
 * stall@500ms both zero, TTFA p95 447.4, RTF p95 0.794, required prebuffer p95
 * 2.8 ms, drift +0.0022 over ten windows.
 *
 * The superseded 30 ms sweep, kept because it is the evidence for the shape:
 *
 *   cap ms   max_gap p95   max_gap max   TTFA p95   (C94, ten minutes each)
 *      60        131 ms        179 ms     495-500
 *      40        121           148.6      494
 *      30        119           142.9      498
 *      20        119           133.9      496
 *
 * The earlier claim in this comment -- that too tight a cap starves prefills and
 * time to first audio pays for it -- was measured at 16-token slices and does NOT
 * survive at 32: TTFA is flat across the whole column. The reason is that the
 * per-slice budget already bounds one slot at 32 tokens, so a single prefill
 * rarely reaches 30 ms on its own; the cap binds only when several coincide on
 * one worker. It bounds the tail and leaves the median path alone.
 *
 * (that sweep was taken at C94 with the f16 backbone, and it qualified C96.) */
static double prefill_step_budget_s(void) {
    static double cached = -1.0;
    if (cached >= 0.0) return cached;
    const char *env = getenv("MYNAH_PREFILL_STEP_MS");
    double v = 40.0;
    if (env != NULL && *env != '\0') {
        char *end = NULL;
        const double parsed = strtod(env, &end);
        if (end != env && parsed >= 0.0 && parsed < 100000.0) v = parsed;
    }
    cached = v / 1000.0;
    return cached;
}

/* One slice of prefill, for as many waiting slots as the step's budget allows.
 *
 * Runs between admission and the step, so a slice and a step alternate and no
 * resident slot waits more than the budget for its next frame. A slot that
 * finishes here becomes active and is stepped in the SAME iteration, which keeps
 * the added time-to-first-audio to the slicing itself rather than to a round trip
 * through the loop.
 *
 * `*rr` rotates the starting slot so that when the budget cannot serve everyone,
 * it is a different prefill that waits each time. Without it the lowest slot
 * index would always be served and a request unlucky in its slot could be
 * starved for as long as its neighbours keep arriving. */
/* WHICH waiting prefill gets the step's budget.
 *
 * Round-robin -- the original -- is processor sharing, and processor sharing is
 * the policy that MAXIMISES the number of jobs in flight: every prefill finishes
 * near the time the last one would have, instead of in turn. FIFO to completion
 * should win the mean by construction, and the tail through Little's law, since
 * a lower mean means fewer prefills resident means less competition. The
 * per-step worst case is identical either way, because MYNAH_PREFILL_STEP_MS
 * bounds it.
 *
 * MEASURED, C120, paired ten-minute soaks on the shipped default:
 *
 *   TTFA p95   round-robin  445 ms   ->  FIFO  318 ms      -127 ms, -29%
 *
 * The suspicion that had to be checked first was head-of-line blocking -- a long
 * text's prefill delaying a short one admitted behind it -- because an aggregate
 * p95 cannot tell that apart from a real win when `long` is 12% of the bank and
 * `short` plus `medium` are 70%. The per-class report was added to
 * tools/serving_profile.py for exactly this decision, and it says the suspicion
 * was wrong:
 *
 *   class            rr p50/p95      fifo p50/p95     delta p95
 *   short            165.2/191.6     116.8/134.8      -57 ms  (-30%)
 *   medium           179.5/209.0     129.8/156.5      -53 ms  (-25%)
 *   conversational   175.4/200.6     126.0/150.2      -50 ms  (-25%)
 *   long             430.7/627.8     282.4/488.8     -139 ms  (-22%)
 *
 * EVERY class improves, and the two that carry 70% of the traffic improve by
 * more in proportion than the long ones do. Nobody pays.
 *
 * The one thing that does move the wrong way is the required client prebuffer,
 * 2 ms -> 28 ms at p95, and it is not congestion: prefills now finish sooner
 * and in groups, so more slots go active together and the step widens slightly
 * (RTF p95 0.796 -> 0.807). Twenty-eight milliseconds against a 250 ms contract,
 * with stalls still at zero.
 *
 * MYNAH_PREFILL_ORDER=rr restores the old policy exactly. */
static int prefill_fifo(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("MYNAH_PREFILL_ORDER");
        cached = (env != NULL && strcmp(env, "rr") == 0) ? 0 : 1;
    }
    return cached;
}

static void slots_prefill_slice(const mynah_tts_engine *engine,
                                mynah_engine_scratch *scratch,
                                synth_slot *slots, size_t resident_rows,
                                size_t batch_limit, int dump, size_t *rr) {
    if (resident_rows == 0u) return;
    const size_t budget = prefill_slice_budget();
    const double step_budget = prefill_step_budget_s();
    const double t0 = (step_budget > 0.0) ? mynah_phase_seconds() : 0.0;
    const int fifo = prefill_fifo();

    /* A CUDA engine can do the same prefill unit for several rows while the
     * driver still retains its scalar hook as the compatibility fallback.  The
     * selected rows are one gang, not one request repeated in a loop.  Keep the
     * batch bounded by the engine arithmetic width: the active slot capacity is
     * intentionally allowed to be wider than one microbatch. */
    if (engine->prepare_slice_batch != NULL && batch_limit > 1u) {
        if (batch_limit > MYNAH_GRAPH_MAX_JOBS) batch_limit = MYNAH_GRAPH_MAX_JOBS;
        mynah_engine_ctx *batch_ctxs[MYNAH_GRAPH_MAX_JOBS];
        size_t batch_slots[MYNAH_GRAPH_MAX_JOBS];
        int batch_done[MYNAH_GRAPH_MAX_JOBS];
        size_t batch_count = 0u;
        for (size_t pass = 0u; pass < resident_rows && batch_count < batch_limit;
             ++pass) {
            size_t selected = resident_rows;
            if (fifo) {
                unsigned long long best_seq = 0ull;
                for (size_t k = 0u; k < resident_rows; ++k) {
                    const synth_slot *candidate = &slots[k];
                    if (!candidate->in_use || !candidate->preparing) continue;
                    int already = 0;
                    for (size_t j = 0u; j < batch_count; ++j)
                        if (batch_slots[j] == k) already = 1;
                    if (already) continue;
                    if (selected == resident_rows || candidate->prep_seq < best_seq) {
                        selected = k;
                        best_seq = candidate->prep_seq;
                    }
                }
            } else {
                const size_t candidate = (*rr + pass) % resident_rows;
                if (slots[candidate].in_use && slots[candidate].preparing)
                    selected = candidate;
            }
            if (selected == resident_rows) continue;
            batch_slots[batch_count] = selected;
            batch_ctxs[batch_count] = slots[selected].ctx;
            batch_done[batch_count] = 0;
            ++batch_count;
        }
        if (batch_count > 1u) {
            char batch_error[256];
            batch_error[0] = '\0';
            const int batch_rc = engine->prepare_slice_batch(
                batch_ctxs, batch_count, budget, batch_done, scratch,
                batch_error, sizeof(batch_error));
            if (batch_rc == 0) {
                for (size_t j = 0u; j < batch_count; ++j) {
                    synth_slot *slot = &slots[batch_slots[j]];
                    if (!batch_done[j]) continue;
                    slot->preparing = 0;
                    if (dump && engine->debug_dump != NULL) {
                        engine->debug_dump(slot->ctx, "encoder");
                        engine->debug_dump(slot->ctx, "prefill");
                    }
                    slot->active = 1;
                }
                *rr = (batch_slots[batch_count - 1u] + 1u) % resident_rows;
                /* A successful batched call owns this slice.  This avoids
                 * immediately taking a remaining row through the scalar hook
                 * and gives the next service tick a chance to form a new gang. */
                return;
            }
            if (batch_rc < 0) {
                for (size_t j = 0u; j < batch_count; ++j) {
                    slots[batch_slots[j]].preparing = 0;
                    (void)slot_fail(&slots[batch_slots[j]],
                                    batch_error[0] != '\0' ? batch_error
                                                            : "batched prefill failed");
                }
                return;
            }
            /* 1 means not eligible and must not have changed any row.  The
             * established scalar loop below is the compatibility path. */
        }
    }

    size_t served = 0;
    /* FIFO keeps picking the SAME oldest prefill until it finishes, which is
     * the whole point: serving each waiting slot once per step in a different
     * order would still be processor sharing, just with the turns renamed. The
     * loop bound is generous rather than exact -- the step budget is what
     * actually stops this, and a slot that completes clears its `preparing`
     * flag so the next pick moves on by itself. */
    const size_t passes = fifo ? resident_rows * 4u : resident_rows;
    for (size_t n = 0; n < passes; ++n) {
        size_t i;
        if (fifo) {
            size_t best = resident_rows;
            unsigned long long best_seq = 0ull;
            for (size_t k = 0; k < resident_rows; ++k) {
                const synth_slot *c = &slots[k];
                if (!c->in_use || !c->preparing) continue;
                if (best == resident_rows || c->prep_seq < best_seq) {
                    best = k;
                    best_seq = c->prep_seq;
                }
            }
            if (best == resident_rows) break;   /* nothing left to prefill */
            i = best;
        } else {
            i = (*rr + n) % resident_rows;
        }
        synth_slot *slot = &slots[i];
        if (!slot->in_use || !slot->preparing) continue;
        if (step_budget > 0.0 && served > 0u &&
            (mynah_phase_seconds() - t0) >= step_budget) {
            *rr = i;                      /* resume here next step */
            break;
        }
        ++served;
        int done = 0;
        if (engine->prepare_slice(slot->ctx, budget, &done,
                                  slot->error, slot->error_capacity) != 0) {
            slot->preparing = 0;
            (void)slot_fail(slot, NULL);
            continue;
        }
        if (!done) continue;
        slot->preparing = 0;
        if (dump && engine->debug_dump != NULL) {
            engine->debug_dump(slot->ctx, "encoder");
            engine->debug_dump(slot->ctx, "prefill");
        }
        slot->active = 1;
    }
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

/* Push decoded PCM at the sink, WITHOUT moving the frame cursor.
 *
 * The split exists for the decoder lane. Inline, a range is decoded and
 * delivered in one breath and the two cursors move together. With the lane the
 * frames are charged to the slot when the decode is SUBMITTED -- otherwise the
 * next step would compute the same range as still pending and hand it out a
 * second time -- while the samples only exist once the unit comes back. */
static int slot_emit(synth_slot *slot, float *audio, size_t produced) {
    if (emit_stream_samples(slot->callback, slot->user_data, audio, produced,
                            slot->chunk_samples, slot->error,
                            slot->error_capacity) != 0) {
        return slot_fail(slot, NULL);
    }
    slot->streamed_samples += produced;
    return 0;
}

/* Hand one gang member its PCM. */
static int slot_deliver(synth_slot *slot, float *audio, size_t produced,
                        size_t frames) {
    if (slot_emit(slot, audio, produced) != 0) return -1;
    slot->streamed_frames += frames;
    return 0;
}

/* Reap this slot's lane unit: deliver its PCM and release its mailbox entry.
 *
 * `blocking` is the whole policy. The driver reaps non-blocking at the top of
 * every iteration, so a finished unit costs nothing to collect; it blocks in
 * exactly two places, and both are about THIS slot -- when this slot has a
 * full quantum waiting behind the unit and cannot go further, and when this
 * slot is about to be retired and its context freed. NO SLOT EVER BLOCKS ON
 * ANOTHER SLOT'S DECODE. That is the property that makes the lane a latency
 * win rather than a second serialization point, and it is also why
 * "never free state the lane is decoding" is enforceable at all: there is
 * exactly one place where the context goes away, and it drains first. */
static void lane_reap(synth_slot *slot, size_t index, int blocking) {
    if (!slot->lane_busy) return;
    if (!blocking && mynah_lane_finished((int)index) != 1) return;
    /* The only place the loop thread ever blocks on the lane. Non-blocking
     * reaps return above without opening the region, so this row is the real
     * stall and not the polling. */
    const int waited = blocking;
    if (waited) mynah_region_begin(MYNAH_RGN_LANE_WAIT);
    mynah_lane_wait((int)index);
    if (waited) mynah_region_end(MYNAH_RGN_LANE_WAIT);
    slot->lane_busy = 0;
    lane_unit *u = &slot->unit;
    if (u->failed) {
        slot_fail(slot, u->error[0] != '\0' ? u->error : "decoding audio failed");
    } else {
        slot_emit(slot, u->pcm, u->produced);
    }
    free(u->pcm);
    u->pcm = NULL;
    u->produced = 0;
    u->failed = 0;
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
                        const mynah_engine_step_result *results, size_t live,
                        int lane_on) {
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

    /* ---- the decoder lane (E5-21) ---------------------------------------
     *
     * WHO decodes and HOW MUCH is decided above, by the one policy, and is not
     * touched here: the lane changes only WHERE the work runs. Each member's
     * range is exactly the range the inline path would have passed, in the same
     * order, to the same `decode_audio`, so the audio is byte-identical -- the
     * lane is a placement decision, never a numerical one.
     *
     * The lane is per slot, so it cannot be combined with an engine that has a
     * real `decode_audio_batch`; `serve` refuses to turn it on for one rather
     * than silently discarding that engine's batching. Today no engine in the
     * tree has one. */
    if (lane_on) {
        for (size_t g = 0; g < count; ++g) {
            const size_t index = step_slot[member[g]];
            synth_slot *slot = &slots[index];
            /* PER-SLOT BLOCKING ONLY. This slot has accumulated a full quantum
             * behind a unit still in flight, which is exactly the bound the
             * mailbox enforces, so THIS slot waits. Every other slot's step
             * has already happened and none of them waits for this. */
            lane_reap(slot, index, 1);
            if (slot->failed) continue;
            slot->unit.engine = engine;
            slot->unit.ctx = slot->ctx;
            slot->unit.first = first[g];
            slot->unit.frames = want[g];
            if (mynah_lane_submit((int)index, lane_decode, &slot->unit) != 0) {
                /* The lane refused -- it is off, or the bound was violated and
                 * said so. Decode inline rather than drop the audio: a lane
                 * refusal is a scheduling failure and must never become a
                 * correctness one. */
                float *one = NULL;
                size_t made = 0;
                if (engine->decode_audio(slot->ctx, first[g], want[g], &one, &made,
                                         slot->error, slot->error_capacity) != 0) {
                    slot_fail(slot, NULL);
                } else {
                    slot_deliver(slot, one, made, want[g]);
                }
                free(one);
                continue;
            }
            slot->lane_busy = 1;
            /* Charged now, delivered later: see slot_emit. */
            slot->streamed_frames += want[g];
        }
        return;
    }

    char shared_error[256];
    shared_error[0] = '\0';
    /* One wide decode call standing in for `count` narrow ones. `units` is the
     * gang width, so the report can answer "did the gang ever actually gang"
     * with units/dispatches rather than with the fact that the code path
     * exists -- a banner is not a count. */
    mynah_region_begin(MYNAH_RGN_DECODE_GANG);
    mynah_region_pool_at(MYNAH_RGN_DECODE_GANG, 1, (long long)count);
    mynah_region_units_at(MYNAH_RGN_DECODE_GANG, (long long)count);
    const int gang_depth = mynah_region_depth();
    const int gang_failed =
        mynah_engine_decode_gang(engine, gang, count, first, want, pcm, produced,
                                 decode_failed, scratch, shared_error,
                                 sizeof(shared_error)) != 0;
    mynah_region_unwind(gang_depth);
    mynah_region_end(MYNAH_RGN_DECODE_GANG);
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
static int slot_finalize_inner(const mynah_tts_engine *engine, synth_slot *slot,
                               int dump);

static int slot_finalize(const mynah_tts_engine *engine, synth_slot *slot, int dump) {
    mynah_region_begin(MYNAH_RGN_FINALIZE);
    const int depth = mynah_region_depth();
    const int rc = slot_finalize_inner(engine, slot, dump);
    mynah_region_unwind(depth);
    mynah_region_end(MYNAH_RGN_FINALIZE);
    return rc;
}

static int slot_finalize_inner(const mynah_tts_engine *engine, synth_slot *slot,
                               int dump) {
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
                      mynah_engine_step_result *results, size_t live, int dump,
                      int lane_on) {
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
    stream_gang(engine, caps, scratch, slots, step_slot, results, live, lane_on);
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
                 mynah_graph_sink *sink, size_t want_batch,
                 size_t active_capacity, int strict_batch, int dump_all) {
    if (sink == NULL || sink->next_job == NULL) return -1;
    if (want_batch == 0u) return 0;
    if (want_batch > MYNAH_GRAPH_MAX_JOBS) return -1;
    if (active_capacity == 0u) active_capacity = want_batch;
    if (active_capacity > MYNAH_GRAPH_MAX_ACTIVE) return -1;
    if (active_capacity < want_batch) want_batch = active_capacity;
    if (engine == NULL) {
        return refuse_all(sink, "model.json names an engine this build does not have");
    }
    char shared_error[256];
    shared_error[0] = '\0';
    mynah_engine_state *state = NULL;
    mynah_region_begin(MYNAH_RGN_MODEL_LOAD);
    const int load_failed =
        engine->model_init(model, &state, shared_error, sizeof(shared_error)) != 0;
    mynah_region_end(MYNAH_RGN_MODEL_LOAD);
    if (load_failed) {
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
    size_t max_batch = want_batch < caps.max_batch ? want_batch : caps.max_batch;
    if (max_batch > active_capacity) max_batch = active_capacity;
    if (max_batch == 0u) {
        engine->model_free(state);
        return refuse_all(sink, "the engine cannot serve an empty microbatch");
    }
    /* The active-slot array is intentionally independent of the arithmetic
     * width. It is a capacity reservation, not a promise that one engine call
     * contains every resident request. */
    const size_t slot_capacity = active_capacity;

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

    /* ---- decoder lane (E5-21), decided once per service -----------------
     *
     * The lane must EXIST -- mynah_lane_split_prepare() refuses a split that
     * would be too narrow or that it cannot pin, so a zero here is already the
     * considered answer and not an absence of one -- and the mailbox must have
     * an entry per slot, which it does by construction. The second check is
     * here so that raising MYNAH_GRAPH_MAX_JOBS fails loudly instead of
     * writing past the mailbox.
     *
     * THE LANE AND THE DECODE GANG ARE MUTUALLY EXCLUSIVE, and that is the one
     * thing to understand before turning either on. Both attack the same cost
     * -- the codec is the largest per-slot term in a frame -- from opposite
     * directions. The gang makes one wide call where there would have been N
     * narrow ones; the lane keeps the N calls but takes them off the loop
     * thread so they overlap the next AR steps. A lane unit is per slot by
     * construction (that is what "at most one in flight per slot" means), so
     * running the lane means calling `decode_audio` per context and not
     * `decode_audio_batch`.
     *
     * That is a SCHEDULING choice and never a numerical one: the seam requires
     * `decode_audio_batch` to be bit-identical per context to `decode_audio`
     * on the same range (tts_engine.h), so whichever runs, each request gets
     * the same bytes. Which one is faster is a property of the host and the
     * model and can only be settled by measurement, so the default is OFF --
     * the gang, which is what ships -- and the lane is an explicit opt-in that
     * says out loud what it is giving up.
     *
     * Nothing about the AUDIO depends on this flag. The lane changes which
     * threads run the decode and when its result is delivered, never the
     * ranges or their order, which is why the goldens do not care. */
    const int lane_on = mynah_lane_width() > 0 &&
                        slot_capacity <= (size_t)MYNAH_LANE_SLOTS;
    if (lane_on) {
        fprintf(stderr,
                "driver: decoder lane ON (%d pinned threads). Decodes run per "
                "slot on the lane and overlap the AR steps%s\n",
                mynah_lane_width(),
                engine->decode_audio_batch != NULL
                    ? ", so this engine's BATCHED codec is not used -- the two "
                      "are alternatives, not layers, and only a measurement on "
                      "this host can say which wins"
                    : "");
    }

    const int timing = getenv("MYNAH_TIMING") != NULL;
    const double t_start = timing ? mynah_phase_seconds() : 0.0;
    double t_prep = t_start, t_ar = t_start;
    size_t admitted = 0;
    /* Rotating cursor for the prefill pass; see slots_prefill_slice. */
    size_t prefill_rr = 0;
    /* Active capacity may exceed the engine microbatch width. The step arrays
     * are intentionally sized to max_batch, so walk the resident slots in a
     * fair rotating order and submit at most one microbatch per iteration. */
    size_t step_rr = 0;
    unsigned long long prep_seq_next = 0;

    synth_slot slots[MYNAH_GRAPH_MAX_ACTIVE];
    mynah_engine_ctx *step_ctxs[MYNAH_GRAPH_MAX_JOBS];
    size_t step_slot[MYNAH_GRAPH_MAX_JOBS];
    mynah_engine_step_result results[MYNAH_GRAPH_MAX_JOBS];
    memset(slots, 0, sizeof(slots));

    int result = 0;
    size_t used = 0;      /* slots holding a request, live or just finished */
    int drained = 0;      /* the sink said there will be no more work */
    /* CUDA's batched decoder does not use the CPU lane, so its physical slot
     * array can stay dense: when a row retires, the last resident row is
     * swapped into the hole.  This is the scheduler half of a row arena; the
     * engine contexts still own their model/KV state and are never copied. */
    const int compact_rows = !lane_on;

    /* ---- serving-loop occupancy (E10-11), MYNAH_SERVE_PROFILE=1 ----------
     *
     * The cost map answers "where did the wall go". It cannot answer "how many
     * slots were live while it went there", and those are different questions:
     * a worker at RTF 0.9 with one slot live is starved, and a worker at RTF
     * 0.9 with sixteen live is full, and the fix for each is the opposite of
     * the fix for the other. Our burst-TTFA model is arithmetic that lands
     * within 4% of the measurement, which is a good model and still a model.
     *
     * Counters only, off unless asked, read once at the end. The row that
     * matters most is `no work queued`: it is a share of WALL, not of work, so
     * a large number there means the box was idle rather than saturated -- the
     * distinction that decides whether a level failed on capacity or on
     * variance. */
    const int serve_profile = getenv("MYNAH_SERVE_PROFILE") != NULL;
    size_t occ_hist[MYNAH_GRAPH_MAX_JOBS + 1u];
    size_t occ_frames = 0, occ_admits = 0, occ_free_nothing_queued = 0;
    double occ_blocked_s = 0.0;
    memset(occ_hist, 0, sizeof(occ_hist));
    /* The histogram says how many slots were live; it cannot say what a step at
     * that width COST, and above a certain concurrency that is the only question
     * left. A step must finish inside one frame period or every slot in it falls
     * behind playback together, so what decides a stall at high C is not the mean
     * width but the width at which T_frame crosses the deadline. Time and lateness
     * per width, same counters-only discipline. */
    double occ_time[MYNAH_GRAPH_MAX_JOBS + 1u];
    size_t occ_late[MYNAH_GRAPH_MAX_JOBS + 1u];
    double occ_worst[MYNAH_GRAPH_MAX_JOBS + 1u];
    memset(occ_time, 0, sizeof(occ_time));
    memset(occ_late, 0, sizeof(occ_late));
    memset(occ_worst, 0, sizeof(occ_worst));
    const double occ_deadline_s =
        (caps.frame_rate > 0.0)
            ? (double)(caps.frames_per_step ? caps.frames_per_step : 1u) / caps.frame_rate
            : 0.0;

    for (;;) {
        /* ---- reap whatever the decoder lane finished --------------------
         * Non-blocking, and first, so that a unit that completed while the
         * batch was stepping is delivered before anything else looks at the
         * slot's cursors. A slot whose unit is still running is simply left
         * alone; it is not waited for here and never on another slot's
         * account. */
        if (lane_on) {
            for (size_t i = 0; i < slot_capacity; ++i) {
                if (slots[i].in_use) lane_reap(&slots[i], i, 0);
            }
        }

        /* ---- admission ------------------------------------------------
         * At the top of the step, not before the loop. `block` is set only
         * when there is nothing else to do, so a running batch is never held
         * up waiting for an arrival that may not come. */
        const unsigned long long t_admit =
            mynah_costmap_level() ? mynah_costmap_now_ns() : 0ull;
        while (!drained && used < slot_capacity &&
               (sink->running == NULL || sink->running(sink->ud) != 0)) {
            size_t index = slot_capacity;
            if (compact_rows) {
                index = used;
            } else {
                for (size_t i = 0; i < slot_capacity; ++i) {
                    if (!slots[i].in_use) { index = i; break; }
                }
            }
            if (index == slot_capacity) break;

            mynah_graph_job job;
            memset(&job, 0, sizeof(job));
            void *tag = NULL;
            const int block = (used == 0u);
            const double t_block = (serve_profile && block) ? mynah_phase_seconds() : 0.0;
            const int got = sink->next_job(sink->ud, &job, &tag, block);
            if (serve_profile && block) occ_blocked_s += mynah_phase_seconds() - t_block;
            if (got != 1) {
                /* Nothing available. If we asked it to block and it still had
                 * nothing, the service is over.  A free slot that stayed free
                 * because the queue was empty is the loop telling us the box is
                 * ahead of its arrivals, which is the opposite of saturation. */
                if (serve_profile) ++occ_free_nothing_queued;
                if (block) drained = 1;
                break;
            }
            if (serve_profile) ++occ_admits;
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
            if (slot_start(engine, model, state, &caps, slot, dump_all,
                           &prep_seq_next) != 0) {
                /* A request that cannot start never occupies the batch. */
                if (slot_retire(engine, sink, slot, dump_all) != 0) result = -1;
                --used;
            }
        }
        /* Admission is submitted rather than bracketed: it is declared
         * "derived" in the table because the region it sits under -- the
         * request -- is itself derived, and a blocking next_job() waiting for
         * an arrival is not this service's work. The report prints the mode,
         * so this row is never read as if it had been measured on one stack
         * alongside the decode rows. */
        if (t_admit != 0ull) {
            mynah_region_add_ns(MYNAH_RGN_RT_ADMISSION,
                                mynah_costmap_now_ns() - t_admit);
        }
        if (timing && t_prep == t_start) t_prep = mynah_phase_seconds();
        if (used == 0u) break;

        /* ---- cancellation --------------------------------------------- */
        if (sink->cancelled != NULL) {
            for (size_t i = 0; i < slot_capacity; ++i) {
                /* A slot still prefilling is cancellable too, and has to be:
                 * otherwise a client that disconnects during a long prefill
                 * keeps a slot slicing to completion before anyone notices. */
                if (!slots[i].in_use || (!slots[i].active && !slots[i].preparing)) continue;
                if (sink->cancelled(sink->ud, slots[i].tag) != 0) {
                    slots[i].cancelled = 1;
                    slots[i].active = 0;
                    slots[i].preparing = 0;
                }
            }
        }

        /* ---- finish the prefills that are in flight -------------------- */
        if (engine->prepare_slice != NULL) {
            const size_t prefill_rows = compact_rows ? used : slot_capacity;
            if (prefill_rows != 0u)
                slots_prefill_slice(engine, scratch, slots, prefill_rows,
                                    max_batch, dump_all, &prefill_rr);
        }

        /* ---- one bounded step over the live set -----------------------
         * A continuous service may retain 128 request contexts while the
         * engine accepts B16 arithmetic. Never pass the resident count to
         * the fixed B16 step arrays: rotate the selected slice so every live
         * request advances without widening the engine call. */
        size_t live = 0;
        size_t next_step_rr = step_rr;
        const size_t resident_rows = compact_rows ? used : slot_capacity;
        for (size_t offset = 0; offset < resident_rows && live < max_batch;
             ++offset) {
            const size_t i = (step_rr + offset) % resident_rows;
            if (!slots[i].in_use || !slots[i].active) continue;
            step_slot[live] = i;
            step_ctxs[live] = slots[i].ctx;
            ++live;
            next_step_rr = (i + 1u) % resident_rows;
        }
        if (live > 0u) step_rr = next_step_rr;
        else step_rr = (step_rr + 1u) % resident_rows;
        if (serve_profile) {
            ++occ_frames;
            occ_hist[live <= max_batch ? live : max_batch] += 1u;
        }
        if (live > 0u) {
            const double t_step = serve_profile ? mynah_phase_seconds() : 0.0;
            step_live(engine, &caps, scratch, slots, step_ctxs, step_slot, results,
                      live, dump_all, lane_on);
            if (serve_profile) {
                const double took = mynah_phase_seconds() - t_step;
                const size_t b = live <= max_batch ? live : max_batch;
                occ_time[b] += took;
                if (took > occ_worst[b]) occ_worst[b] = took;
                if (occ_deadline_s > 0.0 && took > occ_deadline_s) ++occ_late[b];
            }
        }

        /* ---- retire, per slot, as soon as it stops ---------------------
         * Not after the whole group: the slot is the unit of capacity, and
         * holding a finished one until its neighbours catch up is exactly the
         * wait continuous admission exists to remove. */
        if (compact_rows) {
            /* Dense rows make this a real swap-remove.  Do not advance `i`
             * after the move: the last row may itself already be finished. */
            size_t i = 0u;
            while (i < used) {
                if (!slots[i].in_use || slots[i].active || slots[i].preparing) {
                    ++i;
                    continue;
                }
                if (slot_retire(engine, sink, &slots[i], dump_all) != 0)
                    result = -1;
                --used;
                if (i != used) {
                    slots[i] = slots[used];
                    memset(&slots[used], 0, sizeof(slots[used]));
                }
            }
        } else {
            for (size_t i = 0; i < slot_capacity; ++i) {
                /* `preparing` is the third state this loop has to know about:
                 * not active, and not finished either. */
                if (!slots[i].in_use || slots[i].active || slots[i].preparing)
                    continue;
                /* NEVER FREE STATE THE LANE IS DECODING. slot_retire truncates
                 * the frame history and frees the context; a unit still
                 * reading it would be reading freed memory. */
                lane_reap(&slots[i], i, 1);
                if (slot_retire(engine, sink, &slots[i], dump_all) != 0)
                    result = -1;
                --used;
            }
        }
    }
    if (serve_profile) {
        const double wall = mynah_phase_seconds() - t_start;
        size_t slot_frames = 0;
        for (size_t b = 0; b <= max_batch; ++b) slot_frames += occ_hist[b] * b;
        fprintf(stderr,
                "[SERVE] frames=%zu admitted=%zu max_batch=%zu mean_live=%.2f "
                "(1.00 = never batched)\n",
                occ_frames, occ_admits, max_batch,
                occ_frames ? (double)slot_frames / (double)occ_frames : 0.0);
        fprintf(stderr, "[SERVE] live-slot histogram, share of frames:");
        for (size_t b = 0; b <= max_batch; ++b) {
            if (occ_hist[b] == 0u) continue;
            fprintf(stderr, "  B%zu=%.1f%%", b,
                    100.0 * (double)occ_hist[b] / (double)(occ_frames ? occ_frames : 1u));
        }
        fprintf(stderr, "\n");
        /* T_frame(B) = a + b*B, read off the serving loop rather than modelled.
         * `late` is the share of steps at that width that overran the frame
         * period: a width whose MEAN is inside the deadline can still be the
         * one producing every stall, so both columns are printed. */
        if (occ_deadline_s > 0.0) {
            fprintf(stderr,
                    "[SERVE] step cost per width (deadline %.1f ms = %u frame(s) "
                    "at %.2f Hz)\n", occ_deadline_s * 1e3,
                    caps.frames_per_step ? caps.frames_per_step : 1u, caps.frame_rate);
            for (size_t b = 1; b <= max_batch; ++b) {
                if (occ_hist[b] == 0u) continue;
                fprintf(stderr,
                        "[SERVE]   B%-2zu  n=%-8zu mean %6.1f ms  worst %7.1f ms  "
                        "late %5.2f%%  per-slot %5.1f ms\n",
                        b, occ_hist[b], 1e3 * occ_time[b] / (double)occ_hist[b],
                        1e3 * occ_worst[b],
                        100.0 * (double)occ_late[b] / (double)occ_hist[b],
                        1e3 * occ_time[b] / (double)occ_hist[b] / (double)b);
            }
        }
        fprintf(stderr,
                "[SERVE] a free slot found no work queued %zu times; the loop "
                "spent %.1f%% of wall blocked waiting for an arrival -- high "
                "here means the box is IDLE, not saturated\n",
                occ_free_nothing_queued, wall > 0.0 ? 100.0 * occ_blocked_s / wall : 0.0);
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
    return serve(engine, model, sink, max_batch, max_batch, strict_batch, 0);
}

int mynah_graph_serve_continuous(const mynah_tts_model *model, size_t max_batch,
                                 mynah_graph_sink *sink) {
    return mynah_graph_serve_continuous_capacity(model, max_batch, 0u, sink);
}

int mynah_graph_serve_continuous_capacity(const mynah_tts_model *model,
                                          size_t max_batch,
                                          size_t active_capacity,
                                          mynah_graph_sink *sink) {
    if (model == NULL) return -1;
    if (max_batch == 0u) max_batch = 1u;
    if (max_batch > MYNAH_GRAPH_MAX_JOBS) max_batch = MYNAH_GRAPH_MAX_JOBS;
    if (active_capacity == 0u) active_capacity = max_batch;
    if (active_capacity > MYNAH_GRAPH_MAX_ACTIVE)
        active_capacity = MYNAH_GRAPH_MAX_ACTIVE;
    return serve(mynah_engine_lookup(model->info.engine), model, sink,
                 max_batch, active_capacity, 0, 0);
}

/* See mynah_tts.h. Builds the model-owned caches and throws the rest away.
 *
 * The per-ENGINE-STATE caches -- the quantized weight cache above all -- are
 * freed with the state, so this does not pre-build those: they are per worker
 * by construction today, and making them shared is a different change with a
 * different ownership question. What survives is what the model owns, which is
 * the dtype conversion cache, and that is the larger of the two. */
int mynah_tts_model_warm(mynah_tts_model *model, char *error,
                         size_t error_capacity) {
    if (model == NULL) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "warm: no model");
        return -1;
    }
    const mynah_tts_engine *engine = mynah_engine_lookup(model->info.engine);
    if (engine == NULL || engine->model_init == NULL ||
        engine->model_free == NULL) {
        /* Not an error: an engine without the seam simply has nothing to warm,
         * and the caller is no worse off than before it asked. */
        return 0;
    }
    mynah_engine_state *state = NULL;
    char local[256];
    local[0] = '\0';
    if (engine->model_init(model, &state, local, sizeof local) != 0) {
        if (error != NULL && error_capacity > 0)
            snprintf(error, error_capacity, "warm: %s", local);
        return -1;
    }
    engine->model_free(state);
    return 0;
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
    return serve(mynah_engine_lookup(model->info.engine), model, &sink, count,
                 count, 1, count == 1u);
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
