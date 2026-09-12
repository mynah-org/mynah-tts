/* mynah-tts HTTP server — OpenAI-compatible speech synthesis.
 *
 *   POST /v1/audio/speech   OpenAI shape: {model,input,voice,response_format,speed}
 *   GET  /v1/voices         the pack's speakers, by id and name
 *   GET  /v1/models         OpenAI-shaped model listing
 *   GET  /health            liveness
 *
 * Concurrency note, stated plainly because it shapes the design: a
 * mynah_tts_model carries mutable caches (quantized weights, codec filters,
 * projection cache) that synthesis populates, so it is NOT safe to synthesize
 * on one model from several threads. Connections are accepted and parsed
 * concurrently; synthesis happens on one thread only.
 *
 * That single thread is not a bottleneck the way a mutex was. A decode step is
 * bound by the weight bytes it reads, not by arithmetic, so requests taking
 * turns each paid their own trip to memory for the same weights. The scheduler
 * below instead runs mynah_graph_serve_continuous, which reads those weights
 * once per step and serves every request in flight from that one read, with
 * each receiving byte-identical audio to what it would have received alone.
 *
 * Continuous means the batch has an admission point: the driver asks this file
 * for more work at the top of every decoder step, so a request that arrives
 * while four others are mid-utterance joins them on the next step instead of
 * waiting for the group to drain. Its cost to the others is the width of the
 * step, not the length of their request.
 *
 * STREAMING IS NOT A SECOND PATH. A streaming request is the same job with a
 * callback sink instead of a buffer sink, queued in the same queue and stepped
 * in the same batch (CLAUDE.md rule 7). That is what removed the global
 * synthesis mutex: there is nothing left to serialize, because nothing but the
 * scheduler thread ever enters the model. The invariant is asserted rather
 * than commented -- see synth_assert_scheduler().
 *
 * Socket I/O for a stream does NOT run on the synthesis thread: the callback
 * only copies PCM into a bounded queue and a dedicated writer thread owns the
 * fd from there on (server/stream_out.c). A client that stops reading loses its
 * own stream and nothing else.
 *
 * Three things follow from that split, and they are the shape of this file:
 *
 *   - HTTP parsing and synthesis are separate roles. A worker reads the
 *     request, validates it, tokenizes it and builds a job; the scheduler does
 *     nothing but synthesize. A worker never waits on a stream: it hands the
 *     job over and goes back to parsing.
 *   - A job is a heap object with a refcount, and it owns the client
 *     descriptor. Two threads hold it and, with a deadline, either may be the
 *     last to let go. The descriptor leaves the job exactly once, through an
 *     atomic claim, which is what makes a double close impossible and a leaked
 *     descriptor detectable.
 *   - A client that goes away takes its slot with it. The peer hangup is
 *     detected on the socket rather than inferred from a failed write, and it
 *     is observed at the frame boundary through the same cancellation callback
 *     a deadline uses, so a slot is freed within one decoder step and never in
 *     the middle of one. Cancellation is asked before any other policy and
 *     nothing can overrule it: work with no consumer has nothing to trade off
 *     against. On by default; --no-cancel-on-disconnect turns it off and
 *     /health says which is in force.
 *   - Every request has a wall-clock deadline (--request-timeout-ms). A
 *     request past it is answered 504 and its worker is freed; a queued job
 *     past it is never synthesized at all, and one already in the batch is
 *     cancelled out of it rather than finishing for nobody.
 */
#include "http_util.h"
#include "prefork.h"
#include "stream_out.h"

#include "graph.h"
#include "mynah_tts.h"
#include "tokenizer_sentencepiece.h"
#include "tokenizer.h"
#include "threads.h"

#include <assert.h>

#include <errno.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define MAX_BODY (1u << 20)          /* 1 MiB of JSON is far beyond any prompt */
#define MAX_TEXT 8192u
#define STREAM_CHUNK 4096u           /* samples per streamed chunk */
#define CONN_QUEUE_CAP 256u          /* connections parked before we shed load */
#define CLIENT_TIMEOUT_S 30          /* recv deadline, so a silent client cannot pin a worker */
#define JOB_QUEUE_CAP 256u           /* jobs parked before we answer 503 */
#define REQUEST_TIMEOUT_MS 300000u   /* wall-clock ceiling per request; 0 disables */

typedef struct {
    char name[64];
    unsigned id;
} voice_entry;

static struct {
    mynah_tts_model *model;
    mynah_tokenizer *tokenizer;   /* Magpie: per-language G2P + vocabularies */
    mynah_sp *sp;                 /* PocketTTS: one SentencePiece model per pack */
    mynah_tts_model_info info;
    char model_id[128];
    voice_entry voices[64];
    size_t voice_count;
    unsigned default_speaker;
    int worker_count;
    size_t max_batch;
    size_t max_pending;
    unsigned request_timeout_ms;
    int cancel_on_disconnect;   /* default on; --no-cancel-on-disconnect turns it off */

    /* ---- language residency (E5-9) ----
     *
     * THIS process holds exactly one pack, so it holds exactly one set of
     * weights, so every batch it forms is one language. That is not a rule
     * anything here enforces -- it is the shape of the process, and it is why
     * there is no per-slot language field anywhere below to get mixed.
     *
     * `language` is what the pack declares its weights are bound to, or "" for
     * a pack whose weights are not language-specific (one Magpie pack serves
     * twelve languages, choosing only a tokenizer). Empty means "serve any
     * language the tokenizer knows"; non-empty means "refuse anything else",
     * and refusing is the entire fix: before this, a PocketTTS server answered
     * 200 to a request for a language it did not hold, with audio from the
     * wrong model. */
    const char *language;       /* never NULL; "" when the pack is unbound */
    int         pack_index;     /* which -m this worker kept; 0 single-pack  */
    int         fleet_packs;    /* how many the fleet holds; 1 = one language */
} g;

/* The synthesis thread, published once by the scheduler before it enters the
 * driver. Nothing locks the model any more, so the property that makes that
 * safe -- exactly one thread inside it -- is checked instead of described:
 * every callback the driver makes into this file asserts it is running here.
 * A future change that synthesizes from an HTTP worker again fails loudly on
 * the first chunk rather than corrupting a cache silently. */
static pthread_t g_synth_thread;
static atomic_int g_synth_thread_set;

static void synth_assert_scheduler(void) {
    assert(atomic_load(&g_synth_thread_set) != 0 &&
           pthread_equal(pthread_self(), g_synth_thread) != 0);
}

/* Counters published by /health. Plain relaxed atomics: they are diagnostics,
 * not synchronization, and a reader wants a cheap snapshot rather than a
 * consistent one. */
static struct {
    atomic_ulong queued;        /* gauge: jobs waiting for the scheduler */
    atomic_ulong active;        /* gauge: jobs inside a running batch */
    atomic_ulong completed;
    atomic_ulong rejected;      /* queue full, at either queue */
    atomic_ulong timed_out;     /* deadline passed before the audio did */
    /* Kept apart from timed_out on purpose. A deadline is the server giving
     * up on a client; a disconnect is the client giving up on the server, and
     * an operator reading one as the other would tune the wrong number. It is
     * also the only direct evidence that cancel-on-disconnect is doing
     * anything, so the disconnect test has something to assert on. */
    atomic_ulong disconnected;  /* peer hung up; slot freed at a frame boundary */
    atomic_ulong failed;
    atomic_ulong streams_active;
    atomic_ulong streams_total;
    /* Its own counter, never folded into `rejected`. A capacity refusal says
     * "come back"; this one says "this server will never serve that, look at
     * /health". An operator who sees these climbing has a routing or a
     * residency problem, not a load problem, and the two have opposite fixes. */
    atomic_ulong language_refused;
} g_stats;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* ------------------------------------------------------- synthesis jobs
 *
 * An offline request that reaches synthesis is a heap `synth_job`, never a
 * stack ticket. Two threads look at one: the HTTP worker that built it and the
 * scheduler that synthesizes it. They used to be in lockstep -- the worker
 * blocked until the scheduler was done -- so a stack object was sound. With a
 * wall-clock deadline that stops being true: the worker can walk away while
 * the scheduler still holds the pointer. An object with two independent owners
 * needs a refcount, so it has one, and the last owner out frees it.
 *
 * The job also owns the client descriptor, and that is the part worth being
 * pedantic about, because the two classic failures here are both silent:
 *
 *   - the DOUBLE CLOSE. Two owners each close; by the time the second one
 *     runs, that small integer has been recycled by accept() and the close
 *     lands on an unrelated client. Ruled out by `job_claim_fd`, an atomic
 *     exchange: exactly one caller ever receives the descriptor, and every
 *     later caller gets -1 and must not touch it. Every response path for a
 *     job goes through a claim -- there is no other way to reach the socket.
 *
 *   - the LEAK. Each owner assumes the other answered, and the descriptor is
 *     never closed; the process runs out of them under load, which shows up
 *     far from the cause. Ruled out by `job_release`: when the last reference
 *     goes and nobody ever claimed the descriptor, it closes it and says so on
 *     stderr. That line should never appear; if it does, a path returned
 *     without answering a client.
 *
 * Jobs are drained by one scheduler thread, which no longer collects a batch
 * and then runs it: it hands this queue to the driver as a sink and the driver
 * takes what it can fit at the top of every step. There is therefore no batch
 * window to tune and no "next batch" to wait for -- a request waits for a free
 * slot, and for nothing else. */
static void send_error(int fd, const char *status, const char *type,
                       const char *message);

/* A stream's sink: the writer it feeds, and the wall clock it must respect.
 * Lives inside the job, because with the worker gone the job is the only owner
 * left by the time the first chunk is produced. */
typedef struct {
    stream_out *out;
    double deadline_ms;      /* monotonic, 0 when no deadline was configured */
    int expired;
    /* Written and read only on the scheduler thread (sink_cancelled and
     * sink_on_done are both driver callbacks), so it needs no atomic. It
     * exists so the outcome can say WHY the stream stopped: a client that
     * hung up and a deadline that expired are different operational events. */
    int peer_gone;
} stream_sink;

typedef struct synth_job {
    /* Filled in before the job is published; read-only from then on. */
    mynah_tts_request request;
    int *text_ids;                 /* owned: the request points into this */
    int want_pcm;

    /* Streaming requests differ from batch ones in their sink and in nothing
     * else: same queue, same scheduler, same batch. The response header is
     * built by the worker and sent by the writer the scheduler starts at
     * admission, so a client learns it has been admitted when it has, rather
     * than when the first PCM arrives. */
    int is_stream;
    char header[512];
    stream_sink sink;              /* sink.out is set at admission */

    int fd;                        /* owned; leaves only through job_claim_fd */
    atomic_int fd_claimed;

    /* Set when the submitter walked away (deadline). Read by the scheduler on
     * every step to drop the request out of the batch, so it is an atomic
     * rather than a field behind `mu`: a mutex per slot per decoder step is a
     * lock in the hot loop, which rule 4 rules out. */
    atomic_int gave_up;

    /* Written by the scheduler, read by the submitter once `done` is set. */
    float *samples;                /* owned: freed by job_release */
    size_t count;
    char error[256];
    int result;

    /* One condvar per job rather than one broadcast for all of them. At eight
     * requests the difference is noise; at sixty-four, a broadcast wakes every
     * waiting worker on every completion so that sixty-three of them can look
     * at a flag and go back to sleep. */
    pthread_mutex_t mu;
    pthread_cond_t done_cv;
    int done;
    int abandoned;                 /* submitter gave up: discard the result */

    atomic_int refs;
    struct synth_job *next;
} synth_job;

static struct {
    pthread_mutex_t mu;
    pthread_cond_t arrived;
    synth_job *head;
    synth_job *tail;
    size_t pending;
    int stop;
    atomic_int stopping;   /* the same flag, readable without the mutex */
    pthread_t thread;
    int thread_started;
} g_batch;

static synth_job *job_new(int fd) {
    synth_job *j = (synth_job *)calloc(1, sizeof(*j));
    if (j == NULL) return NULL;
    j->fd = fd;
    atomic_init(&j->fd_claimed, 0);
    atomic_init(&j->gave_up, 0);
    atomic_init(&j->refs, 1);
    if (pthread_mutex_init(&j->mu, NULL) != 0) { free(j); return NULL; }
    if (pthread_cond_init(&j->done_cv, NULL) != 0) {
        pthread_mutex_destroy(&j->mu);
        free(j);
        return NULL;
    }
    return j;
}

/* The descriptor, once, to the first caller. -1 to everyone after that. */
static int job_claim_fd(synth_job *j) {
    if (j == NULL) return -1;
    if (atomic_exchange(&j->fd_claimed, 1) != 0) return -1;
    return j->fd;
}

static void job_retain(synth_job *j) {
    atomic_fetch_add(&j->refs, 1);
}

static void job_release(synth_job *j) {
    if (j == NULL) return;
    if (atomic_fetch_sub(&j->refs, 1) != 1) return;
    const int fd = job_claim_fd(j);
    if (fd >= 0) {
        /* The safety net described above: a path dropped the job without
         * answering. Close rather than leak, and make the noise audible. */
        fprintf(stderr, "job freed with an unanswered connection: closing fd %d\n", fd);
        close(fd);
    }
    /* The connection this job owned is finished with. In a prefork worker
     * that is the parent's only signal that the slot is free, and it must
     * happen exactly once per connection: a job is built exactly once per
     * connection that gets past validation, and this is its last reference, so
     * this line is that "once". Its counterpart, for a connection that never
     * became a job, is conn_close(). Outside a prefork worker it is a no-op. */
    mynah_prefork_conn_done();
    free(j->text_ids);
    mynah_tts_free_samples(j->samples);
    pthread_cond_destroy(&j->done_cv);
    pthread_mutex_destroy(&j->mu);
    free(j);
}

/* Answers on the job's descriptor and closes it. Safe to call from any path:
 * if someone already answered, the claim fails and this does nothing. */
static void job_respond_error(synth_job *j, const char *status,
                              const char *type, const char *message) {
    const int fd = job_claim_fd(j);
    if (fd < 0) return;
    send_error(fd, status, type, message);
    close(fd);
}

/* 0 when the job is queued (and the queue holds a reference), -1 when the
 * queue is full or shutting down and the caller must answer 503. */
static int job_enqueue(synth_job *j) {
    pthread_mutex_lock(&g_batch.mu);
    if (g_batch.stop || g_batch.pending >= g.max_pending) {
        pthread_mutex_unlock(&g_batch.mu);
        return -1;
    }
    job_retain(j);
    j->next = NULL;
    if (g_batch.tail == NULL) g_batch.head = j;
    else g_batch.tail->next = j;
    g_batch.tail = j;
    ++g_batch.pending;
    atomic_store(&g_stats.queued, (unsigned long)g_batch.pending);
    pthread_cond_signal(&g_batch.arrived);
    pthread_mutex_unlock(&g_batch.mu);
    return 0;
}

/* Waits for the scheduler, at most `timeout_ms` (0 waits forever). Returns 0
 * with the result in the job, or -1 having marked it abandoned -- after which
 * the job belongs to the scheduler alone and its result is discarded. */
static int job_wait(synth_job *j, unsigned timeout_ms) {
    struct timespec deadline;
    memset(&deadline, 0, sizeof(deadline));
    if (timeout_ms > 0u) {
        /* CLOCK_REALTIME because that is what a default-attribute condvar
         * measures its timeout against. */
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += (time_t)(timeout_ms / 1000u);
        deadline.tv_nsec += (long)(timeout_ms % 1000u) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            deadline.tv_nsec -= 1000000000L;
            ++deadline.tv_sec;
        }
    }
    int expired = 0;
    pthread_mutex_lock(&j->mu);
    while (!j->done && !expired) {
        if (timeout_ms == 0u) {
            pthread_cond_wait(&j->done_cv, &j->mu);
        } else if (pthread_cond_timedwait(&j->done_cv, &j->mu, &deadline) == ETIMEDOUT) {
            expired = !j->done;   /* a completion that raced the timeout wins */
        }
    }
    /* Set under the same mutex the scheduler takes to publish `done`, so the
     * two can never both believe they own the outcome. The atomic mirror is
     * what the scheduler polls per step; it is only ever set, never cleared,
     * so reading it without the mutex cannot produce a false cancellation. */
    if (expired) {
        j->abandoned = 1;
        atomic_store(&j->gave_up, 1);
    }
    pthread_mutex_unlock(&j->mu);
    return expired ? -1 : 0;
}

/* Publishes a finished (or discarded) job and drops the queue's reference. */
static void job_finish(synth_job *j, int abandoned_result, const char *abandoned_error) {
    pthread_mutex_lock(&j->mu);
    if (abandoned_error != NULL) {
        j->result = abandoned_result;
        snprintf(j->error, sizeof(j->error), "%s", abandoned_error);
    }
    j->done = 1;
    pthread_cond_signal(&j->done_cv);
    pthread_mutex_unlock(&j->mu);
    job_release(j);
}

/* --------------------------------------------------------------- streaming */

/* The whole streaming sink: hand the samples to the writer thread and return.
 * The driver already emits stable causal prefixes, so one call is exactly what
 * has become final; the writer decides how that maps onto HTTP chunks.
 *
 * Returning -1 aborts this request -- and only this request -- which is what a
 * full queue, a dead socket or an expired deadline must do: backpressure here
 * is cancellation, never a blocking write. Runs on the scheduler thread, like
 * everything else the driver calls, and says so. */
static int stream_callback(const float *samples, size_t count, void *user_data) {
    stream_sink *sink = (stream_sink *)user_data;
    synth_assert_scheduler();
    if (sink->deadline_ms > 0.0 && now_ms() > sink->deadline_ms) {
        sink->expired = 1;
        return -1;
    }
    if (count == 0) return stream_out_failed(sink->out) ? -1 : 0;
    return stream_out_enqueue(sink->out, samples, count);
}

/* ------------------------------------------------------- the driver's sink
 *
 * These four calls are the whole interface between the HTTP side of this file
 * and synthesis. The driver owns the loop; this owns the queue. */

/* Admission. Returns 1 having filled `job` and `tag`, or 0 when there is
 * nothing to admit -- which, when the driver asked us to block, means the
 * server is stopping and is what ends the driver.
 *
 * Two things happen here rather than earlier, and both are deliberate:
 * a queued request whose deadline has already passed is dropped without ever
 * being synthesized, and a streaming request's writer is started now, so its
 * response header goes out when it is admitted rather than when it was
 * parsed. A client's first byte therefore means "you have a slot". */
static int sink_next_job(void *ud, mynah_graph_job *job, void **tag, int block) {
    (void)ud;
    synth_assert_scheduler();
    for (;;) {
        pthread_mutex_lock(&g_batch.mu);
        while (block && g_batch.head == NULL && !g_batch.stop) {
            pthread_cond_wait(&g_batch.arrived, &g_batch.mu);
        }
        synth_job *j = g_batch.stop ? NULL : g_batch.head;
        if (j == NULL) {
            pthread_mutex_unlock(&g_batch.mu);
            return 0;
        }
        g_batch.head = j->next;
        if (g_batch.head == NULL) g_batch.tail = NULL;
        j->next = NULL;
        --g_batch.pending;
        atomic_store(&g_stats.queued, (unsigned long)g_batch.pending);
        pthread_mutex_unlock(&g_batch.mu);

        /* Nobody is waiting for this any more: the slot it would have taken
         * goes to a request someone still wants. */
        if (atomic_load(&j->gave_up) != 0) {
            job_release(j);
            continue;
        }

        if (j->is_stream) {
            const int fd = job_claim_fd(j);
            if (fd < 0) { job_release(j); continue; }
            j->sink.out = stream_out_start(fd, j->header);
            if (j->sink.out == NULL) {
                send_error(fd, "500 Internal Server Error", "server_error",
                           "cannot start the stream writer");
                close(fd);
                atomic_fetch_add(&g_stats.failed, 1ul);
                job_release(j);
                continue;
            }
            atomic_fetch_add(&g_stats.streams_active, 1ul);
        }

        memset(job, 0, sizeof(*job));
        job->request = &j->request;
        if (j->is_stream) {
            job->callback = stream_callback;
            job->user_data = &j->sink;
            job->chunk_samples = STREAM_CHUNK;
        } else {
            job->samples = &j->samples;
            job->sample_count = &j->count;
        }
        job->error = j->error;
        job->error_capacity = sizeof(j->error);
        *tag = j;
        atomic_fetch_add(&g_stats.active, 1ul);
        return 1;
    }
}

/* Completion, exactly once per admitted request. A streaming job has no
 * submitter left to wake -- its worker went back to parsing the moment the job
 * was queued -- so this is where its response is finished and its reference
 * dropped. */
static void sink_on_done(void *ud, void *tag, int result) {
    (void)ud;
    synth_job *j = (synth_job *)tag;
    synth_assert_scheduler();
    atomic_fetch_sub(&g_stats.active, 1ul);

    if (j->is_stream) {
        stream_out *out = j->sink.out;
        stream_out_finish(out);
        atomic_fetch_sub(&g_stats.streams_active, 1ul);
        stream_out_stats stats;
        memset(&stats, 0, sizeof(stats));   /* the getter is a no-op on NULL */
        stream_out_get_stats(out, &stats);
        /* A mid-stream failure cannot become an HTTP status -- the header is
         * long gone -- so the writer truncates the body and the reason is
         * logged rather than sent. An aborted stream is never silent. */
        if (j->sink.expired) {
            atomic_fetch_add(&g_stats.timed_out, 1ul);
            fprintf(stderr, "stream hit the %u ms deadline after %zu bytes; "
                            "slot released\n", g.request_timeout_ms, stats.sent_bytes);
        } else if (j->sink.peer_gone) {
            /* The event cancel-on-disconnect exists for, logged as its own
             * line so a disconnect under load is distinguishable from a
             * timeout in a log nobody was watching live. */
            atomic_fetch_add(&g_stats.disconnected, 1ul);
            fprintf(stderr, "client disconnected after %zu bytes; stream "
                            "cancelled at the frame boundary and the slot freed\n",
                    stats.sent_bytes);
        } else if (result == MYNAH_GRAPH_CANCELLED) {
            atomic_fetch_add(&g_stats.timed_out, 1ul);
            fprintf(stderr, "stream cancelled after %zu bytes (queue peak %zu, "
                            "%lu refused chunks)\n",
                    stats.sent_bytes, stats.peak_bytes, stats.failed_enqueues);
        } else if (result != MYNAH_GRAPH_OK && !stats.failed) {
            atomic_fetch_add(&g_stats.failed, 1ul);
            fprintf(stderr, "stream failed after %lu queued chunks: %s\n",
                    stats.enqueued_chunks, j->error);
        } else if (result == MYNAH_GRAPH_OK) {
            atomic_fetch_add(&g_stats.completed, 1ul);
        }
        j->sink.out = NULL;
        stream_out_release(out);
        job_release(j);
        return;
    }

    if (result == MYNAH_GRAPH_OK) atomic_fetch_add(&g_stats.completed, 1ul);
    else if (result == MYNAH_GRAPH_CANCELLED) atomic_fetch_add(&g_stats.timed_out, 1ul);
    else atomic_fetch_add(&g_stats.failed, 1ul);
    /* After this the submitter may free the job: read nothing from it here. */
    job_finish(j, result == MYNAH_GRAPH_OK ? 0 : -1, NULL);
}

/* Polled once per request per decoder step. A request nobody is listening for
 * any more leaves the batch within one frame instead of finishing for an empty
 * socket -- and is reported as cancelled, not as a synthesis failure.
 *
 * This is the ONLY preemption point, and that is deliberate. The driver calls
 * this between steps, never inside one, so a slot that answers "cancelled" is
 * retired by slot_retire after the step it is already in has completed. No
 * state a decoder is mid-way through is ever taken from under it; the cost of
 * a disconnect is therefore bounded by one frame, and never by a torn buffer.
 *
 * Cancellation also WINS over everything else: it is asked first and nothing
 * downstream can overrule it. A deadline, a queue policy or an admission
 * decision can only ever delay work, whereas this says the work has no
 * consumer left -- there is nothing to trade off against. */
static int sink_cancelled(void *ud, void *tag) {
    (void)ud;
    synth_job *j = (synth_job *)tag;
    synth_assert_scheduler();

    /* The submitter walked away (batch deadline). Cheapest check, and it
     * covers both request shapes. */
    if (atomic_load(&j->gave_up) != 0) return 1;
    if (!j->is_stream || j->sink.out == NULL) return 0;

    /* Asked BEFORE the generic failed check, because both can be true at once
     * and only this one knows which. The writer usually beats this poll to the
     * discovery -- it is the thread with the socket in its hands -- so a
     * hangup normally arrives here as an already-recorded EPIPE rather than as
     * a live POLLHUP. Checking "failed" first would collapse the two and
     * report a disconnect as a timeout. */
    if (g.cancel_on_disconnect && stream_out_peer_gone(j->sink.out)) {
        j->sink.peer_gone = 1;
        return 1;
    }

    /* Dead for some other reason: a full queue, a send timeout, out of memory.
     * Still a cancellation, just not a disconnect. Note this is deliberately
     * NOT behind the flag: --no-cancel-on-disconnect suppresses the proactive
     * hangup poll, and never the fact that a stream with no working socket has
     * nowhere left to put its audio. */
    return stream_out_failed(j->sink.out);
}

/* Whether to keep admitting. Requests already in flight always finish. */
static int sink_running(void *ud) {
    (void)ud;
    return atomic_load(&g_batch.stopping) == 0;
}

/* The one thread that enters the model, for the whole life of the process. */
static void *scheduler_main(void *arg) {
    (void)arg;
    g_synth_thread = pthread_self();
    atomic_store(&g_synth_thread_set, 1);
    /* The one thread inside the model. Named so that a `top -H` on a busy
     * server answers "which thread is hot" without a debugger, which is the
     * whole point of naming the other three as well. */
    mynah_thread_set_name("mynah-sched");

    mynah_graph_sink sink;
    memset(&sink, 0, sizeof(sink));
    sink.next_job = sink_next_job;
    sink.on_done = sink_on_done;
    sink.cancelled = sink_cancelled;
    sink.running = sink_running;
    /* Returns -1 if any single request failed, which is routine; the only
     * interesting case is coming back before anyone asked it to stop. */
    (void)mynah_graph_serve_continuous(g.model, g.max_batch, &sink);
    if (atomic_load(&g_batch.stopping) == 0) {
        fprintf(stderr, "the synthesis driver stopped on its own; "
                        "queued requests will be refused\n");
        pthread_mutex_lock(&g_batch.mu);
        g_batch.stop = 1;
        pthread_mutex_unlock(&g_batch.mu);
        atomic_store(&g_batch.stopping, 1);
    }
    return NULL;
}

/* Whatever never reached the driver, answered rather than dropped. Runs after
 * the scheduler has been joined, so nothing else touches the queue. */
static void drain_pending_jobs(void) {
    for (;;) {
        pthread_mutex_lock(&g_batch.mu);
        synth_job *j = g_batch.head;
        if (j != NULL) {
            g_batch.head = j->next;
            if (g_batch.head == NULL) g_batch.tail = NULL;
            j->next = NULL;
            --g_batch.pending;
        }
        pthread_mutex_unlock(&g_batch.mu);
        if (j == NULL) break;
        job_respond_error(j, "503 Service Unavailable", "server_error",
                          "server is shutting down");
        job_finish(j, -1, "server is shutting down");
    }
    atomic_store(&g_stats.queued, 0ul);
}

/* ------------------------------------------------------------------ voices */

/* speakers.json is a flat {"Name": id} object. Parsed here rather than in the
 * core so the runtime keeps no opinion about presentation names. */
static void load_voices(const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/speakers.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (f == NULL) return;
    char buf[8192];
    const size_t n = fread(buf, 1, sizeof(buf) - 1u, f);
    fclose(f);
    buf[n] = '\0';

    /* Two shapes in the wild: Magpie writes a flat {"Name": id} object, the
     * PocketTTS converter writes {"voices": [{"name": ..., ...}, ...]} where the
     * index in the array is the id. Detect rather than assume, because guessing
     * wrong here silently maps every request to voice 0. */
    const char *voices_key = strstr(buf, "\"voices\"");
    if (voices_key != NULL) {
        const char *p = voices_key;
        unsigned index = 0;
        while (g.voice_count < sizeof(g.voices) / sizeof(g.voices[0])) {
            const char *name_key = strstr(p, "\"name\"");
            if (name_key == NULL) break;
            const char *q = strchr(name_key + 6, '"');
            if (q == NULL) break;
            const char *end = strchr(q + 1, '"');
            if (end == NULL) break;
            const size_t len = (size_t)(end - q - 1);
            if (len < sizeof(g.voices[0].name)) {
                memcpy(g.voices[g.voice_count].name, q + 1, len);
                g.voices[g.voice_count].name[len] = '\0';
                g.voices[g.voice_count].id = index;
                ++g.voice_count;
            }
            ++index;
            p = end + 1;
        }
        return;
    }

    const char *p = buf;
    while (g.voice_count < sizeof(g.voices) / sizeof(g.voices[0])) {
        const char *q = strchr(p, '"');
        if (q == NULL) break;
        const char *end = strchr(q + 1, '"');
        if (end == NULL) break;
        const size_t len = (size_t)(end - q - 1);
        const char *colon = strchr(end, ':');
        if (colon == NULL) break;
        char *stop = NULL;
        const long id = strtol(colon + 1, &stop, 10);
        if (stop == colon + 1 || id < 0) { p = end + 1; continue; }
        if (len < sizeof(g.voices[0].name)) {
            memcpy(g.voices[g.voice_count].name, q + 1, len);
            g.voices[g.voice_count].name[len] = '\0';
            g.voices[g.voice_count].id = (unsigned)id;
            ++g.voice_count;
        }
        p = stop;
    }
}

static int ci_equal(const char *a, const char *b) {
    for (; *a != '\0' && *b != '\0'; ++a, ++b) {
        int ca = (*a >= 'A' && *a <= 'Z') ? *a - 'A' + 'a' : *a;
        int cb = (*b >= 'A' && *b <= 'Z') ? *b - 'A' + 'a' : *b;
        if (ca != cb) return 0;
    }
    return *a == '\0' && *b == '\0';
}

/* Accepts a voice name ("Sofia", case-insensitive) or a numeric id ("4"). */
static int resolve_voice(const char *voice, unsigned *out) {
    if (voice == NULL || voice[0] == '\0') { *out = g.default_speaker; return 0; }
    for (size_t i = 0; i < g.voice_count; ++i) {
        if (ci_equal(voice, g.voices[i].name)) { *out = g.voices[i].id; return 0; }
    }
    char *end = NULL;
    const long id = strtol(voice, &end, 10);
    if (end != voice && *end == '\0' && id >= 0 &&
        (unsigned)id < g.info.speaker_count) {
        *out = (unsigned)id;
        return 0;
    }
    return -1;
}

/* -------------------------------------------------------------------- http */

static int write_all(int fd, const void *data, size_t len) {
    const char *p = (const char *)data;
    while (len > 0) {
        const ssize_t n = send(fd, p, len, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static void send_status(int fd, const char *status, const char *ctype,
                        const char *body, size_t len) {
    char head[512];
    const int n = snprintf(head, sizeof(head),
                           "HTTP/1.1 %s\r\n"
                           "Content-Type: %s\r\n"
                           "Content-Length: %zu\r\n"
                           "Access-Control-Allow-Origin: *\r\n"
                           "Connection: close\r\n\r\n",
                           status, ctype, len);
    if (n <= 0) return;
    if (write_all(fd, head, (size_t)n) != 0) return;
    if (len > 0) write_all(fd, body, len);
}

static void send_error(int fd, const char *status, const char *type,
                       const char *message) {
    char escaped[512];
    if (mynah_json_escape(message, escaped, sizeof(escaped)) == (size_t)-1) {
        snprintf(escaped, sizeof(escaped), "request failed");
    }
    char body[768];
    const int n = snprintf(body, sizeof(body),
                           "{\"error\":{\"message\":\"%s\",\"type\":\"%s\"}}",
                           escaped, type);
    if (n <= 0) return;
    send_status(fd, status, "application/json", body, (size_t)n);
}

/* The same shape with an `error.code`, for the refusals that share a
 * vocabulary with the prefork admission ladder. A client matching on
 * `error.code` must see the same token whether the router refused it or the
 * worker did, so the token comes from mynah_prefork_refusal_code() rather than
 * being spelled out a second time here. */
static void send_error_code(int fd, const char *status, const char *type,
                            const char *code, const char *message) {
    char escaped[512];
    if (mynah_json_escape(message, escaped, sizeof(escaped)) == (size_t)-1) {
        snprintf(escaped, sizeof(escaped), "request failed");
    }
    char body[832];
    const int n = snprintf(body, sizeof(body),
                           "{\"error\":{\"message\":\"%s\",\"type\":\"%s\","
                           "\"code\":\"%s\"}}",
                           escaped, type, code);
    if (n <= 0) return;
    send_status(fd, status, "application/json", body, (size_t)n);
}

/* ------------------------------------------------------------------ audio */

static void wav_header(unsigned char h[44], uint32_t data_bytes, unsigned rate) {
    const uint32_t riff = 36u + data_bytes;
    const uint16_t channels = 1, bits = 16;
    const uint32_t byte_rate = rate * channels * (bits / 8u);
    memcpy(h, "RIFF", 4);
    h[4] = (unsigned char)(riff);       h[5] = (unsigned char)(riff >> 8);
    h[6] = (unsigned char)(riff >> 16); h[7] = (unsigned char)(riff >> 24);
    memcpy(h + 8, "WAVEfmt ", 8);
    h[16] = 16; h[17] = 0; h[18] = 0; h[19] = 0;
    h[20] = 1;  h[21] = 0;
    h[22] = (unsigned char)channels; h[23] = 0;
    h[24] = (unsigned char)(rate);       h[25] = (unsigned char)(rate >> 8);
    h[26] = (unsigned char)(rate >> 16); h[27] = (unsigned char)(rate >> 24);
    h[28] = (unsigned char)(byte_rate);       h[29] = (unsigned char)(byte_rate >> 8);
    h[30] = (unsigned char)(byte_rate >> 16); h[31] = (unsigned char)(byte_rate >> 24);
    h[32] = (unsigned char)(channels * bits / 8u); h[33] = 0;
    h[34] = (unsigned char)bits; h[35] = 0;
    memcpy(h + 36, "data", 4);
    h[40] = (unsigned char)(data_bytes);       h[41] = (unsigned char)(data_bytes >> 8);
    h[42] = (unsigned char)(data_bytes >> 16); h[43] = (unsigned char)(data_bytes >> 24);
}

static int16_t to_pcm16(float v) {
    if (!(v > -1.0f)) v = -1.0f;
    if (!(v < 1.0f)) v = 1.0f;
    return (int16_t)(v * 32767.0f);
}

/* ------------------------------------------------------------------ routes */

/* Serves both the OpenAI shape ("input"/"voice") and the native one
 * ("text"/"speaker"), so a caller need not pretend to be OpenAI to be clear.
 *
 * Returns 1 when the descriptor is no longer the caller's -- it was answered
 * and closed here, handed to the stream writer, or handed to a job that
 * outlives this call -- and 0 when it is still the caller's to close. The
 * early validation failures are the only cases that return 0: they answer on
 * the caller's descriptor and leave the closing to it, exactly as the other
 * routes do. */
static int handle_speech(int fd, const char *body) {
    char text[MAX_TEXT];
    if (mynah_json_string(body, "input", text, sizeof(text)) != 0 &&
        mynah_json_string(body, "text", text, sizeof(text)) != 0) {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "missing 'input' (or 'text')");
        return 0;
    }
    if (text[0] == '\0') {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "empty 'input'");
        return 0;
    }

    char voice[64] = {0};
    if (mynah_json_string(body, "voice", voice, sizeof(voice)) != 0) {
        (void)mynah_json_string(body, "speaker", voice, sizeof(voice));
    }
    unsigned speaker = 0;
    if (resolve_voice(voice, &speaker) != 0) {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "unknown 'voice' — GET /v1/voices lists the available ones");
        return 0;
    }

    /* LANGUAGE IS A ROUTING KEY, not a tokenizer argument, whenever the pack's
     * weights are bound to one. It used to default to "en" unconditionally,
     * which was harmless while it only chose a Magpie tokenizer and would be a
     * disaster here: every request that omitted it would ask an Italian server
     * for English. Unspecified now means "whatever this server holds".
     *
     * The empty string is therefore load bearing and is not a missing value. */
    char language[32] = {0};
    (void)mynah_json_string(body, "language", language, sizeof(language));
    if (g.language[0] != '\0') {
        /* Bound weights. A mismatch is refused BEFORE a job exists, so a slot
         * is never charged for a request that cannot be served, and so no
         * batch can ever contain two languages -- there is only one model in
         * this process and only requests for it get past this point. */
        if (language[0] != '\0') {
            const char *resident = g.language;
            if (mynah_prefork_language_match(&resident, 1, language) != 0) {
                atomic_fetch_add(&g_stats.language_refused, 1ul);
                char msg[320];
                const char *plan = mynah_prefork_language_plan();
                snprintf(msg, sizeof(msg),
                         "this server holds '%s' and was asked for '%.32s'%s%s"
                         "; GET /health lists what is resident",
                         g.language, language,
                         plan[0] != '\0' ? "; the fleet holds " : "",
                         plan[0] != '\0' ? plan : "");
                send_error_code(fd, "400 Bad Request", "invalid_request_error",
                                mynah_prefork_refusal_code(
                                    MYNAH_PREFORK_REFUSE_LANGUAGE_NOT_SERVED),
                                msg);
                return 0;
            }
        }
        /* Nothing downstream consumes it: a bound pack ships its own
         * tokenizer, so the language is already baked into the weights and the
         * vocabulary alike. */
    } else if (language[0] == '\0') {
        /* Unbound weights (Magpie): the tokenizer still needs a language, and
         * "en" is the default this server has always used. Unchanged on
         * purpose -- this is the path the goldens cover. */
        snprintf(language, sizeof(language), "en");
    }

    char format[32] = "wav";
    (void)mynah_json_string(body, "response_format", format, sizeof(format));
    const int want_pcm = ci_equal(format, "pcm");
    if (!want_pcm && !ci_equal(format, "wav")) {
        /* Be explicit rather than silently returning WAV under another name:
         * mp3/opus/aac/flac would need an encoder this runtime does not embed. */
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "response_format must be 'wav' or 'pcm'");
        return 0;
    }

    int stream = 0;
    (void)mynah_json_bool(body, "stream", &stream);

    double temperature = g.info.default_temperature;
    (void)mynah_json_number(body, "temperature", &temperature);
    double seed = 42.0;
    (void)mynah_json_number(body, "seed", &seed);
    double topk = (double)g.info.default_topk;
    (void)mynah_json_number(body, "top_k", &topk);
    double max_steps = 0.0;
    (void)mynah_json_number(body, "max_steps", &max_steps);

    int *ids = NULL;
    size_t id_count = 0;
    char err[512];
    const int encode_failed = g.sp != NULL
        ? mynah_sp_encode(g.sp, text, strlen(text), &ids, &id_count, err, sizeof(err))
        : mynah_tokenizer_encode(g.tokenizer, language, text, &ids, &id_count,
                                 err, sizeof(err));
    if (encode_failed != 0) {
        send_error(fd, "400 Bad Request", "invalid_request_error", err);
        return 0;
    }

    mynah_tts_request request;
    memset(&request, 0, sizeof(request));
    request.text_ids = ids;
    request.text_length = id_count;
    request.speaker = speaker;
    request.max_steps = max_steps > 0.0 ? (unsigned)max_steps : 0u;
    request.temperature = (float)temperature;
    request.topk = topk > 0.0 ? (unsigned)topk : g.info.default_topk;
    request.use_local_transformer = 1;
    request.seed = (uint64_t)seed;

    /* From here the descriptor belongs to the job. Every exit below answers
     * through job_claim_fd, so no path can close it twice, and job_release
     * closes it if somehow none of them ran.
     *
     * A streaming request is built exactly like a batch one -- same job, same
     * queue, same scheduler -- and differs only in carrying the response
     * header the writer will send and in nobody waiting for it here. */
    synth_job *job = job_new(fd);
    if (job == NULL) {
        free(ids);
        send_error(fd, "503 Service Unavailable", "server_error",
                   "out of memory accepting the request");
        return 0;
    }
    job->request = request;
    job->text_ids = ids;             /* the job owns the ids, and outlives us */
    job->request.text_ids = job->text_ids;
    job->want_pcm = want_pcm;
    job->is_stream = stream;
    /* Rung 4 of the admission ladder, enforced where it is legal to enforce it.
     * The parent watches each worker's oldest outstanding dispatch and counts a
     * breach, but only this process can STOP at a frame boundary, and the frame
     * boundary is the only preemption point the engine has. So the effective
     * cap is the tighter of the operator's per-request timeout and the prefork
     * service cap, and sink_cancelled -- which the driver polls between steps --
     * is what observes it. A cap of 0 from either side means "no cap", so the
     * minimum has to skip zeros rather than take them literally. */
    {
        unsigned cap_ms = g.request_timeout_ms;
        const int service_cap = mynah_prefork_service_cap_ms();
        if (service_cap > 0 &&
            (cap_ms == 0u || (unsigned)service_cap < cap_ms)) {
            cap_ms = (unsigned)service_cap;
        }
        job->sink.deadline_ms = cap_ms > 0u ? now_ms() + (double)cap_ms : 0.0;
    }

    if (stream) {
        /* Chunked PCM: the client gets audio as it is produced. A WAV header
         * needs the total length up front, so streaming is raw PCM only. The
         * header is sent by the writer thread the scheduler starts when this
         * job is admitted, so the client's first byte means "you have a slot"
         * rather than "someone parsed your JSON". */
        const int hn = snprintf(job->header, sizeof(job->header),
                                "HTTP/1.1 200 OK\r\n"
                                "Content-Type: audio/pcm\r\n"
                                "X-Sample-Rate: %u\r\n"
                                "X-Bits-Per-Sample: 16\r\n"
                                "X-Channels: 1\r\n"
                                "Transfer-Encoding: chunked\r\n"
                                "Access-Control-Allow-Origin: *\r\n"
                                "Connection: close\r\n\r\n",
                                g.info.sample_rate);
        if (hn <= 0 || (size_t)hn >= sizeof(job->header)) {
            job_respond_error(job, "500 Internal Server Error", "server_error",
                              "cannot build the response header");
            job_release(job);
            return 1;
        }
    }

    if (job_enqueue(job) != 0) {
        atomic_fetch_add(&g_stats.rejected, 1ul);
        job_respond_error(job, "503 Service Unavailable", "server_error",
                          "server at capacity: the synthesis queue is full");
        job_release(job);
        return 1;
    }

    if (stream) {
        /* Nothing left to wait for: the scheduler starts the writer, feeds it
         * and finishes the response. Going straight back to accept()ing is the
         * point -- a worker blocked here for the length of an utterance is a
         * server whose parallelism is its worker count, not its batch width. */
        atomic_fetch_add(&g_stats.streams_total, 1ul);
        job_release(job);
        return 1;
    }

    if (job_wait(job, g.request_timeout_ms) != 0) {
        /* The deadline passed. The job is marked abandoned: if it is still
         * queued the scheduler drops it without synthesizing, and if it is
         * already in the batch the scheduler's cancel poll retires it within
         * one decoder step. Our reference goes now, which is the point -- this
         * worker thread is free to serve someone else instead of waiting on a
         * request nobody is still listening for. */
        atomic_fetch_add(&g_stats.timed_out, 1ul);
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "request exceeded the server deadline of %u ms",
                 g.request_timeout_ms);
        job_respond_error(job, "504 Gateway Timeout", "server_error", msg);
        job_release(job);
        return 1;
    }

    if (job->result != 0) {
        job_respond_error(job, "500 Internal Server Error", "server_error", job->error);
        job_release(job);
        return 1;
    }

    const size_t count = job->count;
    const size_t pcm_bytes = count * sizeof(int16_t);
    unsigned char *out = (unsigned char *)malloc(44u + pcm_bytes);
    if (out == NULL) {
        job_respond_error(job, "500 Internal Server Error", "server_error", "out of memory");
        job_release(job);
        return 1;
    }
    size_t offset = 0;
    if (!want_pcm) {
        wav_header(out, (uint32_t)pcm_bytes, g.info.sample_rate);
        offset = 44;
    }
    int16_t *pcm = (int16_t *)(out + offset);
    for (size_t i = 0; i < count; ++i) pcm[i] = to_pcm16(job->samples[i]);

    const int reply_fd = job_claim_fd(job);
    if (reply_fd >= 0) {
        send_status(reply_fd, "200 OK", want_pcm ? "audio/pcm" : "audio/wav",
                    (const char *)out, offset + pcm_bytes);
        close(reply_fd);
    }
    free(out);
    job_release(job);                /* frees the samples with the last reference */
    return 1;
}

static void handle_voices(int fd) {
    char body[4096];
    size_t n = (size_t)snprintf(body, sizeof(body), "{\"voices\":[");
    for (size_t i = 0; i < g.voice_count && n < sizeof(body) - 128u; ++i) {
        n += (size_t)snprintf(body + n, sizeof(body) - n,
                              "%s{\"id\":%u,\"name\":\"%s\"}",
                              i == 0 ? "" : ",", g.voices[i].id, g.voices[i].name);
    }
    n += (size_t)snprintf(body + n, sizeof(body) - n, "],\"default\":%u}",
                          g.default_speaker);
    send_status(fd, "200 OK", "application/json", body, n);
}

static void handle_models(int fd) {
    char body[512];
    const int n = snprintf(body, sizeof(body),
                           "{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
                           "\"object\":\"model\",\"owned_by\":\"mynah\"}]}",
                           g.model_id);
    if (n > 0) send_status(fd, "200 OK", "application/json", body, (size_t)n);
}

/* Liveness plus the counters worth paging on: how much work is waiting, how
 * much is running, and how much the server refused or gave up on. A monitor
 * that only ever sees "ok" cannot tell a busy server from a stuck one. */
static void handle_health(int fd) {
    /* Grown with the disconnect counter and the cancel-on-disconnect flag.
     * snprintf would truncate rather than overflow, but a truncated /health is
     * invalid JSON, which a monitor reads as "the server is broken". */
    /* The resident languages, as a list, because "which languages does this
     * server actually hold" is the first question a multi-language deployment
     * asks and the one it must not have to infer from a filename. A bound pack
     * lists its one language; an unbound one lists what its tokenizer can
     * encode, which for Magpie is the twelve the weights genuinely serve.
     * `bound` is the difference between "these are the only ones" and "these
     * all run on the same weights", and it is the field a client should branch
     * on rather than counting entries. */
    char langs[512];
    size_t ln = 0;
    langs[0] = '\0';
    if (g.language[0] != '\0') {
        ln = (size_t)snprintf(langs, sizeof(langs), "\"%s\"", g.language);
    } else if (g.tokenizer != NULL) {
        const char *const *names = mynah_tokenizer_languages(g.tokenizer);
        for (size_t i = 0; names != NULL && names[i] != NULL &&
                           ln < sizeof(langs) - 24u; ++i) {
            ln += (size_t)snprintf(langs + ln, sizeof(langs) - ln, "%s\"%s\"",
                                   i == 0 ? "" : ",", names[i]);
        }
    }
    (void)ln;
    /* The language a request that names none lands on: this process's own.
     * JSON `null` when the pack is unbound, because "" would read as a
     * language whose name is the empty string. */
    char fallback[48];
    if (g.language[0] != '\0') {
        snprintf(fallback, sizeof(fallback), "\"%s\"", g.language);
    } else {
        snprintf(fallback, sizeof(fallback), "null");
    }

    char body[2048];
    const int n = snprintf(body, sizeof(body),
                           "{\"status\":\"ok\",\"model\":\"%s\",\"engine\":\"%s\","
                           "\"sample_rate\":%u,\"voices\":%zu,"
                           /* Residency, then how the fleet was divided. Both
                            * printed rather than assumed: the capacity split
                            * is a number this process CHOSE (W defaults to the
                            * pack count), and config that is only assumed has
                            * been wrong twice in this repository already. */
                           "\"languages\":{\"resident\":[%s],\"bound\":%s,"
                           "\"fleet\":\"%s\",\"default\":%s},"
                           "\"jobs\":{\"queued\":%lu,\"active\":%lu,\"completed\":%lu,"
                           "\"failed\":%lu,\"rejected\":%lu,\"timed_out\":%lu,"
                           "\"disconnected\":%lu,\"language_refused\":%lu},"
                           "\"streams\":{\"active\":%lu,\"total\":%lu},"
                           /* A policy that is on by default has to be
                            * READABLE, or an operator cannot tell a server
                            * that frees slots on hangup from one that does
                            * not -- and the two behave identically until a
                            * client actually goes away. */
                           "\"limits\":{\"max_batch\":%zu,\"queue_capacity\":%zu,"
                           "\"workers\":%d,\"request_timeout_ms\":%u,"
                           "\"cancel_on_disconnect\":%s},"
                           /* Which PROCESS answered. Under prefork the counters
                            * above are that worker's, not the machine's, so a
                            * caller polling /health samples a different worker
                            * each time and needs to be told which one. -1 means
                            * a single-process server, where they are the whole
                            * picture. */
                           "\"process\":{\"pid\":%d,\"prefork_worker\":%d,"
                           "\"synthesis_threads\":%d}}",
                           g.model_id, g.info.engine, g.info.sample_rate,
                           g.voice_count,
                           langs,
                           g.language[0] != '\0' ? "true" : "false",
                           mynah_prefork_language_plan(),
                           fallback,
                           atomic_load(&g_stats.queued),
                           atomic_load(&g_stats.active),
                           atomic_load(&g_stats.completed),
                           atomic_load(&g_stats.failed),
                           atomic_load(&g_stats.rejected),
                           atomic_load(&g_stats.timed_out),
                           atomic_load(&g_stats.disconnected),
                           atomic_load(&g_stats.language_refused),
                           atomic_load(&g_stats.streams_active),
                           atomic_load(&g_stats.streams_total),
                           g.max_batch, g.max_pending, g.worker_count,
                           g.request_timeout_ms,
                           g.cancel_on_disconnect ? "true" : "false",
                           (int)getpid(), mynah_prefork_worker_index(),
                           mynah_prefork_worker_threads() > 0
                               ? mynah_prefork_worker_threads()
                               : mynah_num_threads());
    if (n > 0) send_status(fd, "200 OK", "application/json", body, (size_t)n);
}

/* ------------------------------------------------------------- routing
 *
 * Routing compares a whole path against a table, not a prefix against the
 * request line. The prefix test this replaces matched "POST /v1/audio/speech"
 * inside "POST /v1/audio/speechXYZ", so a typo or a probe was synthesized as
 * if it were the real route. That was merely wrong before; now that a matched
 * route can hand the descriptor to a job or to the stream writer, a route that
 * matches by accident is a descriptor with an owner nobody intended.
 *
 * Method and path are also separate tokens now. "GET /v1/audio/speech" used to
 * fall through to 404 only because the prefix included the method; a table
 * lets it answer 405, which is the honest reply. */
typedef enum {
    ROUTE_SPEECH,
    ROUTE_VOICES,
    ROUTE_MODELS,
    ROUTE_HEALTH,
    ROUTE_NONE,        /* no such path: the caller answers 404 */
    ROUTE_ANSWERED     /* the precheck already replied (405/415/400) */
} route_id;

static const struct {
    const char *path;
    const char *method;
    route_id id;
} ROUTES[] = {
    { "/v1/audio/speech", "POST", ROUTE_SPEECH },
    { "/v1/tts",          "POST", ROUTE_SPEECH },
    { "/v1/voices",       "GET",  ROUTE_VOICES },
    { "/v1/models",       "GET",  ROUTE_MODELS },
    { "/health",          "GET",  ROUTE_HEALTH },
};

/* Resolves method+path to a route, answering the protocol-level refusals
 * itself so that no request reaches a handler with a body the handler would
 * have to re-validate. */
static route_id http_precheck(int fd, const char *method, const char *path,
                              const char *headers, size_t headers_len,
                              const char *body) {
    const char *allow = NULL;
    route_id id = ROUTE_NONE;
    for (size_t i = 0; i < sizeof(ROUTES) / sizeof(ROUTES[0]); ++i) {
        if (strcmp(path, ROUTES[i].path) == 0) {
            allow = ROUTES[i].method;
            id = ROUTES[i].id;
            break;
        }
    }
    if (allow == NULL) return ROUTE_NONE;

    if (strcmp(method, allow) != 0) {
        char msg[160];
        snprintf(msg, sizeof(msg), "method not allowed: %s takes %s", path, allow);
        send_error(fd, "405 Method Not Allowed", "invalid_request_error", msg);
        return ROUTE_ANSWERED;
    }
    if (strcmp(allow, "POST") != 0) return id;

    /* A Content-Type we do not serve is refused here rather than being parsed
     * as if it were JSON: form data and multipart bodies would otherwise be
     * read by the JSON helpers as a body with no fields and answered 400,
     * which tells the caller nothing about what is actually wrong. An absent
     * header stays acceptable -- plenty of clients omit it. */
    char ctype[128];
    if (mynah_http_header(headers, headers_len, "Content-Type",
                          ctype, sizeof(ctype)) == 0 && ctype[0] != '\0') {
        if (!mynah_http_media_type_is(ctype, "application/json")) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "unsupported Content-Type '%.80s': this endpoint takes "
                     "application/json only", ctype);
            send_error(fd, "415 Unsupported Media Type", "invalid_request_error", msg);
            return ROUTE_ANSWERED;
        }
    }

    const char *b = body != NULL ? body : "";
    while (*b == ' ' || *b == '\t' || *b == '\r' || *b == '\n') ++b;
    if (*b != '{') {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   *b != '\0' ? "body is not a JSON object"
                              : "empty body: expected a JSON object");
        return ROUTE_ANSWERED;
    }
    return id;
}

/* ------------------------------------------------------- connection queue */

/* A fixed worker pool draining a bounded queue, rather than one thread per
 * connection.  Thread-per-connection has no ceiling: a client that only opens
 * sockets spawns threads until the process dies -- no exploit needed, just a
 * loop.  When the queue is full we answer 503 and close, because shedding load
 * is a better failure mode than unbounded growth. */
typedef struct {
    int fds[CONN_QUEUE_CAP];
    size_t head, tail, count;
    int shutdown;
    pthread_mutex_t mu;
    pthread_cond_t not_empty;
} conn_queue;

static conn_queue g_queue;

static void queue_init(conn_queue *q) {
    q->head = q->tail = q->count = 0;
    q->shutdown = 0;
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}

/* 0 when parked, -1 when full or shutting down and the caller must close. */
static int queue_push(conn_queue *q, int fd) {
    pthread_mutex_lock(&q->mu);
    if (q->count == CONN_QUEUE_CAP || q->shutdown) {
        pthread_mutex_unlock(&q->mu);
        return -1;
    }
    q->fds[q->tail] = fd;
    q->tail = (q->tail + 1u) % CONN_QUEUE_CAP;
    ++q->count;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
    return 0;
}

/* A client fd, or -1 once shut down and drained. */
static int queue_pop(conn_queue *q) {
    pthread_mutex_lock(&q->mu);
    while (q->count == 0 && !q->shutdown) {
        pthread_cond_wait(&q->not_empty, &q->mu);
    }
    if (q->count == 0) {
        pthread_mutex_unlock(&q->mu);
        return -1;
    }
    const int fd = q->fds[q->head];
    q->head = (q->head + 1u) % CONN_QUEUE_CAP;
    --q->count;
    pthread_mutex_unlock(&q->mu);
    return fd;
}

/* ------------------------------------------------------------- connection */

/* The end of a connection that never became a synthesis job: a validation
 * refusal, a malformed request, one of the non-synthesis routes. Pairs with
 * the notification in job_release(); between the two, every descriptor this
 * worker was handed is reported finished exactly once, which is what the
 * prefork parent's slot accounting is. */
static void conn_close(int fd) {
    if (fd < 0) return;
    close(fd);
    mynah_prefork_conn_done();
}

static void handle_connection(int fd) {
    char *buf = (char *)malloc(MAX_BODY + 8192u);
    if (buf == NULL) { conn_close(fd); return; }

    size_t len = 0;
    const char *head_end = NULL;
    while (len < MAX_BODY + 8192u - 1u) {
        const ssize_t n = recv(fd, buf + len, MAX_BODY + 8192u - 1u - len, 0);
        if (n <= 0) break;
        len += (size_t)n;
        buf[len] = '\0';
        head_end = mynah_memmem(buf, len, "\r\n\r\n", 4);
        if (head_end != NULL) break;
    }
    if (head_end == NULL) { free(buf); conn_close(fd); return; }

    const size_t head_len = (size_t)(head_end - buf);
    char length_header[32] = {0};
    size_t content_length = 0;
    if (mynah_http_header(buf, head_len, "Content-Length",
                          length_header, sizeof(length_header)) == 0) {
        const long v = strtol(length_header, NULL, 10);
        if (v > 0 && (size_t)v <= MAX_BODY) content_length = (size_t)v;
        else if (v > (long)MAX_BODY) {
            send_error(fd, "413 Payload Too Large", "invalid_request_error",
                       "request body too large");
            free(buf); conn_close(fd); return;
        }
    }

    size_t body_have = len - (head_len + 4u);
    while (body_have < content_length && len < MAX_BODY + 8192u - 1u) {
        const ssize_t n = recv(fd, buf + len, MAX_BODY + 8192u - 1u - len, 0);
        if (n <= 0) break;
        len += (size_t)n;
        body_have = len - (head_len + 4u);
    }
    buf[len] = '\0';
    char *body = buf + head_len + 4u;

    /* Set when a route has handed the descriptor to another owner (the stream
     * writer thread). Closing it here as well would be a double close, and the
     * number it names may already belong to a different connection. */
    int handed_off = 0;

    char method[16];
    char path[256];
    if (mynah_http_request_line(buf, len, method, sizeof(method),
                                path, sizeof(path)) != 0) {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "malformed request line");
        free(buf);
        conn_close(fd);
        return;
    }

    if (strcmp(method, "OPTIONS") == 0) {
        const char *pre =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Connection: close\r\n\r\n";
        write_all(fd, pre, strlen(pre));
    } else {
        switch (http_precheck(fd, method, path, buf, head_len, body)) {
            case ROUTE_SPEECH:   handed_off = handle_speech(fd, body); break;
            case ROUTE_VOICES:   handle_voices(fd); break;
            case ROUTE_MODELS:   handle_models(fd); break;
            case ROUTE_HEALTH:   handle_health(fd); break;
            case ROUTE_ANSWERED: break;
            case ROUTE_NONE:
            default:
                send_error(fd, "404 Not Found", "invalid_request_error",
                           "unknown route");
                break;
        }
    }

    free(buf);
    /* The one place a connection is closed when it was not handed to another
     * owner. `handed_off` is set only by a route that transferred ownership
     * (to a job or to the stream writer) or already closed it after replying;
     * closing here as well would land on whatever connection accept() has
     * since given that number. */
    if (!handed_off) conn_close(fd);
}

static void *worker_main(void *arg) {
    /* The HTTP workers are interchangeable, so the index is the only thing
     * that distinguishes them -- and it is exactly what is wanted when three
     * of four are parked in recv() and one is not. */
    {
        char name[16];
        snprintf(name, sizeof(name), "mynah-http%d", (int)(intptr_t)arg);
        mynah_thread_set_name(name);
    }
    for (;;) {
        const int fd = queue_pop(&g_queue);
        if (fd < 0) break;
        handle_connection(fd);
    }
    return NULL;
}

/* -------------------------------------------------------------------- main */

/* SIGINT/SIGTERM ask the accept loop to stop, so the process unwinds through
 * the normal shutdown path: workers joined, scheduler stopped, parked
 * descriptors closed, model freed. Without it the process dies inside accept()
 * and a leak checker run with --atExit has nothing to report on.
 *
 * The handler only sets a flag. Closing the listening socket from a handler
 * looks tempting and does not work: a blocking accept() is not woken by
 * another thread closing its descriptor, and signal() installs a restarting
 * handler, so the call simply resumes. The loop below therefore waits in
 * poll() with a timeout and re-reads the flag, which needs nothing from the
 * handler beyond an async-signal-safe store. */
static volatile sig_atomic_t g_shutdown = 0;

static void on_signal(int sig) {
    (void)sig;
    g_shutdown = 1;
}

/* SIGUSR1 asks for a statistics dump. In a prefork tree the parent gets the
 * signal, prints its routing table and forwards it to every worker, so one
 * `kill -USR1 <parent>` produces the whole machine's view: the parent's line
 * says how work was distributed, each worker's line says what it did with it.
 * The handler only sets a flag; the printing happens in the accept loop, which
 * is where it is allowed to call fprintf. */
static volatile sig_atomic_t g_dump_stats = 0;

static void on_usr1_dump(int sig) {
    (void)sig;
    g_dump_stats = 1;
}

static void dump_local_stats(void) {
    const int idx = mynah_prefork_worker_index();
    char who[48];
    if (idx >= 0) snprintf(who, sizeof(who), "worker %d pid %d", idx, (int)getpid());
    else snprintf(who, sizeof(who), "server pid %d", (int)getpid());
    fprintf(stderr,
            "[%s] queued=%lu active=%lu completed=%lu failed=%lu rejected=%lu "
            "timed_out=%lu disconnected=%lu streams=%lu/%lu · threads=%d max_batch=%zu\n",
            who,
            atomic_load(&g_stats.queued), atomic_load(&g_stats.active),
            atomic_load(&g_stats.completed), atomic_load(&g_stats.failed),
            atomic_load(&g_stats.rejected), atomic_load(&g_stats.timed_out),
            atomic_load(&g_stats.disconnected),
            atomic_load(&g_stats.streams_active), atomic_load(&g_stats.streams_total),
            mynah_prefork_worker_threads() > 0 ? mynah_prefork_worker_threads()
                                               : mynah_num_threads(),
            g.max_batch);
    fflush(stderr);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL_DIR [-m MODEL_DIR ...] [-p PORT] [--host ADDR] [-w WORKERS]\n"
            "       [--device cpu|metal|cuda] [--max-batch N] [--max-pending N]\n"
            "       [--request-timeout-ms MS] [--no-cancel-on-disconnect]\n"
            "       [--prefork W] [--prefork-threads T] [--prefork-plan]\n"
            "\n"
            "  -m MODEL_DIR       repeatable: ONE PACK PER LANGUAGE. The pack's\n"
            "                     model.json says which language its weights are\n"
            "                     bound to; a request naming another is refused 400\n"
            "                     language_not_served rather than served from the\n"
            "                     wrong model. The FIRST pack is the default, used by\n"
            "                     a request that names no language.\n"
            "                     Several packs imply --prefork: one worker holds one\n"
            "                     pack, which is what makes a batch always one\n"
            "                     language. W defaults to the number of packs.\n"
            "                     A Magpie pack declares no language (one set of\n"
            "                     weights serves twelve) and must be served alone.\n"
            "\n"
            "  --no-cancel-on-disconnect\n"
            "                     stop watching streaming sockets for a peer hangup.\n"
            "                     By default the scheduler checks each streaming slot\n"
            "                     once per decoder step and frees it at the next frame\n"
            "                     boundary when the client has gone. This turns off\n"
            "                     that check only: a stream whose socket has actually\n"
            "                     failed still ends, because it has nowhere to write.\n"
            "                     /health reports which is in force.\n"
            "\n"
            "  --prefork W        serve from W pinned worker processes instead of one\n"
            "                     process. The parent loads the pack, forks W workers\n"
            "                     that share it, routes each connection to the least\n"
            "                     loaded one, and never synthesizes. Linux pins each\n"
            "                     worker to its own core slice; elsewhere it runs\n"
            "                     unpinned and says so.\n"
            "  --prefork-threads T  pool width per worker. Default: allowed cpus / W,\n"
            "                     which subscribes the machine exactly once.\n"
            "  --prefork-plan     print this machine\'s topology and the measurement\n"
            "                     procedure for choosing W, then exit. No model needed.\n"
            "\n"
            "  POST /v1/audio/speech   {\"input\":\"...\",\"voice\":\"Sofia\"}\n"
            "  POST /v1/tts            {\"text\":\"...\",\"speaker\":\"Sofia\"}\n"
            "  GET  /v1/voices\n"
            "  GET  /v1/models\n"
            "  GET  /health\n", argv0);
}

/* At most this many language packs in one fleet. Six PocketTTS languages exist
 * today; the bound is generous and explicit so that a typo in a launch script
 * is refused rather than silently growing an array. */
#define MAX_PACKS 16

int main(int argc, char **argv) {
    const char *pack_dir[MAX_PACKS];
    int pack_count = 0;
    int port = 8080;
    const char *host = "127.0.0.1";
    mynah_tts_device device = MYNAH_TTS_DEVICE_CPU;
    g.worker_count = 4;
    g.max_batch = 8;
    g.max_pending = JOB_QUEUE_CAP;
    g.request_timeout_ms = REQUEST_TIMEOUT_MS;
    /* Default ON. Synthesizing for a socket whose peer is gone spends a slot
     * -- the scarcest thing the server has -- on output nobody will ever
     * hear, and the alternative to freeing it is holding it for the length of
     * an utterance. The escape hatch exists because the detection is a
     * judgement about a socket, and an operator who believes it is wrong
     * needs a way to say so without rebuilding. */
    g.cancel_on_disconnect = 1;
    int prefork_workers = 0;
    int prefork_threads = 0;
    int prefork_plan_only = 0;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            /* REPEATABLE, one pack per language. The first is the default: the
             * pack a request that names no language lands on. Order is
             * therefore a decision, not an accident, and it is printed. */
            if (pack_count >= MAX_PACKS) {
                fprintf(stderr, "at most %d model packs\n", MAX_PACKS);
                return 2;
            }
            pack_dir[pack_count++] = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if ((strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--workers") == 0) && i + 1 < argc) {
            g.worker_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-batch") == 0 && i + 1 < argc) {
            g.max_batch = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-pending") == 0 && i + 1 < argc) {
            const int v = atoi(argv[++i]);
            g.max_pending = v > 0 ? (size_t)v : JOB_QUEUE_CAP;
        } else if (strcmp(argv[i], "--request-timeout-ms") == 0 && i + 1 < argc) {
            const long v = strtol(argv[++i], NULL, 10);
            /* 0 disables the deadline; negative is a typo, not an intent. */
            g.request_timeout_ms = v >= 0 ? (unsigned)v : REQUEST_TIMEOUT_MS;
        } else if (strcmp(argv[i], "--prefork") == 0 && i + 1 < argc) {
            prefork_workers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefork-threads") == 0 && i + 1 < argc) {
            prefork_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefork-plan") == 0) {
            prefork_plan_only = 1;
        } else if (strcmp(argv[i], "--no-cancel-on-disconnect") == 0) {
            g.cancel_on_disconnect = 0;
        } else if (strcmp(argv[i], "--cancel-on-disconnect") == 0) {
            g.cancel_on_disconnect = 1;   /* the default, spelled out */
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            const char *d = argv[++i];
            if (strcmp(d, "metal") == 0) device = MYNAH_TTS_DEVICE_METAL;
            else if (strcmp(d, "cuda") == 0) device = MYNAH_TTS_DEVICE_CUDA;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (prefork_plan_only) {
        /* A plan is about the machine, not about a pack: it must work before
         * anyone has downloaded 219 MB of weights, because the first question
         * on a new server is "what shape should this be", not "does it run". */
        mynah_prefork_config plan;
        memset(&plan, 0, sizeof(plan));
        plan.listen_fd = -1;
        plan.workers = prefork_workers;
        plan.threads_per = prefork_threads;
        plan.slots_per = (int)g.max_batch;
        mynah_prefork_print_plan(&plan, stdout);
        return 0;
    }
    if (pack_count == 0 || port <= 0 || port > 65535) { usage(argv[0]); return 2; }
    if (prefork_workers < 0) { usage(argv[0]); return 2; }
    /* MORE THAN ONE PACK IMPLIES PREFORK, because the whole design is that a
     * process holds one pack. If W was not given it becomes the pack count,
     * and the banner says that it did -- a topology this process chose is
     * exactly the kind of thing that must be printed. If W was given and is
     * too small, that is an error and not a silently dropped language. */
    if (pack_count > 1) {
        if (prefork_workers == 0) {
            prefork_workers = pack_count;
            fprintf(stderr, "%d model packs: enabling --prefork %d "
                            "(one worker per language; each worker holds one pack, "
                            "so a batch is always one language)\n",
                    pack_count, prefork_workers);
        } else if (prefork_workers < pack_count) {
            fprintf(stderr, "--prefork %d cannot serve %d languages: one worker holds "
                            "exactly one pack, so a language would have no worker. "
                            "Use --prefork %d or more.\n",
                    prefork_workers, pack_count, pack_count);
            return 2;
        }
    }
    if (g.worker_count < 1) g.worker_count = 1;
    if (g.worker_count > 64) g.worker_count = 64;
    if (g.max_batch < 1u) g.max_batch = 1u;
    if (g.max_batch > mynah_tts_max_batch()) g.max_batch = mynah_tts_max_batch();

    signal(SIGPIPE, SIG_IGN);   /* a client hanging up mid-stream is routine */

    /* The prefork plan is resolved HERE, before the pack is opened, and not at
     * fork time. The shared thread pool resolves its width once, on first use,
     * and caches it for the life of the process; opening the pack can be that
     * first use. A setenv("MYNAH_THREADS") after the fork would then be a
     * silent no-op and every worker would quietly run at the parent's width --
     * the kind of failure that reads as "prefork did not help" rather than as
     * a bug. Doing it before the open costs the parent nothing: it never
     * synthesizes. */
    mynah_prefork_config pf;
    memset(&pf, 0, sizeof(pf));
    pf.listen_fd = -1;
    pf.workers = prefork_workers;
    pf.threads_per = prefork_threads;
    if (prefork_workers > 0) {
        mynah_prefork_reserve_threads(&pf);
        prefork_workers = pf.workers;
    }

    /* ---- open every pack, BEFORE the fork ----
     *
     * Before, so the mapped weights are one physical copy behind the whole
     * tree rather than one per worker, and so that a bad path is a startup
     * failure rather than a worker that dies after the router is already
     * listening. A worker keeps exactly one of these and closes the rest; the
     * router keeps them all and never enters any of them. */
    char err[512];
    mynah_tts_model *packs[MAX_PACKS];
    mynah_tts_model_info pack_info[MAX_PACKS];
    const char *pack_lang[MAX_PACKS];
    for (int i = 0; i < pack_count; ++i) packs[i] = NULL;

    for (int i = 0; i < pack_count; ++i) {
        if (mynah_tts_model_open_device(pack_dir[i], device, &packs[i],
                                        err, sizeof(err)) != 0) {
            fprintf(stderr, "cannot open model %s: %s\n", pack_dir[i], err);
            for (int k = 0; k < i; ++k) mynah_tts_model_close(packs[k]);
            return 1;
        }
        mynah_tts_model_get_info(packs[i], &pack_info[i]);
        pack_lang[i] = pack_info[i].language;
    }

    /* ---- and validate the fleet, before anything is forked ----
     *
     * Two packs claiming one language, or a second pack whose weights are not
     * bound to any, both mean a request could not be routed: the server would
     * have to pick one silently, and picking silently is how a client gets a
     * whole utterance from the wrong model and a 200 to go with it. */
    if (pack_count > 1) {
        for (int i = 0; i < pack_count; ++i) {
            if (pack_lang[i][0] == '\0') {
                fprintf(stderr,
                    "%s declares no language in model.json, so it cannot be one of "
                    "several packs: its weights serve any language, and there would "
                    "be no way to decide which requests belong to it. Serve it alone, "
                    "with a single -m.\n", pack_dir[i]);
                for (int k = 0; k < pack_count; ++k) mynah_tts_model_close(packs[k]);
                return 1;
            }
            for (int k = 0; k < i; ++k) {
                if (mynah_prefork_language_match(&pack_lang[k], 1, pack_lang[i]) == 0) {
                    fprintf(stderr,
                        "%s and %s both claim language '%s'; a request naming it could "
                        "go to either, so refusing to start\n",
                        pack_dir[k], pack_dir[i], pack_lang[i]);
                    for (int n = 0; n < pack_count; ++n) mynah_tts_model_close(packs[n]);
                    return 1;
                }
            }
        }
    }

    /* The engine's own ceiling, not the runtime's: a continuous-latent engine
     * declares 1 until its batching is measured, and handing it more is an
     * error rather than a slower path.
     *
     * This has to happen HERE and not with the other argument clamping: it
     * asks the model, and before the open above there is no model to ask.
     * Asking too early did not fail, it answered 1 -- the conservative default
     * for "no model" -- so the server silently ran every request alone with a
     * --max-batch the operator had set and /health cheerfully reported.
     *
     * The MINIMUM over the fleet, because every worker runs the same
     * --max-batch and the router's slot accounting is one number. A fleet
     * whose packs disagreed would otherwise hand one engine more slots than it
     * declared. */
    for (int i = 0; i < pack_count; ++i) {
        const size_t engine_max = mynah_tts_model_max_batch(packs[i]);
        if (g.max_batch > engine_max) g.max_batch = engine_max;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGUSR1, on_usr1_dump);

    const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = strcmp(host, "0.0.0.0") == 0 ? INADDR_ANY : inet_addr(host);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        return 1;
    }
    if (listen(listen_fd, 64) != 0) { perror("listen"); return 1; }
    /* Non-blocking so that a spurious poll() readiness cannot park the accept
     * loop, and so the loop always comes back to check the shutdown flag. */
    const int listen_flags = fcntl(listen_fd, F_GETFL, 0);
    if (listen_flags >= 0) fcntl(listen_fd, F_SETFL, listen_flags | O_NONBLOCK);

    /* ------------------------------------------------------------- prefork
     *
     * This is the last point at which this process is still single-threaded,
     * and that is exactly why the fork is here. Two constraints pin it:
     *
     *   - AFTER the pack is open, so the mapped weights are one physical copy
     *     behind every worker rather than W copies of 219 MB;
     *   - BEFORE the scheduler thread, the HTTP workers and the shared pool's
     *     own threads exist, and before a single synthesis has run, because
     *     fork() copies only the calling thread: a child of a multi-threaded
     *     process inherits every mutex in whatever state it was in, including
     *     locked by a thread that no longer exists.
     *
     * A worker comes back from here with a channel instead of a listening
     * socket and is, from the next line on, an ordinary single-process server.
     * The parent does not come back at all: it routes until it is asked to
     * stop and then returns PARENT_DONE with nothing left to unwind but the
     * pack it loaded for its children. */
    int chan_fd = -1;
    int accept_fd = listen_fd;
    if (prefork_workers > 0) {
        pf.listen_fd = listen_fd;
        pf.slots_per = (int)g.max_batch;
        /* The authoritative answer to "is a GPU backend resident in this
         * process?", which prefork refuses to fork across. Its own scan of the
         * CUDA runtime is a backstop for a library someone else pulled in;
         * this flag is the one that also catches Metal, where a mapped
         * framework is not a device and only the caller knows a device was
         * actually opened. Forking with GPU state alive produces a wrong
         * answer rather than a crash, which is the worse failure. */
        pf.gpu_backend_open = (device != MYNAH_TTS_DEVICE_CPU);
        /* The fleet's languages, in pack order. With one pack this is left
         * NULL and every language-aware branch in the router stays dormant --
         * which is what keeps the single-language server the one that shipped. */
        if (pack_count > 1) {
            pf.languages = pack_lang;
            pf.language_count = pack_count;
        }
        const mynah_prefork_role role = mynah_prefork_run(&pf, &g_shutdown, &chan_fd);
        if (role == MYNAH_PREFORK_ERROR) {
            close(listen_fd);
            for (int i = 0; i < pack_count; ++i) mynah_tts_model_close(packs[i]);
            return 1;
        }
        if (role == MYNAH_PREFORK_PARENT_DONE) {
            /* The router closed the listening socket itself, has no threads to
             * join and never entered any model. */
            for (int i = 0; i < pack_count; ++i) mynah_tts_model_close(packs[i]);
            return 0;
        }
        accept_fd = -1;              /* a worker never accepts; the parent routes */
        /* WHICH PACK IS MINE. Decided by the parent before the fork and read
         * back here, so a worker can never disagree with the router about what
         * it holds -- a disagreement would mean requests routed to a worker
         * that refuses them, or worse, one that served them from the wrong
         * weights. */
        if (pack_count > 1) g.pack_index = mynah_prefork_worker_language();
    }

    /* ---- from here this process holds ONE pack ----
     *
     * Everything else is closed. Not for memory -- the mappings are shared and
     * mostly clean -- but because a second model in a process that serves one
     * language is a thing a future edit could reach for, and the shortest way
     * to keep "a batch is one language" true is to leave nothing else here to
     * batch with. */
    if (g.pack_index < 0 || g.pack_index >= pack_count) g.pack_index = 0;
    g.model = packs[g.pack_index];
    g.info = pack_info[g.pack_index];
    g.language = pack_info[g.pack_index].language;
    g.fleet_packs = pack_count;
    for (int i = 0; i < pack_count; ++i) {
        if (i != g.pack_index) mynah_tts_model_close(packs[i]);
    }
    const char *model_dir = pack_dir[g.pack_index];

    /* Which tokenizer applies is a property of the engine, so the pack decides.
     * Opening Magpie's unconditionally rejected a valid PocketTTS pack for a
     * missing english_phoneme.tsv it has no reason to carry.
     *
     * Opened AFTER the fork, and only for this worker's own pack: a tokenizer
     * is per pack, and the parent would otherwise build N of them it never
     * uses. The process is still single threaded here, so this is safe ground
     * -- the fork ordering constraint is about threads and mutexes, not about
     * opening files. */
    if (strcmp(g.info.engine, "pocket") == 0) {
        char sp_path[1024];
        snprintf(sp_path, sizeof(sp_path), "%s/tokenizer.model", model_dir);
        if (mynah_sp_open(sp_path, &g.sp, err, sizeof(err)) != 0) {
            fprintf(stderr, "cannot open tokenizer: %s\n", err);
            mynah_tts_model_close(g.model);
            return 1;
        }
    } else {
        g.tokenizer = mynah_tokenizer_open(model_dir, err, sizeof(err));
        if (g.tokenizer == NULL) {
            fprintf(stderr, "cannot open tokenizer: %s\n", err);
            mynah_tts_model_close(g.model);
            return 1;
        }
    }
    load_voices(model_dir);
    snprintf(g.model_id, sizeof(g.model_id), "%s-%s", g.info.engine, g.info.revision);
    g.default_speaker = g.voice_count > 0 ? g.voices[0].id : 0u;

    /* The main thread's role is decided by the branch just above, so it names
     * itself here rather than at the top of main(): a prefork worker does not
     * accept anything, it receives descriptors down a channel, and calling
     * both of those "accept" would put the wrong answer in `top`.
     *
     * One consequence to know about rather than discover: on Linux the main
     * thread's name IS /proc/<pid>/comm, so `ps -o comm` and a bare `top` will
     * show this instead of the binary name, and `pkill mynah-tts-server` stops
     * matching. That is a deliberate trade -- under prefork it is the only
     * thing that distinguishes a worker from the router in a process list,
     * since every one of them is the same executable. Match on the full
     * command line (pkill -f) if you need the old behaviour. On macOS the
     * process keeps its name and `sample` reports the main thread by its
     * dispatch-queue label, so the name is visible only to a debugger. */
    mynah_thread_set_name(chan_fd >= 0 ? "mynah-recv" : "mynah-accept");

    pthread_mutex_init(&g_batch.mu, NULL);
    pthread_cond_init(&g_batch.arrived, NULL);
    if (pthread_create(&g_batch.thread, NULL, scheduler_main, NULL) != 0) {
        fprintf(stderr, "cannot start the synthesis scheduler\n");
        if (accept_fd >= 0) close(accept_fd);
        if (chan_fd >= 0) close(chan_fd);
        mynah_tokenizer_close(g.tokenizer);
        mynah_sp_close(g.sp);
        mynah_tts_model_close(g.model);
        return 1;
    }
    g_batch.thread_started = 1;

    /* One line that says what this process serves and how the fleet was
     * divided. Assembled rather than branched into three fprintf calls so the
     * banner has one shape whatever the topology, and so the multi-language
     * case cannot be the one nobody printed. */
    char lang_banner[640];
    if (g.language[0] != '\0') {
        const char *plan = mynah_prefork_language_plan();
        snprintf(lang_banner, sizeof(lang_banner),
                 "%s — this pack's weights are BOUND to it, so a request naming "
                 "another is refused 400 %s%s%s%s",
                 g.language,
                 mynah_prefork_refusal_code(MYNAH_PREFORK_REFUSE_LANGUAGE_NOT_SERVED),
                 plan[0] != '\0' ? "\n      fleet: " : "",
                 plan[0] != '\0' ? plan : "",
                 plan[0] != '\0'
                     ? " (workers per language; a batch is one process's slots,"
                       "\n      so a batch is always one language)"
                     : "");
    } else {
        snprintf(lang_banner, sizeof(lang_banner),
                 "not bound — one set of weights serves every language this pack's "
                 "tokenizer\n      knows; `language` selects a tokenizer and defaults "
                 "to en");
    }

    fprintf(stderr,
            "mynah-tts server on http://%s:%d  model=%s  device=%s  voices=%zu  "
            "workers=%d  max_batch=%zu  queue=%zu  timeout=%ums%s\n"
            "note: one scheduler thread synthesizes; requests join the running\n"
            "      batch as slots free up, streaming and batch alike\n"
            "transport: TCP_NODELAY on every accepted socket; threads named\n"
            "      (mynah-accept/recv, mynah-http*, mynah-sched, mynah-out*)\n"
            "language: %s\n"
            "cancel-on-disconnect: %s\n",
            host, port, g.model_id, mynah_tts_device_name(device), g.voice_count,
            g.worker_count, g.max_batch, g.max_pending, g.request_timeout_ms,
            chan_fd >= 0 ? "  [prefork worker]" : "",
            /* Residency, printed whichever shape the server is in. "which
             * languages does this thing hold" has to be answerable from the
             * log of a server that has since died, and for a bound pack it is
             * also the answer to "why was my request refused 400". */
            lang_banner,
            /* Printed whichever way it is set: a policy visible only when
             * enabled tells an operator nothing when a stream outlives its
             * client and they are trying to work out why. */
            g.cancel_on_disconnect
                ? "on — each streaming slot is checked for a peer hangup once per\n"
                  "      decoder step and freed at the next frame boundary\n"
                  "      (--no-cancel-on-disconnect)"
                : "OFF — streaming sockets are not watched for a hangup; a stream\n"
                  "      ends only when a write to it actually fails");

    queue_init(&g_queue);
    pthread_t workers[64];
    int worker_count = g.worker_count;
    for (int i = 0; i < worker_count; ++i) {
        /* The index travels as the argument rather than through a shared
         * counter: the thread names itself, so it must know which one it is
         * before anything else races it. */
        if (pthread_create(&workers[i], NULL, worker_main, (void *)(intptr_t)i) != 0) {
            worker_count = i;
            break;
        }
    }
    if (worker_count == 0) { fprintf(stderr, "cannot start workers\n"); return 1; }

    /* Where a connection comes from is the ONLY difference between a
     * single-process server and a prefork worker. A worker has no listening
     * socket: the parent accepted, chose it, and passed the descriptor down a
     * socketpair. Everything after the handover -- the bounded queue, the HTTP
     * workers, the scheduler, the writer -- is the same code in both shapes,
     * which is the point: there is no second server to keep in step. */
    while (!g_shutdown) {
        if (g_dump_stats) { g_dump_stats = 0; dump_local_stats(); }

        int fd;
        if (chan_fd >= 0) {
            fd = mynah_prefork_recv_conn(chan_fd, 200);
            if (fd == -2) continue;        /* timeout or signal: re-read the flag */
            if (fd < 0) {
                /* The parent is gone. Nothing will ever arrive again, so stop
                 * rather than spin on a dead channel; the requests already in
                 * flight are finished by the shutdown path below. */
                fprintf(stderr, "prefork worker %d: the router closed the channel; "
                                "shutting down\n", mynah_prefork_worker_index());
                break;
            }
        } else {
            struct pollfd pfd;
            pfd.fd = accept_fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            const int ready = poll(&pfd, 1, 200);
            if (ready < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (ready == 0) continue;   /* nothing yet: re-read the shutdown flag */

            fd = accept(accept_fd, NULL, NULL);
            if (fd < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
                    errno == ECONNABORTED) continue;
                break;
            }
        }
        /* BSD hands the accepted socket the listener's O_NONBLOCK; Linux does
         * not. Clear it either way, because everything downstream reads and
         * writes this descriptor blocking, with a timeout. A descriptor that
         * arrived over SCM_RIGHTS keeps whatever the parent's listener gave it,
         * so it needs exactly the same treatment -- which is why this is one
         * block below the branch rather than two copies inside it. */
        const int cf = fcntl(fd, F_GETFL, 0);
        if (cf >= 0) fcntl(fd, F_SETFL, cf & ~O_NONBLOCK);
        /* A client that connects and never sends must not pin a worker
         * forever; without this the bounded queue would still fill up. */
        struct timeval tv = {CLIENT_TIMEOUT_S, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        /* Nagle off, on every accepted socket and on every descriptor the
         * prefork parent passes down -- which is why this sits below the
         * branch with the timeouts rather than inside either arm.
         *
         * The latency argument is the obvious one: a streamed chunk is a small
         * write followed by nothing, exactly the shape Nagle holds back
         * waiting for a companion. The measurement argument is the one that
         * actually forced it. Our cadence metrics are a histogram of when
         * chunks ARRIVED, and Nagle merges two server emissions into one
         * client-visible arrival. A run whose coalesced-read share is high is
         * declared not quotable for cadence percentages
         * (.work/streaming-cadence.md §5), so without this the transport is
         * free to invent numbers the scheduler never produced. */
        const int nodelay = 1;
        if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay)) != 0) {
            /* Not fatal -- the stream is still correct, only burstier -- but
             * it must not pass unnoticed, because the cadence numbers taken
             * afterwards would silently be transport artefacts. */
            static int warned = 0;
            if (!warned) {
                warned = 1;
                fprintf(stderr, "warning: TCP_NODELAY refused (%s); cadence "
                                "measurements from this run are not quotable\n",
                        strerror(errno));
            }
        }
        if (queue_push(&g_queue, fd) != 0) {
            atomic_fetch_add(&g_stats.rejected, 1ul);
            /* Queue full: shed the connection rather than grow without bound.
             *
             * This used to write the 503 and close immediately, which is how a
             * refusal reaches the client as an RST instead of as a status: the
             * request body is still in flight, so close() on unread data sends
             * a reset and curl reports a broken pipe rather than the 503 we
             * carefully wrote. mynah_prefork_refuse_and_close() does the
             * shutdown-drain-close dance without blocking this thread, and
             * emits the same machine-readable error.code as every other rung
             * of the ladder, so a client matching on the code sees one
             * vocabulary whichever rung refused it.
             *
             * It takes ownership of the descriptor, so the slot must be given
             * back separately -- a shed connection is still one the router
             * charged to this worker. */
            mynah_prefork_refuse_and_close(fd, MYNAH_PREFORK_REFUSE_AT_CAPACITY);
            mynah_prefork_conn_done();
        }
    }

    /* Shutdown, in the one order that leaves nothing dangling: stop taking
     * connections, let the workers finish the requests they are already
     * waiting on (they hold job references), then stop the scheduler -- which
     * finishes what is in the batch and stops admitting -- then answer
     * whatever never got served. */
    pthread_mutex_lock(&g_queue.mu);
    g_queue.shutdown = 1;
    pthread_cond_broadcast(&g_queue.not_empty);
    pthread_mutex_unlock(&g_queue.mu);
    for (int i = 0; i < worker_count; ++i) pthread_join(workers[i], NULL);

    pthread_mutex_lock(&g_batch.mu);
    g_batch.stop = 1;
    pthread_mutex_unlock(&g_batch.mu);
    atomic_store(&g_batch.stopping, 1);
    pthread_cond_broadcast(&g_batch.arrived);
    if (g_batch.thread_started) pthread_join(g_batch.thread, NULL);
    /* Streaming jobs have no submitter waiting, so anything still queued is
     * this thread's to answer. */
    drain_pending_jobs();

    /* Connections parked but never picked up: their descriptors are still
     * ours, and this is the last chance to close them. */
    for (int parked = queue_pop(&g_queue); parked >= 0; parked = queue_pop(&g_queue)) {
        conn_close(parked);
    }

    if (accept_fd >= 0) close(accept_fd);
    if (chan_fd >= 0) close(chan_fd);
    mynah_tokenizer_close(g.tokenizer);
    mynah_sp_close(g.sp);
    mynah_tts_model_close(g.model);
    pthread_cond_destroy(&g_batch.arrived);
    pthread_mutex_destroy(&g_batch.mu);
    pthread_cond_destroy(&g_queue.not_empty);
    pthread_mutex_destroy(&g_queue.mu);
    return 0;
}
