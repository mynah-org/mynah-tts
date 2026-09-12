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
 * below instead hands whatever is queued to mynah_tts_synthesize_batch, which
 * reads those weights once and serves every waiting request from cache --
 * measured 1.63x aggregate throughput at eight in flight, with each request
 * receiving byte-identical audio to what it would have received alone.
 *
 * Streaming still runs one request at a time (it needs its callback interleaved
 * with generation) and takes the same lock, so it never overlaps a batch.
 *
 * Socket I/O for a stream does NOT run on the synthesis thread: the callback
 * only copies PCM into a bounded queue and a dedicated writer thread owns the
 * fd from there on (server/stream_out.c). A client that stops reading loses its
 * own stream and nothing else -- before this, it held the synthesis lock for
 * the whole SO_SNDTIMEO.
 *
 * Three things follow from that split, and they are the shape of this file:
 *
 *   - HTTP parsing and synthesis are separate roles. A worker reads the
 *     request, validates it, tokenizes it and builds a job; the scheduler does
 *     nothing but synthesize.
 *   - A job is a heap object with a refcount, and it owns the client
 *     descriptor. Two threads hold it and, with a deadline, either may be the
 *     last to let go. The descriptor leaves the job exactly once, through an
 *     atomic claim, which is what makes a double close impossible and a leaked
 *     descriptor detectable.
 *   - Every request has a wall-clock deadline (--request-timeout-ms). A
 *     request past it is answered 504 and its worker is freed; a queued job
 *     past it is never synthesized at all.
 */
#include "http_util.h"
#include "stream_out.h"

#include "mynah_tts.h"
#include "tokenizer.h"

#include <errno.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
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
    mynah_tokenizer *tokenizer;
    mynah_tts_model_info info;
    char model_id[128];
    voice_entry voices[64];
    size_t voice_count;
    unsigned default_speaker;
    int worker_count;
    size_t max_batch;
    size_t max_pending;
    unsigned request_timeout_ms;
    pthread_mutex_t synth_lock;
} g;

/* Counters published by /health. Plain relaxed atomics: they are diagnostics,
 * not synchronization, and a reader wants a cheap snapshot rather than a
 * consistent one. */
static struct {
    atomic_ulong queued;        /* gauge: jobs waiting for the scheduler */
    atomic_ulong active;        /* gauge: jobs inside a running batch */
    atomic_ulong completed;
    atomic_ulong rejected;      /* queue full, at either queue */
    atomic_ulong timed_out;     /* deadline passed before the audio did */
    atomic_ulong failed;
    atomic_ulong streams_active;
    atomic_ulong streams_total;
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
 * Jobs are drained by one scheduler thread, so requests that arrive while a
 * batch is running are grouped into the next one -- the queue depth does the
 * batching, and BATCH_WINDOW_US only helps requests that land near
 * simultaneously on an idle server. */
#define BATCH_WINDOW_US 2000

static void send_error(int fd, const char *status, const char *type,
                       const char *message);

typedef struct synth_job {
    /* Filled in before the job is published; read-only from then on. */
    mynah_tts_request request;
    int *text_ids;                 /* owned: the request points into this */
    int want_pcm;

    int fd;                        /* owned; leaves only through job_claim_fd */
    atomic_int fd_claimed;

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
    pthread_t thread;
    int thread_started;
} g_batch;

static synth_job *job_new(int fd) {
    synth_job *j = (synth_job *)calloc(1, sizeof(*j));
    if (j == NULL) return NULL;
    j->fd = fd;
    atomic_init(&j->fd_claimed, 0);
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
     * two can never both believe they own the outcome. */
    if (expired) j->abandoned = 1;
    pthread_mutex_unlock(&j->mu);
    return expired ? -1 : 0;
}

static void batch_run(synth_job **taken, size_t count) {
    mynah_tts_batch_job jobs[16];
    if (count > (sizeof(jobs) / sizeof(jobs[0]))) count = sizeof(jobs) / sizeof(jobs[0]);
    for (size_t i = 0; i < count; ++i) {
        jobs[i].request = &taken[i]->request;
        jobs[i].samples = &taken[i]->samples;
        jobs[i].sample_count = &taken[i]->count;
        jobs[i].error = taken[i]->error;
        jobs[i].error_capacity = sizeof(taken[i]->error);
        jobs[i].result = 0;
    }
    /* Streaming holds the same lock, so a batch never overlaps one. */
    pthread_mutex_lock(&g.synth_lock);
    mynah_tts_synthesize_batch(g.model, jobs, count);
    pthread_mutex_unlock(&g.synth_lock);
    for (size_t i = 0; i < count; ++i) taken[i]->result = jobs[i].result;
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

static void *batch_scheduler(void *arg) {
    (void)arg;
    synth_job *taken[16];
    for (;;) {
        pthread_mutex_lock(&g_batch.mu);
        while (g_batch.pending == 0u && !g_batch.stop) {
            pthread_cond_wait(&g_batch.arrived, &g_batch.mu);
        }
        const int stopping = g_batch.stop;
        if (stopping && g_batch.pending == 0u) {
            pthread_mutex_unlock(&g_batch.mu);
            break;
        }
        /* Give near-simultaneous arrivals a moment to join this batch. */
        if (!stopping && g_batch.pending < g.max_batch) {
            pthread_mutex_unlock(&g_batch.mu);
            usleep(BATCH_WINDOW_US);
            pthread_mutex_lock(&g_batch.mu);
        }
        size_t count = 0;
        const size_t take = stopping ? (sizeof(taken) / sizeof(taken[0])) : g.max_batch;
        while (count < take && g_batch.head != NULL) {
            synth_job *j = g_batch.head;
            g_batch.head = j->next;
            if (g_batch.head == NULL) g_batch.tail = NULL;
            j->next = NULL;
            taken[count++] = j;
            --g_batch.pending;
        }
        atomic_store(&g_stats.queued, (unsigned long)g_batch.pending);
        pthread_mutex_unlock(&g_batch.mu);

        if (stopping) {
            for (size_t i = 0; i < count; ++i) {
                job_finish(taken[i], -1, "server is shutting down");
            }
            continue;
        }

        /* A job whose deadline passed while it queued never enters the batch.
         * This is the part of the deadline that actually frees capacity: the
         * slot it would have occupied goes to a request someone still wants. */
        size_t live = 0;
        for (size_t i = 0; i < count; ++i) {
            synth_job *j = taken[i];
            pthread_mutex_lock(&j->mu);
            const int gone = j->abandoned;
            pthread_mutex_unlock(&j->mu);
            if (gone) { job_release(j); continue; }
            taken[live++] = j;
        }
        if (live == 0u) continue;

        atomic_fetch_add(&g_stats.active, (unsigned long)live);
        batch_run(taken, live);
        atomic_fetch_sub(&g_stats.active, (unsigned long)live);

        for (size_t i = 0; i < live; ++i) {
            if (taken[i]->result == 0) atomic_fetch_add(&g_stats.completed, 1ul);
            else atomic_fetch_add(&g_stats.failed, 1ul);
            job_finish(taken[i], 0, NULL);
        }
    }
    return NULL;
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

/* --------------------------------------------------------------- streaming */

/* A stream's sink: the writer it feeds, and the wall clock it must respect. */
typedef struct {
    stream_out *out;
    double deadline_ms;      /* monotonic, 0 when no deadline was configured */
    int expired;
} stream_sink;

/* The whole streaming sink: hand the samples to the writer thread and return.
 * The caller already emits stable causal prefixes, so one call is exactly what
 * has become final; the writer decides how that maps onto HTTP chunks.
 *
 * Returning -1 aborts synthesis, which is what a full queue, a dead socket or
 * an expired deadline must do -- backpressure here is cancellation, never a
 * blocking write. The deadline check lives here because this is the only place
 * the streaming path yields between decoder steps: the synthesis runs on this
 * thread, so aborting from here is what releases the synthesis lock. */
static int stream_callback(const float *samples, size_t count, void *user_data) {
    stream_sink *sink = (stream_sink *)user_data;
    if (sink->deadline_ms > 0.0 && now_ms() > sink->deadline_ms) {
        sink->expired = 1;
        return -1;
    }
    if (count == 0) return stream_out_failed(sink->out) ? -1 : 0;
    return stream_out_enqueue(sink->out, samples, count);
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

    char language[16] = "en";
    (void)mynah_json_string(body, "language", language, sizeof(language));

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
    if (mynah_tokenizer_encode(g.tokenizer, language, text, &ids, &id_count,
                               err, sizeof(err)) != 0) {
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

    if (stream) {
        /* Chunked PCM: the client gets audio as it is produced. A WAV header
         * needs the total length up front, so streaming is raw PCM only. The
         * header itself is sent by the writer thread, not from here. */
        char head[512];
        const int hn = snprintf(head, sizeof(head),
                                "HTTP/1.1 200 OK\r\n"
                                "Content-Type: audio/pcm\r\n"
                                "X-Sample-Rate: %u\r\n"
                                "X-Bits-Per-Sample: 16\r\n"
                                "X-Channels: 1\r\n"
                                "Transfer-Encoding: chunked\r\n"
                                "Access-Control-Allow-Origin: *\r\n"
                                "Connection: close\r\n\r\n",
                                g.info.sample_rate);
        if (hn <= 0) {
            free(ids);
            send_error(fd, "500 Internal Server Error", "server_error",
                       "cannot build the response header");
            return 0;
        }

        /* From here the fd belongs to the writer thread: it sends the header,
         * the chunks and the terminator, then closes. Nothing below may write
         * to it or close it -- that double close is the whole hazard of
         * handing a descriptor to a detached thread. */
        stream_out *out = stream_out_start(fd, head);
        if (out == NULL) {
            free(ids);
            send_error(fd, "500 Internal Server Error", "server_error",
                       "cannot start the stream writer");
            return 0;
        }

        mynah_tts_stream *st = NULL;
        /* The stream takes its text through push(), not through the request:
         * leaving text_ids set here would feed the same tokens twice and
         * produce a longer utterance than the batch path for the same input. */
        mynah_tts_request stream_request = request;
        stream_request.text_ids = NULL;
        stream_request.text_length = 0;
        stream_sink sink;
        sink.out = out;
        sink.deadline_ms = g.request_timeout_ms > 0u
                         ? now_ms() + (double)g.request_timeout_ms : 0.0;
        sink.expired = 0;
        atomic_fetch_add(&g_stats.streams_active, 1ul);
        atomic_fetch_add(&g_stats.streams_total, 1ul);
        pthread_mutex_lock(&g.synth_lock);
        int rc = mynah_tts_stream_open(g.model, &stream_request, STREAM_CHUNK,
                                       stream_callback, &sink, &st,
                                       err, sizeof(err));
        if (rc == 0) {
            rc = mynah_tts_stream_push(st, ids, id_count, err, sizeof(err));
            if (rc == 0) rc = mynah_tts_stream_flush(st, err, sizeof(err));
            mynah_tts_stream_close(st);
        }
        pthread_mutex_unlock(&g.synth_lock);
        atomic_fetch_sub(&g_stats.streams_active, 1ul);
        free(ids);

        /* Hand the tail of the response to the writer and let go. A mid-stream
         * failure cannot become an HTTP status because the header is long
         * gone, so the writer terminates the body or truncates it. */
        stream_out_finish(out);
        if (sink.expired) {
            atomic_fetch_add(&g_stats.timed_out, 1ul);
            stream_out_stats stats;
            stream_out_get_stats(out, &stats);
            fprintf(stderr, "stream hit the %u ms deadline after %zu bytes; "
                            "slot released\n",
                    g.request_timeout_ms, stats.sent_bytes);
        } else if (rc != 0 && !stream_out_failed(out)) {
            /* A synthesis failure, not a client one: the writer logs its own. */
            atomic_fetch_add(&g_stats.failed, 1ul);
            stream_out_stats stats;
            stream_out_get_stats(out, &stats);
            fprintf(stderr, "stream failed after %lu queued chunks: %s\n",
                    stats.enqueued_chunks, err);
        } else if (rc == 0) {
            atomic_fetch_add(&g_stats.completed, 1ul);
        }
        /* The descriptor is the writer's now: it sends the terminator and
         * closes. Nothing here may touch it again. */
        stream_out_release(out);
        return 1;
    }

    /* From here the descriptor belongs to the job. Every exit below answers
     * through job_claim_fd, so no path can close it twice, and job_release
     * closes it if somehow none of them ran. */
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

    if (job_enqueue(job) != 0) {
        atomic_fetch_add(&g_stats.rejected, 1ul);
        job_respond_error(job, "503 Service Unavailable", "server_error",
                          "server at capacity: the synthesis queue is full");
        job_release(job);
        return 1;
    }

    if (job_wait(job, g.request_timeout_ms) != 0) {
        /* The deadline passed. The job is marked abandoned: if it is still
         * queued the scheduler drops it without synthesizing, and if it is
         * mid-batch the scheduler discards the samples when it finishes. Our
         * reference goes now, which is the point -- this worker thread is free
         * to serve someone else instead of waiting on a request nobody is
         * still listening for. */
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
    char body[768];
    const int n = snprintf(body, sizeof(body),
                           "{\"status\":\"ok\",\"model\":\"%s\",\"engine\":\"%s\","
                           "\"sample_rate\":%u,\"voices\":%zu,"
                           "\"jobs\":{\"queued\":%lu,\"active\":%lu,\"completed\":%lu,"
                           "\"failed\":%lu,\"rejected\":%lu,\"timed_out\":%lu},"
                           "\"streams\":{\"active\":%lu,\"total\":%lu},"
                           "\"limits\":{\"max_batch\":%zu,\"queue_capacity\":%zu,"
                           "\"workers\":%d,\"request_timeout_ms\":%u}}",
                           g.model_id, g.info.engine, g.info.sample_rate,
                           g.voice_count,
                           atomic_load(&g_stats.queued),
                           atomic_load(&g_stats.active),
                           atomic_load(&g_stats.completed),
                           atomic_load(&g_stats.failed),
                           atomic_load(&g_stats.rejected),
                           atomic_load(&g_stats.timed_out),
                           atomic_load(&g_stats.streams_active),
                           atomic_load(&g_stats.streams_total),
                           g.max_batch, g.max_pending, g.worker_count,
                           g.request_timeout_ms);
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

static void handle_connection(int fd) {
    char *buf = (char *)malloc(MAX_BODY + 8192u);
    if (buf == NULL) { close(fd); return; }

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
    if (head_end == NULL) { free(buf); close(fd); return; }

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
            free(buf); close(fd); return;
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
        close(fd);
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
    if (!handed_off) close(fd);
}

static void *worker_main(void *arg) {
    (void)arg;
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

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL_DIR [-p PORT] [--host ADDR] [-w WORKERS]\n"
            "       [--device cpu|metal|cuda] [--max-batch N] [--max-pending N]\n"
            "       [--request-timeout-ms MS]\n"
            "\n"
            "  POST /v1/audio/speech   {\"input\":\"...\",\"voice\":\"Sofia\"}\n"
            "  POST /v1/tts            {\"text\":\"...\",\"speaker\":\"Sofia\"}\n"
            "  GET  /v1/voices\n"
            "  GET  /v1/models\n"
            "  GET  /health\n", argv0);
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    int port = 8080;
    const char *host = "127.0.0.1";
    mynah_tts_device device = MYNAH_TTS_DEVICE_CPU;
    g.worker_count = 4;
    g.max_batch = 8;
    g.max_pending = JOB_QUEUE_CAP;
    g.request_timeout_ms = REQUEST_TIMEOUT_MS;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_dir = argv[++i];
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
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            const char *d = argv[++i];
            if (strcmp(d, "metal") == 0) device = MYNAH_TTS_DEVICE_METAL;
            else if (strcmp(d, "cuda") == 0) device = MYNAH_TTS_DEVICE_CUDA;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (model_dir == NULL || port <= 0 || port > 65535) { usage(argv[0]); return 2; }
    if (g.worker_count < 1) g.worker_count = 1;
    if (g.worker_count > 64) g.worker_count = 64;
    if (g.max_batch < 1u) g.max_batch = 1u;
    if (g.max_batch > mynah_tts_max_batch()) g.max_batch = mynah_tts_max_batch();
    if (g.max_batch > 16u) g.max_batch = 16u;   /* batch_run's job array */

    signal(SIGPIPE, SIG_IGN);   /* a client hanging up mid-stream is routine */

    char err[512];
    if (mynah_tts_model_open_device(model_dir, device, &g.model, err, sizeof(err)) != 0) {
        fprintf(stderr, "cannot open model: %s\n", err);
        return 1;
    }
    mynah_tts_model_get_info(g.model, &g.info);
    g.tokenizer = mynah_tokenizer_open(model_dir, err, sizeof(err));
    if (g.tokenizer == NULL) {
        fprintf(stderr, "cannot open tokenizer: %s\n", err);
        mynah_tts_model_close(g.model);
        return 1;
    }
    load_voices(model_dir);
    snprintf(g.model_id, sizeof(g.model_id), "%s-%s", g.info.engine, g.info.revision);
    g.default_speaker = g.voice_count > 0 ? g.voices[0].id : 0u;
    pthread_mutex_init(&g.synth_lock, NULL);
    pthread_mutex_init(&g_batch.mu, NULL);
    pthread_cond_init(&g_batch.arrived, NULL);
    if (pthread_create(&g_batch.thread, NULL, batch_scheduler, NULL) != 0) {
        fprintf(stderr, "cannot start the synthesis scheduler\n");
        mynah_tokenizer_close(g.tokenizer);
        mynah_tts_model_close(g.model);
        return 1;
    }
    g_batch.thread_started = 1;

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

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

    fprintf(stderr,
            "mynah-tts server on http://%s:%d  model=%s  device=%s  voices=%zu  "
            "workers=%d  max_batch=%zu  queue=%zu  timeout=%ums\n"
            "note: queued requests are synthesized together; streaming runs alone\n",
            host, port, g.model_id, mynah_tts_device_name(device), g.voice_count,
            g.worker_count, g.max_batch, g.max_pending, g.request_timeout_ms);

    queue_init(&g_queue);
    pthread_t workers[64];
    int worker_count = g.worker_count;
    for (int i = 0; i < worker_count; ++i) {
        if (pthread_create(&workers[i], NULL, worker_main, NULL) != 0) {
            worker_count = i;
            break;
        }
    }
    if (worker_count == 0) { fprintf(stderr, "cannot start workers\n"); return 1; }

    while (!g_shutdown) {
        struct pollfd pfd;
        pfd.fd = listen_fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        const int ready = poll(&pfd, 1, 200);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ready == 0) continue;   /* nothing yet: re-read the shutdown flag */

        const int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
                errno == ECONNABORTED) continue;
            break;
        }
        /* BSD hands the accepted socket the listener's O_NONBLOCK; Linux does
         * not. Clear it either way, because everything downstream reads and
         * writes this descriptor blocking, with a timeout. */
        const int cf = fcntl(fd, F_GETFL, 0);
        if (cf >= 0) fcntl(fd, F_SETFL, cf & ~O_NONBLOCK);
        /* A client that connects and never sends must not pin a worker
         * forever; without this the bounded queue would still fill up. */
        struct timeval tv = {CLIENT_TIMEOUT_S, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (queue_push(&g_queue, fd) != 0) {
            atomic_fetch_add(&g_stats.rejected, 1ul);
            /* Queue full: shed the connection rather than grow without bound. */
            const char *busy =
                "HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: 62\r\n"
                "Retry-After: 1\r\n"
                "Connection: close\r\n\r\n"
                "{\"error\":{\"message\":\"server busy\",\"type\":\"server_error\"}}";
            (void)!write(fd, busy, strlen(busy));
            close(fd);
        }
    }

    /* Shutdown, in the one order that leaves nothing dangling: stop taking
     * connections, let the workers finish the requests they are already
     * waiting on (they hold job references), then stop the scheduler, then
     * close whatever never got served. */
    pthread_mutex_lock(&g_queue.mu);
    g_queue.shutdown = 1;
    pthread_cond_broadcast(&g_queue.not_empty);
    pthread_mutex_unlock(&g_queue.mu);
    for (int i = 0; i < worker_count; ++i) pthread_join(workers[i], NULL);

    pthread_mutex_lock(&g_batch.mu);
    g_batch.stop = 1;
    pthread_cond_signal(&g_batch.arrived);
    pthread_mutex_unlock(&g_batch.mu);
    if (g_batch.thread_started) pthread_join(g_batch.thread, NULL);

    /* Connections parked but never picked up: their descriptors are still
     * ours, and this is the last chance to close them. */
    for (int parked = queue_pop(&g_queue); parked >= 0; parked = queue_pop(&g_queue)) {
        close(parked);
    }

    close(listen_fd);
    mynah_tokenizer_close(g.tokenizer);
    mynah_tts_model_close(g.model);
    pthread_cond_destroy(&g_batch.arrived);
    pthread_mutex_destroy(&g_batch.mu);
    pthread_cond_destroy(&g_queue.not_empty);
    pthread_mutex_destroy(&g_queue.mu);
    pthread_mutex_destroy(&g.synth_lock);
    return 0;
}
