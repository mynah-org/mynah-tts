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
 */
#include "http_util.h"

#include "mynah_tts.h"
#include "tokenizer.h"

#include <errno.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_BODY (1u << 20)          /* 1 MiB of JSON is far beyond any prompt */
#define MAX_TEXT 8192u
#define STREAM_CHUNK 4096u           /* samples per streamed chunk */
#define CONN_QUEUE_CAP 256u          /* connections parked before we shed load */
#define CLIENT_TIMEOUT_S 30          /* recv deadline, so a silent client cannot pin a worker */

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
    pthread_mutex_t synth_lock;
} g;

/* -------------------------------------------------- batched synthesis queue
 *
 * Offline requests are parked here instead of synthesizing on their own
 * thread. One scheduler thread drains the queue, so requests that arrive while
 * a batch is running are grouped into the next one -- the queue depth does the
 * batching, and BATCH_WINDOW_US only helps requests that land near-simultaneously
 * on an idle server. */
#define BATCH_WINDOW_US 2000

typedef struct synth_ticket {
    mynah_tts_request request;
    float *samples;
    size_t count;
    char error[256];
    int result;
    int done;
    struct synth_ticket *next;
} synth_ticket;

static struct {
    pthread_mutex_t mu;
    pthread_cond_t arrived;
    pthread_cond_t completed;
    synth_ticket *head;
    synth_ticket *tail;
    size_t pending;
    int stop;
    int running;
    pthread_t thread;
} g_batch;

static void batch_run(synth_ticket **taken, size_t count) {
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

static void *batch_scheduler(void *arg) {
    (void)arg;
    synth_ticket *taken[16];
    for (;;) {
        pthread_mutex_lock(&g_batch.mu);
        while (g_batch.pending == 0u && !g_batch.stop) {
            pthread_cond_wait(&g_batch.arrived, &g_batch.mu);
        }
        if (g_batch.stop && g_batch.pending == 0u) {
            pthread_mutex_unlock(&g_batch.mu);
            break;
        }
        /* Give near-simultaneous arrivals a moment to join this batch. */
        if (g_batch.pending < g.max_batch) {
            pthread_mutex_unlock(&g_batch.mu);
            usleep(BATCH_WINDOW_US);
            pthread_mutex_lock(&g_batch.mu);
        }
        size_t count = 0;
        while (count < g.max_batch && g_batch.head != NULL) {
            synth_ticket *t = g_batch.head;
            g_batch.head = t->next;
            if (g_batch.head == NULL) g_batch.tail = NULL;
            t->next = NULL;
            taken[count++] = t;
            --g_batch.pending;
        }
        g_batch.running = 1;
        pthread_mutex_unlock(&g_batch.mu);

        batch_run(taken, count);

        pthread_mutex_lock(&g_batch.mu);
        g_batch.running = 0;
        for (size_t i = 0; i < count; ++i) taken[i]->done = 1;
        pthread_cond_broadcast(&g_batch.completed);
        pthread_mutex_unlock(&g_batch.mu);
    }
    return NULL;
}

/* Park one request and block until the scheduler has synthesized it. */
static int batch_submit(synth_ticket *ticket) {
    ticket->next = NULL;
    ticket->done = 0;
    ticket->samples = NULL;
    ticket->count = 0;
    ticket->result = 0;
    ticket->error[0] = '\0';
    pthread_mutex_lock(&g_batch.mu);
    if (g_batch.tail == NULL) g_batch.head = ticket;
    else g_batch.tail->next = ticket;
    g_batch.tail = ticket;
    ++g_batch.pending;
    pthread_cond_signal(&g_batch.arrived);
    while (!ticket->done) pthread_cond_wait(&g_batch.completed, &g_batch.mu);
    pthread_mutex_unlock(&g_batch.mu);
    return ticket->result;
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

typedef struct {
    int fd;
    int failed;
    int failure_errno;   /* why the write failed, for the log */
    size_t sent_bytes;   /* how far the stream got before it stopped */
} stream_sink;

/* One HTTP chunk per callback: the caller already emits stable causal
 * prefixes, so a chunk is exactly what has become final. */
static int stream_callback(const float *samples, size_t count, void *user_data) {
    stream_sink *sink = (stream_sink *)user_data;
    if (sink->failed || count == 0) return sink->failed ? -1 : 0;

    int16_t *pcm = (int16_t *)malloc(count * sizeof(*pcm));
    if (pcm == NULL) {
        sink->failed = 1;
        sink->failure_errno = ENOMEM;
        return -1;
    }
    for (size_t i = 0; i < count; ++i) pcm[i] = to_pcm16(samples[i]);

    char size_line[32];
    const int n = snprintf(size_line, sizeof(size_line), "%zx\r\n",
                           count * sizeof(*pcm));
    if (n <= 0 ||
        write_all(sink->fd, size_line, (size_t)n) != 0 ||
        write_all(sink->fd, pcm, count * sizeof(*pcm)) != 0 ||
        write_all(sink->fd, "\r\n", 2) != 0) {
        sink->failed = 1;
        sink->failure_errno = errno;
        free(pcm);
        return -1;
    }
    sink->sent_bytes += count * sizeof(*pcm);
    free(pcm);
    return 0;
}

/* ------------------------------------------------------------------ routes */

/* Serves both the OpenAI shape ("input"/"voice") and the native one
 * ("text"/"speaker"), so a caller need not pretend to be OpenAI to be clear. */
static void handle_speech(int fd, const char *body) {
    char text[MAX_TEXT];
    if (mynah_json_string(body, "input", text, sizeof(text)) != 0 &&
        mynah_json_string(body, "text", text, sizeof(text)) != 0) {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "missing 'input' (or 'text')");
        return;
    }
    if (text[0] == '\0') {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "empty 'input'");
        return;
    }

    char voice[64] = {0};
    if (mynah_json_string(body, "voice", voice, sizeof(voice)) != 0) {
        (void)mynah_json_string(body, "speaker", voice, sizeof(voice));
    }
    unsigned speaker = 0;
    if (resolve_voice(voice, &speaker) != 0) {
        send_error(fd, "400 Bad Request", "invalid_request_error",
                   "unknown 'voice' — GET /v1/voices lists the available ones");
        return;
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
        return;
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
        return;
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
         * needs the total length up front, so streaming is raw PCM only. */
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
        if (hn <= 0 || write_all(fd, head, (size_t)hn) != 0) { free(ids); return; }

        stream_sink sink = {fd, 0, 0, 0};
        mynah_tts_stream *st = NULL;
        /* The stream takes its text through push(), not through the request:
         * leaving text_ids set here would feed the same tokens twice and
         * produce a longer utterance than the batch path for the same input. */
        mynah_tts_request stream_request = request;
        stream_request.text_ids = NULL;
        stream_request.text_length = 0;
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
        free(ids);

        /* Terminate the chunked body either way; a mid-stream failure cannot
         * become an HTTP status because the header is long gone. */
        write_all(fd, "0\r\n\r\n", 5);
        /* A stream that stops early must never be silent: this used to return
         * without a word when the socket write was what failed, which makes a
         * truncated response indistinguishable from a short utterance. */
        if (sink.failed) {
            fprintf(stderr, "stream aborted after %zu bytes: %s\n",
                    sink.sent_bytes,
                    sink.failure_errno != 0 ? strerror(sink.failure_errno)
                                            : "client write failed");
        } else if (rc != 0) {
            fprintf(stderr, "stream failed after %zu bytes: %s\n",
                    sink.sent_bytes, err);
        }
        return;
    }

    synth_ticket ticket;
    memset(&ticket, 0, sizeof(ticket));
    ticket.request = request;
    const int rc = batch_submit(&ticket);
    float *samples = ticket.samples;
    const size_t count = ticket.count;
    free(ids);
    if (rc != 0) {
        send_error(fd, "500 Internal Server Error", "server_error", ticket.error);
        return;
    }

    const size_t pcm_bytes = count * sizeof(int16_t);
    unsigned char *out = (unsigned char *)malloc(44u + pcm_bytes);
    if (out == NULL) {
        mynah_tts_free_samples(samples);
        send_error(fd, "500 Internal Server Error", "server_error", "out of memory");
        return;
    }
    size_t offset = 0;
    if (!want_pcm) {
        wav_header(out, (uint32_t)pcm_bytes, g.info.sample_rate);
        offset = 44;
    }
    int16_t *pcm = (int16_t *)(out + offset);
    for (size_t i = 0; i < count; ++i) pcm[i] = to_pcm16(samples[i]);
    mynah_tts_free_samples(samples);

    send_status(fd, "200 OK", want_pcm ? "audio/pcm" : "audio/wav",
                (const char *)out, offset + pcm_bytes);
    free(out);
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

static void handle_health(int fd) {
    char body[256];
    const int n = snprintf(body, sizeof(body),
                           "{\"status\":\"ok\",\"model\":\"%s\",\"engine\":\"%s\","
                           "\"sample_rate\":%u,\"voices\":%zu}",
                           g.model_id, g.info.engine, g.info.sample_rate,
                           g.voice_count);
    if (n > 0) send_status(fd, "200 OK", "application/json", body, (size_t)n);
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

    if (strncmp(buf, "OPTIONS ", 8) == 0) {
        const char *pre =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Connection: close\r\n\r\n";
        write_all(fd, pre, strlen(pre));
    } else if (strncmp(buf, "POST /v1/audio/speech", 21) == 0 ||
               strncmp(buf, "POST /v1/tts", 12) == 0) {
        handle_speech(fd, body);
    } else if (strncmp(buf, "GET /v1/voices", 14) == 0) {
        handle_voices(fd);
    } else if (strncmp(buf, "GET /v1/models", 14) == 0) {
        handle_models(fd);
    } else if (strncmp(buf, "GET /health", 11) == 0) {
        handle_health(fd);
    } else {
        send_error(fd, "404 Not Found", "invalid_request_error", "unknown route");
    }

    free(buf);
    close(fd);
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

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL_DIR [-p PORT] [--host ADDR] [-w WORKERS]\n"
            "       [--device cpu|metal|cuda] [--max-batch N]\n"
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
    pthread_cond_init(&g_batch.completed, NULL);
    if (pthread_create(&g_batch.thread, NULL, batch_scheduler, NULL) != 0) {
        fprintf(stderr, "cannot start the synthesis scheduler\n");
        mynah_tts_model_close(g.model);
        return 1;
    }

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

    fprintf(stderr,
            "mynah-tts server on http://%s:%d  model=%s  device=%s  voices=%zu  "
            "workers=%d  max_batch=%zu\n"
            "note: queued requests are synthesized together; streaming runs alone\n",
            host, port, g.model_id, mynah_tts_device_name(device), g.voice_count,
            g.worker_count, g.max_batch);

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

    for (;;) {
        const int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            break;
        }
        /* A client that connects and never sends must not pin a worker
         * forever; without this the bounded queue would still fill up. */
        struct timeval tv = {CLIENT_TIMEOUT_S, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (queue_push(&g_queue, fd) != 0) {
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

    pthread_mutex_lock(&g_queue.mu);
    g_queue.shutdown = 1;
    pthread_cond_broadcast(&g_queue.not_empty);
    pthread_mutex_unlock(&g_queue.mu);
    for (int i = 0; i < worker_count; ++i) pthread_join(workers[i], NULL);

    close(listen_fd);
    mynah_tokenizer_close(g.tokenizer);
    mynah_tts_model_close(g.model);
    pthread_mutex_destroy(&g.synth_lock);
    return 0;
}
