/* Asynchronous chunked-output writer. See stream_out.h for the ownership and
 * cancellation contract.
 *
 * The queue is a single byte ring allocated once at start, not a list of
 * malloc'd chunks: a stream produces a chunk every few milliseconds and the
 * hot path should not touch the allocator at all. The producer converts into
 * a reusable PCM16 scratch buffer outside the lock and only holds the mutex
 * for the memcpy into the ring.
 *
 * The writer drains the contiguous readable span and writes it straight out of
 * the ring while NOT holding the lock -- safe because the span is only handed
 * back to the producer (queued -= span) after the write has completed, so the
 * region the writer is reading is never one the producer may fill.
 */
#include "stream_out.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define STREAM_OUT_DEFAULT_BYTES      (1u << 20)   /* 1 MiB */
#define STREAM_OUT_MIN_BYTES          4096u
#define STREAM_OUT_MAX_BYTES          (1u << 26)   /* 64 MiB; it is allocated up front */
#define STREAM_OUT_DEFAULT_TIMEOUT_MS 5000

struct stream_out {
    int fd;
    int send_timeout_ms;

    char *header;
    size_t header_len;

    /* Producer-only, so no lock: grown on demand, never per chunk. */
    int16_t *convert;
    size_t convert_samples;

    pthread_mutex_t mu;
    pthread_cond_t cv;

    /* Guarded by mu. */
    unsigned char *ring;
    size_t capacity;
    size_t head;              /* the writer's read cursor */
    size_t queued;            /* readable bytes; the tail is head + queued */
    size_t peak_bytes;
    size_t sent_bytes;
    unsigned long enqueued_chunks;
    unsigned long failed_enqueues;
    int producer_done;
    int failed;
    int failure_errno;

    _Atomic int refs;         /* producer + detached writer */
};

/* ------------------------------------------------------------ configuration */

static size_t stream_out_capacity_bytes(void) {
    const char *e = getenv("MYNAH_STREAM_OUT_MAX_BYTES");
    if (e == NULL || e[0] == '\0') return STREAM_OUT_DEFAULT_BYTES;
    char *end = NULL;
    const unsigned long long v = strtoull(e, &end, 10);
    if (end == e || *end != '\0' ||
        v < STREAM_OUT_MIN_BYTES || v > STREAM_OUT_MAX_BYTES) {
        return STREAM_OUT_DEFAULT_BYTES;
    }
    return (size_t)v;
}

static int stream_out_timeout_ms(void) {
    const char *e = getenv("MYNAH_STREAM_OUT_SEND_TIMEOUT_MS");
    if (e == NULL || e[0] == '\0') return STREAM_OUT_DEFAULT_TIMEOUT_MS;
    char *end = NULL;
    const long v = strtol(e, &end, 10);
    if (end == e || *end != '\0' || v < 1 || v > 120000) {
        return STREAM_OUT_DEFAULT_TIMEOUT_MS;
    }
    return (int)v;
}

/* ------------------------------------------------------------------ socket */

/* Writes everything or reports the peer as gone. A SO_SNDTIMEO expiry arrives
 * as EAGAIN/EWOULDBLOCK: that is a client which has stopped reading, not a
 * transient condition to retry, so it ends the stream like any other failure.
 * Retrying would reinstate exactly the 30-second stall this module removes. */
static int write_all_or_gone(int fd, const void *data, size_t len) {
    const char *p = (const char *)data;
    while (len > 0) {
        const ssize_t n = send(fd, p, len, 0);
        if (n > 0) {
            p += (size_t)n;
            len -= (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n == 0) errno = EPIPE;
        return -1;
    }
    return 0;
}

/* One HTTP chunk: size line, body, CRLF. */
static int write_chunk(int fd, const unsigned char *data, size_t len) {
    char size_line[32];
    const int n = snprintf(size_line, sizeof(size_line), "%zx\r\n", len);
    if (n <= 0 || (size_t)n >= sizeof(size_line)) { errno = EINVAL; return -1; }
    if (write_all_or_gone(fd, size_line, (size_t)n) != 0) return -1;
    if (write_all_or_gone(fd, data, len) != 0) return -1;
    return write_all_or_gone(fd, "\r\n", 2);
}

/* ------------------------------------------------------------- lifetime */

static void stream_out_destroy(stream_out *out) {
    pthread_cond_destroy(&out->cv);
    pthread_mutex_destroy(&out->mu);
    free(out->ring);
    free(out->header);
    free(out->convert);
    free(out);
}

void stream_out_release(stream_out *out) {
    if (out == NULL) return;
    if (atomic_fetch_sub(&out->refs, 1) != 1) return;
    stream_out_destroy(out);
}

/* Abandon the stream: the producer's next enqueue fails and the writer stops
 * without emitting a terminator, because a truncated chunked body is how a
 * reader learns the response is incomplete. */
static void mark_failed_locked(stream_out *out, int err) {
    if (!out->failed) {
        out->failed = 1;
        out->failure_errno = err;
    }
    out->producer_done = 1;
    pthread_cond_broadcast(&out->cv);
}

/* -------------------------------------------------------------- the writer */

static void *stream_out_writer_main(void *arg) {
    stream_out *out = (stream_out *)arg;

    if (out->header_len > 0 &&
        write_all_or_gone(out->fd, out->header, out->header_len) != 0) {
        pthread_mutex_lock(&out->mu);
        mark_failed_locked(out, errno);
        pthread_mutex_unlock(&out->mu);
    }

    for (;;) {
        pthread_mutex_lock(&out->mu);
        while (out->queued == 0 && !out->producer_done && !out->failed) {
            pthread_cond_wait(&out->cv, &out->mu);
        }
        if (out->failed) { pthread_mutex_unlock(&out->mu); break; }
        if (out->queued == 0) { pthread_mutex_unlock(&out->mu); break; }
        /* Only the span up to the ring's end; a wrap becomes a second chunk. */
        size_t span = out->queued;
        if (span > out->capacity - out->head) span = out->capacity - out->head;
        const size_t offset = out->head;
        pthread_mutex_unlock(&out->mu);

        if (write_chunk(out->fd, out->ring + offset, span) != 0) {
            const int err = errno;
            pthread_mutex_lock(&out->mu);
            mark_failed_locked(out, err);
            pthread_mutex_unlock(&out->mu);
            break;
        }

        pthread_mutex_lock(&out->mu);
        out->head = (out->head + span) % out->capacity;
        out->queued -= span;
        out->sent_bytes += span;
        pthread_mutex_unlock(&out->mu);
    }

    pthread_mutex_lock(&out->mu);
    int failed = out->failed;
    pthread_mutex_unlock(&out->mu);

    if (!failed && write_all_or_gone(out->fd, "0\r\n\r\n", 5) != 0) {
        const int err = errno;
        pthread_mutex_lock(&out->mu);
        mark_failed_locked(out, err);
        pthread_mutex_unlock(&out->mu);
        failed = 1;
    }

    close(out->fd);

    /* A stream that stops early must never be silent: a truncated response is
     * otherwise indistinguishable from a short utterance. */
    if (failed) {
        stream_out_stats st;
        stream_out_get_stats(out, &st);
        fprintf(stderr,
                "stream aborted after %zu bytes: %s "
                "(queue %zu/%zu peak, chunks=%lu, refused=%lu)\n",
                st.sent_bytes,
                st.failure_errno != 0 ? strerror(st.failure_errno)
                                      : "client stopped reading",
                st.peak_bytes, st.capacity_bytes,
                st.enqueued_chunks, st.failed_enqueues);
    }

    stream_out_release(out);
    return NULL;
}

/* -------------------------------------------------------------- the producer */

stream_out *stream_out_start(int fd, const char *header) {
    if (fd < 0) return NULL;
    stream_out *out = (stream_out *)calloc(1, sizeof(*out));
    if (out == NULL) return NULL;

    out->fd = fd;
    out->send_timeout_ms = stream_out_timeout_ms();
    out->capacity = stream_out_capacity_bytes();
    out->ring = (unsigned char *)malloc(out->capacity);
    if (out->ring == NULL) { free(out); return NULL; }

    if (header != NULL && header[0] != '\0') {
        out->header_len = strlen(header);
        out->header = (char *)malloc(out->header_len + 1u);
        if (out->header == NULL) { free(out->ring); free(out); return NULL; }
        memcpy(out->header, header, out->header_len + 1u);
    }

    if (pthread_mutex_init(&out->mu, NULL) != 0) {
        free(out->header); free(out->ring); free(out);
        return NULL;
    }
    if (pthread_cond_init(&out->cv, NULL) != 0) {
        pthread_mutex_destroy(&out->mu);
        free(out->header); free(out->ring); free(out);
        return NULL;
    }
    atomic_init(&out->refs, 2);   /* producer + writer */

    struct timeval tv;
    tv.tv_sec = out->send_timeout_ms / 1000;
    tv.tv_usec = (out->send_timeout_ms % 1000) * 1000;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    pthread_t thread;
    if (pthread_create(&thread, NULL, stream_out_writer_main, out) != 0) {
        /* The fd was never handed over: destroy never closes it, so it goes
         * back to the caller intact. */
        stream_out_destroy(out);
        return NULL;
    }
    pthread_detach(thread);
    return out;
}

int stream_out_enqueue(stream_out *out, const float *samples, size_t count) {
    if (out == NULL || samples == NULL || count == 0) return -1;
    if (count > SIZE_MAX / sizeof(int16_t)) return -1;
    const size_t bytes = count * sizeof(int16_t);

    if (count > out->convert_samples) {
        int16_t *grown = (int16_t *)realloc(out->convert, bytes);
        if (grown == NULL) {
            pthread_mutex_lock(&out->mu);
            ++out->failed_enqueues;
            mark_failed_locked(out, ENOMEM);
            pthread_mutex_unlock(&out->mu);
            return -1;
        }
        out->convert = grown;
        out->convert_samples = count;
    }
    /* Conversion happens outside the lock; the writer must never wait on it. */
    for (size_t i = 0; i < count; ++i) {
        float v = samples[i];
        if (!(v > -1.0f)) v = -1.0f;
        if (!(v < 1.0f)) v = 1.0f;
        out->convert[i] = (int16_t)(v * 32767.0f);
    }

    pthread_mutex_lock(&out->mu);
    if (out->failed || out->producer_done) {
        ++out->failed_enqueues;
        pthread_mutex_unlock(&out->mu);
        return -1;
    }
    if (bytes > out->capacity - out->queued) {
        /* Backpressure is cancellation: a reader too slow to keep up loses its
         * own stream rather than holding the synthesis thread hostage. */
        ++out->failed_enqueues;
        mark_failed_locked(out, 0);
        pthread_mutex_unlock(&out->mu);
        return -1;
    }

    const size_t tail = (out->head + out->queued) % out->capacity;
    const size_t first = bytes < out->capacity - tail ? bytes : out->capacity - tail;
    memcpy(out->ring + tail, out->convert, first);
    if (first < bytes) {
        memcpy(out->ring, (const unsigned char *)out->convert + first, bytes - first);
    }
    out->queued += bytes;
    if (out->queued > out->peak_bytes) out->peak_bytes = out->queued;
    ++out->enqueued_chunks;
    pthread_cond_signal(&out->cv);
    pthread_mutex_unlock(&out->mu);
    return 0;
}

int stream_out_failed(stream_out *out) {
    if (out == NULL) return 1;
    pthread_mutex_lock(&out->mu);
    const int failed = out->failed;
    pthread_mutex_unlock(&out->mu);
    return failed;
}

void stream_out_finish(stream_out *out) {
    if (out == NULL) return;
    pthread_mutex_lock(&out->mu);
    out->producer_done = 1;
    pthread_cond_broadcast(&out->cv);
    pthread_mutex_unlock(&out->mu);
}

void stream_out_get_stats(stream_out *out, stream_out_stats *stats) {
    if (out == NULL || stats == NULL) return;
    pthread_mutex_lock(&out->mu);
    stats->capacity_bytes = out->capacity;
    stats->peak_bytes = out->peak_bytes;
    stats->sent_bytes = out->sent_bytes;
    stats->enqueued_chunks = out->enqueued_chunks;
    stats->failed_enqueues = out->failed_enqueues;
    stats->failed = out->failed;
    stats->failure_errno = out->failure_errno;
    pthread_mutex_unlock(&out->mu);
}
