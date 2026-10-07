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

#include "http_util.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <unistd.h>

#define STREAM_OUT_DEFAULT_BYTES      (1u << 20)   /* 1 MiB */
#define STREAM_OUT_MIN_BYTES          4096u
#define STREAM_OUT_MAX_BYTES          (1u << 26)   /* 64 MiB; it is allocated up front */
#define STREAM_OUT_DEFAULT_TIMEOUT_MS 5000

/* One unit of work for a delivery helper (MYNAH_STREAM_DELIVER_THREADS): a
 * chunk to enqueue, or the stream's close. Chunk items are recycled through
 * the helper's free list, so after warm-up the producer never allocates; the
 * close item is embedded in the stream_out, so closing cannot fail. */
typedef struct deliver_item {
    struct deliver_item *next;
    stream_out *out;
    float *pcm;
    size_t count;
    size_t cap;               /* floats `pcm` can hold; grown, never shrunk */
    int close;
} deliver_item;

struct stream_out {
    /* Guarded by `mu` for READING from another thread; the writer thread is
     * the only one that mutates it, and only to -1 when it closes. Anyone
     * asking about the socket must hold `mu`, which is what keeps the close
     * and a peer-hangup poll from overlapping. */
    int fd;
    int send_timeout_ms;
    int one_write;            /* MYNAH_STREAM_OUT_WRITEV, read at start */

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

    /* MYNAH_STREAM_DELIVER_THREADS. `lane` is the helper this stream is
     * pinned to for life; `dead` mirrors `failed` without the mutex, so the
     * producer can stop copying chunks for a stream nobody will send. It is
     * only ever set, so a stale zero costs one wasted copy, never a lost
     * failure. */
    unsigned lane;
    _Atomic int dead;
    deliver_item close_item;
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

/* MYNAH_STREAM_OUT_WRITEV (default on; =0 is the rollback): send each HTTP
 * chunk (size line, PCM, CRLF) with one writev() instead of three send()
 * calls. The bytes on the wire are the same; what changes is a third of the
 * syscalls on the writer threads, which at hundreds of streams share the
 * host's cores with the scheduler thread. */
static int stream_out_one_write(void) {
    const char *e = getenv("MYNAH_STREAM_OUT_WRITEV");
    return e == NULL || strcmp(e, "0") != 0;
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

/* The delivery pool (MYNAH_STREAM_DELIVER_THREADS); empty unless started. */
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    deliver_item *head;       /* FIFO, guarded by mu */
    deliver_item *tail;
    deliver_item *spare;      /* recycled chunk items, guarded by mu */
    int idle;                 /* the helper is parked on cv */
    unsigned index;
} deliver_lane;

static deliver_lane *g_lanes;
static unsigned g_lane_count;      /* written once, before any stream exists */
static _Atomic unsigned g_next_lane;

/* Round-robin rather than a hash of the pointer: streams are admitted one at
 * a time, so this balances the helpers exactly, and a stream keeps the lane
 * it was given for as long as it lives -- which is the ordering guarantee. */
static unsigned stream_out_deliver_pick_lane(void) {
    if (g_lane_count == 0u) return 0u;
    return atomic_fetch_add_explicit(&g_next_lane, 1u, memory_order_relaxed) %
           g_lane_count;
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

/* The same contract as write_all_or_gone, for a gather list: everything or
 * the peer is gone. A short write advances through the vector in place. */
static int writev_all_or_gone(int fd, struct iovec *iov, int count) {
    while (count > 0) {
        const ssize_t n = writev(fd, iov, count);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0) errno = EPIPE;
            return -1;
        }
        size_t left = (size_t)n;
        while (count > 0 && left >= iov->iov_len) {
            left -= iov->iov_len;
            ++iov;
            --count;
        }
        if (count > 0) {
            iov->iov_base = (char *)iov->iov_base + left;
            iov->iov_len -= left;
        }
    }
    return 0;
}

/* One HTTP chunk: size line, body, CRLF. */
static int write_chunk(int fd, const unsigned char *data, size_t len, int one_write) {
    char size_line[32];
    const int n = snprintf(size_line, sizeof(size_line), "%zx\r\n", len);
    if (n <= 0 || (size_t)n >= sizeof(size_line)) { errno = EINVAL; return -1; }
    if (one_write) {
        struct iovec iov[3];
        iov[0].iov_base = size_line;
        iov[0].iov_len = (size_t)n;
        iov[1].iov_base = (void *)data;
        iov[1].iov_len = len;
        iov[2].iov_base = (void *)"\r\n";
        iov[2].iov_len = 2u;
        return writev_all_or_gone(fd, iov, 3);
    }
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
    atomic_store_explicit(&out->dead, 1, memory_order_relaxed);
    out->producer_done = 1;
    pthread_cond_broadcast(&out->cv);
}

/* -------------------------------------------------------------- the writer */

static void *stream_out_writer_main(void *arg) {
    stream_out *out = (stream_out *)arg;

    /* One writer per stream, so the descriptor is the only identity available
     * and also the useful one: it is what `lsof` and the server's own logs
     * name. Set from this thread because that is the only spelling both
     * platforms share -- see mynah_thread_set_name(). */
    {
        char name[16];
        snprintf(name, sizeof(name), "mynah-out%d", out->fd);
        mynah_thread_set_name(name);
    }

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

        if (write_chunk(out->fd, out->ring + offset, span, out->one_write) != 0) {
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

    /* Closed under the mutex, and the field retired to -1 in the same critical
     * section. stream_out_peer_gone() may be polling this descriptor from the
     * scheduler thread; it holds `mu` while it does, so the close either
     * happens entirely before its poll (and it sees -1 and reports gone) or
     * entirely after (and its poll ran on a descriptor that was still ours).
     * Without the lock the number could be reissued by accept() between the
     * two, and the poll would be asking about a stranger's connection. */
    pthread_mutex_lock(&out->mu);
    const int doomed = out->fd;
    out->fd = -1;
    if (doomed >= 0) close(doomed);
    pthread_mutex_unlock(&out->mu);

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
    out->one_write = stream_out_one_write();
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
    atomic_init(&out->dead, 0);
    out->lane = stream_out_deliver_pick_lane();

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

/* Which recorded failures mean "the peer went away" as opposed to "the peer is
 * still there and not keeping up". The distinction is the whole point of the
 * disconnect counter: a hangup frees a slot nobody wanted, a slow reader is a
 * capacity problem, and an operator who sees them as one number cannot tell a
 * flaky client population from an undersized machine.
 *
 * Deliberately NOT in this set: errno 0, which is the queue-overflow
 * cancellation (a reader too slow, socket still open), and EAGAIN, which is an
 * SO_SNDTIMEO expiry -- same thing, arriving by a different route. */
static int failure_is_hangup(int err) {
    switch (err) {
        case EPIPE:
        case ECONNRESET:
        case ENOTCONN:
#ifdef ESHUTDOWN
        case ESHUTDOWN:
#endif
            return 1;
        default:
            return 0;
    }
}

int stream_out_peer_gone(stream_out *out) {
    if (out == NULL) return 1;

    pthread_mutex_lock(&out->mu);
    /* Already failed. The answer comes from WHY, not from the socket: by the
     * time a write has returned EPIPE the descriptor may already be closed,
     * and in practice the writer usually discovers the hangup before this poll
     * does -- it is the thread actually touching the socket. Reporting only
     * what the poll caught would have credited one disconnect in four and
     * filed the rest under "timed out", which is how the counter read before
     * this branch existed. */
    if (out->failed) {
        const int gone = failure_is_hangup(out->failure_errno);
        pthread_mutex_unlock(&out->mu);
        return gone;
    }
    const int fd = out->fd;
    if (fd < 0) { pthread_mutex_unlock(&out->mu); return 1; }

    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
#ifdef POLLRDHUP
    pfd.events |= POLLRDHUP;   /* Linux: the half-close, without a read */
#endif
    pfd.revents = 0;

    int gone = 0;
    const int ready = poll(&pfd, 1, 0);   /* zero timeout: never blocks on mu */
    if (ready < 0) {
        /* EINTR and EAGAIN are "ask again"; anything else means this
         * descriptor can no longer be polled, which is a dead stream. */
        gone = (errno != EINTR && errno != EAGAIN);
    } else if (ready > 0) {
        if ((pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
            gone = 1;
#ifdef POLLRDHUP
        } else if ((pfd.revents & POLLRDHUP) != 0) {
            gone = 1;
#endif
        } else if ((pfd.revents & POLLIN) != 0) {
            /* The portable half. Readable means either the peer sent
             * something (a pipelined byte we will never read) or it closed.
             * Only a zero-length peek tells the two apart, and it cannot block
             * because poll() just said the socket is readable. */
            char probe;
            const ssize_t n = recv(fd, &probe, 1, MSG_PEEK);
            if (n == 0) {
                gone = 1;
            } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                       errno != EINTR) {
                gone = 1;
            }
        }
    }

    /* Failing the stream here is what frees the socket promptly: the writer is
     * parked on the condvar waiting for PCM nobody will read, and this wakes
     * it to close and go. The decoder is untouched -- it learns at its own
     * frame boundary, through the server's cancellation callback. */
    if (gone) mark_failed_locked(out, EPIPE);
    pthread_mutex_unlock(&out->mu);
    return gone;
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

/* ------------------------------------------------- the delivery helpers */

/* One helper. It takes the whole queue at once, so a step's worth of chunks
 * costs it one lock rather than one per chunk, and runs each item exactly as
 * the producer would have: stream_out_enqueue() for a chunk -- the only
 * caller of it for this stream, which keeps `convert` single-owner -- and
 * finish + release for a close. Nothing here touches the model, the driver or
 * the request; the stream_out and this lane are all a helper ever sees. */
static void *stream_out_deliver_main(void *arg) {
    deliver_lane *lane = (deliver_lane *)arg;
    {
        char name[16];
        snprintf(name, sizeof(name), "mynah-dlv%u", lane->index);
        mynah_thread_set_name(name);
    }
    pthread_mutex_lock(&lane->mu);
    for (;;) {
        while (lane->head == NULL) {
            lane->idle = 1;
            pthread_cond_wait(&lane->cv, &lane->mu);
            lane->idle = 0;
        }
        deliver_item *it = lane->head;
        lane->head = NULL;
        lane->tail = NULL;
        pthread_mutex_unlock(&lane->mu);

        deliver_item *spent = NULL;
        deliver_item *spent_tail = NULL;
        while (it != NULL) {
            /* Read before the item can go away: a close item lives inside
             * the stream it releases. */
            deliver_item *next = it->next;
            if (it->close) {
                stream_out *out = it->out;
                stream_out_finish(out);
                stream_out_release(out);
            } else {
                /* A refusal marks the stream failed inside enqueue, exactly
                 * as it did on the producer's thread; the producer learns it
                 * from stream_out_failed() at its next step. */
                (void)stream_out_enqueue(it->out, it->pcm, it->count);
                it->out = NULL;
                it->next = NULL;
                if (spent_tail != NULL) spent_tail->next = it;
                else spent = it;
                spent_tail = it;
            }
            it = next;
        }

        pthread_mutex_lock(&lane->mu);
        if (spent_tail != NULL) {
            spent_tail->next = lane->spare;
            lane->spare = spent;
        }
    }
    return NULL;
}

int stream_out_deliver_init(unsigned threads) {
    if (threads == 0u || g_lanes != NULL) return 0;
    deliver_lane *lanes = (deliver_lane *)calloc(threads, sizeof(*lanes));
    if (lanes == NULL) return -1;
    for (unsigned i = 0; i < threads; ++i) {
        lanes[i].index = i;
        if (pthread_mutex_init(&lanes[i].mu, NULL) != 0 ||
            pthread_cond_init(&lanes[i].cv, NULL) != 0) {
            /* Nothing has started yet; the few primitives initialised so far
             * are leaked rather than tracked, once, at start-up. */
            free(lanes);
            return -1;
        }
    }
    for (unsigned i = 0; i < threads; ++i) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, stream_out_deliver_main, &lanes[i]) != 0) {
            /* Helpers already running are parked on an empty queue and will
             * never be handed work: g_lane_count stays 0, so every stream
             * keeps synchronous delivery. They cost a stack each. */
            return -1;
        }
        pthread_detach(thread);
    }
    g_lanes = lanes;
    g_lane_count = threads;
    return 0;
}

int stream_out_deliver_enabled(void) {
    return g_lane_count > 0u;
}

int stream_out_deliver(stream_out *out, const float *samples, size_t count) {
    if (out == NULL || samples == NULL || count == 0) return -1;
    if (atomic_load_explicit(&out->dead, memory_order_relaxed)) {
        /* What an enqueue on a failed stream answers, counted the same way. */
        pthread_mutex_lock(&out->mu);
        ++out->failed_enqueues;
        pthread_mutex_unlock(&out->mu);
        return -1;
    }
    deliver_lane *lane = &g_lanes[out->lane];

    pthread_mutex_lock(&lane->mu);
    deliver_item *it = lane->spare;
    if (it != NULL) lane->spare = it->next;
    pthread_mutex_unlock(&lane->mu);

    if (it == NULL) it = (deliver_item *)calloc(1, sizeof(*it));
    if (it != NULL && count > it->cap) {
        float *grown = count <= SIZE_MAX / sizeof(float)
            ? (float *)realloc(it->pcm, count * sizeof(float)) : NULL;
        if (grown == NULL) {
            pthread_mutex_lock(&lane->mu);
            it->next = lane->spare;
            lane->spare = it;
            pthread_mutex_unlock(&lane->mu);
            it = NULL;
        } else {
            it->pcm = grown;
            it->cap = count;
        }
    }
    if (it == NULL) {
        pthread_mutex_lock(&out->mu);
        ++out->failed_enqueues;
        mark_failed_locked(out, ENOMEM);
        pthread_mutex_unlock(&out->mu);
        return -1;
    }
    /* The copy is what makes the caller's buffer free again on return. */
    memcpy(it->pcm, samples, count * sizeof(float));
    it->out = out;
    it->count = count;
    it->close = 0;
    it->next = NULL;

    pthread_mutex_lock(&lane->mu);
    if (lane->tail != NULL) lane->tail->next = it;
    else lane->head = it;
    lane->tail = it;
    /* Only a parked helper needs the futex: a busy one finds this item when
     * it comes back for the next batch. That is most of the saving at width,
     * one wake per step per helper instead of one per stream. */
    if (lane->idle) pthread_cond_signal(&lane->cv);
    pthread_mutex_unlock(&lane->mu);
    return 0;
}

void stream_out_deliver_close(stream_out *out) {
    if (out == NULL) return;
    if (g_lane_count == 0u) {
        stream_out_finish(out);
        stream_out_release(out);
        return;
    }
    deliver_lane *lane = &g_lanes[out->lane];
    deliver_item *it = &out->close_item;
    it->out = out;
    it->count = 0;
    it->close = 1;
    it->next = NULL;
    pthread_mutex_lock(&lane->mu);
    if (lane->tail != NULL) lane->tail->next = it;
    else lane->head = it;
    lane->tail = it;
    if (lane->idle) pthread_cond_signal(&lane->cv);
    pthread_mutex_unlock(&lane->mu);
}
