/* Asynchronous chunked-output writer for one streaming HTTP response.
 *
 * Why this exists: synthesis used to write PCM to the client socket from the
 * synthesis thread, inside the server's synthesis critical section. A client
 * that stopped reading stalled that write until SO_SNDTIMEO expired, and with
 * it every other request in the process. Socket I/O therefore moves onto a
 * thread of its own, and the synthesis thread only ever copies bytes into a
 * bounded queue.
 *
 * The contract, stated plainly because it is the part that bites:
 *
 *   - stream_out_start() TAKES OWNERSHIP of the fd on success. The writer
 *     thread sends the HTTP header, the chunks, the terminating "0\r\n\r\n",
 *     and then closes the fd. Nobody else may write to it or close it. On
 *     failure the fd is untouched and still belongs to the caller.
 *   - The writer thread is detached, so it can outlive the request that
 *     started it. Lifetime is an atomic refcount of two: the producer drops
 *     its reference with stream_out_release(), the writer drops its own when
 *     it is finished, and the last one out frees everything.
 *   - Backpressure is cancellation, not blocking. When the queue is full the
 *     enqueue fails and the stream is marked failed; the producer discovers
 *     this (from the enqueue return value or stream_out_failed()) and aborts
 *     the request. A slow reader costs its own stream, never the process.
 *   - SIGPIPE must be ignored process-wide; the writer relies on EPIPE.
 *
 * Queue capacity defaults to 1 MiB (MYNAH_STREAM_OUT_MAX_BYTES) and the send
 * timeout to 5000 ms (MYNAH_STREAM_OUT_SEND_TIMEOUT_MS).
 */
#ifndef MYNAH_SERVER_STREAM_OUT_H
#define MYNAH_SERVER_STREAM_OUT_H

#include <stddef.h>

typedef struct stream_out stream_out;

typedef struct {
    size_t capacity_bytes;        /* queue ceiling, the backpressure limit */
    size_t peak_bytes;            /* deepest the queue ever got */
    size_t sent_bytes;            /* PCM bytes actually written to the socket */
    unsigned long enqueued_chunks;
    unsigned long failed_enqueues;
    int failed;
    int failure_errno;            /* 0 when the failure was not a socket error */
} stream_out_stats;

/* Starts the detached writer and hands it `fd`. `header` (may be NULL) is
 * copied and sent by the writer before the first chunk. Returns NULL without
 * touching `fd` if the writer could not be started. */
stream_out *stream_out_start(int fd, const char *header);

/* Converts `count` floats to PCM16 and copies them into the queue. Returns 0,
 * or -1 when the stream is gone -- queue full, socket dead, or out of memory.
 * Never blocks on the socket. */
int stream_out_enqueue(stream_out *out, const float *samples, size_t count);

/* Non-zero once the stream has been abandoned. Cheap enough to poll per chunk. */
int stream_out_failed(stream_out *out);

/* No more chunks will be enqueued: let the writer drain and terminate. */
void stream_out_finish(stream_out *out);

/* Snapshot of the counters. Safe at any time; the writer may still be draining. */
void stream_out_get_stats(stream_out *out, stream_out_stats *stats);

/* Drops the producer's reference. `out` must not be touched afterwards. */
void stream_out_release(stream_out *out);

#endif
