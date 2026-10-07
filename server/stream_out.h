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

/* Non-zero once the peer has hung up. Answers from the socket itself rather
 * than from a failed write, so a client that goes away during a long decode is
 * noticed while it happens instead of at the next chunk.
 *
 * It lives HERE, and not in the server's cancellation callback, because this
 * module owns the descriptor: the writer thread closes it and nothing outside
 * may name it. A poller elsewhere would be racing that close, and the number
 * it polled could by then belong to another client -- the same class of bug
 * job_claim_fd exists to rule out. The check runs under the same mutex the
 * writer takes to close, so the descriptor is guaranteed live for the syscall.
 *
 * Detection marks the stream failed, so the writer stops and releases the
 * socket without waiting for the producer. The poll itself never blocks
 * (zero timeout); it is meant to be called once per decoder step.
 *
 * Portability, stated rather than promised: POLLRDHUP is Linux-only, and
 * POLLHUP on a half-closed TCP socket is not portably raised. The fallback is
 * POLLIN plus a zero-length MSG_PEEK, which is what actually detects a closed
 * peer on macOS and BSD. A client that stops READING but keeps the socket open
 * is invisible to both, by construction -- that case is the queue's ceiling,
 * not this function's. */
int stream_out_peer_gone(stream_out *out);

/* No more chunks will be enqueued: let the writer drain and terminate. */
void stream_out_finish(stream_out *out);

/* Snapshot of the counters. Safe at any time; the writer may still be draining. */
void stream_out_get_stats(stream_out *out, stream_out_stats *stats);

/* Drops the producer's reference. `out` must not be touched afterwards. */
void stream_out_release(stream_out *out);

/* ---- MYNAH_STREAM_DELIVER_THREADS: enqueue off the producer's thread ----
 *
 * The producer side of a stream_out -- PCM16 conversion, the ring memcpy and
 * the condvar signal that wakes the writer -- is cheap per chunk and costly
 * per step: at hundreds of streams it is that many conversions, mutexes and
 * futex wakes in a row on the one scheduler thread. These calls move it onto
 * a small pool of helper threads.
 *
 * Ordering is the whole contract. Every stream is pinned to ONE helper for
 * its lifetime (round-robin at stream_out_start), and a helper serves its
 * queue strictly FIFO, so a stream's chunks reach its ring in the order they
 * were handed over and its end-of-stream comes after the last of them. Two
 * streams on different helpers are unordered relative to each other, which
 * nothing can observe: they are different sockets.
 *
 * Nothing changes on the wire. The helper runs the same stream_out_enqueue()
 * on the same samples, and the writer is untouched. */

/* Starts `threads` helpers. Call once, before any stream starts; 0 leaves
 * delivery synchronous. Returns 0, or -1 having started none. */
int stream_out_deliver_init(unsigned threads);

/* Non-zero once stream_out_deliver_init() started helpers. */
int stream_out_deliver_enabled(void);

/* The asynchronous stream_out_enqueue(): copies `count` floats into the
 * stream's helper queue and returns, so the caller's buffer is free again on
 * return. Returns -1 only for a stream already known to be dead or when the
 * copy cannot be allocated (the stream is then marked failed, as an
 * out-of-memory enqueue would). A queue overflow is discovered by the helper
 * and surfaces through stream_out_failed(), one step later than before. */
int stream_out_deliver(stream_out *out, const float *samples, size_t count);

/* stream_out_finish() + stream_out_release(), queued behind the stream's
 * pending chunks on its helper. Takes the producer's reference: `out` must not
 * be touched afterwards. Never fails -- the request for it lives inside the
 * stream_out, so it needs no allocation. */
void stream_out_deliver_close(stream_out *out);

#endif
