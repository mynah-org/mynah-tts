/* A small persistent team for the serving loop's per-row host loops
 * (MYNAH_SERVE_HOST_THREADS).
 *
 * The CUDA serving loop is driven by one scheduler thread. At hundreds of rows
 * its per-row host loops (landing a decode gang's PCM, delivering it, the
 * early noise draws of a step, ...) add up to milliseconds per iteration while
 * the GPU waits. Those loops are embarrassingly parallel: row r reads and
 * writes only row r's own state. This team runs such a loop over a few threads.
 *
 * Contract for a body passed to mynah_hostpool_run:
 *   - fn(ud, begin, end) handles rows [begin, end) and touches nothing that
 *     another row's call touches (its own context, its own result slot);
 *   - it makes no device call (CUDA calls stay on the scheduler thread) and
 *     takes no lock that the scheduler holds across the call;
 *   - anything order-dependent (the first error reported, counters, device
 *     submissions) is left to a serial merge after the call, in row order.
 * Under that contract the result is bit-identical to the serial loop whatever
 * thread runs which rows: a row's inputs and outputs do not depend on who
 * computes it. Rows are handed out in fixed contiguous chunks claimed from an
 * atomic counter, so a worker that wakes late simply claims fewer chunks and
 * the caller (which always participates) never waits for a sleeping thread
 * to start.
 *
 * Off by default (opt-in until measured on a host-bound GPU); with =auto the
 * server picks a count from the cpus it may use. The team is private: it does not share the kernel pool of
 * threads.c, which the GPU server runs at MYNAH_THREADS=1. */
#ifndef MYNAH_TTS_HOSTPOOL_H
#define MYNAH_TTS_HOSTPOOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Threads that run a region, the caller included; 1 = off (everything inline).
 * Resolved once: MYNAH_SERVE_HOST_THREADS=N (0..16, 0 or 1 = off), or =auto
 * for the value set by mynah_hostpool_set_auto; unset = off. */
int mynah_hostpool_threads(void);

/* The server's automatic choice for this process (called before serving). */
void mynah_hostpool_set_auto(int threads);

/* Rows below which a region runs inline (MYNAH_SERVE_HOST_MIN_ROWS, default 64). */
size_t mynah_hostpool_min_rows(void);

/* Run fn over [0, n). Inline when the pool is off, n is below the row
 * threshold, or the call comes from inside a region. */
void mynah_hostpool_run(size_t n, void (*fn)(void *ud, size_t begin, size_t end),
                        void *ud);

/* Regions run in parallel and inline since start (profile line). */
void mynah_hostpool_stats(unsigned long long *parallel, unsigned long long *inline_runs);

#ifdef __cplusplus
}
#endif

#endif
