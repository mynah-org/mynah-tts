/* Minimal pthread parallel-for for independent CPU loops (matmul row blocks,
 * codec channels).  Tasks must write to disjoint regions: the result is
 * BIT-IDENTICAL to the serial loop by construction (same task code, different
 * threads).  Thread count: env MYNAH_THREADS, default = online cores.
 *
 * Ported from the ../mynah house pattern (persistent pool, no hot-path spawn). */
#ifndef MYNAH_TTS_THREADS_H
#define MYNAH_TTS_THREADS_H

int mynah_num_threads(void);

/* Runs fn(ctx, i) for i in [0, n): tasks are distributed over
 * min(n, mynah_num_threads()) threads (the caller participates).
 * With n <= 1 or a single thread it runs inline with no spawn. */
void mynah_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx);

#endif
