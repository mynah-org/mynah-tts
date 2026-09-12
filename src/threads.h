/* Minimal pthread parallel-for for independent CPU loops (matmul row blocks,
 * codec channels).  Tasks must write to disjoint regions: the result is
 * BIT-IDENTICAL to the serial loop by construction (same task code, different
 * threads).  Thread count: env MYNAH_THREADS, default = online cores.
 *
 * Ported from the ../mynah house pattern (persistent pool, no hot-path spawn),
 * then extended with the parts of ../qwen-tts's qwen_tts_thread.c that are
 * correctness and not tuning: a fork handler, several jobs in flight instead of
 * one slot, and capability predicates the pool answers honestly. */
#ifndef MYNAH_TTS_THREADS_H
#define MYNAH_TTS_THREADS_H

int mynah_num_threads(void);

/* Configure direct BLAS calls to match MYNAH_THREADS. */
void mynah_blas_set_threads(int n);
/* 1 when mynah_blas_set_threads() really does control the vendor BLAS: a
 * non-Apple GNU-C build, the weak openblas_set_num_threads resolved, and no
 * explicit OPENBLAS_NUM_THREADS. 0 means the clamp is a silent no-op. */
int mynah_blas_owned(void);

/* Runs fn(ctx, i) for i in [0, n): tasks are distributed over
 * min(n, mynah_num_threads()) threads (the caller participates).
 * With n <= 1 or a single thread it runs inline with no spawn. */
void mynah_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx);

/* ---------------------------------------------------------------------------
 * Everything below is additive: the three functions above keep their contract.
 * ------------------------------------------------------------------------- */

/* Rebuild the pool state in a forked child.  fork() copies only the calling
 * thread, so a child inherits worker bookkeeping with no workers behind it and
 * deadlocks on the first dispatch.  This resets the pool to "never started" so
 * the child builds its own; it allocates nothing and is safe from a
 * pthread_atfork child handler, which is where the pool registers it, so a
 * prefork server gets the fix without calling anything.  Calling it explicitly
 * after fork() is still supported and idempotent. */
void mynah_threadpool_after_fork(void);

/* Capability predicates.  Callers used to have to guess; a pool that is asked
 * one ambiguous question answers it differently on every backend, which is the
 * bug ../qwen-tts documents in its own header.  These are separate questions:
 *
 * concurrent_submit_ok: may two independent threads dispatch at the same time
 *   and both get workers?  (Not "does it not crash" -- the old pool was safe
 *   and silently ran one of the two regions serially.)
 * nested_dispatch_ok: may a task already running on the pool dispatch again?
 * priority_ok: does mynah_parallel_set_low_until() really deprioritise, or is
 *   it a no-op on this backend? */
int mynah_pool_concurrent_submit_ok(void);
int mynah_pool_nested_dispatch_ok(void);
int mynah_pool_priority_ok(void);

/* How many regions can be in flight before a submitter has to run inline. */
int mynah_pool_max_jobs(void);

/* 1 while this thread is executing chunks of a parallel region (caller or
 * worker). */
int mynah_parallel_active(void);

/* Dispatch priority: a submitter whose deadline is in the future asks for LOW,
 * and workers take its chunks only while no normal region has work left.  The
 * caller thread always runs its own chunks, so a LOW region still makes
 * progress alone; past until_ms it is ordinary again, which bounds starvation.
 * until_ms is CLOCK_MONOTONIC milliseconds (mynah_parallel_now_ms); 0 = normal. */
void mynah_parallel_set_low_until(double until_ms);
double mynah_parallel_now_ms(void);

/* Counters, so the degradations are visible instead of being felt as "the
 * decoder got slow".  Any pointer may be NULL.
 *   dispatches       regions handed to the pool
 *   serial           regions run inline because there was nothing to share
 *                    (n == 1, one thread, or no worker was ever created)
 *   inline_fallbacks regions run inline although the pool exists, because all
 *                    job slots were busy.  This is the one that should stay 0.
 *   helper_joins     times a worker joined someone else's region */
void mynah_parallel_stats(long long *dispatches, long long *serial,
                          long long *inline_fallbacks, long long *helper_joins);
void mynah_parallel_stats_reset(void);

#endif
