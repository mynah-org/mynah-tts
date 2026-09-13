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

#include <stddef.h>

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

/* ---------------------------------------------------------------------------
 * SPIN BUDGET -- E5-22
 *
 * A worker with no work spins on the dispatch generation for this many
 * iterations before it parks on the condvar. It is the single
 * highest-leverage knob the reference implementation found, and it is
 * ISA-dependent by default because the measurement is:
 *
 *   QWEN_POOL_SPIN | STREAM p95 | ctx switches/s
 *          256     |   0.999    |    197,000
 *         4096     |   0.893    |     38,000
 *        65536     |   0.808    |      7,600      <- their aarch64 default
 *
 * and independently on ARM: CP 16.0 -> 9.6 ms/frame with context switches
 * 491,320 -> 35,132. Higher buys nothing; 262144 measured worse. On x86 8c
 * the ARM value is worse, and on Graviton5 the curve is NON-MONOTONIC
 * (.work/linux-production.md trap 9). So the default here is 65536 on
 * Linux/aarch64 and 4096 everywhere else -- transferred, not measured by us --
 * and MYNAH_POOL_SPIN overrides it. THE NUMBER DOES NOT PORT: pin the measured
 * value per box.
 *
 * Why it is the right shape for us specifically: our own thread-pool work
 * measured ~20 us of wake-up per region, so a region must be worth >= 200 us to
 * pay for itself. The spin sweep is that same wall from the other side, and it
 * says the fix is not fewer regions but not parking between them.
 *
 * Correctness is independent of the value. The publisher takes the mutex and
 * broadcasts UNCONDITIONALLY, and a worker re-checks the generation under that
 * same mutex before waiting, so the store-buffer litmus that deadlocks a
 * "publish, then peek at sleeping" pool on x86 (.work/linux-production.md trap
 * 8) cannot arise here: the spin is an optimisation layered outside a lock
 * protocol that was already correct. A budget of 0 parks immediately. */
int mynah_pool_spin_budget(void);
/* Where the budget came from, for the dispatch report: "env", "aarch64",
 * "default". Never NULL. */
const char *mynah_pool_spin_source(void);
/* Waits that reached the condvar, and waits the spin absorbed. The ratio is
 * the knob's own evidence; context switches per second is the external one. */
void mynah_pool_wait_stats(long long *parks, long long *spin_wins);

/* ---------------------------------------------------------------------------
 * DECODER LANE -- E5-21
 *
 * A private, PINNED team on the last N cpus of this process's slice, with its
 * own submit lock and a bounded one-unit-per-slot mailbox. Off by default.
 *
 * WHAT IT IS FOR. The decoder is per request and sits on the loop thread, so
 * its cost is multiplied by the number of concurrent streams and overlaps with
 * nothing. The reference measured it at 72-80% of the marginal cost of each
 * extra slot and moved its iteration wall p95 at B4 from 192 ms to 77 ms, and
 * stall@250 from 100% to 0%, by taking it off the critical path.
 *
 * THE TWO ATTEMPTS THAT FAILED FIRST, because they are what the design is
 * avoiding rather than incidental history (.work/serving-design.md §5):
 *
 *   - a second submitter on the ENGINE pool: TTFA p95 167 -> 1200 ms. It
 *     serialized on the submit lock behind the regions and lost the gang.
 *     This is why the lane has its own pool and its own lock, and why adding a
 *     second submitter to the engine pool is on the forbidden list.
 *   - a private team that was NOT pinned: 21 threads on 8 cores. This is why
 *     mynah_lane_split_prepare() REFUSES on a platform that cannot pin instead
 *     of quietly running unpinned.
 *
 * AND WHY IT MUST BE ABLE TO REFUSE. The split size was swept, not guessed: at
 * B4, 4+4 / 5+3 / 6+2 measured STREAM p95 0.997 / 1.113 / 1.364 against 1.203
 * inline, with the producer's mailbox wait going 3.6 -> 8.0 -> 16.6 ms per
 * frame. ON A NARROW LANE THE THING IS SLOWER THAN INLINE. 2 step + 6 decoder
 * was killed earlier still: on two threads the per-slot region work costs
 * +17 ms per slot. So both sides need >= MYNAH_LANE_MIN_CPUS cpus and a request
 * that cannot have them is refused with a reason, never silently narrowed.
 *
 * THE MECHANISM, and it is the part to copy exactly: the redirection happens
 * INSIDE mynah_parallel_for, keyed on a thread-local tag, not by discipline at
 * call sites. Everything a lane thread dispatches -- conv panels, SGEMM
 * slices, im2col, every future call site nobody has written yet -- lands on
 * the lane team because of one branch at the top of the dispatch primitive. A
 * convention would have to be remembered; this cannot be forgotten. */

/* Both sides of the split need at least this many cpus. See the sweep above. */
#define MYNAH_LANE_MIN_CPUS 4
/* One mailbox entry per driver slot; MYNAH_GRAPH_MAX_JOBS is 16. */
#define MYNAH_LANE_SLOTS    16

/* Build the lane team on `cpus[count - lane_cpus .. count)` and narrow the
 * engine pool to the rest.
 *
 * MUST be called with the process already pinned to `cpus[0 .. count)`, BEFORE
 * any pool thread exists, and from the only thread in the process. In a
 * prefork worker that is exactly one place, and the order is not optional:
 *
 *     sched_setaffinity(child) -> mynah_threadpool_after_fork() ->
 *     mynah_lane_split_prepare() -> (first dispatch builds the engine pool)
 *
 * because pthreads inherit the creating thread's mask. Pin first or the pool
 * is born unpinned and nothing later fixes it.
 *
 * `lane_cpus` of 0 turns the lane off and always succeeds. Returns 0 when the
 * lane is up, -1 otherwise, and either way writes a sentence into `why`
 * explaining what is in force. A refusal is not an error: the caller runs the
 * decoder inline, which is the default and is faster than a narrow lane. */
int mynah_lane_split_prepare(const int *cpus, int count, int lane_cpus,
                             char *why, size_t why_capacity);

/* Threads in the lane team, 0 when the lane is off. */
int mynah_lane_width(void);
/* Threads left to the engine pool: mynah_num_threads() minus the lane. */
int mynah_lane_engine_width(void);
/* 1 while THIS thread is a lane thread, i.e. while its dispatches are being
 * redirected. Exists so a caller can assert the redirection rather than
 * describe it. */
int mynah_lane_current(void);

/* Hand `fn(ud)` to the lane for mailbox slot `slot` (0 .. MYNAH_LANE_SLOTS).
 *
 * THE BOUNDED CONTRACT: at most one unit in flight per slot. A second submit
 * for a slot that already has one is refused with -2 and counted, because a
 * mailbox that grows is a queue, and a queue in front of the decoder is the
 * unbounded wait the admission ladder exists to prevent. The caller's job is
 * to block on that slot instead -- PER-SLOT blocking only; another slot's step
 * must never wait for this slot's decode.
 *
 * Returns 0 accepted, -1 the lane is off, -2 the contract would be violated.
 * `fn` runs on a lane thread with the redirection tag set, so everything it
 * dispatches stays on the lane. It must not free anything the submitter still
 * owns, and the submitter must not free anything the lane is still reading:
 * mynah_lane_wait() is the only thing that makes that safe. */
int mynah_lane_submit(int slot, void (*fn)(void *ud), void *ud);
/* 1 when this slot's unit has finished and is waiting to be reaped, 0 when it
 * is still running, -1 when nothing is in flight. Never blocks. */
int mynah_lane_finished(int slot);
/* Block until this slot's unit has finished, then clear the slot. Returns 0
 * when a unit was reaped, -1 when there was nothing in flight. Idempotent, so
 * a retire path can call it unconditionally. */
int mynah_lane_wait(int slot);
/* Times the bounded contract was violated. MUST stay 0 in every run; it is
 * printed rather than asserted so a violation is visible in a production log
 * instead of only in a debug build. */
long long mynah_lane_overruns(void);

/* ---------------------------------------------------------------------------
 * BLAS THREAD TIMEOUT -- E4-16a
 *
 * INTERIM COMPENSATION FOR A DEPENDENCY WE ARE REMOVING, not design. E4-16
 * deletes the BLAS call entirely; until it does, an idle OpenBLAS team spins,
 * and the reference measured what that costs:
 *
 *                       | without | with OPENBLAS_THREAD_TIMEOUT=1
 *   TTFA C=1            | 108 ms, BIMODAL | 66 ms, stable
 *   context switches/s  | 42,500  | 12,000
 *
 * Bimodal is the word that matters: a single run looks definitive whichever
 * mode it draws, so this is invisible to anyone who measures once.
 *
 * CLAIM VERSUS FACT. This process sets the variable if it is unset, as early
 * as it can. That is a backstop and not a guarantee: a shared libopenblas is
 * initialized before the executable's own constructors, so if that build reads
 * the variable in its constructor our write is too late. The only channel that
 * always works is the process environment before exec. The dispatch report
 * says which of the two happened rather than claiming success.
 *
 * Returns the value in force, or NULL when the variable is unset. */
const char *mynah_blas_thread_timeout(void);
/* 1 when this process set it, 0 when it was inherited or absent. */
int mynah_blas_thread_timeout_ours(void);

#endif
