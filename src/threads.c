/* The shared parallel-for pool, the decoder lane, and the two knobs whose
 * values are measurements rather than opinions.
 *
 * There are TWO pools in this file and exactly one place that chooses between
 * them: the branch at the top of mynah_parallel_for(). The engine pool serves
 * the scheduler thread; the lane pool serves the decoder team pinned to the
 * tail of this process's cpu slice. Which one a dispatch lands on is decided by
 * a thread-local tag, never by the call site -- see threads.h on E5-21 for why
 * that distinction is the whole mechanism.
 *
 * Ported from the ../mynah house pattern (persistent pool, no hot-path spawn),
 * then extended with the parts of ../qwen-tts's qwen_tts_thread.c that are
 * correctness and not tuning: a fork handler, several jobs in flight instead of
 * one slot, capability predicates the pool answers honestly, and -- E5-22 --
 * a spin-then-park wait whose budget is a tunable with an ISA-dependent
 * default.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "threads.h"

#include "dispatch.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

#define PF_MAX_THREADS 64
/* Regions that may be in flight at once.  One per synthesis thread is the
 * shape that matters (the server's scheduler plus whatever the codec lane
 * becomes); 8 is far above that and costs 8 * sizeof(pf_job) of .bss. */
#define PF_MAX_JOBS 8

/* Default width of the pool.  Decode is DRAM-bandwidth bound, so on an
 * asymmetric core layout the efficiency cores add little bandwidth while
 * lengthening every barrier to the slowest worker.  On Apple Silicon the
 * performance-core count is the better default: measured on M1 (4P+4E),
 * 4 threads gives 0.482 s versus 0.507 s for all 8.  Any other platform, and
 * any Mac without perflevel reporting, keeps the online-CPU count.
 * MYNAH_THREADS always wins. */
static long default_threads(void) {
#if defined(__APPLE__)
    int perf = 0;
    size_t size = sizeof(perf);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &perf, &size, NULL, 0) == 0 &&
        perf > 0) {
        return perf;
    }
#endif
#if defined(__linux__) && defined(_GNU_SOURCE)
    /* E4-17.  sysconf(_SC_NPROCESSORS_ONLN) counts the machine; it sees neither
     * an inherited taskset mask nor a cpuset cgroup.  A prefork worker pinned to
     * eight cpus was therefore starting THIRTY-TWO threads, and with four
     * workers that is 128 threads on 32 cores -- the same oversubscription we
     * had just finished removing from OpenBLAS, kept in our own pool.
     *
     * Measured on the production box: OpenBLAS sized itself from the affinity
     * mask and correctly picked 8, our pool picked 32, and the clamp we had
     * written to protect us then overwrote their right answer with our wrong
     * one, costing 24 threads and 11% of wall. This is that finding's other
     * half, and the one that was ours.
     *
     * The mask is the honest denominator: it is what the scheduler will
     * actually let us run on, whether that came from taskset, from a cpuset
     * cgroup, or from our own prefork pinning. Fall through to the machine
     * count only if the call fails or reports nothing. */
    {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof allowed, &allowed) == 0) {
            const int n = CPU_COUNT(&allowed);
            if (n > 0) return (long)n;
        }
    }
#endif
    return sysconf(_SC_NPROCESSORS_ONLN);
}

int mynah_num_threads(void) {
    static int nth = 0;
    if (nth == 0) {
        const char *env = getenv("MYNAH_THREADS");
        long n = env ? atol(env) : default_threads();
        if (n < 1) n = 1;
        if (n > PF_MAX_THREADS) n = PF_MAX_THREADS;
        nth = (int)n;
    }
    return nth;
}

double mynah_parallel_now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------------------
 * E5-22: the spin budget.
 *
 * Resolved once and cached, so a dispatch never calls getenv(). The default is
 * transferred from the reference's measurement, not measured by us, and the
 * source is reported alongside the value precisely so nobody reads 65536 here
 * as a number this repository established.
 * ------------------------------------------------------------------------- */
#if defined(__linux__) && defined(__aarch64__)
#define PF_SPIN_DEFAULT 65536
#define PF_SPIN_DEFAULT_WHY "aarch64"
#else
#define PF_SPIN_DEFAULT 4096
#define PF_SPIN_DEFAULT_WHY "default"
#endif

static int g_spin = -1;
static const char *g_spin_why = PF_SPIN_DEFAULT_WHY;

static int spin_budget(void) {
    if (g_spin < 0) {
        const char *env = getenv("MYNAH_POOL_SPIN");
        if (env != NULL && env[0] != '\0') {
            const long v = atol(env);
            /* 0 is meaningful -- park immediately -- so only a negative value
             * falls back to the default. */
            g_spin = v >= 0 ? (int)(v > 1 << 24 ? 1 << 24 : v) : PF_SPIN_DEFAULT;
            g_spin_why = v >= 0 ? "env" : PF_SPIN_DEFAULT_WHY;
        } else {
            g_spin = PF_SPIN_DEFAULT;
            g_spin_why = PF_SPIN_DEFAULT_WHY;
        }
    }
    return g_spin;
}

int mynah_pool_spin_budget(void) { return spin_budget(); }
const char *mynah_pool_spin_source(void) { (void)spin_budget(); return g_spin_why; }

/* `pause` on x86, `yield` on aarch64, a compiler barrier elsewhere. The
 * barrier is not decoration: without it the spin loop's load can be hoisted
 * out and the wait becomes infinite. */
static inline void pf_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

static void pf_name_self(const char *name) {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#elif defined(__APPLE__)
    pthread_setname_np(name);
#else
    (void)name;
#endif
}

/* ---------------------------------------------------------------------------
 * Job slots.
 *
 * The old pool had exactly one slot behind a trylock: a second caller found it
 * busy and ran its whole region inline, single-threaded, with no error.  That
 * is invisible until two threads synthesize at once, and then it reads as "the
 * decoder is slow".  Here every submitter gets its own slot and workers pick
 * whichever live slot still has chunks, so two regions share the pool instead
 * of one of them losing it.  Chunk assignment stays a shared atomic counter per
 * region, so the work each task does is unchanged and the output is still
 * bit-identical to the serial loop.
 * ------------------------------------------------------------------------- */
typedef struct {
    void (*fn)(void *, int);
    void *ctx;
    atomic_int next;   /* next chunk index; also how workers see "drained" */
    int n;
    /* Workers currently inside this region. Mutated only under the pool mutex;
     * ATOMIC so a submitter can spin on it outside the lock before parking. */
    atomic_int refs;
    int live;          /* slot published and not yet reclaimed  (pool->mu)    */
    int low;           /* submitted with a future deadline      (pool->mu)    */
} pf_job;

/* One pool. Two instances exist: the engine's and the decoder lane's. They
 * share no state at all -- separate mutex, separate generation, separate job
 * slots, separate workers -- which is the answer to the reference's first
 * failed attempt, a second submitter on the one pool that took TTFA p95 from
 * 167 to 1200 ms. */
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  job_cv;    /* work available */
    pthread_cond_t  done_cv;   /* a region's refs hit zero */
    pf_job          jobs[PF_MAX_JOBS];
    /* Bumped on every publish; the workers' wakeup key. Mutated only under
     * `mu`, ATOMIC so a spinning worker can read it without the lock. */
    atomic_uint     gen;
    unsigned        rotor;     /* start index for the slot scan, for fairness */
    int             workers;
    const char     *thread_name;
} pf_pool;

static pf_pool g_engine = { .mu = PTHREAD_MUTEX_INITIALIZER,
                            .job_cv = PTHREAD_COND_INITIALIZER,
                            .done_cv = PTHREAD_COND_INITIALIZER };
static pf_pool g_lane   = { .mu = PTHREAD_MUTEX_INITIALIZER,
                            .job_cv = PTHREAD_COND_INITIALIZER,
                            .done_cv = PTHREAD_COND_INITIALIZER };

static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;
/* Atomic so the uncontended fast path in pool_init_once() can read it without
 * the init mutex and still be a well-defined read, not a benign-looking race. */
static _Atomic int g_pool_inited;

static _Atomic long long g_stat_dispatch, g_stat_serial, g_stat_inline,
    g_stat_helper, g_stat_park, g_stat_spin_win;
#define PF_STAT(c) atomic_fetch_add_explicit(&(c), 1, memory_order_relaxed)

void mynah_parallel_stats(long long *dispatches, long long *serial,
                          long long *inline_fallbacks, long long *helper_joins) {
    if (dispatches) *dispatches = atomic_load(&g_stat_dispatch);
    if (serial) *serial = atomic_load(&g_stat_serial);
    if (inline_fallbacks) *inline_fallbacks = atomic_load(&g_stat_inline);
    if (helper_joins) *helper_joins = atomic_load(&g_stat_helper);
}

void mynah_parallel_stats_reset(void) {
    atomic_store(&g_stat_dispatch, 0);
    atomic_store(&g_stat_serial, 0);
    atomic_store(&g_stat_inline, 0);
    atomic_store(&g_stat_helper, 0);
    atomic_store(&g_stat_park, 0);
    atomic_store(&g_stat_spin_win, 0);
}

void mynah_pool_wait_stats(long long *parks, long long *spin_wins) {
    if (parks) *parks = atomic_load(&g_stat_park);
    if (spin_wins) *spin_wins = atomic_load(&g_stat_spin_win);
}

static _Thread_local int g_depth;
static _Thread_local double g_low_until;
/* THE TAG. 1 on every thread of the lane team, and on nothing else. It is read
 * by exactly one branch, at the top of mynah_parallel_for(). */
static _Thread_local int g_on_lane;

int mynah_parallel_active(void) { return g_depth > 0; }
void mynah_parallel_set_low_until(double until_ms) { g_low_until = until_ms; }
int mynah_pool_max_jobs(void) { return PF_MAX_JOBS; }
int mynah_lane_current(void) { return g_on_lane; }

#define PF_LANE_MAX_CPUS PF_MAX_THREADS

typedef struct {
    void (*fn)(void *);
    void *ud;
    /* 0 empty, 1 queued, 2 running, 3 finished and waiting to be reaped. The
     * whole bounded contract is "state must be 0 to submit". */
    int   state;
} lane_box;

static int g_lane_width;                    /* 0 = the lane is off */
static int g_lane_cpus[PF_LANE_MAX_CPUS];
static pthread_mutex_t g_lane_mu = PTHREAD_MUTEX_INITIALIZER;  /* its OWN lock */
static pthread_cond_t  g_lane_cv = PTHREAD_COND_INITIALIZER;   /* consumer */
static pthread_cond_t  g_lane_done_cv = PTHREAD_COND_INITIALIZER; /* producers */
static lane_box g_lane_box[MYNAH_LANE_SLOTS];
static unsigned g_lane_rotor;
static _Atomic long long g_lane_overruns;

int mynah_lane_width(void) { return g_lane_width; }

int mynah_lane_engine_width(void) {
    const int w = mynah_num_threads() - g_lane_width;
    return w < 1 ? 1 : w;
}

long long mynah_lane_overruns(void) { return atomic_load(&g_lane_overruns); }

/* Pin the calling thread to `cpus[first .. first+count)`. 1 on success, 0 when
 * the platform has no such call -- never a silent no-op, because "pinned" is
 * the load-bearing word in the whole item. */
static int lane_pin(const int *cpus, int first, int count) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = first; i < first + count; ++i) CPU_SET(cpus[i], &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    (void)cpus; (void)first; (void)count;
    return 0;
#endif
}

/* Concurrent submitters both get workers (that is the point of the slot array).
 * A nested dispatch is safe because no lock is held while a region runs and a
 * submitter can always finish its own region alone: worst case its slot is the
 * last one and it runs inline.  Priority is real, not a stub. */
int mynah_pool_concurrent_submit_ok(void) { return 1; }
int mynah_pool_nested_dispatch_ok(void) { return 1; }
int mynah_pool_priority_ok(void) { return 1; }

static void pf_run(pf_job *st) {
    g_depth++;
    for (;;) {
        const int i = atomic_fetch_add_explicit(&st->next, 1, memory_order_relaxed);
        if (i >= st->n) break;
        st->fn(st->ctx, i);
    }
    g_depth--;
}

/* Caller holds pool->mu.  Picks the region that most needs a hand: among slots
 * with chunks left, the one with the fewest workers already inside, starting
 * the scan at a rotating index so equal candidates alternate.  Low-priority
 * regions are only considered when nothing normal has work left. */
static pf_job *pick_job(pf_pool *pool) {
    pf_job *best = NULL;
    int best_refs = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < PF_MAX_JOBS; k++) {
            pf_job *j = &pool->jobs[(pool->rotor + (unsigned)k) % PF_MAX_JOBS];
            if (!j->live || (pass == 0 && j->low) || (pass == 1 && !j->low)) continue;
            if (atomic_load_explicit(&j->next, memory_order_relaxed) >= j->n) continue;
            const int refs = atomic_load_explicit(&j->refs, memory_order_relaxed);
            if (best == NULL || refs < best_refs) { best = j; best_refs = refs; }
        }
        if (best) break;
    }
    if (best) {
        atomic_fetch_add_explicit(&best->refs, 1, memory_order_relaxed);
        pool->rotor++;
    }
    return best;
}

/* SPIN, THEN PARK -- E5-22.
 *
 * The lock protocol is unchanged and is what makes this correct: the publisher
 * bumps `gen` and broadcasts while holding `mu`, and this function re-checks
 * `gen` under the same `mu` before waiting. The spin below never decides
 * whether to broadcast, so no wakeup can be lost however the store buffer
 * behaves -- which is the difference between this and the "publish, then peek
 * at sleeping" pool that deadlocks on x86 and silently does not on ARM
 * (.work/linux-production.md trap 8). We develop on ARM; that bug would not
 * have shown up here.
 *
 * Returns with the lock NOT held. */
static void pf_wait_for_work(pf_pool *pool, unsigned seen) {
    const int budget = spin_budget();
    for (int i = 0; i < budget; ++i) {
        if (atomic_load_explicit(&pool->gen, memory_order_acquire) != seen) {
            PF_STAT(g_stat_spin_win);
            return;
        }
        pf_cpu_relax();
    }
    pthread_mutex_lock(&pool->mu);
    if (atomic_load_explicit(&pool->gen, memory_order_relaxed) == seen) {
        PF_STAT(g_stat_park);
        do {
            pthread_cond_wait(&pool->job_cv, &pool->mu);
        } while (atomic_load_explicit(&pool->gen, memory_order_relaxed) == seen);
    }
    pthread_mutex_unlock(&pool->mu);
}

static void *pool_worker(void *arg) {
    pf_pool *pool = (pf_pool *)arg;
    if (pool == &g_lane) {
        /* The tag, set once. From here every dispatch this thread makes is
         * redirected by the branch at the top of mynah_parallel_for(). */
        g_on_lane = 1;
        /* Belt and braces on the mask. The creator was pinned to the lane
         * cpus when it made this thread, so the inherited mask is already
         * right; re-asserting it here means the invariant does not depend on
         * the creator's ordering surviving a future edit. */
        lane_pin(g_lane_cpus, 0, g_lane_width);
    }
    pf_name_self(pool->thread_name != NULL ? pool->thread_name : "mynah-pool");
    for (;;) {
        pthread_mutex_lock(&pool->mu);
        const unsigned seen = atomic_load_explicit(&pool->gen, memory_order_relaxed);
        pf_job *job = pick_job(pool);
        pthread_mutex_unlock(&pool->mu);
        if (job == NULL) {
            /* `seen` was read under the lock, so a publish either happened
             * before the scan (and was picked) or bumps gen after: the
             * re-check inside pf_wait_for_work closes the window. */
            pf_wait_for_work(pool, seen);
            continue;
        }
        PF_STAT(g_stat_helper);
        pf_run(job);
        pthread_mutex_lock(&pool->mu);
        if (atomic_fetch_sub_explicit(&job->refs, 1, memory_order_relaxed) == 1) {
            pthread_cond_broadcast(&pool->done_cv);
        }
        pthread_mutex_unlock(&pool->mu);
    }
    return NULL; /* never reached */
}

/* ---------------------------------------------------------------------------
 * The decoder lane -- E5-21.
 * ------------------------------------------------------------------------- */

/* The lane's single consumer. One unit runs at a time, by construction: the
 * lane's own submit lock is what the reference's first attempt lacked when it
 * put a second submitter on the engine pool. Everything this thread dispatches
 * goes to the lane pool, because g_on_lane is set here and mynah_parallel_for
 * reads it. */
static void *lane_consumer(void *arg) {
    (void)arg;
    g_on_lane = 1;
    lane_pin(g_lane_cpus, 0, g_lane_width);
    pf_name_self("mynah-lane");
    for (;;) {
        pthread_mutex_lock(&g_lane_mu);
        lane_box *box = NULL;
        for (int k = 0; k < MYNAH_LANE_SLOTS; ++k) {
            lane_box *b = &g_lane_box[(g_lane_rotor + (unsigned)k) % MYNAH_LANE_SLOTS];
            if (b->state == 1) { box = b; break; }
        }
        if (box == NULL) {
            pthread_cond_wait(&g_lane_cv, &g_lane_mu);
            pthread_mutex_unlock(&g_lane_mu);
            continue;
        }
        ++g_lane_rotor;
        box->state = 2;
        void (*fn)(void *) = box->fn;
        void *ud = box->ud;
        pthread_mutex_unlock(&g_lane_mu);

        fn(ud);

        pthread_mutex_lock(&g_lane_mu);
        box->state = 3;
        pthread_cond_broadcast(&g_lane_done_cv);
        pthread_mutex_unlock(&g_lane_mu);
    }
    return NULL; /* never reached */
}

int mynah_lane_split_prepare(const int *cpus, int count, int lane_cpus,
                             char *why, size_t why_capacity) {
    if (why != NULL && why_capacity > 0) why[0] = '\0';
    if (lane_cpus <= 0) {
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane off: the decoder runs inline on the engine pool "
            "(the default; a lane is only a win on a wide enough pinned slice)");
        return -1;
    }
    if (cpus == NULL || count <= 0) {
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane refused: no cpu list to split");
        return -1;
    }
    if (g_lane_width != 0) {
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane refused: already prepared with %d cpus", g_lane_width);
        return -1;
    }
    if (count > PF_LANE_MAX_CPUS) {
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane refused: a %d-cpu slice is past this pool's %d-thread "
            "ceiling", count, PF_LANE_MAX_CPUS);
        return -1;
    }
    /* THE POOL MUST NOT EXIST YET. pthreads inherit the creating thread's
     * mask, so a pool built before the split is a pool on the wrong cpus, and
     * nothing later moves it. This is the one ordering in the whole item that
     * is not optional. */
    if (atomic_load_explicit(&g_pool_inited, memory_order_acquire) != 0 ||
        g_engine.workers != 0) {
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane refused: the engine pool already exists, so its "
            "threads inherited the full mask and cannot be narrowed");
        return -1;
    }
    /* THE TWO REFUSALS THAT ARE THE ITEM, and the reason it is a gate rather
     * than a knob. A narrow lane is SLOWER than inline -- 6+2 measured STREAM
     * p95 1.364 against 0.997 at 4+4, with the producer's mailbox wait going
     * 3.6 -> 16.6 ms per frame -- and a narrow engine side is worse still, at
     * +17 ms of per-slot region work on two threads. Both sides are checked
     * against the cpu slice AND against the pool width, because they can
     * disagree: a slice of 8 with MYNAH_THREADS=2 would pass the first test
     * and leave the engine with nothing. */
    const int engine_cpus = count - lane_cpus;
    const int pool_width = mynah_num_threads();
    if (lane_cpus < MYNAH_LANE_MIN_CPUS || engine_cpus < MYNAH_LANE_MIN_CPUS ||
        pool_width - lane_cpus < MYNAH_LANE_MIN_CPUS) {
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane refused: %d cpus (pool width %d) split %d+%d, and both "
            "sides need at least %d. On a narrow lane the decoder is SLOWER than "
            "inline -- 6+2 measured STREAM p95 1.364 against 0.997 at 4+4 -- so "
            "this runs inline instead",
            count, pool_width, engine_cpus, lane_cpus, MYNAH_LANE_MIN_CPUS);
        return -1;
    }

    for (int i = 0; i < lane_cpus; ++i) g_lane_cpus[i] = cpus[engine_cpus + i];
    g_lane_width = lane_cpus;

    /* PIN FIRST, THEN CREATE. The lane threads are created while this thread
     * holds the LANE mask, so they inherit it; this thread is then moved to
     * the engine mask, and the engine pool -- built lazily on its first
     * dispatch, from this same thread -- inherits that. Getting this backwards
     * is how the reference ended up with a private team that was not pinned at
     * all: 21 threads on 8 cores.
     *
     * A platform with no affinity call fails here, deliberately, rather than
     * running an unpinned private team. */
    if (!lane_pin(cpus, engine_cpus, lane_cpus)) {
        g_lane_width = 0;
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane refused: cannot pin to cpus %d..%d (this platform may "
            "have no affinity call). An UNPINNED private team is the reference's "
            "second failed attempt -- 21 threads on 8 cores -- so the decoder "
            "runs inline instead",
            cpus[engine_cpus], cpus[count - 1]);
        return -1;
    }

    g_lane.thread_name = "mynah-lanew";
    int started = 0;
    pthread_t tid;
    if (pthread_create(&tid, NULL, lane_consumer, NULL) == 0) {
        pthread_detach(tid);
        ++started;
    }
    for (int k = 1; k < lane_cpus && started > 0; ++k) {
        if (pthread_create(&tid, NULL, pool_worker, &g_lane) != 0) break;
        pthread_detach(tid);
        ++started;
        ++g_lane.workers;
    }

    /* Back to the engine side before anything else happens on this thread.
     * Done even when thread creation failed: leaving the scheduler pinned to
     * the lane cpus would be worse than having no lane at all. */
    const int narrowed = lane_pin(cpus, 0, engine_cpus);

    if (started == 0) {
        g_lane_width = 0;
        g_lane.workers = 0;
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane refused: no lane thread could be created");
        return -1;
    }
    if (!narrowed) {
        /* The lane exists and is pinned, but the engine side is not confined,
         * so the two overlap. Say so rather than reporting a clean split. */
        if (why != NULL) snprintf(why, why_capacity,
            "decoder lane ON but UNCONFINED: %d lane threads pinned to %d..%d, "
            "and this process could NOT be narrowed to its first %d cpus, so the "
            "engine pool overlaps the lane",
            lane_cpus, g_lane_cpus[0], g_lane_cpus[lane_cpus - 1], engine_cpus);
        return 0;
    }
    if (why != NULL) snprintf(why, why_capacity,
        "decoder lane ON: %d+%d (engine+lane) of %d cpus, lane pinned to %d..%d, "
        "own submit lock, at most one unit in flight per slot",
        engine_cpus, lane_cpus, count, g_lane_cpus[0], g_lane_cpus[lane_cpus - 1]);
    return 0;
}

int mynah_lane_submit(int slot, void (*fn)(void *ud), void *ud) {
    if (g_lane_width == 0 || fn == NULL) return -1;
    if (slot < 0 || slot >= MYNAH_LANE_SLOTS) return -1;
    pthread_mutex_lock(&g_lane_mu);
    lane_box *box = &g_lane_box[slot];
    if (box->state != 0) {
        pthread_mutex_unlock(&g_lane_mu);
        /* The bounded contract, violated. Printed rather than asserted so it
         * is visible in a production log; the counter must stay 0. */
        atomic_fetch_add(&g_lane_overruns, 1);
        fprintf(stderr, "MAILBOX OVERRUN on decoder-lane slot %d: a second unit "
                        "was submitted while one was in flight; the bounded "
                        "contract was violated\n", slot);
        return -2;
    }
    box->fn = fn;
    box->ud = ud;
    box->state = 1;
    pthread_cond_signal(&g_lane_cv);
    pthread_mutex_unlock(&g_lane_mu);
    return 0;
}

int mynah_lane_finished(int slot) {
    if (g_lane_width == 0 || slot < 0 || slot >= MYNAH_LANE_SLOTS) return -1;
    pthread_mutex_lock(&g_lane_mu);
    const int state = g_lane_box[slot].state;
    pthread_mutex_unlock(&g_lane_mu);
    if (state == 0) return -1;
    return state == 3 ? 1 : 0;
}

int mynah_lane_wait(int slot) {
    if (g_lane_width == 0 || slot < 0 || slot >= MYNAH_LANE_SLOTS) return -1;
    pthread_mutex_lock(&g_lane_mu);
    lane_box *box = &g_lane_box[slot];
    if (box->state == 0) {
        pthread_mutex_unlock(&g_lane_mu);
        return -1;
    }
    while (box->state != 3) pthread_cond_wait(&g_lane_done_cv, &g_lane_mu);
    box->state = 0;
    box->fn = NULL;
    box->ud = NULL;
    pthread_mutex_unlock(&g_lane_mu);
    return 0;
}

static void pool_init(pf_pool *pool, int width) {
    for (int k = 0; k < width - 1; k++) {
        pthread_t tid;
        if (pthread_create(&tid, NULL, pool_worker, pool) == 0) {
            pthread_detach(tid);
            pool->workers++;
        }
    }
}

/* fork() keeps the calling thread only.  Without this the child inherits
 * workers > 0 with no workers behind it, broadcasts to nobody and then waits
 * forever for refs to drop: an immediate deadlock, and the reason a prefork
 * server was impossible.  Resetting to "never started" makes the child build
 * its own pool on its next dispatch.  Nothing here allocates or takes a lock
 * that a dead thread might hold, which is what makes it legal from an atfork
 * child handler.
 *
 * The lane is reset to OFF for the same reason and one more: a lane is a
 * property of a cpu slice, and a child has a different slice than its parent.
 * The child re-prepares it after it pins itself. */
void mynah_threadpool_after_fork(void) {
    pf_pool *const pools[2] = { &g_engine, &g_lane };
    for (int p = 0; p < 2; ++p) {
        pthread_mutex_init(&pools[p]->mu, NULL);
        pthread_cond_init(&pools[p]->job_cv, NULL);
        pthread_cond_init(&pools[p]->done_cv, NULL);
        memset(pools[p]->jobs, 0, sizeof(pools[p]->jobs));
        atomic_store_explicit(&pools[p]->gen, 0, memory_order_relaxed);
        pools[p]->rotor = 0;
        pools[p]->workers = 0;
    }
    pthread_mutex_init(&g_init_mu, NULL);
    pthread_mutex_init(&g_lane_mu, NULL);
    pthread_cond_init(&g_lane_cv, NULL);
    pthread_cond_init(&g_lane_done_cv, NULL);
    memset(g_lane_box, 0, sizeof(g_lane_box));
    g_lane_rotor = 0;
    g_lane_width = 0;
    g_depth = 0;
    g_on_lane = 0;
    g_low_until = 0.0;
    atomic_store_explicit(&g_pool_inited, 0, memory_order_relaxed);
}

static void pool_init_once(void) {
    if (atomic_load_explicit(&g_pool_inited, memory_order_acquire)) return;
    pthread_mutex_lock(&g_init_mu);
    if (!atomic_load_explicit(&g_pool_inited, memory_order_relaxed)) {
        static int atfork_registered = 0;  /* survives fork, like the handler */
        if (!atfork_registered) {
            atfork_registered = 1;
            pthread_atfork(NULL, NULL, mynah_threadpool_after_fork);
        }
        g_engine.thread_name = "mynah-pool";
        pool_init(&g_engine, mynah_lane_engine_width());
        atomic_store_explicit(&g_pool_inited, 1, memory_order_release);
    }
    pthread_mutex_unlock(&g_init_mu);
}

void mynah_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
    if (n <= 0) return;

    /* ================== THE REDIRECTION -- E5-21 ==================
     *
     * One branch, at the top of the dispatch primitive, keyed on a
     * thread-local tag. Everything a lane thread dispatches -- conv panels,
     * SGEMM slices, im2col, and every call site nobody has written yet --
     * lands on the lane team because of this line and not because anybody
     * remembered to route it. A convention at the call sites would have to
     * hold for code that does not exist; this holds by construction.
     * ============================================================== */
    pf_pool *const pool = g_on_lane ? &g_lane : &g_engine;
    const int nth = g_on_lane ? g_lane_width : mynah_lane_engine_width();

    if (nth <= 1 || n == 1) {
        PF_STAT(g_stat_serial);
        g_depth++;
        for (int i = 0; i < n; i++) fn(ctx, i);
        g_depth--;
        return;
    }
    /* The lane's threads are created by mynah_lane_split_prepare(); only the
     * engine pool is built lazily. */
    if (!g_on_lane) pool_init_once();

    pf_job *slot = NULL;
    pthread_mutex_lock(&pool->mu);
    if (pool->workers > 0) {
        for (int k = 0; k < PF_MAX_JOBS; k++) {
            if (!pool->jobs[k].live) { slot = &pool->jobs[k]; break; }
        }
    }
    if (slot == NULL) {
        pthread_mutex_unlock(&pool->mu);
        if (pool->workers > 0) PF_STAT(g_stat_inline); else PF_STAT(g_stat_serial);
        g_depth++;
        for (int i = 0; i < n; i++) fn(ctx, i);
        g_depth--;
        return;
    }
    slot->fn = fn;
    slot->ctx = ctx;
    slot->n = n;
    atomic_store_explicit(&slot->refs, 0, memory_order_relaxed);
    slot->live = 1;
    slot->low = (g_low_until > 0.0 && mynah_parallel_now_ms() < g_low_until);
    if (!slot->low) g_low_until = 0.0;
    atomic_store_explicit(&slot->next, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->gen, 1, memory_order_release);
    pthread_cond_broadcast(&pool->job_cv);
    pthread_mutex_unlock(&pool->mu);

    PF_STAT(g_stat_dispatch);
    pf_run(slot); /* the caller works too */

    /* Spin before parking here too: at the end of a region the helpers are
     * microseconds from finishing, and the submitter is the thread on the
     * critical path. */
    const int budget = spin_budget();
    int spun = 0;
    while (spun < budget &&
           atomic_load_explicit(&slot->refs, memory_order_acquire) > 0) {
        pf_cpu_relax();
        ++spun;
    }
    pthread_mutex_lock(&pool->mu);
    /* ctx belongs to the caller's frame: the slot cannot be reclaimed, and this
     * function cannot return, until every helper has left the region. */
    if (atomic_load_explicit(&slot->refs, memory_order_relaxed) > 0) {
        PF_STAT(g_stat_park);
        while (atomic_load_explicit(&slot->refs, memory_order_relaxed) > 0) {
            pthread_cond_wait(&pool->done_cv, &pool->mu);
        }
    } else {
        PF_STAT(g_stat_spin_win);
    }
    slot->live = 0;
    pthread_mutex_unlock(&pool->mu);
}

/* ======================================================================
 * Dispatch predicates
 * ====================================================================== */
/* E5-22. The VALUE, not a boolean: 65536 and 4096 are different servers and a
 * row saying ON would hide which one is running. */
static int probe_pool_spin(char *out, size_t capacity, const char **why) {
    static char text[384];
    const int budget = mynah_pool_spin_budget();
    long long parks = 0, wins = 0;
    mynah_pool_wait_stats(&parks, &wins);
    snprintf(out, capacity, "%d", budget);
    /* TRANSFERRED, and the word is load bearing: their sweep, not ours, and
     * their own note that the curve is non-monotonic on some hosts. */
    snprintf(text, sizeof text,
             "[predicate] spin source=%s, TRANSFERRED not measured here: "
             "STREAM p95 .999/.893/.808 at 256/4096/65536, csw/s 197k/38k/7.6k. "
             "Does not port. %lld parked, %lld spun",
             mynah_pool_spin_source(), parks, wins);
    *why = text;
    return 0;
}

/* E5-21. Off, refused, or the split that is actually in force. */
static int probe_pool_lane(char *out, size_t capacity, const char **why) {
    static char text[240];
    const int lane = mynah_lane_width();
    if (lane == 0) {
        snprintf(out, capacity, "OFF");
        snprintf(text, sizeof text,
                 "[predicate] mynah_lane_width(): the decoder runs inline on the "
                 "engine pool. The lane needs a pinned slice with at least %d "
                 "cpus on each side -- on a narrow lane it is SLOWER than inline "
                 "(6+2 measured 1.364 against 0.997 at 4+4)",
                 MYNAH_LANE_MIN_CPUS);
    } else {
        snprintf(out, capacity, "%d+%d", mynah_lane_engine_width(), lane);
        snprintf(text, sizeof text,
                 "[predicate] mynah_lane_width(): a private PINNED team of %d "
                 "threads with its own submit lock and a one-unit-per-slot "
                 "mailbox; engine pool narrowed to %d. Mailbox overruns: %lld "
                 "(must be 0)",
                 lane, mynah_lane_engine_width(), mynah_lane_overruns());
    }
    *why = text;
    return 0;
}

void mynah_threads_dispatch_probes(void) {
    mynah_dispatch_register_value_probe("pool.spin", probe_pool_spin);
    mynah_dispatch_register_value_probe("pool.decoder_lane", probe_pool_lane);
}
