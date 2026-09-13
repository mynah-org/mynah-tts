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
#include <signal.h>
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
/* THE DEFAULTS ARE THE MEASUREMENTS, not the intentions. On 32 Neoverse-V2,
 * paired interleaved, median of within-round ratios against the pool as it was:
 *
 *   precheck  -0.9% at 8 threads, -2.1% at 16, -2.3% at 32   -> ON
 *   narrow    +1.0% at 8 threads, +0.5% at 16, +0.4% at 32   -> OFF
 *   fastexit  +0.4% at 8 threads, -1.9% at 16, -4.9% at 32   -> OFF, see below
 *
 * NARROW IS OFF BECAUSE IT DID NOT PAY. The count said a 4-chunk region on a
 * 32-wide pool wakes 31 threads and that eight of them run no chunk, which is
 * true and is still true; capping admission simply does not buy wall back,
 * because a helper that leaves while chunks remain then has to be re-admitted
 * and the cap is one more load on the hot pick path. It is kept, switchable
 * and tested, so the next person does not have to rediscover the idea to find
 * out it was already measured. Initialised here rather than in the one-shot so
 * a worker that somehow reads them first sees the default, not zero. */
static int g_narrow = 0;
static int g_precheck = 1;
/* E9-P3. OFF, and the reason is not the measurement -- it is -4.9% at 32
 * threads, the largest single lever in this file. It is off because it is the
 * mirror image of the store-buffer trap this pool documents (trap 8: a pool
 * that peeks at a flag instead of taking the lock deadlocks on x86 and
 * silently does not on ARM), production is x86-64 AND ARM64, and the only box
 * this lane could measure on is ARM. The ordering is Dekker with seq_cst on
 * both halves and is correct by the standard on any conforming target; what is
 * missing is a run. Flip it after an x86 gate, not before. */
static int g_fastexit = 0;
/* E9-P3, OFF until it is measured on the box that will run it. It is the mirror
 * image of the store-buffer trap this file already documents, so it ships dark
 * and is turned on by a number, not by an argument. */
#define PF_FASTEXIT_DEFAULT 0
static int env_flag(const char *name, int fallback);

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
        /* Same one-shot, so the hot path has one guard rather than three. */
        /* The fallbacks are the statics above, so the default lives in exactly
         * one place and cannot drift between the two. */
        g_narrow = env_flag("MYNAH_POOL_NARROW", g_narrow);
        g_precheck = env_flag("MYNAH_POOL_PRECHECK", g_precheck);
        g_fastexit = env_flag("MYNAH_POOL_FASTEXIT", g_fastexit);
    }
    return g_spin;
}

int mynah_pool_spin_budget(void) { return spin_budget(); }
const char *mynah_pool_spin_source(void) { (void)spin_budget(); return g_spin_why; }

/* ---------------------------------------------------------------------------
 * E9-P2 knobs. Resolved once, in the same call that resolves the spin budget,
 * so a dispatch never reaches getenv(). Both default ON: each is a pure
 * reduction in threads woken for nothing, and neither can change a result
 * because chunk assignment is unaffected (threads.h states the invariant).
 * They are switchable so an arm can be run against a binary that is byte for
 * byte the one in production.
 * ------------------------------------------------------------------------- */
static int env_flag(const char *name, int fallback) {
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') return fallback;
    return (v[0] != '0');
}

int mynah_pool_precheck_enabled(void) { (void)spin_budget(); return g_precheck; }
int mynah_pool_narrow_enabled(void) { (void)spin_budget(); return g_narrow; }
int mynah_pool_fastexit_enabled(void) { (void)spin_budget(); return g_fastexit; }

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
    /* ATOMIC, and the reason is E9-P2's lock-free pre-check: a worker now reads
     * `n` and `live` WITHOUT the pool mutex, so leaving them plain ints would be
     * a data race in the language even though every write is still under the
     * lock. Relaxed loads on the scan path, ordinary stores under `mu`. */
    atomic_int n;
    /* Workers currently inside this region. Mutated only under the pool mutex;
     * ATOMIC so a submitter can spin on it outside the lock before parking. */
    atomic_int refs;
    atomic_int live;   /* slot published and not yet reclaimed  (pool->mu)    */
    int low;           /* submitted with a future deadline      (pool->mu)    */
    /* E9-P2. Most helpers allowed INSIDE at once: min(n, width) - 1. A region
     * of four chunks has no use for thirty-one threads. Read relaxed off the
     * lock by the pre-check; written once at publish. */
    atomic_int cap;
    /* E9-P1, written only while the meter is on. `joins` counts threads that
     * entered, `useful` those that ran at least one chunk; the difference is
     * the wake-up that bought nothing. */
    atomic_int joins;
    atomic_int useful;
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
    /* E9-P3. Submitters currently parked on done_cv. A helper that finishes the
     * last chunk of a region reads this to decide whether the mutex and the
     * broadcast are needed at all. SEQ_CST, and that is not decoration: see
     * pf_region_exit(). */
    _Atomic int     done_waiters;
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

/* ===========================================================================
 * THE POOL METER -- E9-P1.  See threads.h for why it exists.
 *
 * Three storage classes, chosen by who writes them:
 *   - per-thread padded slots for park/work wall, because up to 64 workers
 *     write those on every wait and a shared line would be the cost;
 *   - a per-call-site table of relaxed atomics, because only SUBMITTERS write
 *     it and a process has one or two of those (the scheduler thread, and the
 *     lane consumer when the lane is up);
 *   - plain relaxed globals for the aggregate, same reason.
 * Nothing here is touched at all when the meter is off.
 * ========================================================================= */

#define PF_METER_SLOTS 64   /* one per thread that ever waits or works */
#define PF_METER_SITES 64   /* distinct fn pointers; overflow is counted */

static long long pf_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000000LL + (long long)t.tv_nsec;
}

/* -1 = not resolved yet. Resolved once, then it is a plain load on a hot
 * branch that the predictor gets right every time. */
static int g_meter = -1;
static long long g_break_even_ns = 200000; /* 200 us; see threads.h */
static long long g_meter_t0;

static void pf_meter_install_hooks(void);

static int pf_meter_on(void) {
    if (g_meter < 0) {
        const char *env = getenv("MYNAH_POOL_METER");
        g_meter = (env != NULL && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        const char *be = getenv("MYNAH_POOL_BREAK_EVEN_US");
        if (be != NULL && be[0] != '\0') {
            const long v = atol(be);
            if (v > 0) g_break_even_ns = (long long)v * 1000LL;
        }
        if (g_meter) {
            g_meter_t0 = pf_now_ns();
            pf_meter_install_hooks();
        }
    }
    return g_meter;
}

int mynah_pool_meter_enabled(void) { return pf_meter_on(); }

/* One cache line each, and the alignment is on the TYPE rather than on the
 * array: aligning only the array start happens to work while the struct is
 * exactly 64 bytes and stops working silently the moment a counter is added.
 * _Alignas on the struct makes the compiler pad it to a multiple of 64 for us,
 * so the invariant survives an edit. These are written by every worker on every
 * wait; sharing a line between two workers would make the meter's own cost the
 * thing it measures. */
typedef struct {
    /* _Alignas on the first member raises the whole struct's alignment, which
     * makes the compiler round sizeof up to a multiple of it. */
    _Alignas(64) _Atomic long long park_ns;
    _Atomic long long work_ns;
    _Atomic long long chunks;
    _Atomic long long waits;
} pf_tslot;
static pf_tslot g_tslot[PF_METER_SLOTS];
_Static_assert(sizeof(pf_tslot) % 64 == 0,
               "pool meter: per-thread slots must not share a cache line");
static _Thread_local int g_meter_tid = -1;
static _Atomic int g_meter_tids;

static pf_tslot *pf_meter_slot(void) {
    if (g_meter_tid < 0) {
        const int id = atomic_fetch_add_explicit(&g_meter_tids, 1, memory_order_relaxed);
        g_meter_tid = id < PF_METER_SLOTS ? id : PF_METER_SLOTS - 1;
    }
    return &g_tslot[g_meter_tid];
}

typedef struct {
    _Atomic(void *) fn;
    _Atomic long long count, chunks, ns_region, ns_barrier;
    _Atomic long long req, entered, useful, n_sum;
    _Atomic long long buckets[MYNAH_POOL_METER_BUCKETS];
} pf_site;
static pf_site g_site[PF_METER_SITES];
static _Atomic long long g_site_overflow;

/* Open addressing on the function pointer. Claimed with a CAS, so two
 * submitters racing on a new site cannot both take a row. */
static pf_site *pf_meter_site(void *fn) {
    size_t h = ((size_t)fn >> 4) * 0x9E3779B97F4A7C15ULL;
    for (int probe = 0; probe < 8; ++probe) {
        pf_site *s = &g_site[(h + (size_t)probe) % PF_METER_SITES];
        void *cur = atomic_load_explicit(&s->fn, memory_order_relaxed);
        if (cur == fn) return s;
        if (cur == NULL) {
            void *expect = NULL;
            if (atomic_compare_exchange_strong_explicit(&s->fn, &expect, fn,
                                                        memory_order_relaxed,
                                                        memory_order_relaxed))
                return s;
            if (expect == fn) return s;
        }
    }
    atomic_fetch_add_explicit(&g_site_overflow, 1, memory_order_relaxed);
    return NULL;
}

static _Atomic long long g_m_chunks, g_m_req, g_m_entered, g_m_useful,
    g_m_region_ns, g_m_barrier_ns, g_m_under_be;
static _Atomic long long g_m_bucket[MYNAH_POOL_METER_BUCKETS];
static _Atomic long long g_m_capped, g_m_precheck_skips;

#define PF_ADD(c, v) atomic_fetch_add_explicit(&(c), (long long)(v), memory_order_relaxed)

static int pf_bucket_of(long long ns) {
    int b = 0;
    while (ns >= 2 && b < MYNAH_POOL_METER_BUCKETS - 1) { ns >>= 1; ++b; }
    return b;
}

/* Called by the submitter once a region is fully retired, so `entered` and
 * `useful` are stable: every helper has left, which the refs==0 wait
 * established. */
static void pf_meter_region(void *fn, int n, int req, long long chunks,
                            int entered, int useful,
                            long long region_ns, long long barrier_ns) {
    PF_ADD(g_m_chunks, chunks);
    PF_ADD(g_m_req, req);
    PF_ADD(g_m_entered, entered);
    PF_ADD(g_m_useful, useful);
    PF_ADD(g_m_region_ns, region_ns);
    PF_ADD(g_m_barrier_ns, barrier_ns);
    const int b = pf_bucket_of(region_ns);
    PF_ADD(g_m_bucket[b], 1);
    if (region_ns < g_break_even_ns) PF_ADD(g_m_under_be, 1);
    pf_site *s = pf_meter_site(fn);
    if (s == NULL) return;
    PF_ADD(s->count, 1);
    PF_ADD(s->chunks, chunks);
    PF_ADD(s->ns_region, region_ns);
    PF_ADD(s->ns_barrier, barrier_ns);
    PF_ADD(s->req, req);
    PF_ADD(s->entered, entered);
    PF_ADD(s->useful, useful);
    PF_ADD(s->n_sum, n);
    PF_ADD(s->buckets[b], 1);
}

void mynah_pool_narrow_stats(long long *capped, long long *precheck_skips) {
    if (capped) *capped = atomic_load(&g_m_capped);
    if (precheck_skips) *precheck_skips = atomic_load(&g_m_precheck_skips);
}

void mynah_pool_meter_read(mynah_pool_meter *out) {
    if (out == NULL) return;
    memset(out, 0, sizeof *out);
    out->dispatches = atomic_load(&g_stat_dispatch);
    out->serial = atomic_load(&g_stat_serial);
    out->inline_fallbacks = atomic_load(&g_stat_inline);
    out->chunks = atomic_load(&g_m_chunks);
    out->width_requested = atomic_load(&g_m_req);
    out->width_entered = atomic_load(&g_m_entered);
    out->width_useful = atomic_load(&g_m_useful);
    out->region_ns = atomic_load(&g_m_region_ns);
    out->barrier_ns = atomic_load(&g_m_barrier_ns);
    out->parks = atomic_load(&g_stat_park);
    out->spin_wins = atomic_load(&g_stat_spin_win);
    out->under_break_even = atomic_load(&g_m_under_be);
    out->break_even_ns = g_break_even_ns;
    out->meter_ns = g_meter_t0 != 0 ? pf_now_ns() - g_meter_t0 : 0;
    for (int i = 0; i < MYNAH_POOL_METER_BUCKETS; ++i)
        out->buckets[i] = atomic_load(&g_m_bucket[i]);
    const int slots = atomic_load(&g_meter_tids);
    for (int i = 0; i < slots && i < PF_METER_SLOTS; ++i) {
        const long long p = atomic_load(&g_tslot[i].park_ns);
        const long long w = atomic_load(&g_tslot[i].work_ns);
        out->park_ns += p;
        out->work_ns += w;
        if (p != 0 || w != 0) out->worker_threads++;
    }
    for (int i = 0; i < PF_METER_SITES; ++i)
        if (atomic_load(&g_site[i].fn) != NULL) out->sites++;
    out->site_overflow = atomic_load(&g_site_overflow);
}

void mynah_pool_meter_reset(void) {
    mynah_parallel_stats_reset();
    atomic_store(&g_m_chunks, 0); atomic_store(&g_m_req, 0);
    atomic_store(&g_m_entered, 0); atomic_store(&g_m_useful, 0);
    atomic_store(&g_m_region_ns, 0); atomic_store(&g_m_barrier_ns, 0);
    atomic_store(&g_m_under_be, 0);
    atomic_store(&g_m_capped, 0); atomic_store(&g_m_precheck_skips, 0);
    atomic_store(&g_site_overflow, 0);
    for (int i = 0; i < MYNAH_POOL_METER_BUCKETS; ++i) atomic_store(&g_m_bucket[i], 0);
    for (int i = 0; i < PF_METER_SLOTS; ++i) {
        atomic_store(&g_tslot[i].park_ns, 0);
        atomic_store(&g_tslot[i].work_ns, 0);
        atomic_store(&g_tslot[i].chunks, 0);
        atomic_store(&g_tslot[i].waits, 0);
    }
    for (int i = 0; i < PF_METER_SITES; ++i) {
        atomic_store(&g_site[i].fn, NULL);
        atomic_store(&g_site[i].count, 0); atomic_store(&g_site[i].chunks, 0);
        atomic_store(&g_site[i].ns_region, 0); atomic_store(&g_site[i].ns_barrier, 0);
        atomic_store(&g_site[i].req, 0); atomic_store(&g_site[i].entered, 0);
        atomic_store(&g_site[i].useful, 0); atomic_store(&g_site[i].n_sum, 0);
        for (int b = 0; b < MYNAH_POOL_METER_BUCKETS; ++b)
            atomic_store(&g_site[i].buckets[b], 0);
    }
    g_meter_t0 = pf_now_ns();
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

/* Returns the number of chunks THIS thread executed. The count is what makes
 * "entered but did no work" visible; it costs one register. */
static long long pf_run(pf_job *st) {
    long long did = 0;
    const int n = atomic_load_explicit(&st->n, memory_order_relaxed);
    g_depth++;
    for (;;) {
        const int i = atomic_fetch_add_explicit(&st->next, 1, memory_order_relaxed);
        if (i >= n) break;
        st->fn(st->ctx, i);
        ++did;
    }
    g_depth--;
    return did;
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
            if (!atomic_load_explicit(&j->live, memory_order_relaxed) ||
                (pass == 0 && j->low) || (pass == 1 && !j->low)) continue;
            if (atomic_load_explicit(&j->next, memory_order_relaxed) >=
                atomic_load_explicit(&j->n, memory_order_relaxed)) continue;
            const int refs = atomic_load_explicit(&j->refs, memory_order_relaxed);
            /* E9-P2 NARROWING. A performance cap, never a correctness one: the
             * submitter runs its own region to completion, so zero helpers is
             * always a correct outcome and holding one out can only cost time. */
            if (g_narrow && refs >= atomic_load_explicit(&j->cap, memory_order_relaxed)) {
                if (g_meter > 0) PF_ADD(g_m_capped, 1);
                continue;
            }
            if (best == NULL || refs < best_refs) { best = j; best_refs = refs; }
        }
        if (best) break;
    }
    if (best) {
        atomic_fetch_add_explicit(&best->refs, 1, memory_order_relaxed);
        if (g_meter > 0) atomic_fetch_add_explicit(&best->joins, 1, memory_order_relaxed);
        pool->rotor++;
    }
    return best;
}

/* E9-P2 THE PRE-CHECK.  Lock-free "is there anything for me?".
 *
 * THE ORDERING IS THE WHOLE THING, so it is stated rather than assumed. The
 * caller reads `gen` with ACQUIRE **before** calling this. The publisher writes
 * `live`, `n`, `cap` under `mu` and then bumps `gen` with RELEASE. So:
 *   - a region published before that acquire load is visible to this scan
 *     (release/acquire on `gen` carries the plain stores with it);
 *   - a region published after it leaves the caller's `seen` stale, and
 *     pf_wait_for_work() returns immediately on the generation mismatch.
 * Either way a wakeup cannot be lost. A false positive costs one needless
 * mutex; a false negative is impossible for a region we could have helped. */
static int pf_any_work(pf_pool *pool) {
    for (int k = 0; k < PF_MAX_JOBS; k++) {
        const pf_job *j = &pool->jobs[k];
        if (!atomic_load_explicit(&j->live, memory_order_relaxed)) continue;
        if (atomic_load_explicit(&j->next, memory_order_relaxed) >=
            atomic_load_explicit(&j->n, memory_order_relaxed)) continue;
        if (g_narrow &&
            atomic_load_explicit(&j->refs, memory_order_relaxed) >=
            atomic_load_explicit(&j->cap, memory_order_relaxed)) continue;
        return 1;
    }
    return 0;
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

/* E9-P3 THE REGION EXIT.
 *
 * WHAT IT REPLACES. Every helper left a region by taking the pool mutex,
 * decrementing refs under it, and broadcasting done_cv if it was the last out.
 * On 32 threads that is up to 31 acquisitions of one global mutex per region,
 * and the count said a region is 40-60 us and the barrier is a third of its
 * wall. The decrement itself is already atomic; the lock is there purely to
 * close a lost-wakeup window against a submitter that is about to park.
 *
 * WHY IT IS NOT THE TRAP. This is the same SHAPE as the "publish, then peek at
 * sleeping" pool that deadlocks on x86 and silently does not on ARM
 * (.work/linux-production.md trap 8), so the ordering is spelled out rather
 * than assumed. It is Dekker:
 *
 *   submitter:  store done_waiters++   then   load refs
 *   helper:     store refs--           then   load done_waiters
 *
 * With SEQ_CST on all four and a seq_cst fence between each pair, at least one
 * of the two loads must observe the other's store -- so either the submitter
 * sees refs already at zero and never parks, or the helper sees a waiter and
 * takes the mutex to broadcast. Both can happen; neither can be missed.
 * ACQUIRE/RELEASE IS NOT ENOUGH HERE, because store-then-load to two different
 * locations is exactly the pair that release/acquire does not order.
 *
 * The second window -- helper broadcasts between the submitter's check and its
 * cond_wait -- is closed the way it always was: the submitter re-tests refs
 * INSIDE the mutex in its wait loop, and the helper's broadcast is inside the
 * same mutex.
 *
 * Off (the default) the old path runs verbatim. */
static void pf_region_exit(pf_pool *pool, pf_job *job) {
    if (!g_fastexit) {
        pthread_mutex_lock(&pool->mu);
        if (atomic_fetch_sub_explicit(&job->refs, 1, memory_order_relaxed) == 1) {
            pthread_cond_broadcast(&pool->done_cv);
        }
        pthread_mutex_unlock(&pool->mu);
        return;
    }
    const int was = atomic_fetch_sub_explicit(&job->refs, 1, memory_order_seq_cst);
    if (was != 1) return;                     /* not the last one out */
    atomic_thread_fence(memory_order_seq_cst);
    if (atomic_load_explicit(&pool->done_waiters, memory_order_seq_cst) == 0) return;
    pthread_mutex_lock(&pool->mu);
    pthread_cond_broadcast(&pool->done_cv);
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
    (void)spin_budget();        /* resolves the E9-P2 knobs too, once, off the hot path */
    const int meter = pf_meter_on();
    pf_tslot *const ts = meter ? pf_meter_slot() : NULL;
    long long t_idle = meter ? pf_now_ns() : 0;
    for (;;) {
        /* E9-P2 PRE-CHECK. The acquire load of `gen` must come BEFORE the scan;
         * see pf_any_work() for why that single ordering is what makes skipping
         * the mutex safe. */
        unsigned seen;
        pf_job *job = NULL;
        if (g_precheck) {
            seen = atomic_load_explicit(&pool->gen, memory_order_acquire);
            if (!pf_any_work(pool)) {
                if (meter) PF_ADD(g_m_precheck_skips, 1);
                pf_wait_for_work(pool, seen);
                continue;
            }
        }
        pthread_mutex_lock(&pool->mu);
        seen = atomic_load_explicit(&pool->gen, memory_order_relaxed);
        job = pick_job(pool);
        pthread_mutex_unlock(&pool->mu);
        if (job == NULL) {
            /* `seen` was read under the lock, so a publish either happened
             * before the scan (and was picked) or bumps gen after: the
             * re-check inside pf_wait_for_work closes the window. */
            pf_wait_for_work(pool, seen);
            continue;
        }
        PF_STAT(g_stat_helper);
        long long t_start = 0;
        if (meter) {
            t_start = pf_now_ns();
            PF_ADD(ts->park_ns, t_start - t_idle);
            PF_ADD(ts->waits, 1);
        }
        const long long did = pf_run(job);
        if (meter) {
            t_idle = pf_now_ns();
            PF_ADD(ts->work_ns, t_idle - t_start);
            PF_ADD(ts->chunks, did);
            if (did > 0) atomic_fetch_add_explicit(&job->useful, 1, memory_order_relaxed);
        }
        pf_region_exit(pool, job);
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
        /* E9-P3. fork() keeps the calling thread only, so any submitter that
         * was parked on done_cv is gone; its +1 would otherwise survive into
         * the child and make every region exit there take the mutex to
         * broadcast at nobody. Harmless, and exactly the sort of inherited
         * bookkeeping the rest of this function exists to clear. */
        atomic_store_explicit(&pools[p]->done_waiters, 0, memory_order_relaxed);
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
    /* One cached int. Off, this is the meter's ENTIRE cost on this path: no
     * timer is read, no counter outside the four that already existed is
     * touched, and the branch is perfectly predicted. */
    const int meter = pf_meter_on();

    if (nth <= 1 || n == 1) {
        PF_STAT(g_stat_serial);
        long long t0 = meter ? pf_now_ns() : 0;
        g_depth++;
        for (int i = 0; i < n; i++) fn(ctx, i);
        g_depth--;
        if (meter) {
            /* A serial region still costs wall and still has a call site; it is
             * counted so that "dispatches per frame" cannot be gamed by a
             * caller that shrinks n until the pool stops being used. */
            pf_meter_region((void *)(size_t)fn, n, 1, n, 1, 1,
                            pf_now_ns() - t0, 0);
        }
        return;
    }
    /* The lane's threads are created by mynah_lane_split_prepare(); only the
     * engine pool is built lazily. */
    if (!g_on_lane) pool_init_once();

    pf_job *slot = NULL;
    pthread_mutex_lock(&pool->mu);
    if (pool->workers > 0) {
        for (int k = 0; k < PF_MAX_JOBS; k++) {
            if (!atomic_load_explicit(&pool->jobs[k].live, memory_order_relaxed)) {
                slot = &pool->jobs[k];
                break;
            }
        }
    }
    if (slot == NULL) {
        pthread_mutex_unlock(&pool->mu);
        if (pool->workers > 0) PF_STAT(g_stat_inline); else PF_STAT(g_stat_serial);
        long long t0 = meter ? pf_now_ns() : 0;
        g_depth++;
        for (int i = 0; i < n; i++) fn(ctx, i);
        g_depth--;
        if (meter)
            pf_meter_region((void *)(size_t)fn, n, 1, n, 1, 1, pf_now_ns() - t0, 0);
        return;
    }
    /* WIDTH REQUESTED. n chunks cannot occupy more than n threads however wide
     * the pool is; this is the honest denominator for "did the team show up?",
     * and with E9-P2 on it is also the admission cap. */
    const int req = n < nth ? n : nth;
    slot->fn = fn;
    slot->ctx = ctx;
    atomic_store_explicit(&slot->n, n, memory_order_relaxed);
    atomic_store_explicit(&slot->refs, 0, memory_order_relaxed);
    atomic_store_explicit(&slot->cap, req - 1 > 0 ? req - 1 : 1, memory_order_relaxed);
    atomic_store_explicit(&slot->joins, 0, memory_order_relaxed);
    atomic_store_explicit(&slot->useful, 0, memory_order_relaxed);
    atomic_store_explicit(&slot->live, 1, memory_order_relaxed);
    slot->low = (g_low_until > 0.0 && mynah_parallel_now_ms() < g_low_until);
    if (!slot->low) g_low_until = 0.0;
    atomic_store_explicit(&slot->next, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->gen, 1, memory_order_release);
    pthread_cond_broadcast(&pool->job_cv);
    pthread_mutex_unlock(&pool->mu);

    PF_STAT(g_stat_dispatch);
    const long long t_pub = meter ? pf_now_ns() : 0;
    const long long own = pf_run(slot); /* the caller works too */
    /* The BARRIER starts here: the submitter has no chunks left and the region
     * is not over. This subtraction is the number the whole item is about. */
    const long long t_own = meter ? pf_now_ns() : 0;

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
    /* E9-P3, the submitter's half of the Dekker pair in pf_region_exit(): the
     * waiter is announced BEFORE refs is read, with a seq_cst fence between, so
     * a helper that is about to finish cannot both miss this announcement and
     * be missed by the read below. */
    if (g_fastexit) {
        atomic_fetch_add_explicit(&pool->done_waiters, 1, memory_order_seq_cst);
        atomic_thread_fence(memory_order_seq_cst);
    }
    pthread_mutex_lock(&pool->mu);
    /* ctx belongs to the caller's frame: the slot cannot be reclaimed, and this
     * function cannot return, until every helper has left the region. */
    if (atomic_load_explicit(&slot->refs, memory_order_seq_cst) > 0) {
        PF_STAT(g_stat_park);
        while (atomic_load_explicit(&slot->refs, memory_order_relaxed) > 0) {
            pthread_cond_wait(&pool->done_cv, &pool->mu);
        }
    } else {
        PF_STAT(g_stat_spin_win);
    }
    if (g_fastexit)
        atomic_fetch_sub_explicit(&pool->done_waiters, 1, memory_order_seq_cst);
    /* Read the participation counters BEFORE the slot is released: once `live`
     * goes to 0 another submitter may take this slot and reset them. refs is 0
     * and the mutex is held, so every helper's increments are visible here. */
    int entered = 0, useful = 0;
    if (meter) {
        entered = atomic_load_explicit(&slot->joins, memory_order_relaxed);
        useful = atomic_load_explicit(&slot->useful, memory_order_relaxed);
    }
    atomic_store_explicit(&slot->live, 0, memory_order_relaxed);
    pthread_mutex_unlock(&pool->mu);
    if (meter) {
        const long long t_end = pf_now_ns();
        pf_meter_region((void *)(size_t)fn, n, req, n, entered + 1,
                        useful + (own > 0 ? 1 : 0), t_end - t_pub, t_end - t_own);
    }
}

/* ===========================================================================
 * The meter's report.
 *
 * It prints the two numbers the item asked for -- dispatches per unit of work,
 * and the fraction of wall that is barrier and park -- and then the per-call-
 * site table, because "fuse these regions" is a request that has to name a
 * function before anybody can act on it.
 * ========================================================================= */

static int pf_site_cmp(const void *a, const void *b) {
    const pf_site *x = *(const pf_site *const *)a;
    const pf_site *y = *(const pf_site *const *)b;
    const long long xa = atomic_load(&x->count), ya = atomic_load(&y->count);
    return ya > xa ? 1 : (ya < xa ? -1 : 0);
}

static double pf_pct(long long part, long long whole) {
    return whole > 0 ? 100.0 * (double)part / (double)whole : 0.0;
}

int mynah_pool_meter_report(void *out_file) {
    FILE *out = out_file != NULL ? (FILE *)out_file : stderr;
    mynah_pool_meter m;
    mynah_pool_meter_read(&m);
    if (!pf_meter_on()) {
        fprintf(out, "[pool-meter] OFF (set MYNAH_POOL_METER=1)\n");
        return 0;
    }
    long long capped = 0, skips = 0;
    mynah_pool_narrow_stats(&capped, &skips);
    const long long regions = m.dispatches + m.serial + m.inline_fallbacks;
    /* Thread-seconds is the honest denominator for park: a parked worker burns
     * one thread's wall, and there are `worker_threads` of them. */
    const long long thread_ns = m.meter_ns * (m.worker_threads > 0 ? m.worker_threads : 1);
    fprintf(out,
            "[pool-meter] regions=%lld (dispatched %lld, serial %lld, inline-fallback %lld)\n"
            "[pool-meter] chunks=%lld  width: requested %.2f / entered %.2f / useful %.2f\n"
            "[pool-meter]   -> %.1f%% of the threads that joined a region ran no chunk\n"
            "[pool-meter] region wall=%.3f ms total, mean %.1f us; barrier=%.3f ms (%.1f%% of region wall)\n"
            "[pool-meter] worker park=%.3f ms, work=%.3f ms over %lld worker threads (park %.1f%% of their wall)\n"
            "[pool-meter] waits: %lld parked, %lld absorbed by spin (budget %d, source %s)\n"
            "[pool-meter] narrowing: %lld admissions capped, %lld pre-checks skipped the mutex (narrow=%d precheck=%d fastexit=%d)\n"
            "[pool-meter] under break-even (%lld us): %lld of %lld regions = %.1f%%\n"
            "[pool-meter] meter wall %.3f ms; sites %lld (overflow %lld)\n",
            regions, m.dispatches, m.serial, m.inline_fallbacks,
            m.chunks,
            regions > 0 ? (double)m.width_requested / (double)regions : 0.0,
            regions > 0 ? (double)m.width_entered / (double)regions : 0.0,
            regions > 0 ? (double)m.width_useful / (double)regions : 0.0,
            m.width_entered > 0
                ? 100.0 * (double)(m.width_entered - m.width_useful) / (double)m.width_entered
                : 0.0,
            (double)m.region_ns / 1e6,
            regions > 0 ? (double)m.region_ns / (double)regions / 1e3 : 0.0,
            (double)m.barrier_ns / 1e6, pf_pct(m.barrier_ns, m.region_ns),
            (double)m.park_ns / 1e6, (double)m.work_ns / 1e6, m.worker_threads,
            pf_pct(m.park_ns, thread_ns),
            m.parks, m.spin_wins, mynah_pool_spin_budget(), mynah_pool_spin_source(),
            capped, skips, mynah_pool_narrow_enabled(), mynah_pool_precheck_enabled(),
            mynah_pool_fastexit_enabled(),
            m.break_even_ns / 1000, m.under_break_even, regions,
            pf_pct(m.under_break_even, regions),
            (double)m.meter_ns / 1e6, m.sites, m.site_overflow);

    fprintf(out, "[pool-meter] region duration histogram (log2 ns):\n");
    for (int b = 0; b < MYNAH_POOL_METER_BUCKETS; ++b) {
        if (m.buckets[b] == 0) continue;
        const double lo = (double)(1LL << b) / 1e3;
        fprintf(out, "[pool-meter]   %9.2f us .. %9.2f us : %10lld  %5.1f%%%s\n",
                lo, lo * 2.0, m.buckets[b], pf_pct(m.buckets[b], regions),
                (1LL << (b + 1)) <= m.break_even_ns ? "  (below break-even)" : "");
    }

    /* PIE DEFEATS A BARE POINTER. The absolute address in a running process is
     * the link-time address plus a load base that changes every run, so a raw
     * %p cannot be fed to nm. The DELTA between two addresses in the same module
     * survives relocation, so the report prints each site as an offset from
     * mynah_parallel_for(); offline the symbol is
     *     nm <binary> | grep mynah_parallel_for   ->  A
     *     addr2line -fe <binary> $(( A + delta ))
     * which names the file and line without the process being alive. */
    const void *const anchor = (const void *)(size_t)&mynah_parallel_for;
    fprintf(out, "[pool-meter] call-site anchor: mynah_parallel_for; "
                 "resolve with addr2line -fe BINARY $((nm_addr_of_anchor + delta))\n");
    const pf_site *order[PF_METER_SITES];
    int nsites = 0;
    for (int i = 0; i < PF_METER_SITES; ++i)
        if (atomic_load(&g_site[i].fn) != NULL) order[nsites++] = &g_site[i];
    qsort((void *)order, (size_t)nsites, sizeof order[0], pf_site_cmp);
    fprintf(out,
            "[pool-meter] per call site, by dispatch count "
            "(fn: resolve with nm/addr2line against this binary)\n"
            "[pool-meter]   %-18s %10s %8s %8s %9s %9s %8s %7s\n",
            "delta", "regions", "mean n", "mean us", "barrier%", "req w", "used w", "<be%");
    for (int i = 0; i < nsites; ++i) {
        const pf_site *t = order[i];
        const long long c = atomic_load(&t->count);
        if (c == 0) continue;
        long long under = 0;
        for (int b = 0; b < MYNAH_POOL_METER_BUCKETS; ++b)
            if ((1LL << (b + 1)) <= m.break_even_ns) under += atomic_load(&t->buckets[b]);
        fprintf(out,
                "[pool-meter]   %+-18lld %10lld %8.1f %8.1f %9.1f %9.2f %8.2f %7.1f\n",
                (long long)((const char *)atomic_load(&t->fn) - (const char *)anchor), c,
                (double)atomic_load(&t->n_sum) / (double)c,
                (double)atomic_load(&t->ns_region) / (double)c / 1e3,
                pf_pct(atomic_load(&t->ns_barrier), atomic_load(&t->ns_region)),
                (double)atomic_load(&t->req) / (double)c,
                (double)atomic_load(&t->useful) / (double)c,
                pf_pct(under, c));
    }
    return nsites;
}

int mynah_pool_meter_report_json(const char *path) {
    const char *p = path != NULL ? path : getenv("MYNAH_POOL_METER_JSON");
    if (p == NULL || p[0] == '\0') return 0;
    char resolved[1024];
    if (strstr(p, "%d") != NULL) snprintf(resolved, sizeof resolved, p, (int)getpid());
    else snprintf(resolved, sizeof resolved, "%s", p);
    FILE *f = fopen(resolved, "w");
    if (f == NULL) return -1;
    mynah_pool_meter m;
    mynah_pool_meter_read(&m);
    long long capped = 0, skips = 0;
    mynah_pool_narrow_stats(&capped, &skips);
    fprintf(f, "{\n  \"pid\": %d,\n  \"threads\": %d,\n  \"spin\": %d,\n"
               "  \"spin_source\": \"%s\",\n  \"narrow\": %d,\n  \"precheck\": %d,\n  \"fastexit\": %d,\n"
               "  \"dispatches\": %lld,\n  \"serial\": %lld,\n  \"inline_fallbacks\": %lld,\n"
               "  \"chunks\": %lld,\n  \"width_requested\": %lld,\n  \"width_entered\": %lld,\n"
               "  \"width_useful\": %lld,\n  \"region_ns\": %lld,\n  \"barrier_ns\": %lld,\n"
               "  \"park_ns\": %lld,\n  \"work_ns\": %lld,\n  \"parks\": %lld,\n"
               "  \"spin_wins\": %lld,\n  \"capped\": %lld,\n  \"precheck_skips\": %lld,\n"
               "  \"under_break_even\": %lld,\n  \"break_even_ns\": %lld,\n"
               "  \"meter_ns\": %lld,\n  \"worker_threads\": %lld,\n"
               "  \"sites\": %lld,\n  \"site_overflow\": %lld,\n"
               "  \"site_anchor\": \"mynah_parallel_for\",\n  \"buckets\": [",
            (int)getpid(), mynah_num_threads(), mynah_pool_spin_budget(),
            mynah_pool_spin_source(), mynah_pool_narrow_enabled(),
            mynah_pool_precheck_enabled(), mynah_pool_fastexit_enabled(),
            m.dispatches, m.serial, m.inline_fallbacks, m.chunks,
            m.width_requested, m.width_entered, m.width_useful,
            m.region_ns, m.barrier_ns, m.park_ns, m.work_ns, m.parks, m.spin_wins,
            capped, skips, m.under_break_even, m.break_even_ns, m.meter_ns,
            m.worker_threads, m.sites, m.site_overflow);
    for (int b = 0; b < MYNAH_POOL_METER_BUCKETS; ++b)
        fprintf(f, "%s%lld", b ? ", " : "", m.buckets[b]);
    fprintf(f, "],\n  \"sites_detail\": [");
    int first = 1;
    for (int i = 0; i < PF_METER_SITES; ++i) {
        void *fn = atomic_load(&g_site[i].fn);
        if (fn == NULL) continue;
        fprintf(f, "%s\n    {\"delta\": %lld, \"regions\": %lld, \"n_sum\": %lld, "
                   "\"ns_region\": %lld, \"ns_barrier\": %lld, \"req\": %lld, "
                   "\"entered\": %lld, \"useful\": %lld}",
                first ? "" : ",",
                (long long)((const char *)fn -
                            (const char *)(size_t)&mynah_parallel_for),
                atomic_load(&g_site[i].count), atomic_load(&g_site[i].n_sum),
                atomic_load(&g_site[i].ns_region), atomic_load(&g_site[i].ns_barrier),
                atomic_load(&g_site[i].req), atomic_load(&g_site[i].entered),
                atomic_load(&g_site[i].useful));
        first = 0;
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    return 0;
}

/* ---------------------------------------------------------------------------
 * Getting the report out of a process nobody will edit for us.
 *
 * atexit() covers the CLI. A prefork worker is killed with SIGTERM and would
 * otherwise take its numbers with it, so the meter -- AND ONLY THE METER, never
 * a production run -- also installs a SIGTERM handler that dumps and then
 * chains to whatever handler was already installed, so the server still shuts
 * down exactly the way it did before. Registered lazily on the first dispatch,
 * which is after main() has installed its own handlers; that ordering is what
 * makes the chaining possible.
 * ------------------------------------------------------------------------- */
static struct sigaction g_prev_term;
static _Atomic int g_term_chained;

static void pf_meter_dump(void) {
    /* Both, always: the JSON is what a harness parses and the table is what a
     * human reads, and a run that produced only one of them always turns out to
     * be the run you wanted the other from. */
    mynah_pool_meter_report_json(NULL);
    mynah_pool_meter_report(stderr);
}

static void pf_meter_on_term(int sig) {
    if (!atomic_exchange(&g_term_chained, 1)) pf_meter_dump();
    /* Hand the signal back to whoever owned it. Restoring first means a handler
     * that re-raises, or a default disposition, behaves as it always did. */
    sigaction(SIGTERM, &g_prev_term, NULL);
    raise(sig);
}

static _Atomic int g_hooks_done;

static void pf_meter_install_hooks(void) {
    /* The lane team is created before any dispatch, so two threads can reach
     * the one-shot in pf_meter_on(). Install exactly once. */
    if (atomic_exchange(&g_hooks_done, 1)) return;
    atexit(pf_meter_dump);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = pf_meter_on_term;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGTERM, &sa, &g_prev_term);
}

/* ===========================================================================
 * THE LITMUS -- E9-P2.
 *
 * The two levers change WHO shows up to a region. The invariant they must not
 * touch is that every chunk runs EXACTLY ONCE, and the failure mode they could
 * introduce is a lost wakeup -- a worker that skipped the mutex on the strength
 * of a lock-free scan and parked on a generation that had already moved. Both
 * are cheap to hammer and impossible to notice by reading, so they are a test
 * that runs in --self-test on every build and under both sanitizers, rather
 * than a paragraph asserting the ordering is fine.
 *
 * The shape is chosen to be hostile to exactly these bugs: regions of two to
 * five chunks on a pool far wider than that (so the admission cap bites on
 * nearly every one), submitted from several threads at once (so slots are
 * reused under contention), thousands of times (so a rare window is hit).
 * ========================================================================= */

typedef struct {
    _Atomic int *marks;
    int n;
} lt_ctx;

static void lt_task(void *ctx, int i) {
    lt_ctx *c = (lt_ctx *)ctx;
    atomic_fetch_add_explicit(&c->marks[i], 1, memory_order_relaxed);
}

/* Every region is verified before the next one starts, so a failure names the
 * width that broke rather than "something, somewhere, ran twice". */
static int lt_round(int n, _Atomic int *marks) {
    for (int i = 0; i < n; ++i) atomic_store_explicit(&marks[i], 0, memory_order_relaxed);
    lt_ctx c = { marks, n };
    mynah_parallel_for(n, lt_task, &c);
    for (int i = 0; i < n; ++i)
        if (atomic_load_explicit(&marks[i], memory_order_relaxed) != 1) return i;
    return -1;
}

#define LT_MAX_N 5
#define LT_ITERS 4000
/* The short form runs inside the dispatch REPORT, so it has to be a couple of
 * milliseconds, not a couple of hundred. It is the same shape at 1/40th the
 * count: enough to catch a lever that is simply wrong (400 rounds from five
 * submitters is two thousand narrowed regions), not enough to catch a
 * rare window. That is what the full form in mynah_threads_self_test() is for. */
#define LT_ITERS_SHORT 400

static _Atomic int g_lt_bad;
static _Atomic int g_lt_iters = LT_ITERS;

static void *lt_submitter(void *arg) {
    (void)arg;
    const int iters = atomic_load(&g_lt_iters);
    _Atomic int marks[LT_MAX_N];
    for (int it = 0; it < iters; ++it) {
        const int n = 2 + (it % (LT_MAX_N - 1));
        if (lt_round(n, marks) >= 0) atomic_store(&g_lt_bad, 1);
    }
    return NULL;
}

static int lt_run(int iters, char *error, size_t error_capacity) {
    atomic_store(&g_lt_bad, 0);
    atomic_store(&g_lt_iters, iters);
    /* Single submitter first: a failure here is the cap, not a slot race. */
    {
        _Atomic int marks[LT_MAX_N];
        for (int it = 0; it < iters; ++it) {
            const int n = 2 + (it % (LT_MAX_N - 1));
            const int bad = lt_round(n, marks);
            if (bad >= 0) {
                snprintf(error, error_capacity,
                         "pool: chunk %d of a %d-chunk region ran %d times "
                         "(narrow=%d precheck=%d spin=%d) -- a lever changed "
                         "WHAT is computed, not just who computes it",
                         bad, n, atomic_load(&marks[bad]),
                         mynah_pool_narrow_enabled(), mynah_pool_precheck_enabled(),
                         mynah_pool_spin_budget());
                return -1;
            }
        }
    }
    /* Then four submitters at once, which is what reuses job slots under
     * contention and is the only shape that can lose a wakeup. */
    pthread_t th[4];
    int made = 0;
    for (int i = 0; i < 4; ++i)
        if (pthread_create(&th[i], NULL, lt_submitter, NULL) == 0) ++made;
    for (int i = 0; i < made; ++i) pthread_join(th[i], NULL);
    if (atomic_load(&g_lt_bad)) {
        snprintf(error, error_capacity,
                 "pool: a chunk did not run exactly once under %d concurrent "
                 "submitters (narrow=%d precheck=%d)",
                 made, mynah_pool_narrow_enabled(), mynah_pool_precheck_enabled());
        return -1;
    }
    /* The meter must not be able to change the answer, so assert the one thing
     * that would mean it had: a region it saw is a region that ran. */
    if (mynah_pool_meter_enabled()) {
        mynah_pool_meter m;
        mynah_pool_meter_read(&m);
        if (m.width_useful > m.width_entered || m.width_entered < m.dispatches) {
            snprintf(error, error_capacity,
                     "pool meter: entered %lld useful %lld dispatches %lld -- "
                     "participation accounting is inconsistent",
                     m.width_entered, m.width_useful, m.dispatches);
            return -1;
        }
    }
    return 0;
}

int mynah_threads_self_test(char *error, size_t error_capacity) {
    return lt_run(LT_ITERS, error, error_capacity);
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

/* E9-P1/P2. One row for the three things that decide how wide a region runs:
 * whether anybody is counting, and whether the two narrowing levers are on. */
static int probe_pool_meter(char *out, size_t capacity, const char **why) {
    static char text[240];
    mynah_pool_meter m;
    mynah_pool_meter_read(&m);
    const long long regions = m.dispatches + m.serial + m.inline_fallbacks;
    /* `resolved` is 24 bytes and three booleans have to fit in it, so the two
     * levers are abbreviated rather than truncated -- a row that silently lost
     * its last field is exactly the "reads as ON" failure this table exists to
     * prevent. The reason line spells them out. */
    snprintf(out, capacity, "%s n%s p%s x%s",
             mynah_pool_meter_enabled() ? "ON" : "off",
             mynah_pool_narrow_enabled() ? "+" : "-",
             mynah_pool_precheck_enabled() ? "+" : "-",
             mynah_pool_fastexit_enabled() ? "+" : "-");
    if (mynah_pool_meter_enabled()) {
        snprintf(text, sizeof text,
                 "[predicate] MYNAH_POOL_METER: %lld regions, mean width "
                 "requested %.2f entered %.2f useful %.2f, barrier %.1f%% of "
                 "region wall, %.1f%% of regions below the %lld us break-even",
                 regions,
                 regions ? (double)m.width_requested / (double)regions : 0.0,
                 regions ? (double)m.width_entered / (double)regions : 0.0,
                 regions ? (double)m.width_useful / (double)regions : 0.0,
                 m.region_ns ? 100.0 * (double)m.barrier_ns / (double)m.region_ns : 0.0,
                 regions ? 100.0 * (double)m.under_break_even / (double)regions : 0.0,
                 m.break_even_ns / 1000);
    } else {
        snprintf(text, sizeof text,
                 "[predicate] MYNAH_POOL_METER unset: not counting dispatches "
                 "per frame or barrier wall. Set it to 1 (same binary) plus "
                 "MYNAH_POOL_METER_JSON=path. n/p/x = narrow, precheck, "
                 "fastexit (E9-P2/P3)");
    }
    *why = text;
    return 0;
}

/* E9-P2. The levers' invariant, checked in the report itself: a reader who sees
 * narrow=on next to a row that has never been tested has learned nothing. */
static int probe_pool_litmus(char *out, size_t capacity, const char **why) {
    static char text[300];
    char err[256];
    err[0] = '\0';
    const int rc = lt_run(LT_ITERS_SHORT, err, sizeof err);
    snprintf(out, capacity, "%s", rc == 0 ? "PASS" : "FAIL");
    if (rc == 0) {
        snprintf(text, sizeof text,
                 "[predicate] %d short rounds of 2..%d-chunk regions on a "
                 "%d-wide pool from 5 submitters: every chunk ran exactly once "
                 "with narrow=%d precheck=%d fastexit=%d spin=%d. The full form "
                 "runs in mynah_dispatch_self_test()",
                 LT_ITERS_SHORT, LT_MAX_N, mynah_num_threads(),
                 mynah_pool_narrow_enabled(), mynah_pool_precheck_enabled(),
                 mynah_pool_fastexit_enabled(), mynah_pool_spin_budget());
    } else {
        snprintf(text, sizeof text, "[predicate] POOL LITMUS FAILED: %s", err);
    }
    *why = text;
    return 0;
}

void mynah_threads_dispatch_probes(void) {
    mynah_dispatch_register_value_probe("pool.litmus", probe_pool_litmus);
    mynah_dispatch_register_value_probe("pool.spin", probe_pool_spin);
    mynah_dispatch_register_value_probe("pool.decoder_lane", probe_pool_lane);
    mynah_dispatch_register_value_probe("pool.meter", probe_pool_meter);
}
