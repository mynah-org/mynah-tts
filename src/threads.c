#include "threads.h"

#include "dispatch.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
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
    int refs;          /* workers currently inside this region (g_job_mu)     */
    int live;          /* slot published and not yet reclaimed  (g_job_mu)    */
    int low;           /* submitted with a future deadline      (g_job_mu)    */
} pf_job;

static pthread_mutex_t g_job_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_job_cv = PTHREAD_COND_INITIALIZER;   /* work available */
static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;  /* refs hit zero  */
static pf_job g_jobs[PF_MAX_JOBS];
static unsigned g_gen;      /* bumped on every publish; the workers' wakeup key */
static unsigned g_rotor;    /* start index for the slot scan, for fairness      */
static int g_workers;

static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;
/* Atomic so the uncontended fast path in pool_init_once() can read it without
 * the init mutex and still be a well-defined read, not a benign-looking race. */
static _Atomic int g_pool_inited;

static _Atomic long long g_stat_dispatch, g_stat_serial, g_stat_inline,
    g_stat_helper;
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
}

static _Thread_local int g_depth;
static _Thread_local double g_low_until;

int mynah_parallel_active(void) { return g_depth > 0; }
void mynah_parallel_set_low_until(double until_ms) { g_low_until = until_ms; }
int mynah_pool_max_jobs(void) { return PF_MAX_JOBS; }

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

/* Caller holds g_job_mu.  Picks the region that most needs a hand: among slots
 * with chunks left, the one with the fewest workers already inside, starting
 * the scan at a rotating index so equal candidates alternate.  Low-priority
 * regions are only considered when nothing normal has work left. */
static pf_job *pick_job(void) {
    pf_job *best = NULL;
    int best_refs = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < PF_MAX_JOBS; k++) {
            pf_job *j = &g_jobs[(g_rotor + (unsigned)k) % PF_MAX_JOBS];
            if (!j->live || (pass == 0 && j->low) || (pass == 1 && !j->low)) continue;
            if (atomic_load_explicit(&j->next, memory_order_relaxed) >= j->n) continue;
            if (best == NULL || j->refs < best_refs) { best = j; best_refs = j->refs; }
        }
        if (best) break;
    }
    if (best) {
        best->refs++;
        g_rotor++;
    }
    return best;
}

static void *pool_worker(void *arg) {
    (void)arg;
    pthread_mutex_lock(&g_job_mu);
    for (;;) {
        const unsigned seen = g_gen;
        pf_job *job = pick_job();
        if (job == NULL) {
            /* g_gen was read under the same lock hold, so a publish either
             * happened before the scan (and was picked) or bumps g_gen after:
             * no wakeup can be lost here. */
            while (g_gen == seen) pthread_cond_wait(&g_job_cv, &g_job_mu);
            continue;
        }
        PF_STAT(g_stat_helper);
        pthread_mutex_unlock(&g_job_mu);
        pf_run(job);
        pthread_mutex_lock(&g_job_mu);
        if (--job->refs == 0) pthread_cond_broadcast(&g_done_cv);
    }
    return NULL; /* never reached */
}

static void pool_init(void) {
    const int nth = mynah_num_threads();
    for (int k = 0; k < nth - 1; k++) {
        pthread_t tid;
        if (pthread_create(&tid, NULL, pool_worker, NULL) == 0) {
            pthread_detach(tid);
            g_workers++;
        }
    }
}

/* fork() keeps the calling thread only.  Without this the child inherits
 * g_workers > 0 with no workers behind it, broadcasts to nobody and then waits
 * forever for refs to drop: an immediate deadlock, and the reason a prefork
 * server was impossible.  Resetting to "never started" makes the child build
 * its own pool on its next dispatch.  Nothing here allocates or takes a lock
 * that a dead thread might hold, which is what makes it legal from an atfork
 * child handler. */
void mynah_threadpool_after_fork(void) {
    pthread_mutex_init(&g_job_mu, NULL);
    pthread_mutex_init(&g_init_mu, NULL);
    pthread_cond_init(&g_job_cv, NULL);
    pthread_cond_init(&g_done_cv, NULL);
    memset(g_jobs, 0, sizeof g_jobs);
    g_gen = 0;
    g_rotor = 0;
    g_workers = 0;
    g_depth = 0;
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
        pool_init();
        atomic_store_explicit(&g_pool_inited, 1, memory_order_release);
    }
    pthread_mutex_unlock(&g_init_mu);
}

/* Inside a parallel_for the cores belong to the workers: if each worker calls a
 * multi-threaded OpenBLAS we get catastrophic oversubscription.  Force BLAS to a
 * single thread for the region and restore on exit.  Accelerate (macOS) nests
 * via GCD and needs none of this.  Weak symbol as in qwen-tts: resolved only if
 * linked against OpenBLAS; an explicit OPENBLAS_NUM_THREADS always wins.
 *
 * The count is process-global, so "set on entry, restore on exit" was wrong as
 * soon as two regions overlapped: the first one to finish restored the full
 * count under the second, which then oversubscribed.  Region entries are
 * counted instead: the first sets, an overlapping one can only lower, and only
 * the last to leave restores the base. */
#if defined(__GNUC__) && !defined(__APPLE__)
extern void openblas_set_num_threads(int) __attribute__((weak));
#define PF_HAVE_BLAS_KNOB 1
#endif

#ifdef PF_HAVE_BLAS_KNOB
static pthread_mutex_t g_blas_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_blas_depth;
static int g_blas_cur = -1;
static int g_blas_base;

static void blas_apply(int n) {
    if (getenv("OPENBLAS_NUM_THREADS")) return; /* explicit user choice */
    if (n < 1) n = 1;
    if (n == g_blas_cur) return;
    if (openblas_set_num_threads) openblas_set_num_threads(n);
    g_blas_cur = n;
}

static void blas_region_enter(int want) {
    pthread_mutex_lock(&g_blas_mu);
    if (g_blas_base == 0) g_blas_base = mynah_num_threads();
    if (g_blas_depth++ == 0 || want < g_blas_cur) blas_apply(want);
    pthread_mutex_unlock(&g_blas_mu);
}

static void blas_region_leave(void) {
    pthread_mutex_lock(&g_blas_mu);
    if (--g_blas_depth <= 0) {
        g_blas_depth = 0;
        blas_apply(g_blas_base ? g_blas_base : mynah_num_threads());
    }
    pthread_mutex_unlock(&g_blas_mu);
}

void mynah_blas_set_threads(int n) {
    pthread_mutex_lock(&g_blas_mu);
    g_blas_base = n > 0 ? n : 1;
    if (g_blas_depth == 0) blas_apply(g_blas_base);
    pthread_mutex_unlock(&g_blas_mu);
}
#else
static void blas_region_enter(int want) { (void)want; }
static void blas_region_leave(void) {}
void mynah_blas_set_threads(int n) { (void)n; }
#endif

/* Does this process actually own the vendor BLAS's thread count?
 *
 * Three things have to line up, and mynah_blas_set_threads() above is a silent
 * no-op when any of them does not: the build must be a non-Apple GNU-C one
 * (Accelerate nests through GCD and needs no clamping), the weak
 * openblas_set_num_threads must have resolved, and OPENBLAS_NUM_THREADS must
 * be unset -- an explicit user choice always wins over ours.  Exported so the
 * dispatch report reads the same three conditions the setter does instead of
 * restating them, because the failure this hides is oversubscription: every
 * pool worker calling a threaded BLAS. */
int mynah_blas_owned(void) {
#ifdef PF_HAVE_BLAS_KNOB
    if (getenv("OPENBLAS_NUM_THREADS") != NULL) return 0;
    return openblas_set_num_threads != NULL;
#else
    return 0;
#endif
}

void mynah_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
    if (n <= 0) return;
    const int nth = mynah_num_threads();
    if (nth <= 1 || n == 1) {
        PF_STAT(g_stat_serial);
        g_depth++;
        for (int i = 0; i < n; i++) fn(ctx, i);
        g_depth--;
        return;
    }
    pool_init_once();

    const int active = n < nth ? n : nth;
    const int blas_want = active <= 2 ? nth / active : 1;

    pf_job *slot = NULL;
    pthread_mutex_lock(&g_job_mu);
    if (g_workers > 0) {
        for (int k = 0; k < PF_MAX_JOBS; k++) {
            if (!g_jobs[k].live) { slot = &g_jobs[k]; break; }
        }
    }
    if (slot == NULL) {
        pthread_mutex_unlock(&g_job_mu);
        if (g_workers > 0) PF_STAT(g_stat_inline); else PF_STAT(g_stat_serial);
        blas_region_enter(blas_want);
        g_depth++;
        for (int i = 0; i < n; i++) fn(ctx, i);
        g_depth--;
        blas_region_leave();
        return;
    }
    slot->fn = fn;
    slot->ctx = ctx;
    slot->n = n;
    slot->refs = 0;
    slot->live = 1;
    slot->low = (g_low_until > 0.0 && mynah_parallel_now_ms() < g_low_until);
    if (!slot->low) g_low_until = 0.0;
    atomic_store_explicit(&slot->next, 0, memory_order_relaxed);
    g_gen++;
    pthread_cond_broadcast(&g_job_cv);
    pthread_mutex_unlock(&g_job_mu);

    PF_STAT(g_stat_dispatch);
    blas_region_enter(blas_want);
    pf_run(slot); /* the caller works too */

    pthread_mutex_lock(&g_job_mu);
    /* ctx belongs to the caller's frame: the slot cannot be reclaimed, and this
     * function cannot return, until every helper has left the region. */
    while (slot->refs > 0) pthread_cond_wait(&g_done_cv, &g_job_mu);
    slot->live = 0;
    pthread_mutex_unlock(&g_job_mu);
    /* Only now: helpers were still running tasks that may call BLAS, and
     * restoring the count from under them is exactly the clobber this counter
     * exists to prevent. */
    blas_region_leave();
}

/* ======================================================================
 * Dispatch predicate
 * ====================================================================== */
static int probe_blas_owned(const char **why) {
    const int on = mynah_blas_owned();
    if (why != NULL) {
#ifdef PF_HAVE_BLAS_KNOB
        *why = on ? "[predicate] mynah_blas_owned(): the weak "
                    "openblas_set_num_threads resolved and OPENBLAS_NUM_THREADS "
                    "is unset, so src/threads.c holds BLAS at one thread inside "
                    "a parallel region"
                  : "[predicate] mynah_blas_owned(): either OPENBLAS_NUM_THREADS "
                    "is set (an explicit choice that always wins) or the weak "
                    "symbol did not resolve. The vendor BLAS keeps its own team "
                    "and can nest under the pool";
#else
        *why = "[predicate] mynah_blas_owned(): not applicable on this build -- "
               "Accelerate nests through GCD, so there is nothing to clamp";
#endif
    }
    return on;
}

void mynah_threads_dispatch_probes(void) {
    mynah_dispatch_register_probe("blas.threads_owned", probe_blas_owned);
}
