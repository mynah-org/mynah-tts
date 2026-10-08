/* Persistent team for the serving loop's per-row host loops. See hostpool.h.
 *
 * One region at a time (the scheduler thread is the only submitter; a second
 * submitter, or a region started from inside a region, runs inline).
 *
 * A region is one 64-bit word -- generation, next unclaimed row, row count,
 * chunk -- plus `fn`/`ud` and a `done` row counter. Rows are claimed by a
 * compare-and-swap on the word, so a claim succeeds only against the region
 * that is open right now; `fn` and `ud` are read only after a successful
 * claim, and stay valid until the submitter has seen `done` reach the row
 * count, which cannot happen before that claim's rows are finished. A worker
 * that wakes late, or not at all, claims nothing and holds nobody up: the
 * submitter claims too and waits only for rows that were actually claimed.
 *
 * Workers learn of a new region from `gen` (bumped at publication); they spin
 * on it for a short budget after each region, then park on a condition
 * variable. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* pthread_setname_np */
#endif
#include "hostpool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define HP_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define HP_RELAX() __asm__ __volatile__("yield")
#else
#define HP_RELAX() ((void)0)
#endif

#define HP_MAX_THREADS 16
#define HP_MAX_ROWS 32767u   /* next + chunk must fit 16 bits */

/* word: gen (16) | next (16) | n (16) | chunk (16) */
#define HP_W(gen, next, n, chunk) \
    (((uint64_t)((gen) & 0xffffu) << 48) | ((uint64_t)(next) << 32) | \
     ((uint64_t)(n) << 16) | (uint64_t)(chunk))
#define HP_NEXT(w) ((size_t)(((w) >> 32) & 0xffffu))
#define HP_N(w) ((size_t)(((w) >> 16) & 0xffffu))
#define HP_CHUNK(w) ((size_t)((w) & 0xffffu))

static struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int nworkers;                /* threads - 1 */
    _Atomic unsigned long gen;   /* region generation, bumped at publication */
    _Atomic int busy;            /* a region is being submitted (one submitter) */
    _Atomic uint64_t word;       /* the open region's claim word */
    _Atomic size_t done;         /* rows of the open region finished */
    void (*fn)(void *, size_t, size_t);
    void *ud;
    int parked;                  /* workers waiting on cv (under mu) */
    unsigned spin_ns;
    _Atomic unsigned long long parallel, inline_runs;
} g_pool = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
};

static int g_auto_threads = 0;
static int g_threads = -1;          /* resolved count; -1 = not yet */
static size_t g_min_rows = 64u;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static _Thread_local int tl_in_region;

static unsigned long long hp_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
}

/* Claim and run chunks of the open region until none is left. */
static void hp_drain(void) {
    for (;;) {
        uint64_t w = atomic_load_explicit(&g_pool.word, memory_order_acquire);
        const size_t begin = HP_NEXT(w), n = HP_N(w), chunk = HP_CHUNK(w);
        if (begin >= n) return;
        const uint64_t claimed = w + ((uint64_t)chunk << 32);
        if (!atomic_compare_exchange_weak_explicit(&g_pool.word, &w, claimed,
                                                   memory_order_acq_rel,
                                                   memory_order_acquire))
            continue;
        const size_t end = begin + chunk < n ? begin + chunk : n;
        g_pool.fn(g_pool.ud, begin, end);
        atomic_fetch_add_explicit(&g_pool.done, end - begin, memory_order_release);
    }
}

static void *hp_worker(void *arg) {
    (void)arg;
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "mynah-hostpool");
#endif
    tl_in_region = 1;   /* a body never submits a nested region */
    unsigned long seen = atomic_load_explicit(&g_pool.gen, memory_order_acquire);
    for (;;) {
        unsigned long gen = atomic_load_explicit(&g_pool.gen, memory_order_acquire);
        if (gen == seen) {
            /* Spin for the next region, then park. */
            const unsigned long long until = hp_now_ns() + g_pool.spin_ns;
            unsigned k = 0;
            while ((gen = atomic_load_explicit(&g_pool.gen, memory_order_acquire)) == seen) {
                HP_RELAX();
                if ((++k & 63u) == 0u && hp_now_ns() >= until) break;
            }
            if (gen == seen) {
                pthread_mutex_lock(&g_pool.mu);
                g_pool.parked++;
                while ((gen = atomic_load_explicit(&g_pool.gen, memory_order_acquire)) == seen)
                    pthread_cond_wait(&g_pool.cv, &g_pool.mu);
                g_pool.parked--;
                pthread_mutex_unlock(&g_pool.mu);
            }
        }
        seen = gen;
        hp_drain();
    }
    return NULL;
}

static void hp_resolve(void) {
    int threads = 1;   /* unset: off */
    const char *e = getenv("MYNAH_SERVE_HOST_THREADS");
    if (e != NULL && strcmp(e, "auto") == 0) {
        threads = g_auto_threads;
    } else if (e != NULL && e[0] != '\0') {
        char *end = NULL;
        const long v = strtol(e, &end, 10);
        if (end != e && *end == '\0' && v >= 0 && v <= HP_MAX_THREADS) {
            threads = (int)v;
        } else {
            fprintf(stderr, "ignoring MYNAH_SERVE_HOST_THREADS=%s (want 0..%d or auto)\n",
                    e, HP_MAX_THREADS);
        }
    }
    if (threads < 1) threads = 1;
    if (threads > HP_MAX_THREADS) threads = HP_MAX_THREADS;
    const char *m = getenv("MYNAH_SERVE_HOST_MIN_ROWS");
    if (m != NULL && m[0] != '\0') {
        char *end = NULL;
        const unsigned long v = strtoul(m, &end, 10);
        if (end != m && *end == '\0' && v >= 2ul && v <= 1000000ul) g_min_rows = (size_t)v;
    }
    g_pool.spin_ns = 50000u;   /* 50 us: regions of one iteration come in bursts */
    const char *s = getenv("MYNAH_SERVE_HOST_SPIN_US");
    if (s != NULL && s[0] != '\0') {
        char *end = NULL;
        const unsigned long v = strtoul(s, &end, 10);
        if (end != s && *end == '\0' && v <= 10000ul) g_pool.spin_ns = (unsigned)v * 1000u;
    }
    g_pool.nworkers = 0;
    for (int i = 0; i < threads - 1; ++i) {
        pthread_t t;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        const int rc = pthread_create(&t, &attr, hp_worker, NULL);
        pthread_attr_destroy(&attr);
        if (rc != 0) break;
        g_pool.nworkers++;
    }
    g_threads = g_pool.nworkers + 1;
}

int mynah_hostpool_threads(void) {
    pthread_once(&g_once, hp_resolve);
    return g_threads;
}

void mynah_hostpool_set_auto(int threads) {
    g_auto_threads = threads;
}

size_t mynah_hostpool_min_rows(void) {
    pthread_once(&g_once, hp_resolve);
    return g_min_rows;
}

void mynah_hostpool_run(size_t n, void (*fn)(void *ud, size_t begin, size_t end),
                        void *ud) {
    if (n == 0u) return;
    const int threads = mynah_hostpool_threads();
    int expected = 0;
    if (threads <= 1 || n < g_min_rows || n > HP_MAX_ROWS || tl_in_region ||
        !atomic_compare_exchange_strong(&g_pool.busy, &expected, 1)) {
        fn(ud, 0u, n);
        if (threads > 1) atomic_fetch_add_explicit(&g_pool.inline_runs, 1ull,
                                                   memory_order_relaxed);
        return;
    }
    /* ~4 chunks per thread: late wakers still find work, and a chunk is long
     * enough that the claim is noise. */
    size_t chunk = n / ((size_t)threads * 4u);
    if (chunk < 8u) chunk = 8u;
    const unsigned long gen = atomic_load_explicit(&g_pool.gen, memory_order_relaxed) + 1ul;
    g_pool.fn = fn;
    g_pool.ud = ud;
    atomic_store_explicit(&g_pool.done, 0u, memory_order_relaxed);
    /* Publish: the release orders fn/ud/done before the claimable word. */
    atomic_store_explicit(&g_pool.word, HP_W(gen, 0u, n, chunk), memory_order_release);
    atomic_store_explicit(&g_pool.gen, gen, memory_order_release);
    pthread_mutex_lock(&g_pool.mu);
    const int parked = g_pool.parked;
    pthread_mutex_unlock(&g_pool.mu);
    if (parked > 0) pthread_cond_broadcast(&g_pool.cv);
    tl_in_region = 1;
    hp_drain();
    tl_in_region = 0;
    /* Every row is claimed; wait for the rows claimed by workers. */
    while (atomic_load_explicit(&g_pool.done, memory_order_acquire) != n) HP_RELAX();
    atomic_fetch_add_explicit(&g_pool.parallel, 1ull, memory_order_relaxed);
    atomic_store_explicit(&g_pool.busy, 0, memory_order_release);
}

void mynah_hostpool_stats(unsigned long long *parallel, unsigned long long *inline_runs) {
    if (parallel != NULL)
        *parallel = atomic_load_explicit(&g_pool.parallel, memory_order_relaxed);
    if (inline_runs != NULL)
        *inline_runs = atomic_load_explicit(&g_pool.inline_runs, memory_order_relaxed);
}
