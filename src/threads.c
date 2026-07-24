#include "threads.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

#define PF_MAX_THREADS 64

int mynah_num_threads(void) {
    static int nth = 0;
    if (nth == 0) {
        const char *env = getenv("MYNAH_THREADS");
        long n = env ? atol(env) : sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) n = 1;
        if (n > PF_MAX_THREADS) n = PF_MAX_THREADS;
        nth = (int)n;
    }
    return nth;
}

typedef struct {
    void (*fn)(void *, int);
    void *ctx;
    atomic_int next;
    int n;
} pf_state;

static void pf_run(pf_state *st) {
    for (;;) {
        const int i = atomic_fetch_add_explicit(&st->next, 1, memory_order_relaxed);
        if (i >= st->n) break;
        st->fn(st->ctx, i);
    }
}

/* Persistent worker pool: workers are created on the first parallel_for and
 * sleep on a condvar, so there is no pthread_create/join on the hot path.  One
 * dispatch at a time (g_pool_mu): if the pool is busy the caller runs the region
 * inline serial (no oversubscription).  Workers are detached and live until the
 * process exits, like the BLAS thread pools. */
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_job_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_job_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static pf_state *g_job;
static unsigned g_gen;
static int g_pending;
static int g_workers;
static pthread_once_t g_pool_once = PTHREAD_ONCE_INIT;

static void *pool_worker(void *arg) {
    (void)arg;
    unsigned seen = 0;
    pthread_mutex_lock(&g_job_mu);
    for (;;) {
        while (g_gen == seen) pthread_cond_wait(&g_job_cv, &g_job_mu);
        seen = g_gen;
        pf_state *job = g_job;
        pthread_mutex_unlock(&g_job_mu);
        pf_run(job);
        pthread_mutex_lock(&g_job_mu);
        if (--g_pending == 0) pthread_cond_signal(&g_done_cv);
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

/* Inside a parallel_for the cores belong to the workers: if each worker calls a
 * multi-threaded OpenBLAS we get catastrophic oversubscription.  Force BLAS to a
 * single thread for the region and restore on exit.  Accelerate (macOS) nests
 * via GCD and needs none of this.  Weak symbol as in qwen-tts: resolved only if
 * linked against OpenBLAS; an explicit OPENBLAS_NUM_THREADS always wins. */
#if defined(__GNUC__) && !defined(__APPLE__)
extern void openblas_set_num_threads(int) __attribute__((weak));
#endif

static void blas_set_threads(int n) {
#if defined(__GNUC__) && !defined(__APPLE__)
    if (getenv("OPENBLAS_NUM_THREADS")) return; /* explicit user choice */
    if (openblas_set_num_threads) openblas_set_num_threads(n > 0 ? n : 1);
#else
    (void)n;
#endif
}

void mynah_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
    if (n <= 0) return;
    const int nth = mynah_num_threads();
    if (nth <= 1 || n == 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    pthread_once(&g_pool_once, pool_init);

    pf_state st = {.fn = fn, .ctx = ctx, .n = n};
    atomic_init(&st.next, 0);
    const int active = n < nth ? n : nth;
    blas_set_threads(active <= 2 ? nth / active : 1);
    if (g_workers == 0 || pthread_mutex_trylock(&g_pool_mu) != 0) {
        pf_run(&st); /* pool absent or busy: inline */
        blas_set_threads(nth);
        return;
    }
    pthread_mutex_lock(&g_job_mu);
    g_job = &st;
    g_pending = g_workers;
    g_gen++;
    pthread_cond_broadcast(&g_job_cv);
    pthread_mutex_unlock(&g_job_mu);
    pf_run(&st); /* the caller works too */
    pthread_mutex_lock(&g_job_mu);
    while (g_pending > 0) pthread_cond_wait(&g_done_cv, &g_job_mu);
    pthread_mutex_unlock(&g_job_mu);
    pthread_mutex_unlock(&g_pool_mu);
    blas_set_threads(nth);
}
