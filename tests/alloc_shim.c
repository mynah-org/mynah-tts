/* alloc_shim.c — count allocations without the runtime knowing.
 *
 * WHY IT IS NOT IN THE RUNTIME.  Interposing an allocator changes allocator
 * behaviour: it adds a call layer, it can defeat tcache fast paths, and a
 * counter in libmynah_tts would ship in every binary for the benefit of one
 * measurement.  The runtime therefore contains no allocator hook at all.  This
 * object is PRELOADED, never linked, and it pushes the count into the runtime
 * through a registration function it looks up with dlsym — so the dependency
 * points from the tool to the library and never the other way.
 *
 * WHAT IT IS FOR.  .work/linux-build-and-dispatch.md recorded 2,786
 * allocations per request (2,608 malloc + 178 calloc) with an LD_PRELOAD
 * counter.  The number is interesting; the PROPERTY is decisive:
 *
 *     the count is CONSTANT across --max-steps
 *
 * which is the evidence that the autoregressive loop allocates nothing
 * (AGENTS.md rule 4).  A comment claiming that can rot in a week.  `make
 * alloc-constant-test` runs two synthesis lengths under this shim and fails if
 * the counts differ, so the property is checked instead of remembered.
 *
 * RECURSION.  dlsym() may allocate, and it is called from inside malloc.  A
 * static bootstrap arena serves whatever is requested before the real symbols
 * are resolved; free() on a bootstrap pointer is a no-op, which is correct
 * because the arena is never reused.
 *
 * THREADS.  The counters are relaxed atomics.  They are read once, at exit,
 * and an exact count matters more than the couple of nanoseconds: this object
 * is never in a measured build.
 */
#define _GNU_SOURCE
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <dlfcn.h>

static atomic_ullong g_malloc, g_calloc, g_realloc, g_free, g_posix;

/* ---- bootstrap arena, for allocations made while resolving dlsym ---- */
#define BOOT_BYTES (256u * 1024u)
static unsigned char g_boot[BOOT_BYTES];
static size_t        g_boot_used;

static int boot_owns(const void *p) {
    return (const unsigned char *)p >= g_boot &&
           (const unsigned char *)p < g_boot + BOOT_BYTES;
}

static void *boot_alloc(size_t n) {
    const size_t aligned = (n + 15u) & ~(size_t)15u;
    if (g_boot_used + aligned > BOOT_BYTES) return NULL;
    void *p = g_boot + g_boot_used;
    g_boot_used += aligned;
    return p;
}

typedef void *(*malloc_fn)(size_t);
typedef void *(*calloc_fn)(size_t, size_t);
typedef void *(*realloc_fn)(void *, size_t);
typedef void  (*free_fn)(void *);

static malloc_fn  real_malloc;
static calloc_fn  real_calloc;
static realloc_fn real_realloc;
static free_fn    real_free;
static int        resolving;

static void resolve(void) {
    if (real_malloc != NULL || resolving) return;
    resolving = 1;
#if defined(__APPLE__)
    /* NOT dlsym(RTLD_NEXT, ...).  Under DYLD_INSERT_LIBRARIES that returns this
     * shim's OWN function -- measured, all three of RTLD_NEXT, RTLD_DEFAULT and
     * the shim's address come back equal -- and since shim_malloc ends in
     * `return real_malloc(n)`, a tail call, the process spins at 100% CPU with
     * RSS pinned and never allocates again.  It looked like "too slow to
     * finish", which is what tests/census_parity.sh used to say; it was an
     * infinite loop, reproducible on `mynah-tts --version` with no model and no
     * Accelerate involved.
     *
     * Take the addresses directly instead.  A Mach-O interpose table rewrites
     * calls made from OTHER images; calls made from the image that declares the
     * table are not rewritten, so `malloc` here is libsystem's malloc. */
    real_malloc  = malloc;
    real_calloc  = calloc;
    real_realloc = realloc;
    real_free    = free;
#else
    real_malloc  = (malloc_fn)dlsym(RTLD_NEXT, "malloc");
    real_calloc  = (calloc_fn)dlsym(RTLD_NEXT, "calloc");
    real_realloc = (realloc_fn)dlsym(RTLD_NEXT, "realloc");
    real_free    = (free_fn)dlsym(RTLD_NEXT, "free");
#endif
    resolving = 0;
}

/* The total the census reads.  Only the allocating calls are summed: a free is
 * counted and reported, but "allocations per request" means allocations. */
unsigned long long mynah_alloc_shim_total(void) {
    return (unsigned long long)atomic_load_explicit(&g_malloc, memory_order_relaxed) +
           (unsigned long long)atomic_load_explicit(&g_calloc, memory_order_relaxed) +
           (unsigned long long)atomic_load_explicit(&g_realloc, memory_order_relaxed) +
           (unsigned long long)atomic_load_explicit(&g_posix, memory_order_relaxed);
}

/* ---- the interposed entry points ---- */

static void *shim_malloc(size_t n) {
    if (real_malloc == NULL) {
        resolve();
        if (real_malloc == NULL) return boot_alloc(n);
    }
    atomic_fetch_add_explicit(&g_malloc, 1ull, memory_order_relaxed);
    return real_malloc(n);
}

static void *shim_calloc(size_t a, size_t b) {
    if (real_calloc == NULL) {
        resolve();
        if (real_calloc == NULL) {
            void *p = boot_alloc(a * b);
            if (p != NULL) memset(p, 0, a * b);
            return p;
        }
    }
    atomic_fetch_add_explicit(&g_calloc, 1ull, memory_order_relaxed);
    return real_calloc(a, b);
}

static void *shim_realloc(void *p, size_t n) {
    if (real_realloc == NULL) resolve();
    if (boot_owns(p)) {
        /* Migrate out of the arena; the old block is simply abandoned. */
        void *q = real_malloc != NULL ? real_malloc(n) : boot_alloc(n);
        if (q != NULL && p != NULL) memcpy(q, p, n);
        atomic_fetch_add_explicit(&g_realloc, 1ull, memory_order_relaxed);
        return q;
    }
    if (real_realloc == NULL) return boot_alloc(n);
    atomic_fetch_add_explicit(&g_realloc, 1ull, memory_order_relaxed);
    return real_realloc(p, n);
}

static void shim_free(void *p) {
    if (p == NULL || boot_owns(p)) return;
    if (real_free == NULL) {
        resolve();
        if (real_free == NULL) return;
    }
    atomic_fetch_add_explicit(&g_free, 1ull, memory_order_relaxed);
    real_free(p);
}

#if defined(__APPLE__)
/* Mach-O: an interpose table rather than symbol shadowing. */
#define INTERPOSE(new, old)                                                    \
    __attribute__((used)) static struct {                                      \
        const void *n;                                                         \
        const void *o;                                                         \
    } _interpose_##old __attribute__((section("__DATA,__interpose"))) = {       \
        (const void *)(unsigned long)&new, (const void *)(unsigned long)&old}
/* posix_memalign is interposed here too, and it is not optional: the engine's
 * aligned scratch goes through it, and a shim that counts three quarters of the
 * allocations answers a question nobody asked. */
static int shim_posix_memalign(void **out, size_t align, size_t n) {
    static int (*real)(void **, size_t, size_t);
    if (real == NULL) real = posix_memalign;
    atomic_fetch_add_explicit(&g_posix, 1ull, memory_order_relaxed);
    return real(out, align, n);
}

INTERPOSE(shim_malloc, malloc);
INTERPOSE(shim_calloc, calloc);
INTERPOSE(shim_realloc, realloc);
INTERPOSE(shim_free, free);
INTERPOSE(shim_posix_memalign, posix_memalign);
#else
void *malloc(size_t n)              { return shim_malloc(n); }
void *calloc(size_t a, size_t b)    { return shim_calloc(a, b); }
void *realloc(void *p, size_t n)    { return shim_realloc(p, n); }
void  free(void *p)                 { shim_free(p); }
int posix_memalign(void **out, size_t align, size_t n) {
    static int (*real)(void **, size_t, size_t);
    if (real == NULL) real = (int (*)(void **, size_t, size_t))
                                 dlsym(RTLD_NEXT, "posix_memalign");
    if (real == NULL) { *out = boot_alloc(n); return *out != NULL ? 0 : 12; }
    atomic_fetch_add_explicit(&g_posix, 1ull, memory_order_relaxed);
    return real(out, align, n);
}
#endif

/* ---- registration and reporting ---- */

static void shim_report(void) {
    const char *path = getenv("MYNAH_ALLOC_COUNT_FILE");
    const unsigned long long m = atomic_load_explicit(&g_malloc, memory_order_relaxed);
    const unsigned long long c = atomic_load_explicit(&g_calloc, memory_order_relaxed);
    const unsigned long long r = atomic_load_explicit(&g_realloc, memory_order_relaxed);
    const unsigned long long pm = atomic_load_explicit(&g_posix, memory_order_relaxed);
    const unsigned long long f = atomic_load_explicit(&g_free, memory_order_relaxed);
    if (path != NULL && path[0] != 0) {
        FILE *out = fopen(path, "w");
        if (out != NULL) {
            fprintf(out,
                    "{\"total\": %llu, \"malloc\": %llu, \"calloc\": %llu, "
                    "\"realloc\": %llu, \"posix_memalign\": %llu, \"free\": %llu}\n",
                    mynah_alloc_shim_total(), m, c, r, pm, f);
            fclose(out);
        }
        return;
    }
    fprintf(stderr,
            "[ALLOC] total=%llu malloc=%llu calloc=%llu realloc=%llu "
            "posix_memalign=%llu free=%llu\n",
            mynah_alloc_shim_total(), m, c, r, pm, f);
}

__attribute__((constructor)) static void shim_init(void) {
    resolve();
    /* Hand the runtime a way to read the count, if this process has one. The
     * lookup is one-directional on purpose: libmynah_tts exports the setter
     * and knows nothing about this file. */
    void (*set)(unsigned long long (*)(void)) =
        (void (*)(unsigned long long (*)(void)))
            dlsym(RTLD_DEFAULT, "mynah_census_set_alloc_source");
    if (set != NULL) set(mynah_alloc_shim_total);
    atexit(shim_report);
}
