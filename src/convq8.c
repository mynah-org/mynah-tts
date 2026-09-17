/* INT8 tap GEMM for the SEANet codec conv stack.  See src/convq8.h. */

#include "convq8.h"

#include <math.h>
#include <stdarg.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dispatch.h"
#include "qmat.h"
#include "threads.h"

/* Rows one strip of the kernel loop covers.  It bounds the task's stack, and
 * it is NOT the row block the pool plans: a task walks its block in strips, so
 * the two can be chosen independently. */
#define CQ8_ROW_STRIP 64u
/* Activation vectors per kernel call.  mynah_qmat_dots_max_batch() is the
 * authority; this is the compile-time bound on the stack array and the code
 * refuses rather than truncates if they ever disagree. */
#define CQ8_MAX_BATCH 4u
/* Below this many MACs the quantization pass costs more than it saves, and
 * the f32 path is both faster and exact.  The same shape of argument as
 * sgemm.c's SG_MIN_BLOCKED_WORK, two orders larger because there is a whole
 * activation pass to amortise here. */
#define CQ8_MIN_WORK (1u << 18)
#define CQ8_PARALLEL_MIN_WORK (1u << 17)
/* The rowsum overflow argument in qmat.c assumes k <= 8192; mynah_qmat_pack_q8
 * enforces it and refuses, and this is the same bound stated where the shape
 * gate can see it. */
#define CQ8_K_MAX 8192u

/* ------------------------------------------------------------- counters */

typedef struct {
    atomic_ullong calls, ran, refused_host, refused_shape, refused_pack;
    atomic_ullong refused_scratch, packed_tensors, packed_bytes;
} cq8_counters;

static cq8_counters g_cq8;

static void cq8_bump(atomic_ullong *c) {
    atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
}
static unsigned long long cq8_read(const atomic_ullong *c) {
    return atomic_load_explicit((atomic_ullong *)c, memory_order_relaxed);
}

void mynah_convq8_stats_get(mynah_convq8_stats *out) {
    if (out == NULL) return;
    out->calls = cq8_read(&g_cq8.calls);
    out->ran = cq8_read(&g_cq8.ran);
    out->refused_host = cq8_read(&g_cq8.refused_host);
    out->refused_shape = cq8_read(&g_cq8.refused_shape);
    out->refused_pack = cq8_read(&g_cq8.refused_pack);
    out->refused_scratch = cq8_read(&g_cq8.refused_scratch);
    out->packed_tensors = cq8_read(&g_cq8.packed_tensors);
    out->packed_bytes = cq8_read(&g_cq8.packed_bytes);
}

/* --------------------------------------------------------- the host gate
 *
 * int8 is not unconditionally faster than f32.  It is faster where the
 * hardware has a dot-product unit -- SDOT on ARM, VPDPBUSD on x86 -- and it is
 * SLOWER where the fallback is a widen-then-madd pair or a scalar loop,
 * because an f32 FMA already does eight lanes and the int8 path then pays a
 * quantization pass for nothing.  qmat names the kernel this host resolved to
 * and that name is the gate; "avx2" (no VNNI) is refused on purpose and not
 * on measurement, because it cannot be measured from here and a default that
 * might be a regression on half of x86 is not a default.
 *
 * MYNAH_CODEC_CONV_Q8=1 forces it on for exactly that measurement; =0 forces
 * it off.  Resolved once, so nothing in the decode loop calls getenv(). */
static int g_cq8_force = -1;          /* test hook: -1 auto, 0 off, 1 on */
static size_t g_cq8_force_blocks = 0u;/* test hook: 0 = plan it normally    */

static int cq8_kernel_ok(const char *name) {
    return strcmp(name, "neon-sdot") == 0 || strcmp(name, "avx512vnni") == 0 ||
           strcmp(name, "avxvnni") == 0;
}

static int cq8_env(void) {
    static atomic_int cached = -2;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v != -2) return v;
    const char *e = getenv("MYNAH_CODEC_CONV_Q8");
    v = (e == NULL || e[0] == '\0') ? -1 : (e[0] == '0' ? 0 : 1);
    atomic_store_explicit(&cached, v, memory_order_relaxed);
    return v;
}

int mynah_convq8_host_ok(const char **why) {
    static char text[260];
    const char *kwhy = NULL;
    const char *kernel = mynah_qmat_int8_kernel(&kwhy);
    const int forced = (g_cq8_force >= 0) ? g_cq8_force : cq8_env();
    const int natural = cq8_kernel_ok(kernel);
    const int on = (forced >= 0) ? forced : natural;
    if (why != NULL) {
        snprintf(text, sizeof text,
                 "[predicate] src/convq8.c mynah_convq8_host_ok(): int8 kernel "
                 "is \"%s\", which %s a dot-product unit%s. %s",
                 kernel, natural ? "has" : "does NOT have",
                 forced >= 0 ? (forced ? ", but it was forced ON"
                                       : ", and it was forced OFF")
                             : "",
                 on ? "The conv stack may run int8 where the group spec asks "
                      "for it"
                    : "Every conv tap GEMM stays f32: an int8 dot without the "
                      "unit is slower than an f32 FMA and the quantization "
                      "pass is pure loss");
        *why = text;
    }
    return on;
}

int mynah_convq8_force(int mode) {
    const int before = g_cq8_force;
    g_cq8_force = (mode < 0) ? -1 : (mode != 0);
    return before;
}

/* ------------------------------------------------------- the weight memo
 *
 * Keyed on the WEIGHT POINTER, exactly as sea_taps_all() in src/seanet.c is
 * keyed, and for the same two reasons: this layer is handed resolved pointers
 * and never a name, and a weight is mmapped and immutable for the life of the
 * process while the state that uses it is per request.  A per-state copy would
 * cost the pack again per worker per slot.
 *
 * Published fully built and never mutated, so readers need no lock once they
 * hold the pointer.  Entries are held BY POINTER so realloc cannot move one
 * out from under a reader mid-GEMM -- the same argument as qmat's cache.
 *
 * Freed at exit rather than never: `make leaks` is a gate on this project. */
typedef struct {
    const float *weight;
    unsigned long long fingerprint;
    size_t m, k, taps;
    size_t ldw;      /* transposed layout only: the source row stride */
    int trans;       /* 0 = [m][k][taps], 1 = [k][ldw] with taps == 1 */
    int8_t *q;        /* [taps][m][k] */
    float *scale;     /* [taps][m]    */
    int32_t *rowsum;  /* [taps][m]    */
} cq8_entry;

/* THE POINTER IS NOT THE IDENTITY, and finding that out cost a self-test.
 *
 * A weight is identified here by its address because this layer never sees a
 * tensor name.  In production that is sound: weights are mmapped and live as
 * long as the process.  It is not sound in general -- malloc recycles
 * addresses -- and the gate test below proved it by freeing one tensor and
 * allocating another of the SAME shape, which landed at the same address and
 * got the first one's quantized bytes back.  The relative error went from
 * 0.0046 to 1.45, which is what a stale cache looks like when it does not
 * crash.
 *
 * So the key is the address AND a fingerprint of the contents: up to 64
 * elements spread across the tensor, mixed.  That is O(1) per call -- 64 loads
 * against a GEMM of tens of millions of MACs -- and it does not pretend to be
 * a hash of the whole tensor: it is a recycled-address detector, and two
 * different tensors agreeing on 64 spread samples AND on (m, k, taps) is not a
 * failure mode worth a second pass over 7 MB per frame.
 *
 * src/seanet.c's sea_taps_all() keys the f32 tap permutation the same way and
 * has the same hazard; it is noted in PLAN.md rather than changed from here. */
static unsigned long long cq8_fingerprint(const float *w, size_t elems) {
    unsigned long long h = 1469598103934665603ull ^ (unsigned long long)elems;
    const size_t probes = elems < 64u ? elems : 64u;
    const size_t step = elems / probes;
    for (size_t i = 0; i < probes; ++i) {
        unsigned int bits;
        memcpy(&bits, &w[i * step], sizeof bits);
        h ^= (unsigned long long)bits;
        h *= 1099511628211ull;
    }
    return h;
}

static struct {
    cq8_entry **entries;
    size_t count, capacity;
    pthread_mutex_t mutex;
    int atexit_registered;
} g_memo = { NULL, 0, 0, PTHREAD_MUTEX_INITIALIZER, 0 };

static void cq8_entry_free(cq8_entry *e) {
    if (e == NULL) return;
    free(e->q);
    free(e->scale);
    free(e->rowsum);
    free(e);
}

static void cq8_memo_release(void) {
    pthread_mutex_lock(&g_memo.mutex);
    for (size_t i = 0; i < g_memo.count; ++i) cq8_entry_free(g_memo.entries[i]);
    free(g_memo.entries);
    g_memo.entries = NULL;
    g_memo.count = g_memo.capacity = 0;
    pthread_mutex_unlock(&g_memo.mutex);
}

/* Builds [taps][m][k] int8 from the [m][k][taps] f32 tensor.  The gather is
 * the same permutation sea_taps_all() does, done once here and then thrown
 * away: what is kept is a quarter of its size. */
static cq8_entry *cq8_pack(const float *weight, size_t m, size_t k, size_t taps,
                           int trans, size_t ldw, size_t fp_elems) {
    cq8_entry *e = (cq8_entry *)calloc(1, sizeof(*e));
    if (e == NULL) return NULL;
    e->weight = weight;
    e->fingerprint = cq8_fingerprint(weight, fp_elems);
    e->m = m; e->k = k; e->taps = taps;
    e->ldw = ldw; e->trans = trans;
    const size_t rows = m * taps;            /* checked by the caller */
    e->q = (int8_t *)malloc(rows * k);
    e->scale = (float *)malloc(rows * sizeof(float));
    e->rowsum = (int32_t *)malloc(rows * sizeof(int32_t));
    float *gather = (float *)malloc(m * k * sizeof(float));
    if (e->q == NULL || e->scale == NULL || e->rowsum == NULL || gather == NULL) {
        free(gather);
        cq8_entry_free(e);
        return NULL;
    }
    for (size_t t = 0; t < taps; ++t) {
        for (size_t i = 0; i < m; ++i) {
            /* Two source layouts, one destination.  `trans` is the transposed
             * convtranspose weight -- PyTorch stores ConvTranspose1d as
             * [in_channels][out_channels * kernel], so the logical row i is a
             * COLUMN there, strided by ldw. */
            const float *src = trans ? weight + i : weight + i * k * taps + t;
            const size_t step = trans ? ldw : taps;
            float *dst = gather + i * k;
            for (size_t p = 0; p < k; ++p) dst[p] = src[p * step];
        }
        if (mynah_qmat_pack_q8(gather, m, k, e->q + t * m * k,
                               e->scale + t * m, e->rowsum + t * m) != 0) {
            free(gather);
            cq8_entry_free(e);
            return NULL;
        }
    }
    free(gather);
    return e;
}

static const cq8_entry *cq8_memo_get(const float *weight, size_t m, size_t k,
                                     size_t taps, int trans, size_t ldw) {
    const size_t fp_elems = trans ? k * ldw : m * k * taps;
    const unsigned long long fp = cq8_fingerprint(weight, fp_elems);
    pthread_mutex_lock(&g_memo.mutex);
    for (size_t i = 0; i < g_memo.count; ++i) {
        cq8_entry *e = g_memo.entries[i];
        if (e->weight == weight && e->m == m && e->k == k && e->taps == taps &&
            e->trans == trans && e->ldw == ldw) {
            if (e->fingerprint == fp) {
                pthread_mutex_unlock(&g_memo.mutex);
                return e;
            }
            /* Same address, same shape, different contents: the tensor this
             * entry was built for is gone.  Re-quantize in place rather than
             * grow a second entry nothing will ever match. */
            cq8_entry *made = cq8_pack(weight, m, k, taps, trans, ldw, fp_elems);
            if (made == NULL) { pthread_mutex_unlock(&g_memo.mutex); return NULL; }
            g_memo.entries[i] = made;
            cq8_entry_free(e);
            pthread_mutex_unlock(&g_memo.mutex);
            return made;
        }
    }
    if (g_memo.count == g_memo.capacity) {
        const size_t cap = (g_memo.capacity == 0u) ? 16u : g_memo.capacity * 2u;
        cq8_entry **grown =
            (cq8_entry **)realloc(g_memo.entries, cap * sizeof(*grown));
        if (grown == NULL) { pthread_mutex_unlock(&g_memo.mutex); return NULL; }
        g_memo.entries = grown;
        g_memo.capacity = cap;
    }
    cq8_entry *made = cq8_pack(weight, m, k, taps, trans, ldw, fp_elems);
    if (made == NULL) { pthread_mutex_unlock(&g_memo.mutex); return NULL; }
    g_memo.entries[g_memo.count++] = made;
    if (!g_memo.atexit_registered) {
        g_memo.atexit_registered = 1;
        (void)atexit(cq8_memo_release);
    }
    cq8_bump(&g_cq8.packed_tensors);
    atomic_fetch_add_explicit(&g_cq8.packed_bytes,
                              (unsigned long long)(m * k * taps),
                              memory_order_relaxed);
    pthread_mutex_unlock(&g_memo.mutex);
    return made;
}

/* --------------------------------------------- the activation scratch
 *
 * Per thread and grown once, freed by a pthread_key destructor: the same
 * shape as qmat's row scratch, and for the same reason -- the decode loop must
 * not allocate, and a cache that cannot be reclaimed is a leak with a good
 * excuse.  Only the CALLING thread ever touches it: the quantization pass
 * happens before the region is dispatched, because every row block reads every
 * activation column. */
/* `tmp` holds ONE TILE of the transposed window, not one column: see the
 * tiling note in mynah_convq8_conv_taps.  CQ8_TILE_FLOATS is the cap that
 * keeps a tile inside L1 and is what the tile width is derived from, so the
 * scratch and the loop cannot disagree about the size. */
#define CQ8_TILE_FLOATS 8192u

typedef struct {
    unsigned char *xq;  /* [span][k] in this host's activation encoding */
    float *sx;          /* [span] activation scales                     */
    float *tmp;         /* one transposed tile, CQ8_TILE_FLOATS or k    */
    size_t span, k;
} cq8_scratch;

static pthread_key_t g_scratch_key;
static pthread_once_t g_scratch_once = PTHREAD_ONCE_INIT;

static void cq8_scratch_free(void *p) {
    cq8_scratch *s = (cq8_scratch *)p;
    if (s == NULL) return;
    free(s->xq);
    free(s->sx);
    free(s->tmp);
    free(s);
}

static void cq8_scratch_init(void) {
    (void)pthread_key_create(&g_scratch_key, cq8_scratch_free);
}

static cq8_scratch *cq8_scratch_get(size_t span, size_t k) {
    pthread_once(&g_scratch_once, cq8_scratch_init);
    cq8_scratch *s = (cq8_scratch *)pthread_getspecific(g_scratch_key);
    if (s == NULL) {
        s = (cq8_scratch *)calloc(1, sizeof(*s));
        if (s == NULL) return NULL;
        if (pthread_setspecific(g_scratch_key, s) != 0) { free(s); return NULL; }
    }
    if (span > s->span || k > s->k) {
        const size_t sp = span > s->span ? span : s->span;
        const size_t kk = k > s->k ? k : s->k;
        if (sp > (size_t)-1 / kk) return NULL;
        unsigned char *nx = (unsigned char *)realloc(s->xq, sp * kk);
        if (nx == NULL) return NULL;
        s->xq = nx;
        float *ns = (float *)realloc(s->sx, sp * sizeof(float));
        if (ns == NULL) return NULL;
        s->sx = ns;
        size_t tmp_floats = CQ8_TILE_FLOATS;
        if (tmp_floats < kk * 8u) tmp_floats = kk * 8u;
        float *nt = (float *)realloc(s->tmp, tmp_floats * sizeof(float));
        if (nt == NULL) return NULL;
        s->tmp = nt;
        s->span = sp;
        s->k = kk;
    }
    return s;
}


/* --------------------------------------------------------------- profiler
 *
 * Two phases, because there are exactly two things this file does and only
 * one of them is arithmetic: the activation pass walks the window COLUMN by
 * column, and a column is strided by `ldb` floats, so on the wide late stages
 * it touches one cache line per element.  Whether that pass or the GEMM
 * dominates is the whole question of whether the int8 path is worth it, and
 * it cannot be read off the SEANet table, which sees one `conv.q8` row.
 *
 * OFF unless MYNAH_CONVQ8_PROFILE is set: one relaxed load per call, never
 * inside an inner loop.  Same level-0 contract as costmap.h. */
#define CQ8_PROF_MAX 24

typedef struct {
    size_t m, n, k, taps;
    atomic_ullong calls, quant_ns, gemm_ns;
} cq8_prof_row;

static cq8_prof_row g_cq8_prof[CQ8_PROF_MAX];
static atomic_int g_cq8_prof_count;
static atomic_int g_cq8_prof_state = -1;
static atomic_int g_cq8_prof_hooked;
static atomic_flag g_cq8_prof_lock = ATOMIC_FLAG_INIT;

static void cq8_prof_report(void);

static int cq8_prof_on(void) {
    int state = atomic_load_explicit(&g_cq8_prof_state, memory_order_relaxed);
    if (state >= 0) return state;
    const char *env = getenv("MYNAH_CONVQ8_PROFILE");
    state = (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    atomic_store_explicit(&g_cq8_prof_state, state, memory_order_relaxed);
    if (state) {
        int expected = 0;
        if (atomic_compare_exchange_strong(&g_cq8_prof_hooked, &expected, 1)) {
            (void)atexit(cq8_prof_report);
        }
    }
    return state;
}

static unsigned long long cq8_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0u;
    return (unsigned long long)ts.tv_sec * 1000000000ull +
           (unsigned long long)ts.tv_nsec;
}

static void cq8_prof_add(size_t m, size_t n, size_t k, size_t taps,
                         unsigned long long quant_ns, unsigned long long gemm_ns) {
    while (atomic_flag_test_and_set(&g_cq8_prof_lock)) { }
    int count = atomic_load_explicit(&g_cq8_prof_count, memory_order_relaxed);
    int slot = -1;
    for (int i = 0; i < count; ++i) {
        if (g_cq8_prof[i].m == m && g_cq8_prof[i].n == n &&
            g_cq8_prof[i].k == k && g_cq8_prof[i].taps == taps) { slot = i; break; }
    }
    if (slot < 0 && count < CQ8_PROF_MAX) {
        slot = count;
        g_cq8_prof[slot].m = m; g_cq8_prof[slot].n = n;
        g_cq8_prof[slot].k = k; g_cq8_prof[slot].taps = taps;
        atomic_store_explicit(&g_cq8_prof_count, count + 1, memory_order_relaxed);
    }
    if (slot >= 0) {
        atomic_fetch_add_explicit(&g_cq8_prof[slot].calls, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_cq8_prof[slot].quant_ns, quant_ns, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_cq8_prof[slot].gemm_ns, gemm_ns, memory_order_relaxed);
    }
    atomic_flag_clear(&g_cq8_prof_lock);
}

static void cq8_prof_report(void) {
    const int count = atomic_load_explicit(&g_cq8_prof_count, memory_order_relaxed);
    if (count <= 0) return;
    fprintf(stderr,
            "[CONVQ8] the int8 conv tap GEMM, by shape. `quant` is the "
            "transpose-and-quantize\n  pass over the window (strided by ldb, "
            "on the calling thread); `gemm` is the region.\n");
    fprintf(stderr, "  %6s %6s %5s %5s %8s %10s %10s %7s\n",
            "m", "n", "k", "taps", "calls", "quant_ms", "gemm_ms", "quant%");
    double tq = 0.0, tg = 0.0;
    for (int i = 0; i < count; ++i) {
        const double q = (double)cq8_read(&g_cq8_prof[i].quant_ns) / 1e6;
        const double g = (double)cq8_read(&g_cq8_prof[i].gemm_ns) / 1e6;
        tq += q; tg += g;
        fprintf(stderr, "  %6zu %6zu %5zu %5zu %8llu %10.3f %10.3f %6.1f%%\n",
                g_cq8_prof[i].m, g_cq8_prof[i].n, g_cq8_prof[i].k,
                g_cq8_prof[i].taps, cq8_read(&g_cq8_prof[i].calls), q, g,
                (q + g) > 0.0 ? 100.0 * q / (q + g) : 0.0);
    }
    fprintf(stderr, "  total quant %.3f ms, gemm %.3f ms, quant share %.1f%%\n",
            tq, tg, (tq + tg) > 0.0 ? 100.0 * tq / (tq + tg) : 0.0);
}

/* ------------------------------------------------------------- the region */

typedef struct {
    const cq8_entry *e;
    const unsigned char *xq;
    const float *sx;
    const float *bias;
    float *c;
    size_t ldc;
    size_t m, n, k, taps, tap_stride, row_block;
} cq8_job;

/* One strip: `rows` weight rows against `batch` activation columns, summed
 * over every tap.
 *
 * THE TAP LOOP IS INNERMOST HERE, where the f32 fused path has it outermost,
 * and the difference is deliberate.  There the ordering is load-bearing: it
 * keeps the beta chain rounding each tap to f32 so the fused call stays
 * byte-identical to the per-tap calls it replaced.  Here there is nothing to
 * be identical to -- the arithmetic is new -- so the accumulator stays in a
 * register across taps and the output line is written once.  The sum is still
 * over taps in tap order for every element, so the result does not depend on
 * the blocking, on the thread count, or on which strip a row landed in. */
static void cq8_strip(const cq8_job *j, size_t i0, size_t rows, size_t j0,
                      size_t batch) {
    const cq8_entry *e = j->e;
    float acc[CQ8_MAX_BATCH][CQ8_ROW_STRIP];
    int32_t dots[CQ8_MAX_BATCH * CQ8_ROW_STRIP];
    for (size_t b = 0; b < batch; ++b)
        for (size_t r = 0; r < rows; ++r) acc[b][r] = 0.0f;

    for (size_t t = 0; t < j->taps; ++t) {
        const void *xp[CQ8_MAX_BATCH];
        float sa[CQ8_MAX_BATCH];
        for (size_t b = 0; b < batch; ++b) {
            const size_t col = j0 + b + t * j->tap_stride;
            xp[b] = j->xq + col * j->k;
            sa[b] = j->sx[col];
        }
        mynah_qmat_dots_i8(e->q + (t * j->m + i0) * j->k, rows, j->k,
                           e->rowsum + t * j->m + i0, xp, batch, dots, rows);
        const float *ws = e->scale + t * j->m + i0;
        for (size_t b = 0; b < batch; ++b) {
            const int32_t *db = dots + b * rows;
            const float sb = sa[b];
            for (size_t r = 0; r < rows; ++r)
                acc[b][r] += mynah_qmat_epilogue(db[r], ws[r], sb, 0.0f);
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        float *crow = j->c + (i0 + r) * j->ldc + j0;
        const float bv = (j->bias == NULL) ? 0.0f : j->bias[i0 + r];
        for (size_t b = 0; b < batch; ++b) crow[b] = acc[b][r] + bv;
    }
}

static void cq8_task(void *ctx, int index) {
    const cq8_job *j = (const cq8_job *)ctx;
    const size_t i0 = (size_t)index * j->row_block;
    if (i0 >= j->m) return;
    size_t block = j->row_block;
    if (i0 + block > j->m) block = j->m - i0;
    const size_t batch_max = mynah_qmat_dots_max_batch() < CQ8_MAX_BATCH
                                 ? mynah_qmat_dots_max_batch()
                                 : CQ8_MAX_BATCH;
    for (size_t s = 0; s < block; s += CQ8_ROW_STRIP) {
        size_t rows = block - s;
        if (rows > CQ8_ROW_STRIP) rows = CQ8_ROW_STRIP;
        for (size_t j0 = 0; j0 < j->n; j0 += batch_max) {
            size_t batch = j->n - j0;
            if (batch > batch_max) batch = batch_max;
            cq8_strip(j, i0 + s, rows, j0, batch);
        }
    }
}

/* ------------------------------------------------------------ entry point */

static int cq8_mul(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > (size_t)-1 / a) return -1;
    *out = a * b;
    return 0;
}

static int cq8_run(size_t m, size_t n, size_t k, size_t taps, int trans,
                   const float *weight, size_t ldw, const float *b, size_t ldb,
                   size_t tap_stride, const float *bias, float *c, size_t ldc) {
    if (weight == NULL || b == NULL || c == NULL || taps == 0u) return -1;
    cq8_bump(&g_cq8.calls);
    if (m == 0u || n == 0u || k == 0u) { cq8_bump(&g_cq8.refused_shape); return 1; }
    if (ldc < n) return -1;
    if (!mynah_convq8_host_ok(NULL)) { cq8_bump(&g_cq8.refused_host); return 1; }
    if (k > CQ8_K_MAX) { cq8_bump(&g_cq8.refused_shape); return 1; }
    if (mynah_qmat_dots_max_batch() < 1u) { cq8_bump(&g_cq8.refused_shape); return 1; }

    /* The activation columns this call reads: [0, n + (taps-1)*tap_stride). */
    size_t span = 0;
    if (cq8_mul(taps - 1u, tap_stride, &span) != 0) {
        cq8_bump(&g_cq8.refused_shape);
        return 1;
    }
    if (span > (size_t)-1 - n) { cq8_bump(&g_cq8.refused_shape); return 1; }
    span += n;
    if (ldb < span) return -1;

    size_t work = 0, rows_total = 0, src_elems = 0;
    if (cq8_mul(m, n, &work) != 0 || cq8_mul(work, k, &work) != 0 ||
        cq8_mul(work, taps, &work) != 0 || cq8_mul(m, taps, &rows_total) != 0 ||
        rows_total > (size_t)-1 / k) {
        cq8_bump(&g_cq8.refused_shape);
        return 1;
    }
    /* The source tensor is walked by the fingerprint, so its extent has to be
     * representable too -- and for the transposed layout that is k * ldw, not
     * m * k. */
    if (cq8_mul(trans ? k : m, trans ? ldw : k * taps, &src_elems) != 0 ||
        src_elems == 0u) {
        cq8_bump(&g_cq8.refused_shape);
        return 1;
    }
    if (trans && (ldw < m || taps != 1u)) {
        cq8_bump(&g_cq8.refused_shape);
        return 1;
    }
    if (work < CQ8_MIN_WORK) { cq8_bump(&g_cq8.refused_shape); return 1; }
    /* THE TWO SHAPE CONDITIONS, each measured against the f32 path it would
     * replace (BLAS=none, 2 threads, the eight conv shapes one PocketTTS
     * utterance actually runs -- .work/seanet-int8-conv.md has the table):
     *
     *   k * taps  is the DEPTH of one output element's accumulation.  Below
     *   128 the SDOT kernel cannot amortise its per-row setup -- accumulator
     *   init, horizontal reduce, epilogue -- and the f32 panel kernel, which
     *   keeps a whole C tile in registers instead, wins.  Measured: k*taps
     *   3584 -> 4.21x, 768 -> 2.91x, 384 -> 1.41x, 192 -> 1.44x, 128 -> 1.56x,
     *   then 64 -> 0.84x and 32 -> 0.38x.  The sign changes between 128 and 64.
     *
     *   m  is the output channel count, and it is here for TWO reasons that
     *   happen to point the same way.  Speed: m is how many weight rows reuse
     *   one quantized activation column, and the pass is paid once per call
     *   whatever m is -- at m = 1 it IS the call (the last convolution, 1 x
     *   1920, spends 64% of its time quantizing for a single output row, and
     *   comes out 0.28x).  QUALITY: a SEANet decoder narrows toward the
     *   waveform, so the narrow stages are also the LAST ones, and their int8
     *   residual reaches the output with nothing downstream to average it.
     *   Measured on the same utterance, admitting the 32-channel stage
     *   (32 x 1920, k 64, 3 taps) moves log-mel correlation 0.9976 -> 0.9785
     *   -- an eightfold increase in spectral error -- to save 6 ms of 44.
     *   That trade is refused; 64 channels costs nothing measurable.
     *
     * Both numbers are from THIS machine and are a development signal, not a
     * product claim.  What is not host-specific is their shape:
     * too shallow to amortise a kernel, too close to the output to hide a
     * residual. */
    if (k * taps < 128u || m < 64u) {
        cq8_bump(&g_cq8.refused_shape);
        return 1;
    }

    const cq8_entry *e = cq8_memo_get(weight, m, k, taps, trans, ldw);
    if (e == NULL) { cq8_bump(&g_cq8.refused_pack); return 1; }

    cq8_scratch *sc = cq8_scratch_get(span, k);
    if (sc == NULL) { cq8_bump(&g_cq8.refused_scratch); return 1; }

    /* Transpose-and-quantize the window ONCE, on the calling thread: every row
     * block reads every activation column, and the column a tap wants is the
     * same column another tap already quantized, shifted.  A per-tap pass
     * would quantize the same float `taps` times and, worse, could give one
     * column two different scales. */
    const int prof = cq8_prof_on();
    const unsigned long long t_quant = prof ? cq8_now() : 0u;
    /* One column at a time, gathered with a stride of `ldb` floats.
     *
     * TILING THIS WAS TRIED AND IS SLOWER.  A tiled transpose reads each cache
     * line once and uses all of it, which is the right instinct -- but it pays
     * for that by writing with the stride instead of reading with it, and
     * measured 30.5 -> 34.1 ms on the same utterance.  The cost here is not
     * the memory pattern: it is the quantizer, which is two passes of scalar
     * float work per column.  Vectorising THAT is what moved the number
     * (see quantize_act_int8 in src/qmat.c). */
    for (size_t col = 0; col < span; ++col) {
        const float *src = b + col;
        float *tmp = sc->tmp;
        for (size_t p = 0; p < k; ++p) tmp[p] = src[p * ldb];
        sc->sx[col] = mynah_qmat_act_quantize(sc->xq + col * k, tmp, k);
    }
    const unsigned long long t_gemm = prof ? cq8_now() : 0u;

    cq8_job job;
    memset(&job, 0, sizeof job);
    job.e = e;
    job.xq = sc->xq;
    job.sx = sc->sx;
    job.bias = bias;
    job.c = c;
    job.ldc = ldc;
    job.m = m; job.n = n; job.k = k;
    job.taps = taps;
    job.tap_stride = tap_stride;

    /* Row blocking is the only axis the thread count touches, so which strip
     * covers a row never changes the value it gets -- only who computes it. */
    size_t want = 1u;
    const int threads = mynah_num_threads();
    if (threads > 1 && work >= CQ8_PARALLEL_MIN_WORK) want = (size_t)threads * 2u;
    /* The self-test's handle on the row plan: the pool cannot be resized from
     * inside a process, so the blocking is forced instead, which is the same
     * question -- does a row's value depend on which block covered it. */
    if (g_cq8_force_blocks > 0u) want = g_cq8_force_blocks;
    size_t blocks = want;
    if (blocks > m) blocks = m;
    if (blocks == 0u) blocks = 1u;
    size_t row_block = (m + blocks - 1u) / blocks;
    row_block += row_block & 1u;   /* the SDOT kernel pairs rows */
    if (row_block == 0u) row_block = 2u;
    job.row_block = row_block;
    blocks = (m + row_block - 1u) / row_block;
    if (blocks == 0u || blocks > (size_t)0x7fffffff) {
        cq8_bump(&g_cq8.refused_shape);
        return 1;
    }

    if (want == 1u || blocks == 1u) {
        for (size_t t = 0; t < blocks; ++t) cq8_task(&job, (int)t);
    } else {
        mynah_parallel_for((int)blocks, cq8_task, &job);
    }
    if (prof) cq8_prof_add(m, n, k, taps, t_gemm - t_quant, cq8_now() - t_gemm);
    cq8_bump(&g_cq8.ran);
    return 0;
}

int mynah_convq8_conv_taps(size_t m, size_t n, size_t k, size_t taps,
                           const float *weight, const float *b, size_t ldb,
                           size_t tap_stride, const float *bias, float *c,
                           size_t ldc) {
    return cq8_run(m, n, k, taps, 0, weight, 0u, b, ldb, tap_stride, bias, c,
                   ldc);
}

int mynah_convq8_gemm_tn(size_t m, size_t n, size_t k, const float *weight,
                         size_t ldw, const float *b, size_t ldb, float *c,
                         size_t ldc) {
    return cq8_run(m, n, k, 1u, 1, weight, ldw, b, ldb, 0u, NULL, c, ldc);
}

/* ======================================================================
 * Self-test
 *
 * THE ORACLE is an exact f32 triple loop written here, not the sgemm path:
 * comparing an approximation against another optimized kernel proves only
 * that two fast things agree.
 *
 * THE BOUNDS ARE ABSOLUTE.  A gate that compares int8 against "whatever the
 * baseline produced" passes mutations, because a symmetric degradation moves
 * the subject and the baseline together -- E10-6 found exactly that, twice.
 * So each shape carries a number measured on this tree and a margin, and a
 * kernel that loses a lane blows it by an order of magnitude rather than by a
 * few percent.
 *
 * AND ONE EXACT ASSERTION.  Integer accumulation has no rounding, so
 * mynah_qmat_dots_i8 must return the SAME int32 for a row whatever the batch
 * width and whatever path the host compiled.  That is checked with ==, which
 * is the only part of int8 that can be.
 * ====================================================================== */

static float cq8_fake(size_t index, size_t salt) {
    const size_t h = (index * 1103515245u + salt * 12345u + 1013904223u);
    return (float)((double)(h % 2039u) / 1019.5 - 1.0);
}

/* c[i][j] = bias[i] + sum_t sum_p w[i][p][t] * b[p*ldb + j + t*stride] */
static void cq8_reference(size_t m, size_t n, size_t k, size_t taps,
                          const float *w, const float *b, size_t ldb,
                          size_t stride, const float *bias, float *c,
                          size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t t = 0; t < taps; ++t) {
                for (size_t p = 0; p < k; ++p) {
                    acc += (double)w[i * k * taps + p * taps + t] *
                           (double)b[p * ldb + j + t * stride];
                }
            }
            c[i * ldc + j] = (float)acc + (bias == NULL ? 0.0f : bias[i]);
        }
    }
}

static double cq8_rel_l2(const float *got, const float *want, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * (double)want[i];
    }
    if (den == 0.0) return num == 0.0 ? 0.0 : 1.0;
    return sqrt(num / den);
}

typedef struct {
    size_t m, n, k, taps, stride;
    double bound;   /* absolute, measured on this tree plus a margin */
    const char *what;
} cq8_case;

/* Measured relative L2 against the f64 oracle, times a 1.3 margin.  The
 * measured column -- 0.00456 / 0.01006 / 0.01064 / 0.01004 / 0.00574 / 0.01644
 * on this tree -- is reprinted by MYNAH_CONVQ8_DEBUG=1, so a bound is never a
 * number someone remembered.  Fewer taps and a narrower k give a LARGER error,
 * which is the expected direction: the int8 residual averages down over the
 * length of the accumulation, and the one-tap k=256 case has the shortest. */
static const cq8_case g_cq8_cases[] = {
    { 512,  16, 512, 7, 1, 0.0060, "entry conv, 19.5% of the sgemm wall" },
    { 128,  32, 256, 3, 1, 0.0130, "stage-1 residual" },
    {  64,  48, 128, 3, 2, 0.0140, "dilated residual" },
    {  64,  64,  64, 3, 1, 0.0130, "narrow k, at the channel floor" },
    {  96,   8,  96, 5, 1, 0.0075, "odd k, odd taps" },
    { 256,  24, 256, 1, 1, 0.0215, "kernel 1: one tap" }
};

static int cq8_fail(char *error, size_t capacity, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (error != NULL && capacity > 0u) vsnprintf(error, capacity, fmt, ap);
    va_end(ap);
    return -1;
}

/* The exact half: the int32 a row gets must not depend on the batch width. */
static int cq8_test_dots(char *error, size_t capacity) {
    enum { ROWS = 9, COLS = 133 };
    float *w = (float *)malloc((size_t)ROWS * COLS * sizeof(float));
    float *x = (float *)malloc(4u * COLS * sizeof(float));
    int8_t *q = (int8_t *)malloc((size_t)ROWS * COLS);
    float *ws = (float *)malloc(ROWS * sizeof(float));
    int32_t *rs = (int32_t *)malloc(ROWS * sizeof(int32_t));
    unsigned char *xq = (unsigned char *)malloc(4u * COLS);
    int32_t *wide = (int32_t *)malloc(4u * ROWS * sizeof(int32_t));
    int32_t *one = (int32_t *)malloc(4u * ROWS * sizeof(int32_t));
    int rc = -1;
    if (w == NULL || x == NULL || q == NULL || ws == NULL || rs == NULL ||
        xq == NULL || wide == NULL || one == NULL) {
        rc = cq8_fail(error, capacity, "convq8: out of memory in dots test");
        goto done;
    }
    for (size_t i = 0; i < (size_t)ROWS * COLS; ++i) w[i] = cq8_fake(i, 11u);
    for (size_t i = 0; i < 4u * COLS; ++i) x[i] = cq8_fake(i, 23u);
    if (mynah_qmat_pack_q8(w, ROWS, COLS, q, ws, rs) != 0) {
        rc = cq8_fail(error, capacity, "convq8: mynah_qmat_pack_q8 refused %dx%d",
                      ROWS, COLS);
        goto done;
    }
    const void *xp[4];
    for (size_t b = 0; b < 4u; ++b) {
        (void)mynah_qmat_act_quantize(xq + b * COLS, x + b * COLS, COLS);
        xp[b] = xq + b * COLS;
    }
    mynah_qmat_dots_i8(q, ROWS, COLS, rs, xp, 4u, wide, ROWS);
    for (size_t b = 0; b < 4u; ++b) {
        const void *single[1] = { xp[b] };
        mynah_qmat_dots_i8(q, ROWS, COLS, rs, single, 1u, one, ROWS);
        for (size_t r = 0; r < ROWS; ++r) {
            if (one[r] != wide[b * ROWS + r]) {
                rc = cq8_fail(error, capacity,
                              "convq8: int32 depends on the batch width -- "
                              "batch 4 row %zu act %zu gave %d, batch 1 gave %d",
                              r, b, wide[b * ROWS + r], one[r]);
                goto done;
            }
        }
    }
    /* And against the AR loop's own int8 matvec, through the cache path that
     * has the mutation tests behind it: the exported primitives must not be a
     * second implementation that merely agrees to a tolerance.  Both end in
     * qmat_row_epilogue on the same int32, so this is ==, not a bound. */
    {
        mynah_qmat_cache *cache = mynah_qmat_cache_new(1);
        if (cache != NULL && mynah_qmat_cache_qtype(cache) == 1) {
            float *ref = (float *)malloc(ROWS * sizeof(float));
            if (ref == NULL) {
                mynah_qmat_cache_free(cache);
                rc = cq8_fail(error, capacity, "convq8: out of memory");
                goto done;
            }
            char ignored[128];
            const int lrc = mynah_qmat_linear_resolved_qt(
                cache, NULL, "convq8.selftest", w, x, ref, 1u, COLS, ROWS, NULL,
                1 /* int8 */, ignored, sizeof ignored);
            const float sx = mynah_qmat_act_quantize(xq, x, COLS);
            for (size_t r = 0; lrc == 0 && r < ROWS; ++r) {
                const float mine = mynah_qmat_epilogue(wide[r], ws[r], sx, 0.0f);
                if (memcmp(&mine, &ref[r], sizeof mine) != 0) {
                    /* Read BEFORE the free.  It was written the other way
                     * round, and the first mutation run reported the
                     * reference as "0" -- a diagnostic that lied about
                     * exactly the number it exists to show. */
                    const double want = (double)ref[r];
                    free(ref);
                    mynah_qmat_cache_free(cache);
                    rc = cq8_fail(error, capacity,
                                  "convq8: the exported primitives disagree "
                                  "with the cached int8 matvec at row %zu: "
                                  "%.9g vs %.9g",
                                  r, (double)mine, want);
                    goto done;
                }
            }
            free(ref);
        }
        mynah_qmat_cache_free(cache);
    }
    rc = 0;
done:
    free(w); free(x); free(q); free(ws); free(rs); free(xq); free(wide); free(one);
    return rc;
}

/* The SHAPE GATE, asserted rather than commented.  Both conditions exist to
 * keep a measured loss out of the build -- one a speed loss, one a quality
 * loss -- and a threshold nothing checks is a threshold that drifts. */
static int cq8_test_gate(char *error, size_t error_capacity) {
    static const struct { size_t m, n, k, taps; int expect_ran; const char *what; }
    cases[] = {
        { 512, 16, 512, 7, 1, "the entry conv: deep and wide, 4.2x" },
        {  64, 64,  64, 3, 1, "exactly at both floors" },
        {  32, 64,  64, 3, 0, "m below the channel floor: 8x worse log-mel" },
        { 256, 96,  64, 1, 0, "k*taps below the depth floor: 0.84x" },
        {   1, 512, 64, 3, 0, "m == 1: 64% of the call is the activation pass" }
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        const size_t m = cases[i].m, n = cases[i].n, k = cases[i].k;
        const size_t taps = cases[i].taps;
        const size_t span = n + (taps - 1u);
        float *w = (float *)malloc(m * k * taps * sizeof(float));
        float *b = (float *)malloc(k * span * sizeof(float));
        float *c = (float *)malloc(m * n * sizeof(float));
        if (w == NULL || b == NULL || c == NULL) {
            free(w); free(b); free(c);
            return cq8_fail(error, error_capacity, "convq8: out of memory");
        }
        for (size_t t = 0; t < m * k * taps; ++t) w[t] = cq8_fake(t, 17u);
        for (size_t t = 0; t < k * span; ++t) b[t] = cq8_fake(t, 29u);
        const int rc = mynah_convq8_conv_taps(m, n, k, taps, w, b, span, 1u,
                                              NULL, c, n);
        free(w); free(b); free(c);
        const int ran = (rc == 0);
        if (ran != cases[i].expect_ran) {
            return cq8_fail(error, error_capacity,
                            "convq8: the shape gate moved -- %zux%zux%zu taps "
                            "%zu (%s) %s but should have been %s",
                            m, n, k, taps, cases[i].what,
                            ran ? "ran" : "refused",
                            cases[i].expect_ran ? "run" : "refused");
        }
    }
    return 0;
}

/* The transposed entry point, against the same f64 oracle.
 *
 * It is a SEPARATE test and not another row in g_cq8_cases because the thing
 * that can be wrong is the pack's gather -- reading a column of the stored
 * tensor where it should read a row -- and a shape whose m and k are equal
 * would not notice.  The shapes here are the three the decoder actually runs,
 * shrunk on n, and none of them is square. */
static int cq8_test_tn(char *error, size_t error_capacity) {
    static const struct { size_t m, n, k; double bound; const char *what; } cases[] = {
        { 3072, 8, 512, 0.0130, "convtr 1: 512ch in, kernel 12" },
        { 1280, 8, 256, 0.0085, "convtr 2: 256ch in, kernel 10" },
        {  512, 8, 128, 0.0330, "convtr 3: 128ch in, kernel 8" }
    };
    for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ++ci) {
        const size_t m = cases[ci].m, n = cases[ci].n, k = cases[ci].k;
        /* ldw > m on purpose: the stored tensor is wider than the block being
         * read, which is what catches a pack that assumes ldw == m. */
        const size_t ldw = m + 3u;
        float *w = (float *)malloc(k * ldw * sizeof(float));
        float *b = (float *)malloc(k * n * sizeof(float));
        float *got = (float *)malloc(m * n * sizeof(float));
        float *want = (float *)malloc(m * n * sizeof(float));
        if (w == NULL || b == NULL || got == NULL || want == NULL) {
            free(w); free(b); free(got); free(want);
            return cq8_fail(error, error_capacity, "convq8: out of memory");
        }
        for (size_t i = 0; i < k * ldw; ++i) w[i] = cq8_fake(i, ci + 211u);
        for (size_t i = 0; i < k * n; ++i) b[i] = cq8_fake(i, ci + 307u);
        const int rc = mynah_convq8_gemm_tn(m, n, k, w, ldw, b, n, got, n);
        if (rc != 0) {
            free(w); free(b); free(got); free(want);
            return cq8_fail(error, error_capacity,
                            "convq8: transposed %zux%zux%zu (%s) refused "
                            "(rc %d)", m, n, k, cases[ci].what, rc);
        }
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (size_t p = 0; p < k; ++p)
                    acc += (double)w[p * ldw + i] * (double)b[p * n + j];
                want[i * n + j] = (float)acc;
            }
        }
        const double rel = cq8_rel_l2(got, want, m * n);
        if (getenv("MYNAH_CONVQ8_DEBUG") != NULL) {
            fprintf(stderr, "[convq8] TN %5zux%4zux%4zu  relL2 %.5f  bound "
                            "%.5f  %s\n", m, n, k, rel, cases[ci].bound,
                    cases[ci].what);
        }
        free(w); free(b); free(got); free(want);
        if (!(rel <= cases[ci].bound)) {
            return cq8_fail(error, error_capacity,
                            "convq8: transposed %zux%zux%zu (%s) relative L2 "
                            "%.5f exceeds the measured bound %.5f",
                            m, n, k, cases[ci].what, rel, cases[ci].bound);
        }
    }
    return 0;
}

int mynah_convq8_self_test(char *error, size_t error_capacity) {
    const int before_force = mynah_convq8_force(1);
    int rc = cq8_test_dots(error, error_capacity);
    if (rc == 0) rc = cq8_test_gate(error, error_capacity);
    if (rc == 0) rc = cq8_test_tn(error, error_capacity);
    if (rc != 0) { (void)mynah_convq8_force(before_force); return rc; }

    double worst = 0.0;
    const char *worst_what = "";
    for (size_t ci = 0; ci < sizeof g_cq8_cases / sizeof g_cq8_cases[0]; ++ci) {
        const cq8_case *sc = &g_cq8_cases[ci];
        const size_t span = sc->n + (sc->taps - 1u) * sc->stride;
        float *w = (float *)malloc(sc->m * sc->k * sc->taps * sizeof(float));
        float *b = (float *)malloc(sc->k * span * sizeof(float));
        float *bias = (float *)malloc(sc->m * sizeof(float));
        float *got = (float *)malloc(sc->m * sc->n * sizeof(float));
        float *alt = (float *)malloc(sc->m * sc->n * sizeof(float));
        float *want = (float *)malloc(sc->m * sc->n * sizeof(float));
        if (w == NULL || b == NULL || bias == NULL || got == NULL ||
            alt == NULL || want == NULL) {
            free(w); free(b); free(bias); free(got); free(alt); free(want);
            (void)mynah_convq8_force(before_force);
            return cq8_fail(error, error_capacity, "convq8: out of memory");
        }
        for (size_t i = 0; i < sc->m * sc->k * sc->taps; ++i) w[i] = cq8_fake(i, ci + 3u);
        for (size_t i = 0; i < sc->k * span; ++i) b[i] = cq8_fake(i, ci + 71u);
        for (size_t i = 0; i < sc->m; ++i) bias[i] = 0.25f * cq8_fake(i, ci + 131u);

        mynah_convq8_stats before, after;
        mynah_convq8_stats_get(&before);
        const int ran = mynah_convq8_conv_taps(sc->m, sc->n, sc->k, sc->taps, w,
                                               b, span, sc->stride, bias, got,
                                               sc->n);
        mynah_convq8_stats_get(&after);
        if (ran != 0) {
            free(w); free(b); free(bias); free(got); free(alt); free(want);
            (void)mynah_convq8_force(before_force);
            return cq8_fail(error, error_capacity,
                            "convq8: shape %zux%zux%zu taps %zu (%s) refused "
                            "(rc %d); a gate that never ran the path proves "
                            "nothing",
                            sc->m, sc->n, sc->k, sc->taps, sc->what, ran);
        }
        if (after.ran != before.ran + 1u) {
            free(w); free(b); free(bias); free(got); free(alt); free(want);
            (void)mynah_convq8_force(before_force);
            return cq8_fail(error, error_capacity,
                            "convq8: the call returned 0 but the ran counter "
                            "did not move (%llu -> %llu)",
                            before.ran, after.ran);
        }

        cq8_reference(sc->m, sc->n, sc->k, sc->taps, w, b, span, sc->stride,
                      bias, want, sc->n);
        const double rel = cq8_rel_l2(got, want, sc->m * sc->n);
        if (rel > worst) { worst = rel; worst_what = sc->what; }
        /* How the bounds in g_cq8_cases are re-measured rather than guessed:
         * MYNAH_CONVQ8_DEBUG=1 prints the column they came from. */
        if (getenv("MYNAH_CONVQ8_DEBUG") != NULL) {
            fprintf(stderr,
                    "[convq8] %4zux%4zux%4zu taps %zu dil %zu  relL2 %.5f  "
                    "bound %.5f  %s\n",
                    sc->m, sc->n, sc->k, sc->taps, sc->stride, rel, sc->bound,
                    sc->what);
        }
        if (!(rel <= sc->bound)) {
            free(w); free(b); free(bias); free(got); free(alt); free(want);
            (void)mynah_convq8_force(before_force);
            return cq8_fail(error, error_capacity,
                            "convq8: %zux%zux%zu taps %zu (%s) relative L2 "
                            "%.5f exceeds the measured bound %.5f",
                            sc->m, sc->n, sc->k, sc->taps, sc->what, rel,
                            sc->bound);
        }

        /* Same shape, a different row plan: the value a row gets must not
         * depend on which block covered it, and here that IS bit-identical --
         * the two arms are the same machine code at different offsets. */
        for (size_t blocks = 1u; blocks <= 8u; blocks *= 2u) {
            g_cq8_force_blocks = blocks;
            const int rc2 = mynah_convq8_conv_taps(sc->m, sc->n, sc->k, sc->taps,
                                                   w, b, span, sc->stride, bias,
                                                   alt, sc->n);
            g_cq8_force_blocks = 0u;
            if (rc2 != 0 || memcmp(alt, got, sc->m * sc->n * sizeof(float)) != 0) {
                free(w); free(b); free(bias); free(got); free(alt); free(want);
                (void)mynah_convq8_force(before_force);
                return cq8_fail(error, error_capacity,
                                "convq8: %zux%zux%zu taps %zu is not "
                                "block-independent at %zu blocks (rc %d)",
                                sc->m, sc->n, sc->k, sc->taps, blocks, rc2);
            }
        }
        free(w); free(b); free(bias); free(got); free(alt); free(want);
    }
    (void)mynah_convq8_force(before_force);
    if (error != NULL && error_capacity > 0u) {
        snprintf(error, error_capacity,
                 "convq8: worst relative L2 %.5f (%s)", worst, worst_what);
    }
    return 0;
}

/* ------------------------------------------------------------- probes */

static int probe_convq8_host(char *out, size_t capacity, const char **why) {
    const char *reason = NULL;
    const int on = mynah_convq8_host_ok(&reason);
    snprintf(out, capacity, "%s", on ? "eligible" : "off");
    *why = reason;
    return 0;
}

static int probe_convq8_path(char *out, size_t capacity, const char **why) {
    static char text[300];
    mynah_convq8_stats st;
    mynah_convq8_stats_get(&st);
    if (st.calls == 0u) {
        snprintf(out, capacity, "n/a");
        *why = "[predicate] src/convq8.c: no conv tap GEMM has asked for int8 "
               "in this process. Either the codec_conv group did not resolve "
               "to int8 or no synthesis has run; --dispatch-map loads no model";
        return 0;
    }
    snprintf(out, capacity, "%llu/%llu", st.ran, st.calls);
    snprintf(text, sizeof text,
             "[predicate] src/convq8.c mynah_convq8_conv_taps(): %llu of %llu "
             "calls ran int8 over %llu packed tensors (%llu KiB). Refused: "
             "%llu host, %llu shape, %llu pack, %llu scratch",
             st.ran, st.calls, st.packed_tensors, st.packed_bytes / 1024u,
             st.refused_host, st.refused_shape, st.refused_pack,
             st.refused_scratch);
    *why = text;
    return 0;
}

void mynah_convq8_dispatch_probes(void) {
    mynah_dispatch_register_value_probe("codec.conv_int8_host", probe_convq8_host);
    mynah_dispatch_register_value_probe("codec.conv_int8_path", probe_convq8_path);
}
