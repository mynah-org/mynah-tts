/* costmap.c — see costmap.h for the semantics, which were fixed before any
 * number was collected.
 *
 * The one thing this file must never do is change what it measures.  Hence:
 *
 *   - Thread blocks come from a STATIC array, claimed with a single relaxed
 *     fetch_add the first time a thread touches a marker.  There is no malloc
 *     anywhere in this file, so there is nothing to free, nothing to leak, and
 *     nothing for an allocator lock to serialize inside a parallel region.
 *   - No atomic read-modify-write on a per-region counter.  Every accumulation
 *     is a plain store to thread-local memory; the merge happens once, at
 *     report time, by walking the static array.
 *   - When the map is off, a marker is one relaxed load of an int and one
 *     predictable branch — the inline wrappers in costmap.h never even call in.
 *   - One clock read per begin and one per end, CLOCK_MONOTONIC.
 */
#include "costmap.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int mynah_costmap_level_v = 0;

/* ======================================================================
 * Static taxonomy.  Append-only: never renumber, never reuse an id.
 * ====================================================================== */

typedef struct {
    int         id;
    const char *name;
    int         parent;
    int         level;
    const char *component;
    const char *mode;      /* "stack" = begin/end pair; "derived" = added ns */
} rgn_info;

static const rgn_info g_rgn[] = {
    /* DERIVED, not stack: a request is opened on the driver loop thread and
     * closed when its context is destroyed, with other requests interleaved in
     * between, so no single thread-local stack can bracket it.  It is written
     * by mynah_region_add_ns() from the context lifetime. */
    { MYNAH_RGN_REQUEST,           "request.total",          MYNAH_RGN_NONE,     1, "driver",   "derived" },
    { MYNAH_RGN_PREPARE,           "request.prepare",        MYNAH_RGN_REQUEST,  1, "driver",   "stack"   },
    { MYNAH_RGN_TOKENIZE,          "prep.tokenize",          MYNAH_RGN_PREPARE,  1, "prep",     "stack"   },
    { MYNAH_RGN_ENCODER,           "prep.encoder",           MYNAH_RGN_PREPARE,  1, "encoder",  "stack"   },
    { MYNAH_RGN_PREFILL,           "prep.decoder_prefill",   MYNAH_RGN_PREPARE,  1, "decoder",  "stack"   },
    { MYNAH_RGN_FINALIZE,          "request.finalize",       MYNAH_RGN_REQUEST,  1, "driver",   "stack"   },
    /* The prefill's four linear projections, and only those.  Its parent's
     * self time is then the prefill's other half -- attention scores, softmax,
     * the RoPE rotation, the KV writes, the norms, GELU and the residuals --
     * which is the half no hook can hand to the pool. */
    { MYNAH_RGN_PREFILL_PROJ,      "prep.prefill_proj",      MYNAH_RGN_PREFILL,  1, "decoder",  "stack"   },

    { MYNAH_RGN_STEP,              "step.total",             MYNAH_RGN_REQUEST,  1, "decoder",  "stack"   },
    { MYNAH_RGN_STEP_EMBED,        "step.embed",             MYNAH_RGN_STEP,     2, "decoder",  "stack"   },
    { MYNAH_RGN_STEP_BACKBONE,     "step.backbone",          MYNAH_RGN_STEP,     2, "decoder",  "stack"   },
    { MYNAH_RGN_STEP_ATTENTION,    "step.attention",         MYNAH_RGN_STEP_BACKBONE, 2, "decoder", "stack" },
    { MYNAH_RGN_STEP_FFN,          "step.ffn",               MYNAH_RGN_STEP_BACKBONE, 2, "decoder", "stack" },
    /* MULTI: a discrete engine runs the head inside step_batch, PocketTTS runs
     * its EOS head inside emit_batch.  Both are correct; declaring one of them
     * flagged the other on every call. */
    { MYNAH_RGN_STEP_HEAD,         "step.head",              MYNAH_RGN_MULTI,    1, "decoder",  "stack"   },
    { MYNAH_RGN_EMIT,              "step.emit",              MYNAH_RGN_REQUEST,  1, "decoder",  "stack"   },

    { MYNAH_RGN_LOCAL,             "local.total",            MYNAH_RGN_MULTI,    1, "local",    "stack"   },
    { MYNAH_RGN_LOCAL_STEP,        "local.stream_step",      MYNAH_RGN_LOCAL,    2, "local",    "stack"   },
    { MYNAH_RGN_LOCAL_PROJ,        "local.stream_proj",      MYNAH_RGN_LOCAL,    2, "local",    "stack"   },

    /* MULTI: under step.emit in the batched path, under step.total when a
     * single context draws its latent alone. */
    { MYNAH_RGN_FLOW,              "flow.head",              MYNAH_RGN_MULTI,    1, "flow",     "stack"   },

    { MYNAH_RGN_CODEC,             "codec.total",            MYNAH_RGN_MULTI,    1, "codec",    "stack"   },
    { MYNAH_RGN_CODEC_EMBED,       "codec.embed",            MYNAH_RGN_CODEC,    2, "codec",    "stack"   },
    { MYNAH_RGN_CODEC_TRANSFORMER, "codec.transformer",      MYNAH_RGN_CODEC,    2, "codec",    "stack"   },
    { MYNAH_RGN_CODEC_CONV,        "codec.conv_stack",       MYNAH_RGN_CODEC,    2, "codec",    "stack"   },
    { MYNAH_RGN_CODEC_POST,        "codec.post",             MYNAH_RGN_CODEC,    2, "codec",    "stack"   },

    { MYNAH_RGN_STREAM_EMIT,       "stream.emit_callback",   MYNAH_RGN_MULTI,    1, "stream",   "stack"   },

    { MYNAH_RGN_RT_REQUEST,        "runtime.request",        MYNAH_RGN_NONE,     1, "runtime",  "derived" },
    { MYNAH_RGN_RT_ADMISSION,      "runtime.admission",      MYNAH_RGN_RT_REQUEST, 1, "runtime", "derived" },
    { MYNAH_RGN_RT_PARALLEL,       "runtime.parallel_for",   MYNAH_RGN_MULTI,    1, "runtime",  "stack"   },
    { MYNAH_RGN_RT_PARALLEL_WAIT,  "runtime.parallel_wait",  MYNAH_RGN_RT_PARALLEL, 1, "runtime", "stack" },
    { MYNAH_RGN_MODEL_LOAD,        "runtime.model_load",     MYNAH_RGN_NONE,     1, "runtime",  "stack"   },
    { MYNAH_RGN_PREPACK,           "runtime.weight_prepack", MYNAH_RGN_MODEL_LOAD, 1, "runtime", "stack"  },

    { MYNAH_RGN_DECODE_GANG,       "driver.decode_gang",     MYNAH_RGN_NONE,     1, "driver",   "stack"   },
    { MYNAH_RGN_LANE_WAIT,         "driver.lane_wait",       MYNAH_RGN_NONE,     1, "driver",   "stack"   },
    { MYNAH_RGN_LANE_DECODE,       "driver.lane_decode",     MYNAH_RGN_NONE,     1, "driver",   "stack"   },

    { MYNAH_RGN_WORK_MATVEC,       "work.matvec_blocks",     MYNAH_RGN_MULTI,    1, "runtime",  "stack"   },
    { MYNAH_RGN_WORK_ARGMAX,       "work.argmax_blocks",     MYNAH_RGN_MULTI,    1, "runtime",  "stack"   },
    { MYNAH_RGN_WORK_CONV,         "work.conv_panels",       MYNAH_RGN_MULTI,    1, "codec",    "stack"   },
};
static const int g_rgn_n = (int)(sizeof g_rgn / sizeof g_rgn[0]);

/* Two rows sharing an id is not a small defect: every merge adds their times
 * together and prints the sum under whichever name this function reaches
 * first, so one region disappears and the other reports work it never did.
 * MYNAH_RGN_MODEL_LOAD and MYNAH_RGN_DECODE_GANG were both 44 from the day the
 * markers landed until this check existed.  Checked once, from the same
 * constructor that reads the level, and reported to stderr unconditionally:
 * a profiler that cannot name its own rows has nothing to say. */
static void rgn_unique_check(void) {
    for (int i = 0; i < g_rgn_n; ++i) {
        if (g_rgn[i].id <= 0 || g_rgn[i].id >= MYNAH_RGN_MAX) {
            fprintf(stderr, "[COSTMAP] BROKEN TABLE: '%s' has id %d, outside "
                            "1..%d; its time will be dropped\n",
                    g_rgn[i].name, g_rgn[i].id, MYNAH_RGN_MAX - 1);
            continue;
        }
        for (int j = 0; j < i; ++j) {
            if (g_rgn[j].id == g_rgn[i].id) {
                fprintf(stderr, "[COSTMAP] BROKEN TABLE: '%s' and '%s' share id "
                                "%d; their times will be summed and reported "
                                "under one name\n",
                        g_rgn[j].name, g_rgn[i].name, g_rgn[i].id);
            }
        }
    }
}

static const rgn_info *rgn_find(int id) {
    for (int i = 0; i < g_rgn_n; ++i) if (g_rgn[i].id == id) return &g_rgn[i];
    return NULL;
}

const char *mynah_region_name(int id) {
    const rgn_info *r = rgn_find(id);
    return r != NULL ? r->name : "region/unnamed";
}
int mynah_region_parent(int id) {
    const rgn_info *r = rgn_find(id);
    return r != NULL ? r->parent : MYNAH_RGN_NONE;
}
int mynah_region_level(int id) {
    const rgn_info *r = rgn_find(id);
    return r != NULL ? r->level : 1;
}
const char *mynah_region_component(int id) {
    const rgn_info *r = rgn_find(id);
    return r != NULL ? r->component : "other";
}
static const char *rgn_mode(int id) {
    const rgn_info *r = rgn_find(id);
    return r != NULL ? r->mode : "stack";
}

/* ======================================================================
 * Per-thread accumulation
 * ====================================================================== */

#define RGN_STACK_MAX 24
/* One more than src/threads.c's PF_MAX_THREADS (64), plus the caller, the
 * server's scheduler and its writer thread. */
#define RGN_TLS_SLOTS 68

typedef struct {
    uint64_t ns[MYNAH_RGN_MAX];
    uint64_t child_ns[MYNAH_RGN_MAX];
    uint64_t calls[MYNAH_RGN_MAX];
    uint64_t units[MYNAH_RGN_MAX];        /* work units this thread claimed  */
    uint64_t tasks[MYNAH_RGN_MAX];        /* units offered by its dispatches */
    uint64_t dispatches[MYNAH_RGN_MAX];
    uint64_t entered[MYNAH_RGN_MAX];      /* workers that entered the body   */
    uint64_t ticks[MYNAH_RGN_MAX];        /* counted, never timed            */
    uint32_t nest_mismatch[MYNAH_RGN_MAX];
    uint32_t pool_nt[MYNAH_RGN_MAX];      /* widest dispatch seen            */
    int      stack_id[RGN_STACK_MAX];
    uint64_t stack_t0[RGN_STACK_MAX];
    int      depth;
    uint64_t overflow;      /* begins dropped because the stack was full   */
    uint64_t unbalanced;    /* ends that did not match the top of stack    */
    uint64_t leaked;        /* regions closed by an unwind, not by an end  */
    int      in_use;
    char     role[24];
    unsigned long tid;
} rgn_tls;

/* Static, so nothing here ever allocates.  A thread past the limit simply does
 * not record — counted in g_tls_overflow and printed, never silently dropped. */
static rgn_tls          g_tls[RGN_TLS_SLOTS];
static atomic_int       g_tls_next;
static atomic_ullong    g_tls_overflow;
static _Thread_local rgn_tls *t_rgn;

static atomic_ullong    g_requests;
static int              g_count_requests = 1;

static inline uint64_t rgn_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

unsigned long long mynah_costmap_now_ns(void) { return rgn_now_ns(); }

/* Claim this thread's block.  One relaxed fetch_add, once per thread, never on
 * the hot path: every later marker reads the thread-local pointer. */
static rgn_tls *rgn_self(void) {
    rgn_tls *t = t_rgn;
    if (t != NULL) return t;
    const int slot = atomic_fetch_add_explicit(&g_tls_next, 1, memory_order_relaxed);
    if (slot >= RGN_TLS_SLOTS) {
        atomic_fetch_add_explicit(&g_tls_overflow, 1ull, memory_order_relaxed);
        return NULL;
    }
    t = &g_tls[slot];
    t->in_use = 1;
    t->tid = (unsigned long)(uintptr_t)pthread_self();
    if (t->role[0] == 0) snprintf(t->role, sizeof t->role, "%s", "unlabelled");
    t_rgn = t;
    return t;
}

void mynah_costmap_init(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    rgn_unique_check();
    const char *e = getenv("MYNAH_COST_MAP");
    int level = 0;
    if (e != NULL && e[0] != 0 && e[0] != '0') level = (e[0] == '2') ? 2 : 1;
    mynah_costmap_level_v = level;
}

int mynah_costmap_level(void) { return mynah_costmap_level_v; }

void mynah_region_thread_role(const char *role) {
    if (!mynah_costmap_level_v || role == NULL) return;
    rgn_tls *t = rgn_self();
    if (t != NULL) snprintf(t->role, sizeof t->role, "%s", role);
}

/* ---- the markers -------------------------------------------------------- */

void mynah_region_begin_(int id) {
    rgn_tls *t = rgn_self();
    if (t == NULL || id <= 0 || id >= MYNAH_RGN_MAX) return;
    if (t->depth >= RGN_STACK_MAX) { ++t->overflow; return; }
    /* The declared nesting is VERIFIED, not trusted.  A mismatch is recorded
     * and the region is still opened: losing the measurement would hide the
     * very call path that is wrong. */
    const int declared = mynah_region_parent(id);
    if (declared != MYNAH_RGN_MULTI) {
        const int dynamic = t->depth > 0 ? t->stack_id[t->depth - 1] : MYNAH_RGN_NONE;
        /* A DERIVED parent is never on a stack, so "the declared parent is not
         * the dynamic one" is guaranteed for its children and says nothing
         * about the call site.  Such a child matches when it opens at the top
         * of the stack; opening it under some other region still fails.  See
         * costmap.h -- getting this wrong flagged 45 regions per request for
         * two months and invalidated every report under the refusal rule. */
        const int ok = (dynamic == declared) ||
                       (dynamic == MYNAH_RGN_NONE && declared != MYNAH_RGN_NONE &&
                        strcmp(rgn_mode(declared), "derived") == 0);
        if (!ok) ++t->nest_mismatch[id];
    }
    t->stack_id[t->depth] = id;
    t->stack_t0[t->depth] = rgn_now_ns();
    ++t->depth;
}

void mynah_region_end_(int id) {
    rgn_tls *t = t_rgn;
    if (t == NULL || id <= 0 || id >= MYNAH_RGN_MAX) return;
    if (t->depth <= 0 || t->stack_id[t->depth - 1] != id) { ++t->unbalanced; return; }
    const uint64_t dt = rgn_now_ns() - t->stack_t0[t->depth - 1];
    --t->depth;
    t->ns[id] += dt;
    ++t->calls[id];
    if (t->depth > 0) t->child_ns[t->stack_id[t->depth - 1]] += dt;
}

int mynah_region_begin_unique_(int id) {
    rgn_tls *t = rgn_self();
    if (t == NULL) return 0;
    for (int i = 0; i < t->depth; ++i) if (t->stack_id[i] == id) return 0;
    mynah_region_begin_(id);
    return 1;
}

int mynah_region_depth(void) {
    const rgn_tls *t = t_rgn;
    return t != NULL ? t->depth : 0;
}

void mynah_region_unwind(int depth) {
    rgn_tls *t = t_rgn;
    if (t == NULL) return;
    while (t->depth > depth) {
        const int id = t->stack_id[t->depth - 1];
        const uint64_t dt = rgn_now_ns() - t->stack_t0[t->depth - 1];
        --t->depth;
        t->ns[id] += dt;
        ++t->calls[id];
        ++t->leaked;
        if (t->depth > 0) t->child_ns[t->stack_id[t->depth - 1]] += dt;
    }
}

void mynah_region_add_ns(int id, unsigned long long ns) {
    if (!mynah_costmap_level_v) return;
    rgn_tls *t = rgn_self();
    if (t == NULL || id <= 0 || id >= MYNAH_RGN_MAX) return;
    t->ns[id] += ns;
    ++t->calls[id];
}

void mynah_region_pool_at_(int id, int threads, long long tasks) {
    rgn_tls *t = rgn_self();
    if (t == NULL || id <= 0 || id >= MYNAH_RGN_MAX) return;
    ++t->dispatches[id];
    if (tasks > 0) t->tasks[id] += (uint64_t)tasks;
    if (threads > 0 && (uint32_t)threads > t->pool_nt[id]) t->pool_nt[id] = (uint32_t)threads;
}

void mynah_region_units_at_(int id, long long n) {
    rgn_tls *t = rgn_self();
    if (t == NULL || id <= 0 || id >= MYNAH_RGN_MAX || n <= 0) return;
    t->units[id] += (uint64_t)n;
}

void mynah_region_workers_at_(int id, int entered) {
    rgn_tls *t = rgn_self();
    if (t == NULL || id <= 0 || id >= MYNAH_RGN_MAX || entered <= 0) return;
    t->entered[id] += (uint64_t)entered;
}

void mynah_region_tick_at_(int id, long long n) {
    rgn_tls *t = rgn_self();
    if (t == NULL || id <= 0 || id >= MYNAH_RGN_MAX || n <= 0) return;
    t->ticks[id] += (uint64_t)n;
}

void mynah_costmap_count_requests(int on) { g_count_requests = on; }

void mynah_costmap_request_done(void) {
    if (!mynah_costmap_level_v || !g_count_requests) return;
    atomic_fetch_add_explicit(&g_requests, 1ull, memory_order_relaxed);
}

static void tls_clear(rgn_tls *t) {
    memset(t->ns, 0, sizeof t->ns);
    memset(t->child_ns, 0, sizeof t->child_ns);
    memset(t->calls, 0, sizeof t->calls);
    memset(t->units, 0, sizeof t->units);
    memset(t->tasks, 0, sizeof t->tasks);
    memset(t->dispatches, 0, sizeof t->dispatches);
    memset(t->entered, 0, sizeof t->entered);
    memset(t->ticks, 0, sizeof t->ticks);
    memset(t->nest_mismatch, 0, sizeof t->nest_mismatch);
    memset(t->pool_nt, 0, sizeof t->pool_nt);
    t->depth = 0;
    t->overflow = t->unbalanced = t->leaked = 0;
}

void mynah_costmap_after_fork(void) {
    /* Single-threaded by definition: fork() left one thread in the child.  The
     * inherited blocks describe the parent's model load and pre-warm, which
     * this worker did not serve and must not report. */
    const int used = atomic_load_explicit(&g_tls_next, memory_order_relaxed);
    const int n = used < RGN_TLS_SLOTS ? used : RGN_TLS_SLOTS;
    for (int i = 0; i < n; ++i) tls_clear(&g_tls[i]);
    atomic_store_explicit(&g_requests, 0ull, memory_order_relaxed);
}

void mynah_costmap_reset(void) {
    for (int i = 0; i < RGN_TLS_SLOTS; ++i) tls_clear(&g_tls[i]);
    atomic_store_explicit(&g_requests, 0ull, memory_order_relaxed);
    atomic_store_explicit(&g_tls_overflow, 0ull, memory_order_relaxed);
}

/* ======================================================================
 * Merge and report
 * ====================================================================== */

static int tls_used(void) {
    const int used = atomic_load_explicit(&g_tls_next, memory_order_relaxed);
    return used < RGN_TLS_SLOTS ? used : RGN_TLS_SLOTS;
}

static int rgn_active(const rgn_tls *t, int i) {
    return t->calls[i] != 0 || t->units[i] != 0 || t->dispatches[i] != 0 ||
           t->entered[i] != 0 || t->ticks[i] != 0 || t->nest_mismatch[i] != 0;
}

int mynah_costmap_merge(mynah_region_stat *out, int capacity) {
    if (out == NULL || capacity <= 0) return -1;
    const int used = tls_used();
    int n = 0;
    for (int id = 1; id < MYNAH_RGN_MAX && n < capacity; ++id) {
        mynah_region_stat s;
        memset(&s, 0, sizeof s);
        int any = 0;
        for (int k = 0; k < used; ++k) {
            const rgn_tls *t = &g_tls[k];
            if (!t->in_use || !rgn_active(t, id)) continue;
            any = 1;
            ++s.threads_seen;
            s.calls         += t->calls[id];
            s.ns            += t->ns[id];
            s.child_ns      += t->child_ns[id];
            s.nest_mismatch += t->nest_mismatch[id];
            s.units         += t->units[id];
            s.tasks         += t->tasks[id];
            s.dispatches    += t->dispatches[id];
            s.entered       += t->entered[id];
            s.ticks         += t->ticks[id];
            if (t->pool_nt[id] > s.pool_threads) s.pool_threads = t->pool_nt[id];
        }
        if (!any) continue;
        s.id        = id;
        s.name      = mynah_region_name(id);
        s.component = mynah_region_component(id);
        s.parent    = mynah_region_parent(id);
        s.level     = mynah_region_level(id);
        s.mode      = rgn_mode(id);
        out[n++] = s;
    }
    return n;
}

void mynah_costmap_health_get(mynah_costmap_health *out) {
    if (out == NULL) return;
    memset(out, 0, sizeof *out);
    const int used = tls_used();
    for (int k = 0; k < used; ++k) {
        const rgn_tls *t = &g_tls[k];
        if (!t->in_use) continue;
        ++out->threads;
        out->stack_overflow += t->overflow;
        out->unbalanced     += t->unbalanced;
        out->leaked         += t->leaked;
        for (int id = 1; id < MYNAH_RGN_MAX; ++id) out->nest_mismatch += t->nest_mismatch[id];
    }
    out->requests = atomic_load_explicit(&g_requests, memory_order_relaxed);
}

/* Declared depth, for indentation.  Bounded by the table, and MULTI stops it. */
static int rgn_depth_static(int id) {
    int depth = 0;
    int cur = mynah_region_parent(id);
    while (cur > 0 && depth < 8) { ++depth; cur = mynah_region_parent(cur); }
    return depth;
}

/* The denominator for the percentage columns.  request.total when it exists,
 * otherwise the largest root region: stated in the header so no reader has to
 * guess what 100% means. */
static unsigned long long report_base(const mynah_region_stat *st, int n,
                                      const char **label) {
    for (int i = 0; i < n; ++i) {
        if (st[i].id == MYNAH_RGN_REQUEST && st[i].ns > 0) {
            *label = st[i].name;
            return st[i].ns;
        }
    }
    unsigned long long best = 0;
    *label = "none";
    for (int i = 0; i < n; ++i) {
        if (st[i].parent == MYNAH_RGN_NONE && st[i].ns > best) {
            best = st[i].ns;
            *label = st[i].name;
        }
    }
    return best;
}

static double ms(unsigned long long ns) { return (double)ns / 1e6; }

/* Append a space-separated token, never past the end.  Returns the new offset. */
static size_t flags_add(char *buf, size_t cap, size_t off, const char *token) {
    if (off + 1 >= cap) return off;
    const int k = snprintf(buf + off, cap - off, "%s%s", off ? " " : "", token);
    if (k < 0) return off;
    off += (size_t)k;
    return off < cap ? off : cap - 1;
}

int mynah_costmap_report(void *out_file) {
    FILE *f = out_file != NULL ? (FILE *)out_file : stderr;
    mynah_region_stat st[MYNAH_RGN_MAX];
    const int n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
    if (n < 0) return -1;

    mynah_costmap_health h;
    mynah_costmap_health_get(&h);
    const char *base_label = "none";
    const unsigned long long base = report_base(st, n, &base_label);

    fprintf(f, "[COSTMAP] v=1 pid=%d level=%d clock=CLOCK_MONOTONIC "
               "semantics=inclusive self=ns-child_ns threads=%llu requests=%llu "
               "base=%s regions=%d\n",
            (int)getpid(), mynah_costmap_level_v,
            (unsigned long long)h.threads, (unsigned long long)h.requests,
            base_label, n);
    fprintf(f, "  %%base is region time SUMMED OVER THREADS against the "
               "single-thread wall of %s: a region running on N workers can "
               "exceed 100%%, and that is the parallel decomposition, not an "
               "error. occupancy = threads that touched the region / widest "
               "dispatch seen.\n", base_label);
    fprintf(f, "  READ THE FLAGS BEFORE ADDING TWO ROWS. `threads=N` (N>1) "
               "means the row is a SUM OVER N THREADS and shares no clock with "
               "a single-threaded row: it may exceed its own parent's wall, and "
               "subtracting it from anything is meaningless. `derived` means "
               "the row was submitted as a duration by threads that handed the "
               "job to each other, not measured by a begin/end pair on one "
               "stack, so its self time is not a measurement and prints as "
               "'-'. Only rows with neither flag, under one parent, sum.\n");

    if (n == 0) {
        fprintf(f, "  (no regions recorded: no instrumentation reached, or "
                   "MYNAH_COST_MAP was set after the work ran)\n");
        fflush(f);
        return 0;
    }

    fprintf(f, "  %-28s %-8s %10s %10s %7s %7s %-9s %s\n",
            "region", "calls", "incl_ms", "self_ms", "%base", "%self",
            "occupancy", "flags");
    for (int i = 0; i < n; ++i) {
        const mynah_region_stat *s = &st[i];
        const unsigned long long self = s->ns > s->child_ns ? s->ns - s->child_ns : 0;
        char name[40];
        const int indent = rgn_depth_static(s->id);
        snprintf(name, sizeof name, "%*s%s", indent * 2, "", s->name);

        char occ[16];
        if (s->pool_threads > 0) {
            snprintf(occ, sizeof occ, "%u/%u", s->threads_seen, s->pool_threads);
        } else {
            snprintf(occ, sizeof occ, "-");
        }

        char flags[96];
        size_t o = 0;
        flags[0] = 0;
        if (strcmp(s->mode, "derived") == 0) o = flags_add(flags, sizeof flags, o, "derived");
        if (s->nest_mismatch != 0) {
            char b[48];
            snprintf(b, sizeof b, "NEST_MISMATCH=%llu",
                     (unsigned long long)s->nest_mismatch);
            o = flags_add(flags, sizeof flags, o, b);
        }
        if (s->units != 0 || s->tasks != 0) {
            char b[48];
            snprintf(b, sizeof b, "units=%llu/%llu",
                     (unsigned long long)s->units, (unsigned long long)s->tasks);
            o = flags_add(flags, sizeof flags, o, b);
        }
        if (s->ticks != 0) {
            char b[48];
            snprintf(b, sizeof b, "ticks=%llu", (unsigned long long)s->ticks);
            o = flags_add(flags, sizeof flags, o, b);
        }
        /* Rows are printed in id order and indented by declared depth, so a
         * row whose parent is not the row above it reads as a child of
         * whatever happens to precede it -- runtime.admission landed under
         * codec.conv_stack that way. Name the parent whenever the indentation
         * alone would lie. */
        if (s->parent != MYNAH_RGN_NONE && s->parent != MYNAH_RGN_MULTI &&
            (i == 0 || st[i - 1].id != s->parent)) {
            char b[48];
            snprintf(b, sizeof b, "under=%s", mynah_region_name(s->parent));
            o = flags_add(flags, sizeof flags, o, b);
        }
        /* The row crossed threads.  This is the flag that stops a reader from
         * adding two numbers that were never on the same clock. */
        if (s->threads_seen > 1) {
            char b[32];
            snprintf(b, sizeof b, "threads=%u", s->threads_seen);
            o = flags_add(flags, sizeof flags, o, b);
        }
        (void)o;

        /* A derived row's ns was submitted, not bracketed, so no child ever
         * accumulated into its child_ns: ns - child_ns would be the whole
         * inclusive time wearing the name "self".  Print '-' instead. */
        const int derived = strcmp(s->mode, "derived") == 0;
        char self_ms[12], self_pct[12];
        if (derived) {
            snprintf(self_ms, sizeof self_ms, "%10s", "-");
            snprintf(self_pct, sizeof self_pct, "%7s", "-");
        } else {
            snprintf(self_ms, sizeof self_ms, "%10.3f", ms(self));
            snprintf(self_pct, sizeof self_pct, "%6.2f%%",
                     base ? 100.0 * (double)self / (double)base : 0.0);
        }
        fprintf(f, "  %-28s %-8llu %10.3f %s %6.2f%% %s %-9s %s\n",
                name, (unsigned long long)s->calls, ms(s->ns), self_ms,
                base ? 100.0 * (double)s->ns / (double)base : 0.0,
                self_pct, occ, flags);
    }

    fprintf(f, "  health: threads=%llu requests=%llu unbalanced=%llu "
               "leaked=%llu stack_overflow=%llu nest_mismatch=%llu "
               "thread_slots_exhausted=%llu\n",
            (unsigned long long)h.threads, (unsigned long long)h.requests,
            (unsigned long long)h.unbalanced, (unsigned long long)h.leaked,
            (unsigned long long)h.stack_overflow,
            (unsigned long long)h.nest_mismatch,
            (unsigned long long)atomic_load_explicit(&g_tls_overflow,
                                                     memory_order_relaxed));
    if (h.nest_mismatch != 0) {
        fprintf(f, "  WARNING: a region was entered under a parent it does not "
                   "declare. The time is recorded, the attribution is not "
                   "trustworthy: fix the declared parent or the call site.\n");
    }
    if (h.unbalanced != 0 || h.leaked != 0) {
        fprintf(f, "  WARNING: unbalanced/leaked regions. An early return left "
                   "a region open; bracket the entry point with "
                   "mynah_region_depth()/mynah_region_unwind().\n");
    }
    if (h.requests != 0) {
        fprintf(f, "  per request: %.3f ms of %s\n",
                ms(base) / (double)h.requests, base_label);
    }
    fflush(f);
    return n;
}

static void json_str(FILE *j, const char *s) {
    fputc('"', j);
    for (; s != NULL && *s; ++s) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\')      { fputc('\\', j); fputc((int)c, j); }
        else if (c == '\n')             { fputs("\\n", j); }
        else if (c < 0x20)              { fprintf(j, "\\u%04x", c); }
        else                            { fputc((int)c, j); }
    }
    fputc('"', j);
}

int mynah_costmap_report_json(void *out_file) {
    FILE *f = out_file != NULL ? (FILE *)out_file : stderr;
    mynah_region_stat st[MYNAH_RGN_MAX];
    const int n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
    if (n < 0) return -1;
    mynah_costmap_health h;
    mynah_costmap_health_get(&h);

    fprintf(f, "{\n  \"v\": 1,\n  \"pid\": %d,\n  \"level\": %d,\n"
               "  \"clock\": \"CLOCK_MONOTONIC\",\n"
               "  \"semantics\": \"inclusive\",\n"
               "  \"exclusive_rule\": \"self_ns = ns - child_ns\",\n"
               "  \"requests\": %llu,\n  \"threads\": %llu,\n"
               "  \"health\": {\"unbalanced\": %llu, \"leaked\": %llu, "
               "\"stack_overflow\": %llu, \"nest_mismatch\": %llu, "
               "\"thread_slots_exhausted\": %llu},\n",
            (int)getpid(), mynah_costmap_level_v,
            (unsigned long long)h.requests, (unsigned long long)h.threads,
            (unsigned long long)h.unbalanced, (unsigned long long)h.leaked,
            (unsigned long long)h.stack_overflow,
            (unsigned long long)h.nest_mismatch,
            (unsigned long long)atomic_load_explicit(&g_tls_overflow,
                                                     memory_order_relaxed));

    fprintf(f, "  \"regions\": [\n");
    for (int i = 0; i < n; ++i) {
        const mynah_region_stat *s = &st[i];
        const unsigned long long self = s->ns > s->child_ns ? s->ns - s->child_ns : 0;
        fprintf(f, "    {\"id\": %d, \"name\": ", s->id); json_str(f, s->name);
        fprintf(f, ", \"component\": "); json_str(f, s->component);
        fprintf(f, ", \"mode\": ");      json_str(f, s->mode);
        fprintf(f, ", \"parent\": %d, \"level\": %d", s->parent, s->level);
        fprintf(f, ", \"calls\": %llu, \"ns\": %llu, \"child_ns\": %llu, "
                   "\"self_ns\": %llu, \"nest_mismatch\": %llu",
                (unsigned long long)s->calls, (unsigned long long)s->ns,
                (unsigned long long)s->child_ns, self,
                (unsigned long long)s->nest_mismatch);
        fprintf(f, ", \"units\": %llu, \"tasks\": %llu, \"dispatches\": %llu, "
                   "\"entered\": %llu, \"ticks\": %llu, \"pool_threads\": %u, "
                   "\"threads_seen\": %u}%s\n",
                (unsigned long long)s->units, (unsigned long long)s->tasks,
                (unsigned long long)s->dispatches,
                (unsigned long long)s->entered, (unsigned long long)s->ticks,
                s->pool_threads, s->threads_seen, i + 1 < n ? "," : "");
    }
    fprintf(f, "  ],\n  \"threads_detail\": [\n");
    const int used = tls_used();
    int first = 1;
    for (int k = 0; k < used; ++k) {
        const rgn_tls *t = &g_tls[k];
        if (!t->in_use) continue;
        int any = 0;
        for (int id = 1; id < MYNAH_RGN_MAX && !any; ++id) any = rgn_active(t, id);
        if (!any) continue;
        if (!first) fprintf(f, ",\n");
        first = 0;
        fprintf(f, "    {\"role\": "); json_str(f, t->role);
        fprintf(f, ", \"tid\": %lu, \"unbalanced\": %llu, \"leaked\": %llu, "
                   "\"stack_overflow\": %llu, \"regions\": [",
                t->tid, (unsigned long long)t->unbalanced,
                (unsigned long long)t->leaked, (unsigned long long)t->overflow);
        int fr = 1;
        for (int id = 1; id < MYNAH_RGN_MAX; ++id) {
            if (!rgn_active(t, id)) continue;
            fprintf(f, "%s{\"id\": %d, \"calls\": %llu, \"ns\": %llu, "
                       "\"child_ns\": %llu, \"units\": %llu}",
                    fr ? "" : ", ", id, (unsigned long long)t->calls[id],
                    (unsigned long long)t->ns[id],
                    (unsigned long long)t->child_ns[id],
                    (unsigned long long)t->units[id]);
            fr = 0;
        }
        fprintf(f, "]}");
    }
    fprintf(f, "\n  ]\n}\n");
    fflush(f);
    return n;
}

static void expand_pid(char *dst, size_t cap, const char *src) {
    size_t o = 0;
    for (size_t i = 0; src[i] != 0 && o + 1 < cap; ++i) {
        if (src[i] == '%' && src[i + 1] == 'd') {
            const int k = snprintf(dst + o, cap - o, "%d", (int)getpid());
            if (k > 0) o += (size_t)k;
            ++i;
        } else {
            dst[o++] = src[i];
        }
    }
    dst[o < cap ? o : cap - 1] = '\0';
}

int mynah_costmap_dump(const char *path) {
    if (path == NULL || path[0] == 0) path = getenv("MYNAH_COSTMAP_JSON");
    if (path == NULL || path[0] == 0) return 0;
    char real[1024];
    expand_pid(real, sizeof real, path);
    FILE *f = fopen(real, "w");
    if (f == NULL) return -1;
    const int r = mynah_costmap_report_json(f);
    fclose(f);
    return r;
}

/* Resolve the level before main(), so no thread can race the first marker, and
 * make an ordinary CLI run report on the way out without every entry point
 * having to remember to. */
int mynah_costmap_trustworthy(char *reason, size_t capacity) {
    mynah_costmap_health h;
    mynah_costmap_health_get(&h);
    const unsigned long long tls_lost =
        atomic_load_explicit(&g_tls_overflow, memory_order_relaxed);
    const char *why = NULL;
    char buf[256];
    if (h.nest_mismatch != 0) {
        snprintf(buf, sizeof buf,
                 "nest_mismatch=%llu: a region was entered under a parent it "
                 "does not declare, so at least one number is attributed to the "
                 "wrong row", (unsigned long long)h.nest_mismatch);
        why = buf;
    } else if (h.stack_overflow != 0) {
        snprintf(buf, sizeof buf,
                 "stack_overflow=%llu: the region stack was full and begins "
                 "were dropped, so their time was charged to an ancestor",
                 (unsigned long long)h.stack_overflow);
        why = buf;
    } else if (h.unbalanced != 0) {
        snprintf(buf, sizeof buf,
                 "unbalanced=%llu: an end did not match the top of the stack",
                 (unsigned long long)h.unbalanced);
        why = buf;
    } else if (tls_lost != 0) {
        snprintf(buf, sizeof buf,
                 "thread_slots_exhausted=%llu: a thread recorded nothing, so "
                 "every total is short by an unknown amount",
                 (unsigned long long)tls_lost);
        why = buf;
    }
    if (why == NULL) return 0;
    if (reason != NULL && capacity > 0) snprintf(reason, capacity, "%s", why);
    return -1;
}

static void costmap_atexit(void) {
    if (!mynah_costmap_level_v) return;
    mynah_costmap_dump(NULL);
    if (getenv("MYNAH_COSTMAP_JSON") == NULL) mynah_costmap_report(stderr);
    const char *strict = getenv("MYNAH_COST_MAP_STRICT");
    if (strict != NULL && strict[0] != 0 && strict[0] != '0') {
        char reason[256];
        if (mynah_costmap_trustworthy(reason, sizeof reason) != 0) {
            fprintf(stderr,
                    "[COSTMAP] REFUSED: %s.\n"
                    "          MYNAH_COST_MAP_STRICT is set, so this run exits "
                    "non-zero rather than let a number nobody should trust be "
                    "recorded as a measurement.\n", reason);
            fflush(stderr);
            _exit(4);
        }
    }
}

__attribute__((constructor)) static void costmap_ctor(void) {
    mynah_costmap_init();
    if (mynah_costmap_level_v) {
        mynah_region_thread_role("main");
        atexit(costmap_atexit);
    }
}

/* ======================================================================
 * Self-test
 * ====================================================================== */

static int cm_fail(char *error, size_t cap, const char *message) {
    if (error != NULL && cap > 0) snprintf(error, cap, "%s", message);
    return -1;
}

static const mynah_region_stat *cm_find(const mynah_region_stat *st, int n, int id) {
    for (int i = 0; i < n; ++i) if (st[i].id == id) return &st[i];
    return NULL;
}

#define CM_THREAD_ITERS 200

static void *cm_worker(void *arg) {
    const int which = *(const int *)arg;
    mynah_region_thread_role(which == 0 ? "cm_worker_a" : "cm_worker_b");
    for (int i = 0; i < CM_THREAD_ITERS; ++i) {
        mynah_region_begin(MYNAH_RGN_CODEC);
        mynah_region_begin(MYNAH_RGN_CODEC_CONV);
        mynah_region_units_at(MYNAH_RGN_CODEC_CONV, 1);
        mynah_region_end(MYNAH_RGN_CODEC_CONV);
        mynah_region_end(MYNAH_RGN_CODEC);
    }
    return NULL;
}

int mynah_costmap_self_test(char *error, size_t error_capacity) {
    const int saved_level = mynah_costmap_level_v;
    mynah_region_stat st[MYNAH_RGN_MAX];
    int rc = 0;

    mynah_costmap_level_v = 2;
    mynah_costmap_reset();

    /* ---- 1. correct nesting is accepted and inclusive time holds ---------- */
    mynah_region_begin(MYNAH_RGN_REQUEST);
    mynah_region_begin(MYNAH_RGN_PREPARE);
    mynah_region_begin(MYNAH_RGN_TOKENIZE);
    mynah_region_end(MYNAH_RGN_TOKENIZE);
    mynah_region_begin(MYNAH_RGN_ENCODER);
    mynah_region_end(MYNAH_RGN_ENCODER);
    mynah_region_end(MYNAH_RGN_PREPARE);
    mynah_region_end(MYNAH_RGN_REQUEST);

    int n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
    if (n < 4) { rc = cm_fail(error, error_capacity, "costmap: regions missing after a correct nest"); goto done; }
    {
        const mynah_region_stat *req = cm_find(st, n, MYNAH_RGN_REQUEST);
        const mynah_region_stat *pre = cm_find(st, n, MYNAH_RGN_PREPARE);
        const mynah_region_stat *tok = cm_find(st, n, MYNAH_RGN_TOKENIZE);
        if (req == NULL || pre == NULL || tok == NULL) {
            rc = cm_fail(error, error_capacity, "costmap: a nested region was not recorded");
            goto done;
        }
        if (req->calls != 1 || pre->calls != 1 || tok->calls != 1) {
            rc = cm_fail(error, error_capacity, "costmap: wrong call count on a correct nest");
            goto done;
        }
        if (req->nest_mismatch != 0 || pre->nest_mismatch != 0 || tok->nest_mismatch != 0) {
            rc = cm_fail(error, error_capacity, "costmap: correct nesting flagged as a mismatch");
            goto done;
        }
        /* INCLUSIVE: the parent must contain the child, and child_ns must be
         * the child's own inclusive time. */
        if (req->ns < pre->ns || pre->ns < tok->ns) {
            rc = cm_fail(error, error_capacity, "costmap: parent time is smaller than its child");
            goto done;
        }
        if (req->child_ns != pre->ns) {
            rc = cm_fail(error, error_capacity, "costmap: child_ns does not match the child's inclusive time");
            goto done;
        }
    }

    /* ---- 2. WRONG nesting is detected ------------------------------------ */
    mynah_costmap_reset();
    /* codec.conv_stack declares codec.total as its parent; open it under
     * request.total instead. */
    mynah_region_begin(MYNAH_RGN_REQUEST);
    mynah_region_begin(MYNAH_RGN_CODEC_CONV);
    mynah_region_end(MYNAH_RGN_CODEC_CONV);
    mynah_region_end(MYNAH_RGN_REQUEST);
    n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
    {
        const mynah_region_stat *bad = cm_find(st, n, MYNAH_RGN_CODEC_CONV);
        if (bad == NULL || bad->nest_mismatch != 1) {
            rc = cm_fail(error, error_capacity, "costmap: a wrong parent was NOT detected");
            goto done;
        }
        if (bad->calls != 1) {
            rc = cm_fail(error, error_capacity, "costmap: a mismatched region lost its measurement");
            goto done;
        }
        /* A MULTI region must never be flagged. */
        mynah_region_begin(MYNAH_RGN_REQUEST);
        mynah_region_begin(MYNAH_RGN_CODEC);       /* declared MULTI */
        mynah_region_end(MYNAH_RGN_CODEC);
        mynah_region_end(MYNAH_RGN_REQUEST);
        n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
        const mynah_region_stat *multi = cm_find(st, n, MYNAH_RGN_CODEC);
        if (multi == NULL || multi->nest_mismatch != 0) {
            rc = cm_fail(error, error_capacity, "costmap: a MULTI-parent region was flagged as a mismatch");
            goto done;
        }
    }

    /* ---- 3. unbalanced end and unwind ------------------------------------ */
    mynah_costmap_reset();
    mynah_region_begin(MYNAH_RGN_REQUEST);
    mynah_region_end(MYNAH_RGN_STEP);            /* never opened */
    {
        const int depth = mynah_region_depth();
        mynah_region_begin(MYNAH_RGN_STEP);
        mynah_region_begin2(MYNAH_RGN_STEP_BACKBONE);
        mynah_region_unwind(depth);              /* early-return simulation */
        if (mynah_region_depth() != depth) {
            rc = cm_fail(error, error_capacity, "costmap: unwind did not restore the depth");
            goto done;
        }
    }
    mynah_region_end(MYNAH_RGN_REQUEST);
    {
        mynah_costmap_health h;
        mynah_costmap_health_get(&h);
        if (h.unbalanced != 1) {
            rc = cm_fail(error, error_capacity, "costmap: an unmatched end was not counted");
            goto done;
        }
        if (h.leaked != 2) {
            rc = cm_fail(error, error_capacity, "costmap: unwind did not count the leaked regions");
            goto done;
        }
    }

    /* ---- 4. two threads accumulate independently and merge exactly -------- */
    mynah_costmap_reset();
    {
        pthread_t a, b;
        int ia = 0, ib = 1;
        if (pthread_create(&a, NULL, cm_worker, &ia) != 0) {
            rc = cm_fail(error, error_capacity, "costmap: cannot create thread a");
            goto done;
        }
        if (pthread_create(&b, NULL, cm_worker, &ib) != 0) {
            pthread_join(a, NULL);
            rc = cm_fail(error, error_capacity, "costmap: cannot create thread b");
            goto done;
        }
        pthread_join(a, NULL);
        pthread_join(b, NULL);

        n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
        const mynah_region_stat *total = cm_find(st, n, MYNAH_RGN_CODEC);
        const mynah_region_stat *conv  = cm_find(st, n, MYNAH_RGN_CODEC_CONV);
        if (total == NULL || conv == NULL) {
            rc = cm_fail(error, error_capacity, "costmap: the threaded regions were not recorded");
            goto done;
        }
        if (total->calls != 2u * CM_THREAD_ITERS || conv->calls != 2u * CM_THREAD_ITERS) {
            rc = cm_fail(error, error_capacity, "costmap: merged call count is wrong across two threads");
            goto done;
        }
        if (total->threads_seen != 2 || conv->threads_seen != 2) {
            rc = cm_fail(error, error_capacity, "costmap: the two threads did not get separate blocks");
            goto done;
        }
        if (conv->units != 2u * CM_THREAD_ITERS) {
            rc = cm_fail(error, error_capacity, "costmap: unit accounting lost work across threads");
            goto done;
        }
        if (total->ns < conv->ns || total->child_ns != conv->ns) {
            rc = cm_fail(error, error_capacity, "costmap: inclusive accounting broke across threads");
            goto done;
        }
        if (conv->nest_mismatch != 0) {
            rc = cm_fail(error, error_capacity, "costmap: a worker's correct nesting was flagged");
            goto done;
        }
    }

    /* ---- 5. off means off ------------------------------------------------ */
    mynah_costmap_reset();
    mynah_costmap_level_v = 0;
    mynah_region_begin(MYNAH_RGN_REQUEST);
    mynah_region_end(MYNAH_RGN_REQUEST);
    mynah_costmap_level_v = 2;
    n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
    if (n != 0) {
        rc = cm_fail(error, error_capacity, "costmap: a marker recorded while the map was off");
        goto done;
    }

    /* ---- 6. level 2 markers stay off at level 1 --------------------------- */
    mynah_costmap_level_v = 1;
    mynah_region_begin(MYNAH_RGN_STEP);
    mynah_region_begin2(MYNAH_RGN_STEP_BACKBONE);
    mynah_region_end2(MYNAH_RGN_STEP_BACKBONE);
    mynah_region_end(MYNAH_RGN_STEP);
    n = mynah_costmap_merge(st, MYNAH_RGN_MAX);
    if (cm_find(st, n, MYNAH_RGN_STEP) == NULL) {
        rc = cm_fail(error, error_capacity, "costmap: a level-1 region was dropped at level 1");
        goto done;
    }
    if (cm_find(st, n, MYNAH_RGN_STEP_BACKBONE) != NULL) {
        rc = cm_fail(error, error_capacity, "costmap: a level-2 region recorded at level 1");
        goto done;
    }

done:
    mynah_costmap_reset();
    mynah_costmap_level_v = saved_level;
    return rc;
}
