/* costmap.h — the region profiler: where the wall time of one synthesis goes.
 *
 * The semantics below were fixed BEFORE any number was collected, which is the
 * only reason numbers from the driver, the engine, the codec and the server can
 * be put in the same table at all.  Ported from qwen-tts, where the same file
 * answered "the conv decoder is ~40% of wall" — a fact that redirected a whole
 * epic away from optimizing the transformer first.
 *
 * SEMANTICS
 *
 *   INCLUSIVE.  A region's time contains every region entered inside it.
 *   Exclusive ("self") time is DERIVED at report time as ns - child_ns and is
 *   never measured separately, so the two can never be silently mixed.
 *
 *   NESTING IS DECLARED AND VERIFIED.  Each region declares its parent
 *   statically in the table in costmap.c.  A begin whose dynamic parent is not
 *   the declared one increments nest_mismatch for that region and is otherwise
 *   accepted: the report shows the mismatch instead of quietly attributing the
 *   time to the wrong place.  MYNAH_RGN_MULTI declares a region that
 *   legitimately has several parents (the pool dispatch runs under all of them).
 *
 *   A DERIVED PARENT IS SATISFIED BY THE TOP OF THE STACK.  A region whose
 *   mode is "derived" is never on any thread's region stack -- that is what
 *   derived means -- so requiring it to be the dynamic parent of its declared
 *   children is a contradiction in the taxonomy, not a finding about the code.
 *   A child of a derived region therefore matches when it opens at the top of
 *   the stack.  This is not a loosening: the check still fails if such a child
 *   opens underneath some UNRELATED region.
 *
 *   This rule was written after the fact and it cost something to learn.  From
 *   the day the first markers landed until 2026-09 the pocket path reported
 *   nest_mismatch=45 on a clean eleven-step synthesis -- request.total was
 *   declared "stack" while the only thing that ever wrote it was
 *   mynah_region_add_ns(), so step.total, step.emit and request.prepare were
 *   each flagged on every single call.  Under the refusal in
 *   .work/engineering-method.md 4 a non-zero nest_mismatch invalidates the
 *   whole report, so the cost map spent two months rejecting its own numbers
 *   for a reason that was in this table rather than in the engine.
 *
 *   ACCUMULATION IS THREAD-LOCAL.  No atomic read-modify-write on the hot path
 *   and no allocation inside a region: each thread's block is calloc'd once, on
 *   its first marker, and linked into a global list under a mutex that one
 *   thread touches exactly once.  Merging happens only at dump time.
 *
 *   CLOCK_MONOTONIC, nanoseconds, one clock read per begin and one per end.
 *
 *   IDS ARE APPEND-ONLY.  A report produced by an older binary must keep its
 *   meaning.  Never renumber; append, and leave gaps where a family may grow.
 *
 * LEVELS (env MYNAH_COST_MAP)
 *
 *   0 / unset   off.  One relaxed load of an int and a predictable branch per
 *               marker; nothing else runs.
 *   1           macro regions: request, prepare, per-AR-step, codec decode,
 *               streaming emit, server admission.  Nothing inside a per-layer
 *               or per-stream loop.
 *   2           adds the fine regions: the phases inside one AR step, the
 *               per-stream local-transformer iteration, the codec sub-stages.
 *               These sit in hot loops (never in an inner kernel) and are
 *               opt-in precisely so level 1 stays cheap enough to leave on.
 *
 * OUTPUT
 *
 *   A human table on stderr at exit when MYNAH_COST_MAP is set, and/or JSON to
 *   MYNAH_COSTMAP_JSON ("%d" in the path becomes the pid, so a forked server
 *   writes one file per worker).
 *
 * REGIONS THAT CROSS THREADS.  `threads_seen` is carried per region and the
 * report prints it, because the single most available way to misread this
 * table is to add two rows that were never on the same thread.  A row marked
 * `threads=N` (N > 1) is a SUM OVER THREADS: it may exceed the wall clock of
 * its own parent, and subtracting it from a single-threaded row is meaningless.
 * The report says so in the header rather than trusting the reader to know.
 *
 * INSTRUMENTATION.  E4-2b placed the markers along the PocketTTS path:
 * src/inference.c carries the driver and placement regions, src/engine_pocket.c
 * the prepare / prefill / step / flow / codec ones.  Magpie's own graph is not
 * instrumented and its rows stay absent rather than being faked from a
 * neighbouring engine's numbers.
 */
#ifndef MYNAH_TTS_COSTMAP_H
#define MYNAH_TTS_COSTMAP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MYNAH_RGN_NONE = 0,

    /* ---- one request, driver level (src/inference.c) --------------------- */
    MYNAH_RGN_REQUEST = 1,       /* synthesize_slots(): whole batch of slots   */
    MYNAH_RGN_PREPARE,           /* slot_prepare() / engine prepare()          */
    MYNAH_RGN_TOKENIZE,          /* text -> token ids (tokenizer*.c)           */
    MYNAH_RGN_ENCODER,           /* text/context encoder stack                 */
    MYNAH_RGN_PREFILL,           /* decoder KV prefill over the conditioning   */
    MYNAH_RGN_FINALIZE,          /* slot_finalize(): trailing decode + trim    */
    /* The four linear projections OF THE PREFILL, and nothing else: the same
     * hook serves the per-step path and the codec transformer, and this row
     * counts only the calls that ran inside mynah_transformer_ar_prefill.  It
     * is what splits the prefill into the half a thread pool can divide (these)
     * and the half it cannot (attention scores, RoPE, the KV writes and the
     * elementwise stack, which are `prep.decoder_prefill`'s self time). */
    MYNAH_RGN_PREFILL_PROJ = 7,

    /* ---- one autoregressive step (engine step_batch/emit_batch) ---------- */
    MYNAH_RGN_STEP = 10,         /* step_batch(): one AR step, whole batch     */
    MYNAH_RGN_STEP_EMBED,        /* frame/token embedding + conditioning  (L2) */
    MYNAH_RGN_STEP_BACKBONE,     /* the decoder transformer stack         (L2) */
    MYNAH_RGN_STEP_ATTENTION,    /* self + cross attention inside it      (L2) */
    MYNAH_RGN_STEP_FFN,          /* the feed-forward half of the stack    (L2) */
    MYNAH_RGN_STEP_HEAD,         /* head projection + argmax / sampling        */
    MYNAH_RGN_EMIT = 18,         /* emit_batch(): frames appended + EOS        */

    /* ---- local transformer / depth head (src/engine_magpie.c) ------------ */
    MYNAH_RGN_LOCAL = 20,        /* the whole local transformer for one step   */
    MYNAH_RGN_LOCAL_STEP,        /* one stacked-stream iteration          (L2) */
    MYNAH_RGN_LOCAL_PROJ,        /* per-stream out projection + embed     (L2) */

    /* ---- continuous-latent head (src/engine_pocket.c) ---------------------
     * PocketTTS's second weight-bound stage.  It is NOT MYNAH_RGN_LOCAL: a
     * flow head and a local transformer are different graphs, and reporting
     * one under the other's name is the kind of borrowed label that makes a
     * table agree with a story.  Until 2026-09 the pocket call site did
     * exactly that, which is why this id exists.                            */
    MYNAH_RGN_FLOW = 24,         /* mynah_flow_head_forward[_batch]            */

    /* ---- codec (src/codec_nanocodec.c, src/seanet.c, src/conv1d.c) ------- */
    MYNAH_RGN_CODEC = 28,        /* decode_audio(): frames -> PCM              */
    MYNAH_RGN_CODEC_EMBED,       /* codebook lookup / latent projection   (L2) */
    MYNAH_RGN_CODEC_TRANSFORMER, /* the codec's own transformer, if any   (L2) */
    MYNAH_RGN_CODEC_CONV,        /* SEANet / upsampling conv stack        (L2) */
    MYNAH_RGN_CODEC_POST,        /* final conv, windowing, overlap trim   (L2) */

    /* ---- streaming (src/inference.c emit_stream_samples, server/) -------- */
    MYNAH_RGN_STREAM_EMIT = 36,  /* the audio callback: the CONSUMER's time,
                                  * inside the synthesis loop. It is here so a
                                  * slow sink stops looking like slow decode.  */

    /* ---- runtime / server ------------------------------------------------ */
    MYNAH_RGN_RT_REQUEST = 40,   /* accept -> response complete      (derived) */
    MYNAH_RGN_RT_ADMISSION,      /* enqueue -> admitted into a batch (derived) */
    MYNAH_RGN_RT_PARALLEL,       /* mynah_parallel_for dispatch        (MULTI) */
    MYNAH_RGN_RT_PARALLEL_WAIT,  /* caller done, waiting for the workers       */
    /* 52, NOT 44.  It was written without a value under RT_PARALLEL_WAIT=43,
     * which made it 44 -- the value MYNAH_RGN_DECODE_GANG declares explicitly
     * a few lines further down.  Two names, one slot: every merge added
     * the decode gang's time to the model load's and printed the sum under
     * whichever name rgn_find() reached first, so `driver.decode_gang` has
     * never appeared in a report and `runtime.model_load` has never been only
     * a model load.  Renumbering is what the append-only rule forbids, and it
     * is also the only repair: an id that two regions share does not carry a
     * meaning for an old report to keep.  `rgn_unique_check()` in costmap.c now
     * refuses the table at startup so a third region cannot land on a fourth's
     * id. */
    MYNAH_RGN_MODEL_LOAD = 52,   /* mmap + weight resolve, once per process    */
    /* Materialising the quantized weight cache, eagerly, at engine init -- the
     * top of serve(), before any slot is prepared.  It is the same work
     * `cache_insert` used to do lazily inside whichever request touched a
     * tensor first, which for the backbone is the first text prefill.  Having
     * it here is the point: a cost that belongs to loading a model must be
     * reported against loading a model and not against a decoder phase.     */
    MYNAH_RGN_PREPACK = 53,

    /* ---- driver placement (src/inference.c) ------------------------------
     * WHERE the codec decode ran, which is a scheduling fact and never a
     * numerical one.  The gang and the lane are alternatives, so a report in
     * which both are non-zero is itself the finding.                        */
    MYNAH_RGN_DECODE_GANG = 44,  /* stream_gang(): one wide decode call        */
    MYNAH_RGN_LANE_WAIT,         /* lane_reap(blocking): THIS slot waiting     */
    MYNAH_RGN_LANE_DECODE,       /* lane_decode(): the decode, on a lane thread*/

    /* ---- parallel work decomposition (MULTI parents) ---------------------
     * Wall time alone cannot say a region ran on two of six workers. These
     * carry the decomposition itself; see mynah_region_units_at below.      */
    MYNAH_RGN_WORK_MATVEC = 48,  /* qmat row blocks claimed per worker         */
    MYNAH_RGN_WORK_ARGMAX,       /* argmax row blocks claimed per worker       */
    MYNAH_RGN_WORK_CONV,         /* codec conv panels claimed per worker       */

    MYNAH_RGN_MAX = 56
};

/* Declared parent of a region that legitimately has several. */
#define MYNAH_RGN_MULTI (-1)

/* 0 = off, 1 = macro, 2 = macro + fine.  Read directly by the inline markers,
 * which is why it is a plain int and not behind a function call. */
extern int mynah_costmap_level_v;

/* Reads MYNAH_COST_MAP once.  Called from a constructor, so a marker reached
 * before main() still sees the right level; calling it again is harmless. */
void mynah_costmap_init(void);
int  mynah_costmap_level(void);

void mynah_region_begin_(int id);
void mynah_region_end_(int id);

static inline void mynah_region_begin(int id) {
    if (mynah_costmap_level_v) mynah_region_begin_(id);
}
static inline void mynah_region_end(int id) {
    if (mynah_costmap_level_v) mynah_region_end_(id);
}
/* Fine (level 2) markers are separate entry points so a level-1 run never even
 * reaches the call: that is what keeps the per-layer sites off level 1's cost. */
static inline void mynah_region_begin2(int id) {
    if (mynah_costmap_level_v > 1) mynah_region_begin_(id);
}
static inline void mynah_region_end2(int id) {
    if (mynah_costmap_level_v > 1) mynah_region_end_(id);
}

/* Open `id` only if it is not already open on this thread; returns 1 when it
 * opened, and only then must the caller end it.  Needed wherever a public
 * entry point may delegate to another one that opens the same region — the
 * total would otherwise be counted twice. */
int mynah_region_begin_unique_(int id);
static inline int mynah_region_begin_unique(int id) {
    return mynah_costmap_level_v ? mynah_region_begin_unique_(id) : 0;
}

/* Depth of this thread's region stack, and "close everything above `depth`".
 * An entry point records the depth just after opening its own region and
 * unwinds to it on every exit path, so the error returns scattered through a
 * decoder body cannot leave a region open forever.  Whatever is unwound is
 * counted as `leaked` and appears in the report rather than skewing it. */
int  mynah_region_depth(void);
void mynah_region_unwind(int depth);

/* Add a duration that was NOT measured by a begin/end pair on one thread.  The
 * server's request lifecycle is recorded as timestamps by threads that hand the
 * job to each other, so no thread-local stack can bracket it.  Those regions
 * are declared mode="derived" in the table and marked as such in the report, so
 * nobody reads them as if they had been measured the same way as the rest. */
void mynah_region_add_ns(int id, unsigned long long ns);

/* The same clock the regions use, for a caller that accumulates a phase itself
 * and submits it once. */
unsigned long long mynah_costmap_now_ns(void);

/* ---- pool occupancy ------------------------------------------------------
 *
 *   mynah_region_pool_at(id, threads, tasks)  the dispatch: workers asked for,
 *                                             units offered
 *   mynah_region_units_at(id, n)              a worker claiming n units
 *   mynah_region_workers_at(id, entered)      workers that actually entered the
 *                                             job body, counted by the job
 *
 * The id is explicit because a pool worker runs on its own thread with its own
 * region stack: it is not "inside" the caller's region and cannot infer the
 * attribution.  Occupancy is derived at report time as (threads that claimed at
 * least one unit) / (threads the dispatch asked for).  Counting entries rather
 * than inferring them from which threads happened to touch a marker matters: a
 * thread that never reaches a marker leaves no record, and turning that silence
 * into an underfill claim would be a lie. */
void mynah_region_pool_at_(int id, int threads, long long tasks);
void mynah_region_units_at_(int id, long long n);
void mynah_region_workers_at_(int id, int entered);
/* Count an event WITHOUT reading the clock, where the event is frequent but its
 * duration already sits inside a coarser region. */
void mynah_region_tick_at_(int id, long long n);

static inline void mynah_region_pool_at(int id, int threads, long long tasks) {
    if (mynah_costmap_level_v) mynah_region_pool_at_(id, threads, tasks);
}
static inline void mynah_region_units_at(int id, long long n) {
    if (mynah_costmap_level_v) mynah_region_units_at_(id, n);
}
static inline void mynah_region_workers_at(int id, int entered) {
    if (mynah_costmap_level_v) mynah_region_workers_at_(id, entered);
}
static inline void mynah_region_tick_at(int id, long long n) {
    if (mynah_costmap_level_v) mynah_region_tick_at_(id, n);
}
/* Level-2 variants: per-row-block accounting is a micro-event and belongs to
 * the deep run, while level 1 keeps only the coarse dispatch summary. */
static inline void mynah_region_units_at2(int id, long long n) {
    if (mynah_costmap_level_v > 1) mynah_region_units_at_(id, n);
}
static inline void mynah_region_pool_at2(int id, int threads, long long tasks) {
    if (mynah_costmap_level_v > 1) mynah_region_pool_at_(id, threads, tasks);
}

/* Label the calling thread ("main", "worker", "scheduler", "writer", ...).
 * Purely descriptive: regions are accumulated per OS thread regardless. */
void mynah_region_thread_role(const char *role);

/* One completed request, so the report can print ms/request. */
void mynah_costmap_request_done(void);
/* Stop counting completed requests for a while — a server pre-warm runs a full
 * synthesis that nobody asked for, and counting it dilutes every ms/request. */
void mynah_costmap_count_requests(int on);

/* Drop everything accumulated before a fork: a prefork server loads the model
 * and pre-warms in the parent, and merging N workers would otherwise count that
 * work N times. */
void mynah_costmap_after_fork(void);

/* Reset every counter, including the thread blocks of threads that have exited.
 * For tests, and for a benchmark that wants the warmup out of its numbers. */
void mynah_costmap_reset(void);

/* ---- reporting ---------------------------------------------------------- */

/* Human table, merged across threads.  `out_file` is a FILE* (NULL = stderr).
 * Returns the number of regions printed, or -1. */
int mynah_costmap_report(void *out_file);
/* Same data as JSON, one object per thread plus a merged section. */
int mynah_costmap_report_json(void *out_file);
/* JSON to `path`; "%d" becomes the pid.  NULL uses MYNAH_COSTMAP_JSON and is a
 * no-op when that is unset. */
int mynah_costmap_dump(const char *path);

/* ---- merged read-back, for tests and for a caller that wants the numbers -- */
typedef struct {
    int                id;
    const char        *name;
    const char        *component;
    int                parent;
    int                level;
    const char        *mode;          /* "stack" or "derived"                 */
    unsigned long long calls;
    unsigned long long ns;            /* INCLUSIVE                            */
    unsigned long long child_ns;      /* self_ns = ns - child_ns              */
    unsigned long long nest_mismatch;
    unsigned long long units;
    unsigned long long tasks;
    unsigned long long dispatches;
    unsigned long long entered;
    unsigned long long ticks;
    unsigned           pool_threads;  /* widest dispatch seen                 */
    unsigned           threads_seen;  /* threads that touched this region     */
} mynah_region_stat;

/* Merge every thread's block into `out`, one entry per region that has any
 * activity.  Returns the number written, or -1. */
int mynah_costmap_merge(mynah_region_stat *out, int capacity);

/* Totals that are not per-region: use for a health line. */
typedef struct {
    unsigned long long requests;
    unsigned long long threads;
    unsigned long long stack_overflow;  /* begins dropped, stack was full     */
    unsigned long long unbalanced;      /* ends that did not match the top    */
    unsigned long long leaked;          /* regions closed by an unwind        */
    unsigned long long nest_mismatch;   /* declared parent != dynamic parent  */
} mynah_costmap_health;
void mynah_costmap_health_get(mynah_costmap_health *out);

/* Static taxonomy. */
const char *mynah_region_name(int id);
int         mynah_region_parent(int id);
int         mynah_region_level(int id);
const char *mynah_region_component(int id);

/* ---- the cost map's refusal ---------------------------------------------
 *
 * .work/engineering-method.md 4: "the cost map rejects its own numbers if
 * nest_mismatch != 0, if the region stack overflowed, or if there are
 * unbalanced ends."  This is that rule as a call, so every consumer refuses
 * for the same reasons rather than each inventing its own tolerance.
 *
 * Returns 0 when the table may be read, or -1 after writing why it may not.
 * `leaked` is deliberately NOT fatal: an unwind is the designed response to an
 * error return, the time is still attributed to the region that was open, and
 * the count is printed.  A mismatch, an overflow or an unbalanced end are
 * different -- each of them means a number is attributed to the wrong row.
 *
 * MYNAH_COST_MAP_STRICT=1 makes the process exit non-zero when this refuses,
 * so a gate script cannot record a run whose profile was not trustworthy. */
int mynah_costmap_trustworthy(char *reason, size_t capacity);

/* Model-free check: nesting accepted, bad nesting DETECTED, unbalanced end
 * detected, unwind accounting, and two threads accumulating independently.
 * 0 = ok, -1 = error.  It needs an empty map to assert exact counts, so it
 * calls mynah_costmap_reset() on entry and on exit and restores the level:
 * run it from --self-test, never in the middle of a profiled synthesis. */
int mynah_costmap_self_test(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif /* MYNAH_TTS_COSTMAP_H */
