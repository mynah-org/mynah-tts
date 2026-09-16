/*
 * Shared causal autoregressive transformer.  See transformer_ar.h for the
 * contract, the KV-cache layout and how the NaN sentinel is handled.
 */
#include "transformer_ar.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"

/* ------------------------------------------------------------------ utils */

static void tar_set_error(char *error, size_t capacity, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static void tar_set_error(char *error, size_t capacity, const char *format,
                          ...) {
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    (void)vsnprintf(error, capacity, format, args);
    va_end(args);
}

/* Checked size_t arithmetic: every allocation size goes through these. */
static int tar_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > (size_t)-1 / a) return -1;
    *out = a * b;
    return 0;
}

static int tar_add(size_t a, size_t b, size_t *out) {
    if (b > (size_t)-1 - a) return -1;
    *out = a + b;
    return 0;
}

static int tar_finite(const float *values, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- kernels */

void mynah_transformer_ar_rope_angles_f32(float *cosines, float *sines,
                                          size_t half, size_t position,
                                          float max_period) {
    if (cosines == NULL || sines == NULL || half == 0) return;
    /* The reference builds `freqs` from a float32 arange times a python-float
     * scalar, so the whole table is float32 arithmetic.  Reproduce that rather
     * than computing in double and rounding: the angle at position ~150 is
     * large enough that a 1e-7 relative drift in `freqs` is visible in cos(). */
    const float slope =
        (float)(-log((double)max_period) * 2.0 / (double)(2u * half));
    const float t = (float)position;
    for (size_t i = 0; i < half; ++i) {
        const float angle = expf((float)i * slope) * t;
        cosines[i] = cosf(angle);
        sines[i] = sinf(angle);
    }
}

void mynah_transformer_ar_rope_apply_f32(float *values, size_t num_heads,
                                         size_t head_dim, const float *cosines,
                                         const float *sines) {
    if (values == NULL || cosines == NULL || sines == NULL) return;
    const size_t half = head_dim / 2u;
    for (size_t h = 0; h < num_heads; ++h) {
        float *row = values + h * head_dim;
        for (size_t i = 0; i < half; ++i) {
            const float re = row[2u * i];
            const float im = row[2u * i + 1u];
            const float c = cosines[i];
            const float s = sines[i];
            row[2u * i] = re * c - im * s;
            row[2u * i + 1u] = re * s + im * c;
        }
    }
}

/* ------------------------------------------------------------------ state */

/* How many positions one prefill tile covers.
 *
 * It is a tile rather than "the whole prefill" because the scratch is sized at
 * `_state_new` and every live request owns one: an unbounded tile would make
 * the per-request footprint a function of the longest text the process ever
 * sees.  Sixteen is the Mimi decoder's own stride, so the codec prefill -- the
 * one that runs every frame -- is exactly one tile and never splits. */
#define TAR_PREFILL_TILE 16u

size_t mynah_transformer_ar_prefill_tile(void) { return TAR_PREFILL_TILE; }

/* Scratch for a set of rows pushed through the stack together.  Each state owns
 * one (the prefill tile, which the single `_step` also uses as a one-row tile),
 * and each `mynah_transformer_ar_batch` owns one (a cross-request step). */
typedef struct {
    size_t rows_cap;
    float *block; /* one owned allocation backing everything below */
    float *x;     /* [rows][d_model]    residual stream */
    float *norm;  /* [rows][d_model]    */
    float *upd;   /* [rows][d_model]    */
    float *qkv;   /* [rows][3*attn_dim] */
    float *attn;  /* [rows][attn_dim]   */
    float *ffn;   /* [rows][ffn_dim]    */
    float *gelu;  /* [rows][ffn_dim]    vForce scratch for the GELU */
    /* Rebuilt per call, never allocated in the loop: the row-pointer view the
     * cross-request hook takes. */
    const float **in_ptr;
    float **out_ptr;
} tar_rows;

/* One row of a pass: whose KV cache it writes, and at which absolute position.
 * A prefill tile is `rows` refs to one state at consecutive positions; a
 * cross-request step is one ref per state, each at its own position. */
typedef struct {
    mynah_transformer_ar_state *state;
    size_t position;
} tar_row_ref;

/* ------------------------------------------------- the shared RoPE tables
 *
 * cos/sin for a position are a pure function of (position, head_dim,
 * max_period) -- no weight and no request enters them -- so every state with
 * the same config was building and holding an identical copy.  On the pinned
 * pack the codec transformer's table is 24016 positions x 32 halves x 2 x 4
 * bytes = 6.1 MB, and there is one live context per slot per worker: at the
 * shipping topology that is the same 6.1 MB dozens of times over, and after
 * the windowed KV cache above it was the LARGEST per-context allocation left.
 *
 * Shared, immutable, and keyed on exactly what it is a function of.  Built
 * under the lock and published whole, so a reader that holds the pointer needs
 * no lock -- the same contract as qmat's weight cache and seanet's tap memo.
 *
 * Freed at exit rather than never, because `make leaks` is a gate here.  Not
 * refcounted: the set of distinct (positions, half, period) triples in a
 * process is the set of transformers the model has, which is two. */
typedef struct {
    size_t positions, half;
    float max_period;
    float *cos_sin;   /* [positions][half] cos, then [positions][half] sin */
} tar_rope_entry;

static struct {
    tar_rope_entry *entries;
    size_t count, capacity;
    pthread_mutex_t mutex;
    int atexit_registered;
} g_tar_rope = { NULL, 0, 0, PTHREAD_MUTEX_INITIALIZER, 0 };

static void tar_rope_release(void) {
    pthread_mutex_lock(&g_tar_rope.mutex);
    for (size_t i = 0; i < g_tar_rope.count; ++i) free(g_tar_rope.entries[i].cos_sin);
    free(g_tar_rope.entries);
    g_tar_rope.entries = NULL;
    g_tar_rope.count = g_tar_rope.capacity = 0;
    pthread_mutex_unlock(&g_tar_rope.mutex);
}

/* Returns the cos table; the sin table follows it at `positions * half`.
 * NULL means the caller keeps its own copy, so a failure here is memory
 * spent and never a different number. */
static const float *tar_rope_shared(size_t positions, size_t half,
                                    float max_period) {
    if (positions == 0u || half == 0u) return NULL;
    pthread_mutex_lock(&g_tar_rope.mutex);
    for (size_t i = 0; i < g_tar_rope.count; ++i) {
        const tar_rope_entry *e = &g_tar_rope.entries[i];
        if (e->positions == positions && e->half == half &&
            memcmp(&e->max_period, &max_period, sizeof max_period) == 0) {
            const float *p = e->cos_sin;
            pthread_mutex_unlock(&g_tar_rope.mutex);
            return p;
        }
    }
    if (g_tar_rope.count == g_tar_rope.capacity) {
        const size_t cap = (g_tar_rope.capacity == 0u) ? 4u : g_tar_rope.capacity * 2u;
        tar_rope_entry *grown =
            (tar_rope_entry *)realloc(g_tar_rope.entries, cap * sizeof(*grown));
        if (grown == NULL) { pthread_mutex_unlock(&g_tar_rope.mutex); return NULL; }
        g_tar_rope.entries = grown;
        g_tar_rope.capacity = cap;
    }
    size_t floats = 0;
    if (tar_mul(positions, half, &floats) != 0 ||
        tar_mul(floats, 2u, &floats) != 0) {
        pthread_mutex_unlock(&g_tar_rope.mutex);
        return NULL;
    }
    float *table = (float *)calloc(floats, sizeof(float));
    if (table == NULL) { pthread_mutex_unlock(&g_tar_rope.mutex); return NULL; }
    for (size_t p = 0; p < positions; ++p) {
        mynah_transformer_ar_rope_angles_f32(table + p * half,
                                             table + positions * half + p * half,
                                             half, p, max_period);
    }
    g_tar_rope.entries[g_tar_rope.count++] =
        (tar_rope_entry){ positions, half, max_period, table };
    if (!g_tar_rope.atexit_registered) {
        g_tar_rope.atexit_registered = 1;
        (void)atexit(tar_rope_release);
    }
    pthread_mutex_unlock(&g_tar_rope.mutex);
    return table;
}

/* TEST HOOK, not a runtime knob.  Forces the windowed cache off (0) or on (1)
 * for states created after the call, or restores the resolution (-1).
 *
 * WHY IT IS PUBLIC: a windowed cache has to produce BIT-IDENTICAL output to
 * the full-length one, or a request's audio would depend on how much memory
 * the process felt like using.  Proving that needs both caches in ONE process
 * over the same weights and the same inputs, and a compile-time or
 * config-derived choice can only give one of them per run -- the same argument
 * as mynah_qmat_i8mm_force(). */
static int g_tar_kv_window_force = -1;

int mynah_transformer_ar_kv_window_force(int mode) {
    const int before = g_tar_kv_window_force;
    g_tar_kv_window_force = (mode < 0) ? -1 : (mode != 0);
    return before;
}

struct mynah_transformer_ar_state {
    mynah_transformer_ar_config config;
    size_t attn_dim; /* num_heads * head_dim */
    size_t half;     /* head_dim / 2         */
    size_t offset;   /* cached positions, i.e. the next absolute position */

    float *kv;       /* [num_layers][2][kv_positions][attn_dim] */
    size_t kv_half;  /* kv_positions * attn_dim                 */
    size_t kv_layer; /* 2 * kv_half                             */
    /* THE CACHE IS NOT ALWAYS max_seq_len LONG.
     *
     * A windowed transformer -- `context > 0`, which is Mimi's decoder at 250
     * -- can never read further back than `context` positions, but the cache
     * was still allocated for every position the utterance could reach: on the
     * pinned pack that is 24016 positions of a 250-position window, 196 MB of
     * address space per request where 2 MB is reachable.  It never showed up
     * as resident because calloc faults on use, which is exactly why it
     * survived: the allocation is wrong, the RSS is not.
     *
     * So a windowed cache holds `context` plus slack, and slot 0 is the
     * absolute position `kv_base` rather than 0.  When a write would run past
     * the end, the still-reachable tail is moved down and `kv_base` advances:
     * the attention loop keeps reading ONE CONTIGUOUS SPAN, which is why this
     * is a compaction and not a ring buffer -- a ring would put a branch and a
     * wrap in the hottest loop in the codec to save an amortised memmove of
     * one position per step.
     *
     * `context == 0` (the backbone, and every voice-prefix path) keeps
     * kv_positions == max_seq_len and kv_base == 0, so nothing there moves. */
    size_t kv_positions; /* capacity in positions                       */
    size_t kv_base;      /* absolute position held in slot 0            */

    tar_rows rows;      /* the prefill tile; row 0 is also the single step  */
    tar_row_ref *refs;  /* [rows.rows_cap]                                  */

    float *block;     /* scores, and the RoPE tables when they are not shared */
    float *scores;    /* [kv_positions] */
    /* Shared when tar_rope_shared() could build them, owned inside `block`
     * when it could not; either way immutable after construction. */
    const float *rope_cos;  /* [max_seq_len][half]  */
    const float *rope_sin;  /* [max_seq_len][half]  */
};

struct mynah_transformer_ar_batch {
    mynah_transformer_ar_config config;
    size_t attn_dim;
    tar_rows rows;
    tar_row_ref *refs; /* [rows.rows_cap] */
};

static void tar_rows_release(tar_rows *rows) {
    if (rows == NULL) return;
    free(rows->block);
    free(rows->in_ptr);
    free((void *)rows->out_ptr);
    memset(rows, 0, sizeof(*rows));
}

/* Sizes and carves one row-set scratch.  Everything a pass touches lives in the
 * single `block`, so a pass allocates nothing (CLAUDE.md rule 4). */
static int tar_rows_reserve(tar_rows *rows,
                            const mynah_transformer_ar_config *config,
                            size_t attn_dim, size_t count, char *error,
                            size_t error_capacity) {
    memset(rows, 0, sizeof(*rows));
    if (count == 0) count = 1u;
    size_t per = 0, part = 0, total = 0;
    int overflow = 0;
    overflow |= tar_mul(config->d_model, 3u, &part); /* x, norm, upd */
    overflow |= tar_add(per, part, &per);
    overflow |= tar_mul(attn_dim, 4u, &part);        /* qkv (3) + attn */
    overflow |= tar_add(per, part, &per);
    /* ffn only.  There used to be a second ffn_dim here for `gelu`, a scratch
     * buffer mynah_gelu_tanh_array stopped reading -- src/kernels.c:1657,
     * `(void)scratch`, ahead of every ISA variant -- so it was reserved, never
     * written and never read.  For the backbone that is count * 4096 floats per
     * live request: 256 KB at a 16-row tile, with another 128 KB for a
     * codec-transformer state.  Nothing to do with latency; it is serving
     * density, which is what a per-request buffer costs sixteen times over. */
    overflow |= tar_mul(config->ffn_dim, 1u, &part); /* ffn */
    overflow |= tar_add(per, part, &per);
    overflow |= tar_mul(per, count, &total);
    if (overflow != 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: row scratch size overflow");
        return -1;
    }
    rows->block = calloc(total ? total : 1u, sizeof(float));
    rows->in_ptr = calloc(count, sizeof(*rows->in_ptr));
    rows->out_ptr = calloc(count, sizeof(*rows->out_ptr));
    if (rows->block == NULL || rows->in_ptr == NULL || rows->out_ptr == NULL) {
        tar_rows_release(rows);
        tar_set_error(error, error_capacity, "transformer_ar: out of memory");
        return -1;
    }
    float *cursor = rows->block;
    rows->x = cursor;    cursor += count * config->d_model;
    rows->norm = cursor; cursor += count * config->d_model;
    rows->upd = cursor;  cursor += count * config->d_model;
    rows->qkv = cursor;  cursor += count * 3u * attn_dim;
    rows->attn = cursor; cursor += count * attn_dim;
    rows->ffn = cursor;  cursor += count * config->ffn_dim;
    rows->gelu = NULL;   /* see the reservation above: the callee ignores it */
    rows->rows_cap = count;
    return 0;
}

void mynah_transformer_ar_config_defaults(mynah_transformer_ar_config *config) {
    if (config == NULL) return;
    config->d_model = 0;
    config->num_heads = 0;
    config->head_dim = 0;
    config->num_layers = 0;
    config->ffn_dim = 0;
    config->max_seq_len = 0;
    config->context = 0; /* unlimited, which is what the backbone uses */
    config->max_period = 10000.0f;
    config->layernorm_eps = 1e-5f;
}

mynah_transformer_ar_state *mynah_transformer_ar_state_new(
    const mynah_transformer_ar_config *config, char *error,
    size_t error_capacity) {
    if (config == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: null config");
        return NULL;
    }
    mynah_transformer_ar_config resolved = *config;
    if (resolved.head_dim == 0 && resolved.num_heads != 0) {
        resolved.head_dim = resolved.d_model / resolved.num_heads;
    }
    if (resolved.d_model == 0 || resolved.num_heads == 0 ||
        resolved.head_dim == 0 || resolved.num_layers == 0 ||
        resolved.ffn_dim == 0 || resolved.max_seq_len == 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: bad dims d_model=%zu heads=%zu "
                      "head_dim=%zu layers=%zu ffn=%zu max_seq=%zu",
                      resolved.d_model, resolved.num_heads, resolved.head_dim,
                      resolved.num_layers, resolved.ffn_dim,
                      resolved.max_seq_len);
        return NULL;
    }
    if ((resolved.head_dim % 2u) != 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: head_dim %zu must be even for RoPE",
                      resolved.head_dim);
        return NULL;
    }
    if (!(resolved.max_period > 1.0f)) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: max_period must be > 1");
        return NULL;
    }
    if (!(resolved.layernorm_eps > 0.0f)) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: layernorm_eps must be > 0");
        return NULL;
    }

    mynah_transformer_ar_state *state = calloc(1u, sizeof(*state));
    if (state == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: out of memory");
        return NULL;
    }
    state->config = resolved;
    state->half = resolved.head_dim / 2u;
    state->offset = 0;

    size_t attn_dim = 0;
    if (tar_mul(resolved.num_heads, resolved.head_dim, &attn_dim) != 0) {
        tar_set_error(error, error_capacity, "transformer_ar: attn_dim overflow");
        free(state);
        return NULL;
    }
    state->attn_dim = attn_dim;

    /* KV: [layers][2][kv_positions][attn_dim].
     *
     * The slack above `context` is what sets how often a compaction runs: with
     * `context` of slack it runs at most once every `context` positions and
     * moves at most `context` of them, i.e. an amortised one position moved
     * per step.  It is floored at the prefill tile because a single tile must
     * fit alongside the window, and capped at max_seq_len because there is
     * nothing to gain beyond it. */
    size_t kv_positions = resolved.max_seq_len;
    if (resolved.context > 0u && g_tar_kv_window_force != 0) {
        size_t slack = resolved.context;
        if (slack < TAR_PREFILL_TILE) slack = TAR_PREFILL_TILE;
        size_t want = 0;
        if (tar_add(resolved.context, slack, &want) == 0 &&
            want < kv_positions) {
            kv_positions = want;
        }
    }
    state->kv_positions = kv_positions;
    state->kv_base = 0;
    size_t kv_half = 0, kv_layer = 0, kv_total = 0, kv_bytes = 0;
    if (tar_mul(kv_positions, attn_dim, &kv_half) != 0 ||
        tar_mul(kv_half, 2u, &kv_layer) != 0 ||
        tar_mul(kv_layer, resolved.num_layers, &kv_total) != 0 ||
        tar_mul(kv_total, sizeof(float), &kv_bytes) != 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: KV cache size overflow");
        free(state);
        return NULL;
    }
    state->kv_half = kv_half;
    state->kv_layer = kv_layer;
    /* calloc, not malloc: an unread region must still be finite, so that a
     * bug shows up as a wrong number rather than as a NaN avalanche. */
    state->kv = calloc(kv_total, sizeof(float));
    if (state->kv == NULL) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: out of memory for %zu KV bytes",
                      kv_bytes);
        free(state);
        return NULL;
    }

    /* The prefill tile, which the single `_step` uses as a one-row tile. */
    if (tar_rows_reserve(&state->rows, &resolved, attn_dim, TAR_PREFILL_TILE,
                         error, error_capacity) != 0) {
        free(state->kv);
        free(state);
        return NULL;
    }
    state->refs = calloc(state->rows.rows_cap, sizeof(*state->refs));
    if (state->refs == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: out of memory");
        tar_rows_release(&state->rows);
        free(state->kv);
        free(state);
        return NULL;
    }

    /* Position-only scratch: the attention scores, and the two RoPE tables
     * only when they could not be shared. */
    const float *shared_rope =
        tar_rope_shared(resolved.max_seq_len, state->half, resolved.max_period);
    size_t total = 0, part = 0;
    int overflow = 0;
    /* `scores` holds one span, and a span is at most `context` when there is
     * one -- so kv_positions bounds it exactly as it bounds the cache. */
    overflow |= tar_add(total, kv_positions, &total);
    if (shared_rope == NULL) {
        overflow |= tar_mul(resolved.max_seq_len, state->half, &part);
        overflow |= tar_mul(part, 2u, &part);
        overflow |= tar_add(total, part, &total);
    }
    if (overflow != 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: scratch size overflow");
        free(state->refs);
        tar_rows_release(&state->rows);
        free(state->kv);
        free(state);
        return NULL;
    }
    state->block = calloc(total ? total : 1u, sizeof(float));
    if (state->block == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: out of memory");
        free(state->refs);
        tar_rows_release(&state->rows);
        free(state->kv);
        free(state);
        return NULL;
    }

    float *cursor = state->block;
    state->scores = cursor;
    if (shared_rope != NULL) {
        state->rope_cos = shared_rope;
        state->rope_sin = shared_rope + resolved.max_seq_len * state->half;
    } else {
        /* The RoPE table is position-only, so it is built once and the hot
         * loop contains no transcendental at all.  This arm only runs when the
         * shared table could not be allocated. */
        float *cos_table = cursor + kv_positions;
        float *sin_table = cos_table + resolved.max_seq_len * state->half;
        for (size_t p = 0; p < resolved.max_seq_len; ++p) {
            mynah_transformer_ar_rope_angles_f32(cos_table + p * state->half,
                                                 sin_table + p * state->half,
                                                 state->half, p,
                                                 resolved.max_period);
        }
        state->rope_cos = cos_table;
        state->rope_sin = sin_table;
    }
    return state;
}

void mynah_transformer_ar_state_free(mynah_transformer_ar_state *state) {
    if (state == NULL) return;
    tar_rows_release(&state->rows);
    free(state->refs);
    free(state->kv);
    free(state->block);
    free(state);
}

/* ---- cross-request batch scratch ---- */

mynah_transformer_ar_batch *mynah_transformer_ar_batch_new(
    const mynah_transformer_ar_config *config, size_t max_rows, char *error,
    size_t error_capacity) {
    if (config == NULL || max_rows == 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: bad batch arguments");
        return NULL;
    }
    mynah_transformer_ar_config resolved = *config;
    if (resolved.head_dim == 0 && resolved.num_heads != 0) {
        resolved.head_dim = resolved.d_model / resolved.num_heads;
    }
    if (resolved.d_model == 0 || resolved.num_heads == 0 ||
        resolved.head_dim == 0 || resolved.num_layers == 0 ||
        resolved.ffn_dim == 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: batch built from an incomplete config");
        return NULL;
    }
    size_t attn_dim = 0;
    if (tar_mul(resolved.num_heads, resolved.head_dim, &attn_dim) != 0) {
        tar_set_error(error, error_capacity, "transformer_ar: attn_dim overflow");
        return NULL;
    }
    mynah_transformer_ar_batch *batch = calloc(1u, sizeof(*batch));
    if (batch == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: out of memory");
        return NULL;
    }
    batch->config = resolved;
    batch->attn_dim = attn_dim;
    if (tar_rows_reserve(&batch->rows, &resolved, attn_dim, max_rows, error,
                         error_capacity) != 0) {
        free(batch);
        return NULL;
    }
    batch->refs = calloc(batch->rows.rows_cap, sizeof(*batch->refs));
    if (batch->refs == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: out of memory");
        tar_rows_release(&batch->rows);
        free(batch);
        return NULL;
    }
    return batch;
}

void mynah_transformer_ar_batch_free(mynah_transformer_ar_batch *batch) {
    if (batch == NULL) return;
    tar_rows_release(&batch->rows);
    free(batch->refs);
    free(batch);
}

size_t mynah_transformer_ar_batch_capacity(
    const mynah_transformer_ar_batch *batch) {
    return (batch == NULL) ? 0u : batch->rows.rows_cap;
}

void mynah_transformer_ar_state_reset(mynah_transformer_ar_state *state) {
    if (state == NULL) return;
    state->offset = 0;
    /* The window slides with the positions, so a rewind to position 0 has to
     * rewind the base with it, or the next step would compute a slot from a
     * base the cache no longer holds. */
    state->kv_base = 0;
}

const mynah_transformer_ar_config *mynah_transformer_ar_state_config(
    const mynah_transformer_ar_state *state) {
    return (state == NULL) ? NULL : &state->config;
}

size_t mynah_transformer_ar_state_offset(
    const mynah_transformer_ar_state *state) {
    return (state == NULL) ? 0 : state->offset;
}

float *mynah_transformer_ar_state_kv(mynah_transformer_ar_state *state,
                                     size_t layer) {
    if (state == NULL || layer >= state->config.num_layers) return NULL;
    /* The block this returns is documented as [2][max_seq_len][heads][dim]
     * with slot 0 at position 0, which is what makes a voice prefix two
     * memcpys.  A windowed cache is shorter and its slot 0 moves, so the
     * contract does not hold and the honest answer is NULL rather than a
     * pointer whose meaning silently changed.  Every caller in the tree --
     * voice loading and the corruption probe -- is on the backbone, which is
     * unwindowed. */
    if (state->kv_positions < state->config.max_seq_len) return NULL;
    return state->kv + layer * state->kv_layer;
}

size_t mynah_transformer_ar_state_kv_half_floats(
    const mynah_transformer_ar_state *state) {
    return (state == NULL) ? 0 : state->kv_half;
}

int mynah_transformer_ar_state_load_kv(mynah_transformer_ar_state *state,
                                       size_t layer, const float *kv,
                                       size_t positions, char *error,
                                       size_t error_capacity) {
    if (state == NULL || kv == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: null argument");
        return -1;
    }
    if (layer >= state->config.num_layers) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: layer %zu out of range (%zu layers)",
                      layer, state->config.num_layers);
        return -1;
    }
    if (state->kv_positions < state->config.max_seq_len) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: this cache is windowed (context %zu, "
                      "%zu slots) and a KV prefix assumes slot 0 is position 0",
                      state->config.context, state->kv_positions);
        return -1;
    }
    if (positions > state->kv_positions) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: voice prefix %zu exceeds the cache's "
                      "%zu positions",
                      positions, state->kv_positions);
        return -1;
    }
    const size_t span = positions * state->attn_dim;
    /* This is the one place upstream's NaN padding can enter the runtime. */
    if (!tar_finite(kv, 2u * span)) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: layer %zu KV prefix has a non-finite "
                      "value (NaN padding must be trimmed to current_end)",
                      layer);
        return -1;
    }
    float *dst = state->kv + layer * state->kv_layer;
    if (span > 0) {
        memcpy(dst, kv, span * sizeof(float));
        memcpy(dst + state->kv_half, kv + span, span * sizeof(float));
    }
    return 0;
}

int mynah_transformer_ar_state_set_offset(mynah_transformer_ar_state *state,
                                          size_t positions, char *error,
                                          size_t error_capacity) {
    if (state == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: null state");
        return -1;
    }
    if (positions > state->config.max_seq_len) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: offset %zu exceeds max_seq_len %zu",
                      positions, state->config.max_seq_len);
        return -1;
    }
    if (positions > state->kv_positions) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: offset %zu exceeds the cache's %zu "
                      "positions; a windowed cache has no slot for it",
                      positions, state->kv_positions);
        return -1;
    }
    state->offset = positions;
    return 0;
}

int mynah_transformer_ar_check_weights(
    const mynah_transformer_ar_state *state,
    const mynah_transformer_ar_weights *weights, char *error,
    size_t error_capacity) {
    if (state == NULL || weights == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: null argument");
        return -1;
    }
    if (weights->layers == NULL) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: layer array missing");
        return -1;
    }
    for (size_t l = 0; l < state->config.num_layers; ++l) {
        const mynah_transformer_ar_layer *layer = &weights->layers[l];
        const char *missing = NULL;
        if (layer->in_proj_weight == NULL) missing = "self_attn.in_proj";
        else if (layer->out_proj_weight == NULL) missing = "self_attn.out_proj";
        else if (layer->norm1_weight == NULL) missing = "norm1.weight";
        else if (layer->norm2_weight == NULL) missing = "norm2.weight";
        else if (layer->linear1_weight == NULL) missing = "linear1";
        else if (layer->linear2_weight == NULL) missing = "linear2";
        if (missing != NULL) {
            tar_set_error(error, error_capacity,
                          "transformer_ar: layer %zu is missing %s", l,
                          missing);
            return -1;
        }
    }
    if (weights->out_norm_weight == NULL && weights->out_norm_bias != NULL) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: out_norm bias without weight");
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------- forward */

/* Lowest key position a query at `position` may attend to. */
static size_t tar_window_start(size_t position, size_t context) {
    if (context == 0 || position + 1u <= context) return 0;
    return position + 1u - context;
}

/* Make room for absolute position `hi`, keeping everything from `keep_from`
 * up.  Returns 0 when the cache can serve [keep_from, hi], -1 when it cannot,
 * which is a configuration error and not a runtime condition: kv_positions is
 * built as context + slack, so a tile that does not fit would mean a tile
 * wider than the slack.
 *
 * A no-op on an unwindowed cache, which is every path that predates this. */
static int tar_kv_reserve(mynah_transformer_ar_state *state, size_t keep_from,
                          size_t hi) {
    if (state->kv_positions >= state->config.max_seq_len) return 0;
    if (hi < state->kv_base) return -1;   /* positions never rewind */
    if (hi - state->kv_base < state->kv_positions) return 0;
    if (keep_from < state->kv_base) keep_from = state->kv_base;
    const size_t shift = keep_from - state->kv_base;
    if (shift == 0u) return -1;
    /* How much of what is stored survives the move.
     *
     * `kv_positions - shift` is an UNSIGNED subtraction, and it is only safe
     * because a caller advances positions by at most one tile per call, so
     * `shift` stays under the slack.  That is four assumptions about the
     * caller holding up a memmove length, which is not a thing to leave
     * implicit: a `shift` at or past the capacity means the window has moved
     * entirely past what we hold, and the answer to that is to keep nothing
     * and rebase -- not to underflow into a memmove of about 2^64 bytes. */
    size_t keep = 0u;
    if (shift < state->kv_positions && state->offset > keep_from) {
        keep = state->offset - keep_from;
        const size_t room = state->kv_positions - shift;
        if (keep > room) keep = room;
    }
    if (keep > 0u) {
        const size_t row = state->attn_dim * sizeof(float);
        for (size_t l = 0; l < state->config.num_layers; ++l) {
            float *k = state->kv + l * state->kv_layer;
            float *v = k + state->kv_half;
            memmove(k, k + shift * state->attn_dim, keep * row);
            memmove(v, v + shift * state->attn_dim, keep * row);
        }
    }
    state->kv_base = keep_from;
    return (hi - state->kv_base < state->kv_positions) ? 0 : -1;
}

/* Every state in `refs` gets room for the whole tile before ANY row writes.
 *
 * The bound that matters is the LOWEST position in the tile, not the highest:
 * row b reads [window_start(p_b), p_b], so compacting to the last row's window
 * would drop what the first row still needs.  States appear once in a
 * cross-request step and once per position in a prefill tile, so the scan is
 * O(count^2) over at most TAR_PREFILL_TILE rows -- nothing against a layer. */
static int tar_kv_prepare(const mynah_transformer_ar_config *config,
                          const tar_row_ref *refs, size_t count) {
    if (config->context == 0u) return 0;
    for (size_t b = 0; b < count; ++b) {
        mynah_transformer_ar_state *state = refs[b].state;
        int first = 1;
        for (size_t a = 0; a < b; ++a) {
            if (refs[a].state == state) { first = 0; break; }
        }
        if (!first) continue;
        size_t lo = refs[b].position, hi = refs[b].position;
        for (size_t a = b + 1u; a < count; ++a) {
            if (refs[a].state != state) continue;
            if (refs[a].position < lo) lo = refs[a].position;
            if (refs[a].position > hi) hi = refs[a].position;
        }
        if (tar_kv_reserve(state, tar_window_start(lo, config->context), hi) != 0) {
            return -1;
        }
    }
    return 0;
}


/*
 * One layer projection for `rows` stacked rows.
 *
 * `rows == 1` takes exactly the path the single `_step` takes, so the three
 * entry points cannot diverge on a one-row call.  Above that the two batch
 * kinds part ways, and the reason is the whole point of this file:
 *
 *   - `cross_request == 0` -- consecutive positions of ONE sequence.  The row
 *     count is a property of that request alone, so the hook may do whatever a
 *     GEMM does.
 *   - `cross_request == 1` -- one position of N DIFFERENT requests.  Only a
 *     hook that promises bit-exact rows may be used; without one, every row
 *     falls back to the single-row call it would have made alone.
 */
static int tar_linear_rows(const mynah_transformer_ar_weights *weights,
                           size_t layer,
                           mynah_transformer_ar_linear_kind kind,
                           const float *weight, const float *bias, tar_rows *rows,
                           const float *in, float *out, size_t count, size_t k,
                           size_t n, int cross_request) {
    if (count > 1u) {
        if (cross_request) {
            if (weights->linear_rows != NULL) {
                for (size_t b = 0; b < count; ++b) {
                    rows->in_ptr[b] = in + b * k;
                    rows->out_ptr[b] = out + b * n;
                }
                return weights->linear_rows(weights->linear_user, layer, kind,
                                            weight, bias, rows->in_ptr,
                                            rows->out_ptr, count, k, n);
            }
        } else if (weights->linear != NULL) {
            return weights->linear(weights->linear_user, layer, kind, weight, bias,
                                   in, out, count, k, n);
        }
    }
    for (size_t b = 0; b < count; ++b) {
        const float *xr = in + b * k;
        float *orow = out + b * n;
        if (weights->linear != NULL) {
            if (weights->linear(weights->linear_user, layer, kind, weight, bias, xr,
                                orow, 1u, k, n) != 0) {
                return -1;
            }
        } else {
            mynah_matvec_bias_f32(weight, xr, bias, orow, n, k);
        }
    }
    return 0;
}

/*
 * The stack, over `count` rows already loaded into `rows->x`.
 *
 * `refs[b]` says which KV cache row b writes and at which absolute position, so
 * this one body serves a prefill tile (one state, consecutive positions) and a
 * cross-request step (N states, one position each) without either becoming a
 * second graph.  Offsets are NOT advanced here; the caller owns that, because
 * only the caller knows whether it is moving one state by `count` or `count`
 * states by one.
 *
 * The K/V of every row is written before any row attends, which is what makes
 * the prefill tile legal: a query at position p reads `[lo, p]` only, so the
 * later positions of its own tile are present but unreachable.
 */
static int tar_forward_rows(const mynah_transformer_ar_weights *weights,
                            const mynah_transformer_ar_config *config,
                            size_t attn_dim, size_t half, tar_rows *rows,
                            const tar_row_ref *refs, size_t count,
                            int cross_request) {
    const size_t d_model = config->d_model;
    const size_t heads = config->num_heads;
    const size_t head_dim = config->head_dim;
    const size_t ffn_dim = config->ffn_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    /* Before layer 0 writes anything: a compaction moves every layer's cache
     * at once, so it cannot happen between two layers of the same pass. */
    if (tar_kv_prepare(config, refs, count) != 0) return -1;

    for (size_t l = 0; l < config->num_layers; ++l) {
        const mynah_transformer_ar_layer *layer = &weights->layers[l];

        /* --- attention block: x + layer_scale_1(attn(norm1(x))) --- */
        mynah_layernorm_f32(rows->x, layer->norm1_weight, layer->norm1_bias,
                            rows->norm, count, d_model, config->layernorm_eps);
        if (tar_linear_rows(weights, l, MYNAH_TAR_LINEAR_IN_PROJ,
                            layer->in_proj_weight, layer->in_proj_bias, rows,
                            rows->norm, rows->qkv, count, d_model, 3u * attn_dim,
                            cross_request) != 0) {
            return -1;
        }
        for (size_t b = 0; b < count; ++b) {
            mynah_transformer_ar_state *state = refs[b].state;
            const size_t position = refs[b].position;
            float *q = rows->qkv + b * 3u * attn_dim;
            float *k = q + attn_dim;
            float *v = k + attn_dim;
            const float *rope_cos = state->rope_cos + position * half;
            const float *rope_sin = state->rope_sin + position * half;
            mynah_transformer_ar_rope_apply_f32(q, heads, head_dim, rope_cos,
                                                rope_sin);
            mynah_transformer_ar_rope_apply_f32(k, heads, head_dim, rope_cos,
                                                rope_sin);
            float *k_cache = state->kv + l * state->kv_layer;
            float *v_cache = k_cache + state->kv_half;
            const size_t slot = position - state->kv_base;
            memcpy(k_cache + slot * attn_dim, k, attn_dim * sizeof(float));
            memcpy(v_cache + slot * attn_dim, v, attn_dim * sizeof(float));
        }
        /* A key-stationary variant of this loop -- key outer, tile positions
         * inner, so K and V are read once for the whole tile instead of once
         * per (position, head) -- was written and measured: bit-identical, and
         * 194 ms -> 211/219 ms on the Mimi decoder.  It is not here because it
         * lost.  The reason is worth keeping: the per-(position, head) walk
         * marches K with a constant stride that the prefetcher gets right and
         * holds one 64-float query in registers, and at a 250-position window
         * the whole K block is 512 KB, i.e. L2-resident, so the re-reads it
         * "saves" are L2 hits rather than DRAM traffic.  The codec transformer
         * at this point is arithmetic-bound (~13 GFLOP for a 5.2 s utterance
         * against this core's ~100 GFLOP/s f32 roof), not traffic-bound, and
         * reordering traffic cannot move an arithmetic bound. */
        for (size_t b = 0; b < count; ++b) {
            mynah_transformer_ar_state *state = refs[b].state;
            const size_t position = refs[b].position;
            const size_t lo = tar_window_start(position, config->context);
            const size_t span = position - lo + 1u;
            /* Slot 0 is `kv_base`, not position 0: on an unwindowed cache the
             * base is always 0 and this is the same arithmetic as before. */
            const size_t lo_slot = lo - state->kv_base;
            const float *q = rows->qkv + b * 3u * attn_dim;
            const float *k_cache = state->kv + l * state->kv_layer;
            const float *v_cache = k_cache + state->kv_half;
            float *scores = state->scores;
            for (size_t h = 0; h < heads; ++h) {
                const float *qh = q + h * head_dim;
                for (size_t j = 0; j < span; ++j) {
                    const float *kj = k_cache + (lo_slot + j) * attn_dim + h * head_dim;
                    scores[j] = mynah_dot_f32(qh, kj, head_dim) * scale;
                }
                /* Rejects non-finite scores, which is the last line of defence
                 * against a NaN that slipped into the cache. */
                if (mynah_softmax_f32(scores, scores, span) != 0) return -1;
                float *oh = rows->attn + b * attn_dim + h * head_dim;
                memset(oh, 0, head_dim * sizeof(float));
                for (size_t j = 0; j < span; ++j) {
                    const float *vj = v_cache + (lo_slot + j) * attn_dim + h * head_dim;
                    mynah_axpy_f32(oh, vj, scores[j], head_dim);
                }
            }
        }
        if (tar_linear_rows(weights, l, MYNAH_TAR_LINEAR_OUT_PROJ,
                            layer->out_proj_weight, layer->out_proj_bias, rows,
                            rows->attn, rows->upd, count, attn_dim, d_model,
                            cross_request) != 0) {
            return -1;
        }
        if (layer->layer_scale_1 != NULL) {
            for (size_t b = 0; b < count; ++b) {
                float *xr = rows->x + b * d_model;
                const float *ur = rows->upd + b * d_model;
                for (size_t i = 0; i < d_model; ++i) {
                    xr[i] += layer->layer_scale_1[i] * ur[i];
                }
            }
        } else {
            mynah_residual_add_f32(rows->x, rows->upd, count * d_model);
        }

        /* --- feed forward: x + layer_scale_2(linear2(gelu(linear1(norm2)))) */
        mynah_layernorm_f32(rows->x, layer->norm2_weight, layer->norm2_bias,
                            rows->norm, count, d_model, config->layernorm_eps);
        if (tar_linear_rows(weights, l, MYNAH_TAR_LINEAR_FFN1,
                            layer->linear1_weight, layer->linear1_bias, rows,
                            rows->norm, rows->ffn, count, d_model, ffn_dim,
                            cross_request) != 0) {
            return -1;
        }
        mynah_gelu_tanh_array(rows->ffn, count * ffn_dim, rows->gelu);
        if (tar_linear_rows(weights, l, MYNAH_TAR_LINEAR_FFN2,
                            layer->linear2_weight, layer->linear2_bias, rows,
                            rows->ffn, rows->upd, count, ffn_dim, d_model,
                            cross_request) != 0) {
            return -1;
        }
        if (layer->layer_scale_2 != NULL) {
            for (size_t b = 0; b < count; ++b) {
                float *xr = rows->x + b * d_model;
                const float *ur = rows->upd + b * d_model;
                for (size_t i = 0; i < d_model; ++i) {
                    xr[i] += layer->layer_scale_2[i] * ur[i];
                }
            }
        } else {
            mynah_residual_add_f32(rows->x, rows->upd, count * d_model);
        }
    }
    return 0;
}

/* The optional final LayerNorm, applied row by row into the caller's rows. */
static void tar_finish_row(const mynah_transformer_ar_weights *weights,
                           const mynah_transformer_ar_config *config,
                           const float *x, float *out) {
    if (out == NULL) return;
    if (weights->out_norm_weight != NULL) {
        mynah_layernorm_f32(x, weights->out_norm_weight, weights->out_norm_bias,
                            out, 1u, config->d_model, config->layernorm_eps);
    } else {
        memcpy(out, x, config->d_model * sizeof(float));
    }
}

int mynah_transformer_ar_prefill(mynah_transformer_ar_state *state,
                                 const mynah_transformer_ar_weights *weights,
                                 const float *x, size_t n_tokens, float *out) {
    if (state == NULL || weights == NULL || weights->layers == NULL) return -1;
    if (n_tokens == 0) return 0;
    if (x == NULL) return -1;
    const mynah_transformer_ar_config *config = &state->config;
    const size_t d_model = config->d_model;
    size_t end = 0;
    if (tar_add(state->offset, n_tokens, &end) != 0 ||
        end > config->max_seq_len) {
        return -1;
    }
    /* Fail at the boundary rather than letting a BOS sentinel reach a matmul:
     * the caller substitutes bos_emb before input_linear, so anything
     * non-finite arriving here is a bug, not a sentinel. */
    if (!tar_finite(x, n_tokens * d_model)) return -1;

    for (size_t done = 0; done < n_tokens;) {
        size_t rows = n_tokens - done;
        if (rows > state->rows.rows_cap) rows = state->rows.rows_cap;
        const size_t base = state->offset;
        for (size_t b = 0; b < rows; ++b) {
            state->refs[b].state = state;
            state->refs[b].position = base + b;
        }
        memcpy(state->rows.x, x + done * d_model, rows * d_model * sizeof(float));
        if (tar_forward_rows(weights, config, state->attn_dim, state->half,
                             &state->rows, state->refs, rows, 0) != 0) {
            return -1;
        }
        state->offset = base + rows;
        if (out != NULL) {
            for (size_t b = 0; b < rows; ++b) {
                tar_finish_row(weights, config, state->rows.x + b * d_model,
                               out + (done + b) * d_model);
            }
        }
        done += rows;
    }
    return 0;
}

int mynah_transformer_ar_step(mynah_transformer_ar_state *state,
                              const mynah_transformer_ar_weights *weights,
                              const float *x, float *out) {
    if (state == NULL || weights == NULL || weights->layers == NULL ||
        x == NULL || out == NULL) {
        return -1;
    }
    const mynah_transformer_ar_config *config = &state->config;
    if (state->offset >= config->max_seq_len) return -1;
    if (!tar_finite(x, config->d_model)) return -1;
    state->refs[0].state = state;
    state->refs[0].position = state->offset;
    memcpy(state->rows.x, x, config->d_model * sizeof(float));
    if (tar_forward_rows(weights, config, state->attn_dim, state->half,
                         &state->rows, state->refs, 1u, 0) != 0) {
        return -1;
    }
    state->offset += 1u;
    tar_finish_row(weights, config, state->rows.x, out);
    return 0;
}

/* Two configurations describe the same graph, i.e. the same weights back both
 * and their KV caches have the same interior.  `max_seq_len` may differ: it is
 * a capacity, not a shape. */
static int tar_config_same(const mynah_transformer_ar_config *a,
                           const mynah_transformer_ar_config *b) {
    return a->d_model == b->d_model && a->num_heads == b->num_heads &&
           a->head_dim == b->head_dim && a->num_layers == b->num_layers &&
           a->ffn_dim == b->ffn_dim && a->context == b->context &&
           a->max_period == b->max_period && a->layernorm_eps == b->layernorm_eps;
}

int mynah_transformer_ar_step_batch(mynah_transformer_ar_state *const *states,
                                    size_t count,
                                    const mynah_transformer_ar_weights *weights,
                                    mynah_transformer_ar_batch *batch,
                                    const float *const *x, float *const *out) {
    if (count == 0) return 0;
    if (states == NULL || weights == NULL || weights->layers == NULL ||
        x == NULL || out == NULL) {
        return -1;
    }
    if (count == 1u) {
        return mynah_transformer_ar_step(states[0], weights, x[0], out[0]);
    }
    if (batch == NULL || count > batch->rows.rows_cap) return -1;
    const mynah_transformer_ar_config *config = &batch->config;
    const size_t d_model = config->d_model;
    for (size_t b = 0; b < count; ++b) {
        mynah_transformer_ar_state *state = states[b];
        if (state == NULL || x[b] == NULL || out[b] == NULL) return -1;
        if (!tar_config_same(&state->config, config)) return -1;
        if (state->offset >= state->config.max_seq_len) return -1;
        if (!tar_finite(x[b], d_model)) return -1;
        /* Two states appearing twice in one batch would have the second write
         * of a position silently overwrite the first. */
        for (size_t c = 0; c < b; ++c) {
            if (states[c] == state) return -1;
        }
        batch->refs[b].state = state;
        batch->refs[b].position = state->offset;
        memcpy(batch->rows.x + b * d_model, x[b], d_model * sizeof(float));
    }
    if (tar_forward_rows(weights, config, batch->attn_dim,
                         config->head_dim / 2u, &batch->rows, batch->refs, count,
                         1) != 0) {
        return -1;
    }
    for (size_t b = 0; b < count; ++b) {
        states[b]->offset += 1u;
        tar_finish_row(weights, config, batch->rows.x + b * d_model, out[b]);
    }
    return 0;
}

/* -------------------------------------------------------------- self test */

#define TAR_T 6u  /* self-test sequence length  */
#define TAR_D 8u  /* self-test d_model          */
#define TAR_H 2u  /* self-test heads            */
#define TAR_HD 4u /* self-test head_dim         */
#define TAR_F 16u /* self-test ffn_dim          */
#define TAR_L 2u  /* self-test layers           */

/* Deterministic, reproducible, and small enough that the double reference
 * below stays numerically meaningful. */
static float tar_fake(size_t index, unsigned salt) {
    unsigned x = (unsigned)(index * 2654435761u) ^ (salt * 40503u);
    x ^= x >> 13;
    x *= 2246822519u;
    x ^= x >> 17;
    return (float)((double)(x % 20011u) / 20011.0 - 0.5) * 0.5f;
}

static int tar_close(double got, double want, double tolerance) {
    const double diff = fabs(got - want);
    if (diff <= tolerance) return 1;
    const double magnitude = fabs(want) > 1.0 ? fabs(want) : 1.0;
    return diff / magnitude <= tolerance;
}

static void tar_ref_linear(const float *weight, const float *bias,
                           const double *input, double *output, size_t rows,
                           size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        double sum = (bias != NULL) ? (double)bias[r] : 0.0;
        for (size_t c = 0; c < cols; ++c) {
            sum += (double)weight[r * cols + c] * input[c];
        }
        output[r] = sum;
    }
}

static void tar_ref_layernorm(const double *input, const float *weight,
                              const float *bias, double *output, size_t n,
                              double epsilon) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += input[i];
    mean /= (double)n;
    double variance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = input[i] - mean;
        variance += d * d;
    }
    variance /= (double)n;
    const double scale = 1.0 / sqrt(variance + epsilon);
    for (size_t i = 0; i < n; ++i) {
        double value = (input[i] - mean) * scale;
        if (weight != NULL) value *= (double)weight[i];
        if (bias != NULL) value += (double)bias[i];
        output[i] = value;
    }
}

static double tar_ref_gelu_tanh(double x) {
    const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

/* Layer-by-layer, all positions at once: deliberately the *batched* shape, so
 * that agreeing with it proves the token-sequential implementation and a
 * masked SDPA compute the same function. */
static void tar_ref_forward(const mynah_transformer_ar_weights *weights,
                            const double x_in[TAR_T][TAR_D], size_t n_tokens,
                            size_t context, double out[TAR_T][TAR_D]) {
    double x[TAR_T][TAR_D];
    for (size_t t = 0; t < n_tokens; ++t) {
        for (size_t i = 0; i < TAR_D; ++i) x[t][i] = x_in[t][i];
    }
    const size_t half = TAR_HD / 2u;
    const double slope = -log(10000.0) * 2.0 / (double)TAR_HD;

    for (size_t l = 0; l < TAR_L; ++l) {
        const mynah_transformer_ar_layer *layer = &weights->layers[l];
        double kc[TAR_T][TAR_D], vc[TAR_T][TAR_D], qc[TAR_T][TAR_D];
        for (size_t t = 0; t < n_tokens; ++t) {
            double norm[TAR_D], qkv[3u * TAR_D];
            tar_ref_layernorm(x[t], layer->norm1_weight, layer->norm1_bias,
                              norm, TAR_D, 1e-5);
            tar_ref_linear(layer->in_proj_weight, layer->in_proj_bias, norm,
                           qkv, 3u * TAR_D, TAR_D);
            for (size_t h = 0; h < TAR_H; ++h) {
                for (size_t i = 0; i < half; ++i) {
                    const double angle = exp((double)i * slope) * (double)t;
                    const double c = cos(angle), s = sin(angle);
                    const size_t a = h * TAR_HD + 2u * i;
                    const double qr = qkv[a], qi = qkv[a + 1u];
                    const double kr = qkv[TAR_D + a], ki = qkv[TAR_D + a + 1u];
                    qc[t][a] = qr * c - qi * s;
                    qc[t][a + 1u] = qr * s + qi * c;
                    kc[t][a] = kr * c - ki * s;
                    kc[t][a + 1u] = kr * s + ki * c;
                }
            }
            for (size_t i = 0; i < TAR_D; ++i) vc[t][i] = qkv[2u * TAR_D + i];
        }
        for (size_t t = 0; t < n_tokens; ++t) {
            double attn[TAR_D], upd[TAR_D];
            const size_t lo =
                (context == 0 || t + 1u <= context) ? 0 : t + 1u - context;
            for (size_t h = 0; h < TAR_H; ++h) {
                double scores[TAR_T], maximum = -INFINITY, total = 0.0;
                for (size_t j = lo; j <= t; ++j) {
                    double dot = 0.0;
                    for (size_t i = 0; i < TAR_HD; ++i) {
                        dot += qc[t][h * TAR_HD + i] * kc[j][h * TAR_HD + i];
                    }
                    scores[j] = dot / sqrt((double)TAR_HD);
                    if (scores[j] > maximum) maximum = scores[j];
                }
                for (size_t j = lo; j <= t; ++j) {
                    scores[j] = exp(scores[j] - maximum);
                    total += scores[j];
                }
                for (size_t i = 0; i < TAR_HD; ++i) {
                    double sum = 0.0;
                    for (size_t j = lo; j <= t; ++j) {
                        sum += scores[j] * vc[j][h * TAR_HD + i];
                    }
                    attn[h * TAR_HD + i] = sum / total;
                }
            }
            tar_ref_linear(layer->out_proj_weight, layer->out_proj_bias, attn,
                           upd, TAR_D, TAR_D);
            for (size_t i = 0; i < TAR_D; ++i) {
                const double gain = (layer->layer_scale_1 != NULL)
                                        ? (double)layer->layer_scale_1[i]
                                        : 1.0;
                x[t][i] += gain * upd[i];
            }
            double norm[TAR_D], ffn[TAR_F];
            tar_ref_layernorm(x[t], layer->norm2_weight, layer->norm2_bias,
                              norm, TAR_D, 1e-5);
            tar_ref_linear(layer->linear1_weight, layer->linear1_bias, norm,
                           ffn, TAR_F, TAR_D);
            for (size_t i = 0; i < TAR_F; ++i) ffn[i] = tar_ref_gelu_tanh(ffn[i]);
            tar_ref_linear(layer->linear2_weight, layer->linear2_bias, ffn, upd,
                           TAR_D, TAR_F);
            for (size_t i = 0; i < TAR_D; ++i) {
                const double gain = (layer->layer_scale_2 != NULL)
                                        ? (double)layer->layer_scale_2[i]
                                        : 1.0;
                x[t][i] += gain * upd[i];
            }
        }
    }
    for (size_t t = 0; t < n_tokens; ++t) {
        if (weights->out_norm_weight != NULL) {
            tar_ref_layernorm(x[t], weights->out_norm_weight,
                              weights->out_norm_bias, out[t], TAR_D, 1e-5);
        } else {
            for (size_t i = 0; i < TAR_D; ++i) out[t][i] = x[t][i];
        }
    }
}

/* Weight storage for the self test, so the layout is owned in one place. */
typedef struct {
    float in_proj[TAR_L][3u * TAR_D * TAR_D];
    float out_proj[TAR_L][TAR_D * TAR_D];
    float norm1_w[TAR_L][TAR_D], norm1_b[TAR_L][TAR_D];
    float norm2_w[TAR_L][TAR_D], norm2_b[TAR_L][TAR_D];
    float linear1[TAR_L][TAR_F * TAR_D];
    float linear2[TAR_L][TAR_D * TAR_F];
    float out_norm_w[TAR_D], out_norm_b[TAR_D];
    float ones[TAR_D];
    mynah_transformer_ar_layer layers[TAR_L];
    mynah_transformer_ar_weights weights;
} tar_test_weights;

static void tar_build_weights(tar_test_weights *store) {
    unsigned salt = 1u;
    memset(store, 0, sizeof(*store));
    for (size_t l = 0; l < TAR_L; ++l) {
        for (size_t i = 0; i < 3u * TAR_D * TAR_D; ++i) {
            store->in_proj[l][i] = tar_fake(i, salt);
        }
        ++salt;
        for (size_t i = 0; i < TAR_D * TAR_D; ++i) {
            store->out_proj[l][i] = tar_fake(i, salt);
        }
        ++salt;
        for (size_t i = 0; i < TAR_D; ++i) {
            store->norm1_w[l][i] = 1.0f + tar_fake(i, salt);
            store->norm1_b[l][i] = tar_fake(i, salt + 1u);
            store->norm2_w[l][i] = 1.0f + tar_fake(i, salt + 2u);
            store->norm2_b[l][i] = tar_fake(i, salt + 3u);
        }
        salt += 4u;
        for (size_t i = 0; i < TAR_F * TAR_D; ++i) {
            store->linear1[l][i] = tar_fake(i, salt);
            store->linear2[l][i] = tar_fake(i, salt + 1u);
        }
        salt += 2u;

        mynah_transformer_ar_layer *layer = &store->layers[l];
        layer->in_proj_weight = store->in_proj[l];
        layer->out_proj_weight = store->out_proj[l];
        layer->norm1_weight = store->norm1_w[l];
        layer->norm1_bias = store->norm1_b[l];
        layer->norm2_weight = store->norm2_w[l];
        layer->norm2_bias = store->norm2_b[l];
        layer->linear1_weight = store->linear1[l];
        layer->linear2_weight = store->linear2[l];
    }
    for (size_t i = 0; i < TAR_D; ++i) {
        store->out_norm_w[i] = 1.0f + tar_fake(i, 99u);
        store->out_norm_b[i] = tar_fake(i, 100u);
        store->ones[i] = 1.0f;
    }
    store->weights.layers = store->layers;
    store->weights.out_norm_weight = store->out_norm_w;
    store->weights.out_norm_bias = store->out_norm_b;
}

static void tar_test_config(mynah_transformer_ar_config *config,
                            size_t context) {
    mynah_transformer_ar_config_defaults(config);
    config->d_model = TAR_D;
    config->num_heads = TAR_H;
    config->head_dim = TAR_HD;
    config->num_layers = TAR_L;
    config->ffn_dim = TAR_F;
    config->max_seq_len = TAR_T + 2u;
    config->context = context;
}

/* A row-pointer hook that does exactly what the built-in fallback does.  Its
 * only job is to make the hooked path's pointer plumbing observable: if
 * `in_ptr`/`out_ptr` were built wrong, this would produce different numbers
 * from the unhooked run of the same test. */
static int tar_test_linear_rows(void *user, size_t layer,
                                mynah_transformer_ar_linear_kind kind,
                                const float *weight, const float *bias,
                                const float *const *in_rows,
                                float *const *out_rows, size_t batch, size_t k,
                                size_t n) {
    (void)user;
    (void)layer;
    (void)kind;
    for (size_t b = 0; b < batch; ++b) {
        mynah_matvec_bias_f32(weight, in_rows[b], bias, out_rows[b], n, k);
    }
    return 0;
}

#define TAR_FAIL(...)                                       \
    do {                                                    \
        tar_set_error(error, error_capacity, __VA_ARGS__);  \
        return -1;                                          \
    } while (0)

int mynah_transformer_ar_self_test(char *error, size_t error_capacity) {
    /* 1. RoPE angles: the table must be exp(i * -log(P) * 2 / D) * position. */
    {
        float cosines[4], sines[4];
        const size_t half = 4u;
        mynah_transformer_ar_rope_angles_f32(cosines, sines, half, 7u,
                                             10000.0f);
        for (size_t i = 0; i < half; ++i) {
            const double freq =
                exp((double)i * -log(10000.0) * 2.0 / (double)(2u * half));
            const double angle = freq * 7.0;
            if (!tar_close(cosines[i], cos(angle), 1e-5) ||
                !tar_close(sines[i], sin(angle), 1e-5)) {
                TAR_FAIL("rope angles[%zu] = (%.9g, %.9g) want (%.9g, %.9g)", i,
                         (double)cosines[i], (double)sines[i], cos(angle),
                         sin(angle));
            }
        }
        /* Position 0 must be the identity rotation. */
        mynah_transformer_ar_rope_angles_f32(cosines, sines, half, 0, 10000.0f);
        for (size_t i = 0; i < half; ++i) {
            if (cosines[i] != 1.0f || sines[i] != 0.0f) {
                TAR_FAIL("rope at position 0 is not the identity");
            }
        }
    }

    /* 2. RoPE application: interleaved pairs, not split halves. */
    {
        float values[8] = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, -1.0f, 0.5f};
        const float cosines[2] = {0.0f, 1.0f};
        const float sines[2] = {1.0f, 0.0f};
        mynah_transformer_ar_rope_apply_f32(values, 2u, 4u, cosines, sines);
        /* Pair 0 turns by 90 degrees, pair 1 does not:
         *   head 0: (1,0) -> (0,1);  (0,1) -> (0,1).
         *   head 1: (2,3) -> (-3,2); (-1,0.5) -> (-1,0.5).
         * A split-halves RoPE would pair (0,2) and (1,3) and produce
         * {0,0,1,1,...} here, so this distinguishes the two conventions. */
        const float want[8] = {0.0f, 1.0f, 0.0f, 1.0f,
                               -3.0f, 2.0f, -1.0f, 0.5f};
        for (size_t i = 0; i < 8u; ++i) {
            if (!tar_close(values[i], want[i], 1e-6)) {
                TAR_FAIL("rope apply[%zu] = %.9g want %.9g", i,
                         (double)values[i], (double)want[i]);
            }
        }
    }

    /* 3. LayerNorm with bias, at the backbone epsilon. */
    {
        const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float weight[4] = {2.0f, 0.5f, 1.0f, -1.0f};
        const float bias[4] = {0.25f, -0.5f, 1.0f, 0.0f};
        float got[4];
        double want[4];
        double dinput[4];
        for (size_t i = 0; i < 4u; ++i) dinput[i] = (double)input[i];
        mynah_layernorm_f32(input, weight, bias, got, 1u, 4u, 1e-5f);
        tar_ref_layernorm(dinput, weight, bias, want, 4u, 1e-5);
        for (size_t i = 0; i < 4u; ++i) {
            if (!tar_close(got[i], want[i], 1e-5)) {
                TAR_FAIL("layernorm[%zu] = %.9g want %.9g", i, (double)got[i],
                         want[i]);
            }
        }
    }

    /* 4. GELU-tanh: the array kernel has an Accelerate path, so check it and
     * not just the scalar helper it falls back to. */
    {
        float values[6] = {-3.0f, -0.5f, 0.0f, 0.5f, 1.0f, 4.0f};
        float scratch[6];
        double want[6];
        for (size_t i = 0; i < 6u; ++i) want[i] = tar_ref_gelu_tanh(values[i]);
        mynah_gelu_tanh_array(values, 6u, scratch);
        for (size_t i = 0; i < 6u; ++i) {
            if (!tar_close(values[i], want[i], 1e-5)) {
                TAR_FAIL("gelu_tanh[%zu] = %.9g want %.9g", i,
                         (double)values[i], want[i]);
            }
        }
    }

    tar_test_weights *store = calloc(1u, sizeof(*store));
    if (store == NULL) TAR_FAIL("self test: out of memory");
    tar_build_weights(store);

    float input[TAR_T][TAR_D];
    double dinput[TAR_T][TAR_D];
    for (size_t t = 0; t < TAR_T; ++t) {
        for (size_t i = 0; i < TAR_D; ++i) {
            input[t][i] = tar_fake(t * TAR_D + i, 7u) * 4.0f;
            dinput[t][i] = (double)input[t][i];
        }
    }

    /* 5. Causal attention, unlimited and windowed, against the batched
     *    double reference. */
    const size_t contexts[2] = {0, 2u};
    float got[TAR_T][TAR_D];
    for (size_t c = 0; c < 2u; ++c) {
        mynah_transformer_ar_config config;
        tar_test_config(&config, contexts[c]);
        mynah_transformer_ar_state *state =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        if (state == NULL) {
            free(store);
            return -1;
        }
        if (mynah_transformer_ar_check_weights(state, &store->weights, error,
                                               error_capacity) != 0) {
            mynah_transformer_ar_state_free(state);
            free(store);
            return -1;
        }
        if (mynah_transformer_ar_prefill(state, &store->weights, &input[0][0],
                                         TAR_T, &got[0][0]) != 0) {
            mynah_transformer_ar_state_free(state);
            free(store);
            TAR_FAIL("prefill failed at context %zu", contexts[c]);
        }
        /* Read the offset BEFORE the free: the message is the whole point of
         * this branch, and formatting it out of a freed state is a use-after-
         * free that gcc 15 sees and that would print whatever the allocator
         * left behind -- garbage in exactly the report someone is relying on. */
        const size_t got_offset = mynah_transformer_ar_state_offset(state);
        if (got_offset != TAR_T) {
            mynah_transformer_ar_state_free(state);
            free(store);
            TAR_FAIL("prefill left offset %zu, want %u", got_offset, TAR_T);
        }
        mynah_transformer_ar_state_free(state);

        double want[TAR_T][TAR_D];
        tar_ref_forward(&store->weights, dinput, TAR_T, contexts[c], want);
        for (size_t t = 0; t < TAR_T; ++t) {
            for (size_t i = 0; i < TAR_D; ++i) {
                if (!tar_close(got[t][i], want[t][i], 2e-5)) {
                    free(store);
                    TAR_FAIL("context %zu token %zu dim %zu: %.9g want %.9g",
                             contexts[c], t, i, (double)got[t][i], want[t][i]);
                }
            }
        }
    }

    /* The window must actually do something, otherwise test 5 passes with a
     * `context` that is quietly ignored. */
    {
        double unlimited[TAR_T][TAR_D], windowed[TAR_T][TAR_D];
        tar_ref_forward(&store->weights, dinput, TAR_T, 0, unlimited);
        tar_ref_forward(&store->weights, dinput, TAR_T, 2u, windowed);
        int differs = 0;
        for (size_t i = 0; i < TAR_D; ++i) {
            if (fabs(unlimited[TAR_T - 1u][i] - windowed[TAR_T - 1u][i]) >
                1e-9) {
                differs = 1;
            }
        }
        if (!differs) {
            free(store);
            TAR_FAIL("the context window changed nothing: the test is blind");
        }
    }

    /* 6. KV-cache continuity: prefill(N) + step == prefill(N + 1). */
    {
        mynah_transformer_ar_config config;
        tar_test_config(&config, 0);
        mynah_transformer_ar_state *split =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        mynah_transformer_ar_state *whole =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        if (split == NULL || whole == NULL) {
            mynah_transformer_ar_state_free(split);
            mynah_transformer_ar_state_free(whole);
            free(store);
            return -1;
        }
        float last_split[TAR_D];
        float all[TAR_T][TAR_D];
        int failed =
            mynah_transformer_ar_prefill(split, &store->weights, &input[0][0],
                                         TAR_T - 1u, NULL) != 0 ||
            mynah_transformer_ar_step(split, &store->weights,
                                      input[TAR_T - 1u], last_split) != 0 ||
            mynah_transformer_ar_prefill(whole, &store->weights, &input[0][0],
                                         TAR_T, &all[0][0]) != 0;
        mynah_transformer_ar_state_free(split);
        mynah_transformer_ar_state_free(whole);
        if (failed) {
            free(store);
            TAR_FAIL("KV continuity: a forward failed");
        }
        for (size_t i = 0; i < TAR_D; ++i) {
            if (last_split[i] != all[TAR_T - 1u][i]) {
                free(store);
                TAR_FAIL("KV continuity dim %zu: %.9g vs %.9g", i,
                         (double)last_split[i], (double)all[TAR_T - 1u][i]);
            }
        }
    }

    /* 7. layer_scale: NULL must be exactly nn.Identity(), i.e. bit-identical
     *    to an explicit vector of ones, and a real gain must change the
     *    output. */
    {
        mynah_transformer_ar_config config;
        tar_test_config(&config, 0);
        float plain[TAR_T][TAR_D], scaled[TAR_T][TAR_D];
        mynah_transformer_ar_state *state =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        if (state == NULL) {
            free(store);
            return -1;
        }
        int failed = mynah_transformer_ar_prefill(state, &store->weights,
                                                  &input[0][0], TAR_T,
                                                  &plain[0][0]) != 0;
        for (size_t l = 0; l < TAR_L; ++l) {
            store->layers[l].layer_scale_1 = store->ones;
            store->layers[l].layer_scale_2 = store->ones;
        }
        mynah_transformer_ar_state_reset(state);
        failed |= mynah_transformer_ar_prefill(state, &store->weights,
                                               &input[0][0], TAR_T,
                                               &scaled[0][0]) != 0;
        if (failed) {
            mynah_transformer_ar_state_free(state);
            free(store);
            TAR_FAIL("layer_scale: a forward failed");
        }
        for (size_t t = 0; t < TAR_T; ++t) {
            for (size_t i = 0; i < TAR_D; ++i) {
                if (plain[t][i] != scaled[t][i]) {
                    mynah_transformer_ar_state_free(state);
                    free(store);
                    TAR_FAIL("layer_scale NULL != ones at [%zu][%zu]: "
                             "%.9g vs %.9g",
                             t, i, (double)plain[t][i], (double)scaled[t][i]);
                }
            }
        }
        /* A non-unit gain must move the answer, and must match the reference
         * with the same gain. */
        float gains[TAR_D];
        for (size_t i = 0; i < TAR_D; ++i) gains[i] = 0.01f + 0.001f * (float)i;
        for (size_t l = 0; l < TAR_L; ++l) {
            store->layers[l].layer_scale_1 = gains;
            store->layers[l].layer_scale_2 = gains;
        }
        mynah_transformer_ar_state_reset(state);
        failed = mynah_transformer_ar_prefill(state, &store->weights,
                                              &input[0][0], TAR_T,
                                              &scaled[0][0]) != 0;
        mynah_transformer_ar_state_free(state);
        if (failed) {
            free(store);
            TAR_FAIL("layer_scale: the scaled forward failed");
        }
        double want[TAR_T][TAR_D];
        tar_ref_forward(&store->weights, dinput, TAR_T, 0, want);
        int differs = 0;
        for (size_t t = 0; t < TAR_T; ++t) {
            for (size_t i = 0; i < TAR_D; ++i) {
                if (!tar_close(scaled[t][i], want[t][i], 2e-5)) {
                    free(store);
                    TAR_FAIL("layer_scale token %zu dim %zu: %.9g want %.9g", t,
                             i, (double)scaled[t][i], want[t][i]);
                }
                if (scaled[t][i] != plain[t][i]) differs = 1;
            }
        }
        if (!differs) {
            free(store);
            TAR_FAIL("layer_scale had no effect on the output");
        }
        for (size_t l = 0; l < TAR_L; ++l) {
            store->layers[l].layer_scale_1 = NULL;
            store->layers[l].layer_scale_2 = NULL;
        }
    }

    /* 8. The NaN guards: a non-finite input and a non-finite voice prefix must
     *    both be refused, not propagated. */
    {
        mynah_transformer_ar_config config;
        tar_test_config(&config, 0);
        mynah_transformer_ar_state *state =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        if (state == NULL) {
            free(store);
            return -1;
        }
        float poisoned[TAR_D], out[TAR_D];
        for (size_t i = 0; i < TAR_D; ++i) poisoned[i] = input[0][i];
        poisoned[3] = (float)NAN;
        const int rejected =
            mynah_transformer_ar_step(state, &store->weights, poisoned, out);
        float kv[2u * 2u * TAR_D];
        for (size_t i = 0; i < 2u * 2u * TAR_D; ++i) kv[i] = 0.125f;
        kv[5] = (float)NAN;
        const int kv_rejected = mynah_transformer_ar_state_load_kv(
            state, 0, kv, 2u, NULL, 0);
        const int overflow = mynah_transformer_ar_state_set_offset(
            state, config.max_seq_len + 1u, NULL, 0);
        mynah_transformer_ar_state_free(state);
        if (rejected == 0 || kv_rejected == 0 || overflow == 0) {
            free(store);
            TAR_FAIL("a NaN or an out-of-range offset was accepted "
                     "(step=%d kv=%d offset=%d)",
                     rejected, kv_rejected, overflow);
        }
    }

    /* 9. A voice prefix loads with no layout conversion and then shifts the
     *    answer, which is the whole point of the [2][T][H][D] cache. */
    {
        mynah_transformer_ar_config config;
        tar_test_config(&config, 0);
        mynah_transformer_ar_state *state =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        if (state == NULL) {
            free(store);
            return -1;
        }
        float prefix[2u * 2u * TAR_D];
        for (size_t i = 0; i < 2u * 2u * TAR_D; ++i) {
            prefix[i] = tar_fake(i, 42u);
        }
        int failed = 0;
        for (size_t l = 0; l < TAR_L; ++l) {
            failed |= mynah_transformer_ar_state_load_kv(state, l, prefix, 2u,
                                                         error,
                                                         error_capacity) != 0;
        }
        failed |= mynah_transformer_ar_state_set_offset(state, 2u, error,
                                                        error_capacity) != 0;
        float with_prefix[TAR_D];
        failed |= mynah_transformer_ar_step(state, &store->weights, input[0],
                                            with_prefix) != 0;
        const size_t offset = mynah_transformer_ar_state_offset(state);
        /* The K half must be readable back exactly as written. */
        const float *cache = mynah_transformer_ar_state_kv(state, 0);
        const size_t kv_half = mynah_transformer_ar_state_kv_half_floats(state);
        int copied = (cache != NULL) && (kv_half == config.max_seq_len * TAR_D);
        if (copied) {
            for (size_t i = 0; i < 2u * TAR_D; ++i) {
                if (cache[i] != prefix[i] ||
                    cache[kv_half + i] != prefix[2u * TAR_D + i]) {
                    copied = 0;
                }
            }
        }
        mynah_transformer_ar_state_free(state);
        if (failed || offset != 3u || !copied) {
            free(store);
            TAR_FAIL("voice prefix load failed (failed=%d offset=%zu copied=%d)",
                     failed, offset, copied);
        }
        /* Same input, no prefix: the two must differ, or the cached positions
         * were never attended to. */
        state = mynah_transformer_ar_state_new(&config, error, error_capacity);
        if (state == NULL) {
            free(store);
            return -1;
        }
        float bare[TAR_D];
        failed = mynah_transformer_ar_step(state, &store->weights, input[0],
                                          bare) != 0;
        mynah_transformer_ar_state_free(state);
        if (failed) {
            free(store);
            TAR_FAIL("voice prefix baseline step failed");
        }
        int differs = 0;
        for (size_t i = 0; i < TAR_D; ++i) {
            if (with_prefix[i] != bare[i]) differs = 1;
        }
        if (!differs) {
            free(store);
            TAR_FAIL("the loaded voice prefix was not attended to");
        }
    }

    /* 10. THE WINDOWED KV CACHE IS THE SAME TRANSFORMER.
     *
     *     A `context` window means the cache can be `context + slack` slots
     *     with slot 0 at a moving absolute position, instead of one slot per
     *     position the utterance could ever reach.  On the pinned pack that is
     *     500 instead of 24016, i.e. 196 MB of address space that was never
     *     reachable.  It is only allowed to be a memory change: the numbers
     *     must be BIT-IDENTICAL, or a request's audio would depend on how much
     *     memory the process felt like using.
     *
     *     Both caches run in one process over the same weights and inputs, and
     *     the run is long enough to force several compactions -- without that
     *     the two arms would be the same code and the test would be blind, so
     *     the compaction count is asserted too, through the base having moved.
     */
    {
        enum { TAR_CTX = 5u, TAR_LONG = 48u };
        for (size_t l = 0; l < TAR_L; ++l) {
            store->layers[l].layer_scale_1 = NULL;
            store->layers[l].layer_scale_2 = NULL;
        }
        mynah_transformer_ar_config config;
        tar_test_config(&config, TAR_CTX);
        config.max_seq_len = TAR_LONG;
        float full[TAR_LONG][TAR_D];
        float windowed[TAR_LONG][TAR_D];
        size_t kv_full = 0, kv_win = 0;
        int failed = 0;
        for (int arm = 0; arm < 2; ++arm) {
            const int before = mynah_transformer_ar_kv_window_force(arm);
            mynah_transformer_ar_state *state =
                mynah_transformer_ar_state_new(&config, error, error_capacity);
            if (state == NULL) {
                (void)mynah_transformer_ar_kv_window_force(before);
                free(store);
                return -1;
            }
            const size_t slots =
                mynah_transformer_ar_state_kv_half_floats(state) /
                (TAR_H * TAR_HD);
            if (arm == 0) kv_full = slots; else kv_win = slots;
            float *out = (arm == 0) ? &full[0][0] : &windowed[0][0];
            for (size_t t = 0; t < (size_t)TAR_LONG; ++t) {
                float in[TAR_D];
                for (size_t i = 0; i < TAR_D; ++i) in[i] = tar_fake(t * TAR_D + i, 77u);
                failed |= mynah_transformer_ar_step(state, &store->weights, in,
                                                    out + t * TAR_D) != 0;
            }
            mynah_transformer_ar_state_free(state);
            (void)mynah_transformer_ar_kv_window_force(before);
        }
        if (failed) {
            free(store);
            TAR_FAIL("the windowed KV comparison failed to step");
        }
        /* Not vacuous: the windowed arm must really be shorter, and short
         * enough that TAR_LONG positions cannot fit without compacting. */
        if (kv_full != (size_t)TAR_LONG || kv_win >= (size_t)TAR_LONG) {
            free(store);
            TAR_FAIL("the two arms allocated the same cache (%zu vs %zu "
                     "positions): the test is blind",
                     kv_full, kv_win);
        }
        if (memcmp(full, windowed, sizeof full) != 0) {
            size_t bad_t = 0, bad_i = 0;
            for (size_t t = 0; t < (size_t)TAR_LONG && bad_t == 0; ++t) {
                for (size_t i = 0; i < TAR_D; ++i) {
                    if (memcmp(&full[t][i], &windowed[t][i], sizeof(float)) != 0) {
                        bad_t = t + 1u; bad_i = i; break;
                    }
                }
            }
            free(store);
            TAR_FAIL("the windowed KV cache changed the answer at position "
                     "%zu dim %zu: %.9g vs %.9g (%zu slots vs %zu)",
                     bad_t - 1u, bad_i, (double)full[bad_t - 1u][bad_i],
                     (double)windowed[bad_t - 1u][bad_i], kv_full, kv_win);
        }
    }

    /* 9. A prefill wider than one tile is the same function as the same
     *    positions stepped one at a time -- bit for bit.  The tile is a
     *    scheduling unit and must never become a numerical one, and at
     *    TAR_T = 6 nothing above ever crosses a tile boundary. */
    {
        enum { TAR_WIDE = 35u }; /* 16 + 16 + 3 at the current tile */
        if (mynah_transformer_ar_prefill_tile() >= (size_t)TAR_WIDE) {
            free(store);
            TAR_FAIL("the tile grew past the multi-tile test: raise TAR_WIDE");
        }
        for (size_t l = 0; l < TAR_L; ++l) {
            store->layers[l].layer_scale_1 = NULL;
            store->layers[l].layer_scale_2 = NULL;
        }
        mynah_transformer_ar_config config;
        tar_test_config(&config, 0);
        config.max_seq_len = TAR_WIDE;
        float wide[TAR_WIDE][TAR_D];
        float tiled[TAR_WIDE][TAR_D];
        float stepped[TAR_WIDE][TAR_D];
        for (size_t t = 0; t < (size_t)TAR_WIDE; ++t) {
            for (size_t i = 0; i < TAR_D; ++i) {
                wide[t][i] = tar_fake(t * TAR_D + i, 19u) * 4.0f;
            }
        }
        mynah_transformer_ar_state *a =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        mynah_transformer_ar_state *b =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        if (a == NULL || b == NULL) {
            mynah_transformer_ar_state_free(a);
            mynah_transformer_ar_state_free(b);
            free(store);
            return -1;
        }
        int failed = mynah_transformer_ar_prefill(a, &store->weights, &wide[0][0],
                                                  TAR_WIDE, &tiled[0][0]) != 0;
        for (size_t t = 0; t < (size_t)TAR_WIDE && !failed; ++t) {
            failed = mynah_transformer_ar_step(b, &store->weights, wide[t],
                                               stepped[t]) != 0;
        }
        const size_t offset_a = mynah_transformer_ar_state_offset(a);
        const size_t offset_b = mynah_transformer_ar_state_offset(b);
        mynah_transformer_ar_state_free(a);
        mynah_transformer_ar_state_free(b);
        if (failed) {
            free(store);
            TAR_FAIL("multi-tile prefill: a forward failed");
        }
        if (offset_a != (size_t)TAR_WIDE || offset_b != (size_t)TAR_WIDE) {
            free(store);
            TAR_FAIL("multi-tile prefill left offsets %zu / %zu, want %u",
                     offset_a, offset_b, TAR_WIDE);
        }
        for (size_t t = 0; t < (size_t)TAR_WIDE; ++t) {
            for (size_t i = 0; i < TAR_D; ++i) {
                if (tiled[t][i] != stepped[t][i]) {
                    free(store);
                    TAR_FAIL("multi-tile prefill [%zu][%zu]: %.9g vs stepped "
                             "%.9g",
                             t, i, (double)tiled[t][i], (double)stepped[t][i]);
                }
            }
        }
    }

    /* 9b. The same equality with the WINDOW ENGAGED, at a scale no other case
     *     here reaches. Test 9 above runs at `context = 0`, so until this was
     *     added the tile path had never been run windowed at any length, and
     *     the longest sequence in this file was 35 positions. That matters
     *     because the production window is `context = 250` on a transformer
     *     that runs at 200 Hz (12.5 Hz latents x `codec_upsample_stride = 16`),
     *     i.e. it engages after 1.25 s of audio on every utterance -- and a
     *     defect whose threshold sits above a handful of positions, such as a
     *     span capped at a block size or a RoPE table valid only to 256, is
     *     invisible to every other case in this file. Two such mutations were
     *     written and confirmed silent against this self test before this case
     *     existed (`.work/transformer-ar-sliding-window.md`).
     *
     *     `context` deliberately does not divide the tile, so the engage
     *     boundary lands mid-tile rather than on its edge. The exhaustive
     *     version -- an f64 reference, the receptive field, RoPE's base and a
     *     ragged batch, all at C = 250 -- is
     *     `tests/test_transformer_ar_window.c` (`make window-test`); this is
     *     the part that has to travel inside the shipped binary. */
    {
        enum { TAR_LONG = 300u, TAR_CTX = 70u };
        mynah_transformer_ar_config config;
        tar_test_config(&config, TAR_CTX);
        config.max_seq_len = TAR_LONG;
        if (TAR_CTX % mynah_transformer_ar_prefill_tile() == 0u) {
            free(store);
            TAR_FAIL("the tile now divides the test window: the engage "
                     "boundary would never fall inside a tile");
        }
        float *wide = calloc((size_t)TAR_LONG * 3u * TAR_D, sizeof(float));
        if (wide == NULL) {
            free(store);
            TAR_FAIL("out of memory for the long windowed case");
        }
        float *tiled = wide + (size_t)TAR_LONG * TAR_D;
        float *stepped = tiled + (size_t)TAR_LONG * TAR_D;
        for (size_t i = 0; i < (size_t)TAR_LONG * TAR_D; ++i) {
            wide[i] = tar_fake(i, 47u) * 4.0f;
        }
        mynah_transformer_ar_state *a =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        mynah_transformer_ar_state *b =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        int failed = a == NULL || b == NULL;
        int wrapped = 0;
        if (!failed) {
            failed = mynah_transformer_ar_prefill(a, &store->weights, wide,
                                                  TAR_LONG, tiled) != 0;
        }
        for (size_t t = 0; t < (size_t)TAR_LONG && !failed; ++t) {
            failed = mynah_transformer_ar_step(b, &store->weights,
                                               wide + t * TAR_D,
                                               stepped + t * TAR_D) != 0;
        }
        /* A step past the capacity must be refused, not wrapped: the KV cache
         * is indexed by absolute position and `max_seq_len` is a hard cap, not
         * a recycling point.  A window does NOT make the cache a ring. */
        if (!failed) {
            wrapped = mynah_transformer_ar_step(b, &store->weights, wide,
                                                stepped) != -1;
        }
        mynah_transformer_ar_state_free(a);
        mynah_transformer_ar_state_free(b);
        if (failed || wrapped) {
            free(wide);
            free(store);
            TAR_FAIL("long windowed prefill: %s",
                     wrapped ? "a step past max_seq_len was accepted, so the "
                               "KV cache is behaving like a ring"
                             : "a forward failed");
        }
        for (size_t t = 0; t < (size_t)TAR_LONG; ++t) {
            for (size_t i = 0; i < TAR_D; ++i) {
                const size_t at = t * TAR_D + i;
                if (tiled[at] != stepped[at]) {
                    /* Both pointers alias `wide`, so the two values have to be
                     * copied out before it is freed.  Reading them afterwards
                     * is a use-after-free, and it corrupts the one message that
                     * explains the failure. */
                    const double a_val = (double)tiled[at];
                    const double b_val = (double)stepped[at];
                    const size_t start = tar_window_start(t, TAR_CTX);
                    free(wide);
                    free(store);
                    TAR_FAIL("long windowed prefill [%zu][%zu]: %.9g vs "
                             "stepped %.9g (window start %zu)",
                             t, i, a_val, b_val, start);
                }
            }
        }
        free(wide);
    }

    /* 9c. The window's EDGE, asserted directly rather than by agreement
     *     between two of our own paths.  9b compares the tile against the step,
     *     which is a self-consistency check: a defect that moves both -- a span
     *     capped at a block size, say -- passes it, and did, until this was
     *     added.
     *
     *     With ONE layer the set of inputs that can reach the output at p is
     *     exactly the window [p - C + 1, p].  So perturb the input at one
     *     position and look at which outputs move: outside that range every
     *     output must come back bit identical, inside it every output must
     *     change.  Two-sided, so a window one position too wide fails the first
     *     half and one too narrow fails the second.
     *
     *     The perturbation VARIES across the dimension on purpose.  A constant
     *     offset is LayerNorm's null space -- norm1 subtracts the mean -- so it
     *     perturbs nothing and every "it changed" assertion would pass on
     *     rounding noise instead.  Measured: constant bump 7e-7 at p == j and
     *     2e-7 after it, varying bump 3.7 and 0.1-0.5. */
    {
        enum { TAR_FIELD = 220u, TAR_FCTX = 70u, TAR_FJ = 40u };
        mynah_transformer_ar_config config;
        tar_test_config(&config, TAR_FCTX);
        config.max_seq_len = TAR_FIELD;
        config.num_layers = 1u; /* one layer, so the field is exactly the window */
        float *buf = calloc((size_t)TAR_FIELD * 4u * TAR_D, sizeof(float));
        if (buf == NULL) {
            free(store);
            TAR_FAIL("out of memory for the receptive-field case");
        }
        float *base_in = buf;
        float *bump_in = base_in + (size_t)TAR_FIELD * TAR_D;
        float *base_out = bump_in + (size_t)TAR_FIELD * TAR_D;
        float *bump_out = base_out + (size_t)TAR_FIELD * TAR_D;
        for (size_t i = 0; i < (size_t)TAR_FIELD * TAR_D; ++i) {
            base_in[i] = tar_fake(i, 53u) * 4.0f;
            bump_in[i] = base_in[i];
        }
        for (size_t i = 0; i < TAR_D; ++i) {
            bump_in[(size_t)TAR_FJ * TAR_D + i] += tar_fake(i, 59u) * 8.0f;
        }
        mynah_transformer_ar_state *a =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        mynah_transformer_ar_state *b =
            mynah_transformer_ar_state_new(&config, error, error_capacity);
        int failed = a == NULL || b == NULL;
        for (size_t t = 0; t < (size_t)TAR_FIELD && !failed; ++t) {
            failed = mynah_transformer_ar_step(a, &store->weights,
                                               base_in + t * TAR_D,
                                               base_out + t * TAR_D) != 0 ||
                     mynah_transformer_ar_step(b, &store->weights,
                                               bump_in + t * TAR_D,
                                               bump_out + t * TAR_D) != 0;
        }
        mynah_transformer_ar_state_free(a);
        mynah_transformer_ar_state_free(b);
        if (failed) {
            free(buf);
            free(store);
            TAR_FAIL("receptive-field case: a forward failed");
        }
        for (size_t t = 0; t < (size_t)TAR_FIELD; ++t) {
            int moved = 0;
            for (size_t i = 0; i < TAR_D; ++i) {
                if (base_out[t * TAR_D + i] != bump_out[t * TAR_D + i]) moved = 1;
            }
            const size_t lo = tar_window_start(t, TAR_FCTX);
            const int reachable = (t >= (size_t)TAR_FJ) && (lo <= (size_t)TAR_FJ);
            if (moved != reachable) {
                free(buf);
                free(store);
                TAR_FAIL("output %zu %s though position %u is %s its window "
                         "[%zu, %zu]: the window reaches %s",
                         t, moved ? "moved" : "did not move", TAR_FJ,
                         reachable ? "inside" : "outside", lo, t,
                         moved ? "too far back" : "not far enough back");
            }
        }
        free(buf);
    }

    /* 10. `_step_batch` of N states is the same function as those N states
     *     stepped alone -- bit for bit, at ragged offsets, with a sliding
     *     window so the per-row `lo` differs, and once more through a
     *     row-pointer hook so the pointer plumbing is exercised rather than
     *     only the fallback. */
    {
        enum { TAR_NB = 3u };
        const size_t prefix[TAR_NB] = {1u, 3u, 5u};
        mynah_transformer_ar_config config;
        tar_test_config(&config, 2u); /* a window, so `lo` is row-dependent */
        config.max_seq_len = TAR_T + 2u;
        for (int hooked = 0; hooked < 2; ++hooked) {
            store->weights.linear_rows = hooked ? tar_test_linear_rows : NULL;
            store->weights.linear_user = NULL;
            mynah_transformer_ar_state *batched[TAR_NB] = {NULL, NULL, NULL};
            mynah_transformer_ar_state *solo[TAR_NB] = {NULL, NULL, NULL};
            mynah_transformer_ar_batch *scratch =
                mynah_transformer_ar_batch_new(&config, TAR_NB, error,
                                               error_capacity);
            int failed = scratch == NULL;
            for (size_t b = 0; b < (size_t)TAR_NB && !failed; ++b) {
                batched[b] = mynah_transformer_ar_state_new(&config, error,
                                                            error_capacity);
                solo[b] = mynah_transformer_ar_state_new(&config, error,
                                                         error_capacity);
                failed = batched[b] == NULL || solo[b] == NULL ||
                         mynah_transformer_ar_prefill(batched[b], &store->weights,
                                                      &input[0][0], prefix[b],
                                                      NULL) != 0 ||
                         mynah_transformer_ar_prefill(solo[b], &store->weights,
                                                      &input[0][0], prefix[b],
                                                      NULL) != 0;
            }
            float got[TAR_NB][TAR_D], want[TAR_NB][TAR_D];
            const float *xs[TAR_NB];
            float *os[TAR_NB];
            for (size_t b = 0; b < (size_t)TAR_NB; ++b) {
                xs[b] = input[b % TAR_T];
                os[b] = got[b];
            }
            if (!failed) {
                failed = mynah_transformer_ar_step_batch(batched, TAR_NB,
                                                         &store->weights, scratch,
                                                         xs, os) != 0;
            }
            for (size_t b = 0; b < (size_t)TAR_NB && !failed; ++b) {
                failed = mynah_transformer_ar_step(solo[b], &store->weights, xs[b],
                                                   want[b]) != 0;
            }
            size_t offsets[TAR_NB] = {0, 0, 0};
            for (size_t b = 0; b < (size_t)TAR_NB; ++b) {
                offsets[b] = mynah_transformer_ar_state_offset(batched[b]);
                mynah_transformer_ar_state_free(batched[b]);
                mynah_transformer_ar_state_free(solo[b]);
            }
            mynah_transformer_ar_batch_free(scratch);
            if (failed) {
                store->weights.linear_rows = NULL;
                free(store);
                TAR_FAIL("step_batch (hooked=%d): a forward failed", hooked);
            }
            for (size_t b = 0; b < (size_t)TAR_NB; ++b) {
                if (offsets[b] != prefix[b] + 1u) {
                    store->weights.linear_rows = NULL;
                    free(store);
                    TAR_FAIL("step_batch left row %zu at offset %zu, want %zu", b,
                             offsets[b], prefix[b] + 1u);
                }
                for (size_t i = 0; i < TAR_D; ++i) {
                    if (got[b][i] != want[b][i]) {
                        store->weights.linear_rows = NULL;
                        free(store);
                        TAR_FAIL("step_batch (hooked=%d) row %zu dim %zu: %.9g "
                                 "vs solo %.9g",
                                 hooked, b, i, (double)got[b][i],
                                 (double)want[b][i]);
                    }
                }
            }
            /* If every row came out the same the comparison above could not
             * have seen one row's state leaking into another. */
            int distinct = 0;
            for (size_t i = 0; i < TAR_D; ++i) {
                if (got[0][i] != got[1][i] || got[1][i] != got[2][i]) distinct = 1;
            }
            if (!distinct) {
                store->weights.linear_rows = NULL;
                free(store);
                TAR_FAIL("step_batch produced three identical rows: the test is "
                         "blind to cross-talk");
            }
        }
        store->weights.linear_rows = NULL;
    }

    free(store);
    return 0;
}

#undef TAR_FAIL
