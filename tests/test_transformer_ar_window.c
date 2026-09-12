/*
 * The sliding window of `src/transformer_ar.c`, past `context: 250`.
 *
 * `context` is the Mimi decoder transformer's attention window. It is not the
 * long-form edge case its name suggests: that transformer sits after
 * `mimi.upsample` (`codec_upsample_stride = 16`) and so runs at 12.5 * 16 =
 * 200 Hz, which makes 250 positions **1.25 s of audio**. Every PocketTTS
 * utterance longer than that already runs the windowed path in production,
 * while `tools/oracle_pocket.py` records only the first four calls of any
 * module. So the code path that serves essentially all real audio has never
 * been compared against anything.
 *
 * The oracle needs uv, torch and gated weights, none of which are a build
 * dependency. This file therefore builds the guarantee a different way --
 * model-free, with synthetic weights -- which is weaker in one respect (it
 * cannot tell us our window rule is Kyutai's) and stronger in another (it runs
 * in `make test`, and therefore under UBSan and ASan, on every machine with no
 * checkpoint present). What a dump would still settle, and the exact command,
 * is in `.work/transformer-ar-sliding-window.md`.
 *
 * The contract under test, stated there as W1-W7 and reproduced here because a
 * test that does not say what it believes is only a change detector:
 *
 *   W2  a query at absolute position p attends to the closed range [lo, p],
 *       lo = (p + 1 <= C) ? 0 : p + 1 - C,  span = min(p + 1, C).
 *   W3  p = C - 1 is the last position that sees position 0; p = C is the
 *       first that does not.
 *   W4  the KV cache is absolute-indexed and does NOT wrap: max_seq_len is a
 *       hard cap, refused before any mutation, not a recycling point.
 *   W5  RoPE is absolute and never re-based; k is stored post-rotation, so a
 *       key keeps its own position's rotation forever.
 *   W6  `context` is shared by every state in a batch, but `lo` is per row.
 *   W7  the prefill tile is a scheduling unit, never a numerical one.
 *
 * Four of the cases below are tolerance-free on purpose. `receptive-field` and
 * `capacity` assert *bit* equality or a hard `-1`; `tile-vs-step` and
 * `ragged-batch` assert bit equality between two paths that must compute the
 * same function. Only `reference` (f32 against f64) and `translation` (which
 * compares R(p)q.R(i)k against q.R(i-p)k, equal in exact arithmetic and not in
 * f32) carry tolerances, and both print their margins.
 */
#include "kernels.h"
#include "transformer_ar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ reporting */

static int g_failures;
static const char *g_case = "?";

#define FAILF(...)                                       \
    do {                                                 \
        fprintf(stderr, "FAIL [%s]: ", g_case);          \
        fprintf(stderr, __VA_ARGS__);                    \
        fputc('\n', stderr);                             \
        ++g_failures;                                    \
        goto done;                                       \
    } while (0)

#define REQUIRE(cond, ...)                               \
    do {                                                 \
        if (!(cond)) FAILF(__VA_ARGS__);                 \
    } while (0)

/* --------------------------------------------------------------- weights */

/* Deterministic and small, so the f64 reference below stays meaningful. */
static float fake(size_t index, unsigned salt) {
    unsigned x = (unsigned)(index * 2654435761u) ^ (salt * 40503u);
    x ^= x >> 13;
    x *= 2246822519u;
    x ^= x >> 17;
    return (float)((double)(x % 20011u) / 20011.0 - 0.5) * 0.5f;
}

/*
 * d_model != attn_dim on purpose: the self test in transformer_ar.c has
 * d_model == num_heads * head_dim == 8, so a projection that confused the two
 * would pass there. Here d_model = 24 and attn_dim = 16.
 */
#define WD 24u  /* d_model  */
#define WH 2u   /* heads    */
#define WHD 8u  /* head_dim */
#define WF 40u  /* ffn_dim  */
#define WA (WH * WHD)
#define WLMAX 2u

typedef struct {
    float in_proj[WLMAX][3u * WA * WD];
    float out_proj[WLMAX][WD * WA];
    float norm1_w[WLMAX][WD], norm1_b[WLMAX][WD];
    float norm2_w[WLMAX][WD], norm2_b[WLMAX][WD];
    float linear1[WLMAX][WF * WD];
    float linear2[WLMAX][WD * WF];
    float out_norm_w[WD], out_norm_b[WD];
    mynah_transformer_ar_layer layers[WLMAX];
    mynah_transformer_ar_weights weights;
} win_weights;

static void build_weights(win_weights *w, size_t layers) {
    unsigned salt = 11u;
    memset(w, 0, sizeof(*w));
    for (size_t l = 0; l < layers; ++l) {
        for (size_t i = 0; i < 3u * WA * WD; ++i) w->in_proj[l][i] = fake(i, salt);
        ++salt;
        for (size_t i = 0; i < WD * WA; ++i) w->out_proj[l][i] = fake(i, salt);
        ++salt;
        for (size_t i = 0; i < WD; ++i) {
            w->norm1_w[l][i] = 1.0f + fake(i, salt);
            w->norm1_b[l][i] = fake(i, salt + 1u);
            w->norm2_w[l][i] = 1.0f + fake(i, salt + 2u);
            w->norm2_b[l][i] = fake(i, salt + 3u);
        }
        salt += 4u;
        for (size_t i = 0; i < WF * WD; ++i) w->linear1[l][i] = fake(i, salt);
        for (size_t i = 0; i < WD * WF; ++i) w->linear2[l][i] = fake(i, salt + 1u);
        salt += 2u;

        mynah_transformer_ar_layer *layer = &w->layers[l];
        layer->in_proj_weight = w->in_proj[l];
        layer->out_proj_weight = w->out_proj[l];
        layer->norm1_weight = w->norm1_w[l];
        layer->norm1_bias = w->norm1_b[l];
        layer->norm2_weight = w->norm2_w[l];
        layer->norm2_bias = w->norm2_b[l];
        layer->linear1_weight = w->linear1[l];
        layer->linear2_weight = w->linear2[l];
    }
    for (size_t i = 0; i < WD; ++i) {
        w->out_norm_w[i] = 1.0f + fake(i, 97u);
        w->out_norm_b[i] = fake(i, 98u);
    }
    w->weights.layers = w->layers;
    w->weights.out_norm_weight = w->out_norm_w;
    w->weights.out_norm_bias = w->out_norm_b;
}

static void win_config(mynah_transformer_ar_config *config, size_t layers,
                       size_t context, size_t max_seq_len) {
    mynah_transformer_ar_config_defaults(config);
    config->d_model = WD;
    config->num_heads = WH;
    config->head_dim = WHD;
    config->num_layers = layers;
    config->ffn_dim = WF;
    config->max_seq_len = max_seq_len;
    config->context = context;
}

/* ------------------------------------------------------- f64 reference */

/*
 * An independent implementation of the contract, in double precision, over an
 * arbitrary number of positions. It is deliberately written from W1-W7 rather
 * than transcribed from `transformer_ar.c`: the window bound below is the
 * contract's `lo`, spelled out, and nothing here shares a line with the code
 * under test.
 *
 * Layer-by-layer over all positions at once -- the batched shape -- so agreeing
 * with it proves the token-sequential implementation computes what a masked
 * SDPA would.
 */
typedef struct {
    double *x;  /* [n][WD]  */
    double *q;  /* [n][WA]  */
    double *k;  /* [n][WA]  */
    double *v;  /* [n][WA]  */
} ref_scratch;

static void ref_layernorm(const double *in, const float *weight,
                          const float *bias, double *out, size_t n, double eps) {
    double mean = 0.0, variance = 0.0;
    for (size_t i = 0; i < n; ++i) mean += in[i];
    mean /= (double)n;
    for (size_t i = 0; i < n; ++i) {
        const double d = in[i] - mean;
        variance += d * d;
    }
    variance /= (double)n;
    const double scale = 1.0 / sqrt(variance + eps);
    for (size_t i = 0; i < n; ++i) {
        double value = (in[i] - mean) * scale;
        if (weight != NULL) value *= (double)weight[i];
        if (bias != NULL) value += (double)bias[i];
        out[i] = value;
    }
}

static void ref_linear(const float *weight, const float *bias,
                       const double *in, double *out, size_t rows,
                       size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        double sum = (bias != NULL) ? (double)bias[r] : 0.0;
        for (size_t c = 0; c < cols; ++c) {
            sum += (double)weight[r * cols + c] * in[c];
        }
        out[r] = sum;
    }
}

static double ref_gelu_tanh(double x) {
    const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

/* W2, spelled out once, used by the reference and by the expectations below. */
static size_t contract_window_start(size_t position, size_t context) {
    if (context == 0u) return 0u;              /* W1 */
    if (position + 1u <= context) return 0u;   /* W3: still inside the prefix */
    return position + 1u - context;
}

static int ref_forward(const mynah_transformer_ar_weights *weights,
                       const mynah_transformer_ar_config *config,
                       const double *x_in, size_t n, double *out) {
    ref_scratch s;
    s.x = malloc(n * WD * sizeof(double));
    s.q = malloc(n * WA * sizeof(double));
    s.k = malloc(n * WA * sizeof(double));
    s.v = malloc(n * WA * sizeof(double));
    double *scores = malloc(n * sizeof(double));
    if (s.x == NULL || s.q == NULL || s.k == NULL || s.v == NULL ||
        scores == NULL) {
        free(s.x); free(s.q); free(s.k); free(s.v); free(scores);
        return -1;
    }
    memcpy(s.x, x_in, n * WD * sizeof(double));

    const size_t half = WHD / 2u;
    const double slope = -log((double)config->max_period) * 2.0 / (double)WHD;

    for (size_t l = 0; l < config->num_layers; ++l) {
        const mynah_transformer_ar_layer *layer = &weights->layers[l];
        for (size_t t = 0; t < n; ++t) {
            double norm[WD], qkv[3u * WA];
            ref_layernorm(s.x + t * WD, layer->norm1_weight, layer->norm1_bias,
                          norm, WD, (double)config->layernorm_eps);
            ref_linear(layer->in_proj_weight, layer->in_proj_bias, norm, qkv,
                       3u * WA, WD);
            /* W5: both q and k turn by the ABSOLUTE position t, and k is
             * stored already turned. Interleaved pairs (2i, 2i+1). */
            for (size_t h = 0; h < WH; ++h) {
                for (size_t i = 0; i < half; ++i) {
                    const double angle = exp((double)i * slope) * (double)t;
                    const double c = cos(angle), sn = sin(angle);
                    const size_t a = h * WHD + 2u * i;
                    const double qr = qkv[a], qi = qkv[a + 1u];
                    const double kr = qkv[WA + a], ki = qkv[WA + a + 1u];
                    s.q[t * WA + a] = qr * c - qi * sn;
                    s.q[t * WA + a + 1u] = qr * sn + qi * c;
                    s.k[t * WA + a] = kr * c - ki * sn;
                    s.k[t * WA + a + 1u] = kr * sn + ki * c;
                }
            }
            for (size_t i = 0; i < WA; ++i) s.v[t * WA + i] = qkv[2u * WA + i];
        }
        for (size_t t = 0; t < n; ++t) {
            double attn[WA], upd[WD], norm[WD], ffn[WF];
            const size_t lo = contract_window_start(t, config->context);
            for (size_t h = 0; h < WH; ++h) {
                double maximum = -INFINITY, total = 0.0;
                for (size_t j = lo; j <= t; ++j) {
                    double dot = 0.0;
                    for (size_t i = 0; i < WHD; ++i) {
                        dot += s.q[t * WA + h * WHD + i] * s.k[j * WA + h * WHD + i];
                    }
                    scores[j] = dot / sqrt((double)WHD);
                    if (scores[j] > maximum) maximum = scores[j];
                }
                for (size_t j = lo; j <= t; ++j) {
                    scores[j] = exp(scores[j] - maximum);
                    total += scores[j];
                }
                for (size_t i = 0; i < WHD; ++i) {
                    double sum = 0.0;
                    for (size_t j = lo; j <= t; ++j) {
                        sum += scores[j] * s.v[j * WA + h * WHD + i];
                    }
                    attn[h * WHD + i] = sum / total;
                }
            }
            ref_linear(layer->out_proj_weight, layer->out_proj_bias, attn, upd,
                       WD, WA);
            for (size_t i = 0; i < WD; ++i) s.x[t * WD + i] += upd[i];
            ref_layernorm(s.x + t * WD, layer->norm2_weight, layer->norm2_bias,
                          norm, WD, (double)config->layernorm_eps);
            ref_linear(layer->linear1_weight, layer->linear1_bias, norm, ffn, WF,
                       WD);
            for (size_t i = 0; i < WF; ++i) ffn[i] = ref_gelu_tanh(ffn[i]);
            ref_linear(layer->linear2_weight, layer->linear2_bias, ffn, upd, WD,
                       WF);
            for (size_t i = 0; i < WD; ++i) s.x[t * WD + i] += upd[i];
        }
    }
    for (size_t t = 0; t < n; ++t) {
        if (weights->out_norm_weight != NULL) {
            ref_layernorm(s.x + t * WD, weights->out_norm_weight,
                          weights->out_norm_bias, out + t * WD, WD,
                          (double)config->layernorm_eps);
        } else {
            memcpy(out + t * WD, s.x + t * WD, WD * sizeof(double));
        }
    }
    free(s.x); free(s.q); free(s.k); free(s.v); free(scores);
    return 0;
}

/* ------------------------------------------------------------- helpers */

static void fill_input(float *x, double *xd, size_t n, unsigned salt) {
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < WD; ++i) {
            const float value = fake(t * WD + i, salt) * 3.0f;
            x[t * WD + i] = value;
            if (xd != NULL) xd[t * WD + i] = (double)value;
        }
    }
}

/* Runs `n` positions one at a time through `_step`. */
static int run_stepped(const mynah_transformer_ar_config *config,
                       const mynah_transformer_ar_weights *weights,
                       const float *x, size_t n, float *out) {
    char error[256];
    mynah_transformer_ar_state *state =
        mynah_transformer_ar_state_new(config, error, sizeof(error));
    if (state == NULL) {
        fprintf(stderr, "state_new: %s\n", error);
        return -1;
    }
    int rc = 0;
    for (size_t t = 0; t < n && rc == 0; ++t) {
        rc = mynah_transformer_ar_step(state, weights, x + t * WD,
                                       out + t * WD);
    }
    if (rc == 0 && mynah_transformer_ar_state_offset(state) != n) rc = -1;
    mynah_transformer_ar_state_free(state);
    return rc;
}

static int run_prefilled(const mynah_transformer_ar_config *config,
                         const mynah_transformer_ar_weights *weights,
                         const float *x, size_t n, float *out) {
    char error[256];
    mynah_transformer_ar_state *state =
        mynah_transformer_ar_state_new(config, error, sizeof(error));
    if (state == NULL) {
        fprintf(stderr, "state_new: %s\n", error);
        return -1;
    }
    int rc = mynah_transformer_ar_prefill(state, weights, x, n, out);
    if (rc == 0 && mynah_transformer_ar_state_offset(state) != n) rc = -1;
    mynah_transformer_ar_state_free(state);
    return rc;
}

static double max_rel_error(const float *got, const double *want, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double diff = fabs((double)got[i] - want[i]);
        const double magnitude = fabs(want[i]) > 1.0 ? fabs(want[i]) : 1.0;
        const double rel = diff / magnitude;
        if (rel > worst) worst = rel;
    }
    return worst;
}

/* ------------------------------------------------- case: f64 reference */

static int case_reference(size_t layers, size_t context, size_t n) {
    static char name[96];
    snprintf(name, sizeof(name), "window/reference L=%zu C=%zu T=%zu", layers,
             context, n);
    g_case = name;

    win_weights *w = malloc(sizeof(*w));
    float *x = malloc(n * WD * sizeof(float));
    double *xd = malloc(n * WD * sizeof(double));
    float *got = malloc(n * WD * sizeof(float));
    double *want = malloc(n * WD * sizeof(double));
    int rc = -1;
    if (w == NULL || x == NULL || xd == NULL || got == NULL || want == NULL) {
        fprintf(stderr, "FAIL [%s]: out of memory\n", g_case);
        ++g_failures;
        goto done;
    }
    build_weights(w, layers);
    fill_input(x, xd, n, 3u);

    mynah_transformer_ar_config config;
    win_config(&config, layers, context, n);

    REQUIRE(run_stepped(&config, &w->weights, x, n, got) == 0,
            "the stepped forward failed");
    REQUIRE(ref_forward(&w->weights, &config, xd, n, want) == 0,
            "the reference forward failed");

    const double worst = max_rel_error(got, want, n * WD);
    /* f32 against f64 over up to `context` accumulations of a d_model = 24
     * matvec. 2e-4 is loose enough not to be a flake and tight enough that a
     * one-position window error (which changes the softmax denominator by a
     * whole term) cannot hide under it: the negative controls in the note
     * move this to O(1e-1). */
    REQUIRE(worst <= 2e-4, "max relative error %.3g exceeds 2e-4", worst);
    printf("  ok   %-44s max rel %.2e\n", name, worst);
    rc = 0;
done:
    free(w); free(x); free(xd); free(got); free(want);
    return rc;
}

/* --------------------------------------------- case: tile against step */

static int case_tile_vs_step(size_t layers, size_t context, size_t n) {
    static char name[96];
    snprintf(name, sizeof(name), "window/tile-vs-step L=%zu C=%zu T=%zu", layers,
             context, n);
    g_case = name;

    win_weights *w = malloc(sizeof(*w));
    float *x = malloc(n * WD * sizeof(float));
    float *stepped = malloc(n * WD * sizeof(float));
    float *tiled = malloc(n * WD * sizeof(float));
    int rc = -1;
    if (w == NULL || x == NULL || stepped == NULL || tiled == NULL) {
        fprintf(stderr, "FAIL [%s]: out of memory\n", g_case);
        ++g_failures;
        goto done;
    }
    build_weights(w, layers);
    fill_input(x, NULL, n, 7u);

    mynah_transformer_ar_config config;
    win_config(&config, layers, context, n);

    const size_t tile = mynah_transformer_ar_prefill_tile();
    REQUIRE(tile > 1u && n > 3u * tile,
            "the tile is %zu; this case needs several tiles", tile);
    /* The engage boundary must land inside a tile, not on its edge, or the
     * case cannot see a tile that is half windowed and half not. */
    REQUIRE(context == 0u || context % tile != 0u,
            "context %zu is tile-aligned; the boundary would never fall "
            "mid-tile",
            context);

    REQUIRE(run_stepped(&config, &w->weights, x, n, stepped) == 0,
            "the stepped forward failed");
    REQUIRE(run_prefilled(&config, &w->weights, x, n, tiled) == 0,
            "the tiled prefill failed");

    /* W7: bit for bit. The tile is a scheduling unit. */
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < WD; ++i) {
            const size_t at = t * WD + i;
            REQUIRE(tiled[at] == stepped[at],
                    "position %zu dim %zu: tiled %.9g vs stepped %.9g "
                    "(window start %zu)",
                    t, i, (double)tiled[at], (double)stepped[at],
                    contract_window_start(t, context));
        }
    }
    printf("  ok   %-44s bit-identical over %zu tiles\n", name,
           (n + tile - 1u) / tile);
    rc = 0;
done:
    free(w); free(x); free(stepped); free(tiled);
    return rc;
}

/* ------------------------------------------- case: the receptive field */

/*
 * The razor, and the one case that shares no arithmetic with the reference.
 *
 * By W2 a query at p reads keys [p - C + 1, p], so with one layer the set of
 * INPUTS that can reach output p is exactly that range. Perturb the input at
 * one position j and re-run: every output at p - j > C - 1 must come back bit
 * identical, and every output at 0 <= p - j <= C - 1 must differ. Both
 * directions are asserted, which is what pins the edge -- a window one position
 * too wide breaks the first, one too narrow breaks the second.
 *
 * With L layers the reachable set widens to [p - L*(C-1), p] because each layer
 * composes another window. The two-sided pin is therefore only exact at L = 1;
 * at L = 2 the test asserts the widened upper bound and the L-independent lower
 * one.
 */
static int case_receptive_field(size_t layers, size_t context, size_t n,
                                size_t j) {
    static char name[96];
    snprintf(name, sizeof(name), "window/receptive-field L=%zu C=%zu j=%zu",
             layers, context, j);
    g_case = name;

    win_weights *w = malloc(sizeof(*w));
    float *x = malloc(n * WD * sizeof(float));
    float *bumped = malloc(n * WD * sizeof(float));
    float *base_out = malloc(n * WD * sizeof(float));
    float *bump_out = malloc(n * WD * sizeof(float));
    int rc = -1;
    if (w == NULL || x == NULL || bumped == NULL || base_out == NULL ||
        bump_out == NULL) {
        fprintf(stderr, "FAIL [%s]: out of memory\n", g_case);
        ++g_failures;
        goto done;
    }
    build_weights(w, layers);
    fill_input(x, NULL, n, 13u);
    memcpy(bumped, x, n * WD * sizeof(float));
    /* The perturbation must VARY across the dimension. A constant offset is
     * exactly LayerNorm's null space -- norm1 subtracts the mean -- so
     * `+= 2.0f` here perturbs nothing, every output comes back equal to within
     * 2e-7 of rounding noise, and the "must differ" half of the assertion
     * below passes on that noise instead of on the window. Measured, not
     * theorised: with a constant bump the response at p == j was 7e-7 and at
     * every later position 2e-7; with this one it is 3.7 and 0.1-0.5. */
    for (size_t i = 0; i < WD; ++i) bumped[j * WD + i] += fake(i, 41u) * 8.0f;

    mynah_transformer_ar_config config;
    win_config(&config, layers, context, n);
    REQUIRE(context > 1u && j + layers * (context - 1u) + 2u < n,
            "the case needs room for the whole field plus two positions past it");

    REQUIRE(run_stepped(&config, &w->weights, x, n, base_out) == 0,
            "the baseline forward failed");
    REQUIRE(run_stepped(&config, &w->weights, bumped, n, bump_out) == 0,
            "the perturbed forward failed");

    const size_t reach = layers * (context - 1u); /* widest p - j that may move */
    for (size_t p = 0; p < n; ++p) {
        int differs = 0;
        for (size_t i = 0; i < WD; ++i) {
            if (base_out[p * WD + i] != bump_out[p * WD + i]) differs = 1;
        }
        if (p < j) {
            REQUIRE(!differs, "causality: output %zu moved when input %zu did",
                    p, j);
        } else if (p - j > reach) {
            REQUIRE(!differs,
                    "output %zu is %zu past the perturbed input %zu, beyond the "
                    "%zu-position field of a %zu-layer C=%zu window, and moved "
                    "anyway: the window reaches too far back",
                    p, p - j, j, reach, layers, context);
        } else if (p - j <= context - 1u) {
            REQUIRE(differs,
                    "output %zu is only %zu past the perturbed input %zu and "
                    "did not move: the window does not reach far enough back "
                    "(C=%zu, so it must still see it)",
                    p, p - j, j, context);
        }
    }
    printf("  ok   %-44s field exactly [%zu, %zu]\n", name, j, j + reach);
    rc = 0;
done:
    free(w); free(x); free(bumped); free(base_out); free(bump_out);
    return rc;
}

/* ------------------------------------------- case: translation invariance */

/*
 * The RoPE test, and the second, independent pin on the engage boundary.
 *
 * A first attempt fed the SAME vector at every position and expected the output
 * to settle once the window filled. It is vacuous, and the reason is worth
 * keeping: `v` is not rotated, so a constant input gives every position the
 * same `v`, and a convex combination of identical vectors is that vector
 * whatever the attention weights are. The measured "settling curve" was flat at
 * 3e-7 from position 0. A window test must make `v` vary.
 *
 * So: a constant background at every position except one, which holds a
 * different vector `a`. With one layer the pre-RoPE q, k, v are then the same
 * at every background position, and a score is
 *
 *     R(p)q . R(i)k  =  q . R(i - p) k,
 *
 * a function of the RELATIVE offset alone (W5). The output at `p` therefore
 * depends only on `p - j` and on the span. Two consequences, both asserted:
 *
 *   - **translation invariance.** Put `a` at two far-apart absolute positions,
 *     both well past the boundary so both windows are full. The responses at
 *     the same relative offset must match. A RoPE angle taken relative to `lo`,
 *     or one left stale across the boundary, makes the score depend on `i`
 *     absolutely and this drifts apart immediately.
 *   - **the boundary.** Put `a` at position 0 instead. Then position `d` has
 *     span `d + 1`, and by W3 the window is full for the first time at exactly
 *     `d = C - 1`. So the response must differ from the translated one for
 *     every `d <= C - 2` and match it from `d = C - 1` on. Measured at C = 13:
 *     2.4e-2 at d = 11 against 3.0e-7 at d = 12, a margin of five orders of
 *     magnitude, and the transition is at C - 1 and nowhere else.
 *
 * Tolerance-based, because R(p)q . R(i)k and q . R(i-p)k are equal in exact
 * arithmetic and not in f32. Both margins are printed.
 */
static int case_translation(size_t context, size_t n) {
    static char name[96];
    snprintf(name, sizeof(name), "window/translation C=%zu", context);
    g_case = name;

    /* The earliest fully-windowed position, and one two windows later: every
     * `d` compared below is fully windowed in both runs, and keeping the
     * absolute positions as small as that allows keeps the f32 error of
     * cos/sin at large angles out of the margin. */
    const size_t j1 = context - 1u;
    const size_t j2 = 3u * context;
    win_weights *w = malloc(sizeof(*w));
    float *x = malloc(n * WD * sizeof(float));
    float *at1 = malloc(n * WD * sizeof(float));
    float *at2 = malloc(n * WD * sizeof(float));
    float *at0 = malloc(n * WD * sizeof(float));
    int rc = -1;
    if (w == NULL || x == NULL || at1 == NULL || at2 == NULL || at0 == NULL) {
        fprintf(stderr, "FAIL [%s]: out of memory\n", g_case);
        ++g_failures;
        goto done;
    }
    build_weights(w, 1u);
    mynah_transformer_ar_config config;
    win_config(&config, 1u, context, n);
    REQUIRE(context > 2u && j2 + context + 2u < n,
            "the case needs room for two well-separated copies");

    const size_t spots[3] = {j1, j2, 0u};
    float *outs[3] = {at1, at2, at0};
    for (size_t c = 0; c < 3u; ++c) {
        for (size_t t = 0; t < n; ++t) {
            for (size_t i = 0; i < WD; ++i) {
                x[t * WD + i] = (t == spots[c]) ? fake(i, 55u) * 4.0f
                                                : fake(i, 23u) * 3.0f;
            }
        }
        REQUIRE(run_stepped(&config, &w->weights, x, n, outs[c]) == 0,
                "the forward with `a` at %zu failed", spots[c]);
    }

    double invariance = 0.0, matched = 0.0, boundary = 0.0;
    for (size_t d = 0; d < context + 2u; ++d) {
        for (size_t i = 0; i < WD; ++i) {
            const double drift = fabs((double)at1[(j1 + d) * WD + i] -
                                      (double)at2[(j2 + d) * WD + i]);
            if (drift > invariance) invariance = drift;
            const double early = fabs((double)at0[d * WD + i] -
                                      (double)at1[(j1 + d) * WD + i]);
            if (d >= context - 1u) {
                if (early > matched) matched = early;
            } else if (d == context - 2u) {
                if (early > boundary) boundary = early;
            }
        }
    }
    REQUIRE(invariance < 1e-4,
            "the same relative offset gave different answers at absolute %zu "
            "and %zu (drift %.3g): the attended set is not translation "
            "invariant, which is what a RoPE base that is not absolute does",
            j1, j2, invariance);
    REQUIRE(matched < 1e-4,
            "a window that started at position 0 never converged on the "
            "translated one (%.3g at or past offset %zu): the window does not "
            "fill at C - 1",
            matched, context - 1u);
    /* The boundary signal is one key out of `context` missing from a softmax,
     * so it necessarily shrinks as 1/C -- 2.4e-2 at C = 13 against 3.2e-5 at
     * C = 250 -- while the f32 floor does not. The assertion is therefore a
     * ratio, and the bit-exact pin on the same boundary lives in
     * `window/receptive-field`, which does not have this problem because it
     * compares presence against absence rather than two nearby windows. */
    REQUIRE(boundary > 1e-5 && boundary > 10.0 * matched,
            "offset %zu already matches the fully-windowed answer (%.3g vs "
            "%.3g): the window fills one position too early",
            context - 2u, boundary, matched);
    printf("  ok   %-44s invariant to %.1e, full at C-1 (%.1e) but not at C-2 "
           "(%.1e, %.0fx)\n",
           name, invariance, matched, boundary,
           boundary / (matched > 0.0 ? matched : 1e-30));
    rc = 0;
done:
    free(w); free(x); free(at1); free(at2); free(at0);
    return rc;
}

/* ----------------------------------------------- case: cache capacity */

/*
 * W4, made executable. "Where does the cache wrap?" has an answer here and it
 * is "nowhere": the cache is absolute-indexed, `max_seq_len` is a hard cap, and
 * a caller who sizes it to `context` gets a refusal at position C rather than
 * silently recycled -- and therefore wrong -- audio. This case is what stops
 * someone turning it into a ring without noticing the contract changed.
 */
static int case_capacity(size_t context) {
    g_case = "window/capacity";
    char error[256];
    const size_t layers = 2u;
    win_weights *w = malloc(sizeof(*w));
    float *x = malloc((context + 4u) * WD * sizeof(float));
    float out[WD];
    mynah_transformer_ar_state *state = NULL;
    int rc = -1;
    if (w == NULL || x == NULL) {
        fprintf(stderr, "FAIL [%s]: out of memory\n", g_case);
        ++g_failures;
        goto done;
    }
    build_weights(w, layers);
    fill_input(x, NULL, context + 4u, 29u);

    /* max_seq_len deliberately equal to the window: if the window implied a
     * ring, this would be a legal configuration for an unbounded sequence. */
    mynah_transformer_ar_config config;
    win_config(&config, layers, context, context);
    state = mynah_transformer_ar_state_new(&config, error, sizeof(error));
    REQUIRE(state != NULL, "state_new: %s", error);

    for (size_t t = 0; t < context; ++t) {
        REQUIRE(mynah_transformer_ar_step(state, &w->weights, x + t * WD,
                                          out) == 0,
                "step %zu inside the capacity failed", t);
    }
    REQUIRE(mynah_transformer_ar_state_offset(state) == context,
            "offset is %zu after %zu steps",
            mynah_transformer_ar_state_offset(state), context);
    /* Position `context` is exactly where a ring buffer would wrap. */
    REQUIRE(mynah_transformer_ar_step(state, &w->weights, x + context * WD,
                                      out) == -1,
            "the step at position %zu was accepted: the cache is behaving like "
            "a ring, which the contract (W4) says it is not",
            context);
    REQUIRE(mynah_transformer_ar_state_offset(state) == context,
            "the refused step moved the offset to %zu",
            mynah_transformer_ar_state_offset(state));
    /* A refusal must not damage the state: it is a bad request, not a fault. */
    REQUIRE(mynah_transformer_ar_prefill(state, &w->weights, x, 1u, NULL) == -1,
            "a prefill past the capacity was accepted");
    REQUIRE(mynah_transformer_ar_state_offset(state) == context,
            "the refused prefill moved the offset to %zu",
            mynah_transformer_ar_state_offset(state));
    mynah_transformer_ar_state_reset(state);
    REQUIRE(mynah_transformer_ar_state_offset(state) == 0u,
            "reset left the offset at %zu",
            mynah_transformer_ar_state_offset(state));
    REQUIRE(mynah_transformer_ar_step(state, &w->weights, x, out) == 0,
            "the state was unusable after a refusal and a reset");

    /* And an overflowing prefill from a clean state is refused whole. */
    mynah_transformer_ar_state_reset(state);
    REQUIRE(mynah_transformer_ar_prefill(state, &w->weights, x, context + 1u,
                                         NULL) == -1,
            "a prefill of %zu into a capacity of %zu was accepted", context + 1u,
            context);
    REQUIRE(mynah_transformer_ar_state_offset(state) == 0u,
            "the refused prefill partially ran: offset %zu",
            mynah_transformer_ar_state_offset(state));
    printf("  ok   %-44s no ring: hard cap at %zu, refusals are clean\n",
           "window/capacity", context);
    rc = 0;
done:
    mynah_transformer_ar_state_free(state);
    free(w); free(x);
    return rc;
}

/* --------------------------------------------- case: the ragged batch */

/* A hook that does exactly what the fallback does, so that installing it
 * changes the pointer plumbing and nothing else. If `in_rows` / `out_rows` were
 * assembled wrongly the numbers would move. */
static int hook_linear_rows(void *user, size_t layer,
                            mynah_transformer_ar_linear_kind kind,
                            const float *weight, const float *bias,
                            const float *const *in_rows, float *const *out_rows,
                            size_t batch, size_t k, size_t n) {
    (void)user; (void)layer; (void)kind;
    /* The same kernel the unhooked path calls, not an equivalent one written
     * out longhand: the hook contract is bit equality per row, and a plain
     * scalar loop is NOT bit-equal to the SIMD/Accelerate matvec. Writing it
     * out by hand here cost an afternoon of thinking the batch was at fault. */
    for (size_t b = 0; b < batch; ++b) {
        mynah_matvec_bias_f32(weight, in_rows[b], bias, out_rows[b], n, k);
    }
    return 0;
}

/*
 * W6. Several states at very different absolute positions -- some still inside
 * the unwindowed prefix, some well past the boundary -- stepped together, each
 * row compared bit for bit against the same state stepped alone.
 *
 * The prefixes are laid down with `_prefill`, so the tile path and the batched
 * step are composed rather than tested separately, and the batch is stepped
 * several times so that the rows starting at C-2 and C-1 CROSS the engage
 * boundary during the batched run. That is the case where a shared `lo` (rather
 * than a per-row one) would first show up.
 */
#define NROWS 6u

static int case_ragged_batch(size_t layers, size_t context, int hooked) {
    static char name[96];
    snprintf(name, sizeof(name), "window/ragged-batch L=%zu C=%zu hook=%d",
             layers, context, hooked);
    g_case = name;

    const size_t prefix[NROWS] = {0u, context - 2u, context - 1u, context,
                                  3u * context + 5u, 7u * context};
    const size_t steps = 6u;
    const size_t cap = 7u * context + steps + 2u;
    char error[256];

    win_weights *w = malloc(sizeof(*w));
    float *x = malloc(cap * WD * sizeof(float));
    float *xs_store = malloc(NROWS * steps * WD * sizeof(float));
    float got[NROWS][WD], want[NROWS][WD];
    mynah_transformer_ar_state *batched[NROWS] = {0};
    mynah_transformer_ar_state *solo[NROWS] = {0};
    mynah_transformer_ar_batch *scratch = NULL;
    int rc = -1;
    if (w == NULL || x == NULL || xs_store == NULL) {
        fprintf(stderr, "FAIL [%s]: out of memory\n", g_case);
        ++g_failures;
        goto done;
    }
    build_weights(w, layers);
    fill_input(x, NULL, cap, 31u);
    for (size_t i = 0; i < NROWS * steps * WD; ++i) {
        xs_store[i] = fake(i, 37u) * 3.0f;
    }
    w->weights.linear_rows = hooked ? hook_linear_rows : NULL;

    mynah_transformer_ar_config config;
    win_config(&config, layers, context, cap);
    scratch = mynah_transformer_ar_batch_new(&config, NROWS, error,
                                             sizeof(error));
    REQUIRE(scratch != NULL, "batch_new: %s", error);
    for (size_t b = 0; b < NROWS; ++b) {
        batched[b] = mynah_transformer_ar_state_new(&config, error,
                                                    sizeof(error));
        solo[b] = mynah_transformer_ar_state_new(&config, error, sizeof(error));
        REQUIRE(batched[b] != NULL && solo[b] != NULL, "state_new: %s", error);
        /* The prefix goes in through the tile path on both sides, so the
         * comparison is batched-step-after-tile against step-after-tile. */
        REQUIRE(mynah_transformer_ar_prefill(batched[b], &w->weights, x,
                                             prefix[b], NULL) == 0 &&
                    mynah_transformer_ar_prefill(solo[b], &w->weights, x,
                                                 prefix[b], NULL) == 0,
                "prefill of row %zu to %zu failed", b, prefix[b]);
    }

    for (size_t s = 0; s < steps; ++s) {
        const float *xs[NROWS];
        float *os[NROWS];
        for (size_t b = 0; b < NROWS; ++b) {
            xs[b] = xs_store + (s * NROWS + b) * WD;
            os[b] = got[b];
        }
        REQUIRE(mynah_transformer_ar_step_batch(batched, NROWS, &w->weights,
                                                scratch, xs, os) == 0,
                "the batched step %zu failed", s);
        for (size_t b = 0; b < NROWS; ++b) {
            REQUIRE(mynah_transformer_ar_step(solo[b], &w->weights, xs[b],
                                              want[b]) == 0,
                    "the solo step %zu of row %zu failed", s, b);
        }
        for (size_t b = 0; b < NROWS; ++b) {
            const size_t position = prefix[b] + s;
            REQUIRE(mynah_transformer_ar_state_offset(batched[b]) ==
                        position + 1u,
                    "row %zu is at offset %zu after step %zu, want %zu", b,
                    mynah_transformer_ar_state_offset(batched[b]), s,
                    position + 1u);
            for (size_t i = 0; i < WD; ++i) {
                REQUIRE(got[b][i] == want[b][i],
                        "step %zu row %zu (position %zu, window start %zu) dim "
                        "%zu: batched %.9g vs solo %.9g",
                        s, b, position,
                        contract_window_start(position, context), i,
                        (double)got[b][i], (double)want[b][i]);
            }
        }
        /* If every row produced the same numbers the comparison above could
         * not have noticed one row's state leaking into another. */
        int distinct = 0;
        for (size_t i = 0; i < WD && !distinct; ++i) {
            for (size_t b = 1; b < NROWS && !distinct; ++b) {
                if (got[b][i] != got[0][i]) distinct = 1;
            }
        }
        REQUIRE(distinct, "step %zu produced %u identical rows: the case is "
                          "blind to cross-talk", s, NROWS);
    }
    /* The rows that started just below the boundary must have crossed it. */
    REQUIRE(contract_window_start(prefix[1], context) == 0u &&
                contract_window_start(prefix[1] + steps - 1u, context) > 0u,
            "row 1 never crossed the engage boundary; the case proves less "
            "than it claims");
    printf("  ok   %-44s %u rows x %zu steps, bit-identical to solo\n", name,
           NROWS, steps);
    rc = 0;
done:
    for (size_t b = 0; b < NROWS; ++b) {
        mynah_transformer_ar_state_free(batched[b]);
        mynah_transformer_ar_state_free(solo[b]);
    }
    mynah_transformer_ar_batch_free(scratch);
    free(w); free(x); free(xs_store);
    return rc;
}

/* ------------------------------------------------------------------ main */

int main(void) {
    char error[256];
    printf("transformer_ar sliding window (PLAN.md E2-4)\n");

    /* The module's own model-free self test first: if RoPE or LayerNorm is
     * broken, every window result below is noise. */
    if (mynah_transformer_ar_self_test(error, sizeof(error)) != 0) {
        fprintf(stderr, "FAIL [transformer_ar self-test]: %s\n", error);
        ++g_failures;
    } else {
        printf("  ok   %-44s\n", "transformer_ar self-test");
    }

    /* 1. Against an independent f64 implementation of the contract. Three
     *    window lengths, the last of them the production 250, and every one of
     *    them crossed several times over. */
    case_reference(1u, 7u, 64u);     /* 9 window lengths        */
    case_reference(2u, 13u, 59u);    /* 4.5 window lengths      */
    case_reference(2u, 0u, 40u);     /* W1: unlimited, a control */
    case_reference(2u, 250u, 520u);  /* production C, two lengths */

    /* 2. The tile path against the step path, bit for bit, with the window
     *    engaged -- which case 9 of the module's self test does not do, because
     *    it runs at context 0. C is chosen not to divide the 16-row tile so the
     *    boundary lands mid-tile. */
    case_tile_vs_step(2u, 17u, 200u);
    case_tile_vs_step(1u, 250u, 300u);
    case_tile_vs_step(2u, 250u, 300u);

    /* 3. The receptive field, two-sided. j is placed so that the field starts
     *    before the boundary and ends after it. */
    case_receptive_field(1u, 9u, 60u, 0u);   /* from the very first position */
    case_receptive_field(1u, 9u, 60u, 11u);  /* well past the boundary       */
    case_receptive_field(1u, 250u, 520u, 260u);
    case_receptive_field(2u, 9u, 60u, 3u);

    /* 4. RoPE's base, via translation invariance once the window is full --
     *    and, in the same measurement, a second pin on the engage boundary
     *    that shares no arithmetic with case 3. */
    case_translation(13u, 120u);
    case_translation(31u, 240u);
    case_translation(250u, 1800u);

    /* 5. The cache does not wrap, and says so. */
    case_capacity(37u);
    case_capacity(250u);

    /* 6. Ragged batch across the window regime, with and without a row hook. */
    for (int hooked = 0; hooked < 2; ++hooked) {
        case_ragged_batch(2u, 13u, hooked);
        case_ragged_batch(2u, 250u, hooked);
    }

    if (g_failures != 0) {
        fprintf(stderr, "\n%d window case(s) failed\n", g_failures);
        return 1;
    }
    printf("all window cases passed\n");
    return 0;
}
