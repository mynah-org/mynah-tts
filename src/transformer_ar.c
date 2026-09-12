/*
 * Shared causal autoregressive transformer.  See transformer_ar.h for the
 * contract, the KV-cache layout and how the NaN sentinel is handled.
 */
#include "transformer_ar.h"

#include <math.h>
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

struct mynah_transformer_ar_state {
    mynah_transformer_ar_config config;
    size_t attn_dim; /* num_heads * head_dim */
    size_t half;     /* head_dim / 2         */
    size_t offset;   /* cached positions, i.e. the next absolute position */

    float *kv;       /* [num_layers][2][max_seq_len][attn_dim] */
    size_t kv_half;  /* max_seq_len * attn_dim                 */
    size_t kv_layer; /* 2 * kv_half                            */

    float *block; /* one owned scratch allocation */
    float *x;     /* [d_model]      */
    float *norm;  /* [d_model]      */
    float *upd;   /* [d_model]      */
    float *qkv;   /* [3 * attn_dim] */
    float *attn;  /* [attn_dim]     */
    float *ffn;   /* [ffn_dim]      */
    float *gelu;  /* [ffn_dim]      */
    float *scores;    /* [max_seq_len]        */
    float *rope_cos;  /* [max_seq_len][half]  */
    float *rope_sin;  /* [max_seq_len][half]  */
};

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

    /* KV: [layers][2][max_seq_len][attn_dim]. */
    size_t kv_half = 0, kv_layer = 0, kv_total = 0, kv_bytes = 0;
    if (tar_mul(resolved.max_seq_len, attn_dim, &kv_half) != 0 ||
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

    /* Scratch: 3 d_model + 3 attn_dim + 2 ffn + max_seq + 2 max_seq*half. */
    size_t total = 0, part = 0;
    int overflow = 0;
    overflow |= tar_mul(resolved.d_model, 3u, &part);
    overflow |= tar_add(total, part, &total);
    overflow |= tar_mul(attn_dim, 4u, &part); /* qkv (3) + attn (1) */
    overflow |= tar_add(total, part, &total);
    overflow |= tar_mul(resolved.ffn_dim, 2u, &part);
    overflow |= tar_add(total, part, &total);
    overflow |= tar_add(total, resolved.max_seq_len, &total);
    overflow |= tar_mul(resolved.max_seq_len, state->half, &part);
    overflow |= tar_mul(part, 2u, &part);
    overflow |= tar_add(total, part, &total);
    if (overflow != 0) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: scratch size overflow");
        free(state->kv);
        free(state);
        return NULL;
    }
    state->block = calloc(total ? total : 1u, sizeof(float));
    if (state->block == NULL) {
        tar_set_error(error, error_capacity, "transformer_ar: out of memory");
        free(state->kv);
        free(state);
        return NULL;
    }

    float *cursor = state->block;
    state->x = cursor;
    cursor += resolved.d_model;
    state->norm = cursor;
    cursor += resolved.d_model;
    state->upd = cursor;
    cursor += resolved.d_model;
    state->qkv = cursor;
    cursor += 3u * attn_dim;
    state->attn = cursor;
    cursor += attn_dim;
    state->ffn = cursor;
    cursor += resolved.ffn_dim;
    state->gelu = cursor;
    cursor += resolved.ffn_dim;
    state->scores = cursor;
    cursor += resolved.max_seq_len;
    state->rope_cos = cursor;
    cursor += resolved.max_seq_len * state->half;
    state->rope_sin = cursor;

    /* The RoPE table is position-only, so it is built once here and the hot
     * loop contains no transcendental at all. */
    for (size_t p = 0; p < resolved.max_seq_len; ++p) {
        mynah_transformer_ar_rope_angles_f32(
            state->rope_cos + p * state->half, state->rope_sin + p * state->half,
            state->half, p, resolved.max_period);
    }
    return state;
}

void mynah_transformer_ar_state_free(mynah_transformer_ar_state *state) {
    if (state == NULL) return;
    free(state->kv);
    free(state->block);
    free(state);
}

void mynah_transformer_ar_state_reset(mynah_transformer_ar_state *state) {
    if (state == NULL) return;
    state->offset = 0;
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
    if (positions > state->config.max_seq_len) {
        tar_set_error(error, error_capacity,
                      "transformer_ar: voice prefix %zu exceeds max_seq_len %zu",
                      positions, state->config.max_seq_len);
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

/* One position through the whole stack.  `out` may be NULL to discard. */
static int tar_step_one(mynah_transformer_ar_state *state,
                        const mynah_transformer_ar_weights *weights,
                        const float *x, float *out) {
    const mynah_transformer_ar_config *config = &state->config;
    const size_t d_model = config->d_model;
    const size_t heads = config->num_heads;
    const size_t head_dim = config->head_dim;
    const size_t attn_dim = state->attn_dim;
    const size_t position = state->offset;
    const size_t lo = tar_window_start(position, config->context);
    const size_t span = position - lo + 1u;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const float *rope_cos = state->rope_cos + position * state->half;
    const float *rope_sin = state->rope_sin + position * state->half;

    memcpy(state->x, x, d_model * sizeof(float));

    for (size_t l = 0; l < config->num_layers; ++l) {
        const mynah_transformer_ar_layer *layer = &weights->layers[l];
        float *k_cache = state->kv + l * state->kv_layer;
        float *v_cache = k_cache + state->kv_half;

        /* --- attention block: x + layer_scale_1(attn(norm1(x))) --- */
        mynah_layernorm_f32(state->x, layer->norm1_weight, layer->norm1_bias,
                            state->norm, 1u, d_model, config->layernorm_eps);
        mynah_matvec_bias_f32(layer->in_proj_weight, state->norm,
                              layer->in_proj_bias, state->qkv, 3u * attn_dim,
                              d_model);
        float *q = state->qkv;
        float *k = state->qkv + attn_dim;
        float *v = state->qkv + 2u * attn_dim;
        mynah_transformer_ar_rope_apply_f32(q, heads, head_dim, rope_cos,
                                            rope_sin);
        mynah_transformer_ar_rope_apply_f32(k, heads, head_dim, rope_cos,
                                            rope_sin);
        memcpy(k_cache + position * attn_dim, k, attn_dim * sizeof(float));
        memcpy(v_cache + position * attn_dim, v, attn_dim * sizeof(float));

        for (size_t h = 0; h < heads; ++h) {
            const float *qh = q + h * head_dim;
            for (size_t j = 0; j < span; ++j) {
                const float *kj =
                    k_cache + (lo + j) * attn_dim + h * head_dim;
                state->scores[j] = mynah_dot_f32(qh, kj, head_dim) * scale;
            }
            /* Rejects non-finite scores, which is the last line of defence
             * against a NaN that slipped into the cache. */
            if (mynah_softmax_f32(state->scores, state->scores, span) != 0) {
                return -1;
            }
            float *oh = state->attn + h * head_dim;
            memset(oh, 0, head_dim * sizeof(float));
            for (size_t j = 0; j < span; ++j) {
                const float *vj =
                    v_cache + (lo + j) * attn_dim + h * head_dim;
                mynah_axpy_f32(oh, vj, state->scores[j], head_dim);
            }
        }
        mynah_matvec_bias_f32(layer->out_proj_weight, state->attn,
                              layer->out_proj_bias, state->upd, d_model,
                              attn_dim);
        if (layer->layer_scale_1 != NULL) {
            for (size_t i = 0; i < d_model; ++i) {
                state->x[i] += layer->layer_scale_1[i] * state->upd[i];
            }
        } else {
            mynah_residual_add_f32(state->x, state->upd, d_model);
        }

        /* --- feed forward: x + layer_scale_2(linear2(gelu(linear1(norm2)))) */
        mynah_layernorm_f32(state->x, layer->norm2_weight, layer->norm2_bias,
                            state->norm, 1u, d_model, config->layernorm_eps);
        mynah_matvec_bias_f32(layer->linear1_weight, state->norm,
                              layer->linear1_bias, state->ffn, config->ffn_dim,
                              d_model);
        mynah_gelu_tanh_array(state->ffn, config->ffn_dim, state->gelu);
        mynah_matvec_bias_f32(layer->linear2_weight, state->ffn,
                              layer->linear2_bias, state->upd, d_model,
                              config->ffn_dim);
        if (layer->layer_scale_2 != NULL) {
            for (size_t i = 0; i < d_model; ++i) {
                state->x[i] += layer->layer_scale_2[i] * state->upd[i];
            }
        } else {
            mynah_residual_add_f32(state->x, state->upd, d_model);
        }
    }

    state->offset = position + 1u;

    if (out != NULL) {
        if (weights->out_norm_weight != NULL) {
            mynah_layernorm_f32(state->x, weights->out_norm_weight,
                                weights->out_norm_bias, out, 1u, d_model,
                                config->layernorm_eps);
        } else {
            memcpy(out, state->x, d_model * sizeof(float));
        }
    }
    return 0;
}

int mynah_transformer_ar_prefill(mynah_transformer_ar_state *state,
                                 const mynah_transformer_ar_weights *weights,
                                 const float *x, size_t n_tokens, float *out) {
    if (state == NULL || weights == NULL || weights->layers == NULL) return -1;
    if (n_tokens == 0) return 0;
    if (x == NULL) return -1;
    const size_t d_model = state->config.d_model;
    size_t end = 0;
    if (tar_add(state->offset, n_tokens, &end) != 0 ||
        end > state->config.max_seq_len) {
        return -1;
    }
    /* Fail at the boundary rather than letting a BOS sentinel reach a matmul:
     * the caller substitutes bos_emb before input_linear, so anything
     * non-finite arriving here is a bug, not a sentinel. */
    if (!tar_finite(x, n_tokens * d_model)) return -1;
    for (size_t t = 0; t < n_tokens; ++t) {
        float *row = (out == NULL) ? NULL : out + t * d_model;
        if (tar_step_one(state, weights, x + t * d_model, row) != 0) return -1;
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
    if (state->offset >= state->config.max_seq_len) return -1;
    if (!tar_finite(x, state->config.d_model)) return -1;
    return tar_step_one(state, weights, x, out);
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
        if (mynah_transformer_ar_state_offset(state) != TAR_T) {
            mynah_transformer_ar_state_free(state);
            free(store);
            TAR_FAIL("prefill left offset %zu, want %u",
                     mynah_transformer_ar_state_offset(state), TAR_T);
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

    free(store);
    return 0;
}

#undef TAR_FAIL
