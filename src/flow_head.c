/*
 * SimpleMLPAdaLN — the PocketTTS flow head.  See flow_head.h for the contract
 * and for why the normalisation kernels live here instead of src/kernels.c.
 */
#include "flow_head.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"

/* ------------------------------------------------------------------ utils */

static void flow_set_error(char *error, size_t capacity, const char *format,
                           ...) __attribute__((format(printf, 3, 4)));

#include <stdarg.h>

static void flow_set_error(char *error, size_t capacity, const char *format,
                           ...) {
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    (void)vsnprintf(error, capacity, format, args);
    va_end(args);
}

/* Checked size_t multiply / add: every allocation goes through these. */
static int flow_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > (size_t)-1 / a) return -1;
    *out = a * b;
    return 0;
}

static int flow_add(size_t a, size_t b, size_t *out) {
    if (b > (size_t)-1 - a) return -1;
    *out = a + b;
    return 0;
}

/* ---------------------------------------------------------------- kernels */

void mynah_flow_layernorm_f32(const float *input, const float *weight,
                              const float *bias, float *output, size_t n,
                              float epsilon) {
    if (input == NULL || output == NULL || n == 0) return;
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += input[i];
    mean /= (float)n;
    float variance = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = input[i] - mean;
        variance += d * d;
    }
    /* Biased variance: torch var(unbiased=False) inside the custom LayerNorm. */
    variance /= (float)n;
    const float scale = 1.0f / sqrtf(variance + epsilon);
    for (size_t i = 0; i < n; ++i) {
        float value = (input[i] - mean) * scale;
        if (weight != NULL) value *= weight[i];
        if (bias != NULL) value += bias[i];
        output[i] = value;
    }
}

void mynah_flow_var_rmsnorm_f32(const float *input, const float *alpha,
                                float *output, size_t n, float epsilon) {
    if (input == NULL || output == NULL || n < 2u) return;
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += input[i];
    mean /= (float)n;
    float variance = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = input[i] - mean;
        variance += d * d;
    }
    /* Unbiased: torch.var() defaults to correction=1, so divide by n-1. */
    variance /= (float)(n - 1u);
    const float scale = 1.0f / sqrtf(epsilon + variance);
    for (size_t i = 0; i < n; ++i) {
        const float gain = (alpha != NULL) ? alpha[i] : 1.0f;
        output[i] = input[i] * (gain * scale);
    }
}

void mynah_flow_silu_f32(const float *input, float *output, size_t n) {
    if (input == NULL || output == NULL) return;
    for (size_t i = 0; i < n; ++i) {
        const float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

void mynah_flow_modulate_f32(const float *input, const float *shift,
                             const float *scale, float *output, size_t n) {
    if (input == NULL || output == NULL) return;
    for (size_t i = 0; i < n; ++i) {
        const float s = (scale != NULL) ? scale[i] : 0.0f;
        const float b = (shift != NULL) ? shift[i] : 0.0f;
        output[i] = input[i] * (1.0f + s) + b;
    }
}

void mynah_flow_timestep_freqs_f32(float *freqs, size_t half,
                                   float max_period) {
    if (freqs == NULL || half == 0) return;
    const double log_period = log((double)max_period);
    for (size_t i = 0; i < half; ++i) {
        freqs[i] = (float)exp(-log_period * (double)i / (double)half);
    }
}

void mynah_flow_round_bf16_f32(float *values, size_t n) {
    if (values == NULL) return;
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        /* Round half to even on the low 16 bits, exactly like a BF16 cast. */
        const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
        bits = (bits + rounding) & 0xffff0000u;
        memcpy(&values[i], &bits, sizeof(bits));
    }
}

/* ------------------------------------------------------------------ state */

struct mynah_flow_head {
    mynah_flow_head_config config;
    size_t half; /* freq_embed_dim / 2 */

    float *block;   /* single owned allocation */
    float *freqs;   /* [half]                  */
    float *emb;     /* [freq_embed_dim]        */
    float *y;       /* [hidden_dim]            */
    float *y_time;  /* [hidden_dim] memoised   */
    float *silu;    /* [hidden_dim]            */
    float *x;       /* [hidden_dim]            */
    float *norm;    /* [hidden_dim]            */
    float *hidden;  /* [hidden_dim]            */
    float *scratch; /* [hidden_dim]            */
    float *mod;     /* [3 * hidden_dim]        */
    float *cached_times; /* [num_time_conds]   */
    int time_cache_valid;
};

void mynah_flow_head_config_defaults(mynah_flow_head_config *config) {
    if (config == NULL) return;
    config->latent_dim = 0;
    config->cond_dim = 0;
    config->hidden_dim = 0;
    config->depth = 0;
    config->num_time_conds = 2u;
    config->freq_embed_dim = 256u;
    config->max_period = 10000.0f;
    config->layernorm_eps = 1e-6f;
    config->rmsnorm_eps = 1e-5f;
    config->freqs_bf16_rounded = 1;
}

mynah_flow_head *mynah_flow_head_create(const mynah_flow_head_config *config,
                                        char *error, size_t error_capacity) {
    if (config == NULL) {
        flow_set_error(error, error_capacity, "flow head: null config");
        return NULL;
    }
    if (config->latent_dim == 0 || config->cond_dim == 0 ||
        config->hidden_dim < 2u || config->depth == 0) {
        flow_set_error(error, error_capacity,
                       "flow head: bad dims latent=%zu cond=%zu hidden=%zu "
                       "depth=%zu",
                       config->latent_dim, config->cond_dim,
                       config->hidden_dim, config->depth);
        return NULL;
    }
    if (config->num_time_conds > 0 &&
        (config->freq_embed_dim < 2u || (config->freq_embed_dim % 2u) != 0)) {
        flow_set_error(error, error_capacity,
                       "flow head: freq_embed_dim %zu must be even and >= 2",
                       config->freq_embed_dim);
        return NULL;
    }
    if (config->num_time_conds > 0 && !(config->max_period > 1.0f)) {
        flow_set_error(error, error_capacity,
                       "flow head: max_period must be > 1");
        return NULL;
    }

    mynah_flow_head *head = calloc(1u, sizeof(*head));
    if (head == NULL) {
        flow_set_error(error, error_capacity, "flow head: out of memory");
        return NULL;
    }
    head->config = *config;
    head->half = config->freq_embed_dim / 2u;

    const size_t hidden = config->hidden_dim;
    size_t mod_floats = 0;
    size_t total = 0;
    if (flow_mul(hidden, 3u, &mod_floats) != 0) {
        flow_set_error(error, error_capacity, "flow head: hidden_dim overflow");
        free(head);
        return NULL;
    }
    /* freqs + emb + 7 hidden-sized vectors + mod + cached times */
    size_t seven = 0;
    if (flow_mul(hidden, 7u, &seven) != 0 ||
        flow_add(seven, mod_floats, &total) != 0 ||
        flow_add(total, head->half, &total) != 0 ||
        flow_add(total, config->freq_embed_dim, &total) != 0 ||
        flow_add(total, config->num_time_conds, &total) != 0) {
        flow_set_error(error, error_capacity, "flow head: scratch overflow");
        free(head);
        return NULL;
    }
    size_t bytes = 0;
    if (flow_mul(total, sizeof(float), &bytes) != 0) {
        flow_set_error(error, error_capacity, "flow head: scratch overflow");
        free(head);
        return NULL;
    }
    head->block = calloc(total ? total : 1u, sizeof(float));
    if (head->block == NULL) {
        flow_set_error(error, error_capacity, "flow head: out of memory");
        free(head);
        return NULL;
    }

    float *cursor = head->block;
    head->freqs = cursor;
    cursor += head->half;
    head->emb = cursor;
    cursor += config->freq_embed_dim;
    head->y = cursor;
    cursor += hidden;
    head->y_time = cursor;
    cursor += hidden;
    head->silu = cursor;
    cursor += hidden;
    head->x = cursor;
    cursor += hidden;
    head->norm = cursor;
    cursor += hidden;
    head->hidden = cursor;
    cursor += hidden;
    head->scratch = cursor;
    cursor += hidden;
    head->mod = cursor;
    cursor += mod_floats;
    head->cached_times = cursor;

    if (head->half > 0) {
        mynah_flow_timestep_freqs_f32(head->freqs, head->half,
                                      config->max_period);
        if (config->freqs_bf16_rounded) {
            mynah_flow_round_bf16_f32(head->freqs, head->half);
        }
    }
    head->time_cache_valid = 0;
    return head;
}

void mynah_flow_head_destroy(mynah_flow_head *head) {
    if (head == NULL) return;
    free(head->block);
    free(head);
}

const mynah_flow_head_config *mynah_flow_head_get_config(
    const mynah_flow_head *head) {
    return (head == NULL) ? NULL : &head->config;
}

void mynah_flow_head_reset(mynah_flow_head *head) {
    if (head == NULL) return;
    head->time_cache_valid = 0;
}

static int flow_linear_ok(const mynah_flow_linear *linear, int needs_bias) {
    if (linear->weight == NULL) return 0;
    if (needs_bias && linear->bias == NULL) return 0;
    return 1;
}

int mynah_flow_head_check_weights(const mynah_flow_head *head,
                                  const mynah_flow_head_weights *weights,
                                  char *error, size_t error_capacity) {
    if (head == NULL || weights == NULL) {
        flow_set_error(error, error_capacity, "flow head: null argument");
        return -1;
    }
    if (!flow_linear_ok(&weights->cond_embed, 0)) {
        flow_set_error(error, error_capacity, "flow head: cond_embed missing");
        return -1;
    }
    if (!flow_linear_ok(&weights->input_proj, 0)) {
        flow_set_error(error, error_capacity, "flow head: input_proj missing");
        return -1;
    }
    if (!flow_linear_ok(&weights->final_adaln, 0) ||
        !flow_linear_ok(&weights->final_linear, 0)) {
        flow_set_error(error, error_capacity,
                       "flow head: final_layer weights missing");
        return -1;
    }
    if (head->config.num_time_conds > 0) {
        if (weights->time_embed == NULL) {
            flow_set_error(error, error_capacity,
                           "flow head: time_embed array missing");
            return -1;
        }
        for (size_t i = 0; i < head->config.num_time_conds; ++i) {
            const mynah_flow_time_embed_weights *te = &weights->time_embed[i];
            if (!flow_linear_ok(&te->mlp_in, 0) ||
                !flow_linear_ok(&te->mlp_out, 0) || te->alpha == NULL) {
                flow_set_error(error, error_capacity,
                               "flow head: time_embed[%zu] incomplete", i);
                return -1;
            }
        }
    }
    if (weights->res_blocks == NULL) {
        flow_set_error(error, error_capacity,
                       "flow head: res_blocks array missing");
        return -1;
    }
    for (size_t i = 0; i < head->config.depth; ++i) {
        const mynah_flow_res_block_weights *block = &weights->res_blocks[i];
        if (block->in_ln_weight == NULL ||
            !flow_linear_ok(&block->adaln, 0) ||
            !flow_linear_ok(&block->mlp_in, 0) ||
            !flow_linear_ok(&block->mlp_out, 0)) {
            flow_set_error(error, error_capacity,
                           "flow head: res_blocks[%zu] incomplete", i);
            return -1;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- forward */

/* TimestepEmbedder: args = t * freqs; emb = cat([cos(args), sin(args)]);
 * mlp = Linear, SiLU, Linear, variance-RMSNorm. */
static void flow_timestep_embed(mynah_flow_head *head,
                                const mynah_flow_time_embed_weights *weights,
                                float t, float *out) {
    const size_t half = head->half;
    const size_t hidden = head->config.hidden_dim;
    for (size_t i = 0; i < half; ++i) {
        const float arg = t * head->freqs[i];
        head->emb[i] = cosf(arg);        /* cos first, then sin */
        head->emb[half + i] = sinf(arg);
    }
    mynah_matvec_bias_f32(weights->mlp_in.weight, head->emb,
                          weights->mlp_in.bias, head->scratch, hidden,
                          head->config.freq_embed_dim);
    mynah_flow_silu_f32(head->scratch, head->scratch, hidden);
    mynah_matvec_bias_f32(weights->mlp_out.weight, head->scratch,
                          weights->mlp_out.bias, out, hidden, hidden);
    mynah_flow_var_rmsnorm_f32(out, weights->alpha, out, hidden,
                               head->config.rmsnorm_eps);
}

static int flow_time_cache_hit(const mynah_flow_head *head,
                               const float *times) {
    if (!head->time_cache_valid) return 0;
    for (size_t i = 0; i < head->config.num_time_conds; ++i) {
        if (head->cached_times[i] != times[i]) return 0;
    }
    return 1;
}

int mynah_flow_head_forward(mynah_flow_head *head,
                            const mynah_flow_head_weights *weights,
                            const float *cond, const float *times,
                            const float *noise, float *out) {
    if (head == NULL || weights == NULL || cond == NULL || noise == NULL ||
        out == NULL) {
        return -1;
    }
    const mynah_flow_head_config *config = &head->config;
    const size_t hidden = config->hidden_dim;
    const size_t latent = config->latent_dim;
    if (config->num_time_conds > 0 && times == NULL) return -1;

    /* y = cond_embed(c) + sum(time_embed[i](t_i)) / num_time_conds */
    mynah_matvec_bias_f32(weights->cond_embed.weight, cond,
                          weights->cond_embed.bias, head->y, hidden,
                          config->cond_dim);
    if (config->num_time_conds > 0) {
        if (!flow_time_cache_hit(head, times)) {
            memset(head->y_time, 0, hidden * sizeof(float));
            for (size_t i = 0; i < config->num_time_conds; ++i) {
                flow_timestep_embed(head, &weights->time_embed[i], times[i],
                                    head->hidden);
                for (size_t j = 0; j < hidden; ++j) {
                    head->y_time[j] += head->hidden[j];
                }
            }
            const float inv = 1.0f / (float)config->num_time_conds;
            for (size_t j = 0; j < hidden; ++j) head->y_time[j] *= inv;
            for (size_t i = 0; i < config->num_time_conds; ++i) {
                head->cached_times[i] = times[i];
            }
            head->time_cache_valid = 1;
        }
        for (size_t j = 0; j < hidden; ++j) head->y[j] += head->y_time[j];
    }

    /* x = input_proj(noise) */
    mynah_matvec_bias_f32(weights->input_proj.weight, noise,
                          weights->input_proj.bias, head->x, hidden, latent);

    /* SiLU(y) is shared by every adaLN_modulation. */
    mynah_flow_silu_f32(head->y, head->silu, hidden);

    for (size_t b = 0; b < config->depth; ++b) {
        const mynah_flow_res_block_weights *block = &weights->res_blocks[b];
        mynah_matvec_bias_f32(block->adaln.weight, head->silu,
                              block->adaln.bias, head->mod, 3u * hidden,
                              hidden);
        const float *shift = head->mod;
        const float *scale = head->mod + hidden;
        const float *gate = head->mod + 2u * hidden;

        mynah_flow_layernorm_f32(head->x, block->in_ln_weight,
                                 block->in_ln_bias, head->norm, hidden,
                                 config->layernorm_eps);
        mynah_flow_modulate_f32(head->norm, shift, scale, head->norm, hidden);

        mynah_matvec_bias_f32(block->mlp_in.weight, head->norm,
                              block->mlp_in.bias, head->hidden, hidden, hidden);
        mynah_flow_silu_f32(head->hidden, head->hidden, hidden);
        mynah_matvec_bias_f32(block->mlp_out.weight, head->hidden,
                              block->mlp_out.bias, head->scratch, hidden,
                              hidden);

        for (size_t i = 0; i < hidden; ++i) {
            head->x[i] += gate[i] * head->scratch[i];
        }
    }

    /* final: norm_final has no affine parameters. */
    mynah_matvec_bias_f32(weights->final_adaln.weight, head->silu,
                          weights->final_adaln.bias, head->mod, 2u * hidden,
                          hidden);
    mynah_flow_layernorm_f32(head->x, NULL, NULL, head->norm, hidden,
                             config->layernorm_eps);
    mynah_flow_modulate_f32(head->norm, head->mod, head->mod + hidden,
                            head->norm, hidden);
    mynah_matvec_bias_f32(weights->final_linear.weight, head->norm,
                          weights->final_linear.bias, out, latent, hidden);
    return 0;
}

/* -------------------------------------------------------------- self test */

#define FLOW_CHECK(condition, ...)                             \
    do {                                                       \
        if (!(condition)) {                                    \
            flow_set_error(error, error_capacity, __VA_ARGS__); \
            return -1;                                         \
        }                                                      \
    } while (0)

static int flow_close(float a, double b, double tolerance) {
    const double d = (double)a - b;
    return (d < 0 ? -d : d) <= tolerance;
}

/* Naive double-precision reference for the whole forward.  Deliberately
 * written straight off modules/mlp.py with no shared code. */
struct flow_ref_ctx {
    size_t latent;
    size_t cond;
    size_t hidden;
    size_t depth;
    size_t ntc;
    size_t femb;
    double max_period;
    double ln_eps;
    double rms_eps;
};

static void ref_linear(const float *w, const float *bias, const double *in,
                       double *out, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        double acc = (bias != NULL) ? (double)bias[r] : 0.0;
        for (size_t c = 0; c < cols; ++c) acc += (double)w[r * cols + c] * in[c];
        out[r] = acc;
    }
}

static void ref_silu(double *v, size_t n) {
    for (size_t i = 0; i < n; ++i) v[i] = v[i] / (1.0 + exp(-v[i]));
}

static void ref_layernorm(const double *in, const float *w, const float *b,
                          double *out, size_t n, double eps) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += in[i];
    mean /= (double)n;
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) var += (in[i] - mean) * (in[i] - mean);
    var /= (double)n;
    const double inv = 1.0 / sqrt(var + eps);
    for (size_t i = 0; i < n; ++i) {
        double v = (in[i] - mean) * inv;
        if (w != NULL) v *= (double)w[i];
        if (b != NULL) v += (double)b[i];
        out[i] = v;
    }
}

static void ref_var_rmsnorm(const double *in, const float *alpha, double *out,
                            size_t n, double eps) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += in[i];
    mean /= (double)n;
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) var += (in[i] - mean) * (in[i] - mean);
    var /= (double)(n - 1u);
    const double inv = 1.0 / sqrt(eps + var);
    for (size_t i = 0; i < n; ++i) out[i] = in[i] * ((double)alpha[i] * inv);
}

static float flow_fake(size_t index, size_t salt) {
    /* Deterministic, bounded, sign-alternating pseudo weights. */
    const double v = sin((double)(index * 7u + salt * 13u + 1u) * 0.37) * 0.5 +
                     cos((double)(index * 3u + salt * 5u + 2u) * 0.11) * 0.25;
    return (float)v;
}

int mynah_flow_head_self_test(char *error, size_t error_capacity) {
    if (error != NULL && error_capacity > 0) error[0] = '\0';

    /* --- SiLU ------------------------------------------------------- */
    {
        const float in[5] = {0.0f, 1.0f, -1.0f, 2.0f, -3.5f};
        const double want[5] = {0.0, 0.7310585786300049, -0.2689414213699951,
                                1.7615941559557646, -0.10259280762974712};
        float got[5];
        mynah_flow_silu_f32(in, got, 5u);
        for (size_t i = 0; i < 5u; ++i) {
            FLOW_CHECK(flow_close(got[i], want[i], 1e-6),
                       "silu[%zu] = %.9g want %.9g", i, (double)got[i],
                       want[i]);
        }
        float inplace[5] = {0.0f, 1.0f, -1.0f, 2.0f, -3.5f};
        mynah_flow_silu_f32(inplace, inplace, 5u);
        for (size_t i = 0; i < 5u; ++i) {
            FLOW_CHECK(inplace[i] == got[i], "silu not in-place safe at %zu", i);
        }
    }

    /* --- LayerNorm, biased variance, with and without affine --------- */
    {
        const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float w[4] = {2.0f, 3.0f, 4.0f, 5.0f};
        const float b[4] = {0.5f, -0.5f, 1.0f, -1.0f};
        const double plain[4] = {-1.341640249843881, -0.44721341661462705,
                                 0.44721341661462705, 1.341640249843881};
        const double affine[4] = {-2.183280499687762, -1.841640249843881,
                                  2.7888536664585084, 5.708201249219405};
        const double weight_only[4] = {-2.683280499687762, -1.341640249843881,
                                       1.7888536664585082, 6.708201249219405};
        float got[4];
        mynah_flow_layernorm_f32(x, NULL, NULL, got, 4u, 1e-6f);
        for (size_t i = 0; i < 4u; ++i) {
            FLOW_CHECK(flow_close(got[i], plain[i], 1e-5),
                       "layernorm(no affine)[%zu] = %.9g want %.9g", i,
                       (double)got[i], plain[i]);
        }
        mynah_flow_layernorm_f32(x, w, b, got, 4u, 1e-6f);
        for (size_t i = 0; i < 4u; ++i) {
            FLOW_CHECK(flow_close(got[i], affine[i], 1e-5),
                       "layernorm(affine)[%zu] = %.9g want %.9g", i,
                       (double)got[i], affine[i]);
        }
        mynah_flow_layernorm_f32(x, w, NULL, got, 4u, 1e-6f);
        for (size_t i = 0; i < 4u; ++i) {
            FLOW_CHECK(flow_close(got[i], weight_only[i], 1e-5),
                       "layernorm(weight only)[%zu] = %.9g want %.9g", i,
                       (double)got[i], weight_only[i]);
        }
    }

    /* --- variance RMSNorm: unbiased, mean subtracted ----------------- */
    {
        const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float alpha[4] = {0.5f, 1.0f, 1.5f, 2.0f};
        const double want[4] = {0.3872971727309663, 1.5491886909238652,
                                3.4856745545786967, 6.196754763695461};
        /* What the ordinary mean-square rmsnorm in src/kernels.c would give.
         * The two must not be confused: assert they are far apart. */
        const double meansq[4] = {0.1825740641190532, 0.7302962564762128,
                                  1.6431665770714787, 2.921185025904851};
        float got[4];
        mynah_flow_var_rmsnorm_f32(x, alpha, got, 4u, 1e-5f);
        for (size_t i = 0; i < 4u; ++i) {
            FLOW_CHECK(flow_close(got[i], want[i], 1e-5),
                       "var_rmsnorm[%zu] = %.9g want %.9g", i, (double)got[i],
                       want[i]);
        }
        FLOW_CHECK(!flow_close(got[3], meansq[3], 1e-2),
                   "var_rmsnorm collapsed onto mean-square rmsnorm");
        float alt[4];
        mynah_rmsnorm_f32(x, alpha, alt, 4u, 1e-5f);
        FLOW_CHECK(!flow_close(alt[3], want[3], 1e-2),
                   "mynah_rmsnorm_f32 unexpectedly matches the flow RMSNorm; "
                   "the self test would no longer catch a substitution");
    }

    /* --- adaLN modulation ------------------------------------------- */
    {
        const float x[3] = {1.0f, -2.0f, 0.5f};
        const float shift[3] = {0.25f, 0.5f, -1.0f};
        const float scale[3] = {1.0f, -0.5f, 3.0f};
        const double want[3] = {1.0 * 2.0 + 0.25, -2.0 * 0.5 + 0.5,
                                0.5 * 4.0 - 1.0};
        float got[3];
        mynah_flow_modulate_f32(x, shift, scale, got, 3u);
        for (size_t i = 0; i < 3u; ++i) {
            FLOW_CHECK(flow_close(got[i], want[i], 1e-6),
                       "modulate[%zu] = %.9g want %.9g", i, (double)got[i],
                       want[i]);
        }
    }

    /* --- timestep frequencies --------------------------------------- */
    {
        float freqs[4];
        const double want[4] = {1.0, 0.09999999999999998, 0.009999999999999995,
                                0.0009999999999999994};
        mynah_flow_timestep_freqs_f32(freqs, 4u, 10000.0f);
        for (size_t i = 0; i < 4u; ++i) {
            FLOW_CHECK(flow_close(freqs[i], want[i], 1e-7),
                       "freqs[%zu] = %.9g want %.9g", i, (double)freqs[i],
                       want[i]);
        }
        /* BF16 round trip: 8 mantissa bits, round half to even. */
        float rounded[6] = {1.0f, 0.1f, 1.5f, -1.00390625f, 0.0f, -0.0f};
        mynah_flow_round_bf16_f32(rounded, 6u);
        FLOW_CHECK(rounded[0] == 1.0f, "bf16 round changed 1.0");
        FLOW_CHECK(rounded[2] == 1.5f, "bf16 round changed 1.5");
        FLOW_CHECK(rounded[4] == 0.0f, "bf16 round changed 0.0");
        FLOW_CHECK(flow_close(rounded[1], 0.100097656, 1e-9),
                   "bf16 round of 0.1 = %.9g want 0.100097656",
                   (double)rounded[1]);
        /* -1.00390625 is exactly halfway between two BF16 values, so
         * round-half-to-even must land on -1.0, not -1.0078125. */
        FLOW_CHECK(rounded[3] == -1.0f,
                   "bf16 round of -1.00390625 = %.9g want -1 (ties to even)",
                   (double)rounded[3]);
        for (size_t i = 0; i < 6u; ++i) {
            uint32_t bits;
            memcpy(&bits, &rounded[i], sizeof(bits));
            FLOW_CHECK((bits & 0xffffu) == 0, "bf16 round left low bits at %zu",
                       i);
        }
    }

    /* --- full forward against the naive double reference ------------- */
    {
        mynah_flow_head_config config;
        mynah_flow_head_config_defaults(&config);
        config.latent_dim = 4u;
        config.cond_dim = 6u;
        config.hidden_dim = 8u;
        config.depth = 2u;
        config.num_time_conds = 2u;
        config.freq_embed_dim = 6u;

        const size_t latent = config.latent_dim;
        const size_t cond_dim = config.cond_dim;
        const size_t hidden = config.hidden_dim;
        const size_t depth = config.depth;
        const size_t femb = config.freq_embed_dim;

        /* One flat pool so the test allocates once and frees once. */
        size_t pool_floats = 0;
        pool_floats += hidden * cond_dim + hidden;             /* cond_embed */
        pool_floats += hidden * latent + hidden;               /* input_proj */
        pool_floats += 2u * (hidden * femb + hidden + hidden * hidden +
                             hidden + hidden);                 /* time embed */
        pool_floats += depth * (hidden + hidden + 3u * hidden * hidden +
                                3u * hidden + 2u * hidden * hidden +
                                2u * hidden);                  /* res blocks */
        pool_floats += 2u * hidden * hidden + 2u * hidden;     /* final adaln */
        pool_floats += latent * hidden + latent;               /* final lin   */
        pool_floats += cond_dim + latent;                      /* inputs      */

        float *pool = calloc(pool_floats, sizeof(float));
        if (pool == NULL) {
            flow_set_error(error, error_capacity, "self test: out of memory");
            return -1;
        }
        for (size_t i = 0; i < pool_floats; ++i) pool[i] = flow_fake(i, 3u);

        float *cursor = pool;
        mynah_flow_head_weights weights;
        memset(&weights, 0, sizeof(weights));
        weights.cond_embed.weight = cursor;
        cursor += hidden * cond_dim;
        weights.cond_embed.bias = cursor;
        cursor += hidden;
        weights.input_proj.weight = cursor;
        cursor += hidden * latent;
        weights.input_proj.bias = cursor;
        cursor += hidden;

        mynah_flow_time_embed_weights time_embed[2];
        for (size_t i = 0; i < 2u; ++i) {
            time_embed[i].mlp_in.weight = cursor;
            cursor += hidden * femb;
            time_embed[i].mlp_in.bias = cursor;
            cursor += hidden;
            time_embed[i].mlp_out.weight = cursor;
            cursor += hidden * hidden;
            time_embed[i].mlp_out.bias = cursor;
            cursor += hidden;
            time_embed[i].alpha = cursor;
            cursor += hidden;
        }
        weights.time_embed = time_embed;

        mynah_flow_res_block_weights blocks[2];
        for (size_t i = 0; i < depth; ++i) {
            blocks[i].in_ln_weight = cursor;
            cursor += hidden;
            blocks[i].in_ln_bias = cursor;
            cursor += hidden;
            blocks[i].adaln.weight = cursor;
            cursor += 3u * hidden * hidden;
            blocks[i].adaln.bias = cursor;
            cursor += 3u * hidden;
            blocks[i].mlp_in.weight = cursor;
            cursor += hidden * hidden;
            blocks[i].mlp_in.bias = cursor;
            cursor += hidden;
            blocks[i].mlp_out.weight = cursor;
            cursor += hidden * hidden;
            blocks[i].mlp_out.bias = cursor;
            cursor += hidden;
        }
        weights.res_blocks = blocks;
        weights.final_adaln.weight = cursor;
        cursor += 2u * hidden * hidden;
        weights.final_adaln.bias = cursor;
        cursor += 2u * hidden;
        weights.final_linear.weight = cursor;
        cursor += latent * hidden;
        weights.final_linear.bias = cursor;
        cursor += latent;
        const float *cond = cursor;
        cursor += cond_dim;
        const float *noise = cursor;

        mynah_flow_head *head = mynah_flow_head_create(&config, error,
                                                       error_capacity);
        if (head == NULL) {
            free(pool);
            return -1;
        }
        if (mynah_flow_head_check_weights(head, &weights, error,
                                          error_capacity) != 0) {
            mynah_flow_head_destroy(head);
            free(pool);
            return -1;
        }

        const float times[2] = {0.0f, 1.0f};
        float got[8];
        if (mynah_flow_head_forward(head, &weights, cond, times, noise, got) !=
            0) {
            flow_set_error(error, error_capacity, "self test: forward failed");
            mynah_flow_head_destroy(head);
            free(pool);
            return -1;
        }

        /* Reference. */
        double y[8], x[8], tmp[8], tmp2[8], mod[24], emb[8], want[8];
        double freqs[4];
        const size_t half = femb / 2u;
        for (size_t i = 0; i < half; ++i) {
            freqs[i] = exp(-log(10000.0) * (double)i / (double)half);
        }
        {
            double condd[8];
            for (size_t i = 0; i < cond_dim; ++i) condd[i] = (double)cond[i];
            ref_linear(weights.cond_embed.weight, weights.cond_embed.bias,
                       condd, y, hidden, cond_dim);
        }
        {
            double acc[8];
            for (size_t i = 0; i < hidden; ++i) acc[i] = 0.0;
            for (size_t k = 0; k < 2u; ++k) {
                for (size_t i = 0; i < half; ++i) {
                    const double arg = (double)times[k] * freqs[i];
                    emb[i] = cos(arg);
                    emb[half + i] = sin(arg);
                }
                ref_linear(time_embed[k].mlp_in.weight,
                           time_embed[k].mlp_in.bias, emb, tmp, hidden, femb);
                ref_silu(tmp, hidden);
                ref_linear(time_embed[k].mlp_out.weight,
                           time_embed[k].mlp_out.bias, tmp, tmp2, hidden,
                           hidden);
                ref_var_rmsnorm(tmp2, time_embed[k].alpha, tmp2, hidden, 1e-5);
                for (size_t i = 0; i < hidden; ++i) acc[i] += tmp2[i];
            }
            for (size_t i = 0; i < hidden; ++i) y[i] += acc[i] / 2.0;
        }
        {
            double noised[8];
            for (size_t i = 0; i < latent; ++i) noised[i] = (double)noise[i];
            ref_linear(weights.input_proj.weight, weights.input_proj.bias,
                       noised, x, hidden, latent);
        }
        for (size_t b = 0; b < depth; ++b) {
            double sy[8];
            for (size_t i = 0; i < hidden; ++i) sy[i] = y[i];
            ref_silu(sy, hidden);
            ref_linear(blocks[b].adaln.weight, blocks[b].adaln.bias, sy, mod,
                       3u * hidden, hidden);
            ref_layernorm(x, blocks[b].in_ln_weight, blocks[b].in_ln_bias, tmp,
                          hidden, 1e-6);
            for (size_t i = 0; i < hidden; ++i) {
                tmp[i] = tmp[i] * (1.0 + mod[hidden + i]) + mod[i];
            }
            ref_linear(blocks[b].mlp_in.weight, blocks[b].mlp_in.bias, tmp,
                       tmp2, hidden, hidden);
            ref_silu(tmp2, hidden);
            ref_linear(blocks[b].mlp_out.weight, blocks[b].mlp_out.bias, tmp2,
                       tmp, hidden, hidden);
            for (size_t i = 0; i < hidden; ++i) {
                x[i] += mod[2u * hidden + i] * tmp[i];
            }
        }
        {
            double sy[8];
            for (size_t i = 0; i < hidden; ++i) sy[i] = y[i];
            ref_silu(sy, hidden);
            ref_linear(weights.final_adaln.weight, weights.final_adaln.bias, sy,
                       mod, 2u * hidden, hidden);
            ref_layernorm(x, NULL, NULL, tmp, hidden, 1e-6);
            for (size_t i = 0; i < hidden; ++i) {
                tmp[i] = tmp[i] * (1.0 + mod[hidden + i]) + mod[i];
            }
            ref_linear(weights.final_linear.weight, weights.final_linear.bias,
                       tmp, want, latent, hidden);
        }

        int failed = 0;
        for (size_t i = 0; i < latent; ++i) {
            if (!flow_close(got[i], want[i], 2e-5)) {
                flow_set_error(error, error_capacity,
                               "flow forward[%zu] = %.9g want %.9g", i,
                               (double)got[i], want[i]);
                failed = 1;
                break;
            }
        }

        /* The memoised time branch must not change the result. */
        if (!failed) {
            float again[8];
            (void)mynah_flow_head_forward(head, &weights, cond, times, noise,
                                          again);
            for (size_t i = 0; i < latent; ++i) {
                if (again[i] != got[i]) {
                    flow_set_error(error, error_capacity,
                                   "time-embed memoisation changed output at "
                                   "%zu: %.9g vs %.9g",
                                   i, (double)again[i], (double)got[i]);
                    failed = 1;
                    break;
                }
            }
        }
        /* A different time must change the result, i.e. the cache keys work. */
        if (!failed) {
            const float other[2] = {0.0f, 0.5f};
            float shifted[8];
            (void)mynah_flow_head_forward(head, &weights, cond, other, noise,
                                          shifted);
            int differs = 0;
            for (size_t i = 0; i < latent; ++i) {
                if (shifted[i] != got[i]) differs = 1;
            }
            if (!differs) {
                flow_set_error(error, error_capacity,
                               "time-embed memoisation ignored a new time");
                failed = 1;
            }
        }

        mynah_flow_head_destroy(head);
        free(pool);
        if (failed) return -1;
    }

    return 0;
}

#undef FLOW_CHECK
