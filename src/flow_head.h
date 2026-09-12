/*
 * SimpleMLPAdaLN — the PocketTTS flow head (E3-3).
 *
 * One evaluation maps a conditioning vector, `num_time_conds` scalar times and
 * a latent-space noise vector to a latent-space sample.  The released
 * checkpoints are LSD-distilled with two time conditions pinned at s = 0 and
 * t = 1, so a full decode is a *single* call per frame; there is no
 * integration loop here on purpose.
 *
 * Reference forward (kyutai-labs/pocket-tts, modules/mlp.py):
 *
 *     y = cond_embed(c) + (time_embed[0](s) + time_embed[1](t)) / num_time_conds
 *     x = input_proj(noise)
 *     per res block:  shift, scale, gate = adaLN(SiLU(y)).chunk(3)
 *                     h = mlp(in_ln(x) * (1 + scale) + shift)
 *                     x = x + gate * h
 *     final:          shift, scale = adaLN(SiLU(y)).chunk(2)
 *                     out = linear(norm_final(x) * (1 + scale) + shift)
 *
 * Three details are load bearing and are the reason this module owns its own
 * normalisation kernels instead of calling `src/kernels.h`:
 *
 *   1. `in_ln` / `norm_final` are LayerNorm with eps = 1e-6 and a *biased*
 *      variance, and `norm_final` has no affine parameters at all (no such
 *      tensor exists in the checkpoint).  `mynah_layernorm_f32` requires a
 *      non-NULL weight.
 *   2. The time-embedding tail normalisation is **not** RMSNorm.  It is
 *          y = x * alpha * rsqrt(eps + var(x))
 *      with `var` mean-subtracted *and* unbiased (dividing by N-1), i.e.
 *      torch's default.  `mynah_rmsnorm_f32` computes a mean-square norm and
 *      is numerically a different function.
 *   3. `freqs` are the deterministic constants
 *          exp(-log(max_period) * arange(half) / half)
 *      identical in every released checkpoint.  They are computed here at
 *      create time and must not be read from the model pack -- but they are
 *      *stored* in BF16, and the reference runtime uses the rounded values.
 *      Measured against build/oracle-pocket: computing them exactly costs
 *      9.8e-4 on the time embedding and 8.6e-5 on the emitted latent, which
 *      eats almost the whole 1e-4 budget; rounding the computed values to
 *      BF16 reproduces the checkpoint bit for bit and brings the time
 *      embedding back to 4.8e-7.  Hence `freqs_bf16_rounded`, on by default.
 *
 * No dimension is baked in: everything comes from `mynah_flow_head_config`.
 * Weights arrive as already-resolved float pointers; this module never formats
 * a tensor name.
 */
#ifndef MYNAH_TTS_FLOW_HEAD_H
#define MYNAH_TTS_FLOW_HEAD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t latent_dim;     /* in and out channels, e.g. 32                  */
    size_t cond_dim;       /* conditioning width, e.g. 1024                 */
    size_t hidden_dim;     /* flow width, e.g. 512                          */
    size_t depth;          /* number of residual blocks, e.g. 6             */
    size_t num_time_conds; /* 2 for LSD, 1 for flow matching, 0 for none    */
    size_t freq_embed_dim; /* sinusoidal width, even, e.g. 256              */
    float max_period;      /* 10000.0f                                      */
    float layernorm_eps;   /* 1e-6f                                         */
    float rmsnorm_eps;     /* 1e-5f                                         */
    int freqs_bf16_rounded; /* 1: round the computed freqs to BF16, which is
                             * what the released checkpoints store.         */
} mynah_flow_head_config;

/* Fills `config` with the PocketTTS defaults for the given dimensions.  The
 * caller still has to take latent/cond/hidden/depth from `model.json`. */
void mynah_flow_head_config_defaults(mynah_flow_head_config *config);

typedef struct {
    const float *weight; /* [out_features][in_features], row major */
    const float *bias;   /* [out_features] or NULL                 */
} mynah_flow_linear;

typedef struct {
    mynah_flow_linear mlp_in;  /* [hidden_dim][freq_embed_dim] */
    mynah_flow_linear mlp_out; /* [hidden_dim][hidden_dim]     */
    const float *alpha;        /* [hidden_dim], variance-RMSNorm gain */
} mynah_flow_time_embed_weights;

typedef struct {
    const float *in_ln_weight;  /* [hidden_dim]               */
    const float *in_ln_bias;    /* [hidden_dim] or NULL       */
    mynah_flow_linear adaln;    /* [3 * hidden_dim][hidden_dim] */
    mynah_flow_linear mlp_in;   /* [hidden_dim][hidden_dim]   */
    mynah_flow_linear mlp_out;  /* [hidden_dim][hidden_dim]   */
} mynah_flow_res_block_weights;

/* ------------------------------------------------- optional linear override
 *
 * The same contract `transformer_ar` uses, and for the same reason: this
 * module owns no weight cache and formats no tensor name, so an engine that
 * wants a quantized projection installs it from outside.  `index` separates
 * the repeated kinds -- time condition i, residual block b -- and is 0 for the
 * singletons.
 *
 * The callee must compute exactly what `mynah_matvec_bias_f32` computes:
 *     out[n] = in[k] @ weight[n][k]^T + (bias ? bias[n] : 0)
 * Returning non-zero fails the forward.  NULL (what a zeroed weights struct
 * has) keeps the f32 matvec, so nothing here changes until an engine opts in.
 *
 * NOTE on the flow head specifically: `input_proj` reads a latent_dim (32)
 * activation and `final_linear` writes one.  A per-row absmax int8 over k = 32
 * is a different numerical proposition from one over k = 1024, which is why
 * the engine is allowed to quantize these kinds individually rather than as
 * one block. */
typedef enum {
    MYNAH_FLOW_LINEAR_TIME_MLP_IN = 0, /* [hidden][freq_embed], per cond    */
    MYNAH_FLOW_LINEAR_TIME_MLP_OUT,    /* [hidden][hidden], per cond        */
    MYNAH_FLOW_LINEAR_COND_EMBED,      /* [hidden][cond_dim]                */
    MYNAH_FLOW_LINEAR_INPUT_PROJ,      /* [hidden][latent_dim]              */
    MYNAH_FLOW_LINEAR_BLOCK_ADALN,     /* [3*hidden][hidden], per block     */
    MYNAH_FLOW_LINEAR_BLOCK_MLP_IN,    /* [hidden][hidden], per block       */
    MYNAH_FLOW_LINEAR_BLOCK_MLP_OUT,   /* [hidden][hidden], per block       */
    MYNAH_FLOW_LINEAR_FINAL_ADALN,     /* [2*hidden][hidden]                */
    MYNAH_FLOW_LINEAR_FINAL_LINEAR,    /* [latent_dim][hidden]              */
    MYNAH_FLOW_LINEAR_KIND_COUNT
} mynah_flow_linear_kind;

typedef int (*mynah_flow_linear_fn)(void *user, size_t index,
                                    mynah_flow_linear_kind kind,
                                    const float *weight, const float *bias,
                                    const float *in, float *out, size_t k,
                                    size_t n);

/* `linear` takes one row; `linear_rows` takes one row per request and MUST be
 * bit-exact per row -- a request's audio may not depend on who it was batched
 * with, so the callee may not let batch width enter the arithmetic. */
typedef int (*mynah_flow_linear_rows_fn)(void *user, size_t index,
                                         mynah_flow_linear_kind kind,
                                         const float *weight, const float *bias,
                                         const float *const *in_rows,
                                         float *const *out_rows, size_t batch,
                                         size_t k, size_t n);

typedef struct {
    mynah_flow_linear cond_embed; /* [hidden_dim][cond_dim]   */
    mynah_flow_linear input_proj; /* [hidden_dim][latent_dim] */
    const mynah_flow_time_embed_weights *time_embed; /* [num_time_conds] */
    const mynah_flow_res_block_weights *res_blocks;  /* [depth]          */
    mynah_flow_linear final_adaln;  /* [2 * hidden_dim][hidden_dim] */
    mynah_flow_linear final_linear; /* [latent_dim][hidden_dim]     */
    /* All optional; NULL keeps the built-in f32 matvec. */
    mynah_flow_linear_fn linear;
    mynah_flow_linear_rows_fn linear_rows;
    void *linear_user;
} mynah_flow_head_weights;

typedef struct mynah_flow_head mynah_flow_head;

/* Creates the scratch state for one context.  Weights are *not* captured: they
 * stay owned read-only by the model and are passed to every forward, so one
 * weight set can back many concurrent contexts. */
mynah_flow_head *mynah_flow_head_create(const mynah_flow_head_config *config,
                                        char *error, size_t error_capacity);
void mynah_flow_head_destroy(mynah_flow_head *head);

const mynah_flow_head_config *mynah_flow_head_get_config(
    const mynah_flow_head *head);

/* Validates that every pointer the configuration requires is present.
 * Returns 0 on success, -1 with a message in `error` otherwise. */
int mynah_flow_head_check_weights(const mynah_flow_head *head,
                                  const mynah_flow_head_weights *weights,
                                  char *error, size_t error_capacity);

/*
 * One flow-head evaluation.  Allocates nothing.
 *
 *   cond  [cond_dim]        conditioning, e.g. the backbone's out_norm result
 *   times [num_time_conds]  the time conditions, {0, 1} for the released LSD
 *   noise [latent_dim]      the gaussian draw
 *   out   [latent_dim]      the sampled latent
 *
 * `times` may be NULL when num_time_conds == 0.  The time-embedding branch is
 * memoised on the exact bit pattern of `times`, which is free in practice
 * because s and t are constant across a whole utterance.
 */
int mynah_flow_head_forward(mynah_flow_head *head,
                            const mynah_flow_head_weights *weights,
                            const float *cond, const float *times,
                            const float *noise, float *out);

/* Drops the memoised time embedding.  Only needed if weights change under a
 * live head; correctness does not depend on calling it. */
void mynah_flow_head_reset(mynah_flow_head *head);

/*
 * Cross-request batching.
 *
 * The head is ~8.9 M parameters and one evaluation is ~1.2 MFLOP per frame, so
 * a single forward is a 36 MB trip to memory for 36 KFLOP of work: N requests
 * evaluated one after another pay that trip N times for the same bytes.  This
 * evaluates N of them with one pass over the weights.
 *
 * The scratch belongs to the driver, not to a request, and must not be shared
 * between threads that evaluate concurrently.
 */
typedef struct mynah_flow_head_batch mynah_flow_head_batch;

mynah_flow_head_batch *mynah_flow_head_batch_new(
    const mynah_flow_head_config *config, size_t max_rows, char *error,
    size_t error_capacity);
void mynah_flow_head_batch_free(mynah_flow_head_batch *batch);
size_t mynah_flow_head_batch_capacity(const mynah_flow_head_batch *batch);

/*
 * `count` evaluations, one per head, with one pass over the weights.
 *
 *   heads [count]  distinct heads, all created from the same configuration
 *   cond  [count]  one `[cond_dim]` row each
 *   noise [count]  one `[latent_dim]` row each
 *   out   [count]  one `[latent_dim]` row each
 *
 * `times` is ONE vector for the whole batch.  The released LSD checkpoints pin
 * s = 0 and t = 1 for every request, so this is not a restriction; a caller that
 * ever needs per-row times must use the single forward, because the time branch
 * is a memoised per-head computation and batching it would silently make one
 * request's conditioning depend on another's.
 *
 * Every row is computed exactly as `_forward` would have computed it alone.
 * `count == 0` is a no-op; `count == 1` is `_forward`.  Returns 0, or -1.
 */
int mynah_flow_head_forward_batch(mynah_flow_head *const *heads, size_t count,
                                  const mynah_flow_head_weights *weights,
                                  const float *const *cond, const float *times,
                                  const float *const *noise, float *const *out,
                                  mynah_flow_head_batch *batch);

/* ---- kernels, exported because they are new and separately testable ---- */

/* LayerNorm over `n` values, biased variance.  `weight` and `bias` may both be
 * NULL (elementwise_affine=False); `bias` may be NULL on its own. */
void mynah_flow_layernorm_f32(const float *input, const float *weight,
                              const float *bias, float *output, size_t n,
                              float epsilon);

/* y = x * alpha * rsqrt(epsilon + var(x)), var mean-subtracted and unbiased
 * (divided by n-1).  This is NOT mynah_rmsnorm_f32.  Requires n >= 2. */
void mynah_flow_var_rmsnorm_f32(const float *input, const float *alpha,
                                float *output, size_t n, float epsilon);

/* y = x * sigmoid(x), in place when output == input. */
void mynah_flow_silu_f32(const float *input, float *output, size_t n);

/* y = x * (1 + scale) + shift, in place when output == input. */
void mynah_flow_modulate_f32(const float *input, const float *shift,
                             const float *scale, float *output, size_t n);

/* Writes exp(-log(max_period) * i / half) for i in [0, half). */
void mynah_flow_timestep_freqs_f32(float *freqs, size_t half, float max_period);

/* Rounds each value to BF16 precision (round half to even) and widens it back
 * to f32, i.e. what a BF16 checkpoint round trip does. */
void mynah_flow_round_bf16_f32(float *values, size_t n);

/* Model-free self test of every kernel above plus a full tiny forward.
 * Returns 0 on success, -1 with a message in `error`. */
int mynah_flow_head_self_test(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_TTS_FLOW_HEAD_H */
