/*
 * Shared causal autoregressive transformer (E3).
 *
 * This is the backbone shape used by PocketTTS' `flow_lm.transformer` and by
 * Mimi's `decoder_transformer`.  It exists because the transformer helpers in
 * `graph.c` are not actually shared: they take a name prefix and then format
 * NeMo tensor names, so a second engine cannot use them
 * (`.work/engine-seam-refactor.md`, risk 6).  **Every weight here arrives as an
 * already-resolved `const float *`.  This module never formats, parses or
 * looks up a tensor name, and never sees a model pack.**
 *
 * Reference (kyutai-labs/pocket-tts, `modules/transformer.py`,
 * `modules/attention.py`, `modules/rope.py`):
 *
 *     x = x + layer_scale_1(self_attn(norm1(x)))
 *     x = x + layer_scale_2(linear2(gelu_tanh(linear1(norm2(x)))))
 *
 * The details that a from-scratch implementation gets wrong silently:
 *
 *   - **QKV is pre-fused.**  `self_attn.in_proj.weight` is `[3 * attn_dim,
 *     d_model]`, packed as `view(b, t, 3, heads, head_dim)`, so one matvec
 *     produces q, k and v and each third is head-major.
 *   - **The FFN is not gated**: `linear1` -> GELU(approximate="tanh") ->
 *     `linear2`.  No SwiGLU, no third projection.
 *   - **LayerNorm has weight *and* bias and eps is 1e-5**, against the flow
 *     head's affine-free 1e-6 (`src/flow_head.h`).  Getting the epsilon from
 *     the wrong neighbour is a plausible-looking, entirely wrong output.
 *     Unlike the flow head this module does *not* grow its own normalisation
 *     kernel: `mynah_layernorm_f32` already computes exactly this (biased
 *     variance, optional bias, SIMD), it only insists on a non-NULL weight —
 *     which every norm in this family has.  A weight-less LayerNorm is
 *     therefore rejected here rather than silently approximated.
 *   - **RoPE is interleaved** (adjacent pairs `(2i, 2i+1)` of each head), not
 *     split-halves, with `freqs[i] = exp(i * -log(max_period) * 2 / head_dim)`
 *     and the *absolute* position as the angle multiplier.
 *   - **`layer_scale` is per-model, not per-architecture.**  The PocketTTS
 *     backbone builds `nn.Identity()`; Mimi's decoder transformer builds
 *     `LayerScale(d_model, 0.01)` and ships `layer_scale_{1,2}.scale [512]`.
 *     A NULL pointer here means identity.
 *   - **`context` is a sliding window.**  The backbone passes `None`
 *     (unlimited); Mimi's decoder transformer passes 250.  Zero means
 *     unlimited.
 *
 * ## KV cache layout — chosen to match the voice files byte for byte
 *
 * `mynah_transformer_ar_state_kv()` hands back one contiguous block per layer
 * shaped `[2][max_seq_len][num_heads][head_dim]`, K first then V, holding
 * **post-RoPE** keys.  That is exactly `_LinearKVCacheBackend.init_state`'s
 * `[2, B, T, H, D]` at B = 1, verified against
 * `languages/english_2026-04/embeddings/alba.safetensors`
 * (`transformer.layers.N.self_attn/cache`, F32 `[2, 1, 126, 16, 64]`, plus
 * `offset` I64 `[1]` = 126).  So a predefined voice loads with two memcpys per
 * layer and **no layout conversion**: only the T capacity differs, the
 * per-position `[heads][head_dim]` interior is identical.
 *
 * ## NaN as a sentinel — how it is handled here
 *
 * Upstream uses NaN twice: `flow_lm.forward` replaces NaN rows of the latent
 * sequence with `bos_emb`, and `_expand_kv_cache` fills the unused tail of a
 * grown KV cache with NaN.  Neither sentinel exists inside this module:
 *
 *   1. **Validity is explicit.**  The state carries an integer `offset`, the
 *      number of cached positions.  Attention only ever reads `[lo, offset+i]`,
 *      so the uninitialised tail is unreachable by construction rather than by
 *      arithmetic.  The backing store is `calloc`ed, so even a bug reads zeros,
 *      not NaN.
 *   2. **NaN is rejected at every entrance.**  `_load_kv` refuses a
 *      non-finite voice cache, `_prefill`/`_step` refuse a non-finite input,
 *      and the attention softmax (`mynah_softmax_f32`) refuses non-finite
 *      scores.  A NaN therefore fails loudly at the boundary instead of
 *      reaching a matmul and turning the whole utterance into silence.
 *   3. The BOS substitution stays where it belongs: the caller applies
 *      `bos_emb` before `input_linear`, so this module only ever sees a real
 *      `[d_model]` vector.
 *
 * ## Prefill
 *
 * `_prefill` is `_step` run over n tokens.  For causal attention with a KV
 * cache the two are the same function: token i writes its own K/V and then
 * attends to `[0, offset+i]`, which is what a batched masked SDPA computes.
 * Keeping one implementation is deliberate (CLAUDE.md rule 7); a batched GEMM
 * prefill is a performance change to make later, under the oracle, and it must
 * not become a second graph.
 *
 * Nothing is allocated after `_state_new`: `_step` runs entirely out of the
 * scratch owned by the state.
 */
#ifndef MYNAH_TTS_TRANSFORMER_AR_H
#define MYNAH_TTS_TRANSFORMER_AR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t d_model;     /* residual width, e.g. 1024                        */
    size_t num_heads;   /* e.g. 16                                          */
    size_t head_dim;    /* e.g. 64; 0 means d_model / num_heads             */
    size_t num_layers;  /* e.g. 6                                           */
    size_t ffn_dim;     /* e.g. 4096                                        */
    size_t max_seq_len; /* KV capacity in positions, voice prefix included  */
    size_t context;     /* sliding attention window; 0 = unlimited          */
    float max_period;   /* RoPE, 10000.0f                                   */
    float layernorm_eps; /* 1e-5f for this family                           */
} mynah_transformer_ar_config;

/* Fills in the values that are architecture constants rather than model
 * dimensions (max_period, eps, unlimited context).  The caller still has to
 * take every dimension from `model.json`. */
void mynah_transformer_ar_config_defaults(mynah_transformer_ar_config *config);

/*
 * One layer's weights, already resolved.  `attn_dim` below is
 * `num_heads * head_dim`.  A NULL bias means "no bias", which is the
 * PocketTTS case for every projection in this block.
 */
typedef struct {
    const float *in_proj_weight;  /* [3 * attn_dim][d_model], q|k|v head-major */
    const float *in_proj_bias;    /* [3 * attn_dim]  or NULL                   */
    const float *out_proj_weight; /* [d_model][attn_dim]                       */
    const float *out_proj_bias;   /* [d_model]       or NULL                   */
    const float *norm1_weight;    /* [d_model], required                       */
    const float *norm1_bias;      /* [d_model] or NULL                         */
    const float *norm2_weight;    /* [d_model], required                       */
    const float *norm2_bias;      /* [d_model] or NULL                         */
    const float *linear1_weight;  /* [ffn_dim][d_model]                        */
    const float *linear1_bias;    /* [ffn_dim]  or NULL                        */
    const float *linear2_weight;  /* [d_model][ffn_dim]                        */
    const float *linear2_bias;    /* [d_model]  or NULL                        */
    const float *layer_scale_1;   /* [d_model] or NULL = nn.Identity()         */
    const float *layer_scale_2;   /* [d_model] or NULL = nn.Identity()         */
} mynah_transformer_ar_layer;

typedef struct {
    const mynah_transformer_ar_layer *layers; /* [num_layers] */
    /* Optional final LayerNorm applied to the stack output, at the same eps.
     * PocketTTS' `flow_lm.out_norm` is this; Mimi's decoder transformer has no
     * such tensor, so both pointers are NULL there. */
    const float *out_norm_weight;
    const float *out_norm_bias;
} mynah_transformer_ar_weights;

typedef struct mynah_transformer_ar_state mynah_transformer_ar_state;

/* Per-context state: KV cache, RoPE table and every scratch buffer `_step`
 * needs.  Weights are *not* captured; they stay owned read-only by the model
 * and are passed to each forward, so one weight set backs many contexts
 * (CLAUDE.md rule 3). */
mynah_transformer_ar_state *mynah_transformer_ar_state_new(
    const mynah_transformer_ar_config *config, char *error,
    size_t error_capacity);
void mynah_transformer_ar_state_free(mynah_transformer_ar_state *state);

/* Rewinds to position 0.  The cache memory is left alone: nothing past the
 * offset is ever read, and it is overwritten before it becomes readable. */
void mynah_transformer_ar_state_reset(mynah_transformer_ar_state *state);

const mynah_transformer_ar_config *mynah_transformer_ar_state_config(
    const mynah_transformer_ar_state *state);

/* Number of cached positions, i.e. the absolute position the next token gets. */
size_t mynah_transformer_ar_state_offset(const mynah_transformer_ar_state *state);

/* The raw KV block for one layer: `[2][max_seq_len][num_heads][head_dim]`,
 * K then V, post-RoPE.  Exposed so a converter can write a voice prefix
 * straight into it; prefer `_load_kv`, which bounds-checks and rejects NaN. */
float *mynah_transformer_ar_state_kv(mynah_transformer_ar_state *state,
                                     size_t layer);

/* Floats in one half (K or V) of a layer's block: max_seq_len*heads*head_dim. */
size_t mynah_transformer_ar_state_kv_half_floats(
    const mynah_transformer_ar_state *state);

/*
 * Copies a voice prefix into layer `layer`.  `kv` is
 * `[2][positions][num_heads][head_dim]` — the voice safetensors layout with
 * the batch axis dropped.  Rejects a non-finite entry, which is the one place
 * upstream's NaN padding could get in.  Does not move the offset: call
 * `_set_offset(positions)` once all layers are loaded.
 */
int mynah_transformer_ar_state_load_kv(mynah_transformer_ar_state *state,
                                       size_t layer, const float *kv,
                                       size_t positions, char *error,
                                       size_t error_capacity);

/* Declares `positions` cached positions valid, i.e. the voice file's
 * `current_end` / `offset`. */
int mynah_transformer_ar_state_set_offset(mynah_transformer_ar_state *state,
                                          size_t positions, char *error,
                                          size_t error_capacity);

/* Validates that every pointer the configuration requires is present.
 * Returns 0, or -1 with a message in `error`. */
int mynah_transformer_ar_check_weights(
    const mynah_transformer_ar_state *state,
    const mynah_transformer_ar_weights *weights, char *error,
    size_t error_capacity);

/*
 * Runs `n_tokens` consecutive positions starting at the current offset and
 * advances the offset by `n_tokens`.
 *
 *   x    [n_tokens][d_model]  row-major input
 *   out  [n_tokens][d_model]  row-major output, or NULL to discard
 *
 * `n_tokens == 0` is a no-op.  Returns 0, or -1 on a bad argument, an overflow
 * of `max_seq_len`, or a non-finite value reaching the attention.
 */
int mynah_transformer_ar_prefill(mynah_transformer_ar_state *state,
                                 const mynah_transformer_ar_weights *weights,
                                 const float *x, size_t n_tokens, float *out);

/* One autoregressive position.  `x` and `out` are `[d_model]`; `out` may not
 * be NULL.  Allocates nothing. */
int mynah_transformer_ar_step(mynah_transformer_ar_state *state,
                              const mynah_transformer_ar_weights *weights,
                              const float *x, float *out);

/* ---- kernels, exported because they are new and separately testable ---- */

/* cos/sin of `position * freqs[i]` for i in [0, half), with
 * `freqs[i] = exp(i * -log(max_period) * 2 / (2 * half))`.  Computed in f32 to
 * match the reference, which builds the table from a float32 arange. */
void mynah_transformer_ar_rope_angles_f32(float *cosines, float *sines,
                                          size_t half, size_t position,
                                          float max_period);

/* Rotates `[num_heads][head_dim]` in place, interleaved: each adjacent pair
 * `(2i, 2i+1)` is one complex number turned by `(cosines[i], sines[i])`. */
void mynah_transformer_ar_rope_apply_f32(float *values, size_t num_heads,
                                         size_t head_dim, const float *cosines,
                                         const float *sines);

/* Model-free self test: RoPE, LayerNorm with bias, GELU-tanh, causal attention
 * with and without a `context` window, KV-cache continuity (prefill of N plus
 * one step equals prefill of N+1), and layer_scale NULL against an explicit
 * unit vector.  Returns 0, or -1 with a message in `error`. */
int mynah_transformer_ar_self_test(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_TTS_TRANSFORMER_AR_H */
