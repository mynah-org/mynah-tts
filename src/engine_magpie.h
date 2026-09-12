/* Magpie graph: text encoder, incremental decoder with KV cache, local
 * transformer and its sampler.
 *
 * This is `graph.c` renamed to what it always was. The E1 split moved the
 * shared services (mynah_util), the kernels, the causal convolutions (conv1d)
 * and the NanoCodec decoder out; what is left formats NeMo tensor names and
 * assumes Magpie's shapes, so it is an engine, not a shared transformer.
 *
 * The types below are exposed because the driver in inference.c holds a
 * decoder cache and a local-transformer state by value inside each slot. That
 * surface is exactly what the engine vtable will collapse (PLAN.md E1-1); until
 * then it is written out honestly rather than hidden behind a void pointer.
 */
#ifndef MYNAH_TTS_ENGINE_MAGPIE_H
#define MYNAH_TTS_ENGINE_MAGPIE_H

#include <stdint.h>

#include "backend.h"
#include "conv1d.h"
#include "codec_nanocodec.h"
#include "kernels.h"
#include "mynah_tts_internal.h"
#include "mynah_util.h"
#include "weights.h"

/* Upper bound on requests stepped together. Slots are cheap in state but each
 * holds a full KV cache, so this bounds memory as much as the pointer arrays
 * the batched step keeps on the stack. */
#define MYNAH_MAX_BATCH 16u

/* Scratch for the weight-stationary batched projections.  Owned by the caller
 * and sized once for the widest k in the graph, so the step stays
 * allocation-free.  Unused at batch 1, where every projection takes the exact
 * single-row path. */
typedef struct {
    int8_t *qx;   /* MYNAH_MAX_BATCH * k_max */
    float *sx;    /* MYNAH_MAX_BATCH */
    size_t k_max;
    size_t capacity;
} batch_scratch;

/* Small KV cache for the autoregressive local transformer.  It has no cross
 * attention and its sequence is at most stream_count+1 (17) positions, so the
 * attention stays scalar and each of the 16 codebook streams is a single-row
 * step instead of re-running the whole stack from scratch every time. */
typedef struct {
    size_t layers;
    size_t width;
    size_t heads;
    size_t head_width;
    size_t ffn_width;
    size_t capacity;
    size_t length;
    float *k; /* layers * capacity * width */
    float *v; /* layers * capacity * width */
    const float *position; /* local_transformer.position_embeddings.weight */
    /* Pre-resolved weight pointers (eliminate per-step snprintf+lookup). */
    const float *norm_self[4];
    const float *qkv_w[4];
    const float *o_w[4];
    const float *norm_ff[4];
    const float *ffn_up_w[4];
    const float *ffn_down_w[4];
    /* The tensor names are kept beside the pointers so the quantized path can
     * key its converted-weight cache without re-formatting a name per step.
     * These four matmuls are the hot loop: 16 sequential streams per stacked
     * frame re-read every local-transformer weight, so they dominate decode. */
    char qkv_name[4][96];
    char o_name[4][96];
    char ffn_up_name[4][96];
    char ffn_down_name[4][96];
    /* Metal resident local-transformer state.  Host K/V and scratch remain
     * for the scalar path; the GPU path never downloads its hidden row. */
    float *dev_k;
    float *dev_v;
    float *dev_x;
    float *dev_nrm;
    float *dev_qkv;
    float *dev_attn;
    float *dev_proj;
    float *dev_hidden;
    float *dev_logits;
    float *dev_input;
    const mynah_backend *dev_backend;
    int gpu_ready;
} local_cache;

typedef struct {
    float *x;
    float *nrm;
    float *qkv;
    float *attn;
    float *proj;
    float *hidden;
    float *scores;
    float *gelu_scratch;
} local_workspace;

/* Per-request state for the local autoregressive helper.
 *
 * The local transformer restarts at position 0 for every stacked frame, but
 * its KV cache, workspace and sampler scratch are shape-constant for the whole
 * request.  Allocating them once and resetting `length` keeps the decode loop
 * allocation-free -- previously every frame paid two KV allocations plus the
 * workspace, the sampler scratch and, under Metal, ten device allocations. */
typedef struct {
    local_cache cache;
    local_workspace workspace;
    float *row_in;
    float *row_out;
    float *logits;        /* audio_vocab_size */
    size_t *top_indices;  /* top_count, sampling only */
    float *top_logits;    /* top_count, sampling only */
    size_t top_count;
} local_frame_state;

/* One request's slice of a batched local frame. */
typedef struct {
    local_frame_state *state;
    const float *decoder_last;
    const float *decoder_dev_last;
    unsigned *codes;
    size_t raw_offset;
    size_t code_stride;
    size_t generated_raw_length;
    size_t min_raw_length;
    float temperature;
    unsigned topk;
    uint64_t *rng_state;
    /* filled in */
    int saw_eos;
    size_t eos_frame;
    int failed;
} local_batch_item;

/* Pre-resolved per-layer weight names and norm pointers so the AR hot loop
 * never calls snprintf or probes the tensor hash table. */
typedef struct {
    const float *norm_self;
    const float *norm_xattn_query;
    const float *norm_pos_ff;
    const float *qkv_w;
    const float *o_self_w;
    const float *q_cross_w;
    const float *o_cross_w;
    const float *ffn_up_w;
    const float *ffn_down_w;
    char qkv[160];
    char o_self[160];
    char q_cross[160];
    char o_cross[160];
    char ffn_up[160];
    char ffn_down[160];
} decoder_layer_resolved;

typedef struct {
    size_t layers;
    size_t width;
    size_t heads;
    size_t head_width;
    size_t ffn_width;
    size_t xattn_width;
    size_t memory_length;
    size_t capacity;
    size_t length;
    float *self_k;   /* layers * capacity * width */
    float *self_v;   /* layers * capacity * width */
    float *cross_k;  /* layers * memory_length * xattn_width */
    float *cross_v;  /* layers * memory_length * xattn_width */
    const float *position; /* decoder.position_embeddings.weight */
    /* pre-resolved weights */
    decoder_layer_resolved *resolved;
    const float *norm_out;
    /* reusable scratch (sized for scratch_rows) */
    size_t scratch_rows;
    float *scratch_x;
    float *scratch_nrm;
    float *scratch_qkv;
    float *scratch_attn;
    float *scratch_proj;
    float *scratch_q_x;
    float *scratch_xctx;
    float *scratch_hidden;
    float *scratch_scores;
    float *scratch_gelu;
    float *scratch_score_matrix;
    float *scratch_head_ctx;
    /* GPU resident-step device buffers (allocated once, reused per step). */
    float *dev_nrm;
    float *dev_qkv;
    float *dev_attn;
    float *dev_proj;
    float *dev_qx;
    float *dev_xctx;
    float *dev_hidden;
    float *dev_x;
    float *dev_self_k;
    float *dev_self_v;
    float *dev_cross_k;
    float *dev_cross_v;
    int dev_allocated;
    int dev_attention_allocated;
    const mynah_backend *dev_backend; /* for dev_free */
} decoder_cache;
/* ---- batched projection scratch ---- */
int  magpie_batch_scratch_init(batch_scratch *scratch, size_t batch, size_t k_max,
                               char *error, size_t error_capacity);
void magpie_batch_scratch_free(batch_scratch *scratch);

/* ---- graph ---- */
int  magpie_encode_text(const mynah_tts_model *model, const int *ids, size_t count,
                        float **encoded, char *error, size_t error_capacity);
int  magpie_embed_audio_frame(const mynah_tts_model *model, const unsigned *codes,
                              size_t code_stride, size_t frame, float *row,
                              char *error, size_t error_capacity);
size_t magpie_stacked_stream_count(const mynah_tts_model *model);

/* ---- decoder ---- */
int  magpie_decoder_cache_init(const mynah_tts_model *model, decoder_cache *cache,
                               const float *memory, size_t memory_length,
                               size_t capacity, char *error, size_t error_capacity);
void magpie_decoder_cache_free(decoder_cache *cache);
int  magpie_decoder_run(const mynah_tts_model *model, decoder_cache *cache,
                        const float *input_rows, size_t count, float *out_last,
                        float **dev_out, char *error, size_t error_capacity);
int  magpie_decoder_step_batch(const mynah_tts_model *model,
                               decoder_cache *const *caches,
                               const float *const *inputs, float *const *outs,
                               size_t batch, const batch_scratch *scratch,
                               char *error, size_t error_capacity);

/* ---- local transformer ---- */
int  magpie_local_frame_state_init(const mynah_tts_model *model,
                                   local_frame_state *state, size_t top_count,
                                   char *error, size_t error_capacity);
void magpie_local_frame_state_free(local_frame_state *state);
int  magpie_sample_local_frame_batch(const mynah_tts_model *model,
                                     local_batch_item *items, size_t count,
                                     const batch_scratch *scratch,
                                     char *error, size_t error_capacity);

#endif
