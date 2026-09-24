#ifndef MYNAH_BACKEND_H
#define MYNAH_BACKEND_H

#include "mynah_tts.h"
#include "seanet.h"

#include <stddef.h>

typedef struct mynah_backend mynah_backend;
typedef struct mynah_backend_decoder mynah_backend_decoder;

typedef int (*mynah_backend_matmul_fn)(void *, const float *, float *, size_t, size_t,
                                       size_t, const float *, const float *, char *, size_t);
typedef void (*mynah_backend_close_fn)(void *);
typedef int (*mynah_backend_self_test_fn)(void *, char *, size_t);

typedef mynah_conv_weights mynah_backend_decoder_weight;

/* Descriptor for the resident Pocket SEANet decoder. The arrays are borrowed
 * during open only; the backend uploads every weight it needs before the
 * first decode step. Blocks are flattened stage-major, then residual-layer
 * major, with one entry for each conv1/conv2 pair. */
typedef struct {
    size_t channels;
    size_t dimension;
    size_t n_filters;
    size_t n_residual_layers;
    const size_t *ratios;
    size_t n_ratios;
    size_t kernel_size;
    size_t residual_kernel_size;
    size_t last_kernel_size;
    size_t dilation_base;
    size_t compress;
    float elu_alpha;
    mynah_backend_decoder_weight first;
    const mynah_backend_decoder_weight *convtr;
    const mynah_seanet_resblock_weights *blocks;
    mynah_backend_decoder_weight last;
} mynah_backend_decoder_desc;

/* Generic sgemm mirroring cblas_sgemm semantics (row-major).
 * C[m,n] = alpha * op(A) * op(B) + beta * C
 * trans_a/trans_b: 0 = NoTrans, 1 = Trans. */
typedef int (*mynah_backend_sgemm_fn)(void *, int trans_a, int trans_b,
                                      size_t m, size_t n, size_t k,
                                      float alpha,
                                      const float *a, size_t lda,
                                      const float *b, size_t ldb,
                                      float beta,
                                      float *c, size_t ldc,
                                      char *, size_t);

int mynah_backend_open(mynah_tts_device device, mynah_backend **out,
                       char *error, size_t error_capacity);
void mynah_backend_close(mynah_backend *backend);
const char *mynah_backend_name(const mynah_backend *backend);

int mynah_backend_matmul(const mynah_backend *backend, const float *input,
                         float *output, size_t rows, size_t input_width,
                         size_t output_width, const float *weight,
                         const float *bias, char *error, size_t error_capacity);

int mynah_backend_sgemm(const mynah_backend *backend,
                        int trans_a, int trans_b,
                        size_t m, size_t n, size_t k,
                        float alpha,
                        const float *a, size_t lda,
                        const float *b, size_t ldb,
                        float beta,
                        float *c, size_t ldc,
                        char *error, size_t error_capacity);

int mynah_backend_self_test(mynah_tts_device device, char *error,
                            size_t error_capacity);

int mynah_backend_decoder_open(const mynah_backend *backend,
                               const mynah_backend_decoder_desc *desc,
                               size_t max_encoder_frames,
                               mynah_backend_decoder **out,
                               char *error, size_t error_capacity);
void mynah_backend_decoder_close(const mynah_backend *backend,
                                 mynah_backend_decoder *decoder);
int mynah_backend_decoder_reset(const mynah_backend *backend,
                                mynah_backend_decoder *decoder,
                                char *error, size_t error_capacity);
int mynah_backend_decoder_step(const mynah_backend *backend,
                               mynah_backend_decoder *decoder,
                               const float *dev_input,
                               size_t encoder_frames,
                               float *dev_output,
                               char *error, size_t error_capacity);
/* A decoder step recorded inside a CUDA graph is submitted during capture but
 * does not execute until the graph is launched.  Backends use this hook to
 * keep the process-local step counter logical rather than counting capture
 * submissions.  CPU/unsupported backends treat it as a no-op. */
int mynah_backend_decoder_note_step(const mynah_backend *backend,
                                    mynah_backend_decoder *decoder);
/* Record one decoder gang submission. This is a diagnostic seam for the
 * asynchronous CUDA decoder path; CPU and other backends treat it as a no-op. */
int mynah_backend_decoder_note_batch(const mynah_backend *backend,
                                     size_t items, size_t frames);
/* Record one successful cross-request Pocket backbone batch. CPU/Metal are
 * intentionally no-ops; CUDA exposes the counters for server observability. */
int mynah_backend_note_backbone_batch(const mynah_backend *backend,
                                      size_t items);
/* Record one resident Mimi decoder-transformer tile. */
int mynah_backend_note_codec_transformer_batch(const mynah_backend *backend,
                                               size_t items, size_t width);
int mynah_backend_metrics_get(const mynah_backend *backend,
                               mynah_tts_backend_metrics *metrics);

/* Query: does this backend support device-side matmul (resident GPU)? */
int mynah_backend_has_dev_ops(const mynah_backend *backend);

/* Inplace device ops (no copy, no sync — caller syncs when needed). */
int mynah_backend_gelu_inplace(const mynah_backend *, float *, size_t, char *, size_t);
int mynah_backend_residual_inplace(const mynah_backend *, float *, const float *, size_t, char *, size_t);
int mynah_backend_layer_norm_inplace(const mynah_backend *, const float *, float *, const float *, size_t, size_t, char *, size_t);
int mynah_backend_matmul_to_dev(const mynah_backend *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
int mynah_backend_matmul_d2d(const mynah_backend *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
int mynah_backend_im2col(const mynah_backend *, const float *, float *, int, int, int, int, char *, size_t);
int mynah_backend_conv1d(const mynah_backend *, const float *, float *, int, int, int, int, int, const float *, const float *, char *, size_t);
/* Device-resident causal conv1d.  `input` and `output` are backend-owned
 * device buffers; weights and bias remain host model-pack views and are
 * cached by the backend.  This is deliberately separate from conv1d(), whose
 * contract is host input/output and may synchronize. */
int mynah_backend_conv1d_dev(const mynah_backend *, const float *, float *, int, int, int, int, int, const float *, const float *, char *, size_t);
int mynah_backend_conv_transpose_dev(const mynah_backend *, const float *, float *, int, int, int, int, int, int, int, const float *, const float *, char *, size_t);
int mynah_backend_gelu_host(const mynah_backend *, float *, size_t, char *, size_t);
int mynah_backend_gelu_host_f64(const mynah_backend *, float *, size_t, char *, size_t);
int mynah_backend_matmul_graph(const mynah_backend *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);

/* Device-side matvec: out[N] = in[K] @ W[N,K]^T + bias. No sync. */
int mynah_backend_matvec_dev(const mynah_backend *backend,
                             const float *dev_in, float *dev_out,
                             size_t K, size_t N,
                             const float *weight, const float *bias,
                             char *error, size_t error_capacity);

/* ---- Device-side operations (resident GPU inference) ----
 * These keep activations on the device between calls, eliminating
 * per-op H2D/D2H copies.  On CPU backends they are trivial wrappers.
 * The caller uploads input once, chains ops, then downloads output. */

/* Upload n floats from host to a device buffer; returns device pointer. */
int mynah_backend_upload(const mynah_backend *backend, const float *host,
                         size_t n, float **dev_ptr,
                         char *error, size_t error_capacity);
/* Download n floats from device to host. */
int mynah_backend_download(const mynah_backend *backend, const float *dev_ptr,
                           float *host, size_t n,
                           char *error, size_t error_capacity);
/* Block until all queued device work completes. */
int mynah_backend_sync(const mynah_backend *backend,
                       char *error, size_t error_capacity);

/* Begin a resident GPU command batch.  CPU backends are no-ops.  The batch is
 * submitted by mynah_backend_sync at the next CPU-visible boundary. */
int mynah_backend_batch_begin(const mynah_backend *backend,
                              char *error, size_t error_capacity);

/* CUDA-Graph command capture for a resident engine step.  The backend keeps
 * graph objects opaque and keyed by (key, identity); `identity` is normally
 * the engine scratch arena whose device addresses are embedded in the graph.
 * begin returns 0 when graph support is available and sets `replay` to 1 for
 * an existing graph or 0 for a new capture.  It returns 1 when graphs are
 * disabled/unavailable, which is a normal fallback rather than an error. */
int mynah_backend_graph_begin(const mynah_backend *backend, size_t key,
                              const void *identity, int *replay,
                              char *error, size_t error_capacity);
int mynah_backend_graph_end(const mynah_backend *backend, size_t key,
                            const void *identity, char *error,
                            size_t error_capacity);
int mynah_backend_graph_launch(const mynah_backend *backend, size_t key,
                               const void *identity, char *error,
                               size_t error_capacity);
void mynah_backend_graph_abort(const mynah_backend *backend, size_t key,
                               const void *identity);
void mynah_backend_graph_forget(const mynah_backend *backend,
                                const void *identity);

/* Pocket flow-head descriptor. The engine owns the host weights and device
 * scratch; CUDA owns cached weight copies and the chained kernels. No CUDA or
 * cuBLAS type crosses this seam, and the operation is asynchronous. */
typedef struct {
    const float *weight;
    const float *bias;
} mynah_backend_flow_linear;

typedef struct {
    const float *in_ln_weight;
    const float *in_ln_bias;
    mynah_backend_flow_linear adaln;
    mynah_backend_flow_linear mlp_in;
    mynah_backend_flow_linear mlp_out;
} mynah_backend_flow_block;

typedef struct {
    size_t batch;
    size_t latent_dim;
    size_t cond_dim;
    size_t hidden_dim;
    size_t depth;
    size_t num_time_conds;
    size_t freq_embed_dim;
    float layernorm_eps;
    float rmsnorm_eps;
    const float *dev_cond;       /* [batch][cond_dim]      */
    const float *dev_noise;      /* [batch][latent_dim]    */
    const float *dev_time_embed; /* [time][freq_embed_dim] */
    float *dev_y;                /* [batch][hidden]         */
    float *dev_silu;             /* [batch][hidden]         */
    float *dev_x;                /* [batch][hidden]         */
    float *dev_norm;             /* [batch][hidden]         */
    float *dev_hidden;           /* [batch][hidden]         */
    float *dev_scratch;          /* [batch][hidden]         */
    float *dev_mod;              /* [batch][3*hidden]       */
    float *dev_final_mod;        /* [batch][2*hidden]       */
    float *dev_time_hidden;      /* [time][hidden]          */
    float *dev_time_output;      /* [time][hidden]          */
    float *dev_time_sum;         /* [hidden]                */
    float *dev_out;              /* [batch][latent]         */
    const mynah_backend_flow_linear *time_mlp_in;
    const mynah_backend_flow_linear *time_mlp_out;
    const float *const *time_alpha;
    const mynah_backend_flow_linear *cond_embed;
    const mynah_backend_flow_linear *input_proj;
    const mynah_backend_flow_block *blocks;
    const mynah_backend_flow_linear *final_adaln;
    const mynah_backend_flow_linear *final_linear;
} mynah_backend_flow_batch;

int mynah_backend_flow_batch_dev(const mynah_backend *,
                                 const mynah_backend_flow_batch *,
                                 char *, size_t);

/* Device-side matmul: out[rows,ow] = in[rows,iw] @ W[ow,iw]^T + bias.
 * Weight/bias are host pointers (cached on device internally). */
int mynah_backend_matmul_dev(const mynah_backend *backend,
                             const float *dev_in, float *dev_out,
                             size_t rows, size_t iw, size_t ow,
                             const float *weight, const float *bias,
                             char *error, size_t error_capacity);
/* Device-side sgemm (row-major, same semantics as mynah_backend_sgemm). */
int mynah_backend_sgemm_dev(const mynah_backend *backend,
                            int trans_a, int trans_b,
                            size_t m, size_t n, size_t k,
                            float alpha,
                            const float *dev_a, size_t lda,
                            const float *dev_b, size_t ldb,
                            float beta,
                            float *dev_c, size_t ldc,
                            char *error, size_t error_capacity);
/* Element-wise ops on device buffers. */
int mynah_backend_gelu_dev(const mynah_backend *backend,
                           float *dev_data, size_t n,
                           char *error, size_t error_capacity);
int mynah_backend_layer_norm_dev(const mynah_backend *backend,
                                 const float *dev_in, float *dev_out,
                                 const float *gain, const float *bias,
                                 size_t rows, size_t width,
                                 char *error, size_t error_capacity);
int mynah_backend_softmax_dev(const mynah_backend *backend,
                              float *dev_data, size_t rows, size_t cols,
                              size_t valid,
                              char *error, size_t error_capacity);
int mynah_backend_residual_add_dev(const mynah_backend *backend,
                                   float *dev_out, const float *dev_in,
                                   size_t n,
                                   char *error, size_t error_capacity);
/* out[i] += scale[i] * in[i], used by Mimi's LayerScale decoder transformer. */
int mynah_backend_scaled_residual_add_dev(const mynah_backend *backend,
                                          float *dev_out, const float *dev_in,
                                          const float *scale, size_t n,
                                          char *error, size_t error_capacity);
/* Row-wise LayerScale: out[row][i] += scale[i] * in[row][i]. */
int mynah_backend_scaled_residual_rows_dev(const mynah_backend *backend,
                                           float *dev_out, const float *dev_in,
                                           const float *scale, size_t rows,
                                           size_t width, char *error,
                                           size_t error_capacity);
int mynah_backend_snake_dev(const mynah_backend *backend,
                            float *dev_data, const float *alpha,
                            size_t channels, size_t length,
                            size_t snake_channels,
                            char *error, size_t error_capacity);

/* Device-side single-token attention over resident K/V.  The self variant
 * reads Q/K/V from qkv and appends K/V at position; the cross variant reads a
 * separate Q and resident K/V cache.  Neither function synchronizes. */
int mynah_backend_has_attention_dev(const mynah_backend *backend);
int mynah_backend_self_attention_dev(const mynah_backend *backend,
                                     const float *dev_qkv,
                                     float *dev_k_cache, float *dev_v_cache,
                                     size_t position, size_t cache_stride,
                                     size_t valid, size_t heads,
                                     size_t head_width, float scale,
                                     float *dev_out,
                                     char *error, size_t error_capacity);
/* Batched self-attention for independent requests.  QKV is contiguous as
 * [batch][3][heads][head_width], while each request supplies its own resident
 * K/V cache and absolute position. */
int mynah_backend_self_attention_batch_dev(
    const mynah_backend *backend, const float *dev_qkv,
    float *const *dev_k_cache, float *const *dev_v_cache,
    const size_t *positions, const size_t *cache_strides, size_t batch,
    size_t heads, size_t head_width, float scale, float *dev_out,
    char *error, size_t error_capacity);
/* Gather the newly-written K/V slot of each independent request into one
 * fixed device buffer.  The pointer/position metadata is copied by the
 * backend, so the operation remains graph-capturable while requests rotate
 * through server slots.  `out` is [batch][2][heads * head_width]. */
int mynah_backend_gather_kv_batch(
    const mynah_backend *backend, float *const *dev_k_cache,
    float *const *dev_v_cache, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float *dev_out, char *error, size_t error_capacity);
int mynah_backend_cross_attention_dev(const mynah_backend *backend,
                                      const float *dev_q,
                                      const float *dev_k_cache,
                                      const float *dev_v_cache,
                                      size_t valid, size_t cache_stride,
                                      size_t heads, size_t head_width,
                                      float scale, float *dev_out,
                                      char *error, size_t error_capacity);
/* Apply the model's interleaved RoPE to the Q and K thirds of one resident
 * fused-QKV row.  The operation is separate from attention because the
 * position is request state, not a property of the weight graph. */
int mynah_backend_rope_dev(const mynah_backend *backend,
                           float *dev_qkv, size_t position,
                           size_t heads, size_t head_width, float max_period,
                           char *error, size_t error_capacity);
int mynah_backend_rope_batch_dev(const mynah_backend *backend,
                                 float *dev_qkv, const size_t *positions,
                                 size_t batch, size_t heads,
                                 size_t head_width, float max_period,
                                 char *error, size_t error_capacity);

/* Copy n floats from host to a specific device buffer (no scratch). */
int mynah_backend_h2d(const mynah_backend *backend, const float *host,
                      float *dev_ptr, size_t n,
                      char *error, size_t error_capacity);
/* Copy n floats from a specific device buffer to host (no scratch). */
int mynah_backend_d2h(const mynah_backend *backend, const float *dev_ptr,
                      float *host, size_t n,
                      char *error, size_t error_capacity);

/* Device-side buffer helpers and constrained greedy argmax.  The argmax
 * returns one scalar token because the next autoregressive position depends
 * on it; the logits themselves never leave the device. */
int mynah_backend_copy_dev(const mynah_backend *backend, float *dev_dst,
                           const float *dev_src, size_t n,
                           char *error, size_t error_capacity);
int mynah_backend_scale_dev(const mynah_backend *backend, float *dev_data,
                            size_t n, float scale,
                            char *error, size_t error_capacity);
int mynah_backend_clip_dev(const mynah_backend *backend, float *dev_data,
                           size_t n, char *error, size_t error_capacity);
int mynah_backend_argmax_dev(const mynah_backend *backend, const float *dev_logits,
                             size_t vocab, size_t codebook_size, size_t eos_id,
                             int allow_eos, unsigned *argmax,
                             char *error, size_t error_capacity);

/* Allocate a persistent device buffer of n floats.  On CPU returns a
 * malloc'd host buffer; on CUDA a cudaMalloc'd device buffer.
 * The caller owns the buffer and must free it with mynah_backend_dev_free. */
int mynah_backend_dev_alloc(const mynah_backend *backend, size_t n,
                            float **dev_ptr, char *error, size_t error_capacity);
void mynah_backend_dev_free(const mynah_backend *backend, float *dev_ptr);

/* Allocate pinned host staging when the backend can provide it.  CPU and
 * backends without a pinned allocator use malloc/free; CUDA uses cudaHostAlloc
 * so H2D/D2H transfers in the batch driver are genuinely asynchronous. */
int mynah_backend_host_alloc(const mynah_backend *backend, size_t n,
                             float **host_ptr, char *error,
                             size_t error_capacity);
void mynah_backend_host_free(const mynah_backend *backend, float *host_ptr);

/* Which CPU matmul path a shape takes: "parallel" (output rows split over the
 * pool), "simd" (serial in-tree matvec) or "sgemm" (one sgemm call -- whose
 * provider is a separate question, answered by the sgemm.provider row).
 * `why` (optional) receives a static string naming the clause that decided.
 * Exported so the dispatch report can call the real policy -- a rows=1
 * projection that stops taking the matvec path is otherwise silent. */
const char *mynah_cpu_matvec_mode(size_t rows, size_t input_width,
                                  size_t output_width, const char **why);

#endif
