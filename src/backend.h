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

/* MYNAH_CUDA_QUANT: the one operator-facing weight-precision switch of the
 * resident CUDA path.  f32 (default) keeps the raw f32 GEMMs; bf16 keeps the
 * CPU representation f32 and gives the resident backbone, flow net and Mimi
 * transformer BF16 weight copies on tensor cores; int8 turns on the Q8
 * policy, the int8 qmat cache and the resident-compatible int8 groups.  The
 * low-level variables (MYNAH_CUDA_Q8, MYNAH_QUANT, MYNAH_QUANT_GROUPS) remain
 * expert overrides and win when set.  It never changes a CPU backend.
 * MYNAH_CUDA_QUANT_STAGES (Pocket: backbone, flow, mimi; default all) narrows
 * the stages bf16 applies to, e.g. `backbone` for bf16 FlowLM Linears with an
 * fp32 flow head and codec. */
typedef enum {
    MYNAH_CUDA_QUANT_INVALID = -1,
    MYNAH_CUDA_QUANT_F32 = 0,
    MYNAH_CUDA_QUANT_BF16 = 1,
    MYNAH_CUDA_QUANT_INT8 = 2
} mynah_cuda_quant_mode;
mynah_cuda_quant_mode mynah_cuda_quant_from_env(void);
const char *mynah_cuda_quant_name(mynah_cuda_quant_mode mode);

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
/* Submit one causal SEANet step for independent request decoders as one device
 * batch.  The decoder objects retain their own causal tails/workspaces; only
 * the arithmetic is shared.  Return 0 when submitted, 1 when this optional
 * backend path is unavailable before touching decoder state, and -1 after a
 * validation/launch failure.  CPU/unsupported backends return 1. */
int mynah_backend_decoder_step_batch(
    const mynah_backend *backend, mynah_backend_decoder *const *decoders,
    const float *const *dev_inputs, size_t batch, size_t encoder_frames,
    float *const *dev_outputs, char *error, size_t error_capacity);
/* A decoder step recorded inside a CUDA graph is submitted during capture but
 * does not execute until the graph is launched.  Backends use this hook to
 * keep the process-local step counter logical rather than counting capture
 * submissions.  CPU/unsupported backends treat it as a no-op. */
int mynah_backend_decoder_note_step(const mynah_backend *backend,
                                    mynah_backend_decoder *decoder);
/* Record one successful cross-request decoder arithmetic batch. CPU and other
 * backends treat it as a no-op. Width-one/fallback decoder submissions do not
 * increment this counter. */
int mynah_backend_decoder_note_batch(const mynah_backend *backend,
                                     size_t items, size_t frames);
/* Device bytes a resident decoder owns now (`owned`) and would own without
 * MYNAH_CUDA_ROW_MEM_DIET (`legacy`). -1 when the backend has no such
 * decoder (CPU/Metal). */
int mynah_backend_decoder_device_bytes(const mynah_backend *backend,
                                       const mynah_backend_decoder *decoder,
                                       size_t *owned, size_t *legacy);
/* Record one successful cross-request Pocket backbone batch. CPU/Metal are
 * intentionally no-ops; CUDA exposes the counters for server observability. */
int mynah_backend_note_backbone_batch(const mynah_backend *backend,
                                      size_t items);
/* Record one resident Mimi decoder-transformer tile. */
int mynah_backend_note_codec_transformer_batch(const mynah_backend *backend,
                                               size_t items, size_t width);
/* Record the resident quantizer/causal upsample handoff.  `fallback` is one
 * when a submitted optional step had to be imported back to the CPU oracle. */
void mynah_backend_note_codec_upsample(const mynah_backend *backend,
                                       int fallback);
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
/* Device INT8/Q8 matmul. Activations are quantized per row on device, weights
 * are cached as per-output-row symmetric int8 plus scales, and the f32 result
 * is written to `dout` with the optional bias epilogue. CUDA-only: callers
 * must use it only after the backend capability/policy gate accepts Q8. */
int mynah_backend_matmul_q8_d2d(const mynah_backend *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
/* Reserve the persistent Q8 activation/accumulator workspace before a graph
 * capture. It is unsupported on non-CUDA backends. */
int mynah_backend_q8_reserve(const mynah_backend *, size_t activation_count,
                             size_t rows, size_t output_count,
                             char *, size_t);
/* Resident BF16-weight matmul (CUDA only): the f32 weight is converted once
 * to a cached device BF16 copy, the activation rows are rounded to BF16 on
 * device, and cuBLAS runs a BF16 x BF16 -> FP32 tensor-core GEMM with FP32
 * accumulation.  The reserve call sizes the activation workspace before any
 * graph capture, exactly like the Q8 reserve. */
int mynah_backend_matmul_bf16_d2d(const mynah_backend *, const float *, float *, size_t, size_t, size_t, const float *, const float *, char *, size_t);
int mynah_backend_bf16_reserve(const mynah_backend *, size_t activation_count,
                               char *, size_t);
/* Fused BF16 decode linears (MYNAH_CUDA_BF16_FUSE, CUDA only).  The backend
 * owns one STAGED BF16 activation: the buffer mynah_backend_matmul_bf16_d2d
 * casts its input into, reserved by mynah_backend_bf16_reserve before any
 * graph capture.  The producers below write the RNE BF16 rounding of the
 * same FP32 value the unfused path computes straight into it, and
 * mynah_backend_matmul_bf16_staged_d2d runs the very same cuBLAS call as
 * mynah_backend_matmul_bf16_d2d on it, without the bias epilogue.  The bias
 * is then folded into the elementwise kernel that consumes the GEMM output
 * (RoPE, residual add, GELU), with the same FP32 add.  Stream-ordered: a
 * staged value must be consumed by the next staged GEMM before another
 * producer or a BF16 matmul overwrites it.  `bias` arguments are host
 * model-pack views, cached by the backend like the GEMM bias. */
int mynah_backend_has_bf16_fused(const mynah_backend *);
/* LayerNorm of `rows` x `width` into the staged activation. */
int mynah_backend_layer_norm_bf16_stage_dev(const mynah_backend *,
                                            const float *in, const float *gain,
                                            const float *bias, size_t rows,
                                            size_t width, char *, size_t);
/* GELU(in + bias) of `rows` x `cols` into the staged activation; `in` is not
 * written. */
int mynah_backend_bias_gelu_bf16_stage_dev(const mynah_backend *,
                                           const float *in, const float *bias,
                                           size_t rows, size_t cols, char *,
                                           size_t);
/* out[rows][ow] = staged[rows][iw] x W^T, BF16 weight copy, FP32 output, no
 * bias. */
int mynah_backend_matmul_bf16_staged_d2d(const mynah_backend *, float *out,
                                         size_t rows, size_t iw, size_t ow,
                                         const float *weight, char *, size_t);
/* mynah_backend_rope_batch_dev with the fused-QKV bias added first to all
 * three of q, k and v. */
int mynah_backend_rope_bias_batch_dev(const mynah_backend *, float *dev_qkv,
                                      const float *bias,
                                      const size_t *positions, size_t batch,
                                      size_t heads, size_t head_width,
                                      float max_period, char *, size_t);
/* out[r][c] += in[r][c] + bias[c] (bias may be NULL: a plain residual). */
int mynah_backend_residual_bias_add_dev(const mynah_backend *, float *out,
                                        const float *in, const float *bias,
                                        size_t rows, size_t cols, char *,
                                        size_t);
/* mynah_backend_self_attention_bf16_prefix_batch_dev (prefix tables may be
 * NULL: no shared prefix) whose output goes to the staged activation.
 * `dev_scratch` ([batch][heads*head_width] FP32) is used only when the fast
 * kernel cannot take the call: the legacy kernel writes it and a cast stages
 * it, so the staged values are the same either way. */
int mynah_backend_self_attention_bf16_stage_batch_dev(
    const mynah_backend *, const float *dev_qkv, void *const *dev_k_cache,
    void *const *dev_v_cache, void *const *dev_k_prefix,
    void *const *dev_v_prefix, const size_t *prefix_len,
    const size_t *positions, const size_t *cache_strides, size_t batch,
    size_t heads, size_t head_width, float scale, float *dev_scratch, char *,
    size_t);
int mynah_backend_im2col(const mynah_backend *, const float *, float *, int, int, int, int, char *, size_t);
int mynah_backend_conv1d(const mynah_backend *, const float *, float *, int, int, int, int, int, const float *, const float *, char *, size_t);
/* Device-resident causal conv1d.  `input` and `output` are backend-owned
 * device buffers; weights and bias remain host model-pack views and are
 * cached by the backend.  This is deliberately separate from conv1d(), whose
 * contract is host input/output and may synchronize. */
int mynah_backend_conv1d_dev(const mynah_backend *, const float *, float *, int, int, int, int, int, const float *, const float *, char *, size_t);
int mynah_backend_conv_transpose_dev(const mynah_backend *, const float *, float *, int, int, int, int, int, int, int, const float *, const float *, char *, size_t);
/* One causal depthwise ConvTranspose1d sample.  Input is [channels] for one
 * latent frame; output is row-major [stride][channels].  `partial` is the
 * persistent [channels][kernel-stride] tail and is updated in-place. */
int mynah_backend_conv_transpose_causal_step_dev(
    const mynah_backend *, const float *, float *, float *, int, int, int,
    const float *, const float *, char *, size_t);
/* Device-side handoff from transformer row-major output to the decoder's
 * channel-major [width][length] input. */
int mynah_backend_scatter_row_to_channels_dev(
    const mynah_backend *, const float *, float *, size_t, size_t, size_t,
    char *, size_t);
int mynah_backend_scatter_rows_to_channels_dev(
    const mynah_backend *, const float *, float *const *, size_t, size_t,
    size_t, size_t, char *, size_t);
/* Gather one row from each device pointer into a stacked row-major batch.
 * A NULL input pointer leaves that output row untouched, which lets callers
 * mix resident and host-staged requests without a second kernel. */
int mynah_backend_gather_rows_to_batch_dev(
    const mynah_backend *, float *const *, float *, size_t, size_t, char *,
    size_t);
/* Asynchronously clear a device buffer. */
int mynah_backend_zero_dev(const mynah_backend *, float *, size_t, char *,
                           size_t);
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
/* Resident-only projection encoding for BF16 weights.  Deliberately outside
 * the qmat qtype range (0..4): the CPU oracle never sees it, and a CPU qmat
 * bf16 group (qtype 4, weights bf16 x f32 activations) is a different
 * representation from the device path (bf16 x bf16 on tensor cores). */
#define MYNAH_BACKEND_QTYPE_BF16 16

typedef struct {
    const float *weight;
    const float *bias;
    /* 0 = original f32 view, 1 = CUDA Q8 path,
     * MYNAH_BACKEND_QTYPE_BF16 = resident BF16-weight tensor-core path.
     * Other encodings are rejected
     * by the resident backend until a matching device kernel exists. */
    int qtype;
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
/* A causal transformer tile over many independent requests in one call.
 *
 * Each of `rows` requests contributes `positions` consecutive tokens starting
 * at its own absolute position `start[r]`; every request keeps a private K/V
 * ring of `ring` slots per layer (slot = absolute % ring, layout [2][ring][dim])
 * and attends to at most the last `context` positions.  Inputs are per-request
 * device rows [positions][dim]; outputs are written channel-major
 * [dim][positions] into per-request device buffers, or skipped when `output`
 * is NULL (a prefill only wants the cache). Rows may carry fewer than
 * `positions` tokens (`count`). All pointer arrays are host arrays of device
 * pointers; `kv` has rows * layers entries, request-major.
 *
 * The projections use a fixed per-element reduction order, so one request's
 * result does not depend on how many others share the call. Nothing here
 * synchronizes. Returns 1 when the backend has no such path. */
/* Host weight pointers of one pre-norm layer, cached on the device by the
 * backend on first use. Biases and layer scales may be NULL. */
typedef struct {
    const float *norm1_weight, *norm1_bias;
    const float *in_proj_weight, *in_proj_bias;   /* [3*dim][dim] */
    const float *out_proj_weight, *out_proj_bias; /* [dim][dim] */
    const float *layer_scale_1;
    const float *norm2_weight, *norm2_bias;
    const float *linear1_weight, *linear1_bias;   /* [ffn][dim] */
    const float *linear2_weight, *linear2_bias;   /* [dim][ffn] */
    const float *layer_scale_2;
} mynah_transformer_tile_layer;
typedef struct {
    size_t rows, positions, dim, heads, ffn, layers;
    size_t context; /* attention window; 0 = the whole prefix           */
    size_t ring;    /* cache slots per layer; slot = absolute % ring      */
    float max_period, layernorm_eps;
    const mynah_transformer_tile_layer *layer; /* [layers], host pointers */
    const float *const *input;  /* [rows] device, [count][dim]            */
    float *const *output;       /* [rows] device, [dim][positions]; NULL  */
    void *const *kv;            /* [rows * layers] device, f32 or bf16    */
    const size_t *start;        /* [rows] absolute position of token 0    */
    const size_t *count;        /* [rows] tokens per row; NULL = positions */
    const size_t *rings;        /* [rows] slots per row; NULL = `ring`    */
    int kv_bf16;                /* the caches hold BF16 instead of f32    */
    /* 1: the tile GEMMs read resident BF16 copies of the projection weights
     * (same deterministic fp32-accumulating kernel, bf16-rounded weights). */
    int weight_bf16;
    /* Use only fixed-order kernels so the result does not depend on how the
     * rows were split across calls (a prefill pushed in pieces must equal one
     * pushed whole). Costs tensor-core speed; the prefill can afford it. */
    int fixed_order;
    /* Rows whose cache does not store a leading prefix (MYNAH_CUDA_SHARED_VOICE
     * with the voice prefix dropped from the row). NULL = every row stores
     * every position, exactly the layout described above. Otherwise row r
     * reads positions [0, skip[r]) from the shared planes
     * prefix[r * layers + l] (layout [K skip[r]][V skip[r]] x dim, same element
     * type as the cache, read only) and stores absolute position a >= skip[r]
     * in its own cache at slot (a - skip[r]) % ring_r, where ring_r
     * (`rings`/`ring`) then counts the STORED slots per plane. Every start[r]
     * must be >= skip[r]: nothing is ever written below the skip. A row with
     * skip[r] == 0 is the plain layout and its prefix entries are ignored. */
    const size_t *skip;         /* [rows] or NULL                         */
    void *const *prefix;        /* [rows * layers] device, or NULL        */
    /* MYNAH_CUDA_KV_VMM (position-major rows): per row the pair
     * (pitch, voff) in elements: slot s of layer l's K is at kv[r * layers +
     * l] + s * pitch and of its V at kv[...] + voff + s * pitch. NULL = the
     * plain layout above, (dim, ring_r * dim) for every row. CUDA only. */
    const size_t *kv_strides;   /* [rows * 2] or NULL                     */
} mynah_backend_tile_desc;
/* 1 when a row's result cannot depend on the other rows of a batched call
 * (every kernel reduces in a fixed order). 0 when the backend runs cuBLAS
 * algorithm selection or tensor-core (TF32/FP16) GEMMs, whose rounding and
 * blocking follow M: then solo and gang outputs agree only to a tolerance. */
int mynah_backend_batch_invariant(const mynah_backend *backend);
/* Quantizer projection + causal depthwise upsample for several independent
 * requests in one submission.  Row r reads the host vector `host_input[r]`
 * ([in_dim], copied into backend-owned pinned staging before the call
 * returns), projects it with `proj_weight` ([channels][in_dim]) and advances
 * its own device tail `partial[r]` ([channels][kernel-stride], may be NULL
 * only when kernel == stride), writing row-major [stride][channels] to
 * `output[r]`.  Every output element uses exactly the reduction order of
 * mynah_backend_matvec_dev + mynah_backend_conv_transpose_causal_step_dev,
 * so a row's result does not depend on who else is in the call.  One H2D,
 * two kernels, no synchronisation.  Returns 1 when the backend has no such
 * path (nothing was queued), -1 on a queueing failure. */
typedef struct {
    size_t rows, in_dim, channels, kernel, stride;
    const float *const *host_input;
    float *const *output;
    float *const *partial;
    const float *proj_weight, *proj_bias; /* host model-pack views */
    const float *up_weight, *up_bias;
} mynah_backend_upsample_batch_desc;
int mynah_backend_codec_upsample_batch_dev(
    const mynah_backend *backend, const mynah_backend_upsample_batch_desc *desc,
    char *error, size_t error_capacity);
/* Queue one gather of `rows` device vectors of `width` floats into a single
 * backend-owned pinned host block ([rows][width]) with one D2H.  `*host_out`
 * is readable only after the next mynah_backend_sync() and stays valid until
 * the next call.  Returns 1 when unsupported (nothing queued). */
int mynah_backend_gather_rows_d2h(const mynah_backend *backend,
                                  const float *const *dev_rows, size_t rows,
                                  size_t width, const float **host_out,
                                  char *error, size_t error_capacity);
/* Record one cross-request codec gang call: stage 0 is the quantizer +
 * upsample submission, stage 1 the batched PCM collect.  No-op off CUDA. */
void mynah_backend_note_codec_gang(const mynah_backend *backend, int stage,
                                   size_t rows);
int mynah_backend_has_tile_transformer(const mynah_backend *backend);
int mynah_backend_tile_transformer_dev(const mynah_backend *backend,
                                       const mynah_backend_tile_desc *desc,
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
/* BF16-cache variants. QKV, output and attention accumulation remain FP32;
 * only persistent K/V elements use two-byte bfloat16 storage. */
int mynah_backend_self_attention_bf16_dev(
    const mynah_backend *backend, const float *dev_qkv,
    void *dev_k_cache, void *dev_v_cache,
    size_t position, size_t cache_stride, size_t valid, size_t heads,
    size_t head_width, float scale, float *dev_out,
    char *error, size_t error_capacity);
int mynah_backend_self_attention_bf16_batch_dev(
    const mynah_backend *backend, const float *dev_qkv,
    void *const *dev_k_cache, void *const *dev_v_cache,
    const size_t *positions, const size_t *cache_strides, size_t batch,
    size_t heads, size_t head_width, float scale, float *dev_out,
    char *error, size_t error_capacity);
/* mynah_backend_self_attention_bf16_dev with positions [0, prefix_len) read
 * from the shared voice-prefix planes dev_k_prefix / dev_v_prefix (stride
 * heads * head_width) instead of the cache (MYNAH_CUDA_SHARED_VOICE). Same
 * kernel and reduction order as the plain call; the cache is never read or
 * written below prefix_len, so it may be a pointer biased below an
 * allocation that does not store the prefix. Requires prefix_len <= position. */
int mynah_backend_has_self_attention_bf16_prefix(const mynah_backend *backend);
int mynah_backend_self_attention_bf16_prefix_dev(
    const mynah_backend *backend, const float *dev_qkv,
    void *dev_k_cache, void *dev_v_cache, const void *dev_k_prefix,
    const void *dev_v_prefix, size_t prefix_len, size_t position,
    size_t cache_stride, size_t valid, size_t heads, size_t head_width,
    float scale, float *dev_out, char *error, size_t error_capacity);
/* As above, with positions [0, prefix_len[i]) of row i read from the shared
 * voice-prefix planes dev_k_prefix[i] / dev_v_prefix[i] (stride heads *
 * head_width) instead of the row's own cache (MYNAH_CUDA_SHARED_VOICE). A row
 * with prefix_len 0 reads only its cache. Whatever kernel the backend picks,
 * positions below prefix_len[i] of the row's own cache are never read or
 * written (the new position is always >= prefix_len[i]), so a row whose cache
 * does not store the prefix at all may pass cache pointers biased below its
 * allocation by prefix_len[i] positions. */
int mynah_backend_has_self_attention_bf16_prefix_batch(const mynah_backend *backend);
int mynah_backend_self_attention_bf16_prefix_batch_dev(
    const mynah_backend *backend, const float *dev_qkv,
    void *const *dev_k_cache, void *const *dev_v_cache,
    void *const *dev_k_prefix, void *const *dev_v_prefix,
    const size_t *prefix_len, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float scale, float *dev_out, char *error,
    size_t error_capacity);
/* Gather the newly-written K/V slot of each independent request into one
 * fixed device buffer.  The pointer/position metadata is copied by the
 * backend, so the operation remains graph-capturable while requests rotate
 * through server slots.  `out` is [batch][2][heads * head_width]. */
int mynah_backend_gather_kv_batch(
    const mynah_backend *backend, float *const *dev_k_cache,
    float *const *dev_v_cache, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float *dev_out, char *error, size_t error_capacity);
int mynah_backend_gather_kv_bf16_batch(
    const mynah_backend *backend, void *const *dev_k_cache,
    void *const *dev_v_cache, const size_t *positions,
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
int mynah_backend_h2d_bf16(const mynah_backend *backend, const float *host,
                           void *dev_ptr, size_t n,
                           char *error, size_t error_capacity);
int mynah_backend_d2h_bf16(const mynah_backend *backend, const void *dev_ptr,
                           float *host, size_t n,
                           char *error, size_t error_capacity);

/* Device-side buffer helpers and constrained greedy argmax.  The argmax
 * returns one scalar token because the next autoregressive position depends
 * on it; the logits themselves never leave the device. */
int mynah_backend_copy_dev(const mynah_backend *backend, float *dev_dst,
                           const float *dev_src, size_t n,
                           char *error, size_t error_capacity);
/* Raw device-to-device bytes, for caches whose element type the caller owns
 * (BF16 K/V). Asynchronous on the backend stream; returns 1 when the backend
 * has no device memory of its own. */
int mynah_backend_copy_dev_bytes(const mynah_backend *backend, void *dev_dst,
                                 const void *dev_src, size_t bytes,
                                 char *error, size_t error_capacity);
/* Strided device-to-device bytes: `rows` rows of `width` bytes, row r from
 * dev_src + r * src_pitch to dev_dst + r * dst_pitch. Asynchronous; returns
 * 1 when the backend has no such copy (nothing queued). */
int mynah_backend_copy_dev_bytes_2d(const mynah_backend *backend,
                                    void *dev_dst, size_t dst_pitch,
                                    const void *dev_src, size_t src_pitch,
                                    size_t width, size_t rows, char *error,
                                    size_t error_capacity);
/* MYNAH_CUDA_KV_VMM: growable device buffers on the CUDA virtual memory
 * management API (CUDA only).  A buffer is a virtual reservation of
 * `reserve` bytes whose first `mapped` bytes are backed by device memory; it
 * grows by mapping more pages after the mapped ones, so its address never
 * changes and nothing is copied.  Sizes are rounded up to the allocation
 * granularity.  `probe` returns 0 and the granularity when the path is
 * usable, -1 and the reason otherwise (no driver entry points, device
 * without VMM support).  `resize` maps up to `want` bytes, or unmaps whole
 * trailing chunks while the rest still covers `want`; it must not run inside
 * a stream capture, and shrinking (like `free`) requires that no queued work
 * still uses the pages.  mynah_backend_dev_free also releases such a buffer
 * correctly. */
int mynah_backend_kv_vmm_probe(const mynah_backend *backend,
                               size_t *granularity, char *error,
                               size_t error_capacity);
int mynah_backend_kv_vmm_alloc(const mynah_backend *backend, size_t reserve,
                               size_t map, void **dev_ptr, size_t *mapped,
                               char *error, size_t error_capacity);
int mynah_backend_kv_vmm_resize(const mynah_backend *backend, void *dev_ptr,
                                size_t want, size_t *mapped, char *error,
                                size_t error_capacity);
void mynah_backend_kv_vmm_free(const mynah_backend *backend, void *dev_ptr);
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
int mynah_backend_dev_alloc_bytes(const mynah_backend *backend, size_t bytes,
                                  void **dev_ptr, char *error,
                                  size_t error_capacity);
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
