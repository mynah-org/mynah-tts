/* Causal 1-D convolutions and the filter cache they share.
 *
 * Moved out of graph.c by the E1 split. These are not codec-specific: the
 * Magpie conv-FFN, the NanoCodec decoder and the PocketTTS SEANet stack all
 * reduce to the same three primitives — unfold a causal window, apply a causal
 * conv1d, apply a causal transposed conv. Keeping them here is what lets a
 * second engine reuse them instead of copying them.
 *
 * The BNNS filter cache is keyed by (shape, owning thread) and is deliberately
 * not shared across threads: BNNSFilterApply's behaviour with one filter on two
 * threads is undocumented, and a per-thread filter lets the apply run outside
 * the cache lock. Do not "simplify" that — with a shared filter the whole
 * convolution had to sit inside the critical section and every codec call in
 * the process serialized.
 */
#ifndef MYNAH_TTS_CONV1D_H
#define MYNAH_TTS_CONV1D_H

#include <stddef.h>

#include "backend.h"
#include "weights.h"

typedef struct codec_bnns_cache codec_bnns_cache;

/* Optional per-call timing, filled only when MYNAH_TIMING is set. */
typedef struct {
    double pack_seconds;
    double gemm_seconds;
    double transpose_seconds;
    double snake_seconds;
    double bnns_create_seconds;
    double bnns_apply_seconds;
    double bnns_destroy_seconds;
    size_t calls;
    size_t transpose_calls;
    size_t snake_calls;
} codec_conv_profile;

/* col[t][i * kernel + k] is channel i of the input at t - (kernel - 1) + k,
 * zero before the start. The (i, k) ordering matches the [out][in][kernel]
 * weight layout, so the unfolded activation multiplies the stored weight
 * directly. */
void mynah_unfold_causal(const float *input, float *col, size_t length,
                         size_t channels, size_t kernel);

int mynah_conv1d_causal(const mynah_weights *file, const mynah_backend *backend,
                        codec_bnns_cache *bnns_cache,
                        const char *weight_name, const char *bias_name,
                        const float *input, float *output,
                        size_t in_channels, size_t out_channels, size_t length,
                        size_t kernel, size_t dilation,
                        float *columns_workspace, size_t columns_capacity,
                        codec_conv_profile *profile,
                        char *error, size_t error_capacity);

int mynah_conv_transpose_causal(const mynah_weights *file, const char *weight_name,
                                const char *bias_name, const float *input,
                                float *output,
                                size_t in_channels, size_t out_channels, size_t length,
                                size_t kernel, size_t stride, size_t groups,
                                codec_conv_profile *profile,
                                char *error, size_t error_capacity);

/* The cache is per-model and mutable; these keep the names the public model
 * lifecycle already uses. */
void *mynah_graph_codec_cache_new(void);
void mynah_graph_codec_cache_free(void *cache);

int mynah_graph_bnns_self_test(char *error, size_t error_capacity);

/* 1 when the causal conv1d runs im2col + sgemm rather than BNNS: either
 * MYNAH_CODEC_SGEMM forced it, or this build has no BNNS at all.  Read by
 * src/codec_nanocodec.c too, which must size an im2col workspace to match. */
int mynah_conv1d_sgemm_enabled(void);

#endif
