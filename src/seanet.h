/*
 * Causal SEANet decoder and the Mimi up/downsample (E3-4).
 *
 * Everything here is streaming-first: the convolutions never re-run left
 * context, they carry it.  E2-3 measured both designs against the reference
 * (.work/pocket-tts-oracle.md): carrying state is exact down to a single 80 ms
 * frame, while replaying context needs 64 frames (5.12 s) per chunk to become
 * bit-identical.  So there is no "left context frames" knob in this API on
 * purpose — feed contiguous, monotonically increasing frames and the state
 * does the rest.
 *
 * THE STATE HAS TWO PARTS AND BOTH ARE MANDATORY:
 *
 *   1. the convolution ring buffers (the obvious one), and
 *   2. a position counter that advances by `encoder_stride` per *latent*
 *      frame.
 *
 * The second one is the trap.  It is not used by any convolution here; it is
 * the offset the Mimi decoder transformer needs for its KV cache, and it is
 * exactly what the oracle harness forgot.  Forgetting it does not crash and
 * does not warn: the audio simply degrades, smoothly and more the smaller the
 * chunk, which reads like a kernel bug.  `mynah_seanet_state_position` is the
 * only way to get it and `mynah_seanet_state_advance` is the only way to move
 * it, so a caller that ignores it is visibly ignoring it.
 *
 * Per-frame order for one decode step:
 *
 *     offset = mynah_seanet_state_position(state);     // encoder frames
 *     mynah_seanet_upsample(state, ...)                // 12.5 Hz -> 200 Hz
 *     <decoder transformer at `offset`, owned by the caller>
 *     mynah_seanet_decode(state, ...)                  // -> waveform
 *     mynah_seanet_state_advance(state, n_latent_frames);
 *
 * Layout convention everywhere: channel major, `[channels][length]`, which is
 * torch `[1, C, T]` with batch 1.
 *
 * No dimension is baked in; the caller supplies them in the config structs and
 * hands over already-resolved weight pointers.  This module never formats a
 * tensor name.
 */
#ifndef MYNAH_TTS_SEANET_H
#define MYNAH_TTS_SEANET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------- weights */

/* Conv1d:          weight is [out_channels][in_channels / groups][kernel]
 * ConvTranspose1d: weight is [in_channels][out_channels / groups][kernel]
 * exactly as PyTorch stores them; `bias` is [out_channels] or NULL. */
typedef struct {
    const float *weight;
    const float *bias;
} mynah_conv_weights;

/* ------------------------------------------------ causal streaming conv1d */

typedef enum {
    MYNAH_CONV_PAD_ZERO = 0,     /* pad_mode="constant" */
    MYNAH_CONV_PAD_REPLICATE = 1 /* pad_mode="replicate" */
} mynah_conv_pad_mode;

typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t dilation;
    size_t groups;
    mynah_conv_pad_mode pad_mode;
} mynah_conv1d_spec;

typedef struct {
    mynah_conv1d_spec spec;
    size_t tail;        /* (kernel-1)*dilation + 1 - stride  */
    size_t max_in_len;
    float *previous;    /* [in_channels][tail], borrowed     */
    float *window;      /* [in_channels][tail + max_in_len]  */
    int primed;         /* replicate padding: first call seen */
    /* GEMM fast path (added later, see the "one tap at a time" note in
     * seanet.c).  `taps` holds one kernel tap of the weight gathered into a
     * dense [out_channels][in_channels] matrix; it is NULL, and the fast path
     * off, whenever the shape or the build does not qualify.  The scalar loop
     * below stays the reference and is what runs then. */
    float *taps;        /* [out_channels][in_channels] or NULL */
} mynah_causal_conv1d;

/* Number of floats the caller must provide to `_init`. */
size_t mynah_causal_conv1d_scratch(const mynah_conv1d_spec *spec,
                                   size_t max_in_len);
int mynah_causal_conv1d_init(mynah_causal_conv1d *conv,
                             const mynah_conv1d_spec *spec, size_t max_in_len,
                             float *scratch, size_t scratch_floats, char *error,
                             size_t error_capacity);
void mynah_causal_conv1d_reset(mynah_causal_conv1d *conv);
/* in_len must be a positive multiple of stride and at most max_in_len. */
int mynah_causal_conv1d_apply(mynah_causal_conv1d *conv,
                              const mynah_conv_weights *weights,
                              const float *input, size_t in_len, float *output);

/* --------------------------------------- causal streaming convtranspose1d */

typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t groups;
} mynah_convtr1d_spec;

typedef struct {
    mynah_convtr1d_spec spec;
    size_t tail;        /* kernel - stride */
    size_t max_in_len;
    float *partial;     /* [out_channels][tail], borrowed */
    float *full;        /* [out_channels][(max_in_len-1)*stride + kernel] */
    /* GEMM fast path: the un-scattered product [out_channels*kernel][in_len].
     * NULL when the shape or the build does not qualify; the scalar loop then
     * runs unchanged. */
    float *taps;        /* [out_channels * kernel][max_in_len] or NULL */
} mynah_causal_convtr1d;

size_t mynah_causal_convtr1d_scratch(const mynah_convtr1d_spec *spec,
                                     size_t max_in_len);
int mynah_causal_convtr1d_init(mynah_causal_convtr1d *convtr,
                               const mynah_convtr1d_spec *spec,
                               size_t max_in_len, float *scratch,
                               size_t scratch_floats, char *error,
                               size_t error_capacity);
void mynah_causal_convtr1d_reset(mynah_causal_convtr1d *convtr);
/* Writes in_len * stride samples per output channel. */
int mynah_causal_convtr1d_apply(mynah_causal_convtr1d *convtr,
                                const mynah_conv_weights *weights,
                                const float *input, size_t in_len,
                                float *output);

/* ----------------------------------------------------------- SEANet model */

typedef struct {
    size_t channels;             /* waveform channels, 1                 */
    size_t dimension;            /* latent width into the decoder, 512   */
    size_t n_filters;            /* 64                                   */
    size_t n_residual_layers;    /* 1                                    */
    const size_t *ratios;        /* decoder order, e.g. {6, 5, 4}        */
    size_t n_ratios;             /* 3                                    */
    size_t kernel_size;          /* 7                                    */
    size_t residual_kernel_size; /* 3                                    */
    size_t last_kernel_size;     /* 3                                    */
    size_t dilation_base;        /* 2                                    */
    size_t compress;             /* 2                                    */
    float elu_alpha;             /* 1.0f                                 */
} mynah_seanet_config;

/* Mimi's ConvTrUpsample1d / ConvDownsample1d.  kernel_size is always
 * 2 * stride.  The upsample is depthwise (groups == channels), which is why
 * its weight is [512, 1, 32] and not [512, 512, 32]: implementing it densely
 * would be both wrong and 512x the work. */
typedef struct {
    size_t stride;       /* 16 = encoder_frame_rate / frame_rate */
    size_t in_channels;
    size_t out_channels;
    size_t groups;       /* upsample: == out_channels; downsample: 1 */
} mynah_resample_config;

typedef struct {
    mynah_conv_weights conv1; /* [dim/compress][dim][residual_kernel_size] */
    mynah_conv_weights conv2; /* [dim][dim/compress][1]                    */
} mynah_seanet_resblock_weights;

typedef struct {
    mynah_conv_weights first; /* [mult*n_filters][dimension][kernel_size]  */
    const mynah_conv_weights *convtr;            /* [n_ratios]             */
    const mynah_seanet_resblock_weights *blocks; /* [n_ratios * n_residual_layers], stage major */
    mynah_conv_weights last;  /* [channels][n_filters][last_kernel_size]   */
} mynah_seanet_decoder_weights;

typedef struct mynah_seanet_state mynah_seanet_state;

/*
 * One codec context: the upsample state, every decoder convolution's ring
 * buffer, the scratch activations, and the position counter.  `up` may be NULL
 * if the caller only ever runs the decoder, in which case the position counter
 * still exists but the encoder stride defaults to prod(ratios) == 1 frame.
 *
 * `max_latent_frames` bounds one call; it is what every scratch buffer is
 * sized from, so nothing allocates inside the decode loop.
 */
mynah_seanet_state *mynah_seanet_state_create(const mynah_seanet_config *config,
                                              const mynah_resample_config *up,
                                              size_t max_latent_frames,
                                              char *error,
                                              size_t error_capacity);
void mynah_seanet_state_destroy(mynah_seanet_state *state);

/* Clears the ring buffers *and* the position counter. */
void mynah_seanet_state_reset(mynah_seanet_state *state);

/* Position in encoder frames.  Feed this to the decoder transformer. */
size_t mynah_seanet_state_position(const mynah_seanet_state *state);
/* Advances by n_latent_frames * encoder_stride.  Call once per decode step,
 * after decoding, exactly like upstream's increment_steps(). */
void mynah_seanet_state_advance(mynah_seanet_state *state,
                                size_t n_latent_frames);

size_t mynah_seanet_state_encoder_stride(const mynah_seanet_state *state);
/* prod(ratios): waveform samples produced per encoder frame. */
size_t mynah_seanet_state_hop_length(const mynah_seanet_state *state);
/* encoder_stride * hop_length: waveform samples per latent frame. */
size_t mynah_seanet_state_samples_per_latent(const mynah_seanet_state *state);
size_t mynah_seanet_state_max_latent_frames(const mynah_seanet_state *state);

int mynah_seanet_check_decoder_weights(const mynah_seanet_state *state,
                                       const mynah_seanet_decoder_weights *w,
                                       char *error, size_t error_capacity);

/* input  [in_channels][n_latent_frames]
 * output [out_channels][n_latent_frames * stride] */
int mynah_seanet_upsample(mynah_seanet_state *state,
                          const mynah_conv_weights *weights,
                          const float *input, size_t n_latent_frames,
                          float *output);

/* input  [dimension][n_encoder_frames]
 * output [channels][n_encoder_frames * hop_length] */
int mynah_seanet_decode(mynah_seanet_state *state,
                        const mynah_seanet_decoder_weights *weights,
                        const float *input, size_t n_encoder_frames,
                        float *output);

/* --------------------------------------------------- encoder-side helper */

/* ConvDownsample1d: stride S, kernel 2S, groups 1, no bias, and pad_mode
 * "replicate" — not zero padding.  Cloning-only in PocketTTS, but it is the
 * mirror of the upsample and shares the streaming contract. */
typedef struct mynah_seanet_downsample mynah_seanet_downsample;

mynah_seanet_downsample *mynah_seanet_downsample_create(
    const mynah_resample_config *config, size_t max_in_len, char *error,
    size_t error_capacity);
void mynah_seanet_downsample_destroy(mynah_seanet_downsample *down);
void mynah_seanet_downsample_reset(mynah_seanet_downsample *down);
int mynah_seanet_downsample_apply(mynah_seanet_downsample *down,
                                  const mynah_conv_weights *weights,
                                  const float *input, size_t in_len,
                                  float *output);

/* ------------------------------------------------------------- utilities */

/* y = x if x > 0 else alpha * (exp(x) - 1); in place when output == input. */
void mynah_seanet_elu_f32(const float *input, float *output, size_t n,
                          float alpha);

/* Model-free self test: causal conv (padded, dilated, strided, grouped,
 * replicate), depthwise causal transposed conv, ELU, streaming continuity of
 * both primitives and of a whole small SEANet decoder, and the position
 * counter.  Returns 0 on success, -1 with a message in `error`. */
int mynah_seanet_self_test(char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_TTS_SEANET_H */
