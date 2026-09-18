/* NanoCodec decoder: FSQ dequantization, snake activations, residual stack.
 *
 * Moved out of graph.c by the E1 split. This is Magpie's audio decoder and
 * nothing else: the FSQ levels, the 864-channel pre-conv and the five
 * upsampling rates are properties of that codec, not of the runtime. PocketTTS
 * has its own decoder (a causal SEANet over continuous latents) and shares only
 * the convolution primitives in conv1d.h.
 *
 * It still takes the whole model because the resident-device path needs the
 * backend and the codec cache together; narrowing that is part of the engine
 * seam, not of moving the file.
 */
#ifndef MYNAH_TTS_CODEC_NANOCODEC_H
#define MYNAH_TTS_CODEC_NANOCODEC_H

#include <stddef.h>

#include "mynah_tts_internal.h"

/* Decode `raw_length` frames of stacked codebook indices to PCM.
 * `*samples` is malloc'd and owned by the caller. */
int mynah_nanocodec_decode(const mynah_tts_model *model, const unsigned *codes,
                           size_t raw_length, float **samples, size_t *sample_count,
                           char *error, size_t error_capacity);

/* 1 when the SEANet Snake activation runs the vDSP/vForce vector form.
 * 0 means MYNAH_SNAKE_SCALAR forced the scalar rollback, or this build has no
 * vector Snake compiled at all. */
int mynah_snake_vector_enabled(void);

#endif
