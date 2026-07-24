#ifndef MYNAH_TTS_KERNELS_H
#define MYNAH_TTS_KERNELS_H

#include <stddef.h>

float mynah_dot_f32(const float *a, const float *b, size_t n);
void mynah_matvec_f32(const float *weights, const float *input, float *output,
                      size_t rows, size_t cols);
void mynah_rmsnorm_f32(const float *input, const float *weight, float *output,
                       size_t n, float epsilon);
int mynah_softmax_f32(const float *logits, float *probabilities, size_t n);
size_t mynah_argmax_f32(const float *values, size_t n);
int mynah_kernels_self_test(char *error, size_t error_capacity);

#endif
