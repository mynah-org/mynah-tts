#ifndef MYNAH_TTS_AUDIO_H
#define MYNAH_TTS_AUDIO_H

#include <stddef.h>

int mynah_wav_write_f32(const char *path, const float *samples, size_t count,
                        unsigned sample_rate, char *error,
                        size_t error_capacity);

#endif
