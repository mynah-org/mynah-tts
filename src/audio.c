#include "audio.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int write_u16_le(FILE *file, uint16_t value) {
    const unsigned char bytes[2] = {(unsigned char)(value & 0xffu),
                                    (unsigned char)(value >> 8)};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int write_u32_le(FILE *file, uint32_t value) {
    const unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
        (unsigned char)((value >> 16) & 0xffu),
        (unsigned char)(value >> 24),
    };
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

int mynah_wav_write_f32(const char *path, const float *samples, size_t count,
                        unsigned sample_rate, char *error,
                        size_t error_capacity) {
    if (path == NULL || samples == NULL || sample_rate == 0 || count > UINT32_MAX / 2u) {
        snprintf(error, error_capacity, "invalid WAV arguments");
        return -1;
    }
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        snprintf(error, error_capacity, "cannot open WAV output: %s", path);
        return -1;
    }

    const uint32_t data_bytes = (uint32_t)(count * 2u);
    const uint32_t riff_bytes = 36u + data_bytes;
    int failed = fwrite("RIFF", 1, 4, file) != 4 ||
                 write_u32_le(file, riff_bytes) != 0 ||
                 fwrite("WAVEfmt ", 1, 8, file) != 8 ||
                 write_u32_le(file, 16u) != 0 ||
                 write_u16_le(file, 1u) != 0 ||
                 write_u16_le(file, 1u) != 0 ||
                 write_u32_le(file, sample_rate) != 0 ||
                 write_u32_le(file, sample_rate * 2u) != 0 ||
                 write_u16_le(file, 2u) != 0 ||
                 write_u16_le(file, 16u) != 0 ||
                 fwrite("data", 1, 4, file) != 4 ||
                 write_u32_le(file, data_bytes) != 0;

    for (size_t i = 0; !failed && i < count; ++i) {
        float sample = samples[i];
        if (!isfinite(sample)) {
            sample = 0.0f;
        }
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        const int32_t quantized = (int32_t)lrintf(sample * 32767.0f);
        failed = write_u16_le(file, (uint16_t)(int16_t)quantized) != 0;
    }

    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        snprintf(error, error_capacity, "failed while writing WAV: %s", path);
        return -1;
    }
    error[0] = '\0';
    return 0;
}
