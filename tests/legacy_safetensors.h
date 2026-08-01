#ifndef MYNAH_SAFETENSORS_H
#define MYNAH_SAFETENSORS_H

#include <stddef.h>

typedef struct mynah_safetensors mynah_safetensors;

typedef struct {
    const float *data;
    size_t rank;
    size_t shape[4];
    size_t count;
} mynah_tensor;

int mynah_safetensors_open(const char *path, mynah_safetensors **out,
                           char *error, size_t error_capacity);
void mynah_safetensors_close(mynah_safetensors *file);
int mynah_safetensors_get(const mynah_safetensors *file, const char *name,
                          mynah_tensor *out);

#endif
