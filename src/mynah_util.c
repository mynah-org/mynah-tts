#include "mynah_util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void mynah_graph_error(char *error, size_t capacity, const char *message) {
    if (error != NULL && capacity > 0) snprintf(error, capacity, "%s", message);
}

int mynah_graph_sgemm(const mynah_backend *backend,
                      int trans_a, int trans_b,
                      size_t m, size_t n, size_t k,
                      float alpha,
                      const float *a, size_t lda,
                      const float *b, size_t ldb,
                      float beta,
                      float *c, size_t ldc,
                      char *error, size_t error_capacity) {
    return mynah_backend_sgemm(backend, trans_a, trans_b, m, n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc,
                               error, error_capacity);
}

double mynah_phase_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

int mynah_tensor_get(const mynah_weights *file, const char *name,
                     mynah_tensor *out, char *error, size_t error_capacity) {
    if (mynah_weights_get(file, name, out) != 0) {
        snprintf(error, error_capacity, "model tensor is missing: %s", name);
        return -1;
    }
    return 0;
}

float *mynah_alloc_floats(size_t count, char *error, size_t error_capacity) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        mynah_graph_error(error, error_capacity, "invalid graph allocation size");
        return NULL;
    }
    float *value = (float *)calloc(count, sizeof(*value));
    if (value == NULL) mynah_graph_error(error, error_capacity, "out of memory in native graph");
    return value;
}
