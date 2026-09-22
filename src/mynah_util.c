#include "mynah_util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

/* CURRENT resident size.  Linux reads /proc/self/statm, whose second field is
 * resident pages -- one open/read/close of a synthetic file, no allocation, so
 * it is safe to call from a request handler.  macOS has no such file and the
 * answer comes from the Mach task port.  Anywhere else: 0, meaning "this
 * platform did not say", which the caller must render as absent rather than as
 * zero bytes. */
size_t mynah_rss_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/self/statm", "r");
    if (f == NULL) return 0;
    unsigned long total = 0, resident = 0;
    const int got = fscanf(f, "%lu %lu", &total, &resident);
    fclose(f);
    if (got != 2) return 0;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return 0;
    return (size_t)resident * (size_t)page;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (size_t)info.resident_size;
#else
    return 0;
#endif
}

/* PEAK resident size.  getrusage is the portable answer and its unit is not:
 * ru_maxrss is KILOBYTES on Linux and BYTES on macOS/BSD.  Getting that wrong
 * is a factor of 1024 in a capacity plan, so the conversion is explicit per
 * platform rather than inherited from whichever machine it was first read on. */
size_t mynah_rss_peak_bytes(void) {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    if (ru.ru_maxrss <= 0) return 0;
#if defined(__linux__)
    return (size_t)ru.ru_maxrss * 1024u;
#else
    return (size_t)ru.ru_maxrss;
#endif
#else
    return 0;
#endif
}

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
