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
 * platform rather than inherited from whichever machine it was first read on.
 *
 * AND IT CAN READ BELOW THE CURRENT RSS, which is not a contradiction to
 * explain away in a comment -- it is a number an operator would size a machine
 * from. A prefork worker on an EPYC 9254 reported current 267.4 MB and "peak"
 * 266.5 MB on 2026-09-22. Linux's ru_maxrss is the mm's hiwater_rss, which the
 * kernel refreshes at particular points (unmap, exit, accounting) rather than
 * on every fault, and a freshly forked child inherits the parent's watermark
 * and then grows past it through COW faults without the watermark being
 * revisited. So the answer lags, and under prefork it lags exactly where this
 * number is read.
 *
 * The peak is at least the current value, by definition. Saying so costs one
 * comparison and removes a reading that would under-size a host. */
size_t mynah_rss_peak_bytes(void) {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru;
    size_t peak = 0;
    if (getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
#if defined(__linux__)
        peak = (size_t)ru.ru_maxrss * 1024u;
#else
        peak = (size_t)ru.ru_maxrss;
#endif
    }
    const size_t now = mynah_rss_bytes();
    return peak > now ? peak : now;
#else
    return 0;
#endif
}

/* The SHARED half of the resident set: file-backed pages, which under prefork
 * means the mmap'd model weights that every worker maps and none of them owns.
 *
 * This is the field E12-11 actually needs and the first cut of it did not have.
 * "Poll every worker and sum" -- which is what the /health comment said -- is
 * WRONG under prefork: it counts the shared weight mapping once per worker, so
 * sixteen workers over a 600 MB pack report ~10 GB of a machine that is using
 * well under 2. The incremental cost of one more worker is the PRIVATE part,
 * resident minus shared, and the quantized cache this item exists to measure is
 * heap, so it lands there. 0 where the platform cannot say. */
size_t mynah_rss_shared_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/self/statm", "r");
    if (f == NULL) return 0;
    unsigned long total = 0, resident = 0, shared = 0;
    const int got = fscanf(f, "%lu %lu %lu", &total, &resident, &shared);
    fclose(f);
    if (got != 3) return 0;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return 0;
    return (size_t)shared * (size_t)page;
#else
    /* macOS: task_info's resident_size does not separate shared from private,
     * and the split needs a walk of the VM regions. Not worth it for a
     * development platform: 0 means "this platform did not say". */
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
