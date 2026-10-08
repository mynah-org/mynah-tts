/* The serving loop's host team (src/hostpool.h): a region must give the
 * serial loop's result, whatever the team size, the row count, the chunking
 * and the wake-up timing of its workers -- and must not touch its argument
 * once it has returned (the argument lives on the caller's stack).
 *
 * Built with -fsanitize=thread this is also the race check of the team. */
#include "hostpool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const double *in;
    double *out;
    unsigned *visits;
} job;

static void body(void *ud, size_t begin, size_t end) {
    job *j = (job *)ud;
    for (size_t i = begin; i < end; ++i) {
        double x = j->in[i];
        for (int k = 0; k < 40; ++k) x = sin(x) + 0.5 * x;
        j->out[i] = x;
        j->visits[i] += 1u;
    }
}

int main(void) {
    if (getenv("MYNAH_SERVE_HOST_THREADS") == NULL)
        setenv("MYNAH_SERVE_HOST_THREADS", "4", 1);
    setenv("MYNAH_SERVE_HOST_MIN_ROWS", "2", 1);
    setenv("MYNAH_SERVE_HOST_SPIN_US", "20", 1);
    const int threads = mynah_hostpool_threads();
    int failures = 0;
    enum { MAX_ROWS = 1100 };
    for (int it = 0; it < 4000; ++it) {
        const size_t n = 1u + (size_t)((unsigned)it * 37u % MAX_ROWS);
        double in[MAX_ROWS], out[MAX_ROWS], ref[MAX_ROWS];
        unsigned visits[MAX_ROWS], ref_visits[MAX_ROWS];
        for (size_t i = 0; i < n; ++i) in[i] = (double)(i * 3u + (size_t)it);
        memset(out, 0, sizeof(out));
        memset(visits, 0, sizeof(visits));
        memset(ref_visits, 0, sizeof(ref_visits));
        job j = {in, out, visits};
        job r = {in, ref, ref_visits};
        mynah_hostpool_run(n, body, &j);
        body(&r, 0u, n);
        if (memcmp(out, ref, n * sizeof(double)) != 0 ||
            memcmp(visits, ref_visits, n * sizeof(unsigned)) != 0) {
            if (failures++ < 5)
                fprintf(stderr, "FAIL [hostpool]: %zu rows differ from the serial loop\n", n);
        }
        if (it % 500 == 0) {
            /* Let the workers park, so the next region also covers wake-up. */
            volatile unsigned spin = 0;
            for (unsigned q = 0; q < 2000000u; ++q) spin += q;
        }
    }
    unsigned long long parallel = 0, inline_runs = 0;
    mynah_hostpool_stats(&parallel, &inline_runs);
    if (threads > 1 && parallel == 0ull) {
        fprintf(stderr, "FAIL [hostpool]: no region ran in parallel\n");
        ++failures;
    }
    if (failures != 0) return 1;
    printf("hostpool: %d threads, %llu regions in parallel, %llu inline, all "
           "bit-identical to the serial loop\n", threads, parallel, inline_runs);
    return 0;
}
