/* Prefork serving topology. The contract and the "how to choose W" procedure
 * are in prefork.h; this file is the mechanism.
 *
 * Three properties are worth naming here because every failure mode of a
 * prefork server is one of them going wrong:
 *
 *   WHEN WE FORK. After the model pack is open, so the mapped weights are one
 *   physical copy behind W address spaces; before any synthesis and before any
 *   other thread in this process exists, so no child can inherit a mutex that
 *   was locked by a thread the fork did not copy. src/threads.c registers a
 *   pthread_atfork child handler that rebuilds the pool, so a child whose pool
 *   was already running in the parent gets a fresh one rather than an empty
 *   one -- without it the first dispatch in a child broadcasts to nobody and
 *   then waits for it, which is a deadlock with no error message.
 *
 *   HOW THE DESCRIPTOR MOVES. The parent accepts and sends the descriptor over
 *   an AF_UNIX socketpair with SCM_RIGHTS, then closes its own copy. There is
 *   exactly one moment where both processes hold it, between sendmsg and
 *   close, and if sendmsg fails the parent still owns it and closes it. A
 *   descriptor is never in flight and unowned.
 *
 *   HOW LOAD IS COUNTED. The parent's only view of a worker is one byte per
 *   finished connection, sent back up the same socketpair. `active[w]` is
 *   dispatched minus finished, the routing decision is "smallest active", and
 *   a worker at `slots_per` is skipped. That makes the accounting the whole
 *   correctness story of the router: a worker path that forgets to report a
 *   finished connection leaks a slot permanently, and one that reports twice
 *   over-admits. See mynah_prefork_conn_done()'s caller in server/main.c --
 *   there is one, on the job's last reference, which is the only place in that
 *   file where a connection is provably finished with.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "prefork.h"

#include "threads.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#endif

#define PREFORK_MAX_WORKERS 256
/* Upper bound on the cpu ids we will enumerate. Sized for a large server, not
 * for CPU_SETSIZE: the list is a plan, and a plan over 1024 cpus is already
 * well past the point where one process should be routing for all of them. */
#define CPU_LIST_MAX 1024

/* Where the cpu topology lives. A macro and not a literal so a test can point
 * it at a fabricated tree: the SMT sibling ordering below is the one piece of
 * Linux-only logic complicated enough to be wrong, and on a machine with no
 * /sys it would otherwise ship having never been executed. */
#ifndef MYNAH_CPU_TOPOLOGY_ROOT
#define MYNAH_CPU_TOPOLOGY_ROOT "/sys/devices/system/cpu"
#endif

/* ------------------------------------------------------------- cpu topology
 *
 * Everything in this section answers one question -- "which cpus may this
 * process use, and which of them are siblings of the same physical core" --
 * and answers it honestly on a platform that cannot tell us. */

/* Fills `out` with the cpu ids this process is allowed to run on and returns
 * how many. Plans on the MASK, not on the machine: sysconf() sees neither an
 * inherited taskset nor a cpuset cgroup, so a pinned or containerised run
 * would otherwise plan for cores it will never get. Returns the online count
 * with an identity list where no mask can be read. */
static int cpus_allowed(int *out, int max) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = 0;
        for (int c = 0; c < CPU_SETSIZE && n < max; ++c) {
            if (CPU_ISSET(c, &set)) out[n++] = c;
        }
        if (n > 0) return n;
    }
#endif
    {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online < 1) online = 1;
        int n = (int)(online < (long)max ? online : (long)max);
        for (int i = 0; i < n; ++i) out[i] = i;
        return n;
    }
}

#if defined(__linux__)
static long sysfs_cpu_long(int cpu, const char *leaf) {
    char path[160];
    snprintf(path, sizeof(path), "%s/cpu%d/topology/%s",
             MYNAH_CPU_TOPOLOGY_ROOT, cpu, leaf);
    FILE *f = fopen(path, "r");
    if (f == NULL) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}
#endif

/* Reorders `cpus` so that the siblings of one physical core sit next to each
 * other, and returns 1 when it really did read the topology.
 *
 * This is the difference between isolation and the appearance of it. With SMT
 * enabled, Linux commonly enumerates the two threads of core 0 as cpu 0 and
 * cpu N/2. A contiguous slice of logical ids then hands worker 0 one thread of
 * each of the first N/2 cores and worker 1 the *other* thread of the same
 * cores: the two workers contend for every one of them while `taskset` shows
 * disjoint masks. Core-major order gives worker 0 cores 0..k with both their
 * threads instead. Identity order (and 0) when sysfs is unreadable, which is
 * every non-Linux platform. */
static int order_core_major(int *cpus, int n) {
#if defined(__linux__)
    int *pkg = (int *)malloc((size_t)n * sizeof(int));
    int *core = (int *)malloc((size_t)n * sizeof(int));
    int *out = (int *)malloc((size_t)n * sizeof(int));
    if (pkg == NULL || core == NULL || out == NULL) {
        free(pkg); free(core); free(out);
        return 0;
    }
    for (int i = 0; i < n; ++i) {
        pkg[i] = (int)sysfs_cpu_long(cpus[i], "physical_package_id");
        core[i] = (int)sysfs_cpu_long(cpus[i], "core_id");
        if (pkg[i] < 0 || core[i] < 0) {
            free(pkg); free(core); free(out);
            return 0;
        }
    }
    int k = 0;
    for (int i = 0; i < n; ++i) {
        int seen = 0;
        for (int j = 0; j < i; ++j) {
            if (pkg[j] == pkg[i] && core[j] == core[i]) { seen = 1; break; }
        }
        if (seen) continue;
        out[k++] = cpus[i];
        for (int j = i + 1; j < n && k < n; ++j) {
            if (pkg[j] == pkg[i] && core[j] == core[i]) out[k++] = cpus[j];
        }
    }
    const int ok = (k == n);
    if (ok) memcpy(cpus, out, (size_t)n * sizeof(int));
    free(pkg); free(core); free(out);
    return ok;
#else
    (void)cpus; (void)n;
    return 0;
#endif
}

/* Distinct physical cores among `cpus`, or 0 when the topology is unreadable
 * (which is not the same as "one core" and must not be printed as such). */
static int count_physical_cores(const int *cpus, int n) {
#if defined(__linux__)
    int cores = 0;
    for (int i = 0; i < n; ++i) {
        const long pi = sysfs_cpu_long(cpus[i], "physical_package_id");
        const long ci = sysfs_cpu_long(cpus[i], "core_id");
        if (pi < 0 || ci < 0) return 0;
        int dup = 0;
        for (int j = 0; j < i; ++j) {
            if (sysfs_cpu_long(cpus[j], "physical_package_id") == pi &&
                sysfs_cpu_long(cpus[j], "core_id") == ci) { dup = 1; break; }
        }
        if (!dup) ++cores;
    }
    return cores;
#else
    (void)cpus; (void)n;
    return 0;
#endif
}

/* Pins this process to `cpus[first .. first+count)`. Returns 1 when the pin
 * really happened, 0 when the platform has no such call -- never a silent
 * no-op, because "pinned" is the load-bearing word in this whole file. */
static int pin_to_slice(const int *cpus, int first, int count,
                        char *desc, size_t desc_size) {
    desc[0] = '\0';
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    size_t used = 0;
    for (int i = first; i < first + count; ++i) {
        CPU_SET(cpus[i], &set);
        if (used + 8u < desc_size) {
            used += (size_t)snprintf(desc + used, desc_size - used, "%s%d",
                                     used > 0 ? "," : "", cpus[i]);
        }
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        snprintf(desc, desc_size, "unpinned (sched_setaffinity: %s)", strerror(errno));
        return 0;
    }
    return 1;
#else
    (void)cpus; (void)first; (void)count;
    snprintf(desc, desc_size, "unpinned (no cpu affinity API on this platform)");
    return 0;
#endif
}

/* --------------------------------------------------------- fd passing
 *
 * SCM_RIGHTS over an AF_UNIX socketpair. Portable POSIX: this is compiled and
 * exercised on macOS as well as Linux, which is why the untested surface of a
 * prefork build is only the two affinity/topology calls above. */

static int send_fd(int chan, int fd) {
    char dummy = 'F';
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } control;
    memset(&control, 0, sizeof(control));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.buf;
    msg.msg_controllen = sizeof(control.buf);

    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    cm->cmsg_level = SOL_SOCKET;
    cm->cmsg_type = SCM_RIGHTS;
    cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &fd, sizeof(int));

    ssize_t n;
    do { n = sendmsg(chan, &msg, 0); } while (n < 0 && errno == EINTR);
    return n == 1 ? 0 : -1;
}

/* The descriptor, -1 when the channel is closed (the parent is gone), -2 when
 * the caller should simply try again. A short or control-less message is a
 * protocol violation and is reported as a closed channel rather than guessed
 * at: there is no sane recovery from a half-read SCM_RIGHTS. */
static int recv_fd(int chan) {
    char dummy = 0;
    struct iovec iov;
    iov.iov_base = &dummy;
    iov.iov_len = 1;

    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } control;
    memset(&control, 0, sizeof(control));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.buf;
    msg.msg_controllen = sizeof(control.buf);

    const ssize_t n = recvmsg(chan, &msg, 0);
    if (n < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return -2;
        return -1;
    }
    if (n == 0) return -1;
    struct cmsghdr *cm = CMSG_FIRSTHDR(&msg);
    if (cm == NULL || cm->cmsg_level != SOL_SOCKET || cm->cmsg_type != SCM_RIGHTS ||
        cm->cmsg_len != CMSG_LEN(sizeof(int))) {
        return -1;
    }
    int fd = -1;
    memcpy(&fd, CMSG_DATA(cm), sizeof(int));
    return fd;
}

/* ------------------------------------------------------------ worker state */

static int g_worker_index = -1;
static int g_worker_threads = 0;
static int g_worker_chan = -1;
static volatile sig_atomic_t g_dump_request = 0;

int mynah_prefork_worker_index(void) { return g_worker_index; }
int mynah_prefork_worker_threads(void) { return g_worker_threads; }

static void on_usr1(int sig) {
    (void)sig;
    g_dump_request = 1;
}

int mynah_prefork_take_dump_request(void) {
    if (g_dump_request == 0) return 0;
    g_dump_request = 0;
    return 1;
}

/* The worker's half of the load accounting. One byte, best effort: if the
 * parent has already gone the write fails and the worker is shutting down
 * anyway. Deliberately not retried on a full pipe either -- the channel is a
 * stream socket with a socket buffer far larger than the number of slots a
 * worker can possibly have in flight, so a full buffer would mean the parent
 * has stopped reading, which is the same condition as it being gone. */
void mynah_prefork_conn_done(void) {
    if (g_worker_chan < 0) return;
    const char b = 1;
    ssize_t n;
    do { n = write(g_worker_chan, &b, 1); } while (n < 0 && errno == EINTR);
    (void)n;
}

int mynah_prefork_recv_conn(int chan_fd, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = chan_fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    const int ready = poll(&pfd, 1, timeout_ms);
    if (ready < 0) return errno == EINTR ? -2 : -1;
    if (ready == 0) return -2;
    if ((pfd.revents & POLLIN) == 0) return -1;   /* HUP/ERR: parent is gone */
    return recv_fd(chan_fd);
}

/* ------------------------------------------------------------------ plan */

/* Resolves W and T against the cpu mask. Never invents W: the operator gives
 * it, because the whole point of prefork.h's procedure is that no constant in
 * this file could know it. T defaults to the slice width, which is the only
 * choice that leaves the machine exactly subscribed. */
static int plan_topology(const mynah_prefork_config *cfg, int *cpus, int max_cpus,
                         int *out_workers, int *out_threads, int *out_per,
                         int *out_core_major) {
    const int ncpu = cpus_allowed(cpus, max_cpus);
    *out_core_major = order_core_major(cpus, ncpu);

    int workers = cfg->workers;
    if (workers < 1) workers = 1;
    if (workers > PREFORK_MAX_WORKERS) workers = PREFORK_MAX_WORKERS;

    const int per = ncpu / workers > 0 ? ncpu / workers : 1;
    int threads = cfg->threads_per > 0 ? cfg->threads_per : per;

    *out_workers = workers;
    *out_threads = threads;
    *out_per = per;
    return ncpu;
}

void mynah_prefork_reserve_threads(mynah_prefork_config *cfg) {
    int cpus[CPU_LIST_MAX];
    int workers = 0, threads = 0, per = 0, core_major = 0;
    (void)plan_topology(cfg, cpus, CPU_LIST_MAX, &workers, &threads, &per, &core_major);
    cfg->workers = workers;
    cfg->threads_per = threads;

    const char *existing = getenv("MYNAH_THREADS");
    if (existing != NULL && existing[0] != '\0') {
        const int have = atoi(existing);
        if (have != threads) {
            fprintf(stderr,
                    "prefork: MYNAH_THREADS=%s is set in the environment and wins; "
                    "the plan wanted %d threads per worker\n", existing, threads);
        }
        cfg->threads_per = have > 0 ? have : threads;
        return;
    }
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", threads);
    setenv("MYNAH_THREADS", buf, 1);
}

void mynah_prefork_print_plan(const mynah_prefork_config *cfg, FILE *out) {
    int cpus[CPU_LIST_MAX];
    int workers = 0, threads = 0, per = 0, core_major = 0;
    mynah_prefork_config local = *cfg;
    if (local.workers < 1) local.workers = 1;
    const int ncpu = plan_topology(&local, cpus, CPU_LIST_MAX, &workers, &threads,
                                   &per, &core_major);
    const int cores = count_physical_cores(cpus, ncpu);
    /* Only claim an SMT factor when the mask actually covers whole cores.
     * A partial mask (a taskset that took one sibling of some cores and both
     * of others) has no single factor, and rounding it down printed "x 1 SMT"
     * on a host where SMT was plainly on. */
    const int smt = (cores > 0 && ncpu % cores == 0) ? ncpu / cores : 0;

    fprintf(out, "prefork plan\n");
    fprintf(out, "  allowed cpus     %d", ncpu);
    if (smt > 0) fprintf(out, "  (%d physical cores x %d SMT)", cores, smt);
    else if (cores > 0) fprintf(out, "  (%d physical cores; the mask covers them "
                                     "unevenly, so there is no one SMT factor)", cores);
    else fprintf(out, "  (physical topology unavailable on this platform)");
    fprintf(out, "\n");
    fprintf(out, "  cpu order        %s\n",
            core_major ? "core-major: a worker's slice keeps SMT siblings together"
                       : "as enumerated: no sysfs topology, slices are logical ids");
    fprintf(out, "  pinning          %s\n",
#if defined(__linux__)
            "sched_setaffinity"
#else
            "UNAVAILABLE on this platform: workers will float across all cpus"
#endif
    );
    fprintf(out, "  asked for        W=%d workers", workers);
    if (cfg->threads_per > 0) fprintf(out, ", T=%d threads (explicit)", threads);
    else fprintf(out, ", T=%d threads (derived: %d cpus / %d workers)", threads, ncpu, workers);
    fprintf(out, ", %d slots each\n", cfg->slots_per > 0 ? cfg->slots_per : 1);
    if (workers * threads > ncpu) {
        fprintf(out, "  WARNING          W*T = %d > %d allowed cpus: oversubscribed, "
                     "workers will preempt each other\n", workers * threads, ncpu);
    } else if (workers * threads < ncpu) {
        fprintf(out, "  note             W*T = %d < %d allowed cpus: %d cpus idle\n",
                workers * threads, ncpu, ncpu - workers * threads);
    }
    if (cores > 0 && smt > 1) {
        fprintf(out, "  WARNING          SMT is on (%d threads per core). Two threads of "
                     "one core share its execution units; a measurement that does not "
                     "say whether SMT was on is not a measurement.\n", smt);
    }

    /* The slices themselves, because "pinned" is only meaningful if you can
     * see what to. On an SMT host this is also the line that shows the sibling
     * ordering working: worker 0 should hold BOTH threads of its cores, not
     * one thread of twice as many. */
    for (int i = 0; i < workers; ++i) {
        const int first = i * per;
        const int count = (i + 1) * per <= ncpu ? per : ncpu - first;
        if (count <= 0) {
            fprintf(out, "  worker %-9d (no cpus left: W=%d exceeds the %d allowed)\n",
                    i, workers, ncpu);
            continue;
        }
        fprintf(out, "  worker %-9d cpus ", i);
        for (int c = first; c < first + count; ++c) {
            fprintf(out, "%s%d", c > first ? "," : "", cpus[c]);
        }
        fprintf(out, "\n");
    }

    fprintf(out,
        "\n"
        "how to choose W -- the procedure, not a number\n"
        "\n"
        "  W and T multiply to this machine. %d cpus admits", ncpu);
    {
        int printed = 0;
        for (int w = 1; w <= ncpu && printed < 6; w *= 2) {
            if (ncpu % w != 0) continue;
            fprintf(out, "%s %dx%d", printed ? "," : "", w, ncpu / w);
            ++printed;
        }
    }
    fprintf(out,
        " and they are not the same server.\n"
        "\n"
        "  STEP 1  find T*, the width at which ONE worker stops improving.\n"
        "          No prefork. Sweep the pool width and read RTF:\n"
        "            for t in 1 2 4 8 16 32; do\n"
        "              MYNAH_THREADS=$t make serving-profile MODEL_DIR=PACK\n"
        "            done\n"
        "          T* is the SMALLEST t within ~10%% of the best RTF. Decode is\n"
        "          bandwidth bound, so T* is usually far below %d.\n"
        "\n"
        "  STEP 2  sweep W at constant subscription: W*T = %d throughout.\n"
        "          For each W in {%d/T*/2, %d/T*, 2*%d/T*} (rounded to a divisor):\n"
        "            ./mynah-tts-server -m PACK --prefork $W --prefork-threads $((%d/W)) &\n"
        "            make serving-profile MODEL_DIR=PACK   # levels W, 2W, 4W\n"
        "          Record per level: completed/launched, TTFA p50 and p95,\n"
        "          STREAM_RTF p50 and p95, stall rate, and the RSS of the whole\n"
        "          process tree. Medians alone will flatter every configuration.\n"
        "\n"
        "  STEP 3  for the winner, raise concurrency until the verdict stops\n"
        "          being GOOD. THAT concurrency is this machine's answer. Publish\n"
        "          it with W, T, the cpu mask, SMT on/off, the model revision and\n"
        "          the ISA beside it -- it transfers to no other machine.\n"
        "\n"
        "  STEP 4  measure RSS after the sweep, never before: the quantized and\n"
        "          codec caches are built lazily on a worker's first synthesis and\n"
        "          are NOT shared by the fork. Only the mapped pack is. If the\n"
        "          total is far above (pack + W * per-worker cache), something you\n"
        "          believed was shared is not.\n"
        "\n"
        "  Send SIGUSR1 to the parent at any point for the per-worker table\n"
        "  (dispatched, completed, in flight, mean in flight since the last dump).\n",
        ncpu, ncpu, ncpu, ncpu, ncpu, ncpu);
}

/* ----------------------------------------------------------------- routing */

typedef struct {
    pid_t pid;
    int chan;            /* parent's end of the socketpair */
    int active;          /* dispatched minus finished */
    /* Two sets of counters on purpose. The `window_` ones are reset by every
     * SIGUSR1 dump, because "what happened in the last minute" is the question
     * a dump is asked; the totals are never reset, because a FINAL line that
     * prints the last window instead of the run reads as "this server did
     * nothing" to anyone who took a dump before shutting it down. */
    long long assigned;
    long long completed;
    long long window_assigned;
    long long window_completed;
    double area;         /* integral of `active` dt, for the mean in flight */
} worker_state;

static double mono_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* The one thing the parent ever writes to a client socket. Assembled rather
 * than written out as a literal so Content-Length cannot drift away from the
 * body it describes -- a hand-counted length is exactly the kind of constant
 * that survives an edit to the message and silently truncates the response. */
static void reject_busy(int fd) {
    static const char body[] =
        "{\"error\":{\"message\":\"all workers at capacity\",\"type\":\"server_error\"}}";
    char msg[512];
    const int len = snprintf(msg, sizeof(msg),
                             "HTTP/1.1 503 Service Unavailable\r\n"
                             "Content-Type: application/json\r\n"
                             "Content-Length: %zu\r\n"
                             "Retry-After: 1\r\n"
                             "Connection: close\r\n\r\n%s",
                             sizeof(body) - 1u, body);
    if (len <= 0) return;
    const char *p = msg;
    size_t left = (size_t)len;
    while (left > 0) {
        const ssize_t n = send(fd, p, left, 0);
        if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
        p += (size_t)n;
        left -= (size_t)n;
    }
}

static void dump_table(const worker_state *w, int workers, long long dispatched,
                       long long rejected, double window) {
    fprintf(stderr, "[prefork] dispatched=%lld rejected=%lld", dispatched, rejected);
    for (int i = 0; i < workers; ++i) {
        if (w[i].pid <= 0) { fprintf(stderr, " w%d[dead]", i); continue; }
        fprintf(stderr, " w%d[pid=%d asg=%lld done=%lld inflight=%d mean=%.2f]",
                i, (int)w[i].pid, w[i].window_assigned, w[i].window_completed,
                w[i].active, window > 0.0 ? w[i].area / window : 0.0);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
}

mynah_prefork_role mynah_prefork_run(const mynah_prefork_config *cfg,
                                     volatile sig_atomic_t *stop, int *chan_fd) {
    int cpus[CPU_LIST_MAX];
    int workers = 0, threads = 0, per = 0, core_major = 0;
    const int ncpu = plan_topology(cfg, cpus, CPU_LIST_MAX, &workers, &threads,
                                   &per, &core_major);
    const int slots = cfg->slots_per > 0 ? cfg->slots_per : 1;

    if (!cfg->quiet) {
        const int cores = count_physical_cores(cpus, ncpu);
        fprintf(stderr, "prefork: %d workers x %d threads over %d allowed cpus "
                        "(%d per worker, %s), %d slots each\n",
                workers, threads, ncpu, per,
                core_major ? "core-major slices" : "logical slices", slots);
        if (cores > 0 && ncpu % cores == 0 && ncpu / cores > 1) {
            const int smt = ncpu / cores;
            fprintf(stderr, "prefork: SMT is on (%d per core); each slice covers "
                            "%d physical cores\n", smt,
                    per / smt > 0 ? per / smt : 1);
        }
        if (workers * threads > ncpu) {
            fprintf(stderr, "prefork: WARNING oversubscribed, W*T = %d > %d cpus\n",
                    workers * threads, ncpu);
        }
#if !defined(__linux__)
        fprintf(stderr, "prefork: WARNING this platform has no cpu affinity API. "
                        "Workers are NOT pinned: they float across every cpu and two "
                        "of them will share cores. The handoff and the routing are "
                        "real; the isolation the throughput claim rests on is not.\n");
#endif
        fflush(stderr);
    }

    worker_state *w = (worker_state *)calloc((size_t)workers, sizeof(*w));
    if (w == NULL) return MYNAH_PREFORK_ERROR;
    for (int i = 0; i < workers; ++i) { w[i].pid = -1; w[i].chan = -1; }

    int live = 0;
    for (int i = 0; i < workers; ++i) {
        int sp[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0) {
            perror("prefork: socketpair");
            break;
        }
        /* Flush before forking: a buffered line in the parent's stdio would be
         * duplicated into every child and printed W times. */
        fflush(stderr);
        fflush(stdout);
        const pid_t pid = fork();
        if (pid < 0) {
            perror("prefork: fork");
            close(sp[0]);
            close(sp[1]);
            break;
        }
        if (pid == 0) {
            /* ------------------------------------------------------- child */
            close(sp[0]);
            /* The parent's ends of every earlier worker's channel came along
             * with the fork. Holding them would keep a dead worker's channel
             * open from this process's point of view and, worse, keep the
             * parent's end alive after the parent exits, so the sibling never
             * sees EOF. */
            for (int j = 0; j < i; ++j) {
                if (w[j].chan >= 0) close(w[j].chan);
            }
            free(w);
            close(cfg->listen_fd);   /* a worker must never accept: the parent routes */

            g_worker_index = i;
            g_worker_chan = sp[1];
            *chan_fd = sp[1];

            char slice[192];
            const int pinned = pin_to_slice(cpus, i * per,
                                            (i + 1) * per <= ncpu ? per : ncpu - i * per,
                                            slice, sizeof(slice));

            /* Belt and braces on the pool. src/threads.c registers a
             * pthread_atfork child handler, so this is already done; calling it
             * is idempotent and documented as such, and it keeps the invariant
             * visible at the one place a reader looks for it. */
            mynah_threadpool_after_fork();

            /* The pool resolves its width once and caches it, so MYNAH_THREADS
             * has to have been right BEFORE the model was opened -- which is
             * what mynah_prefork_reserve_threads() is for. Re-apply it here for
             * the case where it had not been resolved yet, then report what is
             * actually in force rather than what was planned. */
            {
                char buf[16];
                snprintf(buf, sizeof(buf), "%d", threads);
                setenv("MYNAH_THREADS", buf, 1);
                g_worker_threads = mynah_num_threads();
                if (g_worker_threads != threads) {
                    fprintf(stderr, "prefork: worker %d WANTED %d threads but the pool "
                                    "is already fixed at %d; export MYNAH_THREADS=%d "
                                    "before starting the server\n",
                            i, threads, g_worker_threads, threads);
                }
            }
            /* SIGUSR1, carefully. A worker must NOT get a handler installed
             * here: it inherits the one server/main.c installed before the
             * fork, and that is the one that prints the worker's counters.
             * Installing a second would mean whichever ran last silently won.
             *
             * But the parent forwards SIGUSR1 to every worker on a dump, and
             * the DEFAULT action for SIGUSR1 is to terminate the process. A
             * caller that used this module without installing a handler would
             * therefore lose its entire fleet the first time anyone asked for
             * statistics -- which is exactly what happened the first time this
             * was exercised. So: leave an inherited handler alone, and ignore
             * the signal only where there is none. A dump must never be able
             * to kill a worker. */
            {
                struct sigaction cur;
                memset(&cur, 0, sizeof(cur));
                if (sigaction(SIGUSR1, NULL, &cur) == 0 &&
                    cur.sa_handler == SIG_DFL) {
                    signal(SIGUSR1, SIG_IGN);
                }
            }
            fprintf(stderr, "prefork: worker %d pid %d threads %d cpus %s%s\n",
                    i, (int)getpid(), g_worker_threads, slice,
                    pinned ? "" : "  <-- NOT PINNED");
            fflush(stderr);
            return MYNAH_PREFORK_CHILD;
        }
        /* ------------------------------------------------------------ parent */
        close(sp[1]);
        w[i].pid = pid;
        w[i].chan = sp[0];
        ++live;
    }

    if (live == 0) {
        fprintf(stderr, "prefork: no worker could be started\n");
        free(w);
        return MYNAH_PREFORK_ERROR;
    }

    signal(SIGPIPE, SIG_IGN);
    signal(SIGUSR1, on_usr1);

    struct pollfd *pfd = (struct pollfd *)calloc((size_t)workers + 1u, sizeof(*pfd));
    if (pfd == NULL) {
        for (int i = 0; i < workers; ++i) if (w[i].pid > 0) kill(w[i].pid, SIGTERM);
        free(w);
        return MYNAH_PREFORK_ERROR;
    }

    long long dispatched = 0, rejected = 0;
    long long window_dispatched = 0, window_rejected = 0;
    double window_start = mono_seconds();
    double prev = window_start;

    while (*stop == 0 && live > 0) {
        int nf = 0, free_slots = 0;
        int map[PREFORK_MAX_WORKERS];
        for (int i = 0; i < workers; ++i) {
            if (w[i].pid <= 0) continue;
            map[nf] = i;
            pfd[nf].fd = w[i].chan;
            pfd[nf].events = POLLIN;
            pfd[nf].revents = 0;
            ++nf;
            if (w[i].active < slots) ++free_slots;
        }
        /* Stop watching the listener while every worker is full. The kernel
         * backlog holds the connection; a client waits rather than being told
         * "busy" by a server that would have had a slot a millisecond later.
         * When the backlog itself overflows the kernel refuses the connection,
         * which is the honest signal at that point. */
        int listen_slot = -1;
        if (free_slots > 0) {
            listen_slot = nf;
            pfd[nf].fd = cfg->listen_fd;
            pfd[nf].events = POLLIN;
            pfd[nf].revents = 0;
            ++nf;
        }

        const int ready = poll(pfd, (nfds_t)nf, 200);

        {   /* Integrate in-flight over wall time whether or not poll returned
             * anything: the mean is only meaningful if idle time counts. */
            const double now = mono_seconds();
            const double dt = now - prev;
            prev = now;
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid > 0) w[i].area += (double)w[i].active * dt;
            }
        }

        if (mynah_prefork_take_dump_request()) {
            dump_table(w, workers, window_dispatched, window_rejected,
                       prev - window_start);
            /* Forward it: one `kill -USR1 <parent>` should produce the whole
             * machine's view, the routing table from here and each worker's own
             * counters from the worker. */
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid > 0) kill(w[i].pid, SIGUSR1);
            }
            for (int i = 0; i < workers; ++i) {
                w[i].window_assigned = 0;
                w[i].window_completed = 0;
                w[i].area = 0.0;
            }
            window_dispatched = 0; window_rejected = 0; window_start = prev;
        }

        if (ready < 0) {
            if (errno == EINTR) continue;
            perror("prefork: poll");
            break;
        }

        for (int k = 0; k < nf; ++k) {
            if (k == listen_slot) continue;
            if ((pfd[k].revents & (POLLIN | POLLHUP | POLLERR)) == 0) continue;
            const int i = map[k];
            char buf[256];
            ssize_t n;
            do { n = read(w[i].chan, buf, sizeof(buf)); } while (n < 0 && errno == EINTR);
            if (n > 0) {
                w[i].completed += n;
                w[i].window_completed += n;
                w[i].active -= (int)n;
                if (w[i].active < 0) {
                    /* Would mean a worker reported a connection it was never
                     * given. Clamp, but say so: it is an accounting bug and it
                     * silently inflates capacity. */
                    fprintf(stderr, "prefork: worker %d reported more completions than "
                                    "dispatches; slot accounting is wrong\n", i);
                    w[i].active = 0;
                }
            } else if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
                fprintf(stderr, "prefork: worker %d (pid %d) channel closed after "
                                "%lld completed\n", i, (int)w[i].pid, w[i].completed);
                /* Reap it now rather than at shutdown. A worker that dies in
                 * the first minute of a week-long run would otherwise sit as a
                 * zombie for the rest of it, which is harmless but is exactly
                 * the sort of thing an operator finds and cannot explain. */
                (void)waitpid(w[i].pid, NULL, WNOHANG);
                close(w[i].chan);
                w[i].chan = -1;
                w[i].pid = -1;
                w[i].active = 0;
                --live;
            }
        }

        if (listen_slot < 0 || (pfd[listen_slot].revents & POLLIN) == 0) continue;

        const int cfd = accept(cfg->listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
                errno == ECONNABORTED) continue;
            perror("prefork: accept");
            break;
        }

        int best = -1;
        for (int i = 0; i < workers; ++i) {
            if (w[i].pid <= 0 || w[i].active >= slots) continue;
            if (best < 0 || w[i].active < w[best].active) best = i;
        }
        if (best < 0) {
            /* Raced: the last free slot filled between poll and accept. */
            ++rejected;
            ++window_rejected;
            reject_busy(cfd);
            close(cfd);
            continue;
        }
        if (send_fd(w[best].chan, cfd) != 0) {
            fprintf(stderr, "prefork: handing fd to worker %d failed: %s\n",
                    best, strerror(errno));
            ++rejected;
            ++window_rejected;
            reject_busy(cfd);
            close(cfd);
            continue;
        }
        /* Our copy goes now: from here the worker is the only owner, and the
         * client sees a close only when the worker closes. */
        close(cfd);
        ++w[best].active;
        ++w[best].assigned;
        ++w[best].window_assigned;
        ++dispatched;
        ++window_dispatched;
    }

    /* ------------------------------------------------------------- shutdown */
    for (int i = 0; i < workers; ++i) {
        if (w[i].pid > 0) kill(w[i].pid, SIGTERM);
    }
    for (int i = 0; i < workers; ++i) {
        if (w[i].pid <= 0) continue;
        int status = 0;
        pid_t got;
        do { got = waitpid(w[i].pid, &status, 0); } while (got < 0 && errno == EINTR);
        (void)got;
        (void)status;
        if (w[i].chan >= 0) close(w[i].chan);
        w[i].chan = -1;
    }
    /* Totals for the run, not for the window since the last dump. A leftover
     * `still-in-flight` here is the interesting number: it means a worker was
     * handed a connection it never reported finished, which is a leaked slot. */
    fprintf(stderr, "prefork: final  dispatched=%lld rejected=%lld\n",
            dispatched, rejected);
    for (int i = 0; i < workers; ++i) {
        fprintf(stderr, "  worker %d: assigned=%lld completed=%lld still-in-flight=%d\n",
                i, w[i].assigned, w[i].completed, w[i].active);
    }
    close(cfg->listen_fd);
    free(pfd);
    free(w);
    return MYNAH_PREFORK_PARENT_DONE;
}
