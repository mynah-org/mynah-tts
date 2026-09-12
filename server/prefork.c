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

#include "costmap.h"
#include "threads.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
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
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach/mach.h>
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

/* Same idea for the cgroup tree. Unlike the topology read, parsing `cpu.max`
 * is not Linux-only *code* -- it is an ordinary file read and a division, so
 * it compiles and RUNS everywhere and simply finds nothing where there is no
 * cgroupfs. Pointing this at a fabricated directory is therefore a real
 * execution of the parser on any platform, which is the only way the quota
 * arithmetic gets exercised on a developer machine at all. */
#ifndef MYNAH_CGROUP_ROOT
#define MYNAH_CGROUP_ROOT "/sys/fs/cgroup"
#endif

/* Both roots take an environment override in preference to the compile-time
 * default, so a test does not need its own build of this translation unit. */
#if defined(__linux__)   /* the only caller is the sysfs reader below */
static const char *topology_root(void) {
    const char *e = getenv("MYNAH_CPU_TOPOLOGY_ROOT");
    return (e != NULL && e[0] != '\0') ? e : MYNAH_CPU_TOPOLOGY_ROOT;
}
#endif
static const char *cgroup_root(void) {
    const char *e = getenv("MYNAH_CGROUP_ROOT");
    return (e != NULL && e[0] != '\0') ? e : MYNAH_CGROUP_ROOT;
}

/* ------------------------------------------------------------- cpu budget
 *
 * The residual the reference names and did not close: sched_getaffinity says
 * WHICH cpus we may use, the cgroup quota says HOW MUCH of them. They are
 * different limits and a container routinely sets only the second. A pod with
 * `cpu: "4"` and no cpuset has an affinity mask covering all 64 cores of its
 * host and a budget of four: planning W*T from the mask alone builds a server
 * that is throttled the instant it works, and the throttling appears as
 * latency with no CPU-bound thread to blame it on.
 *
 * We do not silently re-plan on the quota -- it is a rate over a period, not a
 * set of cpus, and W workers of T threads is still a legitimate shape under
 * one. We WARN, because "you asked for 16 threads inside a 4-cpu budget" is a
 * sentence an operator can act on and a 4x RTF regression is not.
 *
 * Returns 1 and fills *cpus when a finite quota is in force, 0 otherwise. */
static int cgroup_cpu_budget(double *cpus, char *detail, size_t detail_cap) {
    const char *root = cgroup_root();
    char path[512];
    if (detail != NULL && detail_cap > 0) detail[0] = '\0';

    /* cgroup v2: one file, "$MAX $PERIOD", MAX either a number or "max". */
    snprintf(path, sizeof(path), "%s/cpu.max", root);
    FILE *f = fopen(path, "r");
    if (f != NULL) {
        char quota[64];
        long period = 0;
        quota[0] = '\0';
        const int got = fscanf(f, "%63s %ld", quota, &period);
        fclose(f);
        if (got == 2 && period > 0 && strcmp(quota, "max") != 0) {
            const double q = atof(quota);
            if (q > 0.0) {
                *cpus = q / (double)period;
                if (detail != NULL) {
                    snprintf(detail, detail_cap, "cgroup v2 cpu.max %s %ld",
                             quota, period);
                }
                return 1;
            }
        }
        if (got >= 1 && strcmp(quota, "max") == 0) return 0;   /* explicitly none */
    }

    /* cgroup v1: two files, and a quota of -1 means unlimited. */
    long q = -1, period = 0;
    snprintf(path, sizeof(path), "%s/cpu/cpu.cfs_quota_us", root);
    f = fopen(path, "r");
    if (f == NULL) {
        snprintf(path, sizeof(path), "%s/cpu.cfs_quota_us", root);
        f = fopen(path, "r");
    }
    if (f != NULL) { if (fscanf(f, "%ld", &q) != 1) q = -1; fclose(f); }
    snprintf(path, sizeof(path), "%s/cpu/cpu.cfs_period_us", root);
    f = fopen(path, "r");
    if (f == NULL) {
        snprintf(path, sizeof(path), "%s/cpu.cfs_period_us", root);
        f = fopen(path, "r");
    }
    if (f != NULL) { if (fscanf(f, "%ld", &period) != 1) period = 0; fclose(f); }
    if (q > 0 && period > 0) {
        *cpus = (double)q / (double)period;
        if (detail != NULL) {
            snprintf(detail, detail_cap, "cgroup v1 cfs_quota/period %ld/%ld",
                     q, period);
        }
        return 1;
    }
    return 0;
}

/* Warns when the CPU budget contradicts the plan. Separate from the planner on
 * purpose: the plan is a statement about cpus, this is a statement about time,
 * and conflating them is how "we planned 32 and measured 4" happens. */
static void warn_cpu_budget(int workers, int threads, int ncpu, FILE *out) {
    double budget = 0.0;
    char detail[128];
    if (!cgroup_cpu_budget(&budget, detail, sizeof(detail))) return;

    fprintf(out, "prefork: cpu budget    %.2f cpus (%s)\n", budget, detail);
    const double want = (double)workers * (double)threads;
    if (want > budget + 0.01) {
        fprintf(out,
            "prefork: WARNING the CPU BUDGET CONTRADICTS THE PLAN. W*T = %d threads, "
            "but this cgroup allows %.2f cpus of runtime. The affinity mask says %d "
            "cpus and the quota says %.2f: the mask is WHICH cpus, the quota is HOW "
            "MUCH, and only the quota is enforced -- by throttling every worker at a "
            "period boundary with no busy thread to blame. Raise the quota, or plan "
            "W*T <= %.0f.\n",
            workers * threads, budget, ncpu, budget, budget);
    } else if ((double)ncpu > budget + 0.01) {
        fprintf(out,
            "prefork: note the affinity mask covers %d cpus but the budget is %.2f; "
            "the plan (W*T = %d) fits it. sysconf-based planning would not have.\n",
            ncpu, budget, workers * threads);
    }
}

/* ---------------------------------------------------- fork preconditions
 *
 * Two things must be true at the instant we fork, and neither fails loudly on
 * its own. They are checked here rather than trusted to the comment in
 * server/main.c that asserts them, because a comment is not a check and the
 * ordering it describes is one refactor away from being wrong.
 *
 * THE MUTEX QUESTION first, because it is what the audit is really about.
 * fork() copies the address space but only the calling thread. Every mutex in
 * the child is a bit-for-bit copy of its state at that instant: one another
 * thread held is inherited LOCKED, by a thread that does not exist in the
 * child, and can never be unlocked. Zeroing such a mutex is NOT a repair -- on
 * glibc a zeroed pthread_mutex_t happens to look unlocked, which is exactly
 * why the bug survives testing there, and on other implementations it is
 * simply corrupt. The only correct repair is pthread_mutex_init(); the only
 * way to not need one is to fork while single-threaded.
 *
 * The audit of this tree, recorded here because it is the answer to E5-13:
 *
 *   src/threads.c  g_job_mu, g_init_mu, g_job_cv, g_done_cv
 *                  REINITIALIZED (not zeroed) by mynah_threadpool_after_fork(),
 *                  registered with pthread_atfork() and also called explicitly
 *                  by the child below. Correct.
 *   src/threads.c  g_blas_mu       NOT reinitialized by anything.
 *   src/qmat.c     g_stats_mutex   NOT reinitialized by anything.
 *   src/costmap.c  no mutex at all: thread-local blocks indexed by an atomic
 *                  counter, so there is nothing to inherit. Its CONTENT is
 *                  still wrong in a child -- it holds the parent's model-load
 *                  and pre-warm regions, which every worker would then report
 *                  as its own, counting one load W times.
 *                  mynah_costmap_after_fork() exists for exactly that and had
 *                  no caller anywhere in the tree. The child below calls it.
 *
 * The two unrepaired mutexes are safe ONLY while the fork is single-threaded,
 * because a lock no other thread can hold cannot be inherited held. That turns
 * "we fork from main, early" from a convention into a load-bearing invariant,
 * so it is checked rather than assumed. */

/* Threads in this process, or -1 where the platform will not say. */
static int process_thread_count(void) {
#if defined(__linux__)
    DIR *d = opendir("/proc/self/task");
    if (d == NULL) return -1;
    int n = 0;
    const struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] != '.') ++n;
    }
    closedir(d);
    return n > 0 ? n : -1;
#elif defined(__APPLE__)
    thread_act_array_t list = NULL;
    mach_msg_type_number_t n = 0;
    if (task_threads(mach_task_self(), &list, &n) != KERN_SUCCESS) return -1;
    for (mach_msg_type_number_t i = 0; i < n; ++i) {
        mach_port_deallocate(mach_task_self(), list[i]);
    }
    vm_deallocate(mach_task_self(), (vm_address_t)list,
                  (vm_size_t)n * sizeof(*list));
    return (int)n;
#else
    return -1;
#endif
}

/* A resident GPU compute runtime, if one can be seen from inside the process.
 *
 * This is the reference's still-open bug and it earns the emphasis: a GPU
 * context does not survive fork() -- CUDA's own documentation says so. The
 * child inherits the handles and the device pointers but not the driver's
 * per-process state behind them, and what follows is not a crash. It is
 * kernels launched against a context that is no longer valid: WRONG AUDIO,
 * which nothing downstream of the decoder can tell from right audio. A wrong
 * answer is a far worse failure than a dead worker. `--backend cuda --prefork
 * N` must refuse, not try.
 *
 * cfg->gpu_backend_open is the authoritative signal and a caller that opens a
 * backend should set it. This scan is the backstop for a caller that does not:
 * a CUDA/NVIDIA runtime mapped into the process is unambiguous. Metal is
 * deliberately NOT grounds for refusal -- Metal.framework is pulled in
 * transitively by unrelated system frameworks, and a mapped library is not a
 * created device.
 *
 * Returns 1 and names what it found, 0 otherwise. */
static int gpu_runtime_resident(char *what, size_t cap) {
    static const char *const markers[] = {
        "libcuda.so", "libcudart", "libnvidia-ml", "libcuda.dylib", NULL
    };
#if defined(__linux__)
    FILE *f = fopen("/proc/self/maps", "r");
    if (f == NULL) return 0;
    char line[1024];
    while (fgets(line, sizeof(line), f) != NULL) {
        for (int i = 0; markers[i] != NULL; ++i) {
            if (strstr(line, markers[i]) != NULL) {
                snprintf(what, cap, "%s", markers[i]);
                fclose(f);
                return 1;
            }
        }
    }
    fclose(f);
    return 0;
#elif defined(__APPLE__)
    const uint32_t n = _dyld_image_count();
    for (uint32_t j = 0; j < n; ++j) {
        const char *name = _dyld_get_image_name(j);
        if (name == NULL) continue;
        for (int i = 0; markers[i] != NULL; ++i) {
            if (strstr(name, markers[i]) != NULL) {
                snprintf(what, cap, "%s", name);
                return 1;
            }
        }
    }
    return 0;
#else
    (void)what; (void)cap; (void)markers;
    return 0;
#endif
}

/* Everything that must hold before the first fork(). Returns 0 to proceed, -1
 * to refuse. Refusing is the point: these produce a WRONG ANSWER rather than a
 * crash, and a wrong answer that appears only under prefork is the most
 * expensive kind of bug this file could ship. */
static int check_fork_preconditions(const mynah_prefork_config *cfg) {
    int bad = 0;

    char gpu[512];
    if (cfg->gpu_backend_open) {
        fprintf(stderr,
            "prefork: REFUSING TO FORK -- a GPU backend is open in this process.\n"
            "  A GPU context does not survive fork(): the child inherits the handles\n"
            "  but not the driver state behind them, so its kernels run against a\n"
            "  context that no longer exists. That does not crash. It produces wrong\n"
            "  audio, which nothing downstream can detect. Run prefork on the CPU\n"
            "  backend, or run the GPU backend without prefork.\n");
        bad = 1;
    } else if (gpu_runtime_resident(gpu, sizeof(gpu))) {
        fprintf(stderr,
            "prefork: REFUSING TO FORK -- a GPU compute runtime is mapped into this\n"
            "  process (%s). Even with no context created yet this is not a shape in\n"
            "  which forked workers can be trusted; see above. Set\n"
            "  MYNAH_PREFORK_ALLOW_GPU=1 only having established that no device state\n"
            "  exists.\n", gpu);
        if (getenv("MYNAH_PREFORK_ALLOW_GPU") == NULL) bad = 1;
        else fprintf(stderr, "prefork: MYNAH_PREFORK_ALLOW_GPU is set; continuing\n");
    }

    const int threads = process_thread_count();
    if (threads > 1) {
        /* Not fatal by itself: the pool registers a pthread_atfork child handler
         * that reinitializes its four locks, so the pool survives. But g_blas_mu
         * (src/threads.c) and g_stats_mutex (src/qmat.c) have no such handler,
         * and a fork taken while another thread holds either gives every worker
         * a mutex that can never be unlocked -- a hang with no error message, in
         * a child, under load. Say exactly that, and let an operator who wants
         * the invariant enforced rather than reported ask for it. */
        fprintf(stderr,
            "prefork: WARNING forking with %d threads in this process, not 1.\n"
            "  fork() copies one thread and every mutex in whatever state it was in.\n"
            "  src/threads.c's pool locks are reinitialized by its atfork handler,\n"
            "  but src/threads.c g_blas_mu and src/qmat.c g_stats_mutex are NOT: if\n"
            "  any thread held one at this instant, every worker inherits it locked\n"
            "  and the first worker to take it hangs forever. The fork belongs before\n"
            "  the pool, the scheduler and the HTTP workers exist.\n"
            "  Set MYNAH_PREFORK_STRICT=1 to make this a refusal.\n", threads);
        if (getenv("MYNAH_PREFORK_STRICT") != NULL) {
            fprintf(stderr, "prefork: MYNAH_PREFORK_STRICT is set; refusing.\n");
            bad = 1;
        }
    } else if (threads < 0) {
        fprintf(stderr, "prefork: note this platform will not report its thread "
                        "count; the single-threaded-fork invariant is unchecked\n");
    }

    return bad ? -1 : 0;
}

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
             topology_root(), cpu, leaf);
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

/* ------------------------------------------------------------ the ladder
 *
 * Resolving the four rungs. Every knob is "0 = unset" so a caller that memsets
 * mynah_prefork_config and never hears of admission control still gets the
 * measured defaults, and every knob has an environment override because the
 * ladder has no command-line flags yet -- those belong to server/main.c's
 * argument parser, which this lane does not own. */

static double mono_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* Inherited by every worker across the fork, which is why a worker never has
 * to be told the cap and can never disagree with the parent about it. */
static int g_service_cap_ms = 0;

static int env_int(const char *name, int *out) {
    const char *e = getenv(name);
    if (e == NULL || e[0] == '\0') return 0;
    char *end = NULL;
    const long v = strtol(e, &end, 10);
    if (end == e) return 0;
    *out = (int)v;
    return 1;
}

void mynah_prefork_apply_env(mynah_prefork_config *cfg) {
    const char *e = getenv("MYNAH_PREFORK_QUEUE");
    if (e != NULL && e[0] != '\0') {
        if (strcmp(e, "unbounded") == 0) {
            cfg->queue_per_worker = MYNAH_PREFORK_QUEUE_UNBOUNDED;
        } else {
            int v = 0;
            if (env_int("MYNAH_PREFORK_QUEUE", &v)) {
                cfg->queue_per_worker = v > 0 ? v
                                     : (v == 0 ? MYNAH_PREFORK_QUEUE_NONE
                                               : MYNAH_PREFORK_QUEUE_UNBOUNDED);
            }
        }
    }
    int v = 0;
    /* A deadline or a cap of zero means DISABLED, not "use the default": an
     * operator who types 0 is turning the rung off. Unset is what asks for the
     * default, and unset is the absence of the variable. */
    if (env_int("MYNAH_PREFORK_QUEUE_MS", &v)) cfg->queue_deadline_ms = v > 0 ? v : -1;
    if (env_int("MYNAH_PREFORK_SERVICE_MS", &v)) cfg->service_cap_ms = v > 0 ? v : -1;
}

/* -1 = unbounded, 0 = no queue, >0 = that many per live worker. */
static int resolve_queue_per_worker(const mynah_prefork_config *cfg) {
    if (cfg->queue_per_worker == MYNAH_PREFORK_QUEUE_NONE) return 0;
    if (cfg->queue_per_worker < 0) return -1;
    if (cfg->queue_per_worker == 0) return MYNAH_PREFORK_QUEUE_DEFAULT;
    return cfg->queue_per_worker;
}
static int resolve_queue_deadline_ms(const mynah_prefork_config *cfg) {
    if (cfg->queue_deadline_ms < 0) return 0;              /* disabled */
    if (cfg->queue_deadline_ms == 0) return MYNAH_PREFORK_QUEUE_DEADLINE_MS;
    return cfg->queue_deadline_ms;
}
static int resolve_service_cap_ms(const mynah_prefork_config *cfg) {
    if (cfg->service_cap_ms < 0) return 0;                 /* disabled */
    if (cfg->service_cap_ms == 0) return MYNAH_PREFORK_SERVICE_CAP_MS;
    return cfg->service_cap_ms;
}

/* Prints the ladder that is actually in force, and shouts about an unbounded
 * queue. The shout is deliberately impossible to skim past: an unbounded
 * admission queue is not a generous setting, it is the listener-backlog
 * mistake wearing different clothes. The request still waits, the wait is
 * still unbounded, and the only thing that changed is that the wait now
 * happens somewhere this process can see and chooses not to act on. */
static void describe_ladder(int workers, int slots, int q_per, int deadline_ms,
                            int service_ms, FILE *out) {
    fprintf(out, "prefork: admission   rung1 slots %d/worker (%d total)",
            slots, slots * workers);
    if (q_per < 0)      fprintf(out, " · rung2 queue UNBOUNDED");
    else if (q_per == 0) fprintf(out, " · rung2 queue disabled (refuse immediately)");
    else fprintf(out, " · rung2 queue %d/worker (%d total)", q_per, q_per * workers);
    if (deadline_ms > 0) fprintf(out, " · rung3 deadline %d ms", deadline_ms);
    else fprintf(out, " · rung3 deadline off");
    if (service_ms > 0) fprintf(out, " · rung4 service cap %d ms", service_ms);
    else fprintf(out, " · rung4 service cap off");
    fprintf(out, "\n");

    if (q_per < 0) {
        fprintf(out,
"prefork: ############################################################\n"
"prefork: ##  THE ADMISSION QUEUE BOUND IS DISABLED.                ##\n"
"prefork: ##                                                        ##\n"
"prefork: ##  An unbounded queue does not add capacity. It converts ##\n"
"prefork: ##  a refusal the client can see into a wait it cannot,   ##\n"
"prefork: ##  which is precisely the failure this ladder exists to  ##\n"
"prefork: ##  prevent -- the reference measured p95 TTFB 4470 ms    ##\n"
"prefork: ##  with over 97%% of the tail invisible before accept().##\n"
"prefork: ##                                                        ##\n"
"prefork: ##  Admission capacity and sustained capacity are not the ##\n"
"prefork: ##  same number. Raising the bound until the rejects stop ##\n"
"prefork: ##  admitted C32 with zero refusals on the reference and  ##\n"
"prefork: ##  nothing above C20 sustained. The refusals were the    ##\n"
"prefork: ##  server telling the truth.                             ##\n"
"prefork: ##                                                        ##\n"
"prefork: ##  Memory now grows with arrival rate, and every queued  ##\n"
"prefork: ##  descriptor is an fd this process must hold.           ##\n"
"prefork: ############################################################\n");
    }
    if (deadline_ms <= 0 && q_per != 0) {
        fprintf(out,
            "prefork: WARNING rung 3 is off: a queued request has no deadline and will "
            "wait for a slot indefinitely. The queue bound is then the only thing "
            "limiting how long a client waits, and a bound is not a deadline.\n");
    }
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
    mynah_prefork_apply_env(&local);
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
    warn_cpu_budget(workers, threads, ncpu, out);
    describe_ladder(workers, cfg->slots_per > 0 ? cfg->slots_per : 1,
                    resolve_queue_per_worker(&local),
                    resolve_queue_deadline_ms(&local),
                    resolve_service_cap_ms(&local), out);

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

/* ------------------------------------------------------- refusal reasons
 *
 * One row per rung. The `code` is the contract: it appears as `error.code` in
 * the JSON body and as the counter name in every stats line, so a client
 * matching on it and an operator reading a dump are looking at the same
 * string. Append-only -- never renumber, never rename.
 *
 * Retry-After is per reason and not a constant, because the reasons say
 * different things about retrying. At-capacity is transient: come back. A
 * service-cap breach is a property of the request itself, and telling a client
 * to retry a request that will fail identically is how a refusal becomes a
 * retry storm. */
static const struct {
    const char *code;
    const char *status;
    const char *message;
    int         retry_after;   /* 0 = omit the header entirely */
} REFUSAL[MYNAH_PREFORK_REFUSE__COUNT] = {
    { "server_at_capacity", "503 Service Unavailable",
      "every worker is at its slot cap and the admission queue is full", 1 },
    { "queued_too_long", "503 Service Unavailable",
      "waited in the admission queue longer than the deadline", 1 },
    { "service_cap_exceeded", "503 Service Unavailable",
      "the request ran past its per-request service cap", 0 },
    { "handoff_failed", "503 Service Unavailable",
      "the chosen worker could not be handed the connection", 1 },
};

const char *mynah_prefork_refusal_code(mynah_prefork_refusal reason) {
    if (reason < 0 || reason >= MYNAH_PREFORK_REFUSE__COUNT) return "unknown";
    return REFUSAL[reason].code;
}

size_t mynah_prefork_refusal_response(mynah_prefork_refusal reason,
                                      char *buf, size_t cap) {
    if (reason < 0 || reason >= MYNAH_PREFORK_REFUSE__COUNT) {
        reason = MYNAH_PREFORK_REFUSE_AT_CAPACITY;
    }
    /* Assembled rather than written out as a literal so Content-Length cannot
     * drift away from the body it describes. A hand-counted length is exactly
     * the constant that survives an edit to the message and then silently
     * truncates every refusal. */
    char body[256];
    const int blen = snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s\",\"type\":\"server_error\",\"code\":\"%s\"}}",
        REFUSAL[reason].message, REFUSAL[reason].code);
    if (blen <= 0 || (size_t)blen >= sizeof(body)) return 0;

    char retry[40];
    retry[0] = '\0';
    if (REFUSAL[reason].retry_after > 0) {
        snprintf(retry, sizeof(retry), "Retry-After: %d\r\n",
                 REFUSAL[reason].retry_after);
    }
    const int n = snprintf(buf, cap,
        "HTTP/1.1 %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "%s"
        "Connection: close\r\n"
        "\r\n%s",
        REFUSAL[reason].status, blen, retry, body);
    if (n <= 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

/* ------------------------------------------------- refusing without an RST
 *
 * See the long note on mynah_prefork_refuse_and_close() in prefork.h. The
 * short version: write, shutdown(SHUT_WR), DRAIN, close -- and a close that
 * skips the drain sends an RST that destroys the response the client was about
 * to read. Both halves of that sequence are bounded, by bytes and by time,
 * because an unbounded drain is a way for one client to occupy the router. */

#define PF_LINGER_MAX   256          /* concurrent lingering closes            */
#define PF_LINGER_MS    500.0        /* per-fd budget for the whole sequence   */
#define PF_DRAIN_MAX    (64u * 1024u)/* bytes we will read before giving up    */
#define PF_QPOLL_MAX    64           /* queued fds watched for a hangup        */

typedef struct {
    int    fd;
    double deadline;      /* mono seconds */
    char   msg[384];
    size_t msg_len;
    size_t msg_sent;
    size_t drained;
    int    shut;          /* SHUT_WR already done */
} pf_linger;

static void set_nonblock(int fd) {
    const int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

static void linger_begin(pf_linger *L, int fd, mynah_prefork_refusal reason,
                         double now) {
    memset(L, 0, sizeof(*L));
    L->fd = fd;
    L->deadline = now + PF_LINGER_MS / 1000.0;
    L->msg_len = mynah_prefork_refusal_response(reason, L->msg, sizeof(L->msg));
    set_nonblock(fd);
}

/* Which direction this fd is waiting on right now. */
static short linger_events(const pf_linger *L) {
    return (L->msg_sent < L->msg_len) ? (short)POLLOUT : (short)POLLIN;
}

/* Advances one lingering close as far as it can go without blocking.
 * Returns 1 when the descriptor has been closed and the slot is free. */
static int linger_step(pf_linger *L, double now) {
    /* 1. the response itself */
    while (L->msg_sent < L->msg_len) {
        const ssize_t n = send(L->fd, L->msg + L->msg_sent,
                               L->msg_len - L->msg_sent, 0);
        if (n > 0) { L->msg_sent += (size_t)n; continue; }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (now >= L->deadline) break;
            return 0;                      /* wait for POLLOUT */
        }
        break;                             /* peer is gone: nothing to deliver */
    }

    /* 2. the FIN. This is what tells the client the response is complete, and
     *    it must come BEFORE the drain: a client waiting to send more will not
     *    finish until it learns we are done talking. */
    if (L->msg_sent >= L->msg_len && !L->shut) {
        shutdown(L->fd, SHUT_WR);
        L->shut = 1;
    }

    /* 3. the drain. The whole point of the exercise: unread bytes in the
     *    receive queue at close() time make the kernel send an RST instead of
     *    a FIN, and the RST discards our response along with it. */
    if (L->shut) {
        char scratch[4096];
        for (;;) {
            const ssize_t n = recv(L->fd, scratch, sizeof(scratch), 0);
            if (n > 0) {
                L->drained += (size_t)n;
                if (L->drained >= PF_DRAIN_MAX) break;   /* enough; not our body */
                continue;
            }
            if (n == 0) break;                           /* clean: peer closed */
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (now >= L->deadline) break;
                return 0;                                /* wait for POLLIN */
            }
            break;
        }
    }

    close(L->fd);
    L->fd = -1;
    return 1;
}

void mynah_prefork_refuse_and_close(int fd, mynah_prefork_refusal reason) {
    if (fd < 0) return;
    /* No globals are touched here: this is callable from any thread, which
     * matters because server/main.c's shed path runs on an HTTP worker. It is
     * bounded-blocking rather than non-blocking -- a worker thread can afford
     * half a second on a connection it is refusing, and the router cannot,
     * which is why the router uses refuse_park() below instead. */
    pf_linger L;
    linger_begin(&L, fd, reason, mono_seconds());
    for (;;) {
        const double now = mono_seconds();
        if (linger_step(&L, now)) return;
        const double left_ms = (L.deadline - now) * 1000.0;
        if (left_ms <= 0.0) break;
        struct pollfd pfd;
        pfd.fd = L.fd;
        pfd.events = linger_events(&L);
        pfd.revents = 0;
        const int r = poll(&pfd, 1, (int)left_ms);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
    }
    if (L.fd >= 0) close(L.fd);
}

int mynah_prefork_service_cap_ms(void) { return g_service_cap_ms; }

/* ----------------------------------------------------------------- routing */

typedef struct {
    int    fd;
    double enqueued;      /* mono seconds, stamped at accept() */
} pf_queued;

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
    /* Rung 4's watchdog. Dispatch timestamps of the connections this worker
     * still owes us a completion for, oldest first. Completions do not arrive
     * in dispatch order, so popping the head on every completion makes this an
     * approximation -- but it approximates in the safe direction: the head is
     * always the oldest outstanding dispatch, so "the head is past the cap" is
     * never a false alarm, it can only be late. */
    double *disp;
    int disp_cap, disp_head, disp_n;
    long long over_cap;  /* outstanding connections seen past the service cap */
} worker_state;

static void disp_push(worker_state *w, double t) {
    if (w->disp == NULL || w->disp_n >= w->disp_cap) return;
    w->disp[(w->disp_head + w->disp_n) % w->disp_cap] = t;
    ++w->disp_n;
}
static void disp_pop(worker_state *w, int n) {
    while (n-- > 0 && w->disp_n > 0) {
        w->disp_head = (w->disp_head + 1) % w->disp_cap;
        --w->disp_n;
    }
}

static void dump_table(const worker_state *w, int workers, long long dispatched,
                       const long long *refused, int queued, double window) {
    fprintf(stderr, "[prefork] dispatched=%lld queued=%d", dispatched, queued);
    for (int r = 0; r < MYNAH_PREFORK_REFUSE__COUNT; ++r) {
        fprintf(stderr, " %s=%lld", REFUSAL[r].code, refused[r]);
    }
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
    mynah_prefork_config local = *cfg;
    mynah_prefork_apply_env(&local);

    int cpus[CPU_LIST_MAX];
    int workers = 0, threads = 0, per = 0, core_major = 0;
    const int ncpu = plan_topology(&local, cpus, CPU_LIST_MAX, &workers, &threads,
                                   &per, &core_major);
    const int slots = local.slots_per > 0 ? local.slots_per : 1;
    const int q_per = resolve_queue_per_worker(&local);      /* -1 = unbounded */
    const int deadline_ms = resolve_queue_deadline_ms(&local);
    g_service_cap_ms = resolve_service_cap_ms(&local);

    if (!local.quiet) {
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
        warn_cpu_budget(workers, threads, ncpu, stderr);
        describe_ladder(workers, slots, q_per, deadline_ms, g_service_cap_ms, stderr);
#if !defined(__linux__)
        fprintf(stderr, "prefork: WARNING this platform has no cpu affinity API. "
                        "Workers are NOT pinned: they float across every cpu and two "
                        "of them will share cores. The handoff and the routing are "
                        "real; the isolation the throughput claim rests on is not.\n");
#endif
        fflush(stderr);
    }

    /* Before anything is forked, and before the listening socket is committed
     * to a topology we cannot undo. */
    if (check_fork_preconditions(&local) != 0) return MYNAH_PREFORK_ERROR;

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
            close(local.listen_fd);  /* a worker must never accept: the parent routes */

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
             * visible at the one place a reader looks for it.
             *
             * Note what this does NOT cover, because the audit above found it:
             * src/threads.c's g_blas_mu and src/qmat.c's g_stats_mutex have no
             * after-fork repair anywhere in the tree. They are safe only while
             * the fork is single-threaded, which check_fork_preconditions()
             * above now verifies rather than assumes. */
            mynah_threadpool_after_fork();

            /* The parent opened the pack and may have pre-warmed. Those regions
             * describe work this worker did not do, and merging W copies of them
             * would count one model load W times. mynah_costmap_after_fork()
             * exists for precisely this and had no caller in the tree until
             * now. It has no lock to repair -- the costmap is thread-local
             * blocks behind an atomic index -- so this is about correctness of
             * the numbers, not of the locking. */
            mynah_costmap_after_fork();

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

    /* Per-worker watchdog rings, parent-only: allocated after the fork so no
     * child ever inherits or frees them. */
    for (int i = 0; i < workers; ++i) {
        w[i].disp_cap = slots + 1;
        w[i].disp = (double *)calloc((size_t)w[i].disp_cap, sizeof(double));
    }

    /* The admission queue (rung 2). Bounded at q_per per LIVE worker; the
     * allocation follows the bound, and only an explicitly unbounded queue
     * grows. */
    int q_alloc = (q_per < 0) ? 16 : (q_per > 0 ? q_per * workers : 1);
    pf_queued *q = (pf_queued *)calloc((size_t)q_alloc, sizeof(*q));
    int q_n = 0;

    const size_t pfd_cap = (size_t)workers + 1u + PF_QPOLL_MAX + PF_LINGER_MAX;
    struct pollfd *pfd = (struct pollfd *)calloc(pfd_cap, sizeof(*pfd));
    if (pfd == NULL || q == NULL) {
        for (int i = 0; i < workers; ++i) if (w[i].pid > 0) kill(w[i].pid, SIGTERM);
        for (int i = 0; i < workers; ++i) free(w[i].disp);
        free(pfd); free(q); free(w);
        return MYNAH_PREFORK_ERROR;
    }

    pf_linger linger[PF_LINGER_MAX];
    int linger_n = 0;
    long long linger_forced = 0;

    long long dispatched = 0;
    long long window_dispatched = 0;
    long long refused[MYNAH_PREFORK_REFUSE__COUNT];
    long long window_refused[MYNAH_PREFORK_REFUSE__COUNT];
    memset(refused, 0, sizeof(refused));
    memset(window_refused, 0, sizeof(window_refused));
    long long queued_total = 0, client_gone = 0;
    int queue_peak = 0;

    double window_start = mono_seconds();
    double prev = window_start;

/* Parks a refusal in the lingering-close set. Never blocks, because the router
 * is single threaded and a blocking refusal would stall every other client --
 * the same disease as a router that stops polling its listener. */
#define PF_REFUSE_PARK(FD, REASON, NOW)                                        \
    do {                                                                       \
        pf_linger L_;                                                          \
        linger_begin(&L_, (FD), (REASON), (NOW));                              \
        if (!linger_step(&L_, (NOW))) {                                        \
            if (linger_n < PF_LINGER_MAX) {                                    \
                linger[linger_n++] = L_;                                       \
            } else {                                                           \
                /* The set is full. Finish what we can without blocking and    \
                 * close; this is the only path that can still produce an RST, \
                 * so it is counted rather than hidden. */                     \
                ++linger_forced;                                               \
                L_.deadline = 0.0;                                             \
                (void)linger_step(&L_, (NOW) + 1.0);                           \
            }                                                                  \
        }                                                                      \
    } while (0)

    while (*stop == 0 && live > 0) {
        int nf = 0, listen_slot = -1;
        int map[PREFORK_MAX_WORKERS];
        for (int i = 0; i < workers; ++i) {
            if (w[i].pid <= 0) continue;
            map[nf] = i;
            pfd[nf].fd = w[i].chan;
            pfd[nf].events = POLLIN;
            pfd[nf].revents = 0;
            ++nf;
        }
        const int worker_slots = nf;

        /* RUNG 1, AND THE WHOLE REASON THIS FILE WAS REVISITED. The listener
         * is in the poll set UNCONDITIONALLY -- not "while a slot is free".
         * Watching it only when there is room is the optimisation that reads
         * as free and costs a measurement campaign: a client left in the
         * kernel backlog has not been accepted, so it has no queue entry, no
         * deadline and no timestamp, and nothing in this process can see it
         * waiting. The reference measured p95 TTFB 4470 ms with >97% of the
         * tail before accept(). Accepting and refusing in 200 microseconds is
         * strictly better than an invisible four-second wait, and it is the
         * only way rungs 2 and 3 get to exist at all. */
        listen_slot = nf;
        pfd[nf].fd = local.listen_fd;
        pfd[nf].events = POLLIN;
        pfd[nf].revents = 0;
        ++nf;

        /* Queued clients, watched for a HANGUP ONLY. events is deliberately 0:
         * a queued client has already sent its request, so asking for POLLIN
         * would make poll() return immediately every single time and spin the
         * router at 100% CPU. POLLERR/POLLHUP/POLLNVAL are reported in revents
         * regardless of events (POSIX), which is exactly the subset we want. */
        const int q_watch = q_n < PF_QPOLL_MAX ? q_n : PF_QPOLL_MAX;
        const int q_first = nf;
        for (int i = 0; i < q_watch; ++i) {
            pfd[nf].fd = q[i].fd;
            pfd[nf].events = 0;
            pfd[nf].revents = 0;
            ++nf;
        }

        const int l_first = nf;
        for (int i = 0; i < linger_n; ++i) {
            pfd[nf].fd = linger[i].fd;
            pfd[nf].events = linger_events(&linger[i]);
            pfd[nf].revents = 0;
            ++nf;
        }

        /* Sleep no longer than the next thing that needs attention, so a queue
         * deadline is honoured even when nothing else happens. */
        int timeout = 200;
        {
            const double now = mono_seconds();
            if (q_n > 0 && deadline_ms > 0) {
                const double left = (double)deadline_ms - (now - q[0].enqueued) * 1000.0;
                if (left < timeout) timeout = left > 0.0 ? (int)left : 0;
            }
            for (int i = 0; i < linger_n; ++i) {
                const double left = (linger[i].deadline - now) * 1000.0;
                if (left < timeout) timeout = left > 0.0 ? (int)left : 0;
            }
            if (timeout < 0) timeout = 0;
        }

        const int ready = poll(pfd, (nfds_t)nf, timeout);

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
            dump_table(w, workers, window_dispatched, window_refused, q_n,
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
            window_dispatched = 0;
            memset(window_refused, 0, sizeof(window_refused));
            window_start = prev;
        }

        if (ready < 0) {
            if (errno == EINTR) continue;
            perror("prefork: poll");
            break;
        }

        const double now = prev;

        /* ---- lingering closes, first: they free descriptors ---- */
        for (int i = linger_n - 1; i >= 0; --i) {
            const int k = l_first + i;
            const int fired = (k < nf) && (pfd[k].revents != 0);
            if (!fired && now < linger[i].deadline) continue;
            if (linger_step(&linger[i], now)) {
                linger[i] = linger[linger_n - 1];
                --linger_n;
            }
        }

        /* ---- completions ---- */
        for (int k = 0; k < worker_slots; ++k) {
            if ((pfd[k].revents & (POLLIN | POLLHUP | POLLERR)) == 0) continue;
            const int i = map[k];
            char buf[256];
            ssize_t n;
            do { n = read(w[i].chan, buf, sizeof(buf)); } while (n < 0 && errno == EINTR);
            if (n > 0) {
                w[i].completed += n;
                w[i].window_completed += n;
                w[i].active -= (int)n;
                disp_pop(&w[i], (int)n);
                if (w[i].active < 0) {
                    /* Would mean a worker reported a connection it was never
                     * given. Clamp, but say so: it is an accounting bug and it
                     * silently inflates capacity. */
                    fprintf(stderr, "prefork: worker %d reported more completions than "
                                    "dispatches; slot accounting is wrong\n", i);
                    w[i].active = 0;
                    w[i].disp_n = 0;
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
                w[i].disp_n = 0;
                --live;
            }
        }

        /* ---- RUNG 4's watchdog: outstanding connections past the cap ----
         * The parent cannot refuse these; the descriptor has belonged to the
         * worker since the handoff. What it can do is make the breach VISIBLE,
         * which is the difference between "the server felt slow" and a counter
         * an operator can point at. Enforcement -- stopping at the next frame
         * boundary -- belongs to the synthesis loop; see the note on
         * mynah_prefork_service_cap_ms(). */
        if (g_service_cap_ms > 0) {
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid <= 0 || w[i].disp_n <= 0 || w[i].disp == NULL) continue;
                const double age = (now - w[i].disp[w[i].disp_head]) * 1000.0;
                if (age >= (double)g_service_cap_ms) {
                    ++w[i].over_cap;
                    ++refused[MYNAH_PREFORK_REFUSE_SERVICE_CAP];
                    ++window_refused[MYNAH_PREFORK_REFUSE_SERVICE_CAP];
                    /* Charge it once: re-stamp the head so the same connection
                     * is not counted again on every tick until it finishes. */
                    w[i].disp[w[i].disp_head] = now;
                    fprintf(stderr, "prefork: worker %d has a connection %.0f ms into "
                                    "service, past the %d ms cap\n",
                            i, age, g_service_cap_ms);
                }
            }
        }

        /* ---- queued clients that hung up while waiting ----
         * Not a refusal: nobody is owed a status. Counted separately so it can
         * never be mistaken for one. */
        for (int i = q_watch - 1; i >= 0; --i) {
            const int k = q_first + i;
            if (k >= nf) continue;
            if ((pfd[k].revents & (POLLERR | POLLHUP | POLLNVAL)) == 0) continue;
            close(q[i].fd);
            ++client_gone;
            memmove(&q[i], &q[i + 1], (size_t)(q_n - i - 1) * sizeof(q[0]));
            --q_n;
        }

        /* ---- RUNG 3, then RUNG 1: drain the queue ----
         * The deadline is checked HERE, at the head, at pop time -- never at
         * push. Checking at push can only refuse on a prediction about a wait
         * that has not happened; checking at pop refuses on a fact. An entry
         * that waited and then got a slot anyway is served, which is the whole
         * reason a queue is better than an immediate refusal. */
        while (q_n > 0) {
            if (deadline_ms > 0 &&
                (now - q[0].enqueued) * 1000.0 >= (double)deadline_ms) {
                ++refused[MYNAH_PREFORK_REFUSE_QUEUED_TOO_LONG];
                ++window_refused[MYNAH_PREFORK_REFUSE_QUEUED_TOO_LONG];
                PF_REFUSE_PARK(q[0].fd, MYNAH_PREFORK_REFUSE_QUEUED_TOO_LONG, now);
                memmove(&q[0], &q[1], (size_t)(q_n - 1) * sizeof(q[0]));
                --q_n;
                continue;
            }
            int best = -1;
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid <= 0 || w[i].active >= slots) continue;
                if (best < 0 || w[i].active < w[best].active) best = i;
            }
            if (best < 0) break;                 /* no slot: it stays queued */

            const int fd = q[0].fd;
            memmove(&q[0], &q[1], (size_t)(q_n - 1) * sizeof(q[0]));
            --q_n;
            /* The descriptor was made non-blocking by nothing so far, but it
             * may have been accepted from a non-blocking listener; server/main.c
             * clears O_NONBLOCK on arrival either way. */
            if (send_fd(w[best].chan, fd) != 0) {
                fprintf(stderr, "prefork: handing fd to worker %d failed: %s\n",
                        best, strerror(errno));
                ++refused[MYNAH_PREFORK_REFUSE_HANDOFF_FAILED];
                ++window_refused[MYNAH_PREFORK_REFUSE_HANDOFF_FAILED];
                PF_REFUSE_PARK(fd, MYNAH_PREFORK_REFUSE_HANDOFF_FAILED, now);
                continue;
            }
            close(fd);
            ++w[best].active;
            ++w[best].assigned;
            ++w[best].window_assigned;
            disp_push(&w[best], now);
            ++dispatched;
            ++window_dispatched;
        }

        if (listen_slot < 0 || listen_slot >= nf ||
            (pfd[listen_slot].revents & POLLIN) == 0) continue;

        const int cfd = accept(local.listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
                errno == ECONNABORTED) continue;
            perror("prefork: accept");
            break;
        }

        /* RUNG 1: the least-loaded worker with a free slot. A new arrival may
         * only go straight through when the queue is EMPTY -- otherwise it
         * would jump ahead of entries that have already waited, and rung 3's
         * deadline would start firing on requests that were merely unlucky
         * rather than late. */
        int best = -1;
        if (q_n == 0) {
            for (int i = 0; i < workers; ++i) {
                if (w[i].pid <= 0 || w[i].active >= slots) continue;
                if (best < 0 || w[i].active < w[best].active) best = i;
            }
        }
        if (best >= 0) {
            if (send_fd(w[best].chan, cfd) != 0) {
                fprintf(stderr, "prefork: handing fd to worker %d failed: %s\n",
                        best, strerror(errno));
                ++refused[MYNAH_PREFORK_REFUSE_HANDOFF_FAILED];
                ++window_refused[MYNAH_PREFORK_REFUSE_HANDOFF_FAILED];
                PF_REFUSE_PARK(cfd, MYNAH_PREFORK_REFUSE_HANDOFF_FAILED, now);
                continue;
            }
            /* Our copy goes now: from here the worker is the only owner, and
             * the client sees a close only when the worker closes. */
            close(cfd);
            ++w[best].active;
            ++w[best].assigned;
            ++w[best].window_assigned;
            disp_push(&w[best], now);
            ++dispatched;
            ++window_dispatched;
            continue;
        }

        /* RUNG 2: no slot. Park it if the queue has room. The arithmetic is
         * the reference's -- running + queued >= slots + queue_cap refuses --
         * evaluated over the parent's own counters, which is possible here
         * because a queued descriptor has not been charged to any worker and
         * therefore cannot consume the capacity it is waiting for. */
        /* Recomputed from LIVE workers, not from the W we started with. A
         * fleet that has lost a worker has lost the slots behind those queue
         * entries too, and a bound that keeps promising capacity the machine
         * no longer has is how a degraded server turns a refusal into a wait. */
        const int q_bound = (q_per < 0) ? -1 : q_per * live;
        if (q_bound != 0 && (q_bound < 0 || q_n < q_bound)) {
            if (q_n >= q_alloc) {
                const int grown = q_alloc * 2;
                pf_queued *bigger = (pf_queued *)realloc(q, (size_t)grown * sizeof(*q));
                if (bigger == NULL) {
                    ++refused[MYNAH_PREFORK_REFUSE_AT_CAPACITY];
                    ++window_refused[MYNAH_PREFORK_REFUSE_AT_CAPACITY];
                    PF_REFUSE_PARK(cfd, MYNAH_PREFORK_REFUSE_AT_CAPACITY, now);
                    continue;
                }
                q = bigger;
                q_alloc = grown;
            }
            q[q_n].fd = cfd;
            q[q_n].enqueued = now;
            ++q_n;
            ++queued_total;
            if (q_n > queue_peak) queue_peak = q_n;
            continue;
        }

        /* RUNGS 1+2 exhausted: every slot busy and the queue full. Refuse now,
         * with a reason and a counter of its own, and -- this is the part the
         * reference still has open -- refuse in a way the client can actually
         * read. See the RST note in prefork.h. */
        ++refused[MYNAH_PREFORK_REFUSE_AT_CAPACITY];
        ++window_refused[MYNAH_PREFORK_REFUSE_AT_CAPACITY];
        PF_REFUSE_PARK(cfd, MYNAH_PREFORK_REFUSE_AT_CAPACITY, now);
    }

    /* ------------------------------------------------------------- shutdown */
    /* Anything still queued was accepted and never answered. Tell it so rather
     * than dropping it: a client that gets a 503 at shutdown knows to retry
     * elsewhere, and one whose socket simply dies does not. */
    for (int i = 0; i < q_n; ++i) {
        mynah_prefork_refuse_and_close(q[i].fd, MYNAH_PREFORK_REFUSE_AT_CAPACITY);
        ++refused[MYNAH_PREFORK_REFUSE_AT_CAPACITY];
    }
    q_n = 0;
    /* Finish the lingering closes properly: the whole point is that the client
     * reads the status, and abandoning them here would reintroduce the RST at
     * exactly the moment an operator is most likely to be watching. */
    for (int i = 0; i < linger_n; ++i) {
        const double end = mono_seconds() + PF_LINGER_MS / 1000.0;
        for (;;) {
            const double t = mono_seconds();
            if (linger_step(&linger[i], t)) break;
            if (t >= end) { if (linger[i].fd >= 0) close(linger[i].fd); break; }
            struct pollfd one;
            one.fd = linger[i].fd;
            one.events = linger_events(&linger[i]);
            one.revents = 0;
            if (poll(&one, 1, 20) < 0 && errno != EINTR) {
                close(linger[i].fd);
                break;
            }
        }
    }
    linger_n = 0;

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

    /* Totals for the run, not for the window since the last dump. Each rung
     * gets its own line: "503 x 900" tells an operator nothing, while
     * "at_capacity 12, queued_too_long 888" says immediately that the queue
     * deadline is the thing to look at. */
    fprintf(stderr, "prefork: final  dispatched=%lld queued_total=%lld "
                    "queue_peak=%d client_gone=%lld\n",
            dispatched, queued_total, queue_peak, client_gone);
    long long refused_all = 0;
    for (int r = 0; r < MYNAH_PREFORK_REFUSE__COUNT; ++r) refused_all += refused[r];
    fprintf(stderr, "prefork: refused=%lld", refused_all);
    for (int r = 0; r < MYNAH_PREFORK_REFUSE__COUNT; ++r) {
        fprintf(stderr, "  %s=%lld", REFUSAL[r].code, refused[r]);
    }
    fprintf(stderr, "\n");
    if (linger_forced > 0) {
        fprintf(stderr, "prefork: WARNING %lld refusals were closed without a full "
                        "lingering close (the %d-slot set was full); those clients "
                        "may have seen a reset instead of the status\n",
                linger_forced, PF_LINGER_MAX);
    }
    /* A leftover `still-in-flight` here is the interesting number: it means a
     * worker was handed a connection it never reported finished, which is a
     * leaked slot. */
    for (int i = 0; i < workers; ++i) {
        fprintf(stderr, "  worker %d: assigned=%lld completed=%lld still-in-flight=%d"
                        " over-service-cap=%lld\n",
                i, w[i].assigned, w[i].completed, w[i].active, w[i].over_cap);
        free(w[i].disp);
    }
    close(local.listen_fd);
    free(pfd);
    free(q);
    free(w);
    return MYNAH_PREFORK_PARENT_DONE;
}

#undef PF_REFUSE_PARK
