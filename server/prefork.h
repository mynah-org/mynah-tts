/* Prefork serving topology: one listening parent that routes, W worker
 * processes that synthesize.
 *
 * WHY A SECOND TOPOLOGY EXISTS AT ALL. The in-process server (server/main.c)
 * scales a single model's decode step across cores with the shared thread
 * pool, and continuous batching amortises the weight read across the requests
 * in flight. Both stop paying somewhere well below 32 cores: a decode step is
 * DRAM-bandwidth bound, so widening one step past the point where it saturates
 * the memory path adds barrier cost and no throughput. Past that point the
 * only thing left to sell is *independent* work -- several narrower batches,
 * each on its own slice of cores, each reading the weights for its own slice.
 * That is what this file is. It is a throughput structure, not a latency one.
 *
 * WHAT IS PORTABLE AND WHAT IS NOT, stated up front because the gate depends
 * on it. Everything here -- forking after the model is resident, passing the
 * accepted descriptor over a socketpair with SCM_RIGHTS, least-loaded routing,
 * the per-worker slot accounting, the shutdown sequence -- is plain POSIX and
 * is compiled and exercised on macOS as well as Linux. Exactly two things are
 * Linux-only and behind `#if defined(__linux__)`:
 *
 *   - `sched_setaffinity`, the pinning itself;
 *   - the sysfs topology read (`/sys/devices/system/cpu/N/topology/`) that
 *     decides the core-major order and reports SMT.
 *
 * On a platform without them this runs UNPINNED and says so once, loudly, in
 * the banner. Unpinned prefork is still useful (it is how the descriptor
 * handoff is tested here) but it is not the configuration the throughput claim
 * is about: without pinning the scheduler migrates workers across the machine
 * and two workers routinely land on one core.
 *
 * ---------------------------------------------------------------------------
 * HOW TO CHOOSE W -- the procedure, because the number is a property of the
 * machine and of the model, and anyone who writes a constant here is guessing.
 * ---------------------------------------------------------------------------
 *
 * The knob is really two knobs that multiply to the machine: W workers by T
 * threads each, with W*T = the cores you are allowed to use. On 32 cores,
 * 8x4 and 16x2 and 4x8 are all fully subscribed and they are not the same
 * server. What separates them:
 *
 *   - T too large: the decode step saturates memory bandwidth before it runs
 *     out of threads, and the extra threads only lengthen the barrier to the
 *     slowest one. Symptom: per-worker RTF barely improves from T to 2T.
 *   - T too small: each worker's batch is narrow, so the weight read is
 *     amortised over fewer requests, and the same bytes are pulled from DRAM W
 *     times per step instead of W/2. Symptom: aggregate throughput falls while
 *     each individual stream looks fine.
 *   - W too large: W copies of every per-process cache. The memory-mapped pack
 *     is shared (see below), but the quantized-weight and codec-filter caches
 *     each worker builds on its first synthesis are NOT. Symptom: RSS grows
 *     roughly linearly in W and the machine starts swapping at some W you did
 *     not predict.
 *
 * So measure. The procedure, in order, and each step answers one question:
 *
 * STEP 0 -- know the machine. `--prefork-plan` prints the logical cpu count,
 *   the physical core count, the SMT factor and the inherited cpu mask, then
 *   the sweep below with those numbers substituted. Run it first. If SMT is on
 *   and you do not intend to measure SMT, turn it off on the host or plan on
 *   physical cores only: two workers sharing one physical core look isolated
 *   in `top` and are not.
 *
 * STEP 1 -- find T*, the width where one worker stops improving. Single
 *   worker, no prefork, sweep MYNAH_THREADS over 1,2,4,8,16,32 and record RTF
 *   at concurrency 1 and at concurrency equal to --max-batch:
 *
 *     for t in 1 2 4 8 16 32; do
 *       MYNAH_THREADS=$t make serving-profile MODEL_DIR=... ARGS="--levels 1,4"
 *     done
 *
 *   T* is the smallest T within ~10% of the best RTF. It is usually much
 *   smaller than the core count, and it is the single most useful number in
 *   this procedure: everything below is a sweep around W = cores / T*.
 *
 * STEP 2 -- sweep W at fixed total subscription. With C allowed cores, run
 *   W in {C/(2 T*), C/T*, 2C/T*} with T = C/W each time, so the machine stays
 *   exactly subscribed:
 *
 *     for w in 4 8 16; do
 *       ./mynah-tts-server -m PACK --prefork $w --prefork-threads $((C/w)) &
 *       make serving-profile MODEL_DIR=PACK ARGS="--levels $((w)),$((2*w)),$((4*w))"
 *     done
 *
 *   Report per level: completed/launched, TTFA p50 and p95, STREAM_RTF p50 and
 *   p95, stall rate, and peak RSS of the whole process tree. The winner is the
 *   largest concurrency level that is still GOOD, not the best median.
 *
 * STEP 3 -- find the cliff. For the best (W,T), raise concurrency until the
 *   verdict stops being GOOD. That concurrency is the machine's answer, and it
 *   is the only number worth publishing. Publish it with W, T, the cpu mask,
 *   the model revision and the ISA beside it, because it transfers to no other
 *   machine.
 *
 * STEP 4 -- check the memory you actually bought. `ps -o rss= -p <parent> and
 *   each child` after the sweep, not before: the caches are lazy. If total RSS
 *   is more than about (pack size + W * per-worker cache), a cache you thought
 *   was shared is not.
 *
 * A note on what the fork does and does not share. The children are forked
 * AFTER the pack is open, so the memory-mapped weights are one physical copy
 * for the whole tree. They are forked BEFORE any synthesis, which is required
 * for correctness (a fork from a thread other than main, or after a synthesis
 * has started, inherits locked mutexes) and which also means the lazily-built
 * quantized and codec caches are per worker. Do not "fix" that by prewarming
 * in the parent: the codec's BNNS filter cache is keyed by `pthread_t` and is
 * never pruned, so filters built by the parent's pool threads would be dead
 * entries inherited by every child and freed only at model close.
 */
#ifndef MYNAH_SERVER_PREFORK_H
#define MYNAH_SERVER_PREFORK_H

#include <signal.h>
#include <stdio.h>

typedef enum {
    MYNAH_PREFORK_DISABLED,     /* not requested: the caller serves in-process */
    MYNAH_PREFORK_CHILD,        /* we are a worker: serve from *chan_fd */
    MYNAH_PREFORK_PARENT_DONE,  /* the router ran to completion: exit */
    MYNAH_PREFORK_ERROR         /* could not set up: the caller must fail */
} mynah_prefork_role;

typedef struct {
    int listen_fd;      /* already bound and listening; the parent keeps it */
    int workers;        /* W; <= 0 asks for a plan based on the cpu mask */
    int threads_per;    /* T; <= 0 derives it from W and the cpu mask */
    int slots_per;      /* requests in flight per worker: the child's max_batch */
    int quiet;          /* suppress the banner (tests) */
} mynah_prefork_config;

/* Prints the topology this machine offers and the sweep that turns it into a
 * choice of W. Never forks; safe before the model is open. */
void mynah_prefork_print_plan(const mynah_prefork_config *cfg, FILE *out);

/* Resolves `threads_per` from `workers` and the cpu mask this process is
 * ALLOWED to use, and publishes it as MYNAH_THREADS so the shared pool -- whose
 * width is resolved once, on first use, and cached for the life of the process
 * -- picks it up.
 *
 * MUST be called before the model is opened, and that is the whole reason it
 * is a separate call rather than part of mynah_prefork_run(): by the time the
 * children exist the pool width may already have been resolved from the
 * parent's environment, and a setenv() in the child would then be a silent
 * no-op. A child re-applies and VERIFIES it anyway, and complains if the value
 * in force is not the one planned. */
void mynah_prefork_reserve_threads(mynah_prefork_config *cfg);

/* Forks W workers, then routes. MUST be called from the main thread, with the
 * model already open and no synthesis yet performed, and before any other
 * thread in this process exists.
 *
 * In a child: closes the listening socket, pins to its core slice, rebuilds
 * the thread pool, publishes its worker index, and returns MYNAH_PREFORK_CHILD
 * with *chan_fd set to its end of the socketpair.
 *
 * In the parent: runs the routing loop until *stop is set or every worker is
 * gone, reaps the children, prints the final per-worker table and returns
 * MYNAH_PREFORK_PARENT_DONE. The parent never synthesizes and never parses
 * HTTP; the only thing it writes to a client socket is the 503 it sends when
 * every worker is full. */
mynah_prefork_role mynah_prefork_run(const mynah_prefork_config *cfg,
                                     volatile sig_atomic_t *stop,
                                     int *chan_fd);

/* This worker's index, or -1 when not a prefork child. Used by /health so a
 * caller can see which worker answered it. */
int mynah_prefork_worker_index(void);

/* The threads-per-worker actually in force, or 0 outside a worker. */
int mynah_prefork_worker_threads(void);

/* Exactly once per connection the worker was handed, when it is finished with
 * it. This is the parent's only view of how loaded a worker is, so a path that
 * forgets it leaks a slot and a path that calls it twice over-admits. No-op
 * outside a prefork worker. */
void mynah_prefork_conn_done(void);

/* Waits up to timeout_ms for the parent to hand over a client descriptor.
 * Returns the descriptor, -1 when the channel is gone (the parent exited and
 * the worker must shut down), or -2 on timeout/interruption: retry. */
int mynah_prefork_recv_conn(int chan_fd, int timeout_ms);

/* Non-zero once since the last call if SIGUSR1 asked for a statistics dump.
 * Both the parent's router loop and a worker's accept loop poll this. */
int mynah_prefork_take_dump_request(void);

#endif
