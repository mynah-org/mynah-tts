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
 *
 * ---------------------------------------------------------------------------
 * THE ADMISSION LADDER -- four refusal points, outermost first.
 * ---------------------------------------------------------------------------
 *
 * The shape is copied from a reference implementation that measured each rung,
 * so what follows is the mechanism and not a variation on it. The ordering is
 * the design: each rung refuses a strictly cheaper class of request than the
 * one below it, and every rung refuses with its OWN reason and its OWN counter.
 * "The server was full" and "you waited too long" are different events, they
 * call for different client behaviour, and a stats line that merges them tells
 * an operator nothing.
 *
 *   1. PARENT SLOT CAP. Pick the least-loaded worker with active[w] < cap.
 *      If none has room the request falls to rung 2, and if that is full too
 *      it is refused immediately -- `server_at_capacity`.
 *
 *      THE CRITICAL DETAIL, and the one that cost the reference a whole
 *      measurement campaign: the parent KEEPS POLLING THE LISTENING SOCKET
 *      while every worker is full. The obvious optimisation -- stop watching
 *      the listener until a slot frees, and let the kernel backlog hold the
 *      connection -- is wrong, and wrong in a way no metric can see. A client
 *      parked in the backlog has not been accepted, so it has no queue entry,
 *      no deadline, and no timestamp: it is invisible to every counter in this
 *      file and to the child's queue deadline alike. The reference measured
 *      TTFB/TTFA p95 of 4470/4635 ms with OVER 97% OF THE TAIL LANDING BEFORE
 *      accept(). After the fix the same load gave eight accepted streams at
 *      TTFA 423/562 ms plus two immediate 503s.
 *
 *      A WAIT YOU CANNOT SEE IS WORSE THAN A REFUSAL YOU CAN. That sentence is
 *      the whole reason this rung exists, and it is why the poll set below
 *      always contains cfg->listen_fd -- never conditionally.
 *
 *   2. BOUNDED QUEUE. A request that finds every slot busy is parked in the
 *      parent's admission queue rather than refused outright, because slots
 *      free on a frame boundary and a request refused a millisecond early is
 *      a refusal that did not need to happen. The queue is bounded at
 *      `queue_per_worker` entries per LIVE worker; the reference's default is
 *      1 and so is ours. Past the bound: `server_at_capacity`, immediately.
 *
 *      The bound is not a tuning knob to be raised when refusals appear.
 *      Admission capacity and sustained capacity are different numbers: the
 *      reference raised its cap until C32 was admitted with zero rejects and
 *      then found that nothing above C20 sustained. Disabling the bound is
 *      possible and produces a warning that is deliberately impossible to
 *      miss, because an unbounded queue converts a refusal you can see into
 *      an unbounded wait you cannot -- rung 1's mistake by another route.
 *
 *   3. QUEUE DEADLINE, CHECKED AT POP AND NEVER AT PUSH. An entry older than
 *      `queue_deadline_ms` when it reaches the head is refused with a distinct
 *      reason -- `queued_too_long` -- and that refusal is REAL: it is the
 *      moment the parent discovered the wait, not a deadline synthesized at
 *      push time from a wait that had not happened yet. Checking at push can
 *      only ever refuse on a prediction; checking at pop refuses on a fact.
 *
 *   4. PER-REQUEST SERVICE CAP, stopping at the next frame boundary. A request
 *      admitted in good faith can still turn out to be unservable, and the
 *      place to stop it is a frame boundary -- never mid-frame, because a
 *      truncated frame is a corrupt stream rather than a refusal. The cap
 *      itself travels to the workers through the fork and is readable with
 *      mynah_prefork_service_cap_ms(); the parent additionally watches the
 *      age of each worker's oldest outstanding connection and counts a breach.
 *      See the SERVICE CAP note on that function for what is enforced here and
 *      what still needs a call site in the synthesis loop.
 *
 * AND THE LADDER IS PER LANGUAGE, when there is more than one. Capacity is a
 * property of a language, because a worker holds one pack: an English worker's
 * free slot is of no use to an Italian request. So rung 1 picks the
 * least-loaded worker OF THAT LANGUAGE'S GROUP, rung 2's bound is
 * `queue_per_worker * live workers of that group`, and rung 3's deadline is
 * checked at that group's own head.
 *
 * The queue is one array with a group tag rather than N arrays, and it is
 * scanned in arrival order per group, which gives every group its own FIFO.
 * That detail is load bearing: a single FIFO would let one entry for a
 * saturated language block a ready entry for an idle one behind it -- a fresh
 * starvation channel introduced by the very partitioning that was chosen to
 * make starvation impossible.
 *
 * A request naming a language no group holds is refused at rung 0, before any
 * of this, with `language_not_served` and a 400. See E5-9 and
 * .work/multi-language-serving.md.
 *
 * WHERE THE QUEUE LIVES, and why that differs from the reference. The
 * reference parks rungs 2 and 3 in the CHILD, because in its topology the
 * parent had nowhere to hold an accepted descriptor. Ours holds them in the
 * PARENT, for two reasons that are properties of this code and not
 * preferences: the parent is the only process that knows every worker's load,
 * so it is the only one that can tell "this worker is busy" from "the machine
 * is full"; and a descriptor queued in the parent has not yet been charged to
 * a worker's slot, so a queued request cannot consume the capacity it is
 * waiting for. The admission arithmetic is the reference's unchanged --
 * `running + queued >= slots + queue_cap` refuses -- evaluated over the
 * parent's own per-worker counters. The child keeps its own bounded queue
 * (server/main.c's `--max-pending`); this ladder sits in front of it.
 *
 * WHAT IS DELIBERATELY NOT HERE. Four mechanisms were built, measured and lost
 * in the reference implementation and must not be reintroduced without new
 * numbers: utilization-aware admission (admitting a transient extra stream to
 * a full worker that looks idle -- it broke the four ESTABLISHED streams,
 * stall@250 0% -> 50%), a fairness or lead/credit gate (withholding ready work
 * parked 95.8% of checks and moved STREAM p95 0.838 -> 0.986), cross-worker
 * batching (requests that could batch at B>=3 coincided 1.6% of the time
 * within +-0.25 ms against a 25% bar -- arrivals are not synchronised), and a
 * priority-based prefill helper (TTFA 435 -> 2379 ms; priority is not
 * isolation). The full table is .work/serving-design.md 9. When the machine is
 * full the answer is to REFUSE, not to get clever.
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

/* Why a request was refused. Every rung of the ladder has its own value, its
 * own counter and its own `code` in the JSON body, because an operator reading
 * "503 x 900" cannot act and an operator reading "at_capacity 12,
 * queued_too_long 888" knows immediately that the queue deadline is the thing
 * to look at. Never collapse two of these into one. */
typedef enum {
    /* Rungs 1+2: no worker had a free slot AND the admission queue was full.
     * The machine is full right now; a client should back off and retry. */
    MYNAH_PREFORK_REFUSE_AT_CAPACITY = 0,
    /* Rung 3: admitted to the queue, then found older than the deadline when
     * it reached the head. Distinct from at-capacity on purpose -- this one
     * says the server is not keeping up, not that it is momentarily busy. */
    MYNAH_PREFORK_REFUSE_QUEUED_TOO_LONG,
    /* Rung 4: the request was admitted and served, and its own service time
     * ran past the cap. Not a capacity signal at all: retrying it unchanged
     * will fail the same way. */
    MYNAH_PREFORK_REFUSE_SERVICE_CAP,
    /* Not a rung: the SCM_RIGHTS handoff to the chosen worker failed. Counted
     * apart from at-capacity because it means a worker channel is broken,
     * which is an incident, not load. */
    MYNAH_PREFORK_REFUSE_HANDOFF_FAILED,
    /* Not a rung either, and the only one of these that is NOT a 503: the
     * request named a language this fleet does not hold. The machine is not
     * full and there is nothing to wait for -- no worker has those weights and
     * none will grow them -- so the answer is 400 with
     * `invalid_request_error`, and Retry-After is deliberately absent. Telling
     * a client to come back for a language the server will never serve is how
     * a refusal becomes a retry storm.
     *
     * Appended last, as this enum's contract requires. See E5-9 and
     * .work/multi-language-serving.md for why languages partition the fleet at
     * all: the six PocketTTS language models share nothing, so a batch that
     * mixed them would read the wrong weights for some of its slots. */
    MYNAH_PREFORK_REFUSE_LANGUAGE_NOT_SERVED,
    MYNAH_PREFORK_REFUSE__COUNT
} mynah_prefork_refusal;

/* Sentinel for `queue_per_worker`. Produces an unmissable warning: see the
 * BOUNDED QUEUE rung above for why an unbounded queue is rung 1's mistake by
 * another route. */
#define MYNAH_PREFORK_QUEUE_UNBOUNDED (-1)
/* And its opposite: no queue at all, refuse the instant no slot is free. A
 * distinct sentinel because 0 already means "unset, use the default", and an
 * operator who asks for zero must not silently get one. */
#define MYNAH_PREFORK_QUEUE_NONE      (-2)

typedef struct {
    int listen_fd;      /* already bound and listening; the parent keeps it */
    int workers;        /* W; <= 0 asks for a plan based on the cpu mask */
    int threads_per;    /* T; <= 0 derives it from W and the cpu mask */
    int slots_per;      /* requests in flight per worker: the child's max_batch */
    int quiet;          /* suppress the banner (tests) */

    /* ---- LANGUAGE GROUPS (E5-9) ----
     *
     * The fleet's resident languages, in the order the caller opened their
     * packs; `languages[0]` is the default, used by a request that names none.
     * NULL, or a count below 2, means a single-language fleet and every
     * mechanism below is dormant -- the router is then byte-for-byte the
     * language-agnostic one, which is what keeps the existing path unchanged.
     *
     * With two or more, workers are divided into contiguous groups, one per
     * language, and a worker holds exactly one pack for its life. That is what
     * makes "a batch never mixes languages" a fact about the address space
     * rather than a check somebody has to remember: a batch is drawn from one
     * process's slots, and that process has one model.
     *
     * The strings are borrowed, not copied, and must outlive the call. */
    const char *const *languages;
    int language_count;

    /* ---- the admission ladder. Every one of these is "0 = unset", so a
     * caller that memsets this struct and never hears of the ladder gets the
     * measured defaults, and so that an explicit 0 can still be expressed
     * through the environment (see mynah_prefork_apply_env). ---- */

    /* Rung 2. Queued connections allowed per LIVE worker, on top of its slots.
     * 0 = unset -> MYNAH_PREFORK_QUEUE_DEFAULT (1, the reference's default).
     * MYNAH_PREFORK_QUEUE_UNBOUNDED = no bound, and a shouted warning. */
    int queue_per_worker;
    /* Rung 3. Milliseconds an entry may sit in the queue, measured from accept
     * and checked when it reaches the head. 0 = unset -> the default below. */
    int queue_deadline_ms;
    /* Rung 4. Milliseconds one request may spend in service. 0 = unset -> the
     * default below; a negative value disables the cap entirely. */
    int service_cap_ms;

    /* ---- DECODER LANE (E5-21). 0 = off, and off is the default. ----
     *
     * How many cpus at the TAIL of each worker's slice become a private,
     * pinned decoder team with its own submit lock and a one-unit-per-slot
     * mailbox. The engine pool gets the rest.
     *
     * This is the one place in the tree where it can be turned on, and that is
     * structural rather than tidy: pthreads inherit the creating thread's
     * mask, so the split has to happen after the worker pins itself and before
     * its pool exists, and this file owns both of those moments. A caller that
     * sets it outside prefork gets nothing, because there is no pinned slice
     * to split.
     *
     * The value is NOT clamped into range here. src/threads.c refuses a split
     * whose lane or engine side would be narrower than MYNAH_LANE_MIN_CPUS,
     * and prints why: on a narrow lane the decoder is slower than inline (the
     * reference measured 6+2 at STREAM p95 1.364 against 0.997 at 4+4), so
     * quietly rounding a bad request into a working one would be the exact
     * failure this gate exists to prevent. A refusal leaves the decoder inline,
     * which is the default and is correct.
     *
     * MYNAH_LANE_SPLIT overrides it; --decoder-lane N is the flag. */
    int lane_cpus;

    /* Preconditions (see mynah_prefork_run). Set by a caller that knows it has
     * opened a GPU backend; prefork then refuses rather than forking a wrong
     * answer. The check does not rely on this being set -- a resident CUDA
     * runtime is detected independently -- but a caller that does set it gets
     * the refusal on every backend rather than only the detectable ones. */
    int gpu_backend_open;
} mynah_prefork_config;

#define MYNAH_PREFORK_QUEUE_DEFAULT        1
#define MYNAH_PREFORK_QUEUE_DEADLINE_MS    2000
#define MYNAH_PREFORK_SERVICE_CAP_MS       30000

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
 * MYNAH_PREFORK_PARENT_DONE. The parent never synthesizes, and the only thing
 * it writes to a client socket is a refusal.
 *
 * IT USED TO SAY "and never parses HTTP", and that is now true only with a
 * qualification, so here is the qualification rather than a quietly deleted
 * clause. In a MULTI-LANGUAGE fleet the router has to know a connection's
 * language before it can choose a worker, because the worker is the thing that
 * holds the weights. What it does is bounded and NON-CONSUMING: a MSG_PEEK of
 * at most a few KiB, a search for one JSON key, and a search of the header
 * block for the two header names that announce a body. It never consumes a
 * byte -- the worker still reads the entire request itself -- and it never
 * interprets a method, a status, a framing rule or a body's meaning. A
 * connection whose prefix has not arrived yet is PARKED with its own deadline,
 * never waited on, because a router that blocks on a slow client is rung 1's
 * disease under another name.
 *
 * In a single-language fleet none of that code runs at all. That is not an
 * optimisation; it is the guarantee that adding languages did not change the
 * server everybody already has. */
mynah_prefork_role mynah_prefork_run(const mynah_prefork_config *cfg,
                                     volatile sig_atomic_t *stop,
                                     int *chan_fd);

/* This worker's index, or -1 when not a prefork child. Used by /health so a
 * caller can see which worker answered it. */
int mynah_prefork_worker_index(void);

/* The threads-per-worker actually in force, or 0 outside a worker. */
int mynah_prefork_worker_threads(void);

/* ------------------------------------------------------- language groups */

/* Which language group this worker was assigned -- an index into the
 * `languages` array the parent was given. 0 outside a prefork worker and in a
 * single-language fleet, which is why a caller with one pack can read it
 * unconditionally and always get the one pack it has.
 *
 * This is the ONLY thing that tells a worker which pack is its own, and it is
 * resolved before the fork so a worker can never disagree with the router
 * about what it holds. */
int mynah_prefork_worker_language(void);

/* The fleet's capacity split as one printable line -- "english=2 italian=2" --
 * or "" when the fleet holds one language. Resolved before the fork and
 * inherited, so a WORKER can print the whole fleet's shape in its own banner
 * and in /health even though it holds one pack. Config that is printed is
 * config that can be argued with; config that is assumed has been wrong twice
 * in this repository already. */
const char *mynah_prefork_language_plan(void);

/* Resolves a requested language name against a set of resident ones. Returns
 * the index, -1 when nothing matches, or -2 when the name is AMBIGUOUS.
 *
 * The rule, in one place because the router and the worker must not drift:
 * case-insensitive exact match first; failing that, a case-insensitive prefix
 * of at least two characters that matches exactly one resident name. So `it`
 * and `Italian` both reach a pack that calls itself `italian`, which matters
 * because PocketTTS packs spell languages out while Magpie packs use ISO
 * codes, and a client should not have to know which spelling a pack chose.
 *
 * Ambiguity is refused rather than resolved: picking one of two plausible
 * languages is a wrong answer, and a wrong answer here is a whole utterance in
 * the wrong voice. `want` of NULL or "" returns -1, meaning "unspecified" --
 * the caller decides that means the default, not that it is an error. */
int mynah_prefork_language_match(const char *const *names, int count,
                                 const char *want);

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

/* ------------------------------------------------------- the refusal itself
 *
 * REFUSING WITHOUT AN RST. This is the transport bug the reference still has
 * open, and it turns every fast-fail into a lie: a 503 that is written and
 * then close()d while the client's request body is still sitting unread in the
 * socket's receive queue makes the kernel -- Linux and BSD both -- send an RST
 * instead of a FIN. The RST discards the send buffer, so the client's next
 * read fails with ECONNRESET and curl reports "Recv failure: Connection reset
 * by peer". The status line the server carefully assembled is never seen. The
 * caller ends up debugging a broken pipe while the server's logs happily count
 * a refusal it believes it delivered.
 *
 * The fix is the ordinary lingering close, and all three steps are load
 * bearing: write the response, shutdown(fd, SHUT_WR) so the peer gets a clean
 * FIN and knows the response is complete, then DRAIN what the client is still
 * sending -- bounded by bytes and by time, since a hostile or merely large
 * body must not become a way to occupy the router -- and only then close().
 *
 * Bounded is the word that matters. The parent's routing loop is single
 * threaded: a blocking drain in it would stall every other client, which is
 * the same disease as rung 1. So this call never blocks. It does as much as
 * can be done without blocking and hands whatever is left to the parent's poll
 * set, which finishes the close in the background.
 *
 * server/main.c's own shed path (`queue_push` failing -> write(busy) ->
 * conn_close) has this exact bug and should call this instead.
 *
 * Takes ownership of `fd`: the caller must not close or use it afterwards. */
void mynah_prefork_refuse_and_close(int fd, mynah_prefork_refusal reason);

/* The stable, machine-readable token for a reason -- "server_at_capacity",
 * "queued_too_long", "service_cap_exceeded", "handoff_failed". It appears as
 * `error.code` in the JSON body and as the counter name in the stats line, so
 * a client matching on it and an operator reading a dump are looking at the
 * same string. Append-only: never renumber and never rename. */
const char *mynah_prefork_refusal_code(mynah_prefork_refusal reason);

/* The full HTTP refusal for a reason, including headers and body, written into
 * `buf`. Returns the length, or 0 if `cap` is too small. Content-Length is
 * computed from the body rather than written as a literal, so an edit to the
 * message cannot silently truncate the response. */
size_t mynah_prefork_refusal_response(mynah_prefork_refusal reason,
                                      char *buf, size_t cap);

/* SERVICE CAP -- rung 4, and an honest account of what is enforced where.
 *
 * Returns the per-request service cap in milliseconds that is in force for
 * this process, or 0 when there is none. The value is resolved in the parent
 * before the fork and inherited by every worker, so a worker never has to be
 * told it and can never disagree with the parent about it.
 *
 * What the PARENT enforces: it tracks the age of each worker's oldest
 * outstanding connection and counts a breach against
 * MYNAH_PREFORK_REFUSE_SERVICE_CAP, which is what makes the cap visible in the
 * dump and the final table. It cannot refuse on the client's behalf -- the
 * descriptor belongs to the worker from the moment it is handed over.
 *
 * What the WORKER must do, and what is still missing: the synthesis loop has
 * to compare elapsed service time against this value AT A FRAME BOUNDARY and
 * stop there. A frame boundary specifically: stopping mid-frame truncates a
 * codec frame and yields a corrupt stream, which is a worse outcome than the
 * overrun. That call site is in the slot driver, not in this file. Until it
 * exists the cap is observed and reported but not enforced, and this comment
 * is the record of that gap rather than a claim that it is closed. */
int mynah_prefork_service_cap_ms(void);

/* Applies the MYNAH_PREFORK_* environment overrides to `cfg` and validates it,
 * shouting about anything dangerous. Called by mynah_prefork_run() and by
 * mynah_prefork_print_plan(); exposed so a caller can resolve the ladder
 * before printing its own banner.
 *
 * The environment is the channel because the ladder has no command-line flags
 * yet: those belong to server/main.c's argument parser.
 *
 *   MYNAH_PREFORK_QUEUE       rung 2, queued connections per worker.
 *                             "0" really means zero -- refuse the instant no
 *                             slot is free. "unbounded" or a negative number
 *                             removes the bound and warns loudly.
 *   MYNAH_PREFORK_QUEUE_MS    rung 3, the queue deadline. 0 disables it.
 *   MYNAH_PREFORK_SERVICE_MS  rung 4, the service cap. 0 disables it.
 *   MYNAH_LANE_SPLIT          E5-21, cpus at the tail of each worker's slice
 *                             that become the private pinned decoder team.
 *                             0 (the default) leaves the decoder inline. */
void mynah_prefork_apply_env(mynah_prefork_config *cfg);

#endif
