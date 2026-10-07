# L40S ROW_CAP=1024: the "silent" scheduler exit at C1024

Status: read-only diagnosis, 2026-10-06. No source changed. Line numbers are
from the worktree at the time of writing (`src/inference.c`, `server/main.c`).

## TL;DR

The driver did not stop on its own. **The accept loop in `main()` exited
silently, most likely on `accept()` returning `EMFILE` (too many open files),
and the normal shutdown path then stopped the scheduler.** Nothing in the
server raises `RLIMIT_NOFILE`, the default soft limit on Linux is 1024, and
C1024 closed-loop clients need 1024 client sockets plus the listener, stdio
and the CUDA driver's descriptors. C896 (~900 + ~40 baseline) fits under 1024.
C1024 does not. That explains why the failure shows up only at the full row
capacity, with or without L13. It has nothing to do with ROW_CAP itself.

## Why it is not the driver

`serve()` (`src/inference.c:1521`) has exactly one exit after start-up:
`if (used == 0u) break;` at **inference.c:1868**, right after `admit_pass`.
`used` can reach 0 with requests still queued in only three ways:

1. `*drained` set at **inference.c:1304**. That needs `next_job(block=1)` to
   return 0. `sink_next_job` (**main.c:657-663**) blocks until a job arrives or
   `g_batch.stop` is set, so this happens only after `g_batch.stop`.
2. `sink->running()` returns 0 (**inference.c:1278**). `sink_running`
   (**main.c:872-875**) reads `g_batch.stopping`, the same flag.
3. `admit_cap` (`MYNAH_ADMIT_PER_ITER`) is reached while every admission of
   the iteration failed `slot_start` (see candidate 3 below).

`scheduler_main` (**main.c:905-914**) tells these apart. If the driver returns
while `g_batch.stopping == 0`, it prints **"the synthesis driver stopped on its
own; queued requests will be refused"**. That line has been there since
39b5d21 (2026-09-12) and **it is not in the log**. So `stopping` was already 1
when the driver returned. Only two places set it:

- **main.c:3592-3595**, the shutdown sequence after the accept loop exits,
- **main.c:911-913**, which needs the driver to return first, so it cannot be
  the cause.

The rest of the symptom matches the shutdown path:

- Workers are told to stop (**main.c:3586-3590**). `queue_pop`
  (**main.c:2511-2525**) still hands out the connections already parked, so
  workers parse them and `job_enqueue` accepts them, because `g_batch.stop` is
  not set yet.
- `stop`/`stopping` are then set (**main.c:3592-3596**). `admit_pass` stops
  admitting, the in-flight rows finish, `used` reaches 0, and `serve()` prints
  the normal `[SERVE]` summary (**inference.c:2023-2110**). This is the
  "silent exit" with a summary.
- `drain_pending_jobs()` (**main.c:3600**, body **920-937**) answers every job
  still queued with `503 "server is shutting down"`. This string appears
  nowhere else. `job_enqueue` refusals reach the client as a different 503.
- While the batch drains, new connections pile up in the listen backlog
  (`accept_fd` is closed only at **main.c:3608**), so the port still looks open.

## Ranked candidates

### 1. (Most likely) Accept loop `break` on `accept()` EMFILE/ENFILE, no log line

**main.c:3513-3518**:

```c
fd = accept(accept_fd, NULL, NULL);
if (fd < 0) {
    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
        errno == ECONNABORTED) continue;
    break;                       /* <-- silent; EMFILE lands here */
}
```

- `grep -rn 'rlimit\|RLIMIT\|EMFILE' server src` finds nothing. The server
  never raises the limit. `tools/gpu/serve.sh:13` execs the server without
  `ulimit -n`.
- The descriptors in use are 1 socket per in-flight request (streaming or
  not), plus the listener, stdio, `/dev/nvidia*` and `nvidia-uvm` (typically a
  few dozen per CUDA context), and log files. Closed-loop clients also overlap
  briefly: the old socket is still open in the writer while the next request
  connects. At C896 the count is about 940, below 1024. At C1024 it is above
  1024 the moment the level ramps up, which matches "at the start of C1024".
- It also matches yesterday's run without the new flags (same exit between
  C896 and C1024) and is independent of L13, the slot pool and width buckets.
- No `select()`/`FD_SET` is used anywhere (all `poll`), so raising the limit
  has no FD_SETSIZE hazard.

### 2. Accept loop `break` on a `poll()` error (main.c:3506-3510)

Also silent. Only `EINTR` is retried. `ENOMEM` (or `EINVAL`, which cannot
happen here) would stop the server the same way. This is unlikely, but the
diagnostic below covers it for free.

### 3. Latent driver bug (would NOT be silent): `admit_cap` with all starts failing

When `MYNAH_ADMIT_PER_ITER=N` is set (**inference.c:1718-1723**) and an
iteration starts with `used == 0`, `admit_pass` can admit N jobs that all fail
`slot_start`/`async_submit`. Each one is retired, with `--used` at
**inference.c:1340**. The `iter_admits < admit_cap` guard
(**inference.c:1277**) then ends the pass with `used == 0` and `drained == 0`,
and **inference.c:1868** breaks out of the service for good. `scheduler_main`
would print "stopped on its own", so this is not today's symptom. It is still
a real bug: a burst of invalid requests on an idle server would stop it
permanently. Fix: break at 1868 only when `drained ||
(sink->running && !sink->running(ud))`, otherwise `continue`.

### 4. Checked and ruled out at the full-capacity edge

- `used == slot_capacity`: `admit_pass` just doesn't run
  (**inference.c:1276**). Late admission is guarded the same way
  (**inference.c:1424**). `wait_arrival` is only a probe, and its 0 return
  means "nothing", not "done" (**main.c:717-735**).
- `occ_hist`/`occ_time`/`occ_late`/`occ_worst` are sized `MAX_JOBS + 1` and
  indexed with `live <= max_batch <= ROW_CAP` (**inference.c:1671-1686, 1952,
  1965**). There is no off-by-one.
- `step_ahead.order[MAX_ACTIVE]`, `ctxs/slot[MAX_JOBS]`: `n <= used <=
  slot_capacity`, and `live <= max_batch` (**inference.c:1457-1499**). This is
  in bounds.
- The retire swap-remove (**inference.c:1990-2004**) never touches index
  `used`.
- An engine step error fails rows. It does not end the loop, and `step_live`
  has no return path to `serve`. A memory corruption at width 1024 would crash
  the process, not print a clean `[SERVE]` summary followed by
  `drain_pending_jobs`.
- `--max-inflight` / `--max-pending`: the server only uses them to shed load
  with a 503 (**main.c:491**, **main.c:3559-3577**). Neither stops the
  scheduler.

## Cheapest discriminating experiment

Run the same C768 → C1024 ladder once with `ulimit -n 65536` set before the
server starts (or `prlimit --nofile=65536 --pid <pid>` on the running server).
If C1024 survives, this is the cause. During C896, the following shows the
headroom directly:

```
grep 'open files' /proc/$(pgrep -f mynah-tts-server)/limits
ls /proc/$(pgrep -f mynah-tts-server)/fd | wc -l
```

## Diagnostics to add (one line each)

1. Before the `break` at **main.c:3517**:
   `fprintf(stderr, "accept loop: accept() failed: %s; stopping the server\n", strerror(errno));`
2. Before the `break` at **main.c:3509**:
   `fprintf(stderr, "accept loop: poll() failed: %s; stopping the server\n", strerror(errno));`
3. After the `while` at **main.c:3579**:
   `fprintf(stderr, "accept loop exited (g_shutdown=%d); draining\n", (int)g_shutdown);`
4. At **inference.c:1868**, before the `break`:
   `fprintf(stderr, "driver: exit, used=0 drained=%d running=%d admitted=%zu\n", drained, sink->running ? sink->running(sink->ud) : 1, admitted);`
5. At start-up, near the banner: print the `getrlimit(RLIMIT_NOFILE)` soft and
   hard values, and warn if the soft limit is below
   `max_active + max_pending + CONN_QUEUE_CAP + 64`.

## Likely fix

- **Raise the limit at start-up**: in `main()`, call `getrlimit`, then set the
  soft `RLIMIT_NOFILE` to the hard limit (capped). Log it, and keep the warning
  from diagnostic 5 for when the hard limit is too low. Until this lands, add
  `ulimit -n 65536` to `tools/gpu/serve.sh` and the L40S harness.
- **Never stop the server on a transient `accept()` error**: treat
  `EMFILE/ENFILE/ENOBUFS/ENOMEM` (and Linux's `EPROTO`, `ENETDOWN`,
  `EHOSTUNREACH`, etc., which `accept(2)` says to retry) as "log rate-limited,
  back off ~10-50 ms, `continue`". The backoff matters, because the listener
  stays readable and the loop would otherwise spin. Count these in `/health`
  (`rejected` or a new `accept_errors`). Break only on `EBADF`/`EINVAL`/
  `ENOTSOCK`, and log it.
- Separately, fix candidate 3 so that `serve()` can only end on
  drained/stopping.
