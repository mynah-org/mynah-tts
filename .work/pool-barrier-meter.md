# E9-P — the pool's own overhead: count it, then cut it

Status: **DONE 2026-09-13** for the count and the pool-side levers; E9-P3 is
implemented, measured and left OFF pending an x86 run. Pinned to `1a93cba`.
Owner files: `src/threads.c`, `src/threads.h`, `src/dispatch.c`, `src/dispatch.h`.

## Problem

The first serving sweep on 32 Neoverse-V2 at C48 found narrow workers beating
wide ones **monotonically**, and wide ones collapsing outright:

| topology | STREAM_RTF p95 | verdict |
|---|---|---|
| 16 workers x 2 threads | 0.736 | best |
| 8 x 4 | 0.931 | |
| 2 x 16 | 1.574 | every frame stalled |
| 1 x 32 | — | 48 of 90 requests completed |

The explanation on the table was that a 109M-parameter model has regions too
short to amortise a wide barrier, and that — unlike weight traffic — that cost
does not shrink with the model, so a small engine pays it *more* often per
second of audio. Our own earlier note put the pool's wake-up at ~20 us per
region, implying a region must be worth ~200 us to pay for its own dispatch.

**That was a hypothesis with a strong prior and no count.** The pool's existing
counters (`dispatches`, `serial`, `inline_fallbacks`, `helper_joins`) count
regions; they cannot say how many regions a frame costs, how wide a region
actually ran, or how much of the wall went into the barrier instead of
arithmetic. The reference engine counts ~225 barriers per frame in its decoder
and treats that as the thing to attack. We had never counted ours.

## What was added

### E9-P1, the meter (`MYNAH_POOL_METER=1`)

Inside `mynah_parallel_for()` and `pool_worker()`, so every call site is covered
with no caller change and no call site can opt out:

- regions, split into dispatched / serial / inline-fallback;
- **width requested** (`min(n, width)`) vs **width entered** vs **width useful**
  (ran >= 1 chunk). Entered-minus-useful is the wake-up that bought nothing —
  the hypothesis stated as a subtraction;
- region wall, the submitter's **barrier** wall, and each worker's **park** vs
  **work** wall;
- a log2 histogram of region durations and the count below the 200 us
  break-even (`MYNAH_POOL_BREAK_EVEN_US`);
- the same, **per call site**, keyed on the `fn` pointer and printed as a delta
  from `mynah_parallel_for` so PIE relocation does not defeat `addr2line`.

Storage is chosen by who writes it: cache-line-padded per-thread slots for the
worker wall (up to 64 writers per wait), relaxed atomics for the per-site table
(only submitters write it, and a process has one or two). Off, the entire cost
is one predicted branch on a cached int per dispatch. The report leaves via
`atexit` and via a chained `SIGTERM` handler installed **only when the meter is
on**, which is what gets numbers out of a prefork worker that is killed rather
than returned from.

### E9-P2, the two levers the count named

- **Pre-check** (`MYNAH_POOL_PRECHECK`, default on). A publish bumps `gen` and
  broadcasts, so every idle worker woke and took the pool mutex to run
  `pick_job()` whatever the region's width. The pre-check is a lock-free scan
  ordered by an **acquire load of `gen` taken before the scan**; the publisher's
  release bump synchronises with it, so a region published before the load is
  visible to the scan and one published after leaves `seen` stale and the wait
  returns at once. A wakeup cannot be lost.
- **Narrowing** (`MYNAH_POOL_NARROW`, default on). A region of `n` chunks admits
  at most `min(n, width) - 1` helpers at a time. It is a performance cap and
  never a correctness one, because the submitter runs its own region to
  completion: **zero helpers is always a correct outcome**, so under-admitting
  can only cost time.

Both required `pf_job.live` and `pf_job.n` to become `_Atomic int`: the
pre-check reads them without the pool mutex, and leaving them plain would be a
data race in the language even though every write is still under the lock.

### The litmus

`mynah_threads_self_test()` hammers 2..5-chunk regions on a much wider pool from
five submitters and asserts every chunk ran **exactly once**. A short form (400
rounds) runs inside the dispatch report as the `pool.litmus` row, so a build that
reports `narrow=on` has actually tested it; the full form (4000 rounds) is
`mynah_threads_self_test()`.

## What was measured

Box: 32 Neoverse-V2, SMT off, gcc 15.2, `SIMD=auto`, pack `models/pocket-en`.

### The count, offline, one request, 73 frames

| profile | threads | regions | **per frame** | mean region | **barrier %** | park % of worker wall | width req -> used | **< 200 us** | synth |
|---|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 1825 | 25.0 | 300.7 us | 0.0 | 0.0 | 1.00 -> 1.00 | 88.0% | 2.88 s |
| f32 | 4 | 5697 | 78.0 | 103.0 us | 3.2 | 54.1 | 4.00 -> 4.00 | 87.6% | 1.30 s |
| f32 | 8 | 5697 | 78.0 | 59.2 us | 8.7 | 70.2 | 8.00 -> 7.83 | 91.4% | 1.07 s |
| f32 | 16 | 5697 | 78.0 | 42.5 us | 22.4 | 82.6 | 15.69 -> 14.24 | 99.4% | 0.99 s |
| f32 | 32 | 5697 | 78.0 | 44.6 us | **35.6** | 90.9 | 27.58 -> **19.01** | 99.5% | 1.07 s |
| int8 | 32 | 5697 | 78.0 | 40.7 us | **39.3** | 92.5 | 27.58 -> 18.39 | 99.4% | 1.05 s |

**78 dispatches per frame of audio**, flat from 4 threads up — the work is cut
into the same number of pieces however many threads there are, so each piece
gets shorter and the fixed per-region cost is paid at the same rate. 32 threads
is *slower* than 16 (1.07 s vs 0.99 s) with barrier at 35.6%.

### The count, at real concurrency (C48, 2 waves, 96 requests)

| topology | STREAM_RTF p95 | verdict | regions | per frame | mean region | **barrier %** | park % | width req -> used | < 200 us |
|---|---|---|---|---|---|---|---|---|---|
| **16 x 2** | **0.723** | PASS | 403899 | 61.8 | **266 us** | **1.2** | 2.4 | 2.00 -> 2.00 | 64.0% |
| 2 x 16 | 1.554 | FAIL, 64/96 | 192067 | 41.0 | 64 us | **16.8** | 36.4 | 15.41 -> 14.70 | 95.7% |
| 1 x 32 | 1.616 | FAIL, 32/96 | 94581 | 41.6 | 60 us | **31.3** | 89.6 | 27.87 -> **19.78** | 98.1% |

**The sweep reproduced and the meter explains it.** The winning topology is the
only one whose mean region (266 us) clears the 200 us break-even, and it spends
**1.2%** of region wall in the barrier. The topology that collapsed spends
**31.3%**, has 98% of its regions below break-even, and wakes 27.9 threads per
region to have 19.8 of them do work — **eight threads woken for nothing, per
region.** Note the wide topologies dispatch *fewer* regions per frame (they batch
more rows per step) and still lose: the tax is per region, not per frame.

### The meter does not move what it measures

Same binary, instrumentation off and on, interleaved, byte-identical output in
every pair (macOS M-series, 8 threads, three rounds: off 0.457/0.446/0.453 s,
on 0.456/0.446/0.449 s — the two arms interleave rather than separate). The
Linux paired arm is in the table above. Off, the meter costs one predicted
branch on a cached `int` per dispatch: no timer is read and no counter outside
the four that already existed is touched.

### Per call site, at 32 threads (same run, same regions, only the width differs)

| call site | regions | mean n | mean us | barrier% | req w -> used w | < 200 us |
|---|---|---|---|---|---|---|
| `qmat_rows_block` (src/qmat.c:1481) | 3240 | 51.4 | **42.1** | **43.1** | 27.29 -> 18.93 | **100%** |
| `sg_task` (src/sgemm.c:670) | 1825 | 43.2 | **41.1** | **36.1** | 29.12 -> 17.08 | **100%** |
| `qmat_batch_block` (src/qmat.c:2345) | 632 | 38.7 | 91.5 | 21.4 | 24.61 -> 24.57 | 94% |

The same three sites at **2 threads** are 119 / 157 / 750 us with **1.2 / 3.3 /
0.5%** barrier and 0-88% below break-even. Same regions, same arithmetic; the
only variable is how finely the work is cut.

## The levers, measured

Paired interleaved, median of within-round ratios, reference = the pool as
`1a93cba` has it. 9-11 rounds each, alternating arms so drift lands on both.

| knob | 2 th | 4 th | 8 th | 16 th | 32 th | default |
|---|---|---|---|---|---|---|
| precheck | — | — | -0.9% | **-2.1%** | **-2.3%** | **ON** |
| narrow | — | — | +1.0% | +0.5% | +0.4% | **OFF** |
| fastexit (E9-P3) | +0.2% | +0.2% | +0.4% | **-1.9%** | **-4.9%** | **OFF** |
| all three | -0.1% | +0.4% | -1.0% | **-3.1%** | **-7.0%** | |

### The spin budget is now OUR number, not a transferred one

Reference = the shipped aarch64 default 65536.

| spin | 2 th | 8 th | 32 th |
|---|---|---|---|
| 0 | +1.8% | +5.8% | +8.1% |
| 256 | +1.2% | +6.5% | +10.6% |
| 1024 | +1.9% | +6.0% | **+18.5%** |
| 4096 | +1.8% | +6.1% | **+16.7%** |
| 16384 | +0.2% | +2.1% | +6.9% |
| **65536** | reference | | |
| 262144 | -0.2% | -0.9% | +0.1% |

65536 is the knee on this box, nothing above it buys anything, and 4096 — the
default this binary uses on every non-aarch64 target — would cost **16.7% at 32
threads**. The transferred value happened to be right here; that is now a
measurement rather than an inheritance.

### E9-P3, and why the largest lever ships OFF

`fastexit` is -4.9% at 32 threads, the biggest single number in this lane. It
is off by default anyway. It is the mirror image of the store-buffer trap this
pool already documents (`.work/linux-production.md` trap 8: peeking at a flag
instead of taking the lock deadlocks on x86 and silently does not on ARM),
production is x86-64 **and** ARM64, and the only box this lane could measure on
is ARM. The ordering is Dekker with `seq_cst` on both halves and is correct by
the standard on any conforming target; what is missing is a run. 120k narrowed
regions under `MYNAH_POOL_SPIN=0` (the setting that maximises the window) did
not hang on ARM. **Flip it after an x86 gate, not before.**

## What the count actually says about the roadmap

The levers in this file are worth 3-7% **at wide pools, and production is
narrow**: at 16x2 all three are within noise, because a 2-wide region is 266 us
and its barrier is 1.2%. So this lane does not move C64 today.

What it does establish is the mechanism and the target. **78 dispatches per
frame** is set by the graph, not by the pool, and at 32 threads every one of
them is under the break-even. The lever that would move the wide topologies is
not in `threads.c` at all: it is fusing the 3240 + 1825 regions per request that
`qmat_rows_block` and `sg_task` dispatch. Until then the count endorses what
production already does — keep the workers narrow.

## What was rejected, and why it is written down

- **`pthread_cond_signal` k times instead of `broadcast`** to wake only the
  helpers a narrow region can use. Rejected for now: the pool's job slots mean
  a signal consumed by a worker that then finds region A drained is a wakeup
  region B does not get, and the failure mode is a stall rather than a wrong
  answer — invisible in a test and expensive in production. The pre-check gets
  most of the same benefit (a woken worker that has nothing to do goes straight
  back to waiting without touching the mutex) with none of that risk.
- **Making the meter a compile-time option.** Rejected because then the
  instrumented binary is not the production binary, and the off/on diff stops
  proving anything about the binary that ships.
- **A caller-side `frame`/`step` marker API.** Rejected as gold-plating: it
  would need every engine to call it, which is a change in files this lane does
  not own, and dividing regions by frames delivered gives the same number from
  data both the CLI and the serving profile already print.

## Acceptance gate

- PocketTTS output **byte-identical to `1a93cba`** under the default and int8
  profiles, `SIMD=auto` and `SIMD=portable`, at 1/2/4/8/16/32 threads.
- `make goldens` (8 checks + batch parity, Magpie byte-identical),
  `--self-test`, `kernels-test`, `qmat-test`, `json-test`, `window-test`,
  `driver-test`, `--pocket-self-check`, UBSan, ASan, macOS `leaks`.
- The `pool.litmus` row PASS under every combination of the knobs, including
  `MYNAH_POOL_SPIN=0` and under both sanitizers.
- No new compiler warning on either SIMD profile (warning sets diffed against
  the base build, not counted).

## Findings handed to other lanes

1. **`mynah_dispatch_self_test()` has no caller anywhere in the repository.**
   `Makefile:745` says `census-test` "exercises it", but `census-test` runs
   `--dispatch-map`, which calls `mynah_dispatch_collect()` only. The function
   — including the census self-test it is documented to carry — has never run
   in a gate. `cli/main.c:523`'s `--self-test` block is the right home; one
   call there would also put this lane's full pool litmus into `make test`,
   `make ubsan` and `make asan`.

2. **Fusion candidates, with the count behind them.** Per request at 32
   threads: `qmat_rows_block` (`src/qmat.c:1481`) dispatches **3240** regions of
   mean 51.4 chunks at 42.1 us, 43.1% of it barrier, 100% below break-even;
   `sg_task` (`src/sgemm.c:670`) dispatches **1825** at 41.1 us, 36.1% barrier,
   100% below break-even. At 2 threads the same sites run 119 us and 157 us with
   1.2% and 3.3% barrier. Neither file belongs to this lane. Expected saving if
   the regions were coarser: the barrier share is an upper bound, so roughly a
   third of those sites' wall at wide widths — the same 3-7% this lane measured,
   but without needing the pool to be wide in the first place.
3. **The E9-P3 x86 gate.** `MYNAH_POOL_FASTEXIT=1` is -4.9% at 32 threads on
   aarch64 and has never run on x86. One `make test` plus the `pool.litmus` row
   on an x86-64 host is the whole gate.
