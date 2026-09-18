# Serving design v2 — what to copy from qwen-tts, and why each piece exists

Source: `~/source/personal/qwen-tts` branch `feature/x86-amx-vnni-oss`, read in
full (server, thread pool, profiles, and fifteen `.work/` campaign reports).
Every number below is theirs, measured, with the file it came from. Ours are
marked as ours.

**The framing that matters before anything else**: there is no `SERVER_V2`
symbol in their tree. "v2" is a *generation*, and in code it is simply a second
serve entry point that coexists with the first. We are not porting a module. We
are copying a set of **ownership decisions**, each of which was bought with a
measurement that killed the obvious alternative.

Their own general law, written by their post-mortem judge, is the shortest
useful summary of the whole campaign:

> Every change that removed work from the serial critical path won. Every change
> that added a second submitter, a gate or a parking policy on the same pool
> lost. Explicit overload boundaries helped by making waits visible.

Read that twice before designing anything here. Most of our instincts — a queue,
a fairness gate, a helper thread, a smarter admission heuristic — are on the
losing side of it, and each was tried and measured there.

Related: [`serving-doctrine.md`](serving-doctrine.md) (the ranking),
[`linux-production.md`](linux-production.md) (the traps),
[`streaming-server-v2.md`](streaming-server-v2.md) (our items).

---

## 1. The shape: processes, not threads

Three nested partitions, decided outside-in, none of them inferred:

| level | unit | who decides | ours today |
|---|---|---|---|
| **A** processes | `--prefork W` pinned worker processes, one contiguous **core-major** CPU slice each | explicit flag, never from `nproc` | does not exist (E5-6) |
| **B** threads | `K-1` pool threads + the calling loop thread = `K` | `--prefork-threads`, default `cpus_in_mask / W` | `MYNAH_THREADS`, global |
| **C** lanes | the last `N` cpus of the worker's mask become a private decoder team | `QWEN_SD_LANE_SPLIT=N` | does not exist |

**Workers are processes and a request never migrates.** No global batch, no
cross-worker state, no stealing, no consolidation. The parent owns the listening
socket, a per-worker `active[w]` counter and the CPU plan — it never parses HTTP
and never touches the model. The channel is one `AF_UNIX` socketpair per worker;
the accepted fd crosses by `SCM_RIGHTS`, and exactly **one byte** crosses back
per completed connection. That byte is the parent's only completion signal.

Weights are packed and quantized **before the fork**, so int8 weights and mmapped
safetensors are copy-on-write shared: 2.0-2.3 GB PSS per worker, 10.5-11.3 GB for
four workers plus parent. Their conclusion is worth keeping because it kills a
tempting idea: *"replication is not a cost here and cache sharing is not a
benefit; the topology effect is bandwidth-per-CCX plus team width."*

Oversubscription is avoided **structurally**, not by tuning: `want = n_threads - 1`.
The pool cannot oversubscribe because it is constructed not to. Nested dispatch
from inside a pool task runs inline for the same reason.

### The one ordering that is not optional

```
sched_setaffinity(child)  →  pool-state reset after fork  →  lane split  →  pool creation
```

because **pthreads inherit the creating thread's mask**. Pin first or the pool is
born unpinned, and nothing later fixes it.

### Core-major, and the bug that makes it non-negotiable

Linux numbers every core's *first* thread before any sibling. On a 12-core SMT-2
host (siblings `N` and `N+12`), `--prefork 2` with a contiguous logical slice gave
worker 0 cpus 0-11 and worker 1 cpus 12-23 — **the same twelve physical cores, one
worker per hyperthread**. The log showed two disjoint ranges. The workers were not
isolated at all, and the per-core AMX tile unit serialised them on top.

So: order cpus by `(physical_package_id, core_id)` read from sysfs, fall back to
identity when sysfs is absent, and **print the mask that was actually set**, not
the slice indices. Their `[TOPOLOGY] v=1 worker=… configured_mask=… actual_mask=…`
record exists because *configured* and *actual* are different claims.

→ our E5-14.

---

## 2. Thread ownership inside one worker

| thread | owns | must never |
|---|---|---|
| acceptor | receives the fd, sets `SO_RCVTIMEO` + `TCP_NODELAY`, pushes to a 256-slot ring | parse, synthesize |
| readers ×`clamp(max_batch,2,16)` | 1 MiB buffer, HTTP parse, JSON validation, job allocation | any inference |
| **scheduler ×1** | **the entire engine**: all B slots, the pool, admission, every per-slot state | read sockets |
| single-job worker ×1 | a *cloned* ctx for requests the batch engine cannot express | touch batch slots |
| decoder lane team | the decoder on the last N cpus, with its own submit lock | touch the engine pool |
| output writers | one per stream: the socket, chunked framing, close | inference |

Threads are named for `/proc` on purpose: *"the thread ownership table of a worker
is then readable without a debugger. Zero cost after creation."* We should do this
— it is free and we have already spent hours guessing which thread was blocked.

**The lane's isolation is enforced in the dispatch primitive, not by discipline at
call sites**:

```c
/* The decoder lane never reaches the engine pool: everything its thread dispatches
 * (conv panels, snake rows, SGEMM slices, im2col, bf16 matmat) runs on the lane team. */
if (g_lane_tls) { lane_parallel(nt, fn, ctx); return; }
```

One `if` at the top of `parallel()`. That is the whole mechanism. Copy this
shape: a thread-local tag that redirects dispatch is robust against every future
call site, and a convention is not.

---

## 3. Waiting: spin, then park

Worker loop and submitter are symmetric — bounded spin on an atomic, then a
condvar. `cpu_relax` is `pause` on x86 and `yield` on aarch64. The generation word
packs (generation, needed-worker-count) into one atomic so a dispatch that needs
`need` workers only broadcasts if a *needed* worker is parked.

Phases inside one region use a **sense-reversing spin barrier**, justified in the
code: every participant is already a running thread and the phases are
microseconds long, so spinning is the correct wait. They measure **225 spin
barriers per Talker frame, 718 per CP frame, ~943 per iteration**.

The spin budget is the single highest-leverage knob they found, and it is
ISA-dependent by default (65536 on aarch64 Linux, 4096 elsewhere):

| `QWEN_POOL_SPIN` | STREAM p95 | ctx switches/s |
|---|---|---|
| 256 | 0.999 | 197,000 |
| 4096 | 0.893 | 38,000 |
| **65536** | **0.808** | **7,600** |

Independently on ARM: CP 16.0 → 9.6 ms/frame, 491,320 → 35,132 context switches.
Higher buys nothing; 262144 is worse.

**This is the number to take seriously for us.** Our own thread-pool work
measured ~20 µs of wake-up per region, i.e. a region must be worth ≥200 µs to pay
for itself. Their spin sweep is the same wall approached from the other side, and
it says the fix is not fewer regions but *not parking between them*.

### Context switches as the shape diagnostic

`1x8` 2-4k/s · `4x8` 8-9k · `1x16` 20.6k · `2x16` 37-42k · `1x32` 75-91k.

A wide pool does not fail gradually. `1x32` at C8 gives STREAM 1.40-1.55 and 62%
of frames stalling past 500 ms — their word is "dead".

---

## 4. The lane law, and why the cost model is the design

For one 8-core CCX, measured undiluted on fixed text:

```
1.7B:  T_frame(B) ≈ 40 ms + 13.5 ms × B        frame = 80 ms of audio
0.6B:  T_frame(B) ≈ 23 ms + 12.3 ms × B
```

The constant is the **weight stream** — bandwidth-bound, identical at 2, 4 and 8
threads. The slope is **the decoder**, 72-80% of the marginal cost of each extra
slot. Batching is nearly free on the constant and pays full price on the slope.

Two facts from this that change what we build:

1. **Above B1 the machine is not memory-limited.** The weight stream is fixed;
   adding a slot costs decoder time, not bandwidth. So the useful work is making
   the decoder cheaper or moving it off the critical path — not compressing
   weights further.
2. **A CCX holds exactly one step-lane.** Two 4-core lanes on the same CCX
   measured STREAM 1.278/1.286 against 0.972 solo; Talker 63.7 ms/step against
   27.8. Two weight streams on one CCX halve each other. Meanwhile a 2-thread
   lane costs the *same* Talker+CP as an 8-thread lane, because the weight stream
   saturates the CCX at two threads — **the other six cores are free for work that
   does not stream weights.** That sentence is the entire justification for the
   decoder lane.

The per-CCX bandwidth roof they measured (c8a.8xlarge, Zen5, 4×8c, 32 MiB L3 each):

| mask | threads | DRAM GB/s | 101 MB cache-resident GB/s |
|---|---|---|---|
| one CCX (0-7) | 8 | 55.0 | 65.7 |
| two CCX (0-15) | 16 | 109.9 | **40.5** |
| all (0-31) | 32 | 208.3 | 333.8 |

The 16-thread mask reads **slower cache-resident than DRAM**: a working set split
across two CCX thrashes between them. That is the first physical reason wide
pools lose, and it is measurable in ten minutes on any host — we should measure
it on ours before choosing W.

---

## 5. The decoder lane: three attempts, two failures, one win

This is the most transferable single piece, because **our decoder is in exactly
the same position**: glue-bound, per-slot, on the loop thread.

| attempt | what it did | measured |
|---|---|---|
| same-pool consumer | second submitter on the engine pool | TTFA p95 167 → **1200 ms**; serialized on the submit lock behind the regions, lost the gang |
| private 2nd team | its own threads, unpinned | 21 threads on 8 cores — **oversubscribed** |
| **lane split** | private **pinned** team on the last N cpus, own submit lock, bounded mailbox | **win** |

Lane results at `1x8`, 1.7B, STREAM p95:

| B | inline | static 4+4 | elastic | + V2 conv |
|---|---|---|---|---|
| 1 | 0.674 | 0.613 | 0.615 | |
| 2 | 0.866 | 0.774 | 0.747 | |
| 3 | 1.020 | 0.871 | 0.859 | |
| 4 | **1.203** | 0.997 | 0.987 | **0.918** |

Iteration wall p95 at B4: 192 → **77 ms**. Stall@250 ms at B4: 100% → **0%**.

**The split size was swept, not guessed.** 4+4 / 5+3 / 6+2 at fixed B4:
0.997 / 1.113 / 1.364, with the producer's mailbox wait going 3.6 → 8.0 → 16.6 ms
per frame. On two cores the lane is *slower than inline*. And 2 step + 6 decoder
was killed before that: on two threads the per-slot region work costs +17 ms per
slot, so a 2-core step lane reaches B4 at ~98 ms before any decoder runs at all.

The contract that makes it safe is small and instrumented:

- **at most one decoder unit in flight per slot**, at most one quantum accumulated
  behind it; a violation prints `MAILBOX OVERRUN … the bounded contract was
  violated` and the counter must stay 0 in every run;
- **per-slot blocking only** — other slots' steps never wait for a decode;
- **never free state the lane is decoding**;
- the **frame boundary is the only preemption point**; there is no intra-call
  preemption anywhere.

Elastic (the engine keeps all cpus and narrows only while a unit is in flight) beat
static by 0.01 — but its real value was a *falsification*: Talker+CP at B4 was
69.5 ms elastic vs 69.8 static, identical, which killed "the static partition is
taxing the step" and left L3 pollution by the decoder as the actual term. A
decoder unit in flight costs the text stage **+11-14 ms per iteration**, and in
pure-overlap iterations that stage runs at exactly its DRAM rate — *"its
L3-resident half is gone."*

---

## 6. The admission ladder, and what fails fast

Four independent refusal points, outermost first:

1. **parent slot cap** — least-loaded worker with `active[w] < cap`, else an
   immediate 503;
2. **child queue bound** — `running + queued >= slots + queue_cap` → 503, default
   queue cap **1**;
3. **queue deadline at pop time** → 503 "queued too long";
4. **per-request service cap** → stop at the next frame boundary.

Plus input-shape admission: the text character limit is derived from *both* the
prompt-token budget and the time cap, and the rejection message states the exact
budget.

The subtle one, and the one we have already half-hit: **the parent keeps polling
the listener while every slot is full.** Before that fix, a full server stopped
accepting, the client waited in the kernel backlog, and the child's queue deadline
could never see that wait. Measured at C5: TTFB/TTFA p95 **4470/4635 ms** entirely
invisible — *">97% of the tail was before `accept()`"*. After: eight accepted with
TTFA 423/562 ms plus two immediate 503s.

**A wait you cannot see is worse than a refusal you can.** That is the same lesson
as our own 66.5 s slow-client stall, from the other end of the pipe.

### Utilization-aware admission: built, measured, failed

They built a shared health page (three atomics, `MAP_SHARED`, parent reads only)
and let the parent admit one transient extra connection to a full worker when its
recent service-loop interval showed headroom. All three thresholds (40/60/80 ms)
did admit, and the extra stream was interactive at TTFA 394-479 ms. **The four
established streams broke**: STREAM p95 0.985-1.004 against a control of 0.835,
stall@250 **50%** against 0%.

Their closing sentence is the one to keep: *"The result is also not a reason to
tune more threshold values."*

→ this is our E5-11, and it says: when the machine is full, **refuse**. Do not
get clever.

---

## 7. Lifecycle, and the two latency bugs hiding in it

```
parent  poll(listen + W channels)
        └─ listener polled iff (free slot || reject-at-parent)
        └─ accept → least-loaded worker → SCM_RIGHTS → active[w]++
child   recv fd → re-read own affinity → resize soft thread budget → ring
reader  parse, validate, route → job queue  (503 if full)
sched   next_job → HEADER SENT HERE → prefill → install KV → lockstep frames
        → decoder (inline or lane) → on_chunk → writer
done    chunked terminator → close → one byte back to the parent
```

Two things in that diagram are latency bugs that we can have too:

**The header goes out at admission, not at the first chunk.** Otherwise TTFB
equals TTFA and the whole admission/prefill cost is invisible to the client
metric. With the fix, C1 TTFB p50/p95 is 0.4/0.6 ms against TTFA 82.6/84.2 ms.

**`TCP_NODELAY` on every accepted socket.** Coalesced reads 6.7% → 0.0%.

And one correctness bug we are *very* likely to reproduce: their warm-up ran on
whatever state the CLI had left in the context, so it primed per-request state for
a configuration no request uses, and **the first real request differed from every
one after it**. Warm up through the same reset the request path uses.

---

## 8. Prefill: the stall everyone hits

Inline prefill inside the frame loop stalls **every established slot** for the
whole prompt — measured 108-240 ms. Two ways out were tried:

- **LOW-priority prefill helper thread**: moved the closed-loop short class from
  0.966 to 0.915 *while raising TTFA p95 from 172 to 683 ms* — and in the serving
  A/B, TTFA p95 435 → **2379 ms**. It *"established the mechanism and disqualified
  that implementation."*
- **sliced admission**: one slice of the prompt per iteration while the others
  stream, the whole remainder when idle. Kept.

Also an invariant worth stealing verbatim: **one pending admission per worker**,
owning the admission block until complete, because the partial KV and the prompt
state live in the context and a second prompt build would destroy both.

---

## 9. The rejected list — do not re-run these

Each was measured on their hardware. Ours differs, but the *mechanism* of each
failure is architectural, not host-specific.

| tried | measured | why it failed |
|---|---|---|
| wide pool `1x32` | C8 STREAM 1.40-1.55, 62% stall@500, 90k csw/s | team width vs LLC domain |
| two lanes per CCX (`8x4`) | 1.278 vs 0.972 solo | one weight stream per CCX |
| 2+6 / 5+3 / 6+2 lane splits | 1.364 at 6+2, mailbox 16.6 ms/frame | decoder needs ≥4 cores to hide |
| same-pool decoder consumer | TTFA 167 → 1200 ms | second submitter on one pool |
| oversubscribed 2nd team | +20-50% STREAM | 21 threads / 8 cores |
| prefill helper + LOW | TTFA 435 → 2379 ms | priority is not isolation |
| cap 3 per worker | 0.835 → 0.969, stall@250 0 → 13% | past the lane law's knee |
| raised cap 8 | admission to C32 with no rejects, **no C20+ sustains** | admission ≠ capacity |
| global cross-worker batching | useful B≥3 coincidence **1.6% @±0.25 ms, 11.7% @±8 ms** vs a 25% bar | arrivals do not coincide |
| utilization-aware admission | stall@250 50% | see §6 |
| lead/credit gate | 0.838 → 0.986, parks 95.8% of checks | withholding ready work loses |
| sub-quantum decode | 0.895 → 1.475 | more rendezvous |
| larger decode quantum q8 | RTF 0.864 **passes**, prebuffer 806 ms, stall@250 100% | the cadence law |
| NTA prefetch / hot workers / direct conv | no effect | cache tricks falsified |
| ragged panel scratch reuse | C3 better, C4 worse | flag removed |

Two entries deserve to be read as conclusions rather than data points.

**Global cross-worker batching is not justified.** Over 1112 steady-state events,
requests that could have been batched at B≥3 coincided within ±0.25 ms only 1.62%
of the time, and within ±8 ms only 11.69%, against a 25% bar. Arrivals are not
synchronised, and waiting to synchronise them is the cadence law's forbidden move.

**Admission capacity and sustained capacity are different numbers.** Raising the
cap let C32 in with zero rejects. Nothing above C20 sustained. A server that
accepts everything and serves nothing in real time has a worse failure mode than
one that refuses.

---

## 10. The measurement protocol — this is the actual deliverable

The reason their numbers mean anything is the protocol, and it costs us almost
nothing to adopt.

- **WAVE** = 3 synchronised waves per concurrency level, all C fired at t=0. This
  is a *screen*, not a qualification.
- **SOAK** = 5-30 minutes at fixed C, with warm-up and a strict drift gate. Only a
  SOAK promotes.
- Deciding metric: **STREAM_RTF p95**, preferred gate **≤ 0.90**, hard gate
  **< 1.0**. Then TTFA p95, required prebuffer p95, safe-play-start p95,
  stall@100/250/500 ms, errors/rejects, core-equivalents, context switches/s.
- Every cell records the actual CPU masks and a strict profile preflight.
- A prediction from their `doctor.py` is *"a starting point … never a claim."*

The gap between screen and soak is not cosmetic: **C16 passed the wave screen at
0.919-0.974 and failed a 30-minute soak** with STREAM p95 1.004, 596 rejects and
111 broken pipes. Their conclusion: *"C16 is the hard-capacity boundary, not a
product point."* Preferred C11, qualified C12, hard capacity C16 — three different
numbers, and only the middle one ships.

They also gate on **audio**, not only on timing: the entire "all-on" Arm profile
was *faster* (C12 at 0.842 vs 1.059) and was **rejected** because mel-correlation
came in at 0.886-0.945 against a 0.98 gate. A serving profile that degrades the
audio is not a serving profile. We need that gate before we tune anything, and we
have the pieces for it already.

---

## 11. What transfers to us, ranked

**Copy the ownership, not the code.** Our dataflow is different — 12.5 Hz latents,
a 1-step flow head, a SEANet decoder instead of an RVQ one — but every structural
decision above is about *who owns what*, and that transfers exactly.

1. **Prefork + pinned core-major slices + fd passing** (E5-6, E5-12, E5-14). This
   is the mechanism that converts cores into streams. Without it more cores are
   not more streams; with the wrong slice order they are actively fewer.
2. **One scheduler thread owning the engine; readers and writers off it** (E5-1,
   E5-2, E5-3 ✓). We have the writer. The scheduler is next.
3. **Spin-then-park with a large spin budget**, and context switches per second as
   a first-class metric. Our ~20 µs wake-up says this is where our glue cost is.
4. **A decoder lane**, once 1-3 are in place. Our decoder is 72% of the frame and
   per-slot — the same position theirs was in. Not before: on a narrow lane it is
   slower than inline.
5. **The admission ladder with fail-fast 503 and a still-polled listener** (E5-11).
6. **The measurement protocol** (§10) — arguably first, since without it none of
   the above can be shown to have worked.

**Do not copy**: global cross-worker batching, any fairness/credit gate, a
priority-based prefill helper, utilization-aware admission, or a second submitter
on the engine pool. All five were built there, measured, and lost.

---

## 12. Open holes they name themselves

Worth knowing, because we will meet them in the same order:

- the synchronous output path does blocking 15 KB writes with **no send timeout** —
  bounded only for healthy clients (we already hit this: 66.5 s);
- their fail-fast 503 writes and closes **without draining the request body**, so
  the client sees an RST instead of the status — clients report `BrokenPipe`
  rather than the refusal. Fix is `shutdown(SHUT_WR)` + a bounded drain;
- health and metadata requests occupy a dispatch slot until close;
- unconditional `fprintf` telemetry from all workers serialises on stderr;
- one banner claims a batched decoder that on VNNI resolves to the sequential
  fallback — a **misleading log line about the thing being measured**.
