# The stall has one cause: prefill runs inside the step loop

Measured on the Axion, `main` at `8aa714a`, pack `models/pocket-en`, mixed v2
bank. This note exists because three concurrency levels (C99, C98, C96) failed
the same mandatory gate while every other gate passed, and lowering the load
bought a factor of ten without reaching zero. A quantity that survives a 3%
reduction in load is not saturation.

## The shape of the defect, from the load numbers alone

At C96 over 30 minutes, with one frame of audio lasting 80 ms:

    max_gap p50   97 ms     -- one frame; the steady state is perfect
    max_gap p95  358 ms     -- four to five frames missing, occasionally

So it is not a slowdown, it is an interruption: every so often a slot stops for
several frame periods and then resumes at the normal rate. `required_prebuffer`
is 0 ms at p50 AND p95 and 535 ms at max -- the same story from the other side.

## What costs 358 ms: the prefill of a long request

Per class, at C1, no contention, 90 s each:

| class | TTFA p50 | TTFA p95 | audio | n |
|---|---|---|---|---|
| short | 28.4 ms | 32.6 ms | 1.44 s | 407 |
| medium | 52.7 ms | 58.6 ms | 3.36 s | 173 |
| long | **211.2 ms** | **230.2 ms** | 15.84 s | 35 |

TTFA at C1 is prefill plus one step plus one frame of codec, so the prefill of a
long text costs about **190 ms** against 28 ms of fixed cost. That is 2.4 frame
periods, and with the frame in flight and the scheduling around it, it is the
358 ms gap.

## And the code says the same thing

`src/inference.c`, the continuous serving loop:

    /* ---- admission ---- */
    while (!drained && used < max_batch && ...) {
        ...
        if (slot_start(engine, model, state, &caps, slot, dump_all) != 0) { ... }
    }
    /* ---- one step over everything still live ---- */

and `slot_start` calls `engine->prepare()` -- the prefill -- synchronously.
**Every resident slot in that worker is frozen for the duration of the new
request's prefill.** The admission loop is also a `while`, so several prefills
can run back to back before the next step, though at the measured arrival rate
(~1.9 admissions per second per worker against ~18 steps per second) that is
rare and the single long prefill is the dominant term.

This also explains the one result that looked like good news and was not: the
English `medium` bank was GOOD at C99 for ten minutes. Medium prefills all cost
~53 ms, well under one frame period, so no resident slot ever starves. The mixed
bank is not "harder" in throughput -- it delivers the same RTF -- it is harder
because 11.9% of its prefills cost four times a frame.

## Why lowering concurrency barely helps

Admissions per worker per second scale with the arrival rate, and the arrival
rate at a fixed RTF scales with concurrency, so C96 has ~3% fewer freezes than
C99. The freeze DURATION is unchanged: it is a property of the text, not of the
load. Ten times fewer stalls came from the cushion, not from fewer freezes --
slots that are further ahead absorb a 200 ms hole. That is why the number falls
without reaching zero, and why it would keep falling all the way down without
ever reaching zero.

## What can be done, with the cost of each

1. **Chunked prefill.** Split `engine->prepare()` into slices and interleave
   them with decode steps, so the longest freeze is one slice instead of one
   prefill. This removes the cause. It costs an engine-seam API change
   (`prepare` becomes resumable) and it raises TTFA for the request being
   admitted: six slices interleaved with six steps is ~190 ms of prefill spread
   over ~520 ms of wall. Coarse slicing (two or three) is the compromise: the
   freeze drops to 60-90 ms, under one frame period, for ~110 ms of TTFA.
2. **Lead-aware admission.** Defer admitting while the least-advanced resident
   slot has less cushion than the incoming prefill will consume. No API change,
   but the arithmetic is unfavourable: a slot gains only `(1 - RTF) x 80 ms`
   = 25 ms of lead per frame, so it needs ~0.5 s of wall to build 250 ms of
   cushion, and with an admission every ~0.5 s per worker there is nearly always
   a vulnerable slot. A guard large enough to protect would defer almost every
   admission; a guard small enough not to would protect almost nothing.
3. **Do nothing and state the client contract.** No request in 53895 needed more
   than 535 ms of lead, so a 600 ms prebuffer plays all of them. This is already
   true and costs nothing, but it is a statement about a 30-minute sample of a
   tail.

(1) is the fix, (3) is what ships until it lands, (2) is recorded so it is not
re-proposed.

---

# The fix, and what it cost: measured the same day

`prepare_slice` is an optional, appended hook on the engine seam. The driver
alternates one slice with one step, so a resident slot never waits longer than a
slice, and a slot mid-prefill is a third state -- neither steppable nor finished.
PocketTTS implements it by resuming the text prefill a tile at a time, which is
sound only because the prefill was already tiled at 16 rows for E5-5 and the
alignment invariant was already asserted in `pocket_text_flush`.

**Bit-identity, which is the contract.** `md5` of a long utterance is unchanged
across `MYNAH_PREFILL_SLICE` 0, 16, 32, 48 and 128, on Apple silicon
(`a3544c9f…`) and on Neoverse-V2 (`274c78ff…`), single and batched. Slicing the
prefill does not change one sample of the audio.

## The slice curve, C96, mixed bank, ten minutes per arm

| slice | stall@500 | stall@250 | max_gap p95 | max_gap max | TTFA p95 | RTF p95 |
|---|---|---|---|---|---|---|
| 0 (old) | **3**/18007 | 81 | 358 ms | 697 ms | **308 ms** | 0.782 |
| 32 | 0 | **21** | 170 | **281** | 497 | 0.763 |
| **48** | 0 | 24 | **158** | 349 | 444 | 0.770 |
| 64 | 0 | 34 | 186 | 431 | **401** | 0.772 |

**32 is dominated by 48 on both axes**, which refutes the reasoning that produced
it: "smaller slices trade first audio for a shorter freeze" is false here. The
slice pass walks every slot still preparing, so a smaller budget keeps more
prefills in flight simultaneously and one step's freeze becomes their sum. The
right shape is a per-step TIME budget shared across the preparing slots; 48 is
the measured optimum without it, and is now the default.

## The qualification: C94, slice 48, thirty minutes

    94  53559/53559  TTFB95 81.3  TTFA95 443  RTF p50 0.668 p95 0.761
        preb95 12  safe95 478  gap95 158  coal 13%  throughput 129.6 audio-s/s
        MARGINAL

    PASS mandatory completed == launched     53559 == 53559
    PASS mandatory STREAM_RTF p95            0.761 < 1.000
    PASS mandatory stall_rate@500ms          (0 of 53559 requests)
    PASS preferred TTFB / TTFA / RTF / prebuffer / safe_play_start
    FAIL preferred stall_rate@250ms          (81 of 53559 requests)
    PASS drift rtf +0.0000, prebuffer +0.0003, over ten 3-minute windows

Against the same soak before the fix (C96, 30 min): `stall@500` **5 -> 0**,
`stall@250` **233 -> 81**, `max_gap` p95 358 -> 158 and max 707 -> 441, RTF p95
0.783 -> 0.761, throughput 130.5 -> 129.6 (-0.7%), TTFA p95 308 -> 443.

**Every mandatory gate passes for the first time in this investigation.** The
configuration has an operating point: C94, `16x2 --max-batch 8`, codec fully
int8, `MYNAH_PREFILL_SLICE=48`.

## One prediction of mine did not hold, and it matters

I said the residual `stall@250ms` would fall with concurrency. It did not:
**0.13% at C96 and 0.15% at C94**, flat. So the 81 remaining stalls are not a
capacity tail either -- they are what is left of the freeze, which is still
158 ms at p95, about two frame periods. Lowering concurrency further will not
close them; shortening the freeze will. That points straight at the per-step time
budget above, and it is the next thing to build, not a lower C.

---

# Concurrency is excluded, by four levels that refuse to line up

The question after the fix was whether the residual `stall_rate@250ms` could be
driven to zero by serving fewer requests. It cannot, and the shape of the data
says so more clearly than a trend would have.

All at `slice 48`, mixed v2 bank, `16x2 --max-batch 8`, ten minutes each except
C94 which is the thirty-minute qualification:

| C | stall@250 | rate | max_gap p95 | max_gap max | TTFA p95 | RTF p95 |
|---|---|---|---|---|---|---|
| 96 | 24 / 17922 | 0.134% | 158 ms | 349 ms | 444 ms | 0.770 |
| 94 | 81 / 53559 | 0.151% | 158 | 441 | 443 | 0.761 |
| **90** | 38 / 17729 | **0.214%** | 157 | 336 | 435 | 0.743 |
| 80 | 19 / 17474 | 0.109% | 148 | 327 | 396 | 0.668 |

**C90 is the worst of the four while carrying less load than C96 and C94.** The
rate wanders between 0.11% and 0.21% with no monotone trend across a 17% span of
load. A capacity tail does not behave like that.

`max_gap` p95 is the reason, and it is a near-constant: **158, 158, 157, 148**.
The freeze duration does not know the load exists, and the freeze is what stalls
a player. Lowering C removes OCCASIONS and leaves DURATION untouched, so the
count drifts down noisily and never reaches zero. With 53559 attempts the
unlucky request is always found.

**A gate that demands zero is a statement about the worst case, and the worst
case needs a bound, not a better distribution.** That is why the next change is a
cap and not a lower operating point.

## The bound: a per-step prefill budget

`MYNAH_PREFILL_STEP_MS`, default 40 ms -- half a frame period. The per-slot token
budget bounded one slice; it never bounded one STEP, because the slice pass walks
every slot still preparing and seven prefills landing together froze a worker for
seven slices. The cap stops the pass when the step's budget is spent and resumes
next step from where it left off, with a rotating cursor so that when the budget
cannot serve everyone it is a different prefill that waits each time -- without
it the lowest slot index is always served and an unlucky request can be starved
for as long as its neighbours keep arriving. One slice always runs even when the
budget is already spent: a cap that can starve a prefill forever is a deadlock,
not a bound.

It also predicts a reversal. The reason small slices lost (32 dominated by 48)
was that they kept more prefills in flight and a step's freeze became their sum.
Under the cap that sum cannot happen, so small slices should win again.

Bit-identity re-verified with the cap in place, across `slice` 0/16/48 x
`step_ms` 0/40, on both architectures. `make test` 23/23.

## An artefact, and a language refusal that is correct

Twelve streamed clips were captured from the live C90 soak -- three per class,
English, all-int8 including `codec_convtr`, seed 11 -- with the texts, the
configuration and the per-clip cadence in a manifest beside them. The two Italian
texts returned **HTTP 400**: `models/pocket-en` declares
`languages.resident = [english]` with `bound: true`, so the server refuses a
language it does not hold instead of synthesising it badly. That is the
`language_refused` counter which reads zero in every soak, doing its job.

The TTFA of those clips (0.9-1.4 s) is NOT the served TTFA. They are the 91st
request against a server already saturated at C90, so they queue. The served
number is the soak's: 435 ms at p95.

---

# The bound closed it: C90 is GOOD over thirty minutes

    90  53265/53265  TTFB95 82.0  TTFA95 494.8  RTF p50 0.659 p95 0.734
        preb50/95 0/10 ms  safe95 527  gap95 129  coal 13%
        throughput 128.9 audio-s/s   bank 277 of 277 texts, each 192x
        GOOD -- the operating point is C90

    PASS mandatory completed == launched      53265 == 53265
    PASS mandatory STREAM_RTF p95             0.734 < 1.000
    PASS mandatory stall_rate@500ms           (0 of 53265 requests)
    PASS preferred TTFB p95                   82.0 <= 100
    PASS preferred TTFA p95                   494.8 <= 500
    PASS preferred STREAM_RTF p95             0.734 <= 0.900
    PASS preferred required_prebuffer p95     10.2 <= 500
    PASS preferred safe_play_start p95        527.3 <= 1000
    PASS preferred stall_rate@250ms           (0 of 53265 requests)
    PASS drift rtf +0.0040, prebuffer +0.0000, over ten 3-minute windows

Config: `16x2 --max-batch 8`, codec fully int8 including `codec_convtr`,
`MYNAH_PREFILL_SLICE=32`, `MYNAH_PREFILL_STEP_MS=60`.

## The cap's prediction, tested and confirmed

The cap said small slices would win again, because the only reason they lost was
the sum across slots. C90, ten minutes per arm:

| slice / cap | stall@250 | max_gap p95 | max_gap MAX | TTFA p95 | verdict |
|---|---|---|---|---|---|
| 48 / none | 38 | 157 ms | 336 ms | 435 ms | MARGINAL |
| 48 / 40 ms | 16 | 152 | 176 | 434 | MARGINAL |
| 16 / 40 ms | **0** | 104 | **122** | 642 | MARGINAL (TTFA) |
| 16 / 80 ms | 2 | 107 | 143 | 637 | MARGINAL (TTFA) |
| **32 / 60 ms** | **0** | 129 | 173 | 496 | **GOOD** |

The cap bounds TOTAL prefill throughput, so too tight a cap starves every prefill
when several compete and first audio pays for it. That is the whole trade: the
freeze and the time to first audio are the same resource seen from two ends.

## What the fix bought, stated exactly

It did NOT raise the concurrency ceiling. C99 already ran: 130 audio-s/s, RTF
0.83, nothing dropped. What did not exist was a level that could be PROMOTED.

| | before | after |
|---|---|---|
| highest qualified level, mixed bank | **none** | **C90 GOOD** |
| `stall@500ms` | 5 of 53895 | **0 of 53265** |
| `stall@250ms` | 233 of 53895 | **0 of 53265** |
| worst freeze (`max_gap` max) | 707 ms | **177 ms** |
| worst client prebuffer | 535 ms | **267 ms** |
| throughput at equal C | 130.5 | 129.6 audio-s/s (-0.7%) |

The client contract changed with it: the interim "600 ms prebuffer" is now
**250 ms**, and it is a bound rather than a sample maximum.

## What binds next, and it is not the machine

At C90 `STREAM_RTF` p95 is **0.734**: the box delivers a third faster than
realtime and has headroom. The gate that binds is **TTFA at 494.8 against 500**,
which is the price of slicing. Raising C from here breaks first audio, not
throughput, so the next lever is not another scheduling knob -- it is the
ABSOLUTE cost of the prefill, still 190 ms for a long text and never optimised.

## One prediction held, for a reason worth keeping

I expected TTFA p95 to survive the move from ten minutes to thirty (496 -> 494.8)
after being burned that morning by a 4.6 ms margin that did not survive. The
difference is the STATISTIC, not the luck: a p95 is stable in sample size, a
maximum is not. What failed that morning was `required_prebuffer` MAX; what held
here was a percentile. Read the statistic before deciding whether a thin margin
is fragile.

## 2026-09-19 — the ceiling was never measured, and it is higher than C90

C90 was promoted yesterday because it was the level being *investigated*, not
because anything had said it was the top. Five ten-minute soaks on a freshly
rebuilt v1.4.0 (box repo reset to `origin/main` at 77187b6, both binaries
rebuilt — `make` alone does not rebuild the server, and the stale one was nearly
measured) settle where the top actually is:

| C (10 min, convtr ON) | verdict | TTFA p95 | RTF p95 | stall@250 | stall@500 |
|---|---|---|---|---|---|
| 90 | **GOOD** | 492 ms | 0.728 | 0 of 17904 | 0 |
| 96 | **GOOD** | 494 ms | 0.756 | 0 of 18032 | 0 |
| 98 | MARGINAL | 499.5 ms | 0.800 | **13** of 18067 | 0 |
| 100 | MARGINAL | 522.6 ms | 0.817 | **22** of 18056 | 0 |

**The failure changes character above C96, and that is the finding.** Between
C96 and C100 the TTFA p95 moves 28 ms — the prefill cost is behaving exactly as
the per-step cap promises, a slow linear drift. What does not behave is the
stall count: 0 → 13 → 22, roughly doubling every two levels of concurrency,
while `max_gap` p95 stays at 130-134 ms against an 80 ms frame. At C98 the TTFA
gate still passes, by half a millisecond, and the run is MARGINAL *only* because
of stalls.

So the bound built yesterday is a bound on **prefill work per step**, and above
C96 the thing that overruns the frame is no longer prefill. Sixteen workers at
C100 carry 6.25 live slots each against 5.6 at C90; the AR step itself starts
missing the 80 ms deadline. Any further work on "zero stalls at a higher C" has
to attack the decode step or the slot count per worker, not the admission path.
The next knob to try is therefore `--prefork 16 --prefork-threads 2` against a
wider `--max-batch` or a nineteenth/twentieth worker, not another prefill knob.

A second reading of the same table: `STREAM_RTF` p95 0.817 at C100 means the box
still has a fifth of a realtime budget in hand at a level it cannot qualify.
Capacity is not what stops us — continuity is, and it always has been.

## 2026-09-19 — a fourth prediction of mine, refuted the same day I wrote it

Earlier today this note and `docs/performance.md` both said that above C96 the
thing overrunning the 80 ms frame was "the AR step itself, as slots per worker
rise". It was stated as mechanism on the strength of an arithmetic story, and it
is wrong.

`MYNAH_SERVE_PROFILE=1` was extended to time each step and bucket it by width.
C94, sixteen workers, 195303 steps:

| B | steps | mean | worst | late | per-slot |
|---|---|---|---|---|---|
| 1 | 981 | 10.8 ms | 13.8 ms | 0.000% | 10.8 ms |
| 4 | 8293 | 33.6 | 44.3 | 0.000% | 8.4 |
| 5 | 53210 | 43.7 | 56.0 | 0.000% | 8.7 |
| 6 | 130739 | 50.0 | 66.8 | 0.000% | 8.3 |

Not one step crossed the deadline; the worst was 66.8 ms against 80. `a = 3.0 ms`
and `b = 7.8 ms` put the crossing at **B ≈ 9.9**, and `--max-batch` is 8 — the
width is already bounded below the deadline. The histogram also stops at B6
(`mean_live` 5.58): a closed-loop load of 94 over 16 workers is much narrower
than a Poisson arrival, so `--max-batch` is not even an active knob here. The
`--max-batch 6` arm was run as a control for exactly this reason.

**What the number actually says.** `max_gap` p95 is 130 ms while the worst step
is 66.8, so the gap is a sum and not a step: one prefill-slice pass (cap 60 ms)
plus one typical step (50 ms) is 110 ms against a frame of 80. The slack per
frame is `80 - 50 = 30`; the cap is 60. Every frame carrying a prefill pass costs
each slot in that worker ~30 ms, and at RTF 0.746 a slot earns 20 ms of lead per
frame, so one pass takes ~1.5 frames to repay. Several arriving close together on
one worker exhaust the cushion — one stall in 54000.

`MYNAH_PREFILL_STEP_MS = 60` was chosen by trying values yesterday. The measured
principle is `cap = deadline - T_frame(B typical)`, ~30 ms on this host at this
concurrency. Whether it can be lowered is a TTFA question, not a continuity one:
the gate stands at 494.9 against 500, and halving the cap doubles the steps a
prefill takes.

**The lesson is the same one as the other three.** Every refuted prediction in
this note shared a shape: an arithmetic story about where the time must be
going, written down as if it had been measured. The instrumentation that settled
this one is fifteen lines and should have existed before the claim did.

## 2026-09-19 (later) — two more things the box said, one of them about the instrument

### The cap is the slack, and it is free

Three ten-minute soaks at C94, shipped quantization, only `MYNAH_PREFILL_STEP_MS`
moving:

| cap | `max_gap` p95 | `max_gap` max | TTFA p95 | stall@250 |
|---|---|---|---|---|
| 60 (default) | 131 ms | 179 ms | 495-500 | 0-1 |
| 40 | 121 | 148.6 | 494 | 1 |
| 30 | 119 | 142.9 | 498 | 0 |
| 20 | 119 | 133.9 | 496 | 1 |

The cap bounds the worst case monotonically -- the MAXIMUM falls 179 → 134 --
and **TTFA does not pay for it**. That refutes the tension predicted one message
earlier ("halving the cap doubles the steps a prefill needs and raises TTFA"),
and the reason it was wrong is worth keeping: the per-slice budget is already
32 tokens, so one slot's prefill rarely reaches 30 ms by itself. The cap only
binds when several prefills coincide on one worker. **It bounds the tail without
touching the median path**, which is the property one wants from a limit and
rarely gets for free.

The stall column says nothing, deliberately: 1/0/1 is the same coin toss as
below. `max_gap` is a statistic over the whole distribution and moves cleanly;
a gate cut through the middle of a distribution does not.

### The same configuration, measured twice, got two different verdicts

The `--max-batch 8` and `--max-batch 6` arms were meant as an experiment and an
arm. The step-cost table shows they were the same experiment twice over: B6 mean
50.0 vs 49.8 ms, 130739 vs 131363 frames at that width, because the loop never
reached 7 or 8 in either. The verdicts:

    --max-batch 8    TTFA p95 500.136 ms    MARGINAL   (failed by 136 microseconds)
    --max-batch 6    TTFA p95 496.0         GOOD

**C94 and C96 do not sit near the TTFA gate, they sit ON it.** TTFA p95 is
495 ± 5 ms against a threshold of 500, so a ten-minute verdict at these levels is
a coin toss, and the difference between "qualified" and "not" was 136 µs of a
percentile. Nothing about the server is unstable -- the instrument is being read
past its resolution.

Consequences, all of them operational:

* a single ten-minute run may not decide anything at C94-C96, not even an A/B,
  when the deciding quantity is TTFA. Thirty minutes, or repeats;
* this morning's `codec_convtr` A/B survives only because its delta was 52 ms,
  ten times the noise. That was luck, not method;
* `--max-batch` is measured and closed as a lever at this concurrency.

## 2026-09-19 (evening) — the ceiling moved, and the knob that set it is now stale

With `backbone:bf16` and the tiled BFMMLA kernel, screens on the shipped default
(nothing exported):

| C | TTFB p95 | TTFA p95 | RTF p95 | stall@250 | verdict |
|---|---|---|---|---|---|
| 100 | 62.5 ms | 386.6 ms | 0.702 | 0 of 20888 | **GOOD** |
| 110 | 68.3 | 478 | 0.721 | 0 of 21176 | **GOOD** |
| 120 | 74.8 | 514 | 0.788 | 0 of 21435 | MARGINAL — TTFA only, by 14 ms |
| 130 | **200.7** | 581 | 0.802 | 2 of 21620 | MARGINAL — TTFB and TTFA |

Against the same soak before bf16, at C100: MARGINAL, TTFA p95 526, RTF 0.820,
19 stalls of 18018, 124.4 audio-s/s. Now GOOD, TTFA 386.6, 0 stalls, 144.5
audio-s/s — **+16% throughput and 139 ms off first audio**.

**Two readings that matter more than the levels.**

*The bottleneck moved and then moved again.* At C100 the gates sit at 62% (TTFB),
77% (TTFA) and 78% (RTF) of their thresholds — balanced, where this morning TTFA
was at 105% and failing alone. At C120 only TTFA fails, with stalls still at
zero: continuity is no longer the limit, first audio is. At C130 **TTFB jumps
74.8 → 200.7 ms**, which is not synthesis at all — it is the admission queue.
16 workers x 8 slots is 128, and 130 requests is the first level that fills it.

*So `--max-batch` stops being inert exactly there.* It was measured this morning
as a non-lever because the loop never reached 7 live slots; at C130 it saturates.
Above C120 it becomes a real knob again and is the first place to look.

### The cap is stale, and nobody re-derived it

`MYNAH_PREFILL_STEP_MS = 30` came from `slack = 80 - T_frame(B6) = 80 - 50`,
where the 50 ms was measured with the **f16** kernel. bf16 made the step cheaper,
so the slack is now LARGER than 30 and the cap is rationing prefill work that the
frame could absorb — against TTFA, which is the only gate still failing at C120.

Re-deriving it costs no code: `MYNAH_SERVE_PROFILE=1` re-measures `T_frame(B)`,
and the cap sweep is the same four ten-minute runs as this morning.

### And a scheduling idea that does need code

`slots_prefill_slice()` serves preparing slots **round-robin**, with a rotating
cursor so none starves. That is processor sharing, and processor sharing is the
policy that maximises the number of jobs in flight — every prefill finishes at
roughly the time the LAST one would have, rather than in turn.

FIFO-to-completion should beat it on the mean immediately (the k-th request
finishes after the k ahead of it, not after all of them) and on the tail through
Little's law: a lower mean prefill time means fewer prefills resident, which
means less competition, which lowers the mean again. The worst case in a single
step is unchanged, because `MYNAH_PREFILL_STEP_MS` already bounds it.

Risk to measure rather than assume: head-of-line blocking, where a long text's
prefill delays a short one behind it. The mixed bank is the right instrument —
`short` is 24% of it and `long` 12%, and per-class TTFA is already reported.
