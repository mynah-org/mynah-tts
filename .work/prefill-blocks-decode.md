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
