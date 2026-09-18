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
