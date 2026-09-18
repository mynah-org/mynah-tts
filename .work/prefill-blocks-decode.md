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
