# What the reference engine learned on ARM yesterday, and what it costs us

Read from `../qwen-tts` at `97c0fa1` (branch `feature/x86-amx-vnni-oss`), file
`.work/arm-sustained-soak-regression-20260913.md`, committed 2026-09-14 10:12.
Their engine, their hardware, their model. Recorded here because three of its
conclusions land directly on decisions we were about to take.

**Their numbers do not transfer and are not quoted as ours.** 1.7B at `4x8`,
C8, ~216 threads: a weight-traffic regime we are not in, and the user has
already corrected us once for borrowing their concurrency figures for a 109M
model. What transfers is the METHOD and the two REJECTED TREATMENTS, because
those are architectural rather than size-dependent.

## 1. A wave result is not a sustained result, and they now have the gap measured

> "True-wave/parallel capacity, sustained closed-loop capacity and realistic
> arrival-load capacity (for example Poisson) are distinct benchmark families.
> A true-wave result is never a sustained qualification."

Their closed-loop runs are *materially* worse than their wave runs at the same
concurrency: STREAM_RTF tails reach or pass 1.0, safe-play-start rises, and
stall@250/@500 become material — with **zero** crashes, rejects or timeouts.
Nothing looks broken; the cadence is simply not sustainable.

**This lands on us directly.** Our C64 result — 192/192, STREAM_RTF p95 0.895,
zero stalls — is a **wave**. `docs/performance.md` already says a screen may
disqualify and never promote, and this is the first outside evidence of how big
the gap can be. C64 is a candidate, not an operating point, and this is the
second independent reason to say so.

**Cheap gate worth stealing:** their regression reproduces in **2-5 minutes** at
C6/C7/C8 closed-loop. A thirty-minute soak is not needed to find it. We should
run a short closed-loop mini-soak at low concurrency long before we run a long
one at high concurrency.

## 2. A prefill helper thread is REJECTED, and this is the second measurement saying so

`QWEN_PREFILL_HELPER=1`, four-minute closed-loop, against inline control:

| arm | STREAM p95 | TTFA p95 | safe-start p95 | stall@250 | stall@500 |
|---|---|---|---|---|---|
| inline control | 1.041 | 214.0 ms | 716.8 ms | 20.1% | 2.8% |
| **helper** | 1.098 | **308.6 ms** | **1129.3 ms** | **46.8%** | **10.1%** |

Worse on every axis: +94.6 ms TTFA p95, +412.5 ms safe-start p95, +26.6 points
of stall@250. Their reading: *"a shared-pool helper is not QoS isolation"* —
inline prefill vanished from their stage records and the long samples stayed,
now dominated by queue wait.

**Why this matters here more than there.** Our prefill wall is real and a helper
thread is the obvious thing to reach for. It is now measured worse twice on
their engine. And our situation is *structurally worse for it*: at `16x2` a
worker has two threads on two cores, so a helper is a third thread on a machine
already at one thread per core — it cannot find idle resource, it can only
take some from playback. That last sentence is reasoning from our own
saturation measurement, not a measurement of ours; but the direction is not in
doubt.

## 3. A fixed-target admission guard is REJECTED too

`QWEN_ADMISSION_GUARD=1`, `TARGET_MS=400`: hold a new request while active
streams' ready-audio lead is at or below the target.

| arm | STREAM p95 | TTFA p95 | safe-start p95 | stall@250 | throughput |
|---|---|---|---|---|---|
| control | 1.061 | 241 ms | 776 ms | 25.5% | 0.666 req/s |
| **guard 400 ms** | 1.053 | **2805 ms** | **3066 ms** | 13.8% | 0.572 req/s |

It does what it says — stall@250 falls 11.7 points, 45.8% relative — and the
bill is TTFA p95 **+2.56 s**, safe-start p95 **+2.29 s**, throughput −14%, and
the stall@500 tail does not improve at all. 111 deferred admissions against 87
immediate, defer p95 ~2.8-2.95 s, worst defer **19.4 s**.

Their classification: PARTIALLY CONFIRMED, not promoted, and explicitly *"do
not begin cooperative slicing from this result."*

## What we do with this

- [ ] **Run a closed-loop mini-soak before trusting C64.** 2-5 minutes, low
      concurrency, continuously replaced streams rather than a wave. This is the
      cheapest disqualifier available and we have never run one.
- [-] **Do not build a prefill helper thread.** Measured worse twice on their
      engine, and our topology gives it nowhere to run.
- [-] **Do not build a fixed-target admission guard.** Measured: it buys short
      gaps with seconds of TTFA.
- [ ] Consider whether our harness can even produce a closed-loop load.
      `tools/serving_profile.py` has `--mode wave` and `--mode soak`; check what
      soak actually does — if it is a wave repeated, it is not this.

## What is still open, on their side and ours

Their strongest remaining hypothesis is a *playback-first* admission policy that
is bounded rather than fixed-target. They have not shipped one. We should not
copy an idea they have not yet made work, but we should watch that file.
