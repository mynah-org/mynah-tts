# x86 Genoa: best serving config, and a mini soak that may qualify it

Board: `PLAN.md` §E14-9 (no model-pack measurement on x86) and §E4-9 (Linux
measurement box, open since July).

## Problem

`.work/x86-genoa-first-numbers.md` has ratios only, because the box was not
quiet: an `md0` RAID-1 resync ran for the whole session and f32 matvec spread
0.243-0.413 ms over identical runs. We have never answered, on x86:

- how many worker threads a single PocketTTS request should use on 24 cores;
- how many concurrent requests the server holds before latency collapses;
- whether a configuration survives minutes rather than seconds.

`doctor.py` PREDICTS a configuration from topology. A prediction is a starting
point for a measurement, never a claim -- the doctor's own words. Nothing has
ever measured that prediction on x86.

## Evidence at the start (2026-09-22 15:35 CEST, box 5.199.172.200)

```
load average: 1.02   md0_resync at 83.0%, finish=52.9min, speed=200060K/sec
```

So the box is NOT quiet yet. One core is pinned by the resync and it is moving
200 MB/s through the memory controller. Same contaminant as last time.

## Plan

Two phases, split exactly on what the resync can corrupt.

- **A, now.** Correctness and configuration *discovery*, none of it a timing
  claim: sync to `origin/main`, clean build, `make info`, `make doctor --json`,
  `self-test`, `dispatch-gate`, `x86-tier-parity` on `models/pocket-en`.
  A pass/fail gate does not care about a busy core.
- **B, after `/proc/mdstat` reports no resync.** The remote script blocks on
  that condition itself, so the measurement starts on a quiet box without
  anyone polling from outside:
  1. thread sweep, single request, `--threads 1..24`, RTF per arm;
  2. `serving-wave` over levels 1,2,4,8,12 -- a SCREEN, may disqualify a level,
     may never promote one;
  3. `serving-soak` ~5 min at the level the wave did not disqualify, warm-up
     discarded, drift gate across 60 s windows -- the only mode that may
     promote a configuration.

## Acceptance gate

- Phase A: every gate PASS, and the doctor's PREDICTED thread count recorded
  verbatim so phase B can be scored against it.
- Phase B: `/proc/mdstat` shows no resync line in the log the run wrote, or the
  numbers are reported as contaminated and nothing is promoted.
- A configuration is promoted only by the soak's drift gate, never by the wave.
- Numbers land in `docs/performance.md` only if the box was quiet; otherwise
  they stay here, labelled.
