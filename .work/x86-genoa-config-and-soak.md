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

## The plan changed at 16:00, and the reason is worth keeping

Phase B blocked on `/proc/mdstat` as designed, and then sat there: thirteen
minutes of a rented box at 0% CPU with twenty-eight more to go. The owner saw
the idle box before I did.

**Waiting idle buys nothing; a second pass buys a number.** So: measure NOW with
the resync running, measure AGAIN when it stops, run the same arms both times,
and let pass2/pass1 BE the size of the contaminant instead of a caveat about it.
Each arm is 1 warm-up plus 5 timed runs at `--max-steps 64`, so every arm does
identical work, and the sweep runs forward AND reverse so drift shows as an
asymmetry rather than hiding in the ordering.

The first pass immediately showed the contaminant is far smaller for THIS
workload than for the micro-bench that produced the 70% spread: the five repeats
inside an arm land within +/-0.5%. A resync that costs one core out of
twenty-four and memory bandwidth hurts a 4096x1024 matvec loop much more than it
hurts a synthesis that is already bound elsewhere.

## What phase A found: three false statements, none visible from a Mac

`make x86-tier-parity` FAILED on this box, and it is not run in CI because it
needs a model pack. Under it were four separate defects, fixed in cba879a:

| defect | what it claimed |
|---|---|
| `isa.arm.bf16` | `compiled no` while `matvec_bf16_dpbf16_x1` executed two rows below it |
| same row | with `MYNAH_QMAT_BF16=off`: `IDLE HARDWARE: isa.arm.bf16 ... this binary has no kernel for`, on an AMD EPYC |
| same row, env column | `-`, while `MYNAH_QMAT_BF16` turned it off -- the map's one promise broken |
| `doctor.py` | keeps `resolved`, drops `reason`: an operator saw `isa.arm.bf16 ON` on an EPYC and no sentence explaining it |
| `vnni_off` allowlist | written before `resolved` meant "runs"; with VNNI off the E14-2 avx512bw tier wins and its two rows flip ON |

And separately, from the same box: **gcc 13.3 emits 23 warnings clang does not**,
seven of them `-Waggressive-loop-optimizations` inside `src/kernels.c` and
`src/qmat.c` ("iteration 4611686018427387903 invokes undefined behavior", which
is `SIZE_MAX/sizeof(float)`). No call site can produce such an `n`; the cost is
that seven warnings sit in the hottest files on the PRODUCTION compiler and
cover real ones. Not yet fixed -- see the open question below.

## Acceptance gate

- Phase A: every gate PASS, and the doctor's PREDICTED thread count recorded
  verbatim so phase B can be scored against it.
- Phase B: `/proc/mdstat` shows no resync line in the log the run wrote, or the
  numbers are reported as contaminated and nothing is promoted.
- A configuration is promoted only by the soak's drift gate, never by the wave.
- Numbers land in `docs/performance.md` only if the box was quiet; otherwise
  they stay here, labelled.

## Open, after this session

- The seven `-Waggressive-loop-optimizations` sites. The candidate fix is a
  zero-cost precondition, `if (n > PTRDIFF_MAX / sizeof(float))
  __builtin_unreachable();`, which states the bound the callers already respect
  rather than adding a branch. It needs gcc to verify and there is no gcc on the
  development Mac, so it waits for the box to be free of measurement.
- E14-8, the flag-parity test, would have caught the `isa.arm.bf16` env column
  by construction: it is exactly "which env flag reaches which row".
