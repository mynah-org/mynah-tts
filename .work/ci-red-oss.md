# CI is red on the OSS repo — deferred until the C100 work on Axion lands

Status: **recorded, not started.** The user's instruction is explicit: finish the
C100/Axion work first, then come back to this. This note exists so that when we
do come back we start from evidence instead of from `gh run list`.

## Problem

`gh run list` on `mynah-org/mynah-tts` shows failures. They are **not on `main`**:
every `main` run is green, the last one being the 2026-08-04 push
(`Build & Test`, `Memory Safety`, `Code Quality`, all success). The red runs are
`workflow_dispatch` runs on the `lane/int8-*` branches from 2026-09-13:

| run | branch | result |
|---|---|---|
| 34756085349 | `lane/int8-baseline-control` | success |
| 34756081030 | `lane/int8-baseline-control` | **failure** |
| 34755962323 | `lane/int8-smmla-vnni` | **failure** |
| 34755854918 | `lane/int8-baseline-control` | success |
| 34755089602 | `lane/int8-smmla-vnni` | **failure** |

The same branch both passes and fails, which already says the failure is not a
property of the code under test.

## Evidence

Both inspected failures are the **same job**: `link-only: x86 SIMD=avx512`.
Run 34755962323 fails at the step *"Start, and collect the dispatch table"*, and
the log says exactly what happened:

```
Run ./build/cpu/mynah-tts --version
    ./build/cpu/mynah-tts --dispatch-map
fatal: this binary requires AVX-512F and this CPU does not have it.
Built as SIMD=avx512->avx512 (x86_64, d5a6d8f); the CPU reports: avx2 fma.
Rebuild with a profile this host supports (make SIMD=portable, or SIMD=avx2 for
a travelling x86 binary) -- without this check the next instruction would have
been SIGILL.
```

**The guard is working correctly. The job is wrong.** A job named *link-only*
executes the artifact it built. The GitHub x86 runner has `avx2 fma` and no
AVX-512F, so a binary compiled for AVX-512 cannot run there — and our own startup
check refuses it rather than taking a SIGILL, which is the behaviour we want and
should keep.

Not yet confirmed: run 34756081030 fails in the same job, but its log tail showed
compiler diagnostics around `src/transformer_ar.c:1126` (`TAR_FAIL` macro notes)
before the non-zero exit. Those are `note:` lines, not necessarily the cause.
**Do not assume it is the same failure as 34755962323 until its log is read.**

## Plan (when we pick this up)

1. Read 34756081030's failing step in full and decide whether it is one bug or two.
2. Fix the job, not the guard. Three candidate shapes, in order of preference:
   - a link-only job **does not run** the binary — build and stop;
   - if it must prove the artifact starts, build the *run* step at `SIMD=avx2`
     and keep `avx512` as a compile/link target only;
   - or assert the guard's refusal as the **expected** outcome for a
     cross-profile binary, i.e. the job passes when it prints that fatal line.
   The third is the only one that keeps testing anything, but it tests the guard,
   not AVX-512 code. Pick deliberately.
3. Decide whether AVX-512 code paths get real execution coverage anywhere. Today
   they do not: no runner we use has AVX-512F. The EPYC Zen 5 box that produced
   `docs/performance.md`'s x86 numbers does. If AVX-512 correctness matters, that
   coverage has to come from a self-hosted runner or a manual gate, and saying so
   out loud is better than a green CI that never executed the instructions.

## Acceptance gate

- `gh run list --branch main` green, and a `workflow_dispatch` on a lane branch
  green **twice in a row** (the flapping above means one green proves nothing).
- The AVX-512 coverage question is answered in writing here — either "covered by
  X" or "deliberately not executed in CI", never left implicit.
