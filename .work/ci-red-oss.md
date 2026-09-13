# CI is red on the OSS repo — one job bug, fixed; awaiting a run to prove it

Status: **fix written and committed locally, NOT yet proven.** The job is
guard-aware now and its logic is unit-tested against four outcomes, but the only
thing that closes this is a real workflow run, and that needs a push — which is
never done without asking.

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

**Confirmed, both runs, one bug.** Run 34756081030 fails in exactly the same
place as 34755962323:

```
Run ./build/cpu/mynah-tts --version
fatal: this binary requires AVX-512F and this CPU does not have it. ...
##[error]Process completed with exit code 1.
```

The compiler diagnostics around `src/transformer_ar.c:1126` earlier in its log
are `note:` lines from the `TAR_FAIL` macro expansion, not the cause. It is one
bug in one matrix entry, not two.

## The fix (committed locally)

`.github/workflows/build.yml`, the *"Start, and collect the dispatch table"*
step. The guard is untouched — it was right. What changed is what the job
accepts as a successful start:

- the binary starts and prints what was asked of it, **or**
- it exits non-zero carrying the ISA guard's own message.

Anything else still fails the job. Crucially that includes **SIGILL (132) with
no message**: if the guard ever stops firing, this step goes red instead of
quietly passing. That is the reason the step is not simply
`continue-on-error: true` — that spelling would accept the refusal and a real
crash with the same shrug.

Two details that are easy to get wrong and were:

- `shell: bash` on GitHub means `bash --noprofile --norc -eo pipefail`. Under
  `-e` a failing command substitution kills the script **before** its exit code
  can be read — which is the exact thing this step exists to inspect. The step
  sets `+e` explicitly.
- the guard writes to stderr, so the capture is `2>&1`.

Verified locally against four fakes before committing: clean exit 0 → pass;
guard message with exit 1 → pass, with a `::notice`; `kill -ILL` → exit 132 →
**fail**; unrelated exit 3 → **fail**.

## The question the failure exposed, answered

**AVX-512 code paths are deliberately not executed in CI.** No GitHub-hosted
runner we use has AVX-512F: the x86 runners report `avx2 fma`. The matrix entry
`x86 SIMD=avx512` therefore proves that the profile **compiles and links**, and
now also that the binary **refuses to run** where it cannot — which is real
coverage of the guard, and is all it ever was.

Execution coverage for AVX-512 comes from the EPYC Zen 5 box by hand; that is
where `docs/performance.md`'s x86 numbers (int8 0.427 vs f32 0.806) were taken.
If that ever needs to be continuous, it needs a self-hosted runner. Written down
here rather than left implicit, because a green matrix that never executed the
instructions is worse than an honest gap.

## Still open, tracked, not done

`--self-test` is still missing from this matrix. The reason it was excluded is
gone: the aarch64 strict-aliasing miscompile in `src/qmat.c` was fixed in
`d95773e`, which is an ancestor of HEAD. Adding it is the right follow-through,
but every profile has to be proven green on real aarch64 and x86 hosts first,
and building sixteen configurations on the project's ARM box while the C100
concurrency measurements are running on it would corrupt those measurements.
Do it after Axion, not during.

## Acceptance gate

- [x] both failures traced to one cause, and that cause is the job
- [x] guard untouched; step logic verified against pass/refuse/SIGILL/other
- [ ] `workflow_dispatch` on a lane branch green **twice in a row** (the matrix
      flapped, so one green proves nothing) — needs a push
- [x] the AVX-512 coverage question answered in writing, above
- [ ] `--self-test` added to the matrix, after the Axion work
