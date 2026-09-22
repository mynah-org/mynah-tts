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

## 2026-09-16 — re-checked, and what the local gates say

`gh run list` is **unchanged**: the same five `workflow_dispatch` runs on
`lane/int8-*` from 2026-09-13, and **every `main` run is still green** (last
one 2026-08-04, `724d677`, all three workflows). Nothing newer has run because
nothing has been pushed: the local branch is **140 commits ahead of
`origin/main`**, so none of E10's work has ever reached a runner.

That answers the obvious first suspicion — the red is not the new code, and it
cannot be: the new code has never been on GitHub.

The job fix is still committed-and-unproven for the same reason. It needs a
push plus two consecutive greens, and a push is never done without asking.

### Running the CI jobs locally instead

| CI job | run locally as | result |
|---|---|---|
| Memory Safety / ASan | `make asan` (Accelerate and `BLAS=none`) | clean, exit 0 |
| Memory Safety / ASan, real workload | the ASan binary over a 17 s synthesis, default spec **and** `codec_convtr:int8` | clean — 220 frames, so ~13 KV compactions ran under ASan |
| Memory Safety / UBSan | `make ubsan` + a real synthesis | clean |
| Code Quality / clang-tidy | not installed; `cc --analyze` runs the same clang-analyzer engine | see below |
| Build & Test | five BLAS/SIMD profiles, `make test`, goldens, batch parity, window-test, the server suite on the real pack | green |

### What the analyzer found, and what was done

The Code Quality job runs clang-tidy with **`-warnings-as-errors=''`**, so none
of these can fail it — and the last `main` run was green with most of them
already present. Treated as advisory, and triaged rather than swept:

Fixed, because they are real:

* `src/qmat.c` `cache_insert()` — `e->name == NULL` did `return NULL` instead of
  `goto fail`, leaking the entry it had just `calloc`'d. A genuine leak on an
  OOM path, pre-existing, exactly what `unix.Malloc` is for.
* `src/qmat.c` `self_test_act_quantize()` — the "find the first differing byte"
  loops could index `k` if the invariant that got them there ever broke. Mine;
  the bound is written down now instead of reasoned about.
* `src/transformer_ar.c` — a dead store left by the RoPE-sharing change. Mine.

Not fixed, with the reason:

* `src/engine_pocket.c:1506` "Attempt to free released memory" — **false
  positive**. `pocket_call_init` takes four separate `calloc`s and
  `pocket_call_release` frees each and then `memset`s the struct, so a double
  release frees NULLs. The analyzer does not track the memset.
* `src/transformer_ar.c:332` dead store — pre-existing, part of a uniform
  `cursor +=` assignment chain where the last increment is naturally unread.
  CI disables `deadcode.DeadStores` explicitly. Breaking the chain's symmetry
  to silence a disabled check is a readability cost for nothing.
* ~25 more across `server/`, `src/engine_magpie.c`, `src/json.c`, `src/sgemm.c`,
  `src/costmap.c`, `src/threads.c`, `cli/main.c` — mostly `unix.Stream`,
  `unix.Errno` and `core.uninitialized.Assign` on paths the analyzer cannot
  prove. **Not swept**: they are pre-existing, advisory, and a 25-file
  drive-by in the middle of E10 is how unrelated regressions get in. They are
  worth their own pass, with each one judged rather than silenced.

After the three fixes `src/qmat.c` and `src/convq8.c` analyze clean, and
`src/transformer_ar.c`'s only remaining hit is the pre-existing disabled one.

## 2026-09-17 — GREEN, and what the runners found

`docs/plan-board-pocket-tts` went up as PR #2 and all three workflows are
green on it: **Build & Test 22/22 jobs, Memory Safety 4/4, Code Quality**.

That closes the item this note opened. The `link-only: x86 SIMD=avx512` job
that produced every red run of 2026-09-13 passes: the guard-aware step accepts
a binary that refuses to start with the ISA guard's message, and fails on a
SIGILL or any other exit. It had been committed and unproven since September
because proving it needed a push.

### Four defects, none reproducible on the development machine

The branch was green locally on five BLAS/SIMD configurations, `make test`,
goldens, leaks and UBSan before it was pushed. The runners found four things
anyway, and the pattern in all four is the same: **a configuration that only
exists somewhere else.**

1. **x86 did not compile at all.** `QMAT_F16_BATCH_LANES` was defined inside
   `#if defined(MYNAH_QMAT_F16_NEON)` while the loop using it is guarded by
   `MYNAH_QMAT_F16`, which x86 satisfies through F16C. One bug, three red
   workflows. The f16 batched lanes had been written on an arm64 laptop and
   never built anywhere else.

2. **A test that was flaky by machine.** The census self-test's "clean table"
   case held one f32 matvec, which is exactly the census R3 refuses -- an int8
   kernel resolved and nothing carried it. It passed wherever the int8 kernel
   is `neon-sdot` and failed on a runner whose kernel is `avx512vnni`. **Two
   runners behind the same `ubuntu-latest` label differ in whether they have
   AVX-512 VNNI** (the same label reported `avx2 fma` on 2026-09-13 and
   `avx512vnni` today), so Build & Test passed on one machine while Memory
   Safety failed on another in the same push.

3. **A report that named the wrong kernel.** `tail_sweep` tracked two metrics
   through one `worst_name`, so the run printed "worst reducing kernel:
   1.05e-07 (axpy)" -- axpy being elementwise -- and the elementwise failure
   named no kernel at all.

4. **A metric that measured the data, not the kernel.** `axpy` was checked in
   ULP of `a + 0.375*b`, a sum that cancels; one rounding of an input is a
   thousand ULP of a cancelled output, and whether you get one rounding or two
   is whether the build has FMA. 1024.0 ULP on baseline x86, no defect behind
   it. The file already carried the correct argument for the dot product two
   cases earlier.

Two of the four were *diagnostics that could not describe what they had
found*, and each cost a round trip through CI to guess at. Both now carry the
reason. That is the lesson worth keeping from this day: **on a configuration
you cannot reproduce, the quality of the failure message is the iteration
time.**

### What the CI now covers that it did not

- Both sanitizers on **both architectures**. ASan and UBSan ran on x86 only,
  so the NEON, SDOT and SMMLA kernels -- the ones production executes -- were
  never sanitized on Linux. The arm runner reports `dotprod` and `i8mm` **ON**,
  so what is sanitized there is the shipped path.
- The nine link-only profiles **run the kernels**, not just `--version` and
  `--dispatch-map`. They are the only place `SIMD=portable`, `SIMD=scalar`,
  `BLAS=scalar` and the armv8-a baseline are built at all, and defect 4 lived
  in exactly that gap.
- Every `start` in that step is **checked**. They were bare calls under
  `set +e`, so only the last one could fail the job.
- Each profile also runs the self-test under **`MYNAH_QMAT_VNNI=scalar`**,
  which forces the portable unsigned int8 encoding -- what x86 executes and
  what an arm runner never resolves on its own.

### The GPU backends are compiled now, and that found a fifth defect

`gpu/cuda/backend_cuda.cu` and `gpu/metal/*` were built by nothing, anywhere,
on a repo whose own contract calls them "optional build variants that must be
validated on the target machine". They do not need the target machine to
COMPILE: `nvcc` emits PTX and cubin with no device present and `xcrun metal`
does the same for shaders. Two compile-only jobs, named so nobody reads them
as "the GPU is tested" -- `nvidia/cuda:12.6.2-devel` at sm_70 and sm_90, and
`make metal` on the macOS runner already in the matrix.

**5. `make cuda` could not link on any machine.** The CUDA rule spelled its
libraries out by hand (`-lm -lcublas`) where the CPU and Metal targets link
through `$(LDLIBS)`, so when `third_party/ingot` became a dependency it was
added to LDLIBS and that rule never saw it:

    undefined reference to `ingot_st_open'

Every `make cuda`, on every machine, with or without a GPU, since ingot
landed. The job found it on its first run -- every object compiled, including
the `.cu`, and it died at the link. **No GPU was needed to find a defect that
made the GPU build unusable.**

A second thing worth keeping: `CUDA_ARCH` defaults to `-arch=native`, and
native means "ask the installed GPU", so `make cuda` cannot work on a machine
without one. Every build farm is such a machine. CI names the architecture
explicitly; a release build should too.

### Still not covered, and honestly

- **VPDPBUSD is executed only by luck.** Some `ubuntu-latest` runners have
  AVX-512 VNNI and some do not, and which you get is not selectable. Good
  enough to have caught defect 2; not something to build a claim on. A
  measured x86 number still needs a real x86 box.
- **Nothing runs on a GPU.** The two jobs above prove the sources compile and
  link. They prove nothing about a kernel launching, being correct, or being
  fast, and each job prints that in its own log so the distinction survives
  being read by someone in a hurry.
- **No model pack**, so every gate that needs weights -- the codec int8
  quality gate, the oracle parity, the WAV smoke -- stays local. That is the
  right trade: `models/` is gitignored and should stay so.

## 2026-09-22 — red again, one root cause and two design defects behind it

`main` has been red since the 2026-09-21 push (`7b4abed`). **Seven jobs, one
cause**, and the cause is a line I added the day before:

```
python3 tools/ternary_feasibility.py self-test
ModuleNotFoundError: No module named 'numpy'
make: *** [Makefile:532: ternary-test] Error 1
```

`ternary-test` went into `test:` as a hard dependency. No GitHub runner has
numpy in its system python — not `ubuntu-latest`, not `ubuntu-24.04-arm`, not
`macos-latest` — so every job that runs `make test` died on an import:

| workflow | jobs red | where |
|---|---|---|
| Build & Test | 3 | `linux-x86_64`, `linux-aarch64`, `macos-arm64`, step *Full test target* |
| Memory Safety | 4 | UBSan and ASan, both arches, inside `make ubsan` / `make asan` |
| Code Quality | 0 | it never calls `make test` |

The two link-only failures of 2026-09-19 (`undefined reference to
`self_test_bf16_convert``) and the `Illegal instruction` in UBSan the same
evening are **both already fixed** — by `Fix the scalar build` and by `Stop
MYNAH_QMAT_VNNI=256 granting a feature the CPU lacks` respectively. All twelve
`link-only` profiles are green on `7b4abed`. Nothing else was red.

### Defect 1 — an offline tool became a build dependency

The repo contract is that `tools/` is offline tooling and the runtime has no
Python dependency. `ternary-test` quietly promoted numpy to a requirement of
`make test`, which is a requirement no developer box is told about either.

Fixed by making the target **skip when numpy is absent, loudly**: it prints the
exact command it did not run and how to enable it, per the testing checklist's
"report the exact skipped command and reason". A skip that CI could reach would
be worse than the failure, so CI installs numpy on the jobs that run `make
test`, and the Code Quality *Python tooling* job — which owns the offline tools
and already has `setup-python` — runs the ternary self-test unconditionally,
where the skip cannot reach it.

### Defect 2 — the Memory Safety gate could go red over a Python import

`make ubsan` and `make asan` ran the full `test` target, which includes four
pure-Python gates the sanitizer does not instrument. So a missing Python module
turned the *memory safety* workflow red while saying nothing whatsoever about
memory safety, and green there had always cost four unsanitized Python runs.

`test` is split: `test-c` is the C gates (self-test, kernels, qmat, driver,
window, json, simd-auto) and is what the sanitizers run; `test` is `test-c`
plus the Python tooling. No coverage is lost — Build & Test still runs all of
it — and a sanitizer job can now only be red about C.

### Two real warnings the red run surfaced

Every Linux build has been printing these; they were never read because the
matrix was green.

* `src/dispatch.c` — `pool.inline_fallbacks`'s explanatory text is built in a
  240-byte buffer and gcc measures the worst case at **321 bytes**. The arm
  that overflows is the "this report dispatched nothing, so it has measured
  nothing" sentence: exactly the one whose loss would let a reader quote an
  empty report as a result. Buffer is 384.
* `src/json.c` — `'type' may be used uninitialized` in `scan_value`. A false
  positive (every path that reads it assigns it first, through `scan_scalar`),
  but an unread initializer costs nothing and a standing false positive is how
  a real warning goes unnoticed. Initialized.

Left alone deliberately: the `-Waggressive-loop-optimizations` "iteration
4611686018427387903" notes in `kernels.c` and `qmat.c` (gcc reasoning about a
`size_t` induction variable at 2^62, unreachable with real memory) and the
`-Wformat-truncation` notes on error-message paths in `tokenizer.c`,
`engine_magpie.c` and `seanet.c`, where truncating a path name into a bounded
error buffer is the intended behaviour.

### Verified locally before committing

`make test` green; `make test-c` green; `make ubsan` green; `make SIMD=scalar`
links; both arms of the numpy gate exercised (with numpy: runs and passes;
with a numpy-less `python3` on PATH: skips, exit 0); the `pool.inline_fallbacks`
row prints its full sentence; all three workflow files parse.

### Acceptance gate

- [x] every red job on `main` traced to one cause
- [x] the cause fixed at the root (offline tool is not a build dependency)
- [x] the gate is still real in CI, and cannot silently skip there
- [x] a sanitizer workflow can no longer be red about anything but C
- [ ] one green push on `main` to prove it — needs a push, which is never done
      without asking
