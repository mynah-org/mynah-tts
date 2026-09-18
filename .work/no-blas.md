# Removing BLAS — decision, and the surface it actually covers

**Decision (author, 2026-09-12):** we do not want OpenBLAS, and ideally no
external BLAS at all. The reason is not performance, it is **ownership**:
OpenBLAS ships its own thread pool with its own policies, and every one of those
policies is a trap you have to remember to check, on every host, forever. The
same removal is a planned future task on qwen-tts.

This note records how big the job actually is here. The answer is much smaller
than it is there, and it gets smaller still if we do it in the right order.

Related: [`linux-production.md`](linux-production.md) §BLAS oversubscription,
[`cpu-kernels-arm-x86.md`](cpu-kernels-arm-x86.md).

---

## 1. The surface, measured by grep, not by memory

**One BLAS function is used in this repo: `cblas_sgemm`. Three call sites.**

| site | what it serves | reached by |
|---|---|---|
| `src/backend.c:377` | the backend vtable's `sgemm` | **everything** that goes through `mynah_graph_sgemm` — `conv1d.c` (2), `engine_magpie.c` (7) |
| `src/backend.c:177` | `matmul_block`, row-blocked multi-row matmul split over our pool | Magpie prefill/encoder |
| `src/seanet.c:53` | `sea_sgemm`, the conv/convtranspose GEMM fast path | **the PocketTTS decoder**, 2 call sites (`:295`, `:538`) |

Plus a fourth thing that is not BLAS but travels with it: the **BNNS filter cache**
in `conv1d.c`, Accelerate-only, therefore macOS-only, therefore not on the
production path at all.

### What this means for PocketTTS specifically

The PocketTTS graph does **not** go through `mynah_graph_sgemm`. Its backbone and
flow-head linears go through `mynah_qmat_linear` / `mynah_qmat_linear_batched` —
our own kernels, our own ISA dispatch, our own pool. Checked:
`engine_pocket.c:622,631,639,682`.

> **The entire BLAS dependency of the PocketTTS production path is
> `sea_sgemm`, two call sites in `seanet.c`.**

**Correction (2026-09-12, from building it):** that holds for the *quantized*
path. `engine_pocket.c:1166,1237` can still reach `mynah_backend_matmul` for an
unquantized projection, so `sea_sgemm` is the whole f32-GEMM surface only when the
groups are doing their job.

Everything else in the table above is Magpie's, and Magpie is not what we are
optimising for Linux servers.

That is the finding that changes the size of this task. "Remove BLAS" sounds like
replacing a library; here it is **writing one f32 GEMM for two known call sites.**

---

## 2. Why it is load-bearing today

Two reasons, and only one of them is real.

**Real:** the SEANet conv stack was 76.4% of wall at RTF 2.03. Rewriting it as one
GEMM per kernel tap took it from 8570 ms to 237 ms — **36×** — and that GEMM is
`cblas_sgemm`. Delete BLAS naively and we give back the single largest win in the
repo.

**Not real, but decisive in practice:** the fallback is a naive `i,j,p` triple
loop (`backend.c:386-399`). It is a *correctness* reference, not an
implementation. So "no BLAS" today does not mean "slightly slower", it means
"unusable", and that is the only reason the dependency looks structural.

Replacing the fallback with a real blocked GEMM is therefore the whole task. There
is no other coupling.

---

## 3. The shapes we have to be good at — measured, and not what I assumed

**Correction.** This section previously said the shapes were "skinny and long",
that `n` large meant the inner loop had plenty of work and the packing would
amortise, and that this was explicitly not the square-GEMM case where a general
BLAS earns its keep. The first and third points survive. **The second is wrong**,
and the histogram says so.

Instrumented over one PocketTTS utterance, 43 frames, **1075 GEMM calls, 11
distinct shapes** (counts are `43 x kernel_size`; 25 GEMMs per frame, 22 conv1d
and 3 convtranspose):

| calls | site | m | n | k | transA |
|---|---|---|---|---|---|
| **301** | conv1d | 512 | **16** | 512 | 0 |
| 129 | conv1d | 64 | 480 | 128 | 0 |
| 129 | conv1d | 32 | 1920 | 64 | 0 |
| 129 | conv1d | 128 | 96 | 256 | 0 |
| 129 | conv1d | 1 | 1920 | 64 | 0 |
| 43 | convtr | 512 | 480 | 128 | 1 |
| 43 | convtr | **3072** | **16** | 512 | 1 |
| 43 | convtr | 1280 | 96 | 256 | 1 |
| 43 | conv1d | 64 | 1920 | 32 | 0 |
| 43 | conv1d | 256 | 96 | 128 | 0 |
| 43 | conv1d | 128 | 480 | 64 | 0 |

The single most frequent shape — **28% of all calls** — is `m=512, n=16, k=512`,
and the widest is `m=3072, n=16, k=512`. That `n=16` is the **frame batch**, not
a long axis. These are narrow-RHS matmuls, much closer to a handful of matvecs
than to a panel GEMM, and no packing cost amortises over sixteen columns.

**Consequence for `mynah_sgemm_f32`: `n=16` and `n=1920` are two different
kernels, not one tuned shape.** The `n=16` family wants the weights streamed once
with the sixteen columns held in registers — which is the same weight-stationary
argument the batching lane makes for the codec transformer, on the same tensors.
The `n=480/1920` family is an ordinary panel GEMM.

Note also `m=1, n=1920, k=64`: 129 calls of a single output row. That is a matvec
wearing a GEMM's clothes and should never reach a packed kernel at all.

## 3b. The inventory, as of 2026-09-13 — author: remove OpenBLAS for good

The author's position is now unconditional: **OpenBLAS should leave the process
permanently, and anything it was doing we write ourselves for x86 and ARM.** This
section is the map of what that actually means, because "BLAS" has come to cover
three different things in this tree and only one of them is OpenBLAS.

### Tier 1 — real BLAS. Two call sites left, and both already have our kernel.

| site | what | status |
|---|---|---|
| `src/backend.c:219` | the vtable `sgemm` | `mynah_sgemm_f32` under `BLAS=none` |
| `src/seanet.c:163` | `sea_sgemm`, the codec conv/convtranspose fast path | same |

`matmul_block` no longer calls cblas at all — it went through `mynah_sgemm_f32`
when `src/sgemm.c` landed. So the OpenBLAS dependency proper is **two lines**, both
already switchable, and `BLAS=none` links neither OpenBLAS nor Accelerate today.

**What stood between here and deleting it was nothing but a measurement**, and
§3c is that measurement. `BLAS=none` is the Linux default as of 2026-09-13;
`BLAS=openblas` survives as a comparison build and nothing else.

### Tier 2 — the thread-pool compensation machinery, which is pure OpenBLAS tax

Listed in §4 below. All of it exists because a second thread pool lives inside our
process. When OpenBLAS goes, every line of it goes with it, and so does
`OPENBLAS_THREAD_TIMEOUT=1`, the `OPENBLAS_NUM_THREADS`-must-be-absent rule in
every profile, and the 21%-of-time-in-the-scheduler oversubscription trap.

### Tier 3 — Accelerate-only, NOT BLAS, and therefore not on the Linux path at all

This is the part that gets conflated. These are macOS vector-library calls with no
OpenBLAS equivalent and no presence on the production target:

| function | count | where | what we owe it |
|---|---|---|---|
| `BNNSFilterCreateLayerConvolution` + apply/destroy | 8 | `conv1d.c`, `conv1d.h` | the macOS conv1d filter cache |
| `vvtanhf` | 4 | `kernels.c` | GELU-tanh over an array |
| `vvsinf`, `vDSP_vsmul`, `vDSP_vsq`, `vDSP_vsma` | 12 | `codec_nanocodec.c` | the SEANet Snake activation |

On Linux these already fall to our own scalar/NEON code, which is why the Linux
build works at all. **They matter only to the macOS development experience**, and
one of them is a trap for the flip: dropping Accelerate swaps `vvtanhf` for libm's
`tanhf` in the GELU, and on a continuous-AR model that compounds. That is a
separate numerical qualification and must not ride along on a GEMM decision.

### So the honest order of work

1. ~~measure `BLAS=none` against `BLAS=openblas` on the Axion box~~ — **done,
   §3c. f32 +22.6%, int8 a tie, 15-vs-8 and 63-vs-32 threads in an 8-cpu
   worker;**
2. ~~flip the default, delete Tier 2 entirely~~ — **done, §3d and §4;**
3. qualify the `vvtanhf` substitution on its own, then drop Accelerate too and
   write the Snake and the conv filter path ourselves for both ISAs;
4. keep `BLAS=openblas` as a comparison build forever, so the A/B never stops
   being available and never becomes the default again.


## 3c. The measurement that unblocked the flip — Linux ARM, 2026-09-13

**Taken, and it says flip.** GCP Axion (`instance-20260912-085327`), 32
Neoverse-V2 cores, SMT off, Ubuntu 7.0.0-1011-gcp, gcc 15.2, `SIMD=auto` →
`-march=native`. Source shipped with `git archive HEAD` into a directory of its
own; no object file crossed from macOS. The host was otherwise idle: load
average 1.20 at the start, nothing above 0.2% CPU except one interactive
`htop`, and the same `htop` for both arms because the arms are interleaved.

`models/pocket-en`, one utterance, speaker 0, seed 42, `--max-steps 100` →
6.240 s of audio, identical in both arms. Worker shaped like production:
`taskset -c 0-7` with `MYNAH_THREADS=8`. `OPENBLAS_NUM_THREADS` absent, as the
old rule required.

**Protocol.** This repo has been burned by a loaded machine before, so nothing
here is an absolute. Arms interleaved **A/B/B/A** within a round, seven rounds
per profile; each arm invocation is itself `--warmup 1 --runs 5` and the CLI's
own median, so a single cold run cannot become an arm's number; the deciding
statistic is the **within-round ratio**, and what is reported is the median of
the seven ratios with the full range beside it.

**There are two honest answers, because there are two OpenBLAS builds**, and
reporting only the first one would flatter us. Run 1 compares against the
OpenBLAS build **as it ships at the pinned head**, clamp machinery and all.
Run 2 compares against the OpenBLAS **comparison build as it stands after this
change**, with the clamp deleted — the forward-looking A/B, and the one that
stays available.

| | profile | ratio OpenBLAS / ours, median | range | RTF OpenBLAS | RTF ours |
|---|---|---|---|---|---|
| run 1, 7 rounds, vs the build that ships today | f32 | **1.226** | 1.208 – 1.247 | 0.416 | **0.339** |
| run 1 | int8 | 1.017 | 0.982 – 1.031 | 0.124 | **0.122** |
| run 2, 5 rounds, vs the comparison build after the flip | f32 | **1.091** | 1.087 – 1.102 | 0.370 | **0.338** |
| run 2 | int8 | **1.026** | 1.014 – 1.035 | 0.123 | **0.121** |

Read honestly:

- **f32: we win in both framings, and never lose a round** — 12 rounds across
  two runs, not one of which went to OpenBLAS. +22.6% against what ships today,
  **+9.1% against the comparison build going forward**, and the second number
  is the one to quote. That margin is the SEANet decoder's `sea_sgemm`, which
  is where the f32 GEMM surface of the PocketTTS path lives.
- **int8: a tie in run 1, a small consistent win in run 2** (+2.6%, every round
  positive). Either way the gate is "not worse" and it passes. This is the
  expected shape: with int8 the backbone and flow head are on
  `mynah_qmat_linear`, so there is much less f32 GEMM left for the choice of
  GEMM to change.
- **Our arm is the same number in both runs** — RTF 0.3394 and 0.3381, a 0.4%
  difference — so the whole gap between the two framings is the OpenBLAS arm
  moving: 0.4156 → 0.3702. **Deleting our own clamp made OpenBLAS 11% faster.**
  See the thread census below; the two facts have one cause.

**Discipline note, because this lane nearly published a bad number.** A third
run of the run-2 comparison was taken while another lane was compiling on the
same box — `cc1` at 95%, load average 12 — and it produced a B-arm outlier
(2.307 against 2.10–2.15) that would have shown up as noise in a ratio. It was
thrown away and re-taken behind a guard that refuses to start until load1 < 1.0
twice in a row and prints the load and the top intruder after every round. Run
2 above is the guarded one: the only process on the box besides ours was an
idle `htop`. **The box is shared. Check it, do not assume it.**

### The ownership number — and it turned out to be worse than the objection

Threads live in one worker, sampled every 5 ms from `/proc/<pid>/status`,
`taskset -c 0-7`, three repeats per cell. Every cell was **identical in all
three repeats**; this is not a noisy measurement.

| build | `MYNAH_THREADS` | peak threads in an 8-cpu worker |
|---|---|---|
| `BLAS=openblas`, clamp present (pre-flip) | unset | **63** — 32 ours, **31 OpenBLAS's** |
| `BLAS=openblas`, clamp deleted (post-flip) | unset | **39** — 32 ours, 7 OpenBLAS's |
| `BLAS=none` (the new default) | unset | **32** — ours only |
| `BLAS=openblas`, clamp present | 8 | **15** — 8 ours, 7 OpenBLAS's |
| `BLAS=openblas`, clamp deleted | 8 | **15** — same |
| `BLAS=none` | 8 | **8** |

Row one is the author's "OpenBLAS creates 32 threads inside a worker pinned to
8 cpus", reproduced exactly.

**Row two is the finding, and it inverts the story the machinery told about
itself.** Deleting the clamp did not make the oversubscription worse. It made
it **better, by 24 threads.** The reason is that the two pools disagree about
what a cpu is:

- OpenBLAS sizes its team from `sched_getaffinity`. Left alone inside a mask of
  8 cpus it creates **8** threads. It was right.
- our pool sizes itself from `sysconf(_SC_NPROCESSORS_ONLN)`, which sees the
  machine and not the mask, so it creates **32**
  ([`linux-production.md`](linux-production.md) §3 — a real bug, still open,
  and **not this lane's**).
- the clamp then called `openblas_set_num_threads(mynah_num_threads())` — that
  is, it overwrote OpenBLAS's correct answer with our wrong one, and turned a
  7-thread foreign pool into a 31-thread one.

So the compensation machinery was not merely unnecessary overhead. It was
**on, working exactly as designed, and quadrupling the problem it existed to
prevent** — and no test in this tree could see that, because the only way to
see it is to count threads in a pinned process.

**It cost time as well as threads.** The same deletion that took the foreign
pool from 31 threads to 7 also made the OpenBLAS build 11% faster: RTF 0.416
with the clamp, 0.370 without it, our own arm unchanged at 0.338 across both
runs. We were paying an 11% tax to make the oversubscription worse.

That is the whole argument for not having a second pool in the address space,
and the machinery made it for us. Note what it is **not**: it is not "OpenBLAS
is slow". Left alone, OpenBLAS sizes itself correctly and comes in 9% behind
our kernel on these shapes. The failure was ours — it was well-intentioned,
carefully written, and carried a comment explaining why it was right — and it
was wrong in a way only a thread census could reveal. A second pool in the
address space is a permanent opportunity to make exactly this mistake.

### Which direction the output moved

The task's own warning, and it is the right one: when output changes on a flip,
check the direction before calling it a regression. Same utterance, seed 42,
60 steps, compared against the in-tree scalar reference (`BLAS=scalar`: naive
triple loop plus the scalar SEANet conv), which is the definition of
correctness:

| profile | | max abs | samples differing |
|---|---|---|---|
| default (f32) | **ours** | **1 LSB** (3.05e-05) | **0.14%** |
| default (f32) | OpenBLAS | 67 LSB (2.05e-03) | 23.53% |
| int8 | **ours** | **1 LSB** | 0.16% |
| int8 | OpenBLAS | 1 LSB | 0.16% |

On f32 the flip moves the output **67× closer to the reference**, which is the
same direction §5b found against Accelerate on macOS (8 LSB / 18.5%) and now
confirmed against OpenBLAS on Linux ARM. On int8 the two are indistinguishable,
for the same reason the RTF is: there is very little f32 GEMM left to differ
about.

**What was NOT measured, and must not be inferred:** x86. The flip is taken on
ARM evidence plus the ownership argument; the EPYC box has to repeat the A/B
before any x86 RTF claim is made. `docs/performance.md` carries the ARM numbers
only.

---

## 3d. The flip, and the two things that did not ride along (2026-09-13)

`BLAS ?= auto` now resolves to **`none` on Linux** and stays **Accelerate on
macOS**. `BLAS=openblas` and `BLAS=accelerate` are explicit comparison builds
and are kept forever, so the A/B above can always be re-run; `auto` no longer
probes the host for `cblas.h`, which is what made the production build a
property of the build machine.

**Refused, explicitly, both of them:**

1. **`vvtanhf` stays.** Dropping Accelerate would swap vForce's `vvtanhf` in the
   GELU for libm's `tanhf`, and on a continuous-AR model that compounds over
   the utterance. That is a numerical qualification with its own oracle run,
   it belongs to the kernels lane, and it is not a GEMM decision. macOS
   therefore keeps Accelerate as its default.
2. **The BNNS conv filter cache stays.** Same reason, same lane.

Neither is a production concern: Linux never had either of them.

**One thing the flip nearly broke, silently.** Three fast paths outside the GEMM
call sites were keyed on `MYNAH_USE_OPENBLAS` when what they meant was "a real
GEMM exists in this build" — the conv1d per-tap GEMM accumulation, its packed
tap cache, and the x86 `rows=1` parallel matvec. Flipping the Linux default
would have switched all three **off on the production target**, with no test
able to see it, because the PocketTTS path goes through `seanet.c` and never
touches them. They are keyed on `MYNAH_HAVE_SGEMM` (`sgemm.h`) now, so the
question is asked in one place. `BLAS=scalar` is deliberately still excluded:
there the serial SIMD matvec and the scalar conv *are* the point.

---

## 4. What removal deletes as a bonus — **DONE, 2026-09-13**

All of this existed only to compensate for a thread pool we do not own, and all
of it is gone. `git show` the flip for the exact lines; the list is kept because
it is also the checklist that says nothing was missed:

- ✅ `src/threads.c` — `PF_HAVE_BLAS_KNOB`, the weak `openblas_set_num_threads`,
  `blas_apply`/`blas_region_enter`/`blas_region_leave` and the `blas_want`
  computation in `mynah_parallel_for`, the `OPENBLAS_NUM_THREADS`-wins rule, the
  `mynah_blas_owned()` predicate that answers "did the clamp actually happen?",
  `mynah_blas_set_threads()`, and `mynah_blas_after_fork()` together with the
  `g_blas_mu` re-initialisation that `mynah_threadpool_after_fork()` owed it —
  a mutex that only existed to guard a foreign thread count, and a fork handler
  that only existed to repair that mutex. Also the `__attribute__((constructor))`
  that set `OPENBLAS_THREAD_TIMEOUT`, and both of its accessors;
- ✅ `src/threads.h` — the three declarations and the thirty-line comment block
  explaining why the claim they made was deliberately weak;
- ✅ `src/dispatch.c` — **five** rows, not three: `blas.accelerate`,
  `blas.openblas`, `blas.scalar_fallback` (all three answered by
  `sgemm.provider`, and answered better, because it names what RAN), plus
  `blas.threads_owned` and `blas.thread_timeout`, one of which was an *unknown*
  because the honest answer was "we cannot tell from here". Also the second weak
  `openblas_set_num_threads` reference this file kept for its own probe. The
  report still resolves **0 UNKNOWN**;
- ✅ `src/backend.c` — the `mynah_blas_set_threads(mynah_num_threads())` call on
  every codec GEMM, whose comment said it existed because that path "bypasses the
  pthread pool";
- ✅ the Makefile's `cblas.h` probe and its three-way `Accelerate / OpenBLAS /
  scalar` split. `BLAS` is now one flat five-way table with an `$(error)` on an
  unknown value and on a platform mismatch, instead of nested `ifeq`s whose
  result depended on what was installed on the build host;
- ✅ from [`linux-production.md`](linux-production.md): `OPENBLAS_THREAD_TIMEOUT=1`
  (TTFA 108 ms **bimodal** → 66 ms stable, 42.5k → 12k context switches/s) and the
  21%-of-time-in-the-scheduler oversubscription trap. Both marked superseded
  there, and **the "`OPENBLAS_NUM_THREADS` must be absent" rule is withdrawn** —
  a profile no longer has to prove the absence of an environment variable to be
  valid, which was the whole objection.

That last one is the point the author made: it is not that OpenBLAS is slow, it
is that **a second thread pool inside our process is a permanent source of
non-reproducible measurements**, and every profile has to carry env vars that
must be *absent* to be valid.

---

## 5. Sequencing — do it in this order, not the obvious one

1. **int8/int4 first.** Every linear that moves to qmat leaves the f32 GEMM
   surface. On PocketTTS that has already happened for the backbone and the flow
   head; the remaining f32 is the SEANet decoder. Quantising the decoder shrinks
   the problem before we solve it.
2. **Dump the shape histogram** (§3). Two call sites.
3. **Write `mynah_sgemm_f32`** — packed, register-blocked, dispatched on our own
   pool, NEON + AVX2/AVX-512 + scalar reference, `--self-test` against the triple
   loop. This replaces `backend.c`'s fallback, so it also serves Magpie for free.
4. ~~**Flip the default to `BLAS=none`**~~ — **done 2026-09-13, on ARM
   evidence (§3c).** `auto` is `none` on Linux and Accelerate on macOS;
   `BLAS=openblas` and `BLAS=accelerate` are comparison builds and stay.
5. ~~**Delete** the compensation machinery in §4~~ — **done, §4.**

Steps 3 and 4 are separable from everything else in E4/E5 and do not block them.

---

## 5b. Landed — the kernel, not the default (`fa3df67`)

`src/sgemm.c` exists: four families chosen by one function the runtime and the
report both call, a scalar reference as the definition of correctness, and a
narrow/panel boundary **derived** from the register file rather than chosen.
`BLAS=none` is a first-class build that links neither Accelerate nor OpenBLAS.
The default is unchanged and byte-identical.

**The numerical result inverted the expected direction.** Isolated against the
in-tree scalar convolution reference, with two controls proving the GEMM is the
only thing that changed:

| | max abs | samples differing |
|---|---|---|
| **our GEMM**, f32 | **1 LSB** | 0.12% |
| **our GEMM**, int8 | **1 LSB** | 0.16% |
| Accelerate, f32 | 8 LSB | 18.5% |
| Accelerate, int8 | 360 LSB | 94.5% |

Twenty to nine hundred times closer to the reference than the incumbent, which has
been shipping since E3-4.

**Two things the default flip must not ride along on.** Dropping Accelerate also
drops vForce's `vvtanhf` in the GELU for libm's `tanhf` — on a continuous-AR model
that compounds, and it is a separate numerical qualification. And the BNNS filter
cache in `conv1d.c` goes with it. Neither is a GEMM decision.

**A false gate worth remembering:** `make ubsan` and `make ubsan BLAS=none` wrote
to the same directory, so the second run re-tested the first one's
Accelerate-linked binary and printed a green PASS on top. The sanitizer directory
now carries the BLAS name and the run prints its link count as proof.

---

## 6. Acceptance gate — result, 2026-09-13

- ✅ a default `make` on Linux **ARM64** links neither `-lopenblas` nor
  `-framework Accelerate`; `make info` says `BLAS=none/mynah-sgemm` and
  `ldd | grep -c openblas` is 0. **x86-64 not verified on hardware** — the
  Makefile change is architecture-independent and CI covers the link, but no
  x86 binary was built for this flip;
- ⚠️ **SEANet oracle parity NOT RE-RUN.** `make oracle-pocket` needs
  `uv run --with pocket-tts` and the gated upstream weights, neither available
  from this lane. Covered instead by the scalar-reference comparison above,
  which is a stronger statement in the direction that matters: the flip moves
  the waveform **towards** the in-tree reference, by 67×;
- ✅ `mynah_sgemm_f32` self-test against the scalar triple loop passes on every
  compiled ISA — `sgemm.selftest PASS`, worst relative deviation 1e-05 on ARM
  NEON, 7.1e-07 on macOS, in every build including `SIMD=portable`;
- ✅ **RTF not worse:** ARM, +22.6% f32 and a tie on int8 (§3c), paired
  interleaved rather than a WAVE screen. ❌ **x86 not measured** — the EPYC box
  has to repeat §3c before any x86 claim;
- ✅ the dispatch report loses **five** rows, not three, and resolves
  **0 UNKNOWN** on both platforms. `sgemm.provider` now reads
  *"mynah_sgemm_f32 (src/sgemm.c, neon kernels) serves cpu_sgemm and
  matmul_block. No external BLAS is linked into this process"* on the Linux
  default.

**The open risk, resolved:** the note warned that a hand-written GEMM beating
OpenBLAS on the general case is hard, and that if the measurement said we lost,
the answer was to keep the comparison build and say so. The measurement did not
say we lost. It also did not need to: nothing in §3c's f32 win is a claim about
general GEMM — it is a claim about eleven measured shapes, 28% of them
`m=512 n=16 k=512`, where a packed panel kernel has nothing to amortise over
sixteen columns and our narrow family does not try to.

---

## 7. Gates run for the flip

Both machines, every BLAS configuration, `SIMD=auto` **and** `SIMD=portable`
(the profile that exposed the qmat strict-aliasing miscompile on this box):

| gate | macOS (M-series, clang) | Linux ARM (Axion, gcc 15.2) |
|---|---|---|
| clean build, warnings | 3 before / 3 after (`BLAS=none` 1) | 22 before / 22 after, byte-identical sets |
| `--self-test` | PASS | PASS in all 7 build configurations |
| `make goldens` | PASS, 8 checks + batch parity | PASS, 8 checks + batch parity, in `BLAS=none`, `SIMD=portable`, `BLAS=openblas` |
| `make window-test` | PASS | PASS |
| `make driver-test` | PASS | PASS |
| `make json-test` | 2819 checks, 0 failures | 2819 checks, 0 failures |
| PocketTTS default + int8 | — | both, `BLAS=none` / `openblas` / `portable` |
| UBSan | PASS, Accelerate **and** `BLAS=none` | PASS, default, `BLAS=openblas`, `SIMD=portable` |
| ASan | — (macOS uses `leaks`) | PASS |
| `leaks` | 0 leaks, Accelerate **and** `BLAS=none` | — |

Build configurations exercised on Linux: default(`none`), `BLAS=openblas`,
`BLAS=scalar`, `SIMD=portable`, `SIMD=portable BLAS=scalar`,
`SIMD=portable BLAS=openblas`, `SIMD=portable EXTRA_CFLAGS=-march=armv8-a`.

### Left for other lanes — deliberately, with the reason

1. **`server/prefork.c:224`, the fork-safety audit, is now stale.** It names
   `g_blas_mu` as "REINITIALIZED by `mynah_blas_after_fork()`", and neither the
   mutex nor the function exists any more. That audit is the answer to E5-13
   and a future reader will go looking for a repair that is gone, so the line
   has to be updated — but `server/` is not this lane's to edit, and an audit
   comment is exactly the kind of thing that should be changed by whoever owns
   the invariant. The same paragraph's closing remark about "a lock inside a
   library that spawned threads of its own" is also moot on the default build
   now. **Handoff to the serving lane.**
2. **`PLAN.md` E4-16 / E4-16a still read as open.** The board is the author's,
   not this lane's; E4-16 is done and E4-16a is deleted rather than done.
3. **The pool still sizes itself from `sysconf(_SC_NPROCESSORS_ONLN)`**, which
   is why `BLAS=none` still puts 32 threads in an 8-cpu worker. That is
   [`linux-production.md`](linux-production.md) §3, it is a real bug, and it is
   now the *only* remaining source of oversubscription in a pinned worker.
   Removing OpenBLAS did not fix it and was never going to — but it did stop
   doubling it.

**Two pre-existing warnings this lane did NOT fix**, recorded so the next
reader does not mistake them for the flip: `src/seanet.c:661,662` `kernel` and
`stride` are unused under `BLAS=scalar` (both platforms, present at the pinned
head). They belong to whoever next touches that file.
