# The method — how the reference implementation avoids fooling itself

Source: qwen-tts `ENGINEERING.md` (222 lines) and `docs/ENGINEERING-METHOD.md`
(226 lines, lineage declared: Abrash's *Graphics Programming Black Book*), plus
the profiling toolchain in `docs/cpu-profiling.md`.

This is the part the author meant by *"c'è moltissima engineer là"*. It is also
the cheapest thing in this whole exercise to adopt, because almost none of it is
code.

---

## 1. The four opening lines

> **Do not optimize the code. Optimize the work.**
> **Do not trust expected behaviour. Observe the machine.**
> **Do not celebrate a faster component until the system is faster.**
> **Every unexplained millisecond is an engineering question.**

With the disclaimer that prevents the usual misreading:

> This is not a low-level-programming aesthetic. Hand-written SIMD is not the
> point, and preferring it for its own sake would be the opposite of the lesson.
> The rule exists to make us **more suspicious, especially of results we like.**

---

## 2. The organising principle

> The agent must not remember the project; the repository must make it impossible
> to forget. `PLAN.md` says what remains, `ENGINEERING.md` says how we work, the
> `.work/` addenda hold a task's detail, and **the program itself proves which
> path it is executing.**

That last clause is the load-bearing one, and it is why their dispatch report,
shape census and cost map all exist.

### Rules worth transplanting, with the reason attached

**Separate plan from evidence.** Every addendum opens with a fixed skeleton:
`Task · Question · Known facts · Unknowns · Files inspected · Evidence ·
Conclusion · Next action`. An addendum never becomes a second global plan. We
already adopted the board/detail split; the skeleton is the missing half.

**Plan integrity is checked by a script.** `tools/check_plan.py` fails on a
missing `.work/` file, a duplicate task id, a `[x]` task pointing at a file that
does not exist, or an addendum naming a task the plan does not have. *"Never
create placeholder links or empty addenda for work that has not been written."*
Ours would already catch one or two stale links.

**Backend claims require a matrix.** Never call a change "cross-platform", "ARM"
or "x86" without reading its gates. Mandatory classification: **A** common
runtime · **B** common design, backend-specific implementation · **C**
backend/ISA-specific. This is the rule our own memory note already gestures at.

**A benchmark is invalid until dispatch is proven.**

> Never infer the active kernel from a profile filename, a requested env flag or
> the build target name. The strongest evidence is in-process: the server's
> resolved table at startup and the kernel census after warm-up, compared against
> an expected/allowed/forbidden manifest.
>
> An explicit request is a request, not a capability. If an explicitly requested
> path differs from the resolved one: **stop, do not run the benchmark. A
> fallback run is never reported as a measurement of the requested path.**

**Fallbacks must be visible.** Every significant optimised operation reports which
implementation ran, from at least: native optimised · project generic · external
BLAS · scalar/f32 fallback · unsupported. *"A silent fallback in a qualification
run is a failure."*

**Never benchmark a contradiction.** Their examples: AVX-512 BF16 requested on a
binary built without `-mavx512bf16`; AMX on a topology whose batch never reaches
the AMX gate; **server batching parity demonstrated with the CLI's paragraph
splitter.** Whenever a contradiction is mechanically detectable, add a preflight
gate that refuses the run.

**Explicit terminology.** WAVE (screening) · SOAK (the production gate) · POISSON
(overload) · DIAGNOSTIC (profiling, never a headline number). **Never a bare
"RTF".** And: *"do not call any of these simply 'concurrency N'. A threshold from
one does not transfer to another."*

**No opportunistic scope expansion.** While executing X do not start Y, do not
change numerical precision, do not change production defaults, do not build a new
profiling framework.

**Numerical changes are separate work.** *"An optimisation that changes the
arithmetic or the output is not a structural optimisation. It needs its own
quality qualification and is never promoted on performance numbers alone."*

**What is committed is what was built.** Canonical SOAK evidence comes from a
clean committed tree; a dirty-tree binary is labelled NON-QUALIFYING. *"Both the
dispatch incident on GCP and the broken-HEAD discovery were evidence produced by
an artifact different from the source state we believed we had."*

**One implementation owner; parallel agents are evidence suppliers.** Dated, with
the incident:

> **Parallel agents are never parallel sources of truth.** On 2026-09-06 two trees
> evolved independently for a day. One held ~60 commits the user's branch did not
> have, so a plan he had been told existed was invisible in the checkout he was
> reading; the other had accumulated ~1261 lines of uncommitted work that any
> fast-forward would have destroyed. Nothing was technically lost, and the day was
> lost anyway.
>
> Canonical state is a **HASH**, never "roughly the latest branch". Before
> delegating, emit `CANONICAL_HEAD=$(git rev-parse HEAD)` and require the analyst
> to prove `HEAD == CANONICAL_HEAD`, clean tree, zero source diff.
>
> Analysts return EVIDENCE / INTERPRETATION / CONFIDENCE / CONTRADICTIONS /
> RECOMMENDATION. **Read the evidence before the recommendation. A contradiction
> an analyst surfaces is priority evidence.**
>
> Never block on an analyst when the hypothesis is narrow, reversible, parity-
> testable and cheap to benchmark.

**Completion rule.** A task is complete only when the final report contains:
`WHAT CHANGED · WHAT PATH ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS UNKNOWN
· VERDICT: PROMOTE / KEEP / INCONCLUSIVE / REJECT`.

---

## 3. The method's own failures, listed by the method

Their §1 opens by listing mistakes they made, which is the reason to trust the
rest:

- two unrelated benchmark numbers fused into a causal story that was never
  measured;
- a harness that made INT8 look slower because its preparation work was
  serialised while the other arm's was not;
- the conclusion reversing once the harness was fixed;
- a *correct* result that still carried a threading confound until a third arm
  was added.

Rules that follow from that:

- **A label is not a causal explanation.** `prefill`, `decode`, `GEMM`, `queue`,
  `conv_up` — *"it is another box to open."*
- **Every optimisation needs a cost model before code**: current cost · suspected
  cause · proposed transformation · **maximum plausible saving** · new work
  introduced · risk · **the smallest experiment capable of killing the idea**. If
  the maximum plausible saving is too small for the product goal, **do not
  optimise**.
- *"Before every experiment ask: **which control would make my preferred
  explanation look stupid if it were wrong?**"*
- *"A ten-line change that removes 15 ms is better engineering than a
  thousand-line subsystem that removes 17."*
- The claim chain: `microbenchmark → component → critical path → server →
  concurrency → sustained workload → quality`. **Never skip a level.**
- *"Do not spend twenty minutes proving what a twenty-second component
  measurement could refuse."*
- The **engineering ledger**: FACT · HYPOTHESIS · TEST · RESULT · DECISION ·
  CLAIM SCOPE. *"Rejected ideas are evidence. Do not silently rediscover and
  re-run them."*

---

## 4. Every tool declares a refusal

This is the most copyable property in the entire toolchain, and it requires no
kernel work:

- the **shape census** exits non-zero if any operation resolved UNKNOWN, or if a
  feature resolved ON whose class never executed;
- the **bandwidth roofs** return `ROOF UNKNOWN` (nothing of that kind measured) or
  `NOT COMPARABLE` (wrong mask, or right mask but different residency) instead of
  dividing. *"There is **no** WORKER-to-HOST fallback anywhere in the path. A
  percentage is never printed without checking the scopes."*
- the **cost map** rejects its own numbers if `nest_mismatch != 0`, if the region
  stack overflowed, or if there are unbalanced `end`s;
- the **doctor** prints `[UNKNOWN]` — *"nothing here supports a number, and the
  doctor says so instead"* — and labels every value `[MEASURED]` / `[CACHED]` /
  `[TRANSFERRED]` / `[PREDICTED]` (±10-25%) / `[UNKNOWN]`;
- the **dispatch gate** aborts qualification on a resolved fallback, and flags
  `SUSPICIOUS` = expected ON, compiled, supported, resolved OFF with no explicit
  env — *a fallback nobody asked for*. Its `--selftest` **replays the historical
  incident** and must print `SELFTEST PASS`.

The sentence that generated the whole toolchain:

> A fast kernel does not make an efficient engine. The two biggest x86 wins of
> 2026-09 were both **around** the kernels — a dispatch predicate silently
> choosing f32/SGEMM prefill on an AVX-512-BF16 host (~400 ms of TTFA), and **a
> transpose that cancelled itself out** — and neither was visible in any existing
> report.

### Cost map semantics, fixed before a single number was collected

Worth reading before we write ours (`qwen_tts_costmap.h:11-21`):

- **inclusive**; self time is **derived** as `ns − child_ns`, never measured
  separately, "so inclusive and exclusive can never be silently mixed";
- **nesting declared statically and verified at runtime** — a `begin` whose
  dynamic parent is not the declared one increments `nest_mismatch` instead of
  being silently accepted;
- thread-local accumulation, merged only at dump: no atomics on the hot path;
- one `clock_gettime` per begin and end, **never inside an inner kernel loop**;
  when off, a marker is an int load and a branch;
- **two levels**, so per-layer sites do not tax the default;
- ids are **append-only**: an old binary's report must keep its meaning;
- regions crossing threads are marked `"mode": "derived"` vs `"mode": "stack"`
  and **the report prints the column**, so the two are never read as one measure;
- shares are given **within a thread role** — *"the roles do **not** sum to 100%
  of the request, and any report presenting them as one flat breakdown **would be
  lying**"*;
- **parity proof**: the load is run twice with the same binary — census only, then
  census + cost map — and the two census reports are diffed. *"Without that proof
  the numbers would be unfalsifiable."*
- **overhead measured A/B/B/A interleaved**, never a clean run followed by an
  instrumented one: *"drift between two identical runs can be larger than the
  effect."* Measured: worst p50 overhead **+1.5%** while the clean arms themselves
  scattered by **9.1%**.

---

## 5. The conversion hunt — the method applied, with the numbers

Their two biggest x86 wins, as an illustration of what the tooling is for. Our own
hunt is E4 work and running separately; these are the *shapes* to look for.

**a. A single-threaded dtype conversion in front of a parallel GEMM.** On an
AVX-512-BF16 host without AMX, prefill fell back to BLAS, which converts every
weight matrix to f32 first — and that conversion is **single-threaded**, so it
became a serial stage in front of a parallel GEMM. Invisible in the flags banner
for weeks; found by source audit.

| | fallback | native |
|---|---|---|
| C1 TTFA p50 | 423 / 410 ms | **125 / 123 ms** |
| C4 TTFA p95 | 950 / 928 ms | **516 / 498 ms** |
| admission + prefill stage | 2844 / 2831 ms | **1122 / 1130 ms** |

STREAM_RTF and absolute step times unchanged → the entire delta is one-time
prefill.

**b. The transpose that cancelled itself out.** One function built `xT[k*B+b]`
from `X[b*in+k]` (scalar transpose); the AVX-512 kernel it fed then built
`Xb[b*cols+k]` from `xT` — undoing it. The composition is the identity. It existed
only because two backends want opposite layouts. Audio **bit-identical** after the
fix; C4 TTFA p50 −5.4%, admission+prefill **−10.1%**. Their self-criticism is the
useful part: *"engagement was proven in a separate run with identical env, not
inside the A/B. Next time the engagement counter goes inside the timed run."*

**c. FP32 materialisation before int8 quantisation.** Activation preparation as a
share of `prep + compute`:

| projection | B1 | B2 | B4 |
|---|---|---|---|
| large model, one projection | 28.4% | 42.8% | 61.6% |
| **small model, same projection** | **52.8%** | **70.3%** | **79.5%** |

**The smaller the model, the more of the time is preparation rather than
arithmetic.** PocketTTS is smaller than either. This is the single most relevant
number in their whole profiling corpus for us: it predicts that our int8 win will
be gated by activation preparation, not by the dot product, and that the fix is to
quantise each contiguous source row directly instead of materialising a transposed
f32 panel first. Theirs kept output **bit-identical** and removed 68-85% of
preparation cost.

**d. Repeated f32→bf16 packing of identical activations.** Q/K/V repacked the same
normalised activation three times, gate/up twice. One packed LHS now feeds a whole
projection group, *"because the packed LHS layout depends on K and B, not on the
weight matrix"*. Paired A/B consistent in both orders; C1 WAVs byte-identical. The
same commit also closed an **instrumentation lie**: missing markers had work
reported as UNACCOUNTED — *"a hole in the instrumentation, not evidence of idle
workers."*

### The asymmetry checklist — nine classes to grep for

1. feature implemented but **wrong default/gate** (worth ~400 ms above);
2. **an extra conversion/transpose/repack present on only one architecture**;
3. automatic on one, behind an env flag on the other;
4. fused on one, several calls on the other;
5. different thread-pool / barrier behaviour;
6. different scratch allocation / copy / gather / scatter;
7. persistent representation on one vs per-call on the other;
8. different batch/shape thresholds;
9. **a generic fallback silently winning dispatch against the better kernel.**

Where to look: every `*_available()` predicate, every gate row with an *opt-in*
env instead of an opt-out, every `#if defined(__ARM_...)` with no `__AVX512...`
sibling, and every ISA-specific call whose counterpart is a **different function**
rather than the same one.

---

## 6. What we should adopt, and what it costs

| item | cost | why now |
|---|---|---|
| addendum skeleton (`Task · Question · … · Next action`) | zero | we already split board/detail |
| `check_plan.py` equivalent | ~30 lines | we have link rot already |
| WAVE / SOAK / DIAGNOSTIC terminology, no bare "RTF" | zero | our numbers are currently unlabelled |
| dispatch gate with `--selftest` replaying our own incident | small | our known risk: a capability reported ON where it does not exist |
| fallback visibility on every optimised op | small | we have silent f16/BLAS paths today |
| cost-map semantics fixed **before** collecting numbers | design only | cheaper now than after |
| the completion rule in every task report | zero | |
| cost model before optimising (max plausible saving) | zero | would have caught the 36x conv win being the wrong target for serving |

The last row is not hypothetical: a 36× on one region took our RTF to 0.245 and
the machine still served two or three real-time streams, because the limit was
never that region. A cost model would have said so before the work, not after.

---

## 7. The local build lies once a second (macOS, 2026-09-14)

A self-test failed on source with no modification in it — `grep -c MUTANT` was
zero and the failure text named a real disagreement at batch 3. Deleting one
object file and rebuilding made it pass, three times.

Apple ships **GNU Make 3.81**, whose file-time comparison is **whole seconds**.
An edit that lands in the same second as the object's last build is not newer by
make's reckoning, so it is silently skipped and the binary keeps the old code
with the new source on disk. The tighter the edit-build-test loop — a scripted
`python3 - <<PY ... PY && make && ./binary --self-test` in one command is exactly
tight enough — the likelier it is.

**It cuts both ways, and the second way is worse.** A stale object can fail a
test that should pass, which is loud and self-correcting. It can also *pass* a
test that should fail, which is silent: a mutation test that reports "the gate
caught it" may have run the unmutated binary, and a gate that reports PASS may
have run yesterday's kernel.

Rules, then:

- any measurement or gate that decides something gets a `make clean` first, or
  at minimum an `rm` of the objects under test;
- a mutation test is only evidence if the mutated build is *observed* to differ
  — same-second builds make "it failed as expected" unfalsifiable;
- when a result is absurd, suspect the build before the code. This is already
  rule 1 of the global method note and it was still not the first thing checked.

None of this applies to the Linux box, which has a modern make — which is its
own trap, since the platform where the numbers are taken is not the platform
where this bites.
