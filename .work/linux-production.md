# Linux production — what qwen-tts learned, and what we should copy

Status: **REFERENCE** · extracted 2026-09-12 from
`qwen-tts@feature/x86-amx-vnni-oss` (HEAD `51f8d6f`).

Production is Linux server, x86-64 and ARM64. This note is the shortest path
from "works on an M1" to "measured on the target", written from a codebase that
already made the trip. Companion: [serving-doctrine.md](serving-doctrine.md)
for the priority order, [serving-design.md](serving-design.md) for the
mechanisms.

## Correction to a number this plan was anchored on

We recorded "C20/C22 on 32 ARM cores" as the calibration point. **That is not
what their repo measures.** On GCP c4a-highcpu-32 (Neoverse-V2) C20/C22/C24 all
*fail* the gate, control and experimental alike. The qualified points are:

| host | cores | ISA | qualified |
|---|---|---|---|
| **AWS c8a.8xlarge** EPYC Zen5 | 32 | VNNI+BF16 | **C12** mandatory, C11 preferred, C16 hard cap |
| AWS **Graviton5** Neoverse-V3 | 32 | i8mm/BF16 | **C4** |
| GCP c4a-highcpu-32 Neoverse-V2 | 32 | i8mm/BF16 | **none promoted** |
| GCP c4-standard-24 Xeon | 12 | AMX | C4 |
| GCP c4-highcpu-16 Xeon | 8 | AMX | C2 (1.7B) / C3 (0.6B) |

So the honest anchor is **C12 on 32 x86 cores with a 0.6-1.7B model**, and ARM
is materially behind x86 on their stack. Ask the author before trusting either
number — they may be remembering a looser gate — but do not plan against C20 on
ARM without checking.

## The prediction that matters most for us

Across all three ISAs the marginal cost per stream is **the per-slot decoder,
not weight bandwidth**:

- Zen5, one CCX: `T_frame(B) ≈ 40 ms + 13.5 ms × B`; decoder **9.7 ms per
  slot-frame = 72% of the frame**. Talker and CP cost the same at 2, 4 and 8
  threads — the weight stream saturates a CCX at 2-4 threads.
- AMX 12c: decoder call 59 → 89 → 117 ms for B=1/2/3, i.e. **+29 ms per slot**.
- Neoverse-V3: **90-93%** of decoder time in the conv stack.
- The architectural verdict: the decoder runs 303 GMAC over 96 frames at
  ~45 GMAC/s per core, about **5% of Zen5's VNNI peak**, because attention,
  RoPE, layer-norm, depthwise conv, residual add and the final conv are scalar
  on the loop thread, with **50-120 pool dispatches per decoder call per item**.
  It is **glue-bound, not compute-bound**.

Their physics ceiling says C28-32 assuming a free decoder; measured is C12. **The
gap between C28 and C12 is the glue.**

For us: our SEANet decoder is causal, CPU-side, and we just made its conv stack
36× faster with GEMM — which moves us *toward* the same regime, where the limit
becomes dispatch count and round-trips rather than sharper GEMM. Independently,
our own thread-pool work measured ~20 µs of wake-up per region, so a region must
be worth ≥200 µs to pay for itself. **Same conclusion from two directions.**

## The traps, ranked by what they cost

### 1. BLAS oversubscription — 21% of time in the scheduler
`perf` on Neoverse-N1 showed 21% in `__sched_yield` + `__schedule`: their pool
(4 threads) and OpenBLAS (4 threads) on 4 cores. Worse, the engine **never
called** `openblas_set_num_threads`, so on a 64-core server with `-j4` it was 4
of ours against 64 of theirs.

The fix is three-part and the third part is the one people skip:
1. weak symbols so macOS still compiles;
2. **structural ownership**: with ownership on, OpenBLAS is pinned to 1 thread
   and SGEMMs are sliced on the engine pool. Their earlier version *returned
   early* if `OPENBLAS_NUM_THREADS` was in the environment — "two schedulers in
   one process, decided by whether someone remembered a variable". The engine
   now wins and reports the override;
3. **claim ≠ fact**: `blas_own_effective()` is true only if ownership is on
   *and* the symbol resolved. On Accelerate it is always false, and the dispatch
   report says "requested, but this build has no BLAS thread control".

**Corollary that inverts the obvious**: lowering BLAS threads improves RTF and
**worsens TTFA by 30%**, because prefill is all BLAS with no concurrent decoder.
A rigid `engine + BLAS = cores` partition is wrong; slight oversubscription
wins. They ended up with a per-phase knob.

### 2. OpenBLAS spins when idle — one environment variable
`OPENBLAS_THREAD_TIMEOUT=1`, pinned in every profile:

| | without | with |
|---|---|---|
| TTFA C=1 | **108 ms, bimodal** | 66 ms, stable |
| context switches/s | 42,500 | 12,000 |

Bimodal means a single run looks definitive whichever value it draws. Best
value-per-effort line in their repo. And `OPENBLAS_NUM_THREADS` must be
**absent**, not set — their harness refuses to start if it is present.

### 3. Planning on the machine instead of the mask
`sysconf(_SC_NPROCESSORS_ONLN)` sees neither an inherited `taskset` mask nor a
cpuset cgroup, so **a pinned run silently escaped its mask and a container
planned the whole host**. Plan from `sched_getaffinity`. Note the residual they
did *not* close: the cgroup **quota** (`cpu.max`) is still not read at runtime.

### 4. pthreads inherit the creator's affinity
Set the mask **before the pool exists**. Their prefork child order is fixed:
`close(listen_fd)` → `sched_setaffinity` → `threadpool_after_fork` →
`costmap_after_fork` → `lane_split_prepare` → `set_threads`. A nastier corollary:
`--prefork 1` skipped the prefork path entirely, so "1x8" meant eight threads
free to roam the host — on a multi-CCX part, a *different bandwidth domain*, not
a smaller one.

### 5. Contiguous logical slices are wrong under SMT
Linux numbers the first thread of every core before any sibling, so on a 12c
SMT-2 host `--prefork 2` gave both workers **the same twelve physical cores, one
per hyperthread**. Fix: read
`/sys/devices/system/cpu/cpuN/topology/{physical_package_id,core_id}` and reorder
core-major. And print the mask **actually set**, not the slice indices.

### 6. Bandwidth belongs to the mask, not the machine
c8a.4xlarge (2 CCX × 8c): triad ~113 GB/s host-wide, but a process confined to
cpu 0-7 gets **~54 GB/s**, and 8-15 likewise. **A worker roof can never be
derived by dividing a host roof.** On Zen5 4-CCX the cache-resident rate across
two CCX (40.5 GB/s) is *below* its own DRAM rate (109.9): a working set split
across two L3s thrashes. That is the physical reason wide pools collapse, and
why `1x32` measures STREAM 1.40-1.55.

### 7. glibc's mmap threshold — 78 mmap + 154 munmap per request
im2col scratch of a few hundred KB per call crosses glibc's mmap threshold, so
every chunk pays page faults and zero-fill. Measured per request at C1:
`posix_memalign` **11,899**, `free` 12,516, `mmap` 78, `munmap` 154, `futex`
30,309. A per-stream bump arena took it to **41 / 624 / 1.3 / 1.3**, output
bit-identical. They fixed it with a data structure, not an allocator swap.
Warning attached: do not blindly pre-reserve large arenas, it trades a small
first-use saving for multi-worker RSS growth.

**This one is a prediction for us**: our codec allocates scratch per call and on
macOS it does not show, because the allocator is different.

### 8. A lost wakeup that ARM cannot see
In a spin-then-sleep pool, publishing `generation` with release and then reading
`sleeping` outside the mutex is the store-buffer litmus. On ARM64 `stlr`→`ldar`
orders it **and the bug never manifests**; on x86 the store sits in the store
buffer past the load, the dispatcher skips the broadcast while the worker sleeps,
and it deadlocks. **We develop on ARM.** Their verdict on the fixed spin pool
was that it bought ~3% — "the diagnosis was right, the cure was secondary: the
first move is not to oversubscribe".

### 9. Spin counts do not port
`QWEN_POOL_SPIN` is 65536 on Linux/aarch64 and 4096 elsewhere. On ARM 16c, 4096
cost **40% of the Code Predictor** (491,320 → 35,132 context switches). On x86
8c the ARM value is *worse*. On Graviton5 the curve is **non-monotonic**. No
rule, only measurement: pin the measured value per box.

### 10. Build breaks invisible to modern hardware
Their tree did not link when neither dotprod nor VNNI was defined — seven
symbols declared and called unconditionally but defined inside the ISA guard,
with the fallback `#else` *inside the same guard*. It broke `SIMD=portable`, the
default non-VNNI x86 target. Their fix includes a CI job whose comment is the
lesson: **"link-only on purpose: this job guards the guard."**

### 11. Mixed binaries from reused objects
`make blas` then `make blas ARCH_FLAGS=...` reused every object and produced a
binary with portable kernels and a native dispatch. Fix: a **stamp file** of the
effective flags that every object depends on, rewritten only when the value
changes, with a `sleep 1` because GNU make 3.81 compares whole seconds.

### 12. libmvec — the grep proposes, the hardware disposes
It looked like Linux paid scalar `expf`/`sinf` where macOS had Accelerate. `nm
-D` showed `_ZGVnN4v_expf@GLIBC_2.38`: **gcc with `-ffast-math` auto-vectorizes
into glibc's libmvec**, and `sinf` never reaches 1% of profile. A hand-written
transcendental kernel would only help with **clang** (which does not emit
libmvec), musl, old glibc, or without `-ffast-math`. We develop with clang on
macOS — so this is exactly the kind of thing we would "fix" for no reason.

## What to copy, in order

**1. A fatal ISA guard.** We have `cpu_has_avx2()` but nothing that *checks*:
a `-mavx2` binary on a CPU without AVX2 gives an opaque SIGILL. Theirs exits
with an actionable message, before any allocation, and `--caps` warns. ~15 lines.

**2. The `link-only` CI job.** Our x86 CI builds the default `-mavx2` and
nothing else — precisely the configuration in which their tree shipped unlinkable
for days. Copy it verbatim, plus `ARCH_FLAGS=-march=armv8-a` on the ARM runner
we already have.

**3. The flag stamp file.** Eight lines, and it prevents a mixed binary.

**4. BLAS from "the weak symbol exists" to structural ownership**, plus
`OPENBLAS_THREAD_TIMEOUT=1` in the deployment env set. Our GEMM fast path in the
decoder goes through BLAS — that is exactly where the two pools collide.

**5. The pool must plan from `sched_getaffinity`**, and agree with the prefork
planner. Add what they did *not* do: read cgroup v2 `cpu.max` and at least warn.

**6. `SIMD=auto` on x86 should read `/proc/cpuinfo`**, with the double test
(kernel flag **and** `cc_ok`) and a line that prints the resolved profile. For
us the value is in VNNI and AVX-512 BF16; AMX costs an order of magnitude more
work and on their numbers the AMX 8-core box does C2 while the VNNI Zen5 32-core
does C12. `-march=native` is fine on Linux ARM and macOS, **never** on x86 meant
to travel.

**7. Before executing prefork**, check that costmap mutexes are *reinitialized*
and not merely zeroed (an inherited locked mutex can never be unlocked), and that
no GPU backend state is created before the fork — that is their still-open bug:
`--backend cuda --prefork N` is silently broken, a wrong answer rather than a
crash.

**8. An arena in the codec, before measuring.** See trap 7.

**9. A dispatch gate and a host profile that refuses to run.** We have the
report; what is missing is a preflight that compares resolved against a declared
expectation and **fails**, and a profile listing the variables that must be
**absent**. Their proof that this is needed: the correct configuration was
already versioned for the box, and **three campaigns were run wrong anyway**
from memory, both workers on the same twelve physical cores. *"The JSON was
right and the experiment was wrong."*

**10. The measurement discipline.** Only **paired A/B, adjacent, same box, after
warm-up**. Their "+8% TTFA" vanished when measured in adjacent pairs. And the
labelling habit from their cost model: every constant tagged `MEASURED` /
`TRANSFERRED` / `GUESS`, plus a table of points where the model was
**falsified** — `1x32 B8 C8 → measured 1.40, predicted 0.50, FALSIFIED: one
32-thread pool collapses; model blind to it`.

## Their rule for a new box
> **"Never start at I on an unknown box."**

Qualification runs A→J: box fingerprint (SMT off, governor, **no cgroup quota** —
otherwise *"you are measuring the QUOTA, not the machine"*), preflight, self-test,
dispatch check, then C1, then C4, then soak. Optimizing before that is
measuring an unknown.
