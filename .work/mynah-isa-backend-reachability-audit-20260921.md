# ISA / backend reachability audit — is Mynah portable, or is it Neoverse-V2-shaped?

Status: **OPEN**, started 2026-09-21. Living document — update it when the code
moves, do not let it go stale after the first round of fixes.

Start SHA `857ba32`. All work here is local; the C126 soak on the Axion box was
running throughout and was not touched.

The question is not "does Mynah compile on x86". It is: **for each production
operation, what implementation actually executes on AVX2, AVX-512 without VNNI,
VNNI, AMX, plain NEON, dotprod, i8mm and Axion — and is that implementation
right for Mynah's real shapes, threading and streaming path?**

---

## 0. A premise of the brief that does not hold here, stated first

The brief asks whether Mynah "replaced or bypassed" PyTorch operators, FBGEMM,
oneDNN, XNNPACK, torchao or ATen, and asks which CPU backend those libraries
dispatch to.

**None of them are in Mynah's runtime.** `CLAUDE.md` is explicit — "The runtime
is CPU-first and has no Python dependency at runtime" — and the code agrees:
across `src/*.c`, `server/*.c` and `cli/*.c` the strings `torch` and `PyTorch`
appear **only inside comments**, describing the upstream tensor layout the
converter has to match.

So the entire "which library backend gets reached" axis is a question about
**upstream `kyutai-labs/pocket-tts`**, not about Mynah. Mynah is pure C11 with
its own kernels, its own quantization, its own thread pool and its own server.
Nothing dispatches on our behalf and nothing can silently pick a good kernel for
us — which cuts both ways, and section 2 is where it bites.

This also changes what the upstream comparison is worth. Upstream's "+27% on x86
from dynamic INT8" is a statement about **FBGEMM**, and it transfers to Mynah
only as motivation, never as a number. Likewise PocketTTS.cpp's 9.2x realtime on
a Ryzen 7 3800X is evidence that *Zen2/AVX2 is a viable target for this model*
— which is a useful prior for section 6 — and is not a benchmark of this engine.

---

## 1. The production math path, from source

`mynah_kernels_*` is the wrong prefix to grep for and it returns almost nothing;
the real f32 entry points are in `src/kernels.h` without that prefix. Callers,
verified per symbol:

| f32 kernel | production callers | on the PocketTTS hot path? |
|---|---|---|
| `mynah_layernorm_f32` | `transformer_ar.c`, `engine_magpie.c`, `backend.c` | **yes** — every layer, every frame |
| `mynah_rmsnorm_f32` | `flow_head.c` | **yes** — every frame |
| `mynah_matvec_bias_f32` | `engine_pocket.c`, `flow_head.c`, `transformer_ar.c`, `backend.c` | **yes** |
| `mynah_dot_f32` | `seanet.c`, `qmat.c`, `sgemm.c`, `transformer_ar.c` | **yes** |
| `mynah_residual_add_f32` | `transformer_ar.c`, `backend.c` | yes |
| `mynah_matvec_f32` | `backend.c`, `voice_clone.c` | cloning + magpie |
| `mynah_gelu_f32` | `backend.c`, `engine_magpie.c` | magpie only |
| `mynah_snake_row_f32` | `codec_nanocodec.c` | magpie codec only |
| `mynah_tanh_f32` | **none** | **P3: no production caller** |

The quantized path is separate and lives in `src/qmat.c`, reached from
`engine_pocket.c` through `mynah_qmat_linear_resolved_qt()`.

---

## 2. The central finding: x86 f32 is compile-time, x86 quantized is runtime

This is the whole audit in one table.

| file | what it holds | x86 gating | consequence |
|---|---|---|---|
| `src/kernels.c` | f32 dot, matvec, layernorm, rmsnorm, gelu, axpy — **187 AVX2 intrinsic sites** | a single `#elif !defined(MYNAH_DISABLE_SIMD) && defined(__AVX2__)` | **compile-time only.** AVX2 or scalar, decided when the binary is built |
| `src/sgemm.c` | the f32 GEMM behind the SEANet/codec path — 7 intrinsic sites | same single `__AVX2__` guard | compile-time only |
| `src/qmat.c` | int8 / int4 / f16 / bf16 | **26 runtime-dispatch sites**, `__attribute__((target(...)))` on `f16c`, `avx2,f16c`, `avx512f,avx512bw,avx512vl,avx512vnni` (EVEX), `avx2,avxvnni` (VEX), selected by CPUID | **runtime.** One binary, right kernel per host |

**On Arm this asymmetry is invisible**, because AdvSIMD is architecturally
guaranteed on aarch64: the compile-time choice is always the right one. The
dispatch map says so in as many words — `isa.arm.neon … [gate] compile-time
selection, there is no runtime fallback path`.

**On x86 it is the portability story.** The three profiles behave differently in
a way nobody would guess from the outside:

| build | `kernels.c` + `sgemm.c` f32 | `qmat.c` quantized | ships safely to an older host? |
|---|---|---|---|
| `SIMD=portable` | **scalar** (`__AVX2__` undefined) | still runtime-dispatched to VNNI/F16C | yes, and silently slow on the f32 half |
| `SIMD=avx2` | AVX2/FMA | runtime-dispatched | **no — SIGILLs on a pre-Haswell host** |
| `SIMD=auto` | whatever the *build host* had | runtime-dispatched | no — pins the artifact to the builder |
| `SIMD=avx512` | AVX2 + wider autovectorization only | runtime-dispatched | no |

So today there is **no single x86 binary that is both safe on an old CPU and
fast on a new one**. On Arm there is, for free. That is the sense in which the
engine is Neoverse-shaped: not because a kernel is V2-specific, but because the
ISA baseline made a design shortcut invisible on the architecture we qualified on.

### What it costs, bounded by measurement

`SIMD=neon` against `SIMD=scalar`, this M1, same text/seed/runs, `BLAS=none`:

```
SIMD=neon     RTF 0.128
SIMD=scalar   RTF 0.266      2.08x slower
```

**Read this as an upper bound, not as the x86 portable number.** `SIMD=scalar`
compiles out *every* intrinsic including `qmat.c`, whereas an x86 `portable`
build keeps the quantized kernels and loses only the f32 ones. The true cost of
`portable` on x86 is a fraction of 2.08x — a fraction we cannot measure from an
Arm host, and which section 7 says how to get.

### `SIMD=avx512` is already known not to be a kernel

From the Makefile's own comment: *"no f32 kernel in src/ dispatches on AVX-512
… so it only widens autovectorization, which is an unmeasured change that also
pins the artifact to the build host."* The X2 question ("AVX-512 without VNNI")
is therefore **already answered at the source level for f32**: there is nothing
to reach. For int8 the EVEX VNNI kernel exists but requires VNNI, which
Skylake-SP does not have — so a Skylake-SP host runs **AVX2 f32 + AVX2/SSE
int8**, and AVX-512 buys it only autovectorization.

---

## 3. What is provable without renting anything

Cross-compilation to x86 is **not available on this machine** (no x86_64
toolchain or SDK; `cc -arch x86_64` fails). So local x86 evidence stops at
source reading.

But CI already covers more than expected. `.github/workflows/`:

| job | arch | profile | what it proves |
|---|---|---|---|
| build.yml | **x86_64** | `SIMD=avx2` (explicit, not `auto`) | COMPILE + tests on x86 AVX2 |
| build.yml | x86_64 | `SIMD=scalar`, and a portable job | COMPILE + tests with no ISA flag |
| build.yml | aarch64 / macOS arm64 | `SIMD=neon` | Arm baseline |
| safety.yml | **x86_64** and aarch64 | UBSan, ASan | memory correctness on both |

So **COMPILE VERIFIED and PARITY VERIFIED on x86 are free in CI.** Renting is
only needed for PERFORMANCE and for ISA tiers CI does not have.

Gaps CI does not cover, and cannot be closed by reading source:

- x86 `SIMD=avx512` — not in the matrix at all;
- **runtime VNNI execution** — the GitHub x86 runner pool is heterogeneous, which
  is exactly how the `MYNAH_QMAT_VNNI=256` SIGILL was found: a runner with
  AVX512-VNNI and no AVX-VNNI. Green CI there is a statement about the machine
  that ran, not about the code;
- AMX — no kernel exists (`isa.x86.amx_int8 … NOT IMPLEMENTED`), so nothing to test;
- older Arm tiers — dotprod-without-i8mm and plain NEON are not in the matrix.

---

## 4. Findings

**P1 — x86 f32 kernels have no runtime dispatch** (§2). One binary cannot be
both safe and fast on x86. `qmat.c` already demonstrates the fix pattern in the
same tree: target attributes plus a CPUID probe. 187 intrinsic sites in
`kernels.c` and 7 in `sgemm.c` make this a real piece of work, not a one-liner,
and it should start with the four kernels that are actually hot on the Pocket
path — `layernorm`, `rmsnorm`, `matvec_bias`, `dot`.

**P1 — the dispatch map does not report the consequence.** On an x86 portable
build `isa.x86.avx2` reads `compiled no`, which is true and insufficient: it
does not say that the f32 hot kernels therefore run scalar while the quantized
ones stay vectorized. A reader checking portability would see one honest row and
draw the wrong conclusion.

**P2 — `isa.arm.bf16`'s row understates the batched path.** It describes BFDOT
("eight MACs per instruction"), but `qmat_rows_job` calls
`matvec_bf16_neon_tile` — BFMMLA, sixteen MACs, eight accumulators
(`src/qmat.c:3638,3654`). Anyone reading the dispatch map to judge whether bf16
is competitive at batch would conclude it is half as wide as it is.

**P3 — `mynah_tanh_f32` has no production caller** (§1). Dead on both engines.

**Not a finding: AMX.** `isa.x86.amx_int8` is honestly reported as NOT
IMPLEMENTED, and PocketTTS's shapes (B≤8, M×K up to 4096×1024) are small enough
that a tile reload is unlikely to pay. Do not build it on ISA-availability
grounds; the E12 result — that at B=8 the weight pass is 5% of the step — says
the arithmetic is not where the remaining time is.

---

## 4b. Threading is already portable — and that matters for the verdict

Audited before assuming. The threading layer is the part of this engine most at
risk of being silently Neoverse-tuned, and it is the part that is **most
carefully portable**:

| layer | what it does | portable? |
|---|---|---|
| thread count | macOS: `hw.perflevel0.logicalcpu` (P-cores). **Linux: `sched_getaffinity`**, falling back to `_SC_NPROCESSORS_ONLN` | **yes** — affinity respects cgroups and `taskset`, which is the right answer in a container |
| parallel library | **none.** No OpenMP, no MKL, no library thread pool — one custom pool | **yes**, and it removes the whole nested-parallelism/oversubscription class the brief asks about |
| spin budget | a **time** (`PF_SPIN_TARGET_US`), with the iteration count derived per host from a measured `ns/relax` | **yes, by construction** |
| decoder lane | `MYNAH_LANE_SPLIT`, currently OFF | n/a |

The spin budget deserves the detail because it is the exact failure the brief
predicts, already found and already fixed. It *used* to be
`__linux__ && __aarch64__ -> 65536, everything else -> 4096` — an iteration
count gated on the OS half of a condition whose ISA half decides the duration.
`pf_cpu_relax()` is `yield` on aarch64 and `pause` on x86 and those differ by
about two orders of magnitude, so one integer necessarily meant two very
different waits: "Linux x86-64, which is half of production, ran 4096 `pause`es
that nobody ever measured." It is now a time, calibrated on the host that will
run it.

**This changes the verdict.** The engine is not broadly Neoverse-shaped. Two of
the three layers that could have been — threading and the quantized kernels —
are portable by design and were built that way deliberately. The gap is *one
specific thing*: f32 kernel selection on x86. That is a much more tractable
statement than "it was tuned for Axion", and it is why §4's P1 is worth doing
rather than despairing of.

## 5. Still to do in this track

- [ ] threading audit (§8 of the brief) — `pool.threads`, `pool.decoder_lane`,
      `pool.spin`, worker/thread topology, and whether the 16x2 optimum is a
      property of the model's `a + b·B` or of Neoverse-V2;
- [ ] the Arm capability ladder A1–A5 using `MYNAH_QMAT_I8MM=0/1` and
      `MYNAH_QMAT_BF16` to simulate tiers on this box;
- [ ] the per-stage reachability matrix for the Mimi decoder ops;
- [ ] the one-command qualification harness (§15);
- [ ] memory/cache classification per stage (§9) — E12 already established that
      `a` is **not** pure weight traffic, so the naive bandwidth model is known
      to be wrong by ~2.7x and must not be reused here.
