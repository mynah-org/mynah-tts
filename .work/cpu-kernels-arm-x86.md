# E4 — CPU kernel layer: ARM and x86 delivered together

Status: **OPEN** · independent of E1-E3, but measure before porting

**Scope rule for this epic: CPU only. ARM and x86 are one step, not two.**
No kernel is "done" until both the ARM and the x86 path exist, self-test against
the scalar reference, and appear in a dispatch report. GPU (Metal/CUDA) is
explicitly **deferred** — the existing backends stay as they are, and no GPU work
is scheduled here. On Apple Silicon Metal already measured *slower* than CPU
(`docs/performance.md:71-80`), which is a further reason not to spend on it now.

## Where we stand

| | mynah-tts today | qwen-tts `feature/x86-amx-vnni-oss` |
|---|---|---|
| Kernel LOC | `kernels.c` 517 + `qmat.c` 1194 | `kernels.c` 13395 + `kleidi.c` 1254 + `q8repack.c` 653 |
| ARM | NEON + SDOT | NEON + SDOT + **i8mm/SMMLA** + **BF16/BFMMLA** + KleidiAI |
| x86 | AVX2 only | AVX2 + AVX-512F + **VNNI** + **AVX-512 BF16** + **AMX-INT8/BF16** |
| Quant | int8 per-row, int4 Q4_0-style, f16 | int8, Q4_0, Q6_0, Q2_0 + VNNI/AMX weight prepack with persistent cache |
| Fused ops | `matvec_argmax` | fused QKV, argmax-matvec, wide matmat |
| Thread pool | 158 LOC, condvar, work-stealing | + lane split, deadline priority, `set_width`, `after_fork`, capability predicates |
| Dispatch visibility | none | `--dispatch-map`: compiled/supported/env/resolved/reason |
| Region profiling | env-var dumps | cost-map, inclusive, nesting verified at runtime |

## Two defects to fix first

1. **The README's AVX-512 VNNI claim is false.** `src/qmat.c` has no `_mm512_*`
   and no `_mm256_dpbusd_epi32`; `SIMD=avx512` (`Makefile:29-31`) only passes
   compiler flags. The EPYC Zen 5 int8 number (0.427) is real, the attribution is
   not. **Correct `README.md` and note the correction in `docs/performance.md`
   before adding a real VNNI path**, otherwise the new measurement has nothing
   honest to be compared against.
2. **No dispatch report exists.** Until one does, any ISA claim is unverifiable.
   Port `--dispatch-map` early, not last: qwen-tts added it after a dispatch bug
   hid ~400 ms of TTFA.

## What to take from qwen-tts

These files contain no `qwen_tts.h` include — the `qwen_` prefix is cosmetic and
the signatures are `(rows, cols, B)` generic. Verified, not assumed.

| Take as-is | Why |
|---|---|
| `qwen_tts_kernels.{c,h}` | GEMV/GEMM, fused QKV, attention variants, RoPE, norms, bf16 conversion, int8/Q4_0/Q6_0/Q2_0, VNNI + AMX prepack |
| `qwen_tts_thread.{c,h}` | pool + lane split + deadline priority + `after_fork` + honest capability predicates |
| `qwen_tts_kleidi.c` | KleidiAI micro-kernels, Apache-2.0 |
| `qwen_tts_q8repack.c` | int8 repacking |
| `qwen_tts_dispatch.c` | the dispatch report |
| `qwen_tts_costmap.{c,h}` | region profiler with semantics fixed before numbers |
| `Makefile` `SIMD=` profiles + `ARCH_STAMP` | rebuild-on-flag-change, so stale objects cannot fake a result |

Do **not** take: `qwen_tts.h` (monolithic struct + Qwen `#define`s), the
`strstr` config parser, any `talker`/`code_predictor`/`speech_decoder` file, the
flat root layout.

Rename on import so the namespace is ours, and record upstream commit + license
in a vendoring note.

## Why PocketTTS suits these kernels better than Qwen did

- QKV is **pre-fused in the checkpoint** → the fused-QKV kernels apply directly.
- FFN is plain and non-gated → straight `linear`.
- One flow step per frame → the AR step is a handful of matvecs.
- The SEANet conv stack maps onto the same conv/transposed-conv kernels that
  dominate qwen-tts's decoder.

## Order of work

**Measure first.** Port `costmap` and `dispatch` before any kernel, then profile
PocketTTS end-to-end. The expectation from qwen-tts is that the **conv decoder
dominates** (~40% of wall there). If that holds here, optimizing the transformer
first would be wasted work.

1. `dispatch` + `costmap` + the README correction
2. baseline profile of the PocketTTS path, per region, ARM and x86
3. thread pool upgrade (lane split, priority, `after_fork` — `after_fork` is a
   prerequisite for E5's prefork)
4. the kernel the profile actually names — scalar reference first, then **NEON/SDOT/i8mm
   and AVX2/AVX-512/VNNI in the same change**, with a self-test proving all paths
   numerically equivalent
5. int8 weight prepack with persistent cache, both ISAs
6. AMX-INT8 (Linux/x86 only) — last, it is the narrowest-platform item

## Acceptance gate

Per kernel:
- scalar reference exists and is the correctness oracle
- NEON **and** x86 paths both implemented, both in `make self-test`
- `--dispatch-map` shows `resolved` from the runtime predicate, never re-derived
- benchmark records commit, OS, CPU, ISA, threads, revision, seed, dtype, RTF,
  TTFA, frames/s, peak RSS
- the delta is stated as byte-identical or quantization-affected, per
  `docs/performance.md` convention

Machines: M1 (macOS arm64) and EPYC Zen 5 (Linux x86-64) are the two that exist
today. ARM server (Graviton/Grace) has never been measured — leave it as a gap in
`docs/performance.md` rather than an assumption.

## E4-2 done — 2026-09-12: the dispatch report

`src/dispatch.{c,h}` and `src/costmap.{c,h}`, wired into `CORE_SOURCES` and
`mynah-tts --dispatch-map [--json]`.

**The rule was implemented as the rule, not as a convenience**: `resolved` is
never `compiled && supported`. Every row declares its provenance in the reason —
`[runtime]` when a real predicate was called (`mynah_num_threads`,
`mynah_backend_open`, the self-tests, sysctl, CPUID+XGETBV), `[gate]` for
compile-time dispatch with no fallback, `[predicate]` when a module registers
its own, and `[UNKNOWN]` when nobody exports one.

The sharpest of those is `mynah_qmat_cache_new(qtype)` +
`mynah_qmat_cache_enabled()`: that is **qmat's own answer**. Ask for f16 on a
build without the NEON converts and the cache silently downgrades to f32 — the
report says so.

**Eight rows resolve to UNKNOWN**, and each names the wrapper that would fix it.
That is E4's real to-do list, not a gap to paper over:
`mynah_cpu_matvec_mode()` (the row that would hide a rows=1 regression),
`mynah_qmat_argmax_mt_resolved()`, `mynah_qmat_cache_row4()`,
`mynah_qmat_cache_qtype()`, `mynah_blas_owned()`,
`mynah_gelu_vector_enabled()`, `mynah_conv1d_sgemm_enabled()`,
`mynah_snake_vector_enabled()`. **UNKNOWN is never replaced by a guess.**

There is also a **drift canary**: the mirrored f16 gate is compared at runtime
against what `qmat.c` actually does, and prints `DRIFT` if per-TU flags ever
diverge.

### The false claim can no longer survive

On x86 with `SIMD=avx512`, `isa.x86.avx512f` reports `compiled=yes` but
`resolved=OFF` with "COMPILER FLAG ONLY: no `_mm512_*` intrinsic exists anywhere
under src/", and `isa.x86.avx512vnni` reports NOT IMPLEMENTED **and cites the
EPYC 0.427 figure as AVX2**. A future attribution error has to get past this
report first.

### Cost map

Regions are declared for *this* runtime with append-only ids and gaps to grow
into, inclusive semantics with `self = ns - child_ns` derived at report time,
nesting declared statically **and verified at runtime**, zero malloc (thread-local
blocks from a static array, one relaxed `fetch_add` to claim a slot), and no
per-region atomics on the hot path. **No hooks are placed yet** — each id carries
the call site it is waiting for, because `inference.c` and the engines were being
refactored in parallel.

It reports its own health: unbalanced regions, leaked regions from an early
return, nest mismatches and exhausted thread slots, each as a counter plus a
warning, so a broken measurement announces itself instead of producing a
plausible number.

### Coverage limit, stated

x86 was verified by **real x86_64 codegen** through the universal SDK (Mach-O
objects, every `#if defined(__x86_64__)` actually compiled, CPUID and XGETBV
included) plus a forced compile of the aarch64-Linux branch. A true
`--target=x86_64-linux-gnu` build is not possible on this machine — there is no
Linux sysroot — so **Linux coverage needs CI or a sysroot**, and is not claimed.

## 2026-09-19 — `MYNAH_QMAT_VNNI=256` could grant a feature the CPU lacks

CI died with **`Illegal instruction (core dumped)`** in the x86 UBSan job, in the
fourth of `qmat-test`'s five invocations. Not a sanitiser diagnostic: a real
SIGILL, and a bug that had been in the file for months.

```c
enum { QMAT_U8_OFF = 0, QMAT_U8_SCALAR = 1, QMAT_U8_VEX = 2, QMAT_U8_EVEX = 3 };

if (strcmp(env, "256") == 0) return detected >= QMAT_U8_VEX ? QMAT_U8_VEX
                                                            : detected;
```

The comparison treats the enum as a ladder. **The block twenty lines above says
in as many words that it is not one**: "EVEX and VEX are INDEPENDENT features,
not a ladder. Ice Lake server has AVX512-VNNI and no AVX-VNNI; Alder Lake has
AVX-VNNI and no AVX-512; Zen 4/5 have both."

On a host with EVEX and no VEX, `detected` is `QMAT_U8_EVEX` (3), which is
numerically above `QMAT_U8_VEX` (2), so the request was granted and the process
executed a VEX `VPDPBUSD` the silicon does not implement. That breaks the
invariant the same function documents: *"A level the CPU cannot run is clamped
down, never up."*

`qmat_u8_vex_level()` already existed and answers the VEX question **alone** —
it was written for exactly this and used only by the self test. The clamp now
calls it. A host with EVEX and no VEX gets `QMAT_U8_OFF`, the AVX2 madd path:
slower, correct, and executable.

### Why it surfaced now

Nothing in the change that exposed it touches VNNI. GitHub's x86 runner pool is
heterogeneous: the invocation had been passing on runners that happen to have
AVX-VNNI, and this push landed on one that does not. That is worth writing down
on its own — **a green CI on a heterogeneous runner pool is a statement about the
machine that ran, not about the code.** The dispatch report prints what resolved
on each run for this reason; what was missing was anyone reading it when the
answer changed.

### The shape of the mistake

An enum whose ORDER encodes capability, next to a comment explaining that the
order means nothing. Both were written by the same hand on the same day. The
comment was right and the code did not follow it, which is the failure mode a
comment cannot prevent — only a call to a predicate that cannot be misread can.
