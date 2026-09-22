# What ../qwen-tts has that Mynah does not, and which of it is worth taking

Written 2026-09-22, before renting an x86 box, to answer one question: is there
kernel work already done next door that would make the rented hours count for
more? Read as source, not as reputation — every claim below is a line number.

## The headline, and it is the opposite of what I expected

**Mynah is AHEAD of qwen-tts on runtime dispatch, and BEHIND it on kernels.**

`qwen_tts_kernels.c` is 14,372 lines and contains **exactly one**
`__attribute__((target(...)))` — on the startup guard, not on a kernel
(`qwen_tts_kernels.c:675`). Every math kernel there is gated by a build macro:

```c
int qwen_avx2_int8_gemv_compiled(void)  { #if defined(__AVX2__) return 1; ... }
int qwen_avx2_int8_gemv_supported(void) { return __builtin_cpu_supports("avx2"); }
int qwen_avx2_int8_gemv_enabled(void)   { return env && compiled && supported; }
```

So qwen needs a build per ISA exactly as Mynah's f32 half did until today. Its
`compiled/supported/enabled` triad is a good idea and Mynah already expresses
the same three facts, in the dispatch map's `compiled / supported / resolved`
columns, where an operator can read them out of a running process instead of
out of a header.

What it does have is **kernels we have not written**.

## Worth taking, in order

### 1. VDPBF16PS — the x86 bf16 matvec. TAKE THIS FIRST.

`qwen_tts_kernels.c:1210 bf16_matvec_dpbf16`, plus `qwen_loadu_pbh` (:1192) and
`qwen_f32_to_bf16_row` (:1197).

Why it matters here more than it does there: **bf16 is Mynah's shipped backbone
dtype.** On Arm the bf16 path is worth +33% at B=8 through BFMMLA, measured.
On x86 our bf16 kernel is `_mm256_slli_epi32` widening under AVX2 — correct, and
not using the instruction that exists for exactly this. `src/qmat.c` says so
itself: *"The x86 VDPBF16PS kernel is unwritten."*

**The layouts already agree.** qwen takes `const uint16_t *W` indexed
`W + o*in_dim` — plain row-major bf16, which is precisely what Mynah's bf16
cache holds, and Mynah's own comment predicted it: *"Weights are kept bf16 in
the cache in plain row-major, the same layout x86's VDPBF16PS will want."* So
this is a port of an inner loop, not a change of data layout — the expensive
half of a kernel port is already done.

**Port it the Mynah way, not the qwen way.** qwen's is `#ifdef __AVX512BF16__`,
so it needs a build flag and a separate artifact. Ours goes behind
`__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))` with the
CPUID probe `src/qmat.c` already runs for VNNI — one binary, right kernel.

Caveat to measure and not assume: AVX512-BF16 is Cooper Lake, Sapphire Rapids
and Zen 4 onward. Zen 3 and Skylake-SP do not have it. The AVX2 widening stays
as the path for everything older, so this adds a tier rather than replacing one.

### 2. AVX-512BW int8 GEMV **without** VNNI — the Skylake-SP tier

`qwen_avx512bw_int8_gemv_compiled` (:6360) and its kernel. Mynah today has AVX2
int8 and VNNI int8 and nothing between, so an AVX-512-without-VNNI host — the
X2 tier the ISA brief asked about by name — runs the AVX2 kernel and the AVX-512
registers sit idle. This is the cheapest way to answer that question with code
rather than with an opinion, and it is one self-contained function.

### 3. AMX-INT8 — real, and still not worth it here

74 `_tile_loadd` / `_tile_dpbssd` / `_tile_stored` sites. It exists and works
over there. **Do not port it yet**, and the reason is a Mynah measurement, not
a shortage of code: E12 established that at B=8 the weight pass is 5% of the
step, so the arithmetic is not where the remaining time is, and PocketTTS's
shapes (B<=8, M x K up to 4096 x 1024) are small enough that a tile reload is
unlikely to pay. Revisit only if a measured x86 profile says otherwise.

### 4. KleidiAI is vendored there (`third_party/kleidiai`) and is NOT a copy

It is Arm's micro-kernel library with its own build, its own packing formats
and its own licence. Taking it is a dependency decision for the whole runtime,
not a kernel port, and Mynah's Arm side is the half that is already good
(BFMMLA, SDOT, SMMLA, all runtime-probed). Out of scope for the pre-rental
work; worth its own item if Arm ever becomes the bottleneck again.

## Two small things worth stealing outright

**`qwen_check_runtime_isa` is compiled for the baseline** —
`__attribute__((target("arch=x86-64")), noinline)` at :675. The point is subtle
and correct: if the guard is compiled with `-march=native`, the compiler may
emit an AVX-512 instruction *inside the guard itself*, and the binary dies
before it can print the message the guard exists to print. Mynah's
`mynah_dispatch_isa_guard` (`src/dispatch.c:402`) has no such attribute and is
compiled at the build's baseline like everything else. It has never bitten us —
`tests/x86_cross.sh` now proves the guard fires correctly on a real no-AVX2
host — but the hazard is real and the fix is one attribute.

**`qwen_ftz_on()` (:59) sets flush-to-zero on both ISAs** — FPCR bit 24 on
aarch64, MXCSR 0x8040 on x86. Mynah deliberately *reports* whether the build
flushes denormals (`mynah_vecmath_denormals_flush`) and never sets it, which is
the more honest default. Keep the default, but note that a denormal storm in
the flow head or the codec is a real x86 cliff, and the x86 box is where to
look for it: the report already tells us which side of the line we are on.

## What Mynah has that qwen does not, so nobody ports it backwards

- **26 runtime-dispatch sites in `src/qmat.c`** against qwen's zero.
- **A dispatch map that an operator can read from a running process**, with
  `compiled / supported / resolved` and the reason a row is off.
- **An ISA guard with a message**, rather than a SIGILL.
- **`tests/x86_cross.sh`**: x86 built and EXECUTED on the development machine.

Neither project has any SVE kernel (`grep -c svfloat32_t` is 0 on both), so the
idle SVE units on Neoverse-V2 are an open question in both trees.

---

## Second sweep, 2026-09-22: the TOOLS, which is where the rest of it was

The first sweep looked at kernels. The better material was next door in
`tools/`, and two items there target exactly the defects the Genoa box found.

### Taken: `dispatch_gate.py` (landed the same day)

qwen's version exists because of an AVX-512-BF16 prefill bug that cost it about
400 ms of TTFA on a c8a.4xlarge, and its expectation file says so in a comment.
The shape it gates on -- *expected ON, compiled AND supported on this host,
resolved OFF* -- is the shape Mynah produced twice this week and could not see.
Ours is `tools/dispatch_gate.py` + `tools/dispatch_expect.json`, wired into
`make test`, with a six-case negative control that caught a bug in the gate on
its first run.

Ours differs in one way worth keeping: **idle hardware is reported, not
failed.** A unit we have no kernel for is a roadmap item; qwen's table has no
equivalent because its `NOT-IMPLEMENTED` rows are fewer.

### NOT taken yet, and it is the best remaining idea: `flag_parity.py`

It derives, from the preprocessor guards around every env-var READ SITE plus one
call hop, which backend families can actually reach each flag -- then emits
`qwen_flag_scope.h`, a table of flag -> scope bitmask, and flags the
disagreements with the documentation. Its origin is a flag that *parsed and did
nothing* on ARM: `QWEN_NO_SIMD_QUANT` existed, looked like a feature, and the
NEON path never consulted it.

Mynah has that problem now and did not last week. This week added
`MYNAH_KERNELS_X86`, `MYNAH_QMAT_AVX512` and `MYNAH_QMAT_BF16DOT`, all three
x86-only, next to `MYNAH_QMAT_I8MM` and `MYNAH_QMAT_BF16`, both Arm-only, and
nothing anywhere states which is which. Setting `MYNAH_QMAT_BF16DOT=off` on
Graviton is silence -- not an error, not a warning, not a row.

The dispatch map covers the flags that have a ROW (it prints `env` and
`env_value` per row, which is better than a static table because it is the
running process). It says nothing about the ones that do not, and there are
many: a first count finds 30+ `MYNAH_*` names in `src/` and `server/`.

**Recommended shape, adapted rather than copied:** do not generate a header.
Generate a *test* -- `tests/test_flag_scope.py` -- that walks the read sites,
derives the scope, and fails when a flag documented for one architecture is
only reachable on another. A header that nothing reads would be a second thing
to keep in sync; a test that fails is the thing this repo already trusts.

### Looked at and deliberately not taken

| tool | why not |
|---|---|
| `backend_matrix.py` | Mynah's `configs/perf/*.json` + `serving_profile.py --profile` already refuses a run whose world disagrees, which is the same guarantee at the level that matters here |
| `census_report.py`, `costmap_*` | Mynah has `census`, `costmap.c` and `MYNAH_COST_MAP` already |
| `check_plan.py`, `check_repo_integrity.py` | governance for a repo with several agents; this one has `PLAN.md` as a board and a human |
| `cpu_qualify.py`, `doctor_wave.py` | `make doctor` covers the static half and refuses to predict a concurrency it has not soaked, which is the stricter behaviour |
| `box_info.sh`, `box_sync.sh` | scp and a git clone, which is what we do |

### The kernel conclusion is unchanged after the second sweep

VDPBF16PS and the AVX-512BW int8 dot were the two worth taking and both are in.
AMX stays out on a measurement (E12: the weight pass is 5% of the step at B=8),
not on effort. KleidiAI stays out as a dependency decision. Neither project has
an SVE kernel.
