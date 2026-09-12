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
