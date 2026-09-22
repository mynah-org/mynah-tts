# The x86 kernel tiers Mynah does not reach, and how to add one honestly

Opened 2026-09-22. Priority: **max** — this is the code that has to exist
before an x86 box is worth renting, because a rented hour spent on a tier we
have no kernel for measures the fallback, not the machine.

## Problem

After the f32 runtime dispatch landed (E4-9), x86 reaches exactly three tiers:

| tier | f32 | int8 | bf16 |
|---|---|---|---|
| pre-AVX2 | scalar | scalar | scalar |
| AVX2 / FMA | AVX2 | `dot_q8_i32_avx2` (cvtepi8 + madd) | AVX2 shift-widen |
| AVX-VNNI / AVX-512 VNNI | AVX2 | VPDPBUSD, VEX or EVEX | AVX2 shift-widen |

Two gaps, and both matter for exactly the hosts an EC2 matrix would choose
between:

1. **AVX-512 without VNNI** — Skylake-SP, Cascade Lake, Zen 3. They have 512-bit
   registers and no VPDPBUSD, so today they run the 256-bit AVX2 int8 dot and
   half the register file idles. This is the brief's X2 question, unanswered in
   code.
2. **AVX512-BF16 (VDPBF16PS)** — Cooper Lake, Sapphire Rapids, Zen 4+. **bf16
   is the dtype the PocketTTS backbone ships**, and on Arm the bf16 path is
   worth +33% at B=8 through BFMMLA. On x86 we widen bf16 to f32 with a shift
   and multiply in f32 — correct, and not the instruction that exists for it.
   `src/qmat.c` already says so: *"The x86 VDPBF16PS kernel is unwritten."*

`../qwen-tts` has both (`.work/qwen-tts-kernel-reuse.md`), which is what makes
this a port rather than a research task.

## The objection this file has to answer first

`src/qmat.c` carries a deliberate refusal, written by whoever stopped here:

> Writing VDPBF16PS blind would put an unexecuted vector kernel on the default
> path, which is the thing the dispatch report exists to prevent.

That is right, and it is not an argument for leaving the kernel unwritten. It
is an argument against an unexecuted kernel **resolving**. Those are different
things, and separating them is the whole design below.

## Plan: the prove-on-first-use gate

A new kernel tier resolves only if, in this process, it has been **executed and
checked against the scalar reference**. Not "the CPUID bit is set" — run it.

    probe = CPUID says the unit is there
          AND a small fixed-shape matvec through the new kernel agrees with
              matvec_*_scalar to the tolerance that kernel's encoding allows
          AND the environment did not turn it off

memoised, so it costs one small matvec per process, once, off the hot path. A
host without the unit never reaches the check. A host with the unit that
disagrees with the reference **does not resolve and says so in the dispatch
map**, which is strictly better than what a blind kernel would do and better
than what a CI-only gate can do, because it is checked on the machine that will
run it rather than on the machine that built it.

This is the same doctrine `--self-test` already applies at startup, moved to
the granularity of one kernel, and it is what makes it defensible to land two
kernels this project cannot execute on any machine it owns today.

## Items

- **E14-1** the ISA guard must be compiled for the baseline. `qwen-tts` carries
  `__attribute__((target("arch=x86-64"), noinline))` on its guard
  (`qwen_tts_kernels.c:675`) and the reason is sound: a guard compiled with
  `-march=native` may itself contain an instruction the host lacks, and then it
  dies before printing the message it exists to print. Ours has no such
  attribute. One line, no behaviour change on a host that passes.
- **E14-2** AVX-512BW signed int8 dot, no VNNI. Same algebra as
  `dot_q8_i32_avx2` at 512 bits: `_mm512_cvtepi8_epi16` + `_mm512_madd_epi16`.
  Exact int32, so the acceptance gate is **bit-identical to the scalar
  reference**, not a tolerance.
- **E14-3** VDPBF16PS bf16 matvec. The interleave problem `src/qmat.c`
  documents is avoided rather than solved: `_mm512_cvtneps_pbh` converts 16 f32
  to 16 bf16 **in order**, so two of them concatenated with
  `_mm512_inserti64x4` give 32 consecutive bf16 — sixteen lanes of two
  k-adjacent values, which is exactly what the instruction pairs. No scratch
  buffer, no allocation in a kernel, and the `__m256bh`/`__m256i` juggling goes
  through a union, which is how qwen does it and compiles on both toolchains.
- **E14-4** `src/sgemm.c` — tracked separately in
  [`sgemm-runtime-dispatch.md`](sgemm-runtime-dispatch.md), because its ISA
  leaks into the data layout and it is not the same kind of change.

## Acceptance gate

- [x] every new kernel compiles for x86_64 from this Mac and the disassembly
      carries the instruction it was written for: **`vpmaddwd` on `zmm` x6** in
      `dot_q8_i32_avx512bw`, **`vdpbf16ps` x4** and **`vcvtneps2bf16` x8** in
      the bf16 path. The ISA guard, in an avx2 build: **0 AVX registers inside
      it, 131 elsewhere in the same object**
- [x] every new kernel is gated on having EXECUTED against the scalar reference
      — `qmat_int8_avx512bw_verify` (bit-identical; the product is exact) and
      `qmat_bf16_dpbf16_verify` (within `C*FLT_EPSILON`; same products, different
      summation order). A host that lacks the unit never reaches the check and
      the row reads `resolved=OFF` with the reason
- [x] `make x86-cross` passes: a no-AVX-512 host is unchanged and reads the new
      rows correctly
- [x] the dispatch map names the new tiers. `isa.x86.avx512f` and `.avx512bw`
      stopped being "COMPILER FLAG ONLY" — that text was true until these
      kernels existed and would have been a lie the moment they landed;
      `isa.x86.avx512bf16` is new; `isa.arm.bf16` distinguishes the three x86
      tiers instead of answering yes for two of them
- [ ] CI green, including the x86 job's VNNI step — needs a push
- [ ] **not closed by any of the above**: a measurement. These kernels are
      written to be measured on a rented box, and until one runs them no
      performance claim may be made from this item
