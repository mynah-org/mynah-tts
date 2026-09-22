# sgemm.c: the ISA is in the data layout, not only in the inner loop

Opened 2026-09-22. Priority: **max**, and the last piece of E4-9's P1.

## Problem

`src/kernels.c` now selects its f32 kernels at runtime on x86. `src/sgemm.c`
does not, and it is not a smaller version of the same job.

`mynah_sgemm_f32` is called six times from `src/seanet.c`, and the codec is
**43% of the wall** on the Pocket path. So on an x86 `portable` build the
largest single consumer of f32 time still runs scalar, and on an `avx2` build
it still SIGILLs on a pre-Haswell host. The binary is only as portable as its
least portable kernel.

## Evidence: why target attributes are not enough

The file is written once against a macro layer (`sgemm.c:76-118`):

    #define sg_load(p)  vld1q_f32(p) | _mm256_loadu_ps(p) | (*(p))
    #define SG_LANES    4            | 8                  | 1
    #define SG_ACC_VECS 16           | 8                  | 16

That is a good design — one micro-kernel body, three instruction sets, and the
scalar build executes the same loop nest in the same accumulation order, which
is coding rule 5 expressed in code instead of in a comment.

But `SG_LANES` does not stay in the inner loop:

    #define SG_NV_MAX     (SG_ACC_VECS / SG_MR)
    #define SG_NARROW_MAX (SG_NV_MAX * SG_LANES)

and `SG_NARROW_MAX` reaches the **packed panel geometry** and the **public**
`mynah_sgemm_narrow_max` in `sgemm.h`. Put a target attribute on the
micro-kernel alone and the packing still has the wrong shape for it.

So multi-versioning sgemm means the packed layout becomes a runtime choice.
That is a different change from splitting a kernel in two, and doing it in the
same commit as one would make a numeric regression unattributable.

## Plan

Two shapes are viable and the choice is a measurement, not a preference:

1. **Two translation units.** Compile `sgemm.c` twice with different `-m`
   flags into `sgemm_avx2.o` and `sgemm_base.o`, rename the entry points
   through a macro, and dispatch. The Makefile already knows how to build one
   source at several ISA settings (the `link-only` matrix does it), and the
   packing follows its own TU's constants with no runtime branch inside the
   loop. Cost: the object is built twice and the two must be kept from being
   mixed — the same hazard `SAN_DIR` already exists to prevent for sanitizers.
2. **One TU, body in an `.inc`, included twice** under different macro sets
   with target attributes on the wrappers. No build changes; the risk is that
   every static in the body needs mangling and a missed one links to the wrong
   version silently.

(1) is more honest about what is happening and is how the Makefile already
thinks. Start there, and keep the scalar TU as the reference.

## Done 2026-09-22 — shape (1), two translation units

`src/sgemm.c` renames its twelve public symbols when `MYNAH_SGEMM_VARIANT` is
defined, and on x86 the Makefile builds it twice — `base` with the build's own
flags, `avx2` with `-mavx2 -mfma` **added** rather than substituted, so a build
that already carries them is not silently narrowed. `src/sgemm_rt.c` owns the
public names and forwards. On aarch64 the file is empty (a typedef, because
`-Wpedantic` rejects an empty translation unit) and nothing is built twice.

**The variant decision is asked of the COMPILER, not of `uname -m`.**
`make x86-cross` builds x86_64 objects on an arm64 host; a decision keyed on the
host would have skipped the very variant that cross build exists to produce,
then linked, run, and looked fine having tested nothing.

**Which variant is not a new probe.** `mynah_kernels_x86_avx2()` already answers
"does the f32 half of this binary run AVX2 here", is memoised, and
`MYNAH_KERNELS_X86=scalar` already forces it down. A second probe with a second
env would let the two halves of the f32 path disagree, and a bisect would then
have to say which one it meant.

## Acceptance gate

- [x] `mynah_sgemm_f32` resolves at runtime on x86 from ONE binary — and the
      two objects are demonstrably different builds of one source: **0 `ymm`
      registers in `sgemm_base.o`, 621 in `sgemm_avx2.o`**, same `src/sgemm.c`
- [x] `sgemm.selftest` passes in the resolution in force; `make x86-cross`
      asserts the baseline resolution and its self-test on a no-AVX2 host, and
      a new CI step runs BOTH resolutions on the x86 runner — the only machine
      in the fleet where they differ, since Rosetta has no AVX2 and resolves
      both settings to scalar
- [x] `mynah_sgemm_narrow_max` answers for the resolution in force: it is one
      of the twelve forwarded symbols, so it cannot disagree with the kernel
- [x] the packed panel produced by one resolution is never consumed by the
      other — by construction, and the construction is stated rather than
      assumed: packing is per-call scratch, and the resolution is memoised for
      the life of the process, so there is no moment at which it could change
      between a pack and its consumer
- [x] `make x86-cross`, `make test`, `make ubsan`, `make server-test` green
- [ ] CI green — needs a push
- [ ] NOT a gate, and stated so it is not claimed: no speedup. The win is that
      one artifact is both safe and fast; the codec numbers come from a box
