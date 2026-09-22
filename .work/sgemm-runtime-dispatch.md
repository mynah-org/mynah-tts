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

## Acceptance gate

- [ ] `mynah_sgemm_f32` resolves at runtime on x86: AVX2 where the CPU has it,
      baseline where it does not, from ONE binary
- [ ] `sgemm.selftest` passes in both resolutions on the same host, forced by
      an env the way `MYNAH_KERNELS_X86=scalar` forces the f32 half
- [ ] `mynah_sgemm_narrow_max` answers for the resolution in force, not for the
      build — it is public API and a caller sizing a buffer from it must get
      the number the kernel will actually use
- [ ] the packed panel produced by one resolution is never consumed by the
      other; a test asserts it rather than a comment forbidding it
- [ ] `make x86-cross` unchanged, `make test`, `make ubsan`, CI green
- [ ] NOT a gate, and stated so it is not claimed: no speedup. The win is that
      one artifact is both safe and fast; the codec numbers come from a box
