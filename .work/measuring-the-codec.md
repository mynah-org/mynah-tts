# Measuring the codec conv stack on macOS misattributes where its time is

Found 2026-09-16, after reporting a conv-stack figure that does not hold in the
configuration production ships.

## The finding

The same request, same machine, same text, two threads — only the linked BLAS
differs. `MYNAH_SEANET_PROFILE=1`, phase table:

| phase | macOS / Accelerate | **`BLAS=none` (Linux default)** |
|---|---|---|
| `decode.total` | 133.0 ms | **309.2 ms** |
| `convtr.gemm` | 36.1 (27.1%) | **140.3 (45.4%)** |
| `conv.gemm` + `conv.taps` | 55.0 (41.4%) | **130.0 (42.0%)** |
| `elu` | 15.5 (**11.6%**) | 17.3 (**5.6%**) |
| `convtr.scatter` | 11.4 (8.6%) | 10.4 (3.4%) |

The region is **2.3× more expensive** with our own sgemm than with Accelerate on
the same silicon, and the composition moves with it. Accelerate on Apple parts
reaches the AMX coprocessor, so its GEMM is not a fair comparison for a NEON
kernel — but that is exactly why the *shares* cannot be carried across.

## What it cost

E10-5a (the vectorised ELU) was reported at **1.28×** on `codec.conv_stack`.
That is an Accelerate number. Re-measured as a paired A/B in the production
configuration — the same tree with only the ELU's vector block disabled —
it is:

    BLAS=none, ELU scalar:  335.3  331.7  334.3 ms
    BLAS=none, ELU vector:  310.0  309.2  309.5 ms   -> 1.08x

The ELU phase itself still improves about 3×. What changed is its share: 11.6%
of the region under Accelerate, **5.6%** under the BLAS that ships.

An Apple Silicon number is a development signal and not a
product claim, and the commit did label it as one. That was not enough. A scale
factor between platforms is expected and survives being labelled; a change in
**which phase dominates** does not, because it decides what to work on next.

## The rule, for the next person and for me

- Any conv-stack or codec measurement is taken with **`BLAS=none`**, because
  that is what Linux ships (`Makefile:105`). Accelerate is for checking that the
  Accelerate build still works, not for deciding where time goes.
- A ratio quoted for a region must name the BLAS it was taken under.
- This does not apply to the backbone or the flow head, which do not go through
  `sea_sgemm` — their kernels are ours in every configuration.

## What it means for E10-5

It makes the item MORE valuable, not less. In the configuration that ships, the
two GEMMs are **81.5%** of the conv stack (140.3 + 130.0 of 309.2), against
68.5% under Accelerate. Our f32 sgemm runs those at roughly 41 GFLOP/s per core
here. An int8 path attacks exactly that share — and the share is bigger than the
Accelerate numbers suggested.

Not yet known, and only the Linux box can say: how our sgemm compares to
OpenBLAS on Neoverse-V2, where there is no AMX and the comparison is between two
NEON kernels rather than against a coprocessor.
