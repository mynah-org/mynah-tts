# E4-20b / E4-21 — the int8 float epilogue, SMMLA, VNNI on silicon, int4 on x86

Status: **DONE 2026-09-13**. Pinned to `10f8f48`.

## The problem

`mynah_qmat_linear_batched` promises that a row's answer does not depend on the
batch it arrived in. It routed a row through `matvec_q8` (SDOT) or
`matvec_q8_pair_i8mm` (SMMLA) according to the batch size and the row's position
in it, and the two kernels did not round alike — so under concurrent serving a
request's audio depended on who else was in flight. SMMLA was therefore switched
off behind `MYNAH_QMAT_I8MM=1` and had been idle since.

Separately: the x86 VNNI kernels (`VPDPBUSD`, both encodings) had **never
executed**, and int4 had **no vector path on x86 at all**.

## What the divergence actually was — and the working hypothesis was wrong

The brief named the *grouping* of the three-factor `(float)s * scales[row] * sx`.
The disassembly supports that reading at first glance: on GCC 15.2/aarch64,
`matvec_q8` emits `fmul sx,scale` first under `SIMD=auto` (`-march=native`) and
`fmul s,sx` first under `SIMD=portable`. **But those are two different binaries**,
and a row's answer cannot depend on a build it was not built by. Within one
binary GCC was consistent.

The cause was the **rounding count**, not the grouping. `dot_q8()` returned the
product and let each caller write `value += bias[row]` itself — two roundings —
while the quad epilogues wrote `... * sx + bias[row]`, which `-ffp-contract=fast`
fuses into one. Evidence, measured on the Neoverse-V2 box:

- with the grouping already forced uniform by a value barrier, `SIMD=portable`
  still failed `self_test_i8mm_identity` by exactly 1 ULP at 96x256;
- rebuilding the identical source with `-ffp-contract=off` turned every profile
  green;
- `tests/qmat_negative_control.sh` confirms it from the other side: removing the
  value barrier **alone** is caught by nothing, because `fmaf()`'s second operand
  must be materialised as a value, so the fma pins `ws * sx` as a unit by itself.

So the barrier is load-bearing only on the branch with no hardware FMA — x86
built without `-mfma`, i.e. `SIMD=portable` on the production target. Both halves
are needed; neither alone is the fix.

## What was done

Every int8 kernel now ends through `qmat_row_scale()` + `qmat_row_epilogue()`:
a value barrier leaves a two-factor product with one grouping, and the bias is
*part of* the epilogue so there is one rounding decision instead of one per
caller. `fmaf()` where `__FP_FAST_FMAF` holds (which is what `SIMD=auto` already
emitted, so production numerics do not move), a frozen multiply-then-add where it
does not. SMMLA is wired into the batched linear **by default**;
`MYNAH_QMAT_I8MM=0` disables it.

int4 had the same hazard in `acc += (float)gi * scales[g]` across four sites and
in `acc * sx + bias`; both are pinned the same way (`qmat_q4_accum`,
`qmat_q4_finish`). It was invisible because int4's existing gates compare against
an f32 dot with a **relative tolerance** — which a lossy format needs, and which
swallows a 1 ULP accumulation difference and would also swallow a
permuted-but-plausible nibble order.

`q4_group_i32_avx2()` gives int4 its first x86 vector path. See the cost model
below for why it is AVX2 and not VNNI.

## Gates

- `self_test_i8mm_identity` now asserts the **float** side with real per-row
  scales, two activation scales and a bias — not just the int32 tile. The old
  version asserted only integers and ran at 13x37, a shape too small to reach the
  epilogue, which is what licensed wiring the broken kernel in.
- `u8_identity_one` asserts **bit-equality** where it allowed 2 ULP.
- `self_test_batch_membership` runs at INT8 with the SMMLA wiring forced **both**
  ways, plus INT4 and F16.
- `self_test_i8mm_ab`: every shape with the wiring off vs on, bit-identical.
- `self_test_q4_identity`: per-group int32 vector-vs-scalar, the int16
  saturation bound the x86 kernel depends on, and the assembled `dot_q4` and
  `matvec_q4`.
- `tests/test_qmat.c` + `make qmat-test`: the grouping pin (which grouping and
  how many roundings the compiler *actually emitted*, against references with
  their own barriers), and a report of which kernel resolved. Run five times
  under `MYNAH_QMAT_I8MM=0/1` and `MYNAH_QMAT_VNNI=256/512`.
- `tests/qmat_negative_control.sh` + `make qmat-negative-control`: six breaks,
  each required to be caught, each naming the profile that caught it.

## Measured

| | |
|---|---|
| Magpie int8 golden | byte-identical under `SIMD=auto/portable/scalar` |
| PocketTTS | byte-identical to `10f8f48`, f32 and int8, batch 1 and 3 |
| `SIMD=scalar` goldens | `offline-f16` fails — **pre-existing at `10f8f48`**, f16 is compiled out and downgrades to f32 |

## CI: VNNI on real silicon, first execution ever

Run 34755089602, job `linux-x86_64 (openblas)`. Runner: **Intel Xeon Platinum
8573C**, flags `avx2 avx512_vnni avx512f avx_vnni`.

- default resolution: `quant.int8_kernel = avx512vnni` (EVEX `VPDPBUSD`)
- `MYNAH_QMAT_VNNI=256` → `avxvnni` **RESOLVED**
- `MYNAH_QMAT_VNNI=512` → `avx512vnni` **RESOLVED**
- `--self-test` PASS at `off` / `scalar` / `256` / `512`. **No kernel bug.**
- the u8 identity's newly-tightened bit-equality passed on the machine class
  where the original 2 ULP gap was seen (lanes r == 3 of each quad)

**The `ubuntu-latest` x86 pool is heterogeneous.** In the same run, the
`link-only: x86 SIMD=avx512` job landed on a runner reporting only `avx2 fma`
and the binary's own guard correctly refused to start. One green VNNI run is
therefore not proof for the pool, and that job is a standing coin-flip
independent of this change (main has been green on luckier draws).

## Emulation: what it can and cannot do

`qemu-user` 10.2 + `gcc-x86-64-linux-gnu` on the ARM box runs a cross-built
x86-64 binary. This is how the AVX2 int8 and int4 kernels and the **non-FMA
epilogue branch** were executed and broken locally.

**QEMU TCG implements no VNNI.** It says so: `TCG doesn't support requested
feature: CPUID[eax=07h,ecx=01h].EAX.avx-vnni [bit 4]`, and the same for
`avx512vnni`; even `-cpu max` reports `AVX512F=0`. So `VPDPBUSD` is reachable
**only** through CI's runner. Recorded so nobody re-tries it.

Also: QEMU's F16C flushes subnormals, so `--self-test` fails there on an f16
pack check that real silicon passes (CI is green). Use `-cpu <model>,-f16c`.

## Cost model — int4 on x86, written before any measurement

Per MAC, instructions:

| | instr / MAC | bytes / weight |
|---|---|---|
| int8 VNNI (`QMAT_U8_BLOCK`=64, one float touch per row) | ~0.03 | 1.0 |
| int4 scalar (today, x86) | ~4.5 | 0.5 |
| int4 AVX2 (this change) | ~0.56 | 0.5 |
| int4 VNNI (not written) | ~0.50 | 0.5 |

**The structural point.** int8 carries one scale per row, so it accumulates int32
across the whole row and touches a float once. int4 carries one scale per group
of 32, so the int32 accumulator **must** be flushed to float every 32 elements.
That flush — horizontal reduce, convert, multiply, add — is ~6 of the ~18
instructions per group, and the nibble unpack is another ~6. The dot itself is 3.

So **`VPDPBUSD` would replace 3 instructions with 2 and move the whole int4
kernel by roughly a tenth.** The 32-element group really is too short for VNNI to
be the story. The prize is simply *having* a vector path: ~8x fewer instructions
than the scalar nibble loop. **The int4 VNNI variant is deliberately not written**
— it is a small, well-understood delta and should be measured on the machine that
would benefit, not guessed at from an ARM box.

Consequence for the recorded number: "int4 buys only 3% over int8 on the codec"
was taken on ARM, where int4 has SDOT. On x86 before this change, int4 was
~5x more instructions per MAC than int8 while carrying half the bytes — it would
have lost, and the number would have been about the missing kernel, not the
format. `mynah_qmat_int4_kernel()` now names the resolution so that is readable
from the dispatch report instead of from the source.

**The correction term is not a row sum.** Q4_0 nibbles are `n - 8` for
`n` in [0,15], and `_mm256_maddubs_epi16` wants its first operand unsigned — so
the nibble goes in unsigned and the activation stays signed:
`sum (n-8)*x == sum n*x - 8*sum x`. Both terms come from `maddubs` against the
same activation, subtracted in int16 before widening, so **there is no
precomputed table with an extent to get wrong** — which is the mistake the int8
path's `+128` row-sum self-test exists to catch. No saturation: the int16
difference lies in [-5888, 5858], and `self_test_q4_identity` asserts the bound
with groups built to reach it.

## Rejected / recorded negatives

- **Hoisting the epilogue into a `static inline` helper** (E4-20's attempt):
  does nothing. The optimizer inlines it and reassociates the resulting DAG per
  site; statement boundaries are not barriers.
- **`-ffp-contract=off` for the build**: fixes it, and changes global f32
  numerics and codegen everywhere for a defect that is local to one file.
- **`noinline` on an epilogue helper**: one call per row in the decode hot loop.
- **Removing the value barrier now that `fmaf()` pins the grouping**: it is still
  load-bearing on the no-hardware-FMA branch, which is `SIMD=portable` on x86.
- **Reducing 8 int4 groups at a time to amortize the horizontal reduce**: would
  change the float accumulation order and break bit-identity with the scalar
  reference. Not worth it for ~20%.

## Open

- `tools/simd-auto.sh` writes `$(BUILD_DIR)/simd-auto.mk` from a `$(shell ...)`
  that runs *after* the `-include` that would read it, so the **first** `make` in
  a tree with no `build/` resolves `SIMD=auto` to no ISA flag at all. On aarch64
  that is the difference between compiling the NEON int4 kernel and not.
  `tests/qmat_negative_control.sh` primes it with `make info`; the Makefile
  itself is not this lane's file.
- int4 VNNI: specified above, not written.
- SMMLA's *value* is a separate question from its correctness — see below.

## Is SMMLA worth having? (cost model, no measurement taken)

SMMLA does 32 MACs per instruction against SDOT's 16, but only because it
consumes **two** activation vectors. So it can only pay in the weight-stationary
batched linear, and only at batch >= 2; claiming it for the single-row decode
matvec would be exactly the unearned ISA attribution the dispatch report exists
to catch, and `qmat_batch_rows()` is the only place it is wired.

What it should save: the int8 dot instruction count in the batched linear,
halved, for pairs of activations. What it cannot save: the weight traffic, which
is what the weight-stationary loop exists to amortize and which is already shared
across the batch. **So the ceiling is the fraction of batched-decode time spent
issuing SDOTs rather than waiting on weight bytes** — and E4's own measurement
that f16 is 1.96x f32 end to end, almost exactly the 2.00x the halved weight
bytes predict, says decode is bound by weight traffic. That is an argument for
expecting **little to nothing**, and for measuring before believing otherwise.

The smallest experiment that would kill the idea: at a fixed batch of 4, run the
batched linear with `MYNAH_QMAT_I8MM=0` and `=1` in paired interleaved arms on a
**quiet** box, pinned with `taskset` and an explicit `MYNAH_THREADS` (the pool
sizes itself from `sysconf`, not the affinity mask, so the default oversubscribes
inside a pinned worker), and report the median of the **within-round ratio**. If
that ratio is within noise of 1.00, SMMLA is a correctness curiosity and the
default should go back to off — the kernel stays compiled and self-tested either
way. **This was not measured**: the box was wanted for a capacity screen, and a
number taken next to a compile is a number about the compile.
