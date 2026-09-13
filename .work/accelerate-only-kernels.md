# E4-16d / E3-6 — the Accelerate-only vector kernels, written ourselves

Item: `PLAN.md` E4-16d (write the Accelerate-only kernels ourselves, both ISAs)
and E3-6 (model-free self-tests for the new kernels).
Inventory: [`no-blas.md`](no-blas.md) §3b tier 3.
Method: [`engineering-method.md`](engineering-method.md) §5 — the cost model
comes before the code.

---

## 1. The problem, stated as a fact and not as a preference

`src/kernels.c` and `src/codec_nanocodec.c` contain two activations whose
**macOS and Linux builds execute different arithmetic**:

| activation | macOS | Linux | call sites |
|---|---|---|---|
| GELU-tanh, array form | Accelerate `vvtanhf` | libm `tanhf`, one element at a time | `kernels.c` ×4 |
| SEANet Snake | `vDSP_vsmul` + `vvsinf` + `vDSP_vsq` + `vDSP_vsma` | libm `sinf`, one element at a time | `codec_nanocodec.c` ×12 |

Neither is BLAS. Neither has an OpenBLAS equivalent. So the flip to
`BLAS=none` does not touch them, and they are not a performance question at
all: they are the reason a number taken on the development platform is not
evidence about the product. The two platforms agree today **by luck** — see §4,
where I measured exactly how much luck.

A third divergence is inside macOS itself and nobody had noticed it: the
elementwise `mynah_gelu_tanh()` uses libm `tanhf` while the array form
`mynah_gelu_tanh_array()` uses `vvtanhf`, and `engine_magpie.c` calls **both**
(`:177` elementwise, `:1064`/`:1997` array). One binary, one model, two GELUs.

## 2. Cost model — before writing anything

**Current cost.** Not a wall-clock cost; a *correctness-jurisdiction* cost.
Every measurement taken on macOS runs a tanh and a sine that do not exist on
the production target, and every measurement taken on Linux runs a scalar loop
that does not exist on the development machine. The reference implementation
measured ~88 M sine evaluations per request in its equivalent of the Snake, so
this is a hot path, not bookkeeping.

**Suspected cause.** Two vendor libraries were the cheapest way to get a
vectorised transcendental on the machine the code was first written on.

**Maximum plausible saving.** Zero RTF is claimed. The saving is that one
kernel exists instead of four, that the Linux path stops being a scalar
fallback nobody benchmarks, and that the per-call `malloc` inside the Snake's
Accelerate branch (`codec_nanocodec.c:96`) disappears — it violates coding
rule 4 and it is on the codec path.

**New work introduced.** An f32 `tanh` and an f32 `sin`, each with a scalar
reference, a NEON path and an AVX2 path, plus argument reduction for the sine
that is correct away from the origin. This is the risk: a transcendental
written badly is worse than a scalar loop, because it is *plausibly* wrong.

**The smallest experiment that would kill the idea.** Measure our kernel, libm
and Accelerate against a double-precision reference in ULP, over the whole
input range. If ours is materially worse than `vvtanhf`/`vvsinf`, the right
answer is to keep Accelerate on macOS and say so. It is not — see §4.

**Verdict: build it.**

## 3. Design

One algorithm per function, three transcriptions of it (C, NEON, AVX2), and
the C one is the definition of correctness.

- **tanh** — Cephes split. `|x| < 0.625`: odd minimax polynomial in `x²`.
  `0.625 <= |x| <= 9.010913`: `1 - 2/(exp(2|x|)+1)`, with an `exp` written for
  the *only* range this reduction can produce, `[1.25, 18.03]` — no overflow
  case, no denormal case, no general-purpose `expf` to get wrong. `|x| > 9.010913`:
  `±1`, which is the correctly rounded f32 answer because `tanh(9) = 1 - 3.07e-8`
  and the largest float below 1 is `1 - 5.96e-8`. NaN passes through; `-0.0`
  survives because the sign is applied by bit, not by arithmetic.
- **sin** — Cephes quadrant reduction: `j = (int)(4x/pi)`, three-part
  Cody-Waite subtraction of pi/4 (`DP1` exactly representable in 8 bits, so
  `y*DP1` is exact while `y` fits in 14 bits), then a sine or cosine polynomial
  chosen by quadrant. **Valid to `|x| < 8192`**, where the exactness of `y*DP1`
  runs out. Beyond that, and for every non-finite input, the lane is redone
  with libm `sinf` — so scalar and vector agree there *by construction* rather
  than by approximation.
- **Snake** is fused: `row[t] += sin²(a·row[t]) / (a + 1e-9)` in one pass, in
  registers, with no scratch array and no allocation.

## 4. Measured — is this fixing macOS or degrading it?

**Fixing it.** Maximum error against a double-precision evaluation of the same
function, `make kernels-test` on the M-series development machine, ~1M samples
per row:

| tanh, over [-4,4] and [-40,40] | worst | absolute |
|---|---|---|
| **ours, NEON** | **1.35 ulp** | 8.04e-08 |
| ours, scalar reference | 1.35 ulp | 8.04e-08 |
| libm `tanhf` | 0.76 ulp | 4.54e-08 |
| **Accelerate `vvtanhf`** | **2.60 ulp** | 8.91e-08 |

| sin, worst absolute | [-2pi,2pi] | [-400,400] | [-65000,65000] |
|---|---|---|---|
| **ours** | **6.11e-08** | **6.81e-08** | **6.76e-08** |
| libm `sinf` | 3.24e-08 | 5.04e-08 | 5.04e-08 |
| **Accelerate `vvsinf`** | **1.31e-07** | **1.31e-07** | **1.28e-07** |

Ours is about **twice as accurate as the vendor routine it replaces**, on both
functions, and about 1.5× less accurate than libm. So the macOS build was the
one carrying the larger error, and the Linux build — a scalar libm loop — was
the more accurate of the two all along. The framing is not "we accepted a
worse tanh to get one code path". It is "the development platform had the
worse tanh, and now it does not".

### The compounding, on a real PocketTTS utterance

6.16 s of audio, `models/pocket-en`, seed 1234, identical text and parameters.
Utterance length, step count and EOS position are **identical in every run
below** — what moves is the waveform.

| comparison | first differing sample | Pearson |
|---|---|---|
| **pinned head macOS vs pinned head LINUX** | **321** (13 ms) | **0.9535** |
| pinned head macOS vs ours | 321 | 0.8651 |
| pinned head macOS vs a pure-libm reference build | 321 | 0.9586 |
| ours vs that same pure-libm reference build | 1660 (69 ms) | 0.8925 |

Read the first row first, because it is the whole argument. **At the pinned
head, the same binary source on macOS and on the Linux production box already
produced different audio**, diverging 13 ms into the utterance and correlating
0.9535 by the end. `no-blas.md` said the two platforms "agree by luck"; they
did not agree at all, and nothing in the test suite could see it.

Against that, the change this note describes is the same size — and it is the
one that removes the cause. The per-eighth RMS of any of these differences is
below -80 dB for the first eighth and reaches the signal level by the middle:
a one-ulp activation change at step 1 is a different sample at step 200,
exactly as predicted. That is a property of a continuous-AR model, not of this
change, and it is why an "our unit test passes" qualification would have been
worthless here.

**Which trajectory is right?** Neither is the model's truth — the oracle is.
Both stay with the pure-libm reference for a while and then leave it, and
after a chaotic divergence the correlation is a coin flip, not a quality
measure. The one non-arbitrary statistic is **how long they stay together**,
and ours stays together five times longer (sample 1660 vs 321) — consistent
with the ULP ranking above. The claim made here is the modest one that the
evidence supports: ours is the more accurate kernel, and the platforms now run
one function instead of two.

## 4b. How the AVX2 path was verified without an x86 machine

Neither machine in this lane is x86. The AVX2 transcription was still executed,
not merely compiled, and it failed on its first run — so "compiles clean" would
have been a false gate.

```sh
# x86-64 AVX2, executed under Rosetta 2 on Apple Silicon:
clang -target x86_64-apple-macos12 -std=c11 -Wall -Wextra -Wpedantic \
      -O3 -mavx2 -mfma -ffast-math -fno-finite-math-only -Isrc \
      src/kernels.c harness.c -lm -o /tmp/avx2_selftest && /tmp/avx2_selftest
```

where `harness.c` is a `main()` calling `mynah_vecmath_self_test`,
`mynah_gelu_self_test` and `mynah_kernels_self_test` plus a one-line stub for
`mynah_dispatch_register_probe`. Rosetta 2 executes AVX2 and FMA. Dropping
`-mavx2 -mfma` gives a second configuration, `ISA = scalar` on x86, which is
the only way to exercise the scalar reference under a non-ARM ABI.

**What it caught on the first run:** the array GELU and the elementwise GELU
landed 6 ulp apart near x = -0.28 under the AVX2 body, because the FMA
spelling and the C spelling round differently and the GELU there is an order of
magnitude smaller than x — the same ULP-of-a-near-zero trap that this file
falls into three other times. The bound is now taken against `1 + |x|`, the
scale `|GELU(x)| <= |x|` actually lives on. Not a defect in the kernel; a
defect in the gate, and one that only an x86 run could surface.

Verified configurations for the transcendentals:

| ISA | machine | compiler | result |
|---|---|---|---|
| NEON | Apple Silicon | clang | PASS |
| NEON | Neoverse V2 | gcc 15.2 | PASS |
| AVX2 + FMA | x86-64 under Rosetta 2 | clang | PASS |
| scalar | x86-64, SSE only | clang | PASS |
| scalar | Neoverse V2, `SIMD=scalar` | gcc 15.2 | PASS |

Still unverified: AVX-512, and AVX2 on real x86 silicon rather than Rosetta's
translation of it.

## 5. What the existing gates could not see

`make goldens` is a byte-exact SHA-256 of an int16 WAV. Measured on this
branch: flipping `MYNAH_SNAKE_SCALAR` (vDSP Snake vs. scalar Snake) and
`MYNAH_GELU_SCALAR` (Padé GELU vs. libm GELU) leaves **all 8 golden hashes
unchanged**, while `MYNAH_TIMING=1` confirms the Snake runs 96 times in that
same run. The gate exercises the code and cannot see the arithmetic: int16
quantisation absorbs a 1e-7 relative change. That is not a criticism of the
goldens — they exist to catch a refactor that moves a tensor — but it does mean
**the goldens are not, and never were, the qualification for a transcendental
substitution.** `tests/test_kernels.c` is.

---

## 6. The IDLE HARDWARE footer — a cost model, and why nothing was written

`--dispatch-map` ends with an IDLE HARDWARE footer naming five units the
production CPU has and this runtime does not use: `bf16`, `sve`, `sve2`,
`svei8mm`, `svebf16`. The instruction was to pick the one with the best ratio
of plausible gain to risk, write the cost model **before** the kernel, and
write the kernel only if the model survives.

**It does not survive. Nothing was written. Here is the model.**

### 6a. One measurement kills four of the five rows

Neoverse V2 (MIDR `0x410fd4f1`), measured on the production box:

```
SVE vector bytes = 16 (128 bits), f32 lanes = 4;  NEON f32 lanes = 4
/proc/sys/abi/sve_default_vector_length = 16
```

**SVE on this CPU is 128 bits wide — exactly NEON's width.** V2 implements SVE
over the same four 128-bit pipes. So `sve`, `sve2`, `svei8mm` and `svebf16`
offer *no lane count to gain*: an SVE f32 kernel does four lanes, which is what
`src/kernels.c` already does; an SVE `SMMLA` does the same 8-bit block
`src/qmat.c` already reaches through NEON `matvec_q8_pair_i8mm`.

What SVE would still buy is predication — the scalar tail loop disappears — and
gather/scatter. On the shapes in this repo the tail is at most three elements
out of hundreds. **Maximum plausible saving: under 1%. Cost: a second kernel
per function, in a dialect neither CI machine can compile-check by default.
Killed by measurement, not by opinion.**

That the footer prints four rows whose upside is near zero is not a fault in
the footer. It says the CPU *has* the unit and we idle it, which is true; it
does not claim the unit is worth having. Confusing "supported" with "worth
using" is exactly what a cost model is for.

### 6b. `bf16` is the only row with a real mechanism, and it still loses

BFMMLA does 16 MACs per 128-bit instruction against FMLA's 4, and a bf16 weight
halves the bytes moved. On paper that is the one row worth chasing. Four facts
kill it:

1. **The bottleneck is bandwidth, not arithmetic.** `no-blas.md` records the
   codec transformer as 43% of the wall while running 16 positions as 16 steps
   over the same ~29 MB. A 2× arithmetic rate buys nothing on a stage that is
   waiting for memory.
2. **The bandwidth win is already taken, twice.** `MYNAH_QUANT=f16` already
   halves the weights on ARM, and `int8` already quarters them (measured 1.9×
   on EPYC, `docs/performance.md`). bf16 offers *the same* 2× as f16 and half
   of int8's.
3. **At that same 2×, bf16 is the less accurate of the two.** bf16 keeps 7
   mantissa bits, f16 keeps 10. The weights themselves would be lossless — the
   checkpoint IS bf16 and `src/weights.c` widens it at load — but BFDOT/BFMMLA
   need *both* operands in bf16, so the activations lose eight bits, on a
   continuous-AR model, where §4 of this note shows what a one-ulp activation
   change does by step 200.
4. **It is not in this lane's files.** A bf16 weight type is `src/weights.c`
   and `src/qmat.c`.

**The smallest experiment that would have killed it** was the one in 6a plus
reading two existing measurements. Cost: twenty minutes. Cost of writing the
kernel first: days, plus a numerical qualification as large as this one.

**The one condition that flips the answer: x86.** `PLAN.md` E3-5c already says
"f16 is a dead end on the target: it is ARM-only". On an AVX-512-BF16 host
there is no f16 alternative, and bf16 becomes the only lossless-weight 2×
available. So the correct owner of the `bf16` row is the x86 lane and the qmat
lane together, not this one, and the correct trigger is "when x86 becomes the
target", not "because the footer printed it".

**Verdict: write nothing. The footer's five rows are now four dead ends and one
deferred hand-off, and that is a more useful state than five unexplained OFFs.**

### 6c. What this lane owes other lanes

- `src/dispatch.c:754` — the `blas.accelerate` row still says Accelerate
  supplies "`cblas_sgemm` ... and `vvtanhf` for the array GELU". After this
  change it supplies `cblas_sgemm` and nothing else in `src/kernels.c`. The row
  that exists to make the dependency visible now overstates it.
- `src/dispatch.c` row table — there is no `kernel.transcendental` id, and a
  probe registered under an unknown id is silently dropped. One row, answering
  `mynah_vecmath_isa()` and `mynah_vecmath_denormals_flush()`, would put "which
  tanh and which sine did this binary run" in the report. Until then the fact
  rides on `codec.snake_vector` and `kernel.gelu_vector`.
- `src/conv1d.h` — `mynah_unfold_causal`'s comment documents the output layout
  and not the input's. The input is time major `[length][channels]`, the output
  channel major within each row; the function transposes. Found by writing the
  test against the header and watching it fail.
- Makefile / build lane — `-ffast-math` on gcc/aarch64 links a startup object
  that sets FPCR flush-to-zero; Apple clang does not. **Denormal inputs vanish
  on the production target and survive on the development machine**, before any
  of our code runs. Not a kernel property and not fixable from here.

---

## 7. The honest result: this closes ONE of the platform divergences, not all

The claim E4-16d is supposed to support is "macOS and Linux run the same code".
After this change they run the same GELU and the same Snake — and they still
do not produce the same audio. Measured, same text, same seed, same pack:

| build | first differing sample | Pearson |
|---|---|---|
| pinned head, default | 321 | 0.9535 |
| **this change, default** | **1660** | 0.8919 |
| this change, `BLAS=none` | 1660 | 0.8919 |
| this change, `BLAS=none` `-O1 -ffp-contract=off` | 321 | 0.8672 |
| ...and `MYNAH_THREADS=1` | 321 | 0.8672 |

Read down that column. The divergence moves **five times later** when the GELU
is unified, so the GELU was a real contributor. It does not go away, and three
candidate explanations are eliminated outright by the rows below it:

- **not BLAS.** `BLAS=none` on both sides is bit-identical to the default
  comparison. Accelerate's `cblas_sgemm` against OpenBLAS's is not what is
  moving this.
- **not FMA contraction, and not `-ffast-math` reassociation.** With `-O1
  -ffp-contract=off` and no fast math on both sides, they still part at 321.
- **not the thread pool.** `MYNAH_THREADS=1` is bit-identical to the
  multi-threaded run on each platform, so our `mynah_parallel_for` reductions
  are already order-deterministic in thread count. That is worth knowing on
  its own.

**What it is, measured rather than inferred.** Dumping a 3M-point bitstream of
each libm function over the range these kernels use:

| | Apple libm | glibc 2.x | same bits? |
|---|---|---|---|
| `sqrtf` | 5.960e-08 rel | 5.960e-08 rel | **yes** (a hardware instruction) |
| `expf` | 6.032e-08 rel | 5.958e-08 rel | **NO** |
| `logf` | 5.959e-08 rel | 7.581e-08 rel | **NO** |

Both libms are accurate to about half an ulp and they round different inputs
different ways. `expf` is on the PocketTTS hot path twice — `mynah_flow_silu_f32`
and `mynah_softmax_f32` — and on a continuous-AR model that is enough, as §4
already established for a change of exactly this size.

**So the next item in this lane is `mynah_exp_f32`, and it is the same job
again**: scalar reference, NEON, AVX2, a ULP table, and a generation-level
qualification. It was never on the §3b tier-3 inventory because that inventory
was built by grepping for *Accelerate* symbols, and `expf` is libm on both
platforms — it is not a macOS-only call, it is a *two different libms* call.
The inventory's question was "what does macOS have that Linux does not"; the
question that actually predicts a divergence is "what does this binary call
that is not the same function on both machines", and those are different
questions. `logf` is the same story and is not yet known to be hot.

Until that lands, the correct statement about this work is the narrow one:
**the two Accelerate-only kernels are ours, they are measurably more accurate
than what they replaced, and one of the two named causes of the macOS/Linux
audio divergence is gone.**
