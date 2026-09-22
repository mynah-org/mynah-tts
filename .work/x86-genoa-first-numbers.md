# What an EPYC 9254 said, and the seven defects it took to get there

Box: AMD EPYC 9254 (Genoa, Zen 4), 24c/48t, 377 GB, Ubuntu 24.04, gcc 13.3.
`avx2 fma avx512f avx512bw avx512vl avx512_vnni avx512_bf16`, no AMX.
2026-09-22. **First machine this project has ever had that can execute the x86
kernels it ships.**

## Read the ratios, not the milliseconds -- the box was not quiet

Checked after the fact, which is the wrong order and is why it is written here
rather than left out: `md0` was **resyncing throughout**, 33.8% done at
206 MB/s with 200 minutes left, and it had been running for 1h43m when the
numbers below were taken -- i.e. for all of them. A second session of the
owner's was also on the machine (a Qwen3-4B ternary feasibility run, 22 cores),
though it started after the bench runs.

What that does and does not touch:

- **The ratios are the claims, and they are comparisons within minutes of each
  other on one machine**, so contention hits both sides. The large ones --
  sgemm 4.44x, bf16 3.04x, VDPBF16PS 2.98x -- are far outside the spread
  observed between repeats.
- **The absolute milliseconds are soft.** f32 matvec came back between 0.243
  and 0.413 ms across runs of the same binary, a 70% spread that is contention
  and not kernel behaviour. Do not quote a ms from this table.
- The 1.12x on axpy and the 1.26x on avx512bw-vs-avx2 are **inside** that
  spread and should be treated as "no measured difference", not as small wins.

A quiet box would settle this in ten minutes. Until then these justify the code
existing and being reached, which was the question, and not a performance
claim.

## The measurement that was the point

`make kernel-bench SIMD=portable` -- ONE binary, built with no ISA flag at all,
on Zen 4. The right-hand column is the same binary with `MYNAH_KERNELS_X86=scalar`.

| kernel | AVX2, runtime-selected | scalar | |
|---|---|---|---|
| f32 matvec_bias 4096x1024 | **0.305 ms** | 0.689 ms | 2.26x |
| f32 layernorm x64 | 0.019 | 0.032 | 1.68x |
| f32 rmsnorm x64 | 0.011 | 0.017 | 1.55x |
| f32 axpy x256 | 0.017 | 0.019 | 1.12x |
| **sgemm 16x1024x1024** | **0.071 ms** | 0.315 ms | **4.44x** |

**The scalar column is what `SIMD=portable` on x86 got before this week.** The
same portable binary also reaches VNNI (int8 0.081 ms) and VDPBF16PS (bf16
0.171 ms), neither of which needed a build flag. sgemm is the biggest single
number and `seanet.c` calls it six times on a codec path that is 43% of the
wall.

### The tiers, each forced on the one host that has all of them

| int8 dots, B=1, 4096x1024 | ms | vs AVX2 |
|---|---|---|
| `avx512vnni` (VPDPBUSD) | 0.049 | 2.27x |
| `avx512bw` (E14-2, the new tier) | 0.088 | 1.26x |
| `avx2` | 0.111 | -- |

| bf16 matvec 4096x1024 | ms | |
|---|---|---|
| `vdpbf16ps` (E14-3) | 0.161 | **2.98x** over the widening kernel |
| `avx2-widen` | 0.479 | |

**No RTF, no serving number, and none may be quoted from this.** E12 measured
the weight pass at 5% of the AR step at B=8; a kernel 3x faster here is not a
synthesis 3x faster. What these justify is the code existing and being reached,
which is what the last week was for.

## Seven defects, and where each was invisible

| # | defect | invisible on |
|---|---|---|
| 1 | build failed: `snprintf` always_inline vs the guard's baseline target | clang (accepts it silently); found by gcc 13 |
| 2 | `a lane width changed a row's answer` -- bf16 kernel reached from one caller of three | any host without AVX512-BF16 |
| 3 | `IDLE HARDWARE: isa.x86.avx512vl` on a host where two kernels require it | any host without AVX-512 |
| 4 | `make simd-auto` printed `avx512_bf16:NOT-IMPLEMENTED` while executing it | any host without AVX512-BF16 |
| 5 | `x86 SIMD=scalar` did not link -- Makefile and C decided the same thing twice | this Mac (aarch64 scalar); found by CI |
| 6 | `compile-only: CUDA` did not link -- CUDA_CPPFLAGS inherits nothing | every machine here; found by CI |
| 7 | `rss_peak_bytes` BELOW `rss_bytes`, and an RSS nobody could sum | a non-prefork process |

Plus two the bench found about itself: it mislabelled the kernel that produced
a number (pattern-matching prose that ends "half the arithmetic of VDPBF16PS"),
and its f32 rows compare two auto-vectorised builds of the same C on a `-mavx2`
build, which it now says out loud.

**The pattern in all of them: a configuration nobody builds is a configuration
nobody protects.** `tests/x86_cross.sh` builds three profiles now, not two.

## The one that was a real pessimisation

bf16 matvec took 0.489 ms where f32 over TWICE the bytes took 0.257 -- bf16
slower than f32 for half the traffic, which is the opposite of the reason the
encoding exists. A single activation went through the x4 kernel with itself in
all four arguments, so four dpbf16 per weight load and three thrown away. On Arm
that is free (memory-bound at one activation); VDPBF16PS is not in that regime.

An x1 form of both x86 bf16 kernels: **0.489 -> 0.161 ms, 3.04x**, and bf16 is
now correctly faster than f32 rather than slower. Determinism is not given up --
each output row is its own accumulator chain in the same order, so the x1 result
is bit-for-bit lane 0 of the x4, and the existing `batch=2 differs from the
row-at-a-time reference` check is what holds it.

## Everything else that was checked and was fine

- **Seven build profiles** (`auto/portable/avx2/avx512/scalar` x `none/openblas`):
  every one builds, self-tests, and reports a coherent ISA class. `portable`
  reaches exactly the kernels `auto` does. `scalar` turns everything off and
  says so. `SIMD=avx512` is indistinguishable from `avx2` here, which is what
  the Makefile has always claimed it would be.
- **sgemm.provider** tracks BLAS: `none`->mynah, `openblas`->OpenBLAS,
  `scalar`->reference.
- **The full server suite on x86**, on a fake pack: stream==batch byte-identical,
  batching, admission, mixed streams, warmup. First time it has ever run there.
- **CLI**: `--version`, `--help`, a bad flag exits 2 with a usage line, a text
  the fake tokenizer cannot encode exits 2 with `invalid token list` rather
  than crashing.
- **`/health`** reports `workers 8, prefork 4, request_places 32` for a 4x2x8
  server -- the fix for the 16x2 misreading holds on x86.
- **UBSan** clean with the new kernels actually executing.
- **`make doctor`**: 0.4 s, no model, and it still refuses to print a safe
  concurrency it has not soaked.

## Still not done here

No model pack, so no RTF, no bench-matrix, no serving wave. That needs the pack
on the box and is the next thing worth an hour.


---

## 2026-09-22, later: the real pack, and correctness instead of numbers

The box has a CPU-heavy job on it, so this half is deliberately contention-proof:
`models/pocket-en` (289 MB, scp'd because `kyutai/pocket-tts` is a gated repo),
and questions whose answers are bits rather than milliseconds.

### The full server suite, on the real pack, on x86 -- PASS

Every check: `stream==batch` byte-identical, `repro x3` byte-identical,
`batching` 4 concurrent == 4 serial and pairwise distinct, `admission`,
`mixed` 2 streams + 2 batches, `warmup` not vacuous. Previously this had only
ever run on Arm and on a fake pack. `--pocket-self-check` passes too.

### Cross-tier audio parity -- the question only this host can be asked

`make x86-tier-parity MODEL_DIR=models/pocket-en`, temperature 0, seed 42.

| check | result |
|---|---|
| the same tier twice | **byte-identical** -- without this nothing else means anything |
| int8 `avx512bw` vs `avx2` | **byte-identical**, as exact int32 requires |
| `bf16_off`, `f32_scalar`, `vnni_off` vs baseline | same length, 119084 B -- the AR loop made the same decisions |

The second row is E14-2's correctness at the level that matters: the new
AVX-512BW kernel is not merely close to the AVX2 one, it is the same audio.
The third is the property an autoregressive model makes fragile -- a one-ULP
disagreement is amplified by every later step, and here it did not change a
single frame decision across four different kernel families.

### What the test found that the audio did not

`MYNAH_QMAT_VNNI=off` and `=scalar` **also turn `codec.conv_int8_host` from
eligible to off**. `src/convq8.c` asks which int8 kernel RESOLVED and reads a
forced-down one as a host with no dot unit, so the whole conv stack stays f32.
The behaviour is deliberate and the dispatch map states it in the row's reason.

The defect was mine, in the method: I ran an A/B that varied two things and was
one step from attributing an audio difference to VPDPBUSD. So the test now
asserts, for every forcing, that it moves only the dispatch rows it is ALLOWED
to move -- and `codec.conv_int8_host` is on the int8 forcings' list, which turns
a surprise into a stated invariant. The reason string for `u8-scalar` says it
too, so a reader of that one row learns it without diffing two reports.

One more thing that check caught first: it reported the baseline as differing
from ITSELF, because `pool.spin` is a measured per-host calibration and moves
between two runs of the same binary. Excluded, with the reason written down.

### Still not done

No RTF and no serving wave, on purpose: the box is busy and a number taken
there would be a number about the other job. E14-9 stays open.
